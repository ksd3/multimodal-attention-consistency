"""
Synthetic Experiment: Multimodal Attention Consistency
=====================================================
Synthetic validation for the nuclear norm consistency penalty
on block cross-attention matrices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ============================================
# CONFIG
# ============================================

@dataclass
class ExperimentConfig:
    # Concepts and data
    num_concepts: int = 200
    num_train_samples: int = 10_000
    num_val_samples: int = 2_000
    num_test_samples: int = 2_000

    # Multi-token setup 
    # Each sample has multiple tokens per modality,
    # simulating audio frames, video patches, text words
    tokens_per_modality: Dict[str, int] = field(default_factory=lambda: {
        "audio": 5,   # e.g. 5 audio frames
        "video": 4,   # e.g. 4 spatial patches (2x2)
        "text":  3,   # e.g. 3 words
    })

    # Dimensions
    embedding_dim: int = 128
    audio_raw_dim: int = 64
    video_raw_dim: int = 128
    text_raw_dim: int = 32

    # Architecture
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    num_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Consistency penalty
    lambda_consistency: float = 0.1
    lambda_attn_entropy: float = 0.1  # penalizes uniform attention (encourages peaked)
    svd_epsilon: float = 1e-6      # for gradient stability
    use_randomized_svd: bool = False  # for speed at scale
    svd_top_k: int = 0              # 0 = full SVD, >0 = truncated

    # Partial observability — each modality sees this fraction of concept dims
    # Forces cross-modal attention to be informative (not decorative)
    modality_visibility: float = 0.5

    # Noise and corruption
    noise_scale: float = 0.3
    corruption_rates: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    )

    # Statistical rigor
    num_seeds: int = 5             # run every experiment 5 times
    seeds: List[int] = field(
        default_factory=lambda: [42, 123, 456, 789, 1024]
    )

    # Modality scaling experiment
    modality_counts: List[int] = field(
        default_factory=lambda: [2, 3, 4, 5]
    )

    # Lambda sweep
    lambda_values: List[float] = field(
        default_factory=lambda: [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    )


cfg = ExperimentConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================
# STEP 1: Multi-Token Synthetic Dataset
# ============================================
# Each sample has MULTIPLE tokens per modality.
#
# Why this matters:
# - Single token → attention is just a scalar → trivial
# - Multi token → attention is a matrix → permutation structure 
#   actually applies
#
# How it works:
# - A concept like "dog" has an underlying representation
# - Audio: 5 tokens representing different aspects of the sound 
#   (bark onset, sustain, pitch, rhythm, decay)
# - Video: 4 tokens representing spatial parts 
#   (head, body, legs, tail)
# - Text: 3 tokens representing words 
#   ("a", "dog", "barks")
# - All tokens within a modality are generated from the same 
#   concept vector but with different sub-projections

class MultiTokenSyntheticDataset(Dataset):
    """
    Each sample:
    - concept_id: int
    - audio_tokens: (T_a, D_a) — multiple audio frames
    - video_tokens: (T_v, D_v) — multiple video patches
    - text_tokens:  (T_t, D_t) — multiple word embeddings
    
    The sub-token structure simulates how real multimodal data works:
    a 2-second audio clip has multiple frames, a video frame has 
    multiple patches, a caption has multiple words.
    """

    def __init__(
        self,
        num_samples: int,
        num_concepts: int,
        tokens_per_modality: Dict[str, int],
        raw_dims: Dict[str, int],
        corruption_rate: float = 0.0,
        corruption_mode: str = "swap_sample",
        seed: int = 42,
        concept_seed: int = 0,
    ):
        # concept_seed: fixed seed for concept vectors and projections
        #   (must be identical across train/val/test splits)
        # seed: per-split seed for sampling concept_ids and noise
        self.rng = np.random.RandomState(seed)

        self.num_samples = num_samples
        self.num_concepts = num_concepts
        self.tokens_per_modality = tokens_per_modality
        self.raw_dims = raw_dims
        self.modality_names = sorted(tokens_per_modality.keys())
        self.corruption_rate = corruption_rate
        self.corruption_mode = corruption_mode

        # Ground truth concept embeddings — FIXED across splits
        gen = torch.Generator().manual_seed(concept_seed)
        self.concept_vectors = F.normalize(
            torch.randn(num_concepts, cfg.embedding_dim, generator=gen), dim=-1
        )

        # Per-modality concept masks — each modality only sees a subset
        # of concept dimensions, forcing the model to combine modalities
        self.modality_masks = {}
        n_visible = int(cfg.modality_visibility * cfg.embedding_dim)
        for mod in self.modality_names:
            perm = torch.randperm(cfg.embedding_dim, generator=gen)
            mask = torch.zeros(cfg.embedding_dim)
            mask[perm[:n_visible]] = 1.0
            self.modality_masks[mod] = mask

        # Per-modality projections — FIXED across splits
        self.projections = {}
        for mod, num_tokens in tokens_per_modality.items():
            raw_dim = self.raw_dims[mod]
            self.projections[mod] = torch.randn(
                num_tokens, cfg.embedding_dim, raw_dim, generator=gen
            ) * 0.1

        # Sample-level randomness uses the per-split seed
        torch.manual_seed(seed)
        self.samples = self._generate()

    def _generate(self):
        samples = []
        for _ in range(self.num_samples):
            concept_id = self.rng.randint(0, self.num_concepts)
            concept_vec = self.concept_vectors[concept_id]

            modality_tokens = {}
            for mod, num_tokens in self.tokens_per_modality.items():
                raw_dim = self.raw_dims[mod]
                tokens = []
                for t in range(num_tokens):
                    # Mask concept (partial observability) then project + noise
                    masked_vec = concept_vec * self.modality_masks[mod]
                    token = masked_vec @ self.projections[mod][t] + \
                            torch.randn(raw_dim) * cfg.noise_scale
                    tokens.append(token)
                modality_tokens[mod] = torch.stack(tokens)  # (T, D_raw)

            # Apply corruption if needed
            if self.rng.random() < self.corruption_rate:
                modality_tokens = self._corrupt(modality_tokens, concept_id)

            samples.append({
                "concept_id": concept_id,
                **modality_tokens,
            })

        return samples

    def _corrupt(self, modality_tokens, original_concept_id):
        """
        Multiple corruption modes to test different failure types.
        Corrupts the first modality (alphabetically sorted).
        """
        # Pick the first modality to corrupt
        corrupt_mod = self.modality_names[0]
        raw_dim = self.raw_dims[corrupt_mod]

        if self.corruption_mode == "swap_sample":
            # Replace with a DIFFERENT concept's tokens
            wrong_id = self.rng.randint(0, self.num_concepts)
            while wrong_id == original_concept_id:
                wrong_id = self.rng.randint(0, self.num_concepts)
            wrong_vec = self.concept_vectors[wrong_id]

            new_tokens = []
            for t in range(self.tokens_per_modality[corrupt_mod]):
                masked_wrong = wrong_vec * self.modality_masks[corrupt_mod]
                token = masked_wrong @ self.projections[corrupt_mod][t] + \
                        torch.randn(raw_dim) * cfg.noise_scale
                new_tokens.append(token)
            modality_tokens[corrupt_mod] = torch.stack(new_tokens)

        elif self.corruption_mode == "shuffle_tokens":
            # Shuffle token ORDER (temporal disorder)
            perm = torch.randperm(self.tokens_per_modality[corrupt_mod])
            modality_tokens[corrupt_mod] = modality_tokens[corrupt_mod][perm]

        elif self.corruption_mode == "partial_noise":
            # Replace 50% of tokens with pure noise
            num_corrupt = self.tokens_per_modality[corrupt_mod] // 2
            indices = self.rng.choice(
                self.tokens_per_modality[corrupt_mod], num_corrupt, replace=False
            )
            for idx in indices:
                modality_tokens[corrupt_mod][idx] = \
                    torch.randn(raw_dim) * cfg.noise_scale

        return modality_tokens

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Return modalities in sorted name order to match model's sorted keys
        return tuple(s[name] for name in self.modality_names) + (s["concept_id"],)


# ============================================
# STEP 2: Model Architecture (Multi-Token)
# ============================================

class ModalityEncoder(nn.Module):
    """
    Encodes raw modality tokens into shared embedding space.
    Includes positional encoding so the model knows token order.
    """
    def __init__(self, input_dim: int, hidden_dim: int, max_tokens: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_tokens, hidden_dim) * 0.02
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x):
        """
        x: (batch, num_tokens, raw_dim)
        returns: (batch, num_tokens, hidden_dim)
        """
        h = self.projection(x)                          # (B, T, D)
        h = h + self.pos_encoding[:, :h.size(1), :]     # add position
        h = self.norm(h + self.mlp(h))                  # FFN + residual
        return h


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention with extractable attention weights.
    Returns both the output AND the raw attention matrix.
    """
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key_value):
        """
        query:     (B, T_q, D)
        key_value: (B, T_k, D)

        returns:
            output:  (B, T_q, D)
            weights: (B, T_q, T_k) — averaged over heads
        """
        B, T_q, D = query.shape
        T_k = key_value.size(1)

        Q = self.q_proj(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        scores = (Q @ K.transpose(-2, -1)) / scale     # (B, H, T_q, T_k)
        weights = F.softmax(scores, dim=-1)             # (B, H, T_q, T_k)

        # Weighted sum of values
        attn_output = (weights @ V).transpose(1, 2).reshape(B, T_q, D)
        output = self.out_proj(attn_output)
        output = self.norm(query + output)  # residual connection

        # Average weights over heads for the P matrix
        avg_weights = weights.mean(dim=1)   # (B, T_q, T_k)

        return output, avg_weights


class MultimodalTransformerMultiToken(nn.Module):
    """
    Full multimodal transformer supporting N modalities,
    each with variable number of tokens.

    Parameterized to support 2-5 modalities for the scaling experiment.
    """
    def __init__(
        self,
        num_concepts: int,
        modality_configs: Dict[str, Tuple[int, int]],
        # e.g. {"audio": (64, 5), "video": (128, 4), "text": (32, 3)}
        # maps modality name → (raw_dim, num_tokens)
    ):
        super().__init__()
        self.modality_names = sorted(modality_configs.keys())
        self.num_modalities = len(self.modality_names)

        # Per-modality encoders
        self.encoders = nn.ModuleDict()
        for name, (raw_dim, num_tokens) in modality_configs.items():
            self.encoders[name] = ModalityEncoder(
                raw_dim, cfg.embedding_dim, num_tokens
            )

        # Cross-attention for ALL ordered pairs of modalities
        # e.g. for 3 modalities: 6 cross-attention blocks
        self.cross_attns = nn.ModuleDict()
        for i, name_i in enumerate(self.modality_names):
            for j, name_j in enumerate(self.modality_names):
                if i != j:
                    key = f"{name_i}_to_{name_j}"
                    self.cross_attns[key] = CrossAttentionLayer(
                        cfg.embedding_dim, cfg.num_heads
                    )

        # Classification head: pool all tokens from all modalities → predict concept
        self.pool = nn.AdaptiveAvgPool1d(1)  # pool over token dimension
        self.classifier = nn.Sequential(
            nn.Linear(cfg.embedding_dim * self.num_modalities, cfg.embedding_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embedding_dim, num_concepts),
        )

    def forward(self, modality_inputs: Dict[str, torch.Tensor]):
        """
        modality_inputs: {"audio": (B, T_a, D_a), "video": (B, T_v, D_v), ...}

        returns:
            logits:    (B, num_concepts)
            attn_dict: {("audio","video"): (B, T_a, T_v), ...} — all cross-attn weights
            embeddings_dict: {"audio": (B, T_a, D), ...} — encoded representations
        """
        # Encode each modality
        encoded = {}
        for name in self.modality_names:
            encoded[name] = self.encoders[name](modality_inputs[name])

        # Cross-attention between all pairs
        attn_dict = {}
        for i, name_i in enumerate(self.modality_names):
            for j, name_j in enumerate(self.modality_names):
                if i != j:
                    key = f"{name_i}_to_{name_j}"
                    output, weights = self.cross_attns[key](
                        encoded[name_i], encoded[name_j]
                    )
                    # Update encoding with cross-attended output
                    encoded[name_i] = output
                    # Store attention weights: (B, T_i, T_j)
                    attn_dict[(name_i, name_j)] = weights

        # Pool and classify
        pooled = []
        for name in self.modality_names:
            # (B, T, D) → (B, D)
            p = encoded[name].mean(dim=1)
            pooled.append(p)

        fused = torch.cat(pooled, dim=-1)  # (B, N*D)
        logits = self.classifier(fused)    # (B, num_concepts)

        return logits, attn_dict, encoded


# ============================================
# STEP 3: Build the Block P Matrix (Multi-Token)
# ============================================
# With multi-token, P is a proper block matrix where 
# each block is (T_i x T_j) — not just scalars.

def build_P_matrix(
    attn_dict: Dict[Tuple[str, str], torch.Tensor],
    modality_names: List[str],
    tokens_per_modality: Dict[str, int],
    batch_idx: int = 0,
) -> torch.Tensor:
    """
    Assemble the (sum(T_i) x sum(T_j)) block matrix P 
    for a SINGLE sample in the batch.

    For 3 modalities with T_a=5, T_v=4, T_t=3:
    P is (12 x 12) with structure:

        |  I_5    A_av   A_at  |
    P = |  A_va   I_4    A_vt  |
        |  A_ta   A_tv   I_3   |

    where A_av is the (5 x 4) cross-attention matrix 
    from audio to video for this sample.
    """
    # Compute total dimension
    sizes = [tokens_per_modality[name] for name in modality_names]
    total = sum(sizes)
    N = len(modality_names)

    P = torch.zeros(total, total)

    # Fill blocks
    row_offset = 0
    for i, name_i in enumerate(modality_names):
        T_i = sizes[i]
        col_offset = 0
        for j, name_j in enumerate(modality_names):
            T_j = sizes[j]
            if i == j:
                # Diagonal: identity
                P[row_offset:row_offset+T_i, col_offset:col_offset+T_j] = \
                    torch.eye(T_i)
            else:
                # Off-diagonal: cross-attention weights for this sample
                # attn_dict[(name_i, name_j)] is (B, T_i, T_j)
                P[row_offset:row_offset+T_i, col_offset:col_offset+T_j] = \
                    attn_dict[(name_i, name_j)][batch_idx]
            col_offset += T_j
        row_offset += T_i

    return P  # (total x total), e.g. (12 x 12)


def build_batch_P_matrices(
    attn_dict: Dict[Tuple[str, str], torch.Tensor],
    modality_names: List[str],
    tokens_per_modality: Dict[str, int],
    batch_size: int,
) -> torch.Tensor:
    """
    Build P for every sample in the batch.
    Returns: (B, total, total) — a batch of P matrices.

    Uses torch.cat to preserve autograd gradient flow from attention
    weights through to the nuclear norm loss.
    """
    # Determine device from attention dict values
    device = next(iter(attn_dict.values())).device

    block_rows = []
    for i, name_i in enumerate(modality_names):
        T_i = tokens_per_modality[name_i]
        row_blocks = []
        for j, name_j in enumerate(modality_names):
            T_j = tokens_per_modality[name_j]
            if i == j:
                row_blocks.append(
                    torch.eye(T_i, device=device).unsqueeze(0).expand(batch_size, -1, -1)
                )
            else:
                row_blocks.append(attn_dict[(name_i, name_j)])
        block_rows.append(torch.cat(row_blocks, dim=2))  # (B, T_i, total)

    P = torch.cat(block_rows, dim=1)  # (B, total, total)
    return P


# ============================================
# STEP 4: Consistency Losses
# ============================================
# We implement ALL methods for fair comparison:
# (a) Nuclear norm (ours)
# (b) Contrastive loss (CLIP-style baseline)
# (c) Cycle-consistency loss (baseline)
# (d) Mutual information proxy (baseline)

# --- (a) OURS: Smoothed Nuclear Norm ---

def nuclear_norm_loss(
    P: torch.Tensor,
    tokens_per_modality: Dict[str, int],
    num_modalities: int,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Smoothed nuclear norm of P, normalized by the ideal value.

    Smoothing: instead of sum(sigma), compute sum(sqrt(sigma^2 + eps)).
    This avoids gradient blowup when singular values are near zero,
    which happens frequently during training.

    The SVD gradient is: d||P||_* / dP = U @ V^T
    When two singular values are close, U and V become ill-conditioned.
    The epsilon smoothing fixes this.

    Args:
        P: (B, D, D) batch of P matrices
        tokens_per_modality: for computing the ideal norm
        num_modalities: N
        epsilon: smoothing constant for gradient stability
    """
    # Compute singular values: (B, min(D,D))
    # torch.linalg.svdvals is differentiable
    sigmas = torch.linalg.svdvals(P)                   # (B, D)

    # Smoothed nuclear norm per sample
    smoothed_norms = torch.sum(
        torch.sqrt(sigmas ** 2 + epsilon), dim=-1
    )                                                   # (B,)

    # Ideal nuclear norm when perfectly aligned:
    # M singular values each equal to N, rest are 0
    # So ideal = M * N (plus epsilon smoothing)
    M = min(tokens_per_modality.values())  # conservative
    ideal = num_modalities * M

    # Loss = excess over ideal, averaged over batch
    loss = (smoothed_norms - ideal).mean()

    return loss


def nuclear_norm_loss_randomized(
    P: torch.Tensor,
    top_k: int = 20,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Approximate nuclear norm using randomized SVD.
    Only computes top-k singular values.

    Much faster for large P matrices (e.g. when N=5, M=10, P is 50x50).
    Theoretically justified because we only care about the EXCESS 
    singular values beyond the first M.

    Uses the Halko-Martinsson-Tropp algorithm.
    """
    # torch.svd_lowrank computes truncated SVD efficiently
    U, S, V = torch.svd_lowrank(P, q=top_k)
    smoothed = torch.sum(torch.sqrt(S ** 2 + epsilon), dim=-1)
    return smoothed.mean()


# --- (b) BASELINE: Contrastive Loss (CLIP-style) ---

def contrastive_loss(
    embeddings_dict: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Standard InfoNCE contrastive loss between all modality pairs.
    This is what CLIP does, extended to N modalities.

    For each pair (mod_i, mod_j):
    - Pool tokens to get one vector per sample: (B, D)
    - Compute cosine similarity matrix: (B, B)
    - Diagonal entries should be highest (same concept)

    LIMITATION THIS PAPER EXPOSES:
    This loss only enforces PAIRWISE alignment.
    It does NOT enforce TRANSITIVE consistency.
    A↔B can be perfect and B↔C can be perfect, 
    but A↔C can still be wrong.
    """
    modality_names = sorted(embeddings_dict.keys())
    total_loss = 0
    num_pairs = 0

    for i, name_i in enumerate(modality_names):
        for j, name_j in enumerate(modality_names):
            if i >= j:
                continue

            # Pool tokens → one vector per sample
            emb_i = embeddings_dict[name_i].mean(dim=1)  # (B, D)
            emb_j = embeddings_dict[name_j].mean(dim=1)  # (B, D)

            # Normalize
            emb_i = F.normalize(emb_i, dim=-1)
            emb_j = F.normalize(emb_j, dim=-1)

            # Cosine similarity matrix
            sim = (emb_i @ emb_j.T) / temperature  # (B, B)

            # Targets: samples with same concept_id should match
            # For simplicity, use diagonal (each sample matches itself)
            targets = torch.arange(sim.size(0), device=sim.device)

            loss_ij = F.cross_entropy(sim, targets)
            loss_ji = F.cross_entropy(sim.T, targets)
            total_loss += (loss_ij + loss_ji) / 2
            num_pairs += 1

    return total_loss / num_pairs


# --- (c) BASELINE: Cycle-Consistency Loss ---
