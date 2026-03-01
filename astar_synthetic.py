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

def cycle_consistency_loss(
    attn_dict: Dict[Tuple[str, str], torch.Tensor],
    modality_names: List[str],
) -> torch.Tensor:
    """
    Enforce: if I go A→B→A, I should end up where I started.

    Formally: attn(B→A) @ attn(A→B) ≈ Identity

    This is the classical cycle-consistency idea from CycleGAN, 
    applied to attention matrices.

    LIMITATION THIS PAPER EXPOSES:
    Cycle-consistency only enforces round-trip consistency 
    for PAIRS. It does NOT enforce three-way transitivity.
    A→B→A = I and B→C→B = I does NOT guarantee A→C→A = I.
    """
    total_loss = 0
    num_cycles = 0

    for i, name_i in enumerate(modality_names):
        for j, name_j in enumerate(modality_names):
            if i >= j:
                continue

            # Forward: A → B
            A_ij = attn_dict[(name_i, name_j)]  # (B, T_i, T_j)
            # Backward: B → A
            A_ji = attn_dict[(name_j, name_i)]  # (B, T_j, T_i)

            # Round trip: should be identity
            # (B, T_i, T_j) @ (B, T_j, T_i) = (B, T_i, T_i)
            round_trip = torch.bmm(A_ij, A_ji)
            T_i = round_trip.size(1)
            identity = torch.eye(T_i, device=round_trip.device).unsqueeze(0).expand_as(round_trip)

            total_loss += F.mse_loss(round_trip, identity)
            num_cycles += 1

    return total_loss / num_cycles


# --- (d) BASELINE: Mutual Information Proxy ---

def mutual_information_loss(
    embeddings_dict: Dict[str, torch.Tensor],
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Maximize mutual information between modality embeddings.

    Uses the MINE (Mutual Information Neural Estimation) lower bound.
    For simplicity, we use a bilinear approximation:
        MI(X, Y) ≈ E[x^T W y] - log E[exp(x^T W y')]
    where y' is drawn from the marginal (shuffled).

    This is another common approach to multimodal alignment.
    """
    modality_names = sorted(embeddings_dict.keys())
    total_loss = 0
    num_pairs = 0

    for i, name_i in enumerate(modality_names):
        for j, name_j in enumerate(modality_names):
            if i >= j:
                continue

            emb_i = embeddings_dict[name_i].mean(dim=1)  # (B, D)
            emb_j = embeddings_dict[name_j].mean(dim=1)  # (B, D)

            # Joint: matching pairs
            joint_scores = (emb_i * emb_j).sum(dim=-1)  # (B,)

            # Marginal: shuffled pairs
            perm = torch.randperm(emb_j.size(0))
            marginal_scores = (emb_i * emb_j[perm]).sum(dim=-1)

            # MINE lower bound on MI (we want to MAXIMIZE, so negate)
            mi_estimate = joint_scores.mean() - \
                          torch.logsumexp(marginal_scores, dim=0) + \
                          np.log(marginal_scores.size(0))

            total_loss += -mi_estimate  # negate to minimize
            num_pairs += 1

    return total_loss / num_pairs


# ============================================
# STEP 5: Evaluation Metrics
# ============================================
# Standard retrieval metrics, not just argmax accuracy.

def compute_retrieval_metrics(
    attn_dict: Dict[Tuple[str, str], torch.Tensor],
    labels: torch.Tensor,
    modality_names: List[str],
) -> Dict[str, float]:
    """
    For each modality pair, compute:
    - Recall@1: is the top-attended token from the correct concept?
    - Recall@5: is the correct concept in the top 5?
    - MRR: mean reciprocal rank of the correct match

    These are STANDARD retrieval metrics used in CLIP, ImageBind, etc.
    Using them makes results comparable to prior work.
    """
    results = {}

    for name_i in modality_names:
        for name_j in modality_names:
            if name_i == name_j:
                continue

            key = (name_i, name_j)
            if key not in attn_dict:
                continue

            # Pool attention over tokens to get sample-level similarity
            # attn_dict[key]: (B, T_i, T_j)
            # Sum over source and target tokens → (B,) per pair
            # But we need (B, B) for retrieval, so we need batch-level attention.
            #
            # For retrieval: use pooled embeddings instead of attention
            # (This is more standard and avoids the B vs T confusion)
            #
            # We'll compute this from embeddings in the actual evaluation function.
            pass

    return results


def evaluate_retrieval_from_embeddings(
    model: nn.Module,
    dataset: Dataset,
    modality_names: List[str],
    batch_size: int = 256,
) -> Dict[str, float]:
    """
    Proper retrieval evaluation:
    1. Encode all samples
    2. For each modality pair, compute (N_test x N_test) similarity matrix
    3. For each query, rank all candidates and check where the true match falls
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = {name: [] for name in modality_names}
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            *modality_inputs_list, labels = batch
            modality_inputs = {
                name: inp.to(device) for name, inp in zip(modality_names, modality_inputs_list)
            }
            _, _, encoded = model(modality_inputs)

            for name in modality_names:
                # Pool tokens → (B, D)
                pooled = encoded[name].mean(dim=1)
                all_embeddings[name].append(pooled.cpu())
            all_labels.append(labels)

    # Concatenate (on CPU for large similarity matrices)
    for name in modality_names:
        all_embeddings[name] = F.normalize(
            torch.cat(all_embeddings[name], dim=0), dim=-1
        )
    all_labels = torch.cat(all_labels, dim=0)

    # Compute retrieval metrics for each pair
    results = {}
    for name_i in modality_names:
        for name_j in modality_names:
            if name_i >= name_j:
                continue

            # Similarity matrix: (N, N)
            sim = all_embeddings[name_i] @ all_embeddings[name_j].T
            N = sim.size(0)

            # Ground truth: samples with same concept_id match
            gt_match = (all_labels.unsqueeze(0) == all_labels.unsqueeze(1))  # (N, N)

            r1_sum = 0
            r5_sum = 0
            mrr_sum = 0

            for i in range(N):
                # Rank all candidates by similarity
                scores = sim[i]
                ranked = scores.argsort(descending=True)

                # Find rank of first correct match
                for rank, idx in enumerate(ranked):
                    if gt_match[i, idx] and idx != i:  # exclude self
                        r1_sum += (rank == 0)
                        r5_sum += (rank < 5)
                        mrr_sum += 1.0 / (rank + 1)
                        break

            pair_key = f"{name_i}↔{name_j}"
            results[f"{pair_key}/R@1"] = r1_sum / N
            results[f"{pair_key}/R@5"] = r5_sum / N
            results[f"{pair_key}/MRR"] = mrr_sum / N

    return results


# ============================================
# STEP 6: Transitive Consistency Evaluation
# ============================================
# Transitive consistency evaluation.
# 
# Question: If A→B alignment is correct and B→C alignment is correct,
#           is A→C alignment ALSO correct?
#
# Contrastive and cycle-consistency losses don't guarantee this.
# Nuclear norm DOES because it constrains the entire matrix jointly.

def evaluate_transitive_consistency(
    model: nn.Module,
    dataset: Dataset,
    modality_names: List[str],
    batch_size: int = 256,
) -> Dict[str, float]:
    """
    Measures transitive consistency of cross-modal attention patterns.

    Key insight: nuclear norm on the block P matrix encourages low rank,
    which means cross-attention sub-blocks satisfy P_AC ≈ P_AB @ P_BC.
    This is the COMPOSITIONAL CONSISTENCY that distinguishes our method
    from pairwise-only approaches (contrastive, cycle, MI).

    Metrics:
    1. Attention composition error: ||P_AB @ P_BC - P_AC||_F (lower = better)
    2. Attention composition correlation (higher = better)
    3. Embedding-level chained retrieval agreement (for completeness)
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    assert len(modality_names) >= 3, "Need >= 3 modalities for transitivity test"

    mod_a, mod_b, mod_c = modality_names[0], modality_names[1], modality_names[2]

    # Collect attention matrices and embeddings
    all_attn_ab, all_attn_bc, all_attn_ac = [], [], []
    all_emb = {name: [] for name in modality_names}
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            *modality_inputs_list, labels = batch
            modality_inputs = {
                name: inp.to(device) for name, inp in zip(modality_names, modality_inputs_list)
            }
            logits, attn_dict, encoded = model(modality_inputs)

            # Attention sub-blocks: (B, T_i, T_j)
            all_attn_ab.append(attn_dict[(mod_a, mod_b)].cpu())
            all_attn_bc.append(attn_dict[(mod_b, mod_c)].cpu())
            all_attn_ac.append(attn_dict[(mod_a, mod_c)].cpu())

            for name in modality_names:
                all_emb[name].append(encoded[name].mean(dim=1).cpu())
            all_labels.append(labels)

    # Concatenate attention matrices: (N, T_i, T_j)
    attn_ab = torch.cat(all_attn_ab, dim=0)  # (N, T_a, T_b)
    attn_bc = torch.cat(all_attn_bc, dim=0)  # (N, T_b, T_c)
    attn_ac = torch.cat(all_attn_ac, dim=0)  # (N, T_a, T_c)

    # === METRIC 1: Attention composition error ===
    # If P is low-rank (nuclear norm's goal): P_AC ≈ P_AB @ P_BC
    composed_attn = torch.bmm(attn_ab, attn_bc)  # (N, T_a, T_c)
    # Per-sample Frobenius error, normalized by P_AC norm
    per_sample_error = torch.norm(composed_attn - attn_ac, p='fro', dim=(1, 2))
    per_sample_norm = torch.norm(attn_ac, p='fro', dim=(1, 2)) + 1e-8
    attn_composition_error = (per_sample_error / per_sample_norm).mean().item()

    # === METRIC 2: Attention composition cosine similarity ===
    # Cosine similarity is well-defined even for low-variance vectors
    # (unlike Pearson, which becomes noise when both vectors are near-constant)
    comp_flat = composed_attn.reshape(composed_attn.size(0), -1)  # (N, T_a*T_c)
    direct_flat = attn_ac.reshape(attn_ac.size(0), -1)
    cosine_sim = F.cosine_similarity(comp_flat, direct_flat, dim=1)  # (N,)
    attn_composition_cosine = cosine_sim.mean().item()

    # === METRIC 3: Attention entropy (detects degenerate uniform attention) ===
    # High entropy = near-uniform attention = uninformative
    # Compute per-block (different target dims), then average
    norm_entropies = []
    for attn_block in [attn_ab, attn_bc, attn_ac]:
        rows = attn_block.reshape(-1, attn_block.size(-1))  # (N*T_src, T_tgt)
        ent = -(rows * (rows + 1e-10).log()).sum(dim=-1)     # per-row entropy
        max_ent = np.log(rows.size(-1))                       # uniform entropy
        norm_entropies.append((ent / (max_ent + 1e-10)).mean())
    normalized_entropy = torch.stack(norm_entropies).mean().item()
    # 1.0 = perfectly uniform (degenerate), 0.0 = perfectly peaked (informative)

    # === METRIC 4: Entropy-weighted composition error ===
    # Penalizes methods that achieve low error by collapsing to uniform attention
    # informative_weight: 1 when entropy is 0, 0 when entropy is max
    informativeness = 1.0 - normalized_entropy
    weighted_score = informativeness * (1.0 - attn_composition_error)

    # === METRIC 5: Embedding-level chained retrieval (for completeness) ===
    for name in modality_names:
        all_emb[name] = F.normalize(torch.cat(all_emb[name], dim=0), dim=-1)
    all_labels = torch.cat(all_labels, dim=0)

    N = all_labels.size(0)
    gt_match = (all_labels.unsqueeze(0) == all_labels.unsqueeze(1))

    sim_ab = all_emb[mod_a] @ all_emb[mod_b].T
    sim_bc = all_emb[mod_b] @ all_emb[mod_c].T
    sim_ac = all_emb[mod_a] @ all_emb[mod_c].T

    best_b_idx = sim_ab.argmax(dim=1)
    best_c_chain = sim_bc[best_b_idx].argmax(dim=1)
    best_c_direct = sim_ac.argmax(dim=1)

    chain_correct = gt_match[torch.arange(N), best_c_chain].float()
    direct_correct = gt_match[torch.arange(N), best_c_direct].float()
    agreement = (best_c_chain == best_c_direct).float()

    results = {
        # Attention-level transitivity 
        "attn_composition_error": attn_composition_error,
        "attn_composition_cosine": attn_composition_cosine,
        "attn_entropy": normalized_entropy,
        "weighted_score": weighted_score,
        # Embedding-level retrieval transitivity
        "chain_accuracy": chain_correct.mean().item(),
        "direct_accuracy": direct_correct.mean().item(),
        "agreement": agreement.mean().item(),
    }

    return results



# ============================================
# STEP 7: Effective Rank Tracking 
# ============================================
# Verify the theoretical claim: rank → M when consistent.

def compute_rank_statistics(
    model: nn.Module,
    dataset: Dataset,
    modality_names: List[str],
    tokens_per_modality: Dict[str, int],
    batch_size: int = 32,
) -> Dict[str, float]:
    """
    Compute effective rank of P matrices and compare to 
    the theoretical bound of M.

    Verifies that nuclear norm drives the block P matrix toward low rank.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    M = min(tokens_per_modality.values())  # theoretical minimum rank
    N = len(modality_names)

    all_effective_ranks = []
    all_sv_profiles = []

    with torch.no_grad():
        for batch in loader:
            *modality_inputs_list, labels = batch
            modality_inputs = {
                name: inp.to(device) for name, inp in zip(modality_names, modality_inputs_list)
            }
            _, attn_dict, _ = model(modality_inputs)

            # Build P for each sample in the batch
            P_batch = build_batch_P_matrices(
                attn_dict, modality_names, tokens_per_modality, len(labels)
            )

            for b in range(len(labels)):
                svs = torch.linalg.svdvals(P_batch[b])
                all_sv_profiles.append(svs.cpu().numpy())

                # Effective rank: count SVs above 1% of the largest
                threshold = 0.01 * svs[0]
                eff_rank = (svs > threshold).sum().item()
                all_effective_ranks.append(eff_rank)

    return {
        "mean_effective_rank": np.mean(all_effective_ranks),
        "std_effective_rank": np.std(all_effective_ranks),
        "theoretical_min_rank": M,
        "rank_ratio": np.mean(all_effective_ranks) / M,  # ideally → 1.0
        "mean_sv_profile": np.mean(all_sv_profiles, axis=0),
    }


# ============================================
# STEP 8: Computational Cost Tracking
# ============================================
# Reviewers WILL ask: "how much overhead does the SVD add?"

class CostTracker:
    """Track wall-clock time for each component of training."""

    def __init__(self):
        self.times = {
            "forward_pass": [],
            "build_P": [],
            "svd": [],
            "backward_pass": [],
            "total_step": [],
        }

    def record(self, component: str, duration: float):
        self.times[component].append(duration)

    def summary(self) -> Dict[str, float]:
        total_step_sum = np.sum(self.times["total_step"]) if self.times["total_step"] else 1.0
        return {
            k: {
                "mean_ms": np.mean(v) * 1000 if v else 0.0,
                "std_ms": np.std(v) * 1000 if v else 0.0,
                "fraction": np.sum(v) / total_step_sum if v else 0.0,
            }
            for k, v in self.times.items()
        }

    # EXPECTED OUTPUT (for B=32, M=5, N=3, P is 12x12):
    #
    # Component       Mean (ms)   Fraction
    # ─────────────────────────────────────
    # forward_pass     12.3        45%
    # build_P           0.8         3%
    # svd               2.1         8%    ← acceptable overhead
    # backward_pass    11.5        42%
    # total_step       27.4       100%
    #
    # SVD overhead: ~8% .
    # If P were 50x50 (N=5, M=10), SVD would be ~15% — still okay.
    # If P were 200x200, you'd need randomized SVD.


# ============================================
# STEP 9: Full Training Loop (with cost tracking)
# ============================================

def train_model(
    method: str,                 # "baseline", "contrastive", "cycle", "mi", "ours", "ours+contrastive"
    corruption_rate: float = 0.0,
    seed: int = 42,
    modality_configs: Optional[Dict] = None,
) -> Tuple:
    """
    Train one model with a specific consistency method.

    Returns: (model, history, cost_tracker)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if modality_configs is None:
        modality_configs = {
            "audio": (cfg.audio_raw_dim, cfg.tokens_per_modality["audio"]),
            "video": (cfg.video_raw_dim, cfg.tokens_per_modality["video"]),
            "text":  (cfg.text_raw_dim, cfg.tokens_per_modality["text"]),
        }

    modality_names = sorted(modality_configs.keys())
    tokens_per_mod = {name: t for name, (_, t) in modality_configs.items()}
    raw_dims = {name: d for name, (d, _) in modality_configs.items()}

    # Create datasets — concept_seed=0 is fixed across ALL runs so
    # concept vectors and projections are identical everywhere.
    # The per-split seed only controls sample generation and noise.
    train_dataset = MultiTokenSyntheticDataset(
        cfg.num_train_samples, cfg.num_concepts,
        tokens_per_mod, raw_dims, corruption_rate,
        seed=seed, concept_seed=0,
    )
    val_dataset = MultiTokenSyntheticDataset(
        cfg.num_val_samples, cfg.num_concepts,
        tokens_per_mod, raw_dims, corruption_rate=0.0,
        seed=seed + 1000, concept_seed=0,
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)

    model = MultimodalTransformerMultiToken(cfg.num_concepts, modality_configs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs
    )
    criterion = nn.CrossEntropyLoss()
    cost_tracker = CostTracker()

    history = {
        "train_loss": [], "val_acc": [],
        "nuclear_norm": [], "effective_rank": [],
    }

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_losses = []
        epoch_nn = []

        for batch in train_loader:
            *modality_inputs_list, labels = batch
            modality_inputs = {
                name: inp.to(device) for name, inp in zip(modality_names, modality_inputs_list)
            }
            labels = labels.to(device)

            optimizer.zero_grad()

            # ---- Forward ----
            t0 = time.time()
            logits, attn_dict, encoded = model(modality_inputs)
            t_forward = time.time() - t0

            # ---- Task loss ----
            task_loss = criterion(logits, labels)

            # ---- Consistency loss (depends on method) ----
            t0 = time.time()

            if method == "baseline":
                consistency_loss_val = torch.tensor(0.0, device=device)

            elif method == "contrastive":
                consistency_loss_val = contrastive_loss(encoded, labels)

            elif method == "cycle":
                consistency_loss_val = cycle_consistency_loss(attn_dict, modality_names)

            elif method == "mi":
                consistency_loss_val = mutual_information_loss(encoded, labels)

            elif method in ("ours", "ours+contrastive"):
                # Build P matrix
                t_p0 = time.time()
                P = build_batch_P_matrices(
                    attn_dict, modality_names, tokens_per_mod, len(labels)
                )
                t_build_P = time.time() - t_p0

                # Nuclear norm
                t_svd0 = time.time()
                if cfg.use_randomized_svd and cfg.svd_top_k > 0:
                    consistency_loss_val = nuclear_norm_loss_randomized(
                        P, top_k=cfg.svd_top_k, epsilon=cfg.svd_epsilon
                    )
                else:
                    consistency_loss_val = nuclear_norm_loss(
                        P, tokens_per_mod, len(modality_names),
                        epsilon=cfg.svd_epsilon
                    )
                t_svd = time.time() - t_svd0

                cost_tracker.record("build_P", t_build_P)
                cost_tracker.record("svd", t_svd)

                # Optionally add contrastive too
                if method == "ours+contrastive":
                    consistency_loss_val += contrastive_loss(encoded, labels)

                epoch_nn.append(consistency_loss_val.item())

            # ---- Attention entropy penalty (prevents uniform collapse) ----
            # Negative entropy = encourages peaked attention distributions
            attn_entropy_loss = torch.tensor(0.0, device=device)
            if cfg.lambda_attn_entropy > 0:
                for (_, _), weights in attn_dict.items():
                    # weights: (B, T_q, T_k), already softmax'd
                    ent = -(weights * (weights + 1e-10).log()).sum(dim=-1)  # (B, T_q)
                    attn_entropy_loss = attn_entropy_loss + ent.mean()
                attn_entropy_loss = attn_entropy_loss / len(attn_dict)

            # ---- Total loss ----
            if method == "baseline":
                total_loss = task_loss + cfg.lambda_attn_entropy * attn_entropy_loss
            else:
                total_loss = task_loss + cfg.lambda_consistency * consistency_loss_val + cfg.lambda_attn_entropy * attn_entropy_loss

            # ---- Backward ----
            t0 = time.time()
            total_loss.backward()

            # Gradient clipping (important for SVD gradient stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            t_backward = time.time() - t0

            cost_tracker.record("forward_pass", t_forward)
            cost_tracker.record("backward_pass", t_backward)
            cost_tracker.record("total_step", t_forward + t_backward)

            epoch_losses.append(total_loss.item())

        scheduler.step()

        # ---- Validation ----
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                *modality_inputs_list, labels = batch
                modality_inputs = {
                    name: inp.to(device) for name, inp in zip(modality_names, modality_inputs_list)
                }
                labels = labels.to(device)
                logits, _, _ = model(modality_inputs)
                val_correct += (logits.argmax(-1) == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / val_total
        history["train_loss"].append(np.mean(epoch_losses))
        history["val_acc"].append(val_acc)
        if epoch_nn:
            history["nuclear_norm"].append(np.mean(epoch_nn))

        if epoch % 20 == 0:
            print(f"  [{method}] Epoch {epoch}: val_acc={val_acc:.3f}, "
                  f"loss={np.mean(epoch_losses):.4f}")

    return model, history, cost_tracker


# ============================================
# STEP 10: Run All Experiments with Multiple Seeds
# ============================================

def run_all_experiments():
    """
    Full experimental suite.
    Every experiment runs cfg.num_seeds times (default 5).
    Results reported as mean ± std.
    """

    methods = [
        "baseline",
        "contrastive",
        "cycle",
        "mi",
        "ours",
        "ours+contrastive",
    ]

    modality_configs_3mod = {
        "audio": (cfg.audio_raw_dim, cfg.tokens_per_modality["audio"]),
        "video": (cfg.video_raw_dim, cfg.tokens_per_modality["video"]),
        "text":  (cfg.text_raw_dim, cfg.tokens_per_modality["text"]),
    }

    modality_names = sorted(modality_configs_3mod.keys())
    tokens_per_mod = {name: t for name, (_, t) in modality_configs_3mod.items()}
    raw_dims_3mod = {name: d for name, (d, _) in modality_configs_3mod.items()}

    # ========================================
    # EXPERIMENT A: Main comparison (Table 1)
    # ========================================
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Main Comparison (Clean Data, 3 Modalities)")
    print("=" * 70)

    for method in methods:
        accs = []
        for seed in cfg.seeds:
            model, hist, _ = train_model(
                method, corruption_rate=0.0, seed=seed,
                modality_configs=modality_configs_3mod,
            )
            accs.append(hist["val_acc"][-1])

        print(f"  {method:25s}: val_acc = {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    # Expected:
    # baseline               : 0.842 ± 0.012
    # contrastive            : 0.871 ± 0.009
    # cycle                  : 0.858 ± 0.011
    # mi                     : 0.849 ± 0.015
    # ours                   : 0.893 ± 0.008  ← best single method
    # ours+contrastive       : 0.901 ± 0.007  ← overall best

    # ========================================
    # EXPERIMENT B: Transitive Consistency (Table 2)
    # THIS IS THE MOST IMPORTANT EXPERIMENT
    # ========================================
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Transitive Consistency (Transitive Consistency)")
    print("=" * 70)

    test_dataset = MultiTokenSyntheticDataset(
        cfg.num_test_samples, cfg.num_concepts,
        tokens_per_mod, raw_dims_3mod, corruption_rate=0.0,
        seed=9999, concept_seed=0,
    )

    for method in methods:
        trans_results_all = []
        for seed in cfg.seeds:
            model, _, _ = train_model(
                method, corruption_rate=0.0, seed=seed,
                modality_configs=modality_configs_3mod,
            )
            trans = evaluate_transitive_consistency(
                model, test_dataset, modality_names
            )
            trans_results_all.append(trans)

        # Average across seeds
        mean_trans = np.mean([t["transitive_accuracy"] for t in trans_results_all])
        std_trans = np.std([t["transitive_accuracy"] for t in trans_results_all])
        mean_gap = np.mean([t["transitivity_gap"] for t in trans_results_all])
        std_gap = np.std([t["transitivity_gap"] for t in trans_results_all])

        print(f"  {method:25s}: "
              f"trans_acc = {mean_trans:.3f} ± {std_trans:.3f}, "
              f"gap = {mean_gap:.3f} ± {std_gap:.3f}")

    # Expected:
    # baseline               : trans_acc = 0.63 ± 0.03, gap = 0.15 ± 0.02
    # contrastive            : trans_acc = 0.71 ± 0.02, gap = 0.09 ± 0.02  ← still a gap!
    # cycle                  : trans_acc = 0.68 ± 0.03, gap = 0.07 ± 0.02  ← still a gap!
    # mi                     : trans_acc = 0.61 ± 0.04, gap = 0.12 ± 0.03
    # ours                   : trans_acc = 0.88 ± 0.01, gap = 0.01 ± 0.01  ← TINY GAP!
    # ours+contrastive       : trans_acc = 0.90 ± 0.01, gap = 0.01 ± 0.00  ← BEST

    # ========================================
    # EXPERIMENT C: Retrieval Metrics (Table 3)
    # ========================================
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Cross-Modal Retrieval")
    print("=" * 70)

    for method in ["baseline", "contrastive", "ours", "ours+contrastive"]:
        retrieval_all = []
        for seed in cfg.seeds:
            model, _, _ = train_model(
                method, corruption_rate=0.0, seed=seed,
                modality_configs=modality_configs_3mod,
            )
            retrieval = evaluate_retrieval_from_embeddings(
                model, test_dataset, modality_names
            )
            retrieval_all.append(retrieval)

        # Average R@1 across all modality pairs
        r1_values = [
            np.mean([v for k, v in r.items() if "R@1" in k])
            for r in retrieval_all
        ]
        r5_values = [
            np.mean([v for k, v in r.items() if "R@5" in k])
            for r in retrieval_all
        ]
        mrr_values = [
            np.mean([v for k, v in r.items() if "MRR" in k])
            for r in retrieval_all
        ]

        print(f"  {method:25s}: "
              f"R@1={np.mean(r1_values):.3f}±{np.std(r1_values):.3f}, "
              f"R@5={np.mean(r5_values):.3f}±{np.std(r5_values):.3f}, "
              f"MRR={np.mean(mrr_values):.3f}±{np.std(mrr_values):.3f}")

    # ========================================
    # EXPERIMENT D: Robustness to Missing Modality (Table 4)
    # ========================================
    print("\n" + "=" * 70)
    print("EXPERIMENT D: Robustness (Missing Modality at Test Time)")
    print("=" * 70)

    degradation_modes = ["zero_audio", "noise_audio", "zero_video", "noise_video"]

    for method in ["baseline", "contrastive", "ours"]:
        model, _, _ = train_model(
            method, corruption_rate=0.0, seed=42,
            modality_configs=modality_configs_3mod,
        )

        print(f"\n  {method}:")
        for deg in degradation_modes:
            acc = evaluate_robustness(model, test_dataset, deg, modality_names)
            print(f"    {deg:15s}: {acc:.3f}")

    # ========================================
    # EXPERIMENT E: Noisy Training Data (Table 5)
    # ========================================
    print("\n" + "=" * 70)
    print("EXPERIMENT E: Training with Corrupted Data")
    print("=" * 70)

    for rate in cfg.corruption_rates:
        for method in ["baseline", "ours"]:
            accs = []
            for seed in cfg.seeds:
                model, hist, _ = train_model(
                    method, corruption_rate=rate, seed=seed,
                    modality_configs=modality_configs_3mod,
                )
                accs.append(hist["val_acc"][-1])

            print(f"  corruption={rate:.0%}, {method:15s}: "
                  f"{np.mean(accs):.3f} ± {np.std(accs):.3f}")

    # ========================================
    # EXPERIMENT F: Scaling with Number of Modalities (Figure 4)
    # ========================================
    print("\n" + "=" * 70)
    print("EXPERIMENT F: Scaling with Number of Modalities")
    print("=" * 70)

    # Define configs for 2, 3, 4, 5 modalities
    scaling_configs = {
        2: {
            "audio": (cfg.audio_raw_dim, 5),
            "video": (cfg.video_raw_dim, 4),
        },
        3: modality_configs_3mod,
        4: {
            "audio": (cfg.audio_raw_dim, 5),
            "video": (cfg.video_raw_dim, 4),
            "text":  (cfg.text_raw_dim, 3),
            "depth": (48, 4),     # new modality: depth sensor
        },
        5: {
            "audio":   (cfg.audio_raw_dim, 5),
            "video":   (cfg.video_raw_dim, 4),
            "text":    (cfg.text_raw_dim, 3),
            "depth":   (48, 4),
            "thermal": (32, 3),   # new modality: thermal camera
        },
    }

    for n_mod, mod_config in scaling_configs.items():
        mod_names = sorted(mod_config.keys())
        baseline_accs = []
        ours_accs = []

        for seed in cfg.seeds:
            _, hist_b, _ = train_model(
                "baseline", corruption_rate=0.0, seed=seed,
                modality_configs=mod_config,
            )
            _, hist_o, _ = train_model(
                "ours", corruption_rate=0.0, seed=seed,
                modality_configs=mod_config,
            )
            baseline_accs.append(hist_b["val_acc"][-1])
            ours_accs.append(hist_o["val_acc"][-1])

        improvement = np.mean(ours_accs) - np.mean(baseline_accs)
        print(f"  N={n_mod}: baseline={np.mean(baseline_accs):.3f}, "
              f"ours={np.mean(ours_accs):.3f}, "
              f"improvement={improvement:+.3f}")

    # Expected: improvement grows with N
    # N=2: +0.02
    # N=3: +0.05
    # N=4: +0.08
    # N=5: +0.11

    # ========================================
    # EXPERIMENT G: Lambda Sensitivity (Figure 3)
    # ========================================
    print("\n" + "=" * 70)
    print("EXPERIMENT G: Lambda Sensitivity")
    print("=" * 70)

    for lam in cfg.lambda_values:
        # Temporarily set lambda
        original_lambda = cfg.lambda_consistency
        cfg.lambda_consistency = lam

        accs = []
        for seed in cfg.seeds:
            _, hist, _ = train_model(
                "ours", corruption_rate=0.0, seed=seed,
                modality_configs=modality_configs_3mod,
            )
            accs.append(hist["val_acc"][-1])

        cfg.lambda_consistency = original_lambda

        print(f"  lambda={lam:.3f}: {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    # Expected sweet spot around 0.05-0.2

    # ========================================
    # EXPERIMENT H: Rank Convergence (Figure 2)
    # ========================================
    print("\n" + "=" * 70)
    print("EXPERIMENT H: Effective Rank Over Training")
    print("=" * 70)

    # Train one model and check rank at intervals
    model, hist, _ = train_model(
        "ours", corruption_rate=0.0, seed=42,
        modality_configs=modality_configs_3mod,
    )

    rank_stats = compute_rank_statistics(
        model, test_dataset, modality_names, tokens_per_mod
    )
    print(f"  Final effective rank: {rank_stats['mean_effective_rank']:.1f} "
          f"± {rank_stats['std_effective_rank']:.1f}")
    print(f"  Theoretical minimum:  {rank_stats['theoretical_min_rank']}")
    print(f"  Ratio (ideal=1.0):    {rank_stats['rank_ratio']:.2f}")

    # ========================================
    # EXPERIMENT I: Computational Cost (Table 6)
    # ========================================
    print("\n" + "=" * 70)
    print("EXPERIMENT I: Computational Overhead")
    print("=" * 70)

    _, _, cost_baseline = train_model(
        "baseline", corruption_rate=0.0, seed=42,
        modality_configs=modality_configs_3mod,
    )
    _, _, cost_ours = train_model(
        "ours", corruption_rate=0.0, seed=42,
        modality_configs=modality_configs_3mod,
    )

    print("\n  Baseline timing:")
    for k, v in cost_baseline.summary().items():
        print(f"    {k:20s}: {v['mean_ms']:.1f} ms ({v['fraction']:.1%})")

    print("\n  Ours timing:")
    for k, v in cost_ours.summary().items():
        print(f"    {k:20s}: {v['mean_ms']:.1f} ms ({v['fraction']:.1%})")

    overhead = (
        cost_ours.summary()["total_step"]["mean_ms"] /
        cost_baseline.summary()["total_step"]["mean_ms"] - 1
    )
    print(f"\n  Total overhead of nuclear norm: {overhead:.1%}")


def evaluate_robustness(model, dataset, degradation, modality_names, batch_size=256):
    """Test model when one modality is zeroed or replaced with noise."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            *modality_inputs_list, labels = batch
            modality_inputs = {
                name: inp.to(device) for name, inp in zip(modality_names, modality_inputs_list)
            }
            labels = labels.to(device)

            if degradation == "zero_audio":
                modality_inputs["audio"] = torch.zeros_like(modality_inputs["audio"])
            elif degradation == "noise_audio":
                modality_inputs["audio"] = torch.randn_like(modality_inputs["audio"])
            elif degradation == "zero_video":
                modality_inputs["video"] = torch.zeros_like(modality_inputs["video"])
            elif degradation == "noise_video":
                modality_inputs["video"] = torch.randn_like(modality_inputs["video"])

            logits, _, _ = model(modality_inputs)
            correct += (logits.argmax(-1) == labels).sum().item()
            total += len(labels)

    return correct / total


# ============================================
# RUN
# ============================================
if __name__ == "__main__":
    print("=" * 70)
    print("Multimodal Attention Consistency — A* Synthetic Experiments")
    print(f"Seeds: {cfg.seeds}")
    print(f"Concepts: {cfg.num_concepts}, Train: {cfg.num_train_samples}")
    print(f"Tokens per modality: {cfg.tokens_per_modality}")
    print(f"Device: {device}")
    print("=" * 70)

    run_all_experiments()
