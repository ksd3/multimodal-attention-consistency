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

