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

