# Multimodal Attention Consistency via Nuclear Norm

Synthetic validation of the nuclear norm penalty on block cross-attention matrices for enforcing transitive multimodal alignment.

## Key Results

| Experiment | Finding |
|---|---|
| Classification (A) | Ours 88.8% vs 84.3% baseline (+4.5%) |
| Transitive Consistency (B) | 2.5x lower attention composition error |
| Modality Scaling (F) | Advantage grows: +2.1% (N=2) to +7.3% (N=5) |
| Corruption Robustness (E) | +4.7% gap at 20% corruption |
| Overhead (I) | 1.1% additional compute |

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv
uv pip install torch numpy matplotlib
```

## Reproducing Results

### Quick single-process test (1 epoch, ~30 seconds)

```bash
uv run python astar_synthetic.py
```

### Full experiment suite (226 runs, 4 GPU workers, ~10 hours)

```bash
uv run python run_parallel.py > experiment_output.log 2>&1
```

This runs all 9 experiments (A-I) across 5 seeds each with 200 epochs. Results are saved to `results.json` and printed as formatted tables in the log.

Requires an NVIDIA GPU. Tested on RTX 5070 (12GB VRAM). Memory usage is ~250MB per worker.

### Experiment B only (transitive consistency, ~1.5 hours)

```bash
uv run python run_exp_b.py > experiment_b_output.log 2>&1
```

### Generate figures

```bash
uv run python make_figures.py
```

Produces `fig1_classification.png` through `fig6_summary.png`.

## Experiments

| ID | Name | Tasks | What it tests |
|---|---|---|---|
| A | Main Comparison | 30 | Classification accuracy across 6 methods |
| B | Transitive Consistency | 30 | Attention composition: P_AC ≈ P_AB @ P_BC |
| C | Cross-Modal Retrieval | 20 | R@1, R@5, MRR across modality pairs |
| D | Robustness | 3 | Missing/noisy modalities at test time |
| E | Corruption Tolerance | 60 | Training with corrupted data (0-50%) |
| F | Modality Scaling | 40 | 2, 3, 4, 5 modalities |
| G | Lambda Sensitivity | 40 | Nuclear norm weight sweep (0.001-1.0) |
| H | Effective Rank | 1 | Rank of P matrix over training |
| I | Computational Overhead | 2 | Per-component timing breakdown |

## Methods Compared

- **Baseline**: Classification loss only
- **Contrastive**: CLIP-style pairwise alignment
- **Cycle**: Round-trip attention consistency (A→B→A ≈ I)
- **MI**: Mutual information maximization (MINE estimator)
- **Ours**: Nuclear norm on block attention matrix P
- **Ours + Contrastive**: Nuclear norm + CLIP-style alignment

## Configuration

Key parameters in `ExperimentConfig` (top of `astar_synthetic.py`):

| Parameter | Default | Description |
|---|---|---|
| `num_concepts` | 200 | Number of distinct concepts |
| `num_epochs` | 200 | Training epochs |
| `modality_visibility` | 0.5 | Fraction of concept dims visible per modality |
| `noise_scale` | 0.3 | Noise added to modality projections |
| `lambda_consistency` | 0.1 | Nuclear norm penalty weight |
| `lambda_attn_entropy` | 0.1 | Attention entropy penalty (prevents uniform collapse) |
