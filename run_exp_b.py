"""
Run Experiment B only (Transitive Consistency) with entropy penalty.
6 methods × 5 seeds = 30 tasks.
"""
import torch.multiprocessing as mp
import numpy as np
import time
import sys
import json
from collections import defaultdict


def worker(task):
    """Execute a single training + evaluation task."""
    import astar_synthetic as a

    method = task["method"]
    seed = task["seed"]

    mc = {
        "audio": (a.cfg.audio_raw_dim, a.cfg.tokens_per_modality["audio"]),
        "video": (a.cfg.video_raw_dim, a.cfg.tokens_per_modality["video"]),
        "text":  (a.cfg.text_raw_dim, a.cfg.tokens_per_modality["text"]),
    }
    modality_names = sorted(mc.keys())
    tokens_per_mod = {name: t for name, (_, t) in mc.items()}
    raw_dims = {name: d for name, (d, _) in mc.items()}

    model, hist, _ = a.train_model(method, corruption_rate=0.0, seed=seed,
                                    modality_configs=mc)

    from astar_synthetic import MultiTokenSyntheticDataset, evaluate_transitive_consistency
    test_ds = MultiTokenSyntheticDataset(
        a.cfg.num_test_samples, a.cfg.num_concepts,
        tokens_per_mod, raw_dims, corruption_rate=0.0,
        seed=9999, concept_seed=0,
    )
    trans = evaluate_transitive_consistency(model, test_ds, modality_names)

    return {
        "method": method,
        "seed": seed,
        "val_acc": hist["val_acc"][-1],
        "attn_composition_error": trans["attn_composition_error"],
        "attn_composition_cosine": trans["attn_composition_cosine"],
        "attn_entropy": trans["attn_entropy"],
        "weighted_score": trans["weighted_score"],
        "chain_accuracy": trans["chain_accuracy"],
        "direct_accuracy": trans["direct_accuracy"],
        "agreement": trans["agreement"],
    }


def main():
    mp.set_start_method('spawn', force=True)

    import astar_synthetic as a
    methods = ["baseline", "contrastive", "cycle", "mi", "ours", "ours+contrastive"]
    tasks = [{"method": m, "seed": s} for m in methods for s in a.cfg.seeds]

    n_tasks = len(tasks)
    n_workers = 4

    print("=" * 70)
    print("Experiment B Only — Transitive Consistency (with entropy penalty)")
    print(f"Total tasks: {n_tasks}, Workers: {n_workers}")
    print("=" * 70)
    sys.stdout.flush()

    t_start = time.time()
    results = []

    with mp.Pool(n_workers) as pool:
        for result in pool.imap_unordered(worker, tasks):
            results.append(result)
            done = len(results)
            elapsed = time.time() - t_start
            rate = done / elapsed
            eta = (n_tasks - done) / rate if rate > 0 else 0
            print(f"  [{done}/{n_tasks}] {elapsed/60:.0f}m elapsed, "
                  f"~{eta/60:.0f}m remaining "
                  f"({result['method']}, seed={result['seed']}, "
                  f"val={result['val_acc']:.3f})",
                  flush=True)

    total_time = time.time() - t_start
    print(f"\nAll {n_tasks} tasks completed in {total_time/60:.0f} minutes.")
    print("=" * 70)

    # Save
    def convert(obj):
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open("results_exp_b.json", "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print("Raw results saved to results_exp_b.json")

    # Print table
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Transitive Consistency (Transitive Consistency)")
    print("=" * 70)
    by_method = defaultdict(list)
    for r in results:
        by_method[r["method"]].append(r)

    for method in methods:
        rs = by_method[method]
        val = np.mean([r["val_acc"] for r in rs])
        val_s = np.std([r["val_acc"] for r in rs])
        err = np.mean([r["attn_composition_error"] for r in rs])
        err_s = np.std([r["attn_composition_error"] for r in rs])
        cos = np.mean([r["attn_composition_cosine"] for r in rs])
        cos_s = np.std([r["attn_composition_cosine"] for r in rs])
        ent = np.mean([r["attn_entropy"] for r in rs])
        ent_s = np.std([r["attn_entropy"] for r in rs])
        scr = np.mean([r["weighted_score"] for r in rs])
        scr_s = np.std([r["weighted_score"] for r in rs])
        print(f"  {method:25s}: val={val:.3f}±{val_s:.3f}, "
              f"comp_err={err:.4f}±{err_s:.4f}, "
              f"cosine={cos:.3f}±{cos_s:.3f}, "
              f"entropy={ent:.3f}±{ent_s:.3f}, "
              f"score={scr:.3f}±{scr_s:.3f}")


if __name__ == "__main__":
    main()
