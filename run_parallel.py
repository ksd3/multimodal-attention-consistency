"""
Parallel runner for the full experiment suite.
Dispatches 226 independent training runs across 4 GPU workers.
"""
import torch.multiprocessing as mp
import numpy as np
import time
import sys
import json
from collections import defaultdict


def worker(task):
    """Execute a single training + evaluation task. Runs in a child process."""
    import astar_synthetic as a

    exp = task["experiment"]
    method = task["method"]
    seed = task["seed"]
    corruption_rate = task.get("corruption_rate", 0.0)
    modality_configs = task.get("modality_configs", None)
    lambda_override = task.get("lambda_override", None)

    # Apply lambda override for experiment G
    if lambda_override is not None:
        a.cfg.lambda_consistency = lambda_override

    # Train
    model, hist, cost_tracker = a.train_model(
        method, corruption_rate=corruption_rate, seed=seed,
        modality_configs=modality_configs,
    )

    result = {
        "experiment": exp,
        "method": method,
        "seed": seed,
        "val_acc": hist["val_acc"][-1],
    }

    # Additional evaluation depending on experiment
    evaluate = task.get("evaluate", [])
    mc = modality_configs
    if mc is None:
        mc = {
            "audio": (a.cfg.audio_raw_dim, a.cfg.tokens_per_modality["audio"]),
            "video": (a.cfg.video_raw_dim, a.cfg.tokens_per_modality["video"]),
            "text":  (a.cfg.text_raw_dim, a.cfg.tokens_per_modality["text"]),
        }
    modality_names = sorted(mc.keys())
    tokens_per_mod = {name: t for name, (_, t) in mc.items()}
    raw_dims = {name: d for name, (d, _) in mc.items()}

    if "transitivity" in evaluate:
        from astar_synthetic import MultiTokenSyntheticDataset, evaluate_transitive_consistency
        test_ds = MultiTokenSyntheticDataset(
            a.cfg.num_test_samples, a.cfg.num_concepts,
            tokens_per_mod, raw_dims, corruption_rate=0.0,
            seed=9999, concept_seed=0,
        )
        trans = evaluate_transitive_consistency(model, test_ds, modality_names)
        result["attn_composition_error"] = trans["attn_composition_error"]
        result["attn_composition_cosine"] = trans["attn_composition_cosine"]
        result["attn_entropy"] = trans["attn_entropy"]
        result["weighted_score"] = trans["weighted_score"]
        result["chain_accuracy"] = trans["chain_accuracy"]
        result["direct_accuracy"] = trans["direct_accuracy"]
        result["agreement"] = trans["agreement"]

    if "retrieval" in evaluate:
        from astar_synthetic import MultiTokenSyntheticDataset, evaluate_retrieval_from_embeddings
        test_ds = MultiTokenSyntheticDataset(
            a.cfg.num_test_samples, a.cfg.num_concepts,
            tokens_per_mod, raw_dims, corruption_rate=0.0,
            seed=9999, concept_seed=0,
        )
        retrieval = evaluate_retrieval_from_embeddings(model, test_ds, modality_names)
        result["r1"] = np.mean([v for k, v in retrieval.items() if "R@1" in k])
        result["r5"] = np.mean([v for k, v in retrieval.items() if "R@5" in k])
        result["mrr"] = np.mean([v for k, v in retrieval.items() if "MRR" in k])

    if "robustness" in evaluate:
        from astar_synthetic import MultiTokenSyntheticDataset, evaluate_robustness
        test_ds = MultiTokenSyntheticDataset(
            a.cfg.num_test_samples, a.cfg.num_concepts,
            tokens_per_mod, raw_dims, corruption_rate=0.0,
            seed=9999, concept_seed=0,
        )
        rob = {}
        for deg in ["zero_audio", "noise_audio", "zero_video", "noise_video"]:
            rob[deg] = evaluate_robustness(model, test_ds, deg, modality_names)
        result["robustness"] = rob

    if "rank_stats" in evaluate:
        from astar_synthetic import MultiTokenSyntheticDataset, compute_rank_statistics
        test_ds = MultiTokenSyntheticDataset(
            a.cfg.num_test_samples, a.cfg.num_concepts,
            tokens_per_mod, raw_dims, corruption_rate=0.0,
            seed=9999, concept_seed=0,
        )
        rank = compute_rank_statistics(model, test_ds, modality_names, tokens_per_mod)
        result["mean_effective_rank"] = rank["mean_effective_rank"]
        result["std_effective_rank"] = rank["std_effective_rank"]
        result["theoretical_min_rank"] = rank["theoretical_min_rank"]
        result["rank_ratio"] = rank["rank_ratio"]

    if "cost" in evaluate:
        result["cost_summary"] = cost_tracker.summary()

    # Extra fields for grouping
    if "corruption_rate" in task:
        result["corruption_rate"] = task["corruption_rate"]
    if "lambda_override" in task:
        result["lambda_override"] = task["lambda_override"]
    if "n_mod" in task:
        result["n_mod"] = task["n_mod"]

    return result


def build_all_tasks():
    """Enumerate all 226 independent tasks."""
    import astar_synthetic as a

    tasks = []

    modality_configs_3mod = {
        "audio": (a.cfg.audio_raw_dim, a.cfg.tokens_per_modality["audio"]),
        "video": (a.cfg.video_raw_dim, a.cfg.tokens_per_modality["video"]),
        "text":  (a.cfg.text_raw_dim, a.cfg.tokens_per_modality["text"]),
    }

    methods_all = ["baseline", "contrastive", "cycle", "mi", "ours", "ours+contrastive"]

    # Experiment A: 6 methods × 5 seeds = 30
    for method in methods_all:
        for seed in a.cfg.seeds:
            tasks.append({
                "experiment": "A", "method": method, "seed": seed,
                "modality_configs": modality_configs_3mod,
            })

    # Experiment B: 6 methods × 5 seeds = 30
    for method in methods_all:
        for seed in a.cfg.seeds:
            tasks.append({
                "experiment": "B", "method": method, "seed": seed,
                "modality_configs": modality_configs_3mod,
                "evaluate": ["transitivity"],
            })

    # Experiment C: 4 methods × 5 seeds = 20
    for method in ["baseline", "contrastive", "ours", "ours+contrastive"]:
        for seed in a.cfg.seeds:
            tasks.append({
                "experiment": "C", "method": method, "seed": seed,
                "modality_configs": modality_configs_3mod,
                "evaluate": ["retrieval"],
            })

    # Experiment D: 3 methods × 1 seed = 3 (each evaluates 4 degradations)
    for method in ["baseline", "contrastive", "ours"]:
        tasks.append({
            "experiment": "D", "method": method, "seed": 42,
            "modality_configs": modality_configs_3mod,
            "evaluate": ["robustness"],
        })

    # Experiment E: 2 methods × 6 corruption rates × 5 seeds = 60
    for rate in a.cfg.corruption_rates:
        for method in ["baseline", "ours"]:
            for seed in a.cfg.seeds:
                tasks.append({
                    "experiment": "E", "method": method, "seed": seed,
                    "corruption_rate": rate,
                    "modality_configs": modality_configs_3mod,
                })

    # Experiment F: 2 methods × 4 modality counts × 5 seeds = 40
    scaling_configs = {
        2: {
            "audio": (a.cfg.audio_raw_dim, 5),
            "video": (a.cfg.video_raw_dim, 4),
        },
        3: modality_configs_3mod,
        4: {
            "audio": (a.cfg.audio_raw_dim, 5),
            "video": (a.cfg.video_raw_dim, 4),
            "text":  (a.cfg.text_raw_dim, 3),
            "depth": (48, 4),
        },
        5: {
            "audio":   (a.cfg.audio_raw_dim, 5),
            "video":   (a.cfg.video_raw_dim, 4),
            "text":    (a.cfg.text_raw_dim, 3),
            "depth":   (48, 4),
            "thermal": (32, 3),
        },
    }
    for n_mod, mod_config in scaling_configs.items():
        for method in ["baseline", "ours"]:
            for seed in a.cfg.seeds:
                tasks.append({
                    "experiment": "F", "method": method, "seed": seed,
                    "modality_configs": mod_config,
                    "n_mod": n_mod,
                })

    # Experiment G: 8 lambda values × 5 seeds = 40
    for lam in a.cfg.lambda_values:
        for seed in a.cfg.seeds:
            tasks.append({
                "experiment": "G", "method": "ours", "seed": seed,
                "modality_configs": modality_configs_3mod,
                "lambda_override": lam,
            })

    # Experiment H: 1 run
    tasks.append({
        "experiment": "H", "method": "ours", "seed": 42,
        "modality_configs": modality_configs_3mod,
        "evaluate": ["rank_stats"],
    })

    # Experiment I: 2 runs
    for method in ["baseline", "ours"]:
        tasks.append({
            "experiment": "I", "method": method, "seed": 42,
            "modality_configs": modality_configs_3mod,
            "evaluate": ["cost"],
        })

    return tasks


def print_results(results):
    """Aggregate and print results in table format."""

    by_exp = defaultdict(list)
    for r in results:
        by_exp[r["experiment"]].append(r)

    # ── EXPERIMENT A ──
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Main Comparison (Clean Data, 3 Modalities)")
    print("=" * 70)
    by_method = defaultdict(list)
    for r in by_exp["A"]:
        by_method[r["method"]].append(r["val_acc"])
    for method in ["baseline", "contrastive", "cycle", "mi", "ours", "ours+contrastive"]:
        accs = by_method[method]
        print(f"  {method:25s}: val_acc = {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    # ── EXPERIMENT B ──
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Transitive Consistency (Transitive Consistency)")
    print("=" * 70)
    by_method = defaultdict(list)
    for r in by_exp["B"]:
        by_method[r["method"]].append(r)
    for method in ["baseline", "contrastive", "cycle", "mi", "ours", "ours+contrastive"]:
        rs = by_method[method]
        attn_err = np.mean([r["attn_composition_error"] for r in rs])
        attn_err_s = np.std([r["attn_composition_error"] for r in rs])
        cosine = np.mean([r["attn_composition_cosine"] for r in rs])
        cosine_s = np.std([r["attn_composition_cosine"] for r in rs])
        entropy = np.mean([r["attn_entropy"] for r in rs])
        entropy_s = np.std([r["attn_entropy"] for r in rs])
        wscore = np.mean([r["weighted_score"] for r in rs])
        wscore_s = np.std([r["weighted_score"] for r in rs])
        print(f"  {method:25s}: comp_err={attn_err:.4f}±{attn_err_s:.4f}, "
              f"cosine={cosine:.3f}±{cosine_s:.3f}, "
              f"entropy={entropy:.3f}±{entropy_s:.3f}, "
              f"score={wscore:.3f}±{wscore_s:.3f}")

    # ── EXPERIMENT C ──
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Cross-Modal Retrieval")
    print("=" * 70)
    by_method = defaultdict(list)
    for r in by_exp["C"]:
        by_method[r["method"]].append(r)
    for method in ["baseline", "contrastive", "ours", "ours+contrastive"]:
        rs = by_method[method]
        r1 = [r["r1"] for r in rs]
        r5 = [r["r5"] for r in rs]
        mrr = [r["mrr"] for r in rs]
        print(f"  {method:25s}: R@1={np.mean(r1):.3f}±{np.std(r1):.3f}, "
              f"R@5={np.mean(r5):.3f}±{np.std(r5):.3f}, "
              f"MRR={np.mean(mrr):.3f}±{np.std(mrr):.3f}")

    # ── EXPERIMENT D ──
    print("\n" + "=" * 70)
    print("EXPERIMENT D: Robustness (Missing Modality at Test Time)")
    print("=" * 70)
    for r in by_exp["D"]:
        print(f"\n  {r['method']}:")
        for deg, acc in r["robustness"].items():
            print(f"    {deg:15s}: {acc:.3f}")

    # ── EXPERIMENT E ──
    print("\n" + "=" * 70)
    print("EXPERIMENT E: Training with Corrupted Data")
    print("=" * 70)
    by_key = defaultdict(list)
    for r in by_exp["E"]:
        by_key[(r["corruption_rate"], r["method"])].append(r["val_acc"])
    for rate in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        for method in ["baseline", "ours"]:
            accs = by_key[(rate, method)]
            print(f"  corruption={rate:.0%}, {method:15s}: "
                  f"{np.mean(accs):.3f} ± {np.std(accs):.3f}")

    # ── EXPERIMENT F ──
    print("\n" + "=" * 70)
    print("EXPERIMENT F: Scaling with Number of Modalities")
    print("=" * 70)
    by_key = defaultdict(list)
    for r in by_exp["F"]:
        by_key[(r["n_mod"], r["method"])].append(r["val_acc"])
    for n_mod in [2, 3, 4, 5]:
        ba = by_key[(n_mod, "baseline")]
        oa = by_key[(n_mod, "ours")]
        imp = np.mean(oa) - np.mean(ba)
        print(f"  N={n_mod}: baseline={np.mean(ba):.3f}, "
              f"ours={np.mean(oa):.3f}, improvement={imp:+.3f}")

    # ── EXPERIMENT G ──
    print("\n" + "=" * 70)
    print("EXPERIMENT G: Lambda Sensitivity")
    print("=" * 70)
    by_lam = defaultdict(list)
    for r in by_exp["G"]:
        by_lam[r["lambda_override"]].append(r["val_acc"])
    for lam in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        accs = by_lam[lam]
        print(f"  lambda={lam:.3f}: {np.mean(accs):.3f} ± {np.std(accs):.3f}")

    # ── EXPERIMENT H ──
    print("\n" + "=" * 70)
    print("EXPERIMENT H: Effective Rank Over Training")
    print("=" * 70)
    r = by_exp["H"][0]
    print(f"  Final effective rank: {r['mean_effective_rank']:.1f} "
          f"± {r['std_effective_rank']:.1f}")
    print(f"  Theoretical minimum:  {r['theoretical_min_rank']}")
    print(f"  Ratio (ideal=1.0):    {r['rank_ratio']:.2f}")

    # ── EXPERIMENT I ──
    print("\n" + "=" * 70)
    print("EXPERIMENT I: Computational Overhead")
    print("=" * 70)
    for r in by_exp["I"]:
        print(f"\n  {r['method']} timing:")
        for k, v in r["cost_summary"].items():
            print(f"    {k:20s}: {v['mean_ms']:.1f} ms ({v['fraction']:.1%})")

    baseline_total = [r for r in by_exp["I"] if r["method"] == "baseline"][0]["cost_summary"]["total_step"]["mean_ms"]
    ours_total = [r for r in by_exp["I"] if r["method"] == "ours"][0]["cost_summary"]["total_step"]["mean_ms"]
    overhead = ours_total / baseline_total - 1
    print(f"\n  Total overhead of nuclear norm: {overhead:.1%}")


def main():
    mp.set_start_method('spawn', force=True)

    tasks = build_all_tasks()
    n_tasks = len(tasks)
    n_workers = 4

    print("=" * 70)
    print("Multimodal Attention Consistency — PARALLEL Experiment Suite")
    print(f"Total tasks: {n_tasks}, Workers: {n_workers}")
    print(f"Estimated time: ~12 hours")
    print("=" * 70)
    sys.stdout.flush()

    t_start = time.time()
    results = []
    completed = 0

    with mp.Pool(n_workers) as pool:
        for result in pool.imap_unordered(worker, tasks):
            completed += 1
            elapsed = time.time() - t_start
            rate = completed / elapsed
            eta = (n_tasks - completed) / rate if rate > 0 else 0
            eta_h = eta / 3600
            elapsed_h = elapsed / 3600

            print(f"  [{completed}/{n_tasks}] "
                  f"{elapsed_h:.1f}h elapsed, "
                  f"~{eta_h:.1f}h remaining, "
                  f"{rate:.2f} runs/sec  "
                  f"(last: exp={result['experiment']} {result['method']})",
                  flush=True)

            results.append(result)

    total_time = time.time() - t_start
    print(f"\n\nAll {n_tasks} tasks completed in {total_time/3600:.1f} hours.")
    print("=" * 70)

    # Save raw results
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print("Raw results saved to results.json")

    # Print formatted tables
    print_results(results)


if __name__ == "__main__":
    main()
