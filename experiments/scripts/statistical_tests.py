"""
Statistical Significance Tests for ChainGuard

Computes p-values using paired t-tests and Wilcoxon signed-rank tests
to determine whether M3 (TH-GNN) is statistically significantly better
than each baseline.

Uses REAL multi-seed experiment results from multi_seed_results.json.
"""

import json
import os
import numpy as np
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results")


def main():
    # Load multi-seed results
    with open(os.path.join(RESULTS_DIR, "multi_seed_results.json")) as f:
        data = json.load(f)

    summary = data["summary"]
    seeds = data["seeds"]
    n_seeds = data["n_seeds"]

    m3_aucs = np.array(summary["M3_RGCN"]["per_seed_auc"])

    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 70)
    print(f"Seeds: {seeds}")
    print(f"M3 (TH-GNN) AUC per seed: {m3_aucs}")
    print(f"M3 mean ± std: {m3_aucs.mean():.4f} ± {m3_aucs.std():.4f}")
    print()

    results = {}

    for method, method_data in summary.items():
        if method == "M3_RGCN":
            continue

        other_aucs = np.array(method_data["per_seed_auc"])
        mean_diff = m3_aucs.mean() - other_aucs.mean()

        print(f"--- M3 vs {method} ---")
        print(f"  {method} AUC per seed: {other_aucs}")
        print(f"  {method} mean ± std: {other_aucs.mean():.4f} ± {other_aucs.std():.4f}")
        print(f"  Mean difference: {mean_diff:+.4f}")

        # Paired t-test (parametric)
        if other_aucs.std() == 0:
            # LR is deterministic — use one-sample t-test
            t_stat, p_val_t = stats.ttest_1samp(m3_aucs, other_aucs[0])
            test_type = "one-sample t-test (baseline is deterministic)"
        else:
            t_stat, p_val_t = stats.ttest_rel(m3_aucs, other_aucs)
            test_type = "paired t-test"

        print(f"  {test_type}: t={t_stat:.4f}, p={p_val_t:.4f}")

        # Wilcoxon signed-rank test (non-parametric)
        # Note: Wilcoxon needs n >= 6 for reliable results; with n=3, it's approximate
        try:
            if other_aucs.std() == 0:
                diffs = m3_aucs - other_aucs[0]
                w_stat, p_val_w = stats.wilcoxon(diffs)
            else:
                w_stat, p_val_w = stats.wilcoxon(m3_aucs, other_aucs)
            wilcoxon_note = ""
        except ValueError:
            w_stat, p_val_w = float("nan"), float("nan")
            wilcoxon_note = " (insufficient data for Wilcoxon)"

        print(f"  Wilcoxon: W={w_stat}, p={p_val_w}{wilcoxon_note}")

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((m3_aucs.std()**2 + other_aucs.std()**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else float("inf")
        effect_label = "large" if abs(cohens_d) > 0.8 else ("medium" if abs(cohens_d) > 0.5 else "small")
        print(f"  Cohen's d: {cohens_d:.4f} ({effect_label} effect)")

        # Significance
        sig_005 = p_val_t < 0.05
        sig_010 = p_val_t < 0.10
        sig_label = "***" if p_val_t < 0.01 else ("**" if sig_005 else ("*" if sig_010 else "n.s."))
        print(f"  Significance: {sig_label}")
        print()

        results[method] = {
            "method": method,
            "m3_mean": round(m3_aucs.mean(), 4),
            "other_mean": round(other_aucs.mean(), 4),
            "mean_diff": round(mean_diff, 4),
            "t_statistic": round(t_stat, 4),
            "p_value_ttest": round(p_val_t, 6),
            "test_type": test_type,
            "p_value_wilcoxon": round(p_val_w, 6) if not np.isnan(p_val_w) else None,
            "cohens_d": round(cohens_d, 4) if not np.isinf(cohens_d) else None,
            "effect_size": effect_label,
            "significant_005": bool(sig_005),
            "significant_010": bool(sig_010),
            "n_seeds": n_seeds,
        }

    # Summary table
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Comparison':<25s} {'M3 Mean':>8s} {'Other':>8s} {'Diff':>8s} {'p-value':>8s} {'Sig':>5s} {'Effect':>8s}")
    print("-" * 75)
    for method, r in results.items():
        print(f"M3 vs {method:<18s} {r['m3_mean']:>8.4f} {r['other_mean']:>8.4f} "
              f"{r['mean_diff']:>+8.4f} {r['p_value_ttest']:>8.4f} "
              f"{'Yes' if r['significant_005'] else 'No':>5s} {r['effect_size']:>8s}")

    # Note about sample size
    print(f"\nNote: With only {n_seeds} seeds, statistical power is limited.")
    print("Results should be interpreted with caution. More seeds (5-10) recommended")
    print("for publication-quality significance testing.")

    # Save results
    output = {
        "description": "Statistical significance tests for M3 vs baselines",
        "method": "Paired t-tests and Wilcoxon signed-rank tests on multi-seed AUC-ROC",
        "n_seeds": n_seeds,
        "seeds": seeds,
        "note": f"With {n_seeds} seeds, statistical power is limited. More seeds recommended for publication.",
        "comparisons": results,
    }

    output_path = os.path.join(RESULTS_DIR, "statistical_tests.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
