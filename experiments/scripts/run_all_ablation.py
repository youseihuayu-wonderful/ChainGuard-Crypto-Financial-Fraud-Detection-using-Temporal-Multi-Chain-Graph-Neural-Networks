"""
Run All Ablation Experiments (M1-M5) and Generate Results Table

This script runs each model variant sequentially and outputs
a combined results table for the paper.

Usage:
    python experiments/scripts/run_all_ablation.py
"""

import json
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EXPERIMENTS = [
    ("M1", "train_gcn_baseline.py", "GCN Baseline"),
    ("M2", "train_m2_temporal.py", "GCN + Temporal Attention"),
    ("M3", "train_m3_hetero.py", "R-GCN + Heterogeneous Edges"),
    ("M4", "train_m4_thgnn.py", "Full TH-GNN (Temporal + Hetero)"),
    ("M5", "train_m5_crosschain.py", "TH-GNN + Label Propagation"),
]


def main():
    print("=" * 70)
    print("ChainGuard Ablation Study — Running All Experiments")
    print("=" * 70)

    results = {}
    for model_id, script, desc in EXPERIMENTS:
        print(f"\n{'='*70}")
        print(f"Running {model_id}: {desc}")
        print(f"{'='*70}")

        script_path = os.path.join(SCRIPT_DIR, script)
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=1800,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"ERROR in {model_id}:")
            print(result.stderr)
            continue

        # Parse AUC from output
        for line in result.stdout.split("\n"):
            if "AUC-ROC:" in line:
                auc = float(line.split("AUC-ROC:")[1].strip())
            if "F1 (illicit):" in line:
                f1 = float(line.split("F1 (illicit):")[1].strip())
            if "Precision (illicit):" in line:
                precision = float(line.split("Precision (illicit):")[1].strip())
            if "Recall (illicit):" in line:
                recall = float(line.split("Recall (illicit):")[1].strip())

        results[model_id] = {
            "name": desc,
            "auc_roc": auc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    # Save results
    results_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Print final table
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    m1_auc = results.get("M1", {}).get("auc_roc", 0)
    print(f"{'Model':<45s} {'AUC':>7s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s} {'Delta':>7s}")
    print("-" * 80)
    for model_id, _, desc in EXPERIMENTS:
        if model_id in results:
            r = results[model_id]
            delta = r["auc_roc"] - m1_auc if model_id != "M1" else 0
            delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}" if delta != 0 else "  —"
            print(f"{model_id}: {desc:<40s} {r['auc_roc']:.4f}  {r['f1']:.4f}  "
                  f"{r['precision']:.4f}  {r['recall']:.4f}  {delta_str}")


if __name__ == "__main__":
    main()
