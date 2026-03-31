"""
Multi-seed experiment for statistical validation.
Runs M1 (GCN), M3 (R-GCN), LR, RF, GB across multiple seeds.
Reports mean ± std and paired t-tests.

Usage: python experiments/scripts/run_multi_seed.py
"""

import os, sys, json, random, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.data.elliptic_loader import load_elliptic_csv, temporal_split, add_temporal_edges
from src.models.baselines.gcn import GCNBaseline
from src.models.hetero_gcn import HeteroGCN

SEEDS = [42, 123, 456]  # 3 seeds for CPU feasibility
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/elliptic_bitcoin_dataset")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../experiments/results")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def get_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def run_gcn(data, train_mask, val_mask, test_mask, class_weight, seed, epochs=100):
    set_seed(seed)
    model = GCNBaseline(in_channels=data.x.shape[1], hidden_channels=128, dropout=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_auc, patience, best_state = 0, 0, None

    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], data.y[train_mask].float(), pos_weight=class_weight)
        loss.backward(); opt.step()

        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                vp = torch.sigmoid(model(data.x, data.edge_index)[val_mask]).numpy()
                va = roc_auc_score(data.y[val_mask].numpy(), vp)
            if va > best_auc:
                best_auc, patience = va, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 15: break

    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(data.x, data.edge_index)[test_mask]).numpy()
    return get_metrics(data.y[test_mask].numpy(), probs)


def run_rgcn(data, train_mask, val_mask, test_mask, class_weight, seed, epochs=50):
    set_seed(seed)
    model = HeteroGCN(in_channels=data.x.shape[1], hidden_channels=80, num_relations=2, dropout=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_auc, patience, best_state = 0, 0, None

    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        logits = model(data.x, data.edge_index, data.edge_type)
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], data.y[train_mask].float(), pos_weight=class_weight)
        loss.backward(); opt.step()

        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                vp = torch.sigmoid(model(data.x, data.edge_index, data.edge_type)[val_mask]).numpy()
                va = roc_auc_score(data.y[val_mask].numpy(), vp)
            if va > best_auc:
                best_auc, patience = va, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 10: break

    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(data.x, data.edge_index, data.edge_type)[test_mask]).numpy()
    return get_metrics(data.y[test_mask].numpy(), probs)


def run_ml(name, X_train, y_train, X_test, y_test, seed):
    if name == "LR":
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)
        m.fit(X_train, y_train)
    elif name == "RF":
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)
        m.fit(X_train, y_train)
    elif name == "GB":
        sw = np.ones(len(y_train))
        sw[y_train == 1] = (y_train == 0).sum() / (y_train == 1).sum()
        m = GradientBoostingClassifier(n_estimators=300, max_depth=6, subsample=0.8,
                                       validation_fraction=0.1, n_iter_no_change=20, random_state=seed)
        m.fit(X_train, y_train, sample_weight=sw)
    return get_metrics(y_test, m.predict_proba(X_test)[:, 1])


def main():
    print(f"Multi-Seed Experiment | Seeds: {SEEDS}", flush=True)
    t0 = time.time()

    # Load data once
    print("Loading data...", flush=True)
    data = load_elliptic_csv(DATA_DIR)
    train_mask, val_mask, test_mask = temporal_split(data)
    cw = (data.y[train_mask] == 0).sum().float() / (data.y[train_mask] == 1).sum().float()

    X = data.x.numpy(); y = data.y.numpy()
    X_tr, y_tr = X[train_mask.numpy()], y[train_mask.numpy()]
    X_te, y_te = X[test_mask.numpy()], y[test_mask.numpy()]

    print("Building augmented graph (one time)...", flush=True)
    data_aug = load_elliptic_csv(DATA_DIR)
    data_aug = add_temporal_edges(data_aug, k=5)
    tr_a, va_a, te_a = temporal_split(data_aug)

    results = {}

    # M1: GCN
    print("\n--- M1_GCN ---", flush=True)
    results["M1_GCN"] = []
    for s in SEEDS:
        t1 = time.time()
        m = run_gcn(data, train_mask, val_mask, test_mask, cw, s)
        print(f"  seed={s}: AUC={m['auc_roc']:.4f} ({time.time()-t1:.0f}s)", flush=True)
        results["M1_GCN"].append(m)

    # M3: R-GCN
    print("\n--- M3_RGCN ---", flush=True)
    results["M3_RGCN"] = []
    for s in SEEDS:
        t1 = time.time()
        m = run_rgcn(data_aug, tr_a, va_a, te_a, cw, s)
        print(f"  seed={s}: AUC={m['auc_roc']:.4f} ({time.time()-t1:.0f}s)", flush=True)
        results["M3_RGCN"].append(m)

    # ML baselines
    for name in ["LR", "RF", "GB"]:
        print(f"\n--- {name} ---", flush=True)
        results[name] = []
        for s in SEEDS:
            t1 = time.time()
            m = run_ml(name, X_tr, y_tr, X_te, y_te, s)
            print(f"  seed={s}: AUC={m['auc_roc']:.4f} ({time.time()-t1:.0f}s)", flush=True)
            results[name].append(m)

    # Compute stats
    print("\n" + "=" * 55, flush=True)
    print("RESULTS: mean ± std", flush=True)
    print("=" * 55, flush=True)

    summary = {}
    for name, runs in results.items():
        aucs = [r["auc_roc"] for r in runs]
        f1s = [r["f1"] for r in runs]
        summary[name] = {
            "auc_mean": round(np.mean(aucs), 4), "auc_std": round(np.std(aucs), 4),
            "f1_mean": round(np.mean(f1s), 4), "f1_std": round(np.std(f1s), 4),
            "per_seed_auc": [round(a, 4) for a in aucs],
        }
        print(f"  {name:10s}: AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  |  F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}", flush=True)

    # Paired t-tests
    print("\nPaired t-tests (M3 vs each):", flush=True)
    from scipy import stats
    m3_aucs = [r["auc_roc"] for r in results["M3_RGCN"]]
    for name in results:
        if name == "M3_RGCN": continue
        other = [r["auc_roc"] for r in results[name]]
        t, p = stats.ttest_rel(m3_aucs, other)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        delta = np.mean(m3_aucs) - np.mean(other)
        print(f"  M3 vs {name:10s}: delta={delta:+.4f}, t={t:.2f}, p={p:.4f} {sig}", flush=True)

    # Save
    output = {"seeds": SEEDS, "n_seeds": len(SEEDS), "summary": summary}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "multi_seed_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
