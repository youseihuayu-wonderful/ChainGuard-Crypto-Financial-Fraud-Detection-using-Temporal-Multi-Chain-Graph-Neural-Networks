"""
SQL Analytics Data Warehouse for ChainGuard
Normalized relational schema over model outputs, predictions, and analyst feedback.
Demonstrates: window functions, CTEs, multi-table JOINs, correlated subqueries,
              conditional aggregation, time-series analysis.
"""

import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chainguard_analytics.db")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "results")


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ═══════════════════════════════════════════
# SCHEMA — 6 normalized tables
# ═══════════════════════════════════════════

def init_analytics_db():
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            node_id     INTEGER PRIMARY KEY,
            risk_score  REAL NOT NULL,
            true_label  INTEGER,
            timestep    INTEGER NOT NULL,
            predicted   INTEGER GENERATED ALWAYS AS (CASE WHEN risk_score > 0.5 THEN 1 ELSE 0 END) STORED
        );

        CREATE TABLE IF NOT EXISTS timestep_stats (
            timestep    INTEGER PRIMARY KEY,
            n_nodes     INTEGER NOT NULL,
            n_edges     INTEGER NOT NULL,
            n_illicit   INTEGER NOT NULL,
            n_licit     INTEGER NOT NULL,
            n_unknown   INTEGER NOT NULL,
            risk_rate   REAL,
            zone        TEXT CHECK(zone IN ('train','val','test'))
        );

        CREATE TABLE IF NOT EXISTS model_results (
            model_id    TEXT PRIMARY KEY,
            model_name  TEXT NOT NULL,
            model_type  TEXT NOT NULL CHECK(model_type IN ('ablation','baseline')),
            auc_roc     REAL,
            f1          REAL,
            precision_  REAL,
            recall_     REAL
        );

        CREATE TABLE IF NOT EXISTS feature_importance (
            model_id    TEXT NOT NULL,
            feature_idx INTEGER NOT NULL,
            feature_name TEXT NOT NULL,
            importance  REAL NOT NULL,
            PRIMARY KEY (model_id, feature_idx),
            FOREIGN KEY (model_id) REFERENCES model_results(model_id)
        );

        CREATE TABLE IF NOT EXISTS node_explanations (
            node_id      INTEGER NOT NULL,
            feature_idx  INTEGER NOT NULL,
            feature_name TEXT,
            value        REAL,
            gradient     REAL,
            contribution REAL,
            PRIMARY KEY (node_id, feature_idx),
            FOREIGN KEY (node_id) REFERENCES predictions(node_id)
        );

        CREATE TABLE IF NOT EXISTS analyst_feedback (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id       INTEGER NOT NULL,
            risk_score    REAL,
            true_label    INTEGER,
            timestep      INTEGER,
            analyst       TEXT DEFAULT 'Analyst',
            feedback_type TEXT NOT NULL CHECK(feedback_type IN ('confirm_fraud','false_positive')),
            notes         TEXT DEFAULT '',
            created_at    TEXT NOT NULL,
            FOREIGN KEY (node_id) REFERENCES predictions(node_id)
        );

        CREATE INDEX IF NOT EXISTS idx_pred_timestep ON predictions(timestep);
        CREATE INDEX IF NOT EXISTS idx_pred_risk ON predictions(risk_score DESC);
        CREATE INDEX IF NOT EXISTS idx_pred_ts_risk ON predictions(timestep, risk_score DESC);
        CREATE INDEX IF NOT EXISTS idx_pred_label ON predictions(true_label);
        CREATE INDEX IF NOT EXISTS idx_feat_model ON feature_importance(model_id);
        CREATE INDEX IF NOT EXISTS idx_expl_node ON node_explanations(node_id);
        CREATE INDEX IF NOT EXISTS idx_fb_node ON analyst_feedback(node_id);
        CREATE INDEX IF NOT EXISTS idx_fb_type ON analyst_feedback(feedback_type);
    """)
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════
# ETL — Load JSON/CSV into SQL tables
# ═══════════════════════════════════════════

def _load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def etl_load_all():
    """One-time ETL: load all experiment results into the analytics database."""
    conn = _get_conn()

    existing = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
    if existing > 0:
        conn.close()
        return False

    # 1. Predictions (8,841 test nodes)
    preds = _load_json("m3_predictions.json")
    if preds and "test_predictions" in preds:
        conn.executemany(
            "INSERT OR IGNORE INTO predictions (node_id, risk_score, true_label, timestep) VALUES (?,?,?,?)",
            [(p["node_id"], p["risk_score"], p["true_label"], p["timestep"])
             for p in preds["test_predictions"]],
        )

    # 2. Timestep stats (49 timesteps)
    ts_stats = _load_json("timestep_stats.json")
    if ts_stats:
        conn.executemany(
            "INSERT OR IGNORE INTO timestep_stats (timestep, n_nodes, n_edges, n_illicit, n_licit, n_unknown, risk_rate, zone) VALUES (?,?,?,?,?,?,?,?)",
            [(int(k), v["nodes"], v.get("edges", 0), v["illicit"], v["licit"], v["unknown"], v["risk_rate"], v["zone"])
             for k, v in ts_stats.items()],
        )

    # 3. Model results (ablation + baselines = 13 models)
    abl = _load_json("ablation_results.json")
    if abl:
        conn.executemany(
            "INSERT OR IGNORE INTO model_results (model_id, model_name, model_type, auc_roc, f1, precision_, recall_) VALUES (?,?,?,?,?,?,?)",
            [(k, v["name"], "ablation", v["auc_roc"], v["f1"], v["precision"], v["recall"])
             for k, v in abl.items()],
        )

    bl = _load_json("baseline_comparison.json")
    if bl and "results" in bl:
        conn.executemany(
            "INSERT OR IGNORE INTO model_results (model_id, model_name, model_type, auc_roc, f1, precision_, recall_) VALUES (?,?,?,?,?,?,?)",
            [(k, v.get("type", k), "baseline", v["auc_roc"], v["f1"], v["precision"], v["recall"])
             for k, v in bl["results"].items() if k != "thgnn_m3_ours"],
        )

    # 4. Feature importance (165 features)
    feat = _load_json("m3_feature_importance.json")
    if feat and "features" in feat:
        conn.executemany(
            "INSERT OR IGNORE INTO feature_importance (model_id, feature_idx, feature_name, importance) VALUES (?,?,?,?)",
            [("M3", f["idx"], f["name"], f["importance"]) for f in feat["features"]],
        )

    # 5. Node explanations (50 nodes × ~10 features each)
    expl = _load_json("m3_node_explanations.json")
    if expl:
        rows = []
        for node in expl:
            for feat_contrib in node.get("top_features", []):
                rows.append((
                    node["node_id"], feat_contrib["feature_idx"],
                    feat_contrib["feature_name"], feat_contrib["value"],
                    feat_contrib["gradient"], feat_contrib["contribution"],
                ))
        conn.executemany(
            "INSERT OR IGNORE INTO node_explanations (node_id, feature_idx, feature_name, value, gradient, contribution) VALUES (?,?,?,?,?,?)",
            rows,
        )

    # 6. Migrate existing feedback from old database
    old_db = os.path.join(os.path.dirname(__file__), "..", "chainguard.db")
    if os.path.exists(old_db):
        old_conn = sqlite3.connect(old_db)
        old_conn.row_factory = sqlite3.Row
        old_rows = old_conn.execute("SELECT * FROM feedback").fetchall()
        old_conn.close()
        for r in old_rows:
            conn.execute(
                "INSERT OR IGNORE INTO analyst_feedback (node_id, risk_score, true_label, timestep, analyst, feedback_type, notes, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (r["node_id"], r["risk_score"], r["true_label"], r["timestep"],
                 r["analyst"], r["feedback_type"], r["notes"], r["created_at"]),
            )

    conn.commit()
    conn.close()
    return True


def get_table_counts():
    conn = _get_conn()
    counts = {}
    for table in ["predictions", "timestep_stats", "model_results", "feature_importance", "node_explanations", "analyst_feedback"]:
        counts[table] = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
    conn.close()
    return counts


# ═══════════════════════════════════════════
# ANALYTICAL QUERIES — 7 advanced SQL techniques
# ═══════════════════════════════════════════

# ── Query 1: Window Functions — Risk ranking within timestep ──

Q1_SQL = """
SELECT node_id, timestep, risk_score, true_label,
    RANK() OVER (PARTITION BY timestep ORDER BY risk_score DESC)
        AS rank_in_timestep,
    NTILE(10) OVER (PARTITION BY timestep ORDER BY risk_score DESC)
        AS risk_decile,
    ROUND(risk_score - AVG(risk_score) OVER (PARTITION BY timestep), 4)
        AS deviation_from_mean,
    ROUND(risk_score - LAG(risk_score) OVER (
        PARTITION BY timestep ORDER BY risk_score DESC
    ), 4) AS gap_to_previous
FROM predictions
WHERE true_label IN (0, 1)
ORDER BY timestep, rank_in_timestep
LIMIT 50
"""

def query_risk_ranking():
    conn = _get_conn()
    rows = conn.execute(Q1_SQL).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Query 2: CTE — Detection precision by risk tier (Pareto) ──

Q2_SQL = """
WITH risk_tiers AS (
    SELECT node_id, true_label, risk_score,
        CASE WHEN risk_score >= 0.95 THEN 'Critical'
             WHEN risk_score >= 0.80 THEN 'High'
             WHEN risk_score >= 0.50 THEN 'Medium'
             ELSE 'Low' END AS tier
    FROM predictions
    WHERE true_label IN (0, 1)
),
tier_metrics AS (
    SELECT tier,
        COUNT(*) AS total_flagged,
        SUM(CASE WHEN true_label = 1 THEN 1 ELSE 0 END) AS true_positives,
        SUM(CASE WHEN true_label = 0 THEN 1 ELSE 0 END) AS false_positives,
        AVG(risk_score) AS avg_score
    FROM risk_tiers
    GROUP BY tier
),
cumulative AS (
    SELECT tier, total_flagged, true_positives, false_positives, avg_score,
        SUM(true_positives) OVER (ORDER BY avg_score DESC) AS running_tp,
        SUM(total_flagged) OVER (ORDER BY avg_score DESC) AS running_total,
        (SELECT SUM(CASE WHEN true_label = 1 THEN 1 ELSE 0 END) FROM predictions
         WHERE true_label IN (0, 1)) AS total_illicit
    FROM tier_metrics
)
SELECT tier, total_flagged, true_positives, false_positives,
    ROUND(true_positives * 100.0 / MAX(total_flagged, 1), 1) AS precision_pct,
    ROUND(running_tp * 100.0 / MAX(total_illicit, 1), 1) AS cumulative_recall_pct,
    ROUND(running_tp * 1.0 / MAX(running_total, 1), 3) AS cumulative_precision
FROM cumulative
ORDER BY avg_score DESC
"""

def query_detection_tiers():
    conn = _get_conn()
    rows = conn.execute(Q2_SQL).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Query 3: Multi-table JOIN — Analyst feedback vs ground truth ──

Q3_SQL = """
SELECT
    p.timestep,
    ts.zone AS data_split,
    ROUND(ts.risk_rate, 2) AS timestep_risk_rate,
    ROUND(p.risk_score, 4) AS risk_score,
    p.true_label,
    f.feedback_type,
    f.analyst,
    f.created_at,
    CASE
        WHEN f.feedback_type = 'confirm_fraud' AND p.true_label = 1 THEN 'True Positive'
        WHEN f.feedback_type = 'confirm_fraud' AND p.true_label = 0 THEN 'False Positive'
        WHEN f.feedback_type = 'false_positive' AND p.true_label = 1 THEN 'Missed Fraud'
        WHEN f.feedback_type = 'false_positive' AND p.true_label = 0 THEN 'True Negative'
    END AS outcome
FROM analyst_feedback f
JOIN predictions p ON f.node_id = p.node_id
JOIN timestep_stats ts ON p.timestep = ts.timestep
ORDER BY f.created_at DESC
"""

def query_feedback_analysis():
    conn = _get_conn()
    rows = conn.execute(Q3_SQL).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Query 4: Correlated Subquery — Model blind spots ──

Q4_SQL = """
SELECT p.node_id, p.timestep, ROUND(p.risk_score, 4) AS risk_score, p.true_label,
    ROUND((SELECT AVG(p2.risk_score) FROM predictions p2
     WHERE p2.timestep = p.timestep), 4) AS timestep_avg,
    ROUND(p.risk_score - (SELECT AVG(p2.risk_score) FROM predictions p2
     WHERE p2.timestep = p.timestep), 4) AS deviation,
    (SELECT COUNT(*) FROM predictions p3
     WHERE p3.timestep = p.timestep AND p3.risk_score > p.risk_score
    ) AS nodes_ranked_higher,
    ts.n_illicit AS illicit_in_timestep
FROM predictions p
JOIN timestep_stats ts ON p.timestep = ts.timestep
WHERE p.true_label = 1
  AND p.risk_score < (
      SELECT AVG(p2.risk_score) FROM predictions p2
      WHERE p2.timestep = p.timestep
  )
ORDER BY p.risk_score ASC
LIMIT 20
"""

def query_blind_spots():
    conn = _get_conn()
    rows = conn.execute(Q4_SQL).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Query 5: Window + CTE — Feature importance Pareto analysis ──

Q5_SQL = """
WITH ranked AS (
    SELECT feature_name, importance,
        ROW_NUMBER() OVER (ORDER BY importance DESC) AS rank,
        SUM(importance) OVER (ORDER BY importance DESC) AS running_sum,
        SUM(importance) OVER () AS grand_total
    FROM feature_importance
    WHERE model_id = 'M3'
)
SELECT feature_name,
    ROUND(importance, 4) AS importance,
    rank,
    ROUND(running_sum / MAX(grand_total, 0.001) * 100, 1) AS cumulative_pct,
    CASE WHEN running_sum / MAX(grand_total, 0.001) <= 0.50 THEN 'Top 50%'
         WHEN running_sum / MAX(grand_total, 0.001) <= 0.80 THEN 'Top 80%'
         WHEN running_sum / MAX(grand_total, 0.001) <= 0.95 THEN 'Top 95%'
         ELSE 'Tail' END AS pareto_group
FROM ranked
WHERE rank <= 30
"""

def query_feature_pareto():
    conn = _get_conn()
    rows = conn.execute(Q5_SQL).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Query 6: Time-series — Risk trend with spike detection ──

Q6_SQL = """
SELECT timestep, zone, n_illicit, n_licit, n_nodes,
    ROUND(risk_rate, 2) AS risk_rate,
    ROUND(AVG(risk_rate) OVER (
        ORDER BY timestep
        ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
    ), 2) AS moving_avg_5,
    ROUND(risk_rate - LAG(risk_rate) OVER (ORDER BY timestep), 2)
        AS period_change,
    CASE
        WHEN LAG(risk_rate) OVER (ORDER BY timestep) IS NULL THEN 'Normal'
        WHEN risk_rate > 1.5 * AVG(risk_rate) OVER (
            ORDER BY timestep ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) THEN 'SPIKE'
        WHEN risk_rate < 0.5 * AVG(risk_rate) OVER (
            ORDER BY timestep ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) THEN 'DROP'
        ELSE 'Normal'
    END AS anomaly_flag
FROM timestep_stats
ORDER BY timestep
"""

def query_risk_trends():
    conn = _get_conn()
    rows = conn.execute(Q6_SQL).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Query 7: Conditional Aggregation — Model comparison report ──

Q7_SQL = """
SELECT model_type,
    COUNT(*) AS n_models,
    ROUND(AVG(auc_roc), 4) AS avg_auc,
    ROUND(MIN(auc_roc), 4) AS worst_auc,
    ROUND(MAX(auc_roc), 4) AS best_auc,
    (SELECT model_name FROM model_results m2
     WHERE m2.model_type = m.model_type
     ORDER BY auc_roc DESC LIMIT 1) AS top_model,
    ROUND(AVG(precision_), 4) AS avg_precision,
    ROUND(AVG(recall_), 4) AS avg_recall,
    SUM(CASE WHEN auc_roc > 0.85 THEN 1 ELSE 0 END) AS models_above_85,
    ROUND(MAX(auc_roc) - MIN(auc_roc), 4) AS auc_spread
FROM model_results m
GROUP BY model_type
HAVING COUNT(*) >= 1
ORDER BY avg_auc DESC
"""

def query_model_comparison():
    conn = _get_conn()
    rows = conn.execute(Q7_SQL).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Initialize on import ──
init_analytics_db()
etl_load_all()
