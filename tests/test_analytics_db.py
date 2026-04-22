"""
Tests for dashboard/_lib/analytics_db.py — SQL analytics warehouse.
Validates schema, ETL integrity, and all 7 analytical query functions.
"""

import sqlite3
import pytest

import _lib.analytics_db as adb


@pytest.fixture(autouse=True)
def isolate_analytics(tmp_path):
    """Point analytics_db at a temp file so tests use an isolated database."""
    original = adb.DB_PATH
    test_path = str(tmp_path / "analytics_test.db")
    adb.DB_PATH = test_path
    adb.init_analytics_db()
    adb.etl_load_all()
    yield
    adb.DB_PATH = original


# ═══════════════════════════════════════════
# Schema validation
# ═══════════════════════════════════════════

class TestSchema:
    """Verify all 6 tables exist with correct structure."""

    EXPECTED_TABLES = [
        "predictions", "timestep_stats", "model_results",
        "feature_importance", "node_explanations", "analyst_feedback",
    ]

    def test_all_six_tables_exist(self):
        conn = sqlite3.connect(adb.DB_PATH)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()]
        conn.close()

        for t in self.EXPECTED_TABLES:
            assert t in tables, f"Table '{t}' missing from schema"

    def test_predictions_has_generated_column(self):
        conn = sqlite3.connect(adb.DB_PATH)
        cols = [r[1] for r in conn.execute("PRAGMA table_xinfo(predictions)").fetchall()]
        conn.close()
        assert "predicted" in cols

    def test_model_results_check_constraint(self):
        """model_type must be 'ablation' or 'baseline'."""
        conn = sqlite3.connect(adb.DB_PATH)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO model_results VALUES (?,?,?,?,?,?,?)",
                ("bad", "Bad Model", "invalid_type", 0.5, 0.5, 0.5, 0.5),
            )
        conn.close()

    def test_foreign_key_enforcement(self):
        """feature_importance must reference a valid model_id."""
        conn = sqlite3.connect(adb.DB_PATH)
        conn.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO feature_importance VALUES (?,?,?,?)",
                ("NONEXISTENT_MODEL", 0, "feat_0", 1.0),
            )
        conn.close()


# ═══════════════════════════════════════════
# ETL data integrity
# ═══════════════════════════════════════════

class TestETL:
    """Verify ETL loaded correct data from experiment JSON files."""

    def test_table_row_counts(self):
        counts = adb.get_table_counts()
        assert counts["predictions"] == 8841
        assert counts["timestep_stats"] == 49
        assert counts["model_results"] >= 10
        assert counts["feature_importance"] == 165
        assert counts["node_explanations"] >= 100

    def test_etl_idempotent(self):
        """Running ETL twice does not duplicate data."""
        counts_before = adb.get_table_counts()
        result = adb.etl_load_all()
        counts_after = adb.get_table_counts()

        assert result is False
        assert counts_before == counts_after

    def test_predictions_risk_score_range(self):
        conn = sqlite3.connect(adb.DB_PATH)
        row = conn.execute(
            "SELECT MIN(risk_score) AS lo, MAX(risk_score) AS hi FROM predictions"
        ).fetchone()
        conn.close()
        assert row[0] >= 0.0
        assert row[1] <= 1.0

    def test_timestep_stats_completeness(self):
        """All 49 timesteps present with no gaps."""
        conn = sqlite3.connect(adb.DB_PATH)
        timesteps = [r[0] for r in conn.execute(
            "SELECT timestep FROM timestep_stats ORDER BY timestep"
        ).fetchall()]
        conn.close()
        assert len(timesteps) == 49
        assert timesteps == list(range(1, 50))


# ═══════════════════════════════════════════
# Query 1: Window Functions — Risk Ranking
# ═══════════════════════════════════════════

class TestQ1RiskRanking:

    def test_returns_results(self):
        results = adb.query_risk_ranking()
        assert len(results) == 50

    def test_rank_is_ordered_within_timestep(self):
        results = adb.query_risk_ranking()
        by_ts = {}
        for r in results:
            by_ts.setdefault(r["timestep"], []).append(r)

        for ts, rows in by_ts.items():
            ranks = [r["rank_in_timestep"] for r in rows]
            assert ranks == sorted(ranks), f"Ranks not sorted in timestep {ts}"

    def test_risk_decile_range(self):
        results = adb.query_risk_ranking()
        for r in results:
            assert 1 <= r["risk_decile"] <= 10


# ═══════════════════════════════════════════
# Query 2: CTE — Detection Tiers
# ═══════════════════════════════════════════

class TestQ2DetectionTiers:

    def test_returns_four_tiers(self):
        tiers = adb.query_detection_tiers()
        tier_names = [t["tier"] for t in tiers]
        assert set(tier_names) == {"Critical", "High", "Medium", "Low"}

    def test_cumulative_recall_is_monotonic(self):
        tiers = adb.query_detection_tiers()
        recalls = [t["cumulative_recall_pct"] for t in tiers]
        for i in range(1, len(recalls)):
            assert recalls[i] >= recalls[i - 1], \
                f"Cumulative recall decreased: {recalls[i-1]} -> {recalls[i]}"

    def test_precision_is_percentage(self):
        tiers = adb.query_detection_tiers()
        for t in tiers:
            assert 0 <= t["precision_pct"] <= 100


# ═══════════════════════════════════════════
# Query 3: Multi-table JOIN — Feedback Analysis
# ═══════════════════════════════════════════

class TestQ3FeedbackAnalysis:

    def test_returns_empty_without_feedback(self):
        results = adb.query_feedback_analysis()
        assert results == []

    def test_returns_results_after_feedback_inserted(self):
        conn = sqlite3.connect(adb.DB_PATH)
        conn.execute("PRAGMA foreign_keys=ON")
        # Pick an existing node_id from predictions
        node = conn.execute("SELECT node_id FROM predictions LIMIT 1").fetchone()[0]
        conn.execute(
            "INSERT INTO analyst_feedback (node_id, risk_score, true_label, timestep, feedback_type, created_at) "
            "VALUES (?, 0.9, 1, 10, 'confirm_fraud', '2026-01-15T10:00:00')",
            (node,),
        )
        conn.commit()
        conn.close()

        results = adb.query_feedback_analysis()
        assert len(results) == 1
        assert results[0]["outcome"] in ["True Positive", "False Positive",
                                          "Missed Fraud", "True Negative"]


# ═══════════════════════════════════════════
# Query 4: Correlated Subquery — Blind Spots
# ═══════════════════════════════════════════

class TestQ4BlindSpots:

    def test_all_are_illicit_below_average(self):
        blind = adb.query_blind_spots()
        for b in blind:
            assert b["true_label"] == 1
            assert b["deviation"] < 0

    def test_returns_at_most_20(self):
        blind = adb.query_blind_spots()
        assert len(blind) <= 20


# ═══════════════════════════════════════════
# Query 5: Feature Importance — Pareto
# ═══════════════════════════════════════════

class TestQ5FeaturePareto:

    def test_cumulative_pct_increases(self):
        pareto = adb.query_feature_pareto()
        assert len(pareto) > 0
        pcts = [p["cumulative_pct"] for p in pareto]
        for i in range(1, len(pcts)):
            assert pcts[i] >= pcts[i - 1]

    def test_last_entry_is_near_100_or_below(self):
        pareto = adb.query_feature_pareto()
        last_pct = pareto[-1]["cumulative_pct"]
        assert last_pct <= 100.1

    def test_pareto_groups_are_valid(self):
        pareto = adb.query_feature_pareto()
        valid = {"Top 50%", "Top 80%", "Top 95%", "Tail"}
        for p in pareto:
            assert p["pareto_group"] in valid


# ═══════════════════════════════════════════
# Query 6: Time-Series — Risk Trends
# ═══════════════════════════════════════════

class TestQ6RiskTrends:

    def test_returns_49_timesteps(self):
        trends = adb.query_risk_trends()
        assert len(trends) == 49

    def test_anomaly_flags_are_valid(self):
        trends = adb.query_risk_trends()
        valid = {"SPIKE", "DROP", "Normal"}
        for t in trends:
            assert t["anomaly_flag"] in valid

    def test_moving_average_is_computed(self):
        trends = adb.query_risk_trends()
        for t in trends:
            assert t["moving_avg_5"] is not None
            assert t["moving_avg_5"] >= 0


# ═══════════════════════════════════════════
# Query 7: Conditional Aggregation — Model Comparison
# ═══════════════════════════════════════════

class TestQ7ModelComparison:

    def test_returns_two_categories(self):
        comp = adb.query_model_comparison()
        types = {c["model_type"] for c in comp}
        assert types == {"ablation", "baseline"}

    def test_best_auc_greater_than_worst(self):
        comp = adb.query_model_comparison()
        for c in comp:
            assert c["best_auc"] >= c["worst_auc"]

    def test_top_model_is_not_empty(self):
        comp = adb.query_model_comparison()
        for c in comp:
            assert c["top_model"] is not None
            assert len(c["top_model"]) > 0
