"""
Tests for dashboard/_lib/database.py — SQLite CRUD for cases and analyst feedback.
"""

import os
import pytest
from datetime import datetime

import _lib.database as db


@pytest.fixture(autouse=True)
def isolate_db(tmp_path):
    """Point database.py at a temp file so tests don't touch production data."""
    original = db.DB_PATH
    db.DB_PATH = str(tmp_path / "test_chainguard.db")
    db.init_db()
    yield
    db.DB_PATH = original


class TestCaseCRUD:
    """Full lifecycle: create → read → update → read back."""

    def test_save_and_retrieve_case(self, sample_case):
        db.save_case(sample_case)
        result = db.get_case("CASE-001")

        assert result is not None
        assert result["id"] == "CASE-001"
        assert result["title"] == "Suspicious mixing pattern"
        assert result["priority"] == "High"
        assert result["assignee"]["name"] == "Jane Doe"
        assert result["linked_nodes"] == [42, 99, 1337]

    def test_get_all_cases_returns_list(self, sample_case):
        assert db.get_all_cases() == []

        db.save_case(sample_case)
        cases = db.get_all_cases()
        assert len(cases) == 1
        assert cases[0]["id"] == "CASE-001"

    def test_update_case_preserves_fields(self, sample_case):
        db.save_case(sample_case)

        sample_case["status"] = "In Progress"
        sample_case["findings"] = "Confirmed mixing with 3 hops"
        sample_case["updated_at"] = datetime(2026, 1, 16, 9, 0, 0)
        db.save_case(sample_case)

        result = db.get_case("CASE-001")
        assert result["status"] == "In Progress"
        assert result["findings"] == "Confirmed mixing with 3 hops"
        assert result["title"] == "Suspicious mixing pattern"

    def test_get_nonexistent_case_returns_none(self):
        assert db.get_case("CASE-999") is None

    def test_next_case_id_increments(self, sample_case):
        assert db.get_next_case_id() == "CASE-001"

        db.save_case(sample_case)
        assert db.get_next_case_id() == "CASE-002"

    def test_multiple_cases_ordered_by_updated(self, sample_case):
        db.save_case(sample_case)

        case2 = sample_case.copy()
        case2["id"] = "CASE-002"
        case2["title"] = "Rapid layering"
        case2["updated_at"] = datetime(2026, 1, 17, 12, 0, 0)
        case2["assignee"] = sample_case["assignee"].copy()
        case2["timeline"] = []
        db.save_case(case2)

        cases = db.get_all_cases()
        assert len(cases) == 2
        assert cases[0]["id"] == "CASE-002"

    def test_timeline_serialization_roundtrip(self, sample_case):
        sample_case["timeline"].append({
            "time": datetime(2026, 1, 15, 14, 0, 0),
            "action": "Escalated to senior",
            "by": "analyst-1",
        })
        db.save_case(sample_case)

        result = db.get_case("CASE-001")
        assert len(result["timeline"]) == 2
        assert result["timeline"][1]["action"] == "Escalated to senior"
        assert isinstance(result["timeline"][0]["time"], datetime)


class TestFeedback:
    """Feedback storage, retrieval, and statistics."""

    def test_save_and_retrieve_feedback(self):
        db.save_feedback(node_id=42, risk_score=0.95, true_label=1,
                         timestep=10, feedback_type="confirm_fraud")

        all_fb = db.get_all_feedback()
        assert len(all_fb) == 1
        assert all_fb[0]["node_id"] == 42
        assert all_fb[0]["feedback_type"] == "confirm_fraud"

    def test_get_feedback_for_specific_node(self):
        db.save_feedback(42, 0.95, 1, 10, "confirm_fraud")
        db.save_feedback(99, 0.30, 0, 10, "false_positive")
        db.save_feedback(42, 0.95, 1, 10, "confirm_fraud", analyst="Analyst-2")

        node_fb = db.get_feedback_for_node(42)
        assert len(node_fb) == 2
        assert all(f["node_id"] == 42 for f in node_fb)

    def test_feedback_stats_aggregation(self):
        db.save_feedback(1, 0.9, 1, 5, "confirm_fraud")
        db.save_feedback(2, 0.8, 1, 5, "confirm_fraud")
        db.save_feedback(3, 0.3, 0, 5, "false_positive")
        db.save_feedback(1, 0.9, 1, 5, "confirm_fraud", analyst="B")

        stats = db.get_feedback_stats()
        assert stats["total"] == 4
        assert stats["confirmed"] == 3
        assert stats["false_positive"] == 1
        assert stats["reviewed_nodes"] == 3

    def test_empty_feedback_stats(self):
        stats = db.get_feedback_stats()
        assert stats["total"] == 0
        assert stats["confirmed"] == 0
        assert stats["false_positive"] == 0
        assert stats["reviewed_nodes"] == 0
