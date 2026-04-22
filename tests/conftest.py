"""
Shared fixtures for ChainGuard test suite.
"""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dashboard"))


@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary SQLite database path that is cleaned up after the test."""
    return str(tmp_path / "test.db")


@pytest.fixture
def sample_case():
    """A minimal valid case dict for database.py CRUD tests."""
    from datetime import datetime
    return {
        "id": "CASE-001",
        "title": "Suspicious mixing pattern",
        "detection_type": "Automated",
        "status": "Open",
        "priority": "High",
        "assignee": {
            "id": "analyst-1",
            "name": "Jane Doe",
            "role": "Senior Analyst",
            "avatar": "👩‍💻",
        },
        "created_at": datetime(2026, 1, 15, 10, 30, 0),
        "updated_at": datetime(2026, 1, 15, 10, 30, 0),
        "linked_nodes": [42, 99, 1337],
        "description": "High-value node with mixing service connections",
        "findings": "",
        "timeline": [
            {"time": datetime(2026, 1, 15, 10, 30, 0), "action": "Created", "by": "System"},
        ],
    }


@pytest.fixture
def analytics_db_path(tmp_path):
    """Initialize a fresh analytics database with ETL data loaded."""
    import _lib.analytics_db as adb

    original_path = adb.DB_PATH
    test_path = str(tmp_path / "analytics_test.db")
    adb.DB_PATH = test_path

    adb.init_analytics_db()
    adb.etl_load_all()

    yield test_path

    adb.DB_PATH = original_path
