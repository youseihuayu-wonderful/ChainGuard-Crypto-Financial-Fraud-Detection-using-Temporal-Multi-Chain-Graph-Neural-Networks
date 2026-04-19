"""
SQLite Persistent Storage for ChainGuard
Stores cases, analyst feedback, and audit logs across server restarts.
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chainguard.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS cases (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            detection_type TEXT DEFAULT 'Manual',
            status TEXT DEFAULT 'Open',
            priority TEXT DEFAULT 'Medium',
            assignee_id TEXT,
            assignee_name TEXT,
            assignee_role TEXT,
            assignee_avatar TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            linked_nodes TEXT DEFAULT '[]',
            description TEXT DEFAULT '',
            findings TEXT DEFAULT '',
            timeline TEXT DEFAULT '[]'
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id INTEGER NOT NULL,
            risk_score REAL,
            true_label INTEGER,
            timestep INTEGER,
            feedback_type TEXT NOT NULL,
            analyst TEXT DEFAULT 'Analyst',
            created_at TEXT NOT NULL,
            notes TEXT DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_feedback_node ON feedback(node_id);
        CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
    """)
    conn.commit()
    conn.close()


# ── Cases ──

def get_all_cases():
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM cases ORDER BY updated_at DESC").fetchall()
    conn.close()
    cases = []
    for r in rows:
        cases.append(_row_to_case(r))
    return cases


def get_case(case_id):
    conn = _get_conn()
    row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    conn.close()
    return _row_to_case(row) if row else None


def save_case(case):
    conn = _get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO cases
        (id, title, detection_type, status, priority,
         assignee_id, assignee_name, assignee_role, assignee_avatar,
         created_at, updated_at, linked_nodes, description, findings, timeline)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        case["id"], case["title"], case.get("detection_type", "Manual"),
        case["status"], case["priority"],
        case["assignee"]["id"], case["assignee"]["name"],
        case["assignee"]["role"], case["assignee"]["avatar"],
        case["created_at"].isoformat() if isinstance(case["created_at"], datetime) else case["created_at"],
        case["updated_at"].isoformat() if isinstance(case["updated_at"], datetime) else case["updated_at"],
        json.dumps(case.get("linked_nodes", [])),
        case.get("description", ""),
        case.get("findings", ""),
        json.dumps(_serialize_timeline(case.get("timeline", []))),
    ))
    conn.commit()
    conn.close()


def get_next_case_id():
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) as cnt FROM cases").fetchone()
    conn.close()
    return f"CASE-{row['cnt'] + 1:03d}"


def _row_to_case(row):
    return {
        "id": row["id"],
        "title": row["title"],
        "detection_type": row["detection_type"],
        "status": row["status"],
        "priority": row["priority"],
        "assignee": {
            "id": row["assignee_id"],
            "name": row["assignee_name"],
            "role": row["assignee_role"],
            "avatar": row["assignee_avatar"],
        },
        "created_at": datetime.fromisoformat(row["created_at"]),
        "updated_at": datetime.fromisoformat(row["updated_at"]),
        "linked_nodes": json.loads(row["linked_nodes"]),
        "description": row["description"],
        "findings": row["findings"],
        "timeline": _deserialize_timeline(json.loads(row["timeline"])),
    }


def _serialize_timeline(timeline):
    return [
        {
            "time": e["time"].isoformat() if isinstance(e["time"], datetime) else e["time"],
            "action": e["action"],
            "by": e["by"],
        }
        for e in timeline
    ]


def _deserialize_timeline(timeline):
    return [
        {
            "time": datetime.fromisoformat(e["time"]) if isinstance(e["time"], str) else e["time"],
            "action": e["action"],
            "by": e["by"],
        }
        for e in timeline
    ]


# ── Feedback ──

def save_feedback(node_id, risk_score, true_label, timestep, feedback_type, analyst="Analyst", notes=""):
    conn = _get_conn()
    conn.execute("""
        INSERT INTO feedback (node_id, risk_score, true_label, timestep, feedback_type, analyst, created_at, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (node_id, risk_score, true_label, timestep, feedback_type, analyst, datetime.now().isoformat(), notes))
    conn.commit()
    conn.close()


def get_all_feedback():
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM feedback ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_feedback_for_node(node_id):
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM feedback WHERE node_id = ? ORDER BY created_at DESC", (node_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_feedback_stats():
    conn = _get_conn()
    stats = {}
    stats["total"] = conn.execute("SELECT COUNT(*) as c FROM feedback").fetchone()["c"]
    stats["confirmed"] = conn.execute("SELECT COUNT(*) as c FROM feedback WHERE feedback_type = 'confirm_fraud'").fetchone()["c"]
    stats["false_positive"] = conn.execute("SELECT COUNT(*) as c FROM feedback WHERE feedback_type = 'false_positive'").fetchone()["c"]
    stats["reviewed_nodes"] = conn.execute("SELECT COUNT(DISTINCT node_id) as c FROM feedback").fetchone()["c"]
    conn.close()
    return stats


init_db()
