"""
Short-Term and Long-Term Memory Systems.

Demonstrates:
1. Short-Term Memory: Ephemeral conversation turns (can be backed by LangGraph State/Checkpointer).
2. Long-Term Memory: SQLite persistence of past problems, recommendations, and user feedback
   which survives across application restarts.
3. Simple Learning / Adaptation via recorded farmer feedback.
"""

import sqlite3
import os
from typing import List, Dict, Any, Optional

DB_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "long_term_memory.db")


def init_long_term_db():
    """Initializes SQLite schema for long-term farm memory."""
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS farm_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farm_id TEXT,
            crop TEXT,
            problem TEXT,
            recommendation TEXT,
            user_feedback TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # Pre-seed historical experience if table is empty
    cursor.execute("SELECT COUNT(*) FROM farm_history")
    count = cursor.fetchone()[0]
    if count == 0:
        cursor.execute("""
            INSERT INTO farm_history (farm_id, crop, problem, recommendation, user_feedback)
            VALUES (
                'farm_01',
                'Tomato',
                'Yellow leaves during hot July week',
                'Applied early morning drip irrigation for 30 mins; checked drainage',
                'Helpful: leaf yellowing stopped after 3 days without needing chemical fungicides'
            )
        """)
        conn.commit()
    conn.close()


def query_long_term_memory(farm_id: str = "farm_01", limit: int = 2) -> List[Dict[str, Any]]:
    """Retrieves previous interventions and recorded feedback for this farm."""
    init_long_term_db()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT crop, problem, recommendation, user_feedback, timestamp
        FROM farm_history
        WHERE farm_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (farm_id, limit))
    rows = cursor.fetchall()
    results = [dict(row) for row in rows]
    conn.close()
    return results


def save_feedback(farm_id: str, crop: str, problem: str, recommendation: str, feedback: str):
    """
    Saves farmer feedback to enable simple adaptation / learning over time.
    """
    init_long_term_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO farm_history (farm_id, crop, problem, recommendation, user_feedback)
        VALUES (?, ?, ?, ?, ?)
    """, (farm_id, crop, problem, recommendation, feedback))
    conn.commit()
    conn.close()
    print(f"  [LONG-TERM MEMORY] Recorded intervention feedback into SQLite database.")
