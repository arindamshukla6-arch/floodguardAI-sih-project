"""
FloodGuard AI - SQLite prediction history storage
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from app.config import DB_PATH, FEATURE_NAMES

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    location_name TEXT,
    {", ".join(f"{f} REAL" for f in FEATURE_NAMES)},
    flood_probability REAL NOT NULL,
    risk_level TEXT NOT NULL,
    top_factors TEXT
);
"""


@contextmanager
def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.execute(SCHEMA)


def save_prediction(features: dict, probability: float, risk_level: str,
                     top_factors: list, location_name: Optional[str] = None) -> int:
    """Persist one prediction to history. Returns the new row id."""
    init_db()
    with get_connection() as conn:
        columns = ["timestamp", "location_name"] + FEATURE_NAMES + \
                  ["flood_probability", "risk_level", "top_factors"]
        values = [
            datetime.now(timezone.utc).isoformat(),
            location_name,
        ] + [features.get(f) for f in FEATURE_NAMES] + [
            probability,
            risk_level,
            json.dumps(top_factors),
        ]
        placeholders = ", ".join("?" for _ in values)
        col_str = ", ".join(columns)
        cursor = conn.execute(
            f"INSERT INTO prediction_history ({col_str}) VALUES ({placeholders})",
            values,
        )
        return cursor.lastrowid


def get_recent_predictions(limit: int = 50) -> list:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM prediction_history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("top_factors"):
                try:
                    d["top_factors"] = json.loads(d["top_factors"])
                except (json.JSONDecodeError, TypeError):
                    d["top_factors"] = []
            result.append(d)
        return result


def clear_history():
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM prediction_history")
