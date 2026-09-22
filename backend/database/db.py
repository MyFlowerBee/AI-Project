"""
RetinaAI - Database Layer
--------------------------
Simple SQLite persistence for analysis history.

We deliberately store the MINIMUM needed to demonstrate the workflow:
no patient names, no personal identifiers - just an analysis id,
timestamp, image reference, predicted class, confidence, per-class
probabilities and a review status. See "Responsible AI" notes in the
README for why we keep this minimal.
"""

import sqlite3
import json
import os
import uuid
from datetime import datetime
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "retina_ai.db")


def init_db():
    """Create the analyses table if it doesn't already exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                image_filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                class_probabilities TEXT NOT NULL,
                review_status TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'demo'
            )
            """
        )
        conn.commit()


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def insert_analysis(image_filename: str, prediction: str, confidence: float,
                     class_probabilities: dict, review_status: str, mode: str) -> dict:
    analysis_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat() + "Z"

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO analyses
                (id, timestamp, image_filename, prediction, confidence,
                 class_probabilities, review_status, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_id,
                timestamp,
                image_filename,
                prediction,
                confidence,
                json.dumps(class_probabilities),
                review_status,
                mode,
            ),
        )
        conn.commit()

    return {
        "id": analysis_id,
        "timestamp": timestamp,
        "image_filename": image_filename,
        "prediction": prediction,
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "review_status": review_status,
        "mode": mode,
    }


def get_history(limit: int = 100) -> list:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()

    results = []
    for row in rows:
        item = dict(row)
        item["class_probabilities"] = json.loads(item["class_probabilities"])
        results.append(item)
    return results


def get_analysis_by_id(analysis_id: str):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()

    if row is None:
        return None

    item = dict(row)
    item["class_probabilities"] = json.loads(item["class_probabilities"])
    return item
