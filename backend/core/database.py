"""
SQLite persistence layer -- v2.

Adds: stored image files (on disk, path in DB), optional lat/lon per
prediction, and a single-record detail lookup for the frontend's
click-to-expand view.
"""

import sqlite3
import os
import datetime
import uuid

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DB_PATH = os.path.join(BASE_DIR, "agrismart.db")
UPLOADS_DIR = os.path.join(BASE_DIR, "backend", "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            class_label TEXT NOT NULL,
            display_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            is_healthy INTEGER NOT NULL,
            precaution TEXT,
            image_filename TEXT,
            latitude REAL,
            longitude REAL,
            top_k_json TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sustainability_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            final_score REAL NOT NULL,
            water_efficiency_score REAL,
            crop_health_score REAL,
            resource_use_score REAL
        )
    """)
    # lightweight migration for DBs created before these columns existed
    existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(predictions)")}
    for col, coltype in [("image_filename", "TEXT"), ("latitude", "REAL"),
                          ("longitude", "REAL"), ("top_k_json", "TEXT")]:
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {coltype}")
    conn.commit()
    conn.close()


def save_uploaded_image(image_bytes, original_filename="upload.jpg"):
    """Saves image bytes to disk with a unique name, returns the filename (not full path)."""
    ext = os.path.splitext(original_filename)[1] or ".jpg"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    full_path = os.path.join(UPLOADS_DIR, unique_name)
    with open(full_path, "wb") as f:
        f.write(image_bytes)
    return unique_name


def save_prediction(class_label, display_name, confidence, is_healthy, precaution,
                     image_filename=None, latitude=None, longitude=None, top_k_json=None):
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO predictions
           (timestamp, class_label, display_name, confidence, is_healthy, precaution,
            image_filename, latitude, longitude, top_k_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.datetime.utcnow().isoformat(), class_label, display_name,
         confidence, int(is_healthy), precaution, image_filename, latitude, longitude, top_k_json),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def save_sustainability_score(final_score, breakdown):
    conn = get_connection()
    conn.execute(
        """INSERT INTO sustainability_scores
           (timestamp, final_score, water_efficiency_score, crop_health_score, resource_use_score)
           VALUES (?, ?, ?, ?, ?)""",
        (datetime.datetime.utcnow().isoformat(), final_score,
         breakdown.get("water_efficiency_score"), breakdown.get("crop_health_score"),
         breakdown.get("resource_use_score")),
    )
    conn.commit()
    conn.close()


def get_prediction_history(limit=50):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_prediction_by_id(pred_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_predictions_with_location():
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE latitude IS NOT NULL AND longitude IS NOT NULL ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_dashboard_stats():
    conn = get_connection()

    total = conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]
    healthy_count = conn.execute("SELECT COUNT(*) as c FROM predictions WHERE is_healthy = 1").fetchone()["c"]
    avg_confidence = conn.execute("SELECT AVG(confidence) as a FROM predictions").fetchone()["a"] or 0

    top_diseases = conn.execute("""
        SELECT display_name, COUNT(*) as count
        FROM predictions
        WHERE is_healthy = 0
        GROUP BY display_name
        ORDER BY count DESC
        LIMIT 5
    """).fetchall()

    avg_sustainability = conn.execute("SELECT AVG(final_score) as a FROM sustainability_scores").fetchone()["a"]

    conn.close()
    return {
        "total_predictions": total,
        "healthy_count": healthy_count,
        "diseased_count": total - healthy_count,
        "average_confidence": round(avg_confidence, 3),
        "top_diseases": [dict(r) for r in top_diseases],
        "average_sustainability_score": round(avg_sustainability, 1) if avg_sustainability else None,
    }
