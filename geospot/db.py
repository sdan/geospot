"""SQLite database writer for training visualization.

Writes training metrics, images, and sample predictions to a SQLite database
that the viz/ frontend reads via SSE.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from PIL import Image

# Paths relative to this file's location
DB_PATH = Path(__file__).parent.parent / "viz" / "live.db"
IMAGES_DIR = Path(__file__).parent.parent / "viz" / "public" / "live" / "images"


def init_db() -> None:
    """Initialize the database schema if it doesn't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            name TEXT,
            type TEXT,  -- 'sft' | 'rl'
            started_at DATETIME,
            config JSON
        );

        CREATE TABLE IF NOT EXISTS steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

            -- Core metrics
            loss REAL,
            mean_reward REAL,
            mean_distance_km REAL,

            -- Training info
            num_tokens INTEGER,
            num_datums INTEGER,
            learning_rate REAL,
            coord_tau REAL,
            elapsed_s REAL,

            FOREIGN KEY (run_id) REFERENCES runs(id)
        );

        -- One row per unique image (shared across 16 rollouts)
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            group_idx INTEGER NOT NULL,
            path TEXT NOT NULL,  -- "/live/images/{run_id}/{step}_{group}.jpg"
            gt_lat REAL,
            gt_lon REAL,
            gt_city TEXT,
            gt_country TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (run_id) REFERENCES runs(id),
            UNIQUE(run_id, step, group_idx)
        );

        -- 16 rows per image (one per rollout sample)
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            sample_idx INTEGER NOT NULL,  -- 0-15 within group

            -- Prediction
            pred_lat REAL,
            pred_lon REAL,
            pred_text TEXT,

            -- Metrics
            distance_km REAL,
            reward REAL,
            format_valid INTEGER,
            mean_logprob REAL,

            FOREIGN KEY (image_id) REFERENCES images(id)
        );

        CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id, step);
        CREATE INDEX IF NOT EXISTS idx_images_run ON images(run_id, step);
        CREATE INDEX IF NOT EXISTS idx_samples_image ON samples(image_id);
    """)
    conn.close()


def save_image(run_id: str, step: int, group_idx: int, image: Image.Image) -> str:
    """Save image to public dir, return URL path for frontend."""
    dir_path = IMAGES_DIR / run_id
    dir_path.mkdir(parents=True, exist_ok=True)
    filename = f"{step:04d}_{group_idx:02d}.jpg"
    file_path = dir_path / filename
    # Convert to RGB if necessary (handles RGBA, P mode, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(file_path, "JPEG", quality=85)
    return f"/live/images/{run_id}/{filename}"


class DBWriter:
    """Writes training data to SQLite for visualization."""

    def __init__(self, run_id: str, run_name: str, run_type: str, config: dict) -> None:
        """Initialize a new training run.

        Args:
            run_id: Unique identifier for this run (e.g. first 8 chars of uuid4)
            run_name: Human-readable name (e.g. HuggingFace repo name)
            run_type: Either 'sft' or 'rl'
            config: Training configuration dict to store as JSON
        """
        init_db()
        self.run_id = run_id
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?)",
            (run_id, run_name, run_type, datetime.now().isoformat(), json.dumps(config)),
        )
        self.conn.commit()

    def log_step(
        self,
        step: int,
        *,
        loss: float | None = None,
        mean_reward: float | None = None,
        mean_distance_km: float | None = None,
        num_tokens: int | None = None,
        num_datums: int | None = None,
        learning_rate: float | None = None,
        coord_tau: float | None = None,
        elapsed_s: float | None = None,
    ) -> None:
        """Log per-step aggregate metrics."""
        self.conn.execute(
            """
            INSERT INTO steps (run_id, step, loss, mean_reward, mean_distance_km,
                               num_tokens, num_datums, learning_rate, coord_tau, elapsed_s)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.run_id,
                step,
                loss,
                mean_reward,
                mean_distance_km,
                num_tokens,
                num_datums,
                learning_rate,
                coord_tau,
                elapsed_s,
            ),
        )
        self.conn.commit()

    def log_image(
        self,
        step: int,
        group_idx: int,
        image: Image.Image,
        gt_lat: float,
        gt_lon: float,
        gt_city: str | None = None,
        gt_country: str | None = None,
    ) -> int:
        """Save image file and insert images row. Returns image_id."""
        path = save_image(self.run_id, step, group_idx, image)
        cursor = self.conn.execute(
            """
            INSERT INTO images (run_id, step, group_idx, path, gt_lat, gt_lon, gt_city, gt_country)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (self.run_id, step, group_idx, path, gt_lat, gt_lon, gt_city, gt_country),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def log_sample(
        self,
        image_id: int,
        sample_idx: int,
        pred_lat: float | None,
        pred_lon: float | None,
        pred_text: str | None,
        distance_km: float | None,
        reward: float,
        format_valid: bool,
        mean_logprob: float | None = None,
    ) -> None:
        """Insert sample row (one of 16 rollouts for an image)."""
        self.conn.execute(
            """
            INSERT INTO samples (image_id, sample_idx, pred_lat, pred_lon, pred_text,
                                 distance_km, reward, format_valid, mean_logprob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                sample_idx,
                pred_lat,
                pred_lon,
                pred_text,
                distance_km,
                reward,
                int(format_valid),
                mean_logprob,
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
