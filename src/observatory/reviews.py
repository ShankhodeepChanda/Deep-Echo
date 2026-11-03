"""
Candidate review persistence (SQLite) and small UI preferences.

Reviews are keyed by the candidate's stable id (observation id + tile origin),
so they survive reruns, restarts and changes of sensitivity. The store lives
under ``data/reviews`` and never touches observation or model files. Every
write either succeeds or raises ``ReviewStoreError``; nothing is dropped silently.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.config import REVIEWS_DIR

PENDING = "pending"
RETAINED = "retained"
FLAGGED = "flagged"
DISMISSED = "dismissed"
STATUSES = (PENDING, RETAINED, FLAGGED, DISMISSED)
STATUS_LABELS = {PENDING: "Pending", RETAINED: "Retained", FLAGGED: "Flagged", DISMISSED: "Dismissed"}


class ReviewStoreError(Exception):
    """A review could not be saved or loaded."""


@dataclass(frozen=True)
class Review:
    candidate_id: str
    obs_id: str
    status: str
    note: str
    updated_utc: str


class ReviewStore:
    def __init__(self, path: Path = REVIEWS_DIR / "reviews.sqlite"):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, timeout=10)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS reviews ("
                "candidate_id TEXT PRIMARY KEY, obs_id TEXT NOT NULL, status TEXT NOT NULL, "
                "note TEXT NOT NULL DEFAULT '', updated_utc TEXT NOT NULL)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_obs ON reviews(obs_id)")
            return conn
        except (sqlite3.Error, OSError) as exc:
            raise ReviewStoreError(f"Review store is unavailable: {exc}") from exc

    def set(self, candidate_id: str, obs_id: str, status: str, note: str = "") -> Review:
        """Save a decision. ``pending`` with no note removes the record."""
        if status not in STATUSES:
            raise ReviewStoreError(f"Unknown review status: {status!r}")
        note = (note or "").strip()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            with closing(self._connect()) as conn, conn:
                if status == PENDING and not note:
                    conn.execute("DELETE FROM reviews WHERE candidate_id = ?", (candidate_id,))
                else:
                    conn.execute(
                        "INSERT INTO reviews(candidate_id, obs_id, status, note, updated_utc) VALUES (?,?,?,?,?) "
                        "ON CONFLICT(candidate_id) DO UPDATE SET status=excluded.status, note=excluded.note, "
                        "updated_utc=excluded.updated_utc",
                        (candidate_id, obs_id, status, note, now),
                    )
        except sqlite3.Error as exc:
            raise ReviewStoreError(f"Could not save the review: {exc}") from exc
        return Review(candidate_id, obs_id, status, note, now)

    def get(self, candidate_id: str) -> Optional[Review]:
        try:
            with closing(self._connect()) as conn:
                row = conn.execute(
                    "SELECT candidate_id, obs_id, status, note, updated_utc FROM reviews WHERE candidate_id = ?",
                    (candidate_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ReviewStoreError(f"Could not read the review: {exc}") from exc
        return Review(*row) if row else None

    def for_observation(self, obs_id: str) -> Dict[str, Review]:
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT candidate_id, obs_id, status, note, updated_utc FROM reviews WHERE obs_id = ?",
                    (obs_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise ReviewStoreError(f"Could not read reviews: {exc}") from exc
        return {r[0]: Review(*r) for r in rows}

    def all(self) -> List[Review]:
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    "SELECT candidate_id, obs_id, status, note, updated_utc FROM reviews ORDER BY updated_utc DESC"
                ).fetchall()
        except sqlite3.Error as exc:
            raise ReviewStoreError(f"Could not read reviews: {exc}") from exc
        return [Review(*r) for r in rows]


# ---------------------------------------------------------------------------
# Preferences (first-run guide)
# ---------------------------------------------------------------------------
class Preferences:
    def __init__(self, path: Path = REVIEWS_DIR / "prefs.json"):
        self.path = Path(path)

    def read(self) -> Dict:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def set(self, key: str, value) -> None:
        data = self.read()
        data[key] = value
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            tmp.replace(self.path)
        except OSError as exc:
            raise ReviewStoreError(f"Could not save preferences: {exc}") from exc


# ---------------------------------------------------------------------------
# Cadence results (so a finished ON/OFF check is not lost on rerun or restart)
# ---------------------------------------------------------------------------
class CadenceLog:
    def __init__(self, path: Path = REVIEWS_DIR / "cadence_results.json"):
        self.path = Path(path)

    def read(self) -> Dict[str, Dict]:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def set(self, candidate_id: str, result: Dict) -> None:
        data = self.read()
        data[candidate_id] = {**result, "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            tmp.replace(self.path)
        except OSError as exc:
            raise ReviewStoreError(f"Could not save the cadence result: {exc}") from exc
