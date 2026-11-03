"""
Observation catalogue: which observations the dashboard can show.

An observation is a data file (bundled in ``data/raw``, uploaded, or opened in
place from disk) optionally joined to stored results (``artifacts/results``).
Header metadata is read from the file itself; nothing is hard-coded per target.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import RAW_DATA_DIR, RESULTS_DIR, UPLOADS_DIR
from src.observatory.sources import SUPPORTED_EXTENSIONS, SourceError, SourceMeta, open_source

logger = logging.getLogger(__name__)

SUMMARY_FILE = "test_summary_all_targets.json"
REGISTRY_FILE = "registry.json"


@dataclass
class ResultsInfo:
    """Stored held-out evaluation for one target."""

    key: str
    summary: Dict
    rankings_path: Optional[Path]
    threshold: Optional[float]


@dataclass
class Observation:
    obs_id: str
    label: str
    origin: str                       # "bundled" | "uploaded" | "external" | "results-only"
    meta: Optional[SourceMeta] = None
    results: Optional[ResultsInfo] = None
    problems: List[str] = field(default_factory=list)

    @property
    def calibrated_threshold(self) -> Optional[float]:
        return self.results.threshold if self.results else None

    @property
    def file_available(self) -> bool:
        return self.meta is not None


def _safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", text).strip("_") or "observation"


def load_results_index(results_dir: Path = RESULTS_DIR) -> Dict[str, ResultsInfo]:
    """Per-target stored results, keyed by source name. Missing or malformed files yield an empty index."""
    path = Path(results_dir) / SUMMARY_FILE
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return {}
    index: Dict[str, ResultsInfo] = {}
    if not isinstance(raw, dict):
        return index
    for key, summary in raw.items():
        if not isinstance(summary, dict):
            continue
        rankings = Path(results_dir) / f"test_rankings_{key}.csv"
        thr = summary.get("threshold")
        index[str(key)] = ResultsInfo(
            key=str(key), summary=summary,
            rankings_path=rankings if rankings.exists() else None,
            threshold=float(thr) if isinstance(thr, (int, float)) else None,
        )
    return index


# ---------------------------------------------------------------------------
# Upload registry
# ---------------------------------------------------------------------------
def _registry_path(uploads_dir: Path) -> Path:
    return Path(uploads_dir) / REGISTRY_FILE


def read_registry(uploads_dir: Path = UPLOADS_DIR) -> List[Dict]:
    path = _registry_path(uploads_dir)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return [d for d in data if isinstance(d, dict) and "id" in d and "path" in d]
    except (OSError, json.JSONDecodeError):
        logger.warning("Upload registry %s is unreadable; ignoring it", path)
        return []


def _write_registry(entries: List[Dict], uploads_dir: Path) -> None:
    path = _registry_path(uploads_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)
    tmp.replace(path)


def content_key(path: Path, sample_bytes: int = 1 << 20) -> str:
    """Identify a file by content without reading all of it: size plus hashes of its first and last megabyte."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(sample_bytes))
        if size > sample_bytes:
            fh.seek(max(size - sample_bytes, 0))
            h.update(fh.read(sample_bytes))
    return h.hexdigest()


def register_observation(
    path: Path,
    original_name: str,
    external: bool,
    uploads_dir: Path = UPLOADS_DIR,
) -> Tuple[str, bool]:
    """Add a validated file to the registry. Returns ``(observation id, is_new)``.

    ``external=True`` means the file stays where it is (read-only use of a
    large local file); otherwise it is a copy saved in the uploads folder.
    A file whose content is already registered returns the existing id with
    ``is_new=False``, so the same observation never appears twice and its
    reviews stay together.
    """
    entries = read_registry(uploads_dir)
    resolved = str(Path(path).resolve())
    key = content_key(Path(path))
    for entry in entries:
        if entry.get("path") == resolved:
            return str(entry["id"]), False
        if entry.get("content_key") == key and Path(entry["path"]).exists():
            return str(entry["id"]), False
    obs_id = f"up-{uuid.uuid4().hex[:8]}"
    entries.append({
        "id": obs_id, "path": resolved, "original_name": original_name, "external": bool(external),
        "content_key": key, "added_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    _write_registry(entries, uploads_dir)
    return obs_id, True


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_observations(
    raw_dir: Path = RAW_DATA_DIR,
    uploads_dir: Path = UPLOADS_DIR,
    results_dir: Path = RESULTS_DIR,
) -> List[Observation]:
    """Bundled files, registered uploads, then targets that only have stored results."""
    results = load_results_index(results_dir)
    observations: List[Observation] = []
    claimed: set = set()

    candidates = sorted(p for p in Path(raw_dir).glob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    for path in candidates:
        try:
            meta = open_source(path).meta
        except SourceError as exc:
            observations.append(Observation(
                obs_id=_safe_id(path.stem), label=path.name, origin="bundled",
                problems=[f"{path.name}: {exc}"],
            ))
            continue
        info = results.get(meta.source_name)
        if info:
            claimed.add(info.key)
        observations.append(Observation(
            obs_id=_safe_id(meta.source_name), label=meta.source_name, origin="bundled", meta=meta, results=info,
        ))

    for entry in read_registry(uploads_dir):
        path = Path(entry["path"])
        label = str(entry.get("original_name") or path.name)
        if not path.exists():
            observations.append(Observation(
                obs_id=str(entry["id"]), label=label, origin="external" if entry.get("external") else "uploaded",
                problems=[f"The saved file is no longer at {path}."],
            ))
            continue
        try:
            meta = open_source(path).meta
        except SourceError as exc:
            observations.append(Observation(
                obs_id=str(entry["id"]), label=label,
                origin="external" if entry.get("external") else "uploaded", problems=[str(exc)],
            ))
            continue
        observations.append(Observation(
            obs_id=str(entry["id"]), label=f"{meta.source_name} ({label})",
            origin="external" if entry.get("external") else "uploaded", meta=meta,
        ))

    for key, info in results.items():
        if key in claimed:
            continue
        observations.append(Observation(
            obs_id=_safe_id(key), label=key, origin="results-only", results=info,
            problems=["Source data file is not available, so waterfall views and tile coordinates are unavailable."],
        ))
    return observations
