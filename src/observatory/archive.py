"""
Optional Breakthrough Listen archive access for choosing an observation.

Thin helpers around ``BreakthroughListenClient``. Nothing is requested until
the caller invokes ``search`` or ``download``; local-file analysis never needs
network access. Downloads are capped by ``max_mb`` and verified by the client.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class ArchiveFile:
    label: str
    url: Optional[str]
    size_mb: Optional[float]
    record: Dict[str, Any]


def _size_mb(value: Any) -> Optional[float]:
    """Archive sizes arrive as GB numbers or strings such as '1.4 GB'. Unknown formats give None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) * 1024.0
    match = re.match(r"\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]*)", str(value))
    if not match:
        return None
    number, unit = float(match.group(1)), match.group(2).upper()
    factor = {"": 1024.0, "GB": 1024.0, "G": 1024.0, "MB": 1.0, "M": 1.0, "KB": 1 / 1024, "TB": 1024.0 * 1024.0}
    return number * factor[unit] if unit in factor else None


def describe(record: Dict[str, Any]) -> ArchiveFile:
    url = record.get("url")
    name = Path(str(url)).name if url else None
    parts = [str(record[k]) for k in ("target", "telescope", "file_type") if record.get(k)]
    label = " · ".join(parts) if parts else (name or "archive file")
    if name:
        label = f"{label} ({name})"
    return ArchiveFile(label, url, _size_mb(record.get("size")), record)


def search(client, target: str, limit: int = 20) -> List[ArchiveFile]:
    """Query the archive for files matching a target name."""
    target = (target or "").strip()
    if not target:
        raise ValueError("Enter a target name to search for.")
    records = client.query_files(target=target, limit=limit)
    return [describe(r) for r in (records or []) if isinstance(r, dict)]


def download(
    client,
    item: ArchiveFile,
    max_mb: float,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Download one archive file, refusing anything larger than ``max_mb``."""
    if not item.url:
        raise ValueError("This archive record has no download link.")
    if item.size_mb is not None and item.size_mb > max_mb:
        raise ValueError(
            f"This file is {item.size_mb:,.0f} MB, above the {max_mb:,.0f} MB limit. "
            "Raise the limit if you want to download it."
        )
    return client.download_file(item.record, max_size_mb=max_mb, progress_callback=progress)
