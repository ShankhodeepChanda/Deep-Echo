"""
Demo excerpts: small rectangular cuts of large observations for hosted demos.

The bundled recordings are 1.4-1.8 GB each, too large for a git repository or
a hosted app. An excerpt keeps one time x channel region of a recording as an
ordinary SIGPROC filterbank (total intensity, float32). Its header's ``fch1``
and ``tstart`` are moved to the region's first channel and sample, so the
frequencies and start time shown for it are the real ones.

A JSON sidecar next to the excerpt records where the region sits in the
original. That is what lets the stored held-out rankings, which index the
*original* file's tile partition (see ``provenance.heldout_origins``), be
mapped onto the excerpt: tiles that fall inside it keep their stored scores,
and the usual position check re-scores a few of them to prove the mapping.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from src.filterbank_reader import read_filterbank_header
from src.observatory.provenance import TILE_F, TILE_T
from src.observatory.sources import FilterbankSource, SourceError

SIDECAR_SUFFIX = ".excerpt.json"
SIDECAR_VERSION = 1

# Header keys by value type, matching ``read_filterbank_header``.
_STR_KEYS = {"source_name", "rawdatafile"}
_INT_KEYS = {"telescope_id", "machine_id", "data_type", "nchans", "nbits", "nifs", "barycentric", "pulsarcentric", "nbeams", "ibeam"}
_DBL_KEYS = {"tstart", "tsamp", "fch1", "foff", "refdm", "period", "src_raj", "src_dej", "az_start", "za_start"}


@dataclass(frozen=True)
class Excerpt:
    """Where an excerpt's region sits inside the original recording."""

    parent_file: str
    parent_n_times: int
    parent_nchans: int
    t_offset: int
    f_offset: int

    def to_local(self, origins: np.ndarray) -> np.ndarray:
        """Original-file (t0, f0) origins -> origins in the excerpt's own coordinates."""
        return np.asarray(origins, dtype=np.int64) - np.array([self.t_offset, self.f_offset], dtype=np.int64)


def sidecar_path(fil_path: Union[str, Path]) -> Path:
    path = Path(fil_path)
    return path.parent / (path.stem + SIDECAR_SUFFIX)


def read_sidecar(fil_path: Union[str, Path]) -> Optional[Excerpt]:
    """The excerpt description for ``fil_path``; None when it has no sidecar. Raises ``ValueError`` if malformed."""
    path = sidecar_path(fil_path)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        ex = Excerpt(
            parent_file=str(raw["parent_file"]),
            parent_n_times=int(raw["parent_n_times"]), parent_nchans=int(raw["parent_nchans"]),
            t_offset=int(raw["t_offset"]), f_offset=int(raw["f_offset"]),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{path.name} is not a valid excerpt description: {exc}") from exc
    if min(ex.parent_n_times, ex.parent_nchans) < 1 or min(ex.t_offset, ex.f_offset) < 0:
        raise ValueError(f"{path.name} has out-of-range dimensions or offsets")
    return ex


def _encode_header(header: Dict) -> bytes:
    def s(text: str) -> bytes:
        raw = text.encode("utf-8")
        return struct.pack("<I", len(raw)) + raw

    out = s("HEADER_START")
    for key, value in header.items():
        if key in _STR_KEYS:
            out += s(key) + s(str(value))
        elif key in _INT_KEYS:
            out += s(key) + struct.pack("<I", int(value))
        elif key in _DBL_KEYS:
            out += s(key) + struct.pack("<d", float(value))
        # Anything else is dropped: the reader stops at keys it does not know.
    return out + s("HEADER_END")


def write_excerpt(
    src_path: Union[str, Path],
    out_path: Union[str, Path],
    t0: int, nt: int, f0: int, nf: int,
    chunk_rows: int = 256,
    extra: Optional[Dict] = None,
) -> Excerpt:
    """Write region t[t0:t0+nt] x channels[f0:f0+nf] of a filterbank as a new filterbank plus sidecar.

    Data are written as total intensity exactly as ``FilterbankSource`` reads
    it, so every window of the excerpt is bit-identical to the same window of
    the original and model scores reproduce exactly.
    """
    source = FilterbankSource(src_path)
    m = source.meta
    if t0 < 0 or f0 < 0 or nt < TILE_T or nf < TILE_F or t0 + nt > m.n_times or f0 + nf > m.nchans:
        raise SourceError(f"Excerpt t[{t0}:{t0 + nt}] f[{f0}:{f0 + nf}] does not fit {m.n_times} x {m.nchans} (min one tile)")

    header, _ = read_filterbank_header(src_path)
    header = dict(header)
    header.update(nchans=int(nf), nifs=1, nbits=32, fch1=m.fch1_mhz + f0 * m.foff_mhz)
    if header.get("tstart") is not None:
        header["tstart"] = float(header["tstart"]) + t0 * m.tsamp_s / 86400.0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(_encode_header(header))
        for r in range(t0, t0 + nt, chunk_rows):
            n = min(chunk_rows, t0 + nt - r)
            fh.write(np.ascontiguousarray(source.read_window(r, n, f0, nf), dtype=np.float32).tobytes())
    tmp.replace(out_path)

    excerpt = Excerpt(Path(src_path).name, m.n_times, m.nchans, int(t0), int(f0))
    record = {
        "version": SIDECAR_VERSION,
        "source_name": m.source_name,
        "parent_file": excerpt.parent_file,
        "parent_n_times": excerpt.parent_n_times,
        "parent_nchans": excerpt.parent_nchans,
        "parent_nifs": m.nifs,
        "t_offset": excerpt.t_offset,
        "f_offset": excerpt.f_offset,
        "n_times": int(nt),
        "nchans": int(nf),
        "note": "Region of the original recording, stored as total intensity. Made by scripts/make_demo_excerpts.py.",
    }
    record.update(extra or {})
    sidecar_path(out_path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return excerpt


def choose_window(
    n_times: int,
    nchans: int,
    ranked_origins: np.ndarray,
    budget_bytes: int,
    bytes_per_value: int = 4,
    top_k: int = 300,
) -> Tuple[int, int, int, int]:
    """Pick a region (t0, nt, f0, nf) within ``budget_bytes`` that holds the most highly ranked tiles.

    ``ranked_origins`` are held-out tile origins in rank order (best first).
    Each tile fully inside the region counts ``1 / sqrt(rank)``, so the top
    candidates dominate while a region with many tiles still beats one with
    a single good tile. Region edges are tried at the tile-aligned positions
    of the ``top_k`` best tiles.
    """
    origins = np.asarray(ranked_origins, dtype=np.int64)
    if origins.size == 0:
        raise ValueError("No tiles to choose a window from")
    t, f = origins[:, 0], origins[:, 1]
    weight = 1.0 / np.sqrt(np.arange(1, len(origins) + 1))
    head = origins[:top_k]

    nt_options = {int(n_times)} | {TILE_T * 2 ** k for k in range(32) if TILE_T * 2 ** k < n_times}
    best: Optional[Tuple[float, int, int, int, int]] = None
    for nt in sorted(nt_options):
        nf = min(int(nchans), (budget_bytes // (nt * bytes_per_value)) // TILE_F * TILE_F)
        if nf < TILE_F:
            continue
        t_starts = np.unique(np.clip(head[:, 0] // TILE_T * TILE_T, 0, n_times - nt))
        f_starts = np.unique(np.clip(head[:, 1] // TILE_F * TILE_F, 0, nchans - nf))
        for T0 in t_starts:
            in_t = (t >= T0) & (t + TILE_T <= T0 + nt)
            if not in_t.any():
                continue
            ft, wt = f[in_t], weight[in_t]
            inside = (ft[None, :] >= f_starts[:, None]) & (ft[None, :] + TILE_F <= f_starts[:, None] + nf)
            gains = inside.astype(np.float64) @ wt
            i = int(np.argmax(gains))
            if best is None or gains[i] > best[0]:
                best = (float(gains[i]), int(T0), int(nt), int(f_starts[i]), int(nf))
    if best is None:
        raise ValueError("Budget is too small for a single tile")
    return best[1:]
