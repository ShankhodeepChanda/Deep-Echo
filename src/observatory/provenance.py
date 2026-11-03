"""
Tile provenance: where in the observation each analysed tile came from.

``scripts/evaluate_all_targets.py`` writes ``test_tile_id = <name>_test_<n>``
where ``n`` indexes the held-out partition produced by
``SequentialTrainer.prepare_dataset_partitions``. That partition is a seeded
subsample plus a seeded permutation of the tile grid, so the (time, channel)
origin of every held-out tile can be recomputed without loading any data.

The ``time_offset_sec`` / ``center_freq_mhz`` columns in those CSVs are derived
from the tile *index*, not from where the tile was measured, so they are not
used. ``verify_against_scores`` checks the recovered positions by re-scoring a
few tiles and comparing with the stored scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

TILE_T = 64
TILE_F = 64
# Parameters used when the stored rankings were produced.
PARTITION_SEED = 42
SAMPLE_LIMIT = 28000
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# Files wider than this were sampled as sub-bands of the first 256 time samples.
SUBBAND_MIN_CHANS = 100_000
SUBBAND_TIME_SAMPLES = 256


def grid_origins(n_times: int, nchans: int) -> np.ndarray:
    """All non-overlapping tile origins (t0, f0), time-major like ``extract_spectrogram_tiles``."""
    n_t = (n_times - TILE_T) // TILE_T + 1
    n_f = (nchans - TILE_F) // TILE_F + 1
    if n_t < 1 or n_f < 1:
        return np.empty((0, 2), dtype=np.int64)
    t0 = np.repeat(np.arange(n_t, dtype=np.int64) * TILE_T, n_f)
    f0 = np.tile(np.arange(n_f, dtype=np.int64) * TILE_F, n_t)
    return np.stack([t0, f0], axis=1)


def grid_shape(n_times: int, nchans: int) -> Tuple[int, int]:
    """(tiles along time, tiles along frequency) for the non-overlapping grid."""
    return (max((n_times - TILE_T) // TILE_T + 1, 0), max((nchans - TILE_F) // TILE_F + 1, 0))


def sampled_origins(n_times: int, nchans: int, limit: int = SAMPLE_LIMIT) -> np.ndarray:
    """Tile origins in the order the training pipeline built them (before subsampling)."""
    if nchans <= SUBBAND_MIN_CHANS:
        return grid_origins(n_times, nchans)
    n_sub = limit // 4
    sub_f0 = np.linspace(0, nchans - TILE_F, n_sub, dtype=int)
    t_starts = np.arange(0, SUBBAND_TIME_SAMPLES - TILE_T + 1, TILE_T, dtype=np.int64)
    t0 = np.repeat(t_starts, len(sub_f0))
    f0 = np.tile(sub_f0.astype(np.int64), len(t_starts))
    return np.stack([t0, f0], axis=1)


def heldout_origins(
    n_times: int,
    nchans: int,
    seed: int = PARTITION_SEED,
    limit: int = SAMPLE_LIMIT,
) -> np.ndarray:
    """(t0, f0) of every held-out test tile, in the partition's own order."""
    origins = sampled_origins(n_times, nchans, limit)
    total = len(origins)
    if total == 0:
        return origins
    if limit and total > limit:
        keep = np.random.RandomState(seed).choice(total, limit, replace=False)
        origins = origins[keep]
        total = len(origins)
    perm = np.random.RandomState(seed).permutation(total)
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)
    return origins[perm[n_train + n_val:]]


def parse_test_tile_index(tile_id: str) -> Optional[int]:
    """``ProxCen_S_test_03968`` -> 3968; None if the id does not follow that pattern."""
    try:
        head, _, tail = str(tile_id).rpartition("_")
        if not head.endswith("_test") or not tail.isdigit():
            return None
        return int(tail)
    except Exception:
        return None


@dataclass(frozen=True)
class Provenance:
    """Result of a provenance check."""

    ok: bool
    checked: int
    max_abs_diff: float
    message: str


def verify_against_scores(
    origins: np.ndarray,
    stored_scores: np.ndarray,
    read_window,
    score_windows,
    sample: int = 3,
    tol: float = 1e-4,
) -> Provenance:
    """Re-score ``sample`` of the top-ranked tiles and compare with the stored scores.

    ``read_window(t0, f0)`` returns the raw (64, 64) window; ``score_windows``
    maps a list of windows to model scores. Returns ok=False with a message if
    the recomputed scores differ, which means the recovered positions are wrong
    (different seed, sampling limit or training data) and must not be shown.
    """
    n = min(sample, len(stored_scores))
    if n == 0:
        return Provenance(False, 0, float("nan"), "No stored scores to check against.")
    order = np.argsort(stored_scores)[::-1][:n]
    windows: List[np.ndarray] = [read_window(int(origins[i][0]), int(origins[i][1])) for i in order]
    fresh = np.asarray(score_windows(windows), dtype=float)
    diff = float(np.max(np.abs(fresh - np.asarray(stored_scores)[order])))
    if diff <= tol:
        return Provenance(True, n, diff, f"Recovered tile positions reproduce the stored scores (max difference {diff:.2e}).")
    return Provenance(False, n, diff, f"Recovered tile positions do not reproduce the stored scores (max difference {diff:.2e}).")
