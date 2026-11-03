"""
Candidate tables, analysis states and the detection-sensitivity mapping.

Sensitivity mapping (documented; it does not recalibrate the model)
-------------------------------------------------------------------
The model produces one unusualness score per 64 x 64 tile. Which tiles are
shown as candidates is a display decision, defined by *how many* tiles pass:

* **Medium** is the reference. If the observation has a validated threshold
  (99th percentile of validation scores, stored with its results) Medium is
  exactly that threshold. If it has none (for example an uploaded file) no
  threshold has been validated for the data, so Medium is the top 1% of the
  observation's own scores, a ranking rule and not a calibration.
* **Low** shows about half as many candidates as Medium: the strongest ones.
* **High** shows about twice as many as Medium (never fewer than 10 tiles, so
  there is always something to inspect when Medium finds none).

Counts are always recomputed from the loaded scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.observatory.observations import Observation
from src.observatory.provenance import (
    TILE_F, TILE_T, Provenance, grid_origins, grid_shape, heldout_origins,
    parse_test_tile_index,
)
from src.observatory.scoring import ScanResult

logger = logging.getLogger(__name__)

LEVELS = ("Low", "Medium", "High")
DEFAULT_LEVEL = "Medium"
RELATIVE_TOP_FRACTION = 0.01       # Medium for data with no validated threshold
MIN_HIGH_CANDIDATES = 10

FRAME_COLUMNS = [
    "candidate_id", "obs_id", "rank", "score", "t0", "f0",
    "t_start_s", "t_end_s", "f_low_mhz", "f_high_mhz", "f_center_mhz", "tile_ref",
]


class AnalysisKind(str, Enum):
    READY = "ready"                      # candidates available (possibly zero above the cutoff)
    NOT_ANALYZED = "not_analyzed"        # file is loaded but no scores exist yet
    SOURCE_MISSING = "source_missing"    # scores exist but the data file does not
    FAILED = "failed"                    # results exist but could not be used


@dataclass
class CandidateSet:
    kind: AnalysisKind
    message: str = ""
    frame: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=FRAME_COLUMNS))
    basis: str = ""                      # "held-out tiles" | "full scan"
    n_scored: int = 0
    n_grid_total: Optional[int] = None
    has_coordinates: bool = False
    warnings: List[str] = field(default_factory=list)
    provenance: Optional[Provenance] = None

    @property
    def coverage_fraction(self) -> Optional[float]:
        if not self.n_grid_total:
            return None
        return self.n_scored / self.n_grid_total


@dataclass(frozen=True)
class Cutoff:
    value: float
    level: str
    basis: str                           # "validated" | "relative"
    description: str


def _kth_cutoff(sorted_desc: np.ndarray, k: int) -> float:
    """Value such that exactly ``k`` scores are strictly above it (ties may give fewer)."""
    if k <= 0:
        return float(sorted_desc[0]) if sorted_desc.size else float("inf")
    if k >= sorted_desc.size:
        return float(sorted_desc[-1]) - 1.0 if sorted_desc.size else float("inf")
    return float(sorted_desc[k])


def sensitivity_cutoff(level: str, scores: Sequence[float], calibrated_threshold: Optional[float]) -> Cutoff:
    """Score above which a tile is shown as a candidate at ``level`` (see module docstring)."""
    if level not in LEVELS:
        raise ValueError(f"Unknown sensitivity level: {level!r}")
    arr = np.asarray(scores, dtype=float)
    arr = np.sort(arr[np.isfinite(arr)])[::-1]
    if arr.size == 0:
        return Cutoff(float("inf"), level, "relative", "no scores available")

    validated = calibrated_threshold is not None and np.isfinite(calibrated_threshold) and calibrated_threshold > 0
    if validated:
        n_medium = int(np.sum(arr > calibrated_threshold))
    else:
        n_medium = max(1, int(np.ceil(RELATIVE_TOP_FRACTION * arr.size)))

    if level == "Medium":
        if validated:
            return Cutoff(float(calibrated_threshold), level, "validated", "the validated threshold for this observation")
        return Cutoff(_kth_cutoff(arr, n_medium), level, "relative",
                      "top 1% of this observation's own scores (no validated threshold exists for this data)")
    if level == "Low":
        k = max(1, n_medium // 2) if n_medium else 0
        what = "about half as many as Medium: the strongest candidates only"
    else:
        k = min(max(2 * n_medium, MIN_HIGH_CANDIDATES), arr.size)
        what = "about twice as many as Medium, including weaker candidates"
    return Cutoff(_kth_cutoff(arr, k), level, "validated" if validated else "relative", what)


def apply_sensitivity(frame: pd.DataFrame, cutoff: Cutoff) -> pd.DataFrame:
    """Rows scoring above the cutoff, best first."""
    return frame[frame["score"] > cutoff.value].sort_values("score", ascending=False)


# ---------------------------------------------------------------------------
# Building the frame
# ---------------------------------------------------------------------------
def _coordinate_columns(t0: np.ndarray, f0: np.ndarray, obs: Observation) -> dict:
    meta = obs.meta
    f_first = meta.channel_freq_mhz(f0)
    f_last = meta.channel_freq_mhz(f0 + TILE_F - 1)
    half = abs(meta.foff_mhz) / 2
    return {
        "t_start_s": t0 * meta.tsamp_s,
        "t_end_s": (t0 + TILE_T) * meta.tsamp_s,
        "f_low_mhz": np.minimum(f_first, f_last) - half,
        "f_high_mhz": np.maximum(f_first, f_last) + half,
        "f_center_mhz": (f_first + f_last) / 2,
    }


def _finish_frame(obs: Observation, scores: np.ndarray, t0, f0, tile_ref, has_coords: bool) -> pd.DataFrame:
    n = len(scores)
    frame = pd.DataFrame({"obs_id": obs.obs_id, "score": scores.astype(float), "tile_ref": tile_ref})
    if has_coords:
        t0 = np.asarray(t0, dtype=np.int64)
        f0 = np.asarray(f0, dtype=np.int64)
        frame["t0"], frame["f0"] = t0, f0
        for col, values in _coordinate_columns(t0, f0, obs).items():
            frame[col] = values
        frame["candidate_id"] = [f"{obs.obs_id}:t{a}:f{b}" for a, b in zip(t0, f0)]
    else:
        for col in ("t0", "f0", "t_start_s", "t_end_s", "f_low_mhz", "f_high_mhz", "f_center_mhz"):
            frame[col] = np.nan
        frame["candidate_id"] = [f"{obs.obs_id}:{r}" for r in tile_ref]
    frame = frame.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    frame["rank"] = np.arange(1, n + 1)
    return frame[FRAME_COLUMNS]


def candidates_from_scan(obs: Observation, scan: ScanResult) -> CandidateSet:
    """Candidates from a full-observation scan (every non-overlapping tile)."""
    meta = obs.meta
    n_t, n_f = grid_shape(meta.n_times, meta.nchans)
    if scan.scores.shape != (n_t, n_f):
        return CandidateSet(AnalysisKind.FAILED, "The saved scan does not match this file's dimensions. Run the analysis again.")
    origins = grid_origins(meta.n_times, meta.nchans)
    scores = scan.scores.reshape(-1)
    frame = _finish_frame(obs, scores, origins[:, 0], origins[:, 1], [f"scan{i}" for i in range(len(scores))], True)
    return CandidateSet(
        AnalysisKind.READY, basis="full scan", frame=frame, n_scored=len(frame),
        n_grid_total=n_t * n_f, has_coordinates=True,
    )


def candidates_from_heldout(
    obs: Observation,
    score_check: Optional[Callable[[np.ndarray, np.ndarray], Provenance]] = None,
) -> CandidateSet:
    """Candidates from the stored held-out ranking CSV, with measured (recovered) coordinates.

    ``score_check(origins, scores)`` re-scores a few tiles from the source file
    and returns a ``Provenance``; when it fails the coordinates are withheld
    rather than shown wrong.
    """
    info = obs.results
    if info is None or info.rankings_path is None:
        return CandidateSet(AnalysisKind.NOT_ANALYZED, "No stored results for this observation.")
    try:
        df = pd.read_csv(info.rankings_path)
    except Exception as exc:
        return CandidateSet(AnalysisKind.FAILED, f"Stored results could not be read: {exc}")
    needed = {"test_tile_id", "reconstruction_error"}
    if not needed.issubset(df.columns):
        return CandidateSet(
            AnalysisKind.FAILED,
            "Stored results are missing columns: " + ", ".join(sorted(needed - set(df.columns))),
        )
    df = df.dropna(subset=["reconstruction_error"])
    indices = np.array([parse_test_tile_index(t) if parse_test_tile_index(t) is not None else -1 for t in df["test_tile_id"]])
    if (indices < 0).any():
        return CandidateSet(AnalysisKind.FAILED, "Stored results contain tile ids in an unrecognised format.")
    scores = df["reconstruction_error"].to_numpy(dtype=float)

    warnings: List[str] = []
    if obs.meta is None:
        frame = _finish_frame(obs, scores, None, None, [f"test{i:05d}" for i in indices], False)
        return CandidateSet(
            AnalysisKind.SOURCE_MISSING,
            "Scores are available, but the source data file is missing, so tile positions cannot be determined.",
            frame=frame, basis="held-out tiles", n_scored=len(frame), has_coordinates=False,
        )

    meta = obs.meta
    origins_all = heldout_origins(meta.n_times, meta.nchans)
    n_grid_total = int(np.prod(grid_shape(meta.n_times, meta.nchans)))
    if len(indices) == 0:
        return CandidateSet(AnalysisKind.FAILED, "Stored results contain no tiles.")
    if indices.max() >= len(origins_all):
        return CandidateSet(
            AnalysisKind.FAILED,
            f"Stored results list tile {int(indices.max())} but this file's held-out partition has {len(origins_all)} tiles.",
        )
    origins = origins_all[indices]

    provenance = None
    has_coords = True
    if score_check is not None:
        try:
            provenance = score_check(origins, scores)
        except Exception as exc:
            provenance = Provenance(False, 0, float("nan"), f"Position check could not run: {exc}")
        if not provenance.ok:
            has_coords = False
            warnings.append(provenance.message + " Tile positions are not shown.")

    frame = _finish_frame(
        obs, scores, origins[:, 0] if has_coords else None, origins[:, 1] if has_coords else None,
        [f"test{i:05d}" for i in indices], has_coords,
    )
    return CandidateSet(
        AnalysisKind.READY, basis="held-out tiles", frame=frame, n_scored=len(frame), n_grid_total=n_grid_total,
        has_coordinates=has_coords, warnings=warnings, provenance=provenance,
    )


def load_candidate_set(
    obs: Observation,
    scan: Optional[ScanResult] = None,
    score_check: Optional[Callable[[np.ndarray, np.ndarray], Provenance]] = None,
) -> CandidateSet:
    """Best available analysis for an observation: a full scan if one exists, else stored held-out results."""
    if obs.meta is not None and scan is not None:
        return candidates_from_scan(obs, scan)
    if obs.results is not None and obs.results.rankings_path is not None:
        return candidates_from_heldout(obs, score_check)
    if obs.meta is not None:
        return CandidateSet(
            AnalysisKind.NOT_ANALYZED,
            "This observation has not been analysed yet.",
            n_grid_total=int(np.prod(grid_shape(obs.meta.n_times, obs.meta.nchans))) or None,
        )
    return CandidateSet(AnalysisKind.FAILED, "; ".join(obs.problems) or "Observation could not be loaded.")


def filter_candidates(
    frame: pd.DataFrame,
    review_status: Optional[dict] = None,
    status_filter: str = "All",
    assessment: Optional[dict] = None,
    category_filter: Optional[Sequence[str]] = None,
    f_range_mhz: Optional[tuple] = None,
) -> pd.DataFrame:
    """Apply review-status, assessment-category and frequency filters to a candidate frame."""
    out = frame
    if f_range_mhz is not None and "f_center_mhz" in out and out["f_center_mhz"].notna().any():
        lo, hi = f_range_mhz
        out = out[(out["f_center_mhz"] >= lo) & (out["f_center_mhz"] <= hi)]
    if status_filter != "All":
        status_of = lambda cid: (review_status or {}).get(cid, "pending")
        if status_filter == "Pending":
            out = out[[status_of(c) == "pending" for c in out["candidate_id"]]]
        elif status_filter == "Reviewed":
            out = out[[status_of(c) != "pending" for c in out["candidate_id"]]]
        else:
            out = out[[status_of(c) == status_filter.lower() for c in out["candidate_id"]]]
    if category_filter and assessment is not None:
        out = out[[assessment.get(c) in category_filter for c in out["candidate_id"]]]
    return out
