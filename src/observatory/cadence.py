"""
ON/OFF cadence analysis on real observations.

Adapts ``src.cadence_filter.analyze_cadence_sequence`` (which compares one
64 x 64 spectrogram per pointing) to the dashboard. For a candidate seen in the
ON pointing, the same frequency range is located in each OFF observation and
its most unusual tile (over the whole OFF recording) is scored. Nothing is
simulated: an OFF observation that does not cover that band at the same
resolution is reported as incompatible and left out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.cadence_filter import analyze_cadence_sequence
from src.observatory.provenance import TILE_F, TILE_T
from src.observatory.scoring import ModelBundle, normalize_tiles, score_windows
from src.observatory.sources import ObservationSource, ProgressCallback, SourceMeta

CONTRAST_THRESHOLD = 2.0
_RESOLUTION_TOLERANCE = 0.01


@dataclass
class Pointing:
    label: str
    pointing_type: str            # "ON" | "OFF"
    t0: int
    f0: int
    score: float
    flagged: bool
    window: np.ndarray            # raw (64, 64)
    meta: SourceMeta


@dataclass
class CadenceOutcome:
    verdict: str                  # CANDIDATE_TECHNOSIGNATURE | LOCAL_RFI | NORMAL_BACKGROUND
    pointings: List[Pointing]
    contrast_ratio: float
    on_score: float
    off_score: float
    threshold: float
    reasons: List[str] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)   # (observation label, reason)


def compatibility(on: SourceMeta, off: SourceMeta, on_f0: int) -> Tuple[bool, str, Optional[int]]:
    """Can the ON tile's frequency range be compared in ``off``? Returns (ok, reason, off tile f0)."""
    if abs(abs(off.foff_mhz) - abs(on.foff_mhz)) > _RESOLUTION_TOLERANCE * abs(on.foff_mhz):
        return False, f"Channel width differs ({abs(off.foff_mhz) * 1000:g} kHz vs {abs(on.foff_mhz) * 1000:g} kHz).", None
    if abs(off.tsamp_s - on.tsamp_s) > _RESOLUTION_TOLERANCE * on.tsamp_s:
        return False, f"Time resolution differs ({off.tsamp_s:g} s vs {on.tsamp_s:g} s).", None
    f_first = float(on.channel_freq_mhz(on_f0))
    f_last = float(on.channel_freq_mhz(on_f0 + TILE_F - 1))
    try:
        i1, i2 = off.freq_to_channel(f_first), off.freq_to_channel(f_last)
    except ValueError:
        return False, f"Does not cover {min(f_first, f_last):.3f}–{max(f_first, f_last):.3f} MHz.", None
    if abs(i1 - i2) + 1 != TILE_F:
        return False, "Channel grid does not line up with the ON observation.", None
    return True, "", int(min(i1, i2))


def most_unusual_tile(
    source: ObservationSource,
    bundle: ModelBundle,
    f0: int,
    progress: Optional[ProgressCallback] = None,
) -> Tuple[int, float, np.ndarray]:
    """Highest-scoring tile in a fixed channel range over the whole recording: (t0, score, window)."""
    meta = source.meta
    n_t = (meta.n_times - TILE_T) // TILE_T + 1
    if n_t < 1:
        raise ValueError("Observation is shorter than one tile.")
    best = (-1.0, 0, None)
    chunk = 256
    for start in range(0, n_t, chunk):
        stop = min(start + chunk, n_t)
        block = source.read_window(start * TILE_T, (stop - start) * TILE_T, f0, TILE_F)
        wins = block.reshape(stop - start, TILE_T, TILE_F)
        scores = score_windows(bundle, list(wins))
        j = int(np.argmax(scores))
        if scores[j] > best[0]:
            best = (float(scores[j]), (start + j) * TILE_T, wins[j].copy())
        if progress:
            progress(stop / n_t)
    return best[1], best[0], best[2]


def run_cadence(
    on_label: str,
    on_source: ObservationSource,
    on_t0: int,
    on_f0: int,
    off_sources: Sequence[Tuple[str, ObservationSource]],
    bundle: ModelBundle,
    threshold: float,
    progress: Optional[ProgressCallback] = None,
) -> CadenceOutcome:
    """Run the cadence filter for one ON tile against the compatible OFF observations."""
    on_window = on_source.read_window(on_t0, TILE_T, on_f0, TILE_F)
    on_score = float(score_windows(bundle, [on_window])[0])
    pointings = [Pointing(on_label, "ON", on_t0, on_f0, on_score, on_score > threshold, on_window, on_source.meta)]
    skipped: List[Tuple[str, str]] = []

    usable = []
    for label, src in off_sources:
        ok, reason, off_f0 = compatibility(on_source.meta, src.meta, on_f0)
        if ok:
            usable.append((label, src, off_f0))
        else:
            skipped.append((label, reason))
    if not usable:
        raise ValueError("No OFF observation can be compared with this candidate: " + "; ".join(f"{l}: {r}" for l, r in skipped))

    for i, (label, src, off_f0) in enumerate(usable):
        t0, score, window = most_unusual_tile(
            src, bundle, off_f0, (lambda p, i=i: progress((i + p) / len(usable)) if progress else None),
        )
        pointings.append(Pointing(label, "OFF", t0, off_f0, score, score > threshold, window, src.meta))

    observations = [
        {"target": p.label, "pointing_type": p.pointing_type, "spectrogram": normalize_tiles(p.window[None])[0]}
        for p in pointings
    ]
    result = analyze_cadence_sequence(bundle.model, observations, threshold, CONTRAST_THRESHOLD, device="cpu")

    off_flagged = [p for p in pointings if p.pointing_type == "OFF" and p.flagged]
    reasons: List[str] = []
    n_off = sum(1 for p in pointings if p.pointing_type == "OFF")
    if result.verdict == "CANDIDATE_TECHNOSIGNATURE":
        reasons.append(f"The ON tile scores {on_score:.5f}, above the threshold {threshold:.5f}.")
        reasons.append(f"None of the {n_off} OFF observation(s) has a tile in this band above the threshold.")
        reasons.append(f"ON/OFF contrast is {result.contrast_ratio:.1f}× (needs ≥ {CONTRAST_THRESHOLD:g}×).")
        reasons.append("Retained for further investigation. Absence from the OFF pointings does not establish an origin.")
    elif result.verdict == "LOCAL_RFI":
        reasons.append(f"The ON tile scores {on_score:.5f}, above the threshold {threshold:.5f}.")
        if off_flagged:
            names = ", ".join(p.label for p in off_flagged)
            reasons.append(f"An OFF observation also has an unusual tile in this band ({names}), so the signal is not confined to the target.")
        else:
            reasons.append(f"ON/OFF contrast is only {result.contrast_ratio:.1f}× (needs ≥ {CONTRAST_THRESHOLD:g}×).")
    else:
        reasons.append(f"The ON tile scores {on_score:.5f}, which is not above the threshold {threshold:.5f}.")
    return CadenceOutcome(
        verdict=result.verdict, pointings=pointings, contrast_ratio=float(result.contrast_ratio),
        on_score=float(result.on_target_error), off_score=float(result.off_target_error),
        threshold=float(threshold), reasons=reasons, skipped=skipped,
    )
