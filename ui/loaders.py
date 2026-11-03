"""
Cached access to the Observatory data layer for the Streamlit app.

Wrappers only: the logic lives in ``src.observatory``. Caches are keyed by file
fingerprints, so replacing a file or re-running a scan invalidates them.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import streamlit as st

from src.observatory.assessment import Assessment, assess_candidate, characterize
from src.observatory.candidates import CandidateSet, load_candidate_set
from src.observatory.export import data_uri, tile_png
from src.observatory.observations import Observation
from src.observatory.provenance import TILE_F, TILE_T, verify_against_scores
from src.observatory.reviews import CadenceLog, Preferences, ReviewStore
from src.observatory.scoring import (
    ModelBundle, ModelUnavailableError, load_cached_scan, load_model, normalize_tiles, scan_cache_path, score_windows,
)
from src.observatory.sources import ObservationSource, Overview, build_overview, open_source


@st.cache_resource(show_spinner=False)
def model_bundle() -> Tuple[Optional[ModelBundle], Optional[str]]:
    """(bundle, None) or (None, reason)."""
    try:
        return load_model(), None
    except ModelUnavailableError as exc:
        return None, str(exc)


@st.cache_resource(show_spinner=False)
def review_store() -> ReviewStore:
    return ReviewStore()


@st.cache_resource(show_spinner=False)
def cadence_log() -> CadenceLog:
    return CadenceLog()


@st.cache_resource(show_spinner=False)
def preferences() -> Preferences:
    return Preferences()


@st.cache_resource(show_spinner=False, max_entries=12)
def _open(path: str, fingerprint: str) -> ObservationSource:
    return open_source(path)


def source_for(obs: Optional[Observation]) -> Optional[ObservationSource]:
    if obs is None or obs.meta is None:
        return None
    return _open(str(obs.meta.path), obs.meta.fingerprint)


def _mtime(path) -> float:
    try:
        return path.stat().st_mtime
    except (OSError, AttributeError):
        return 0.0


def scan_stamp(obs: Observation, bundle: Optional[ModelBundle]) -> float:
    if obs.meta is None or bundle is None:
        return 0.0
    return _mtime(scan_cache_path(obs.meta.fingerprint, bundle.signature))


def _position_check(src: ObservationSource, bundle: ModelBundle):
    """Callback that re-scores a few tiles to confirm recovered tile positions."""
    def check(origins, scores):
        return verify_against_scores(
            origins, scores, lambda t0, f0: src.read_window(t0, TILE_T, f0, TILE_F), lambda w: score_windows(bundle, w),
        )
    return check


@st.cache_data(show_spinner=False, max_entries=8)
def _candidate_set(obs_id: str, fingerprint: str, results_stamp: float, scan_stamp_: float, _obs: Observation) -> CandidateSet:
    bundle, _ = model_bundle()
    src = source_for(_obs)
    scan = load_cached_scan(_obs.meta.fingerprint, bundle.signature) if (bundle and _obs.meta) else None
    check = None
    if bundle is not None and src is not None:
        check = _position_check(src, bundle)
    return load_candidate_set(_obs, scan, check)


def candidate_set(obs: Observation) -> CandidateSet:
    bundle, _ = model_bundle()
    results_stamp = _mtime(obs.results.rankings_path) if obs.results and obs.results.rankings_path else 0.0
    return _candidate_set(obs.obs_id, obs.meta.fingerprint if obs.meta else "-", results_stamp, scan_stamp(obs, bundle), obs)


@st.cache_data(show_spinner=False, max_entries=6)
def _overview(path: str, fingerprint: str) -> Overview:
    return build_overview(_open(path, fingerprint))


def overview_for(obs: Observation) -> Overview:
    """Decimated waterfall (computed once per file, then cached on disk)."""
    return _overview(str(obs.meta.path), obs.meta.fingerprint)


@st.cache_data(show_spinner=False, max_entries=3000)
def window(path: str, fingerprint: str, t0: int, f0: int, nt: int = TILE_T, nf: int = TILE_F) -> np.ndarray:
    return _open(path, fingerprint).read_window(int(t0), int(nt), int(f0), int(nf))


def window_for(obs: Observation, t0: int, f0: int, nt: int = TILE_T, nf: int = TILE_F) -> np.ndarray:
    return window(str(obs.meta.path), obs.meta.fingerprint, int(t0), int(f0), int(nt), int(nf))


@st.cache_data(show_spinner=False, max_entries=3000)
def _thumb(path: str, fingerprint: str, t0: int, f0: int) -> str:
    w = _open(path, fingerprint).read_window(int(t0), TILE_T, int(f0), TILE_F)
    return data_uri(tile_png(normalize_tiles(w[None])[0]))


def thumb_uri(obs: Observation, t0, f0) -> Optional[str]:
    if obs.meta is None or t0 is None or f0 is None or not np.isfinite(t0) or not np.isfinite(f0):
        return None
    try:
        return _thumb(str(obs.meta.path), obs.meta.fingerprint, int(t0), int(f0))
    except Exception:
        return None


@st.cache_data(show_spinner=False, max_entries=3000)
def _assessment(
    path: str, fingerprint: str, t0: float, f0: float, score: float, cutoff: float,
    f_low: float, f_high: float, verdict: Optional[str], _meta,
) -> Assessment:
    chars = None
    f0_channel = None
    if path and np.isfinite(t0) and np.isfinite(f0):
        try:
            chars = characterize(_open(path, fingerprint).read_window(int(t0), TILE_T, int(f0), TILE_F), _meta)
            f0_channel = int(f0)
        except Exception:
            chars = None
    return assess_candidate(score, cutoff, f_low, f_high, chars, _meta, f0_channel, verdict)


def assessment_for(obs: Observation, row, cutoff: float, verdict: Optional[str] = None) -> Assessment:
    meta = obs.meta
    return _assessment(
        str(meta.path) if meta else "", meta.fingerprint if meta else "-",
        float(row.t0) if np.isfinite(row.t0) else float("nan"), float(row.f0) if np.isfinite(row.f0) else float("nan"),
        float(row.score), float(cutoff), float(row.f_low_mhz), float(row.f_high_mhz), verdict, meta,
    )
