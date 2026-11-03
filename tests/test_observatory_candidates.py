import json

import numpy as np
import pandas as pd
import pytest

from src.observatory.candidates import (
    AnalysisKind, LEVELS, MIN_HIGH_CANDIDATES, apply_sensitivity, candidates_from_heldout, candidates_from_scan,
    filter_candidates, load_candidate_set, sensitivity_cutoff,
)
from src.observatory.observations import Observation, ResultsInfo, discover_observations, load_results_index
from src.observatory.provenance import Provenance, heldout_origins
from src.observatory.scoring import ScanResult
from src.observatory.sources import FilterbankSource
from tests.obs_helpers import noise, write_fil


def scores(n=1000, seed=0):
    return np.random.RandomState(seed).gamma(2.0, 0.003, n)


def test_medium_with_validated_threshold_is_that_threshold():
    s = scores()
    c = sensitivity_cutoff("Medium", s, 0.012)
    assert c.value == 0.012 and c.basis == "validated"
    assert (s > c.value).sum() == int((s > 0.012).sum())


def test_levels_are_ordered_and_scale_with_medium():
    s = scores()
    n = {lvl: int((s > sensitivity_cutoff(lvl, s, 0.012).value).sum()) for lvl in LEVELS}
    assert n["Low"] < n["Medium"] < n["High"]
    assert n["Low"] == max(1, n["Medium"] // 2)
    assert n["High"] == 2 * n["Medium"]


def test_relative_basis_when_no_validated_threshold():
    s = scores(2000)
    med = sensitivity_cutoff("Medium", s, None)
    assert med.basis == "relative"
    assert int((s > med.value).sum()) == 20           # top 1% of 2000
    assert "no validated threshold" in med.description


def test_high_always_shows_something_when_medium_finds_none():
    s = scores(500)
    thr = float(s.max() + 1)
    assert int((s > sensitivity_cutoff("Medium", s, thr).value).sum()) == 0
    assert int((s > sensitivity_cutoff("Low", s, thr).value).sum()) == 0
    assert int((s > sensitivity_cutoff("High", s, thr).value).sum()) == MIN_HIGH_CANDIDATES


def test_sensitivity_validation_and_empty():
    with pytest.raises(ValueError):
        sensitivity_cutoff("Extreme", [1.0], None)
    assert sensitivity_cutoff("Medium", [], None).value == float("inf")
    assert sensitivity_cutoff("Medium", [np.nan], None).value == float("inf")


@pytest.fixture
def heldout_setup(tmp_path):
    """A real tiny .fil plus a rankings CSV for its held-out partition."""
    data = np.random.RandomState(3).rand(64 * 12, 64 * 6).astype(np.float32)
    fil = write_fil(tmp_path / "t.fil", data, fch1=1500.0, foff=-0.1, tsamp=0.5, source="TGT")
    meta = FilterbankSource(fil).meta
    origins = heldout_origins(meta.n_times, meta.nchans, limit=28000)
    sc = np.random.RandomState(4).rand(len(origins)) * 0.01
    order = np.argsort(sc)[::-1]
    pd.DataFrame({
        "test_rank": np.arange(1, len(order) + 1),
        "test_tile_id": [f"TGT_test_{i:05d}" for i in order],
        "reconstruction_error": sc[order],
    }).to_csv(tmp_path / "test_rankings_TGT.csv", index=False)
    obs = Observation("TGT", "TGT", "bundled", meta,
                      ResultsInfo("TGT", {}, tmp_path / "test_rankings_TGT.csv", 0.008))
    return obs, origins, sc, order


def test_heldout_candidates_carry_measured_coordinates(heldout_setup):
    obs, origins, sc, order = heldout_setup
    cs = candidates_from_heldout(obs)
    assert cs.kind == AnalysisKind.READY and cs.has_coordinates and cs.basis == "held-out tiles"
    top = cs.frame.iloc[0]
    t0, f0 = origins[order[0]]
    assert (top.t0, top.f0) == (t0, f0)
    assert top.t_start_s == pytest.approx(t0 * 0.5) and top.t_end_s == pytest.approx((t0 + 64) * 0.5)
    first, last = 1500.0 - f0 * 0.1, 1500.0 - (f0 + 63) * 0.1
    assert top.f_center_mhz == pytest.approx((first + last) / 2)
    assert top.f_low_mhz == pytest.approx(last - 0.05) and top.f_high_mhz == pytest.approx(first + 0.05)
    assert top.candidate_id == f"TGT:t{t0}:f{f0}"
    assert cs.frame["rank"].tolist() == list(range(1, len(cs.frame) + 1))
    assert cs.n_grid_total == 12 * 6 and 0 < cs.coverage_fraction <= 1


def test_ids_are_stable_across_sensitivity(heldout_setup):
    obs, *_ = heldout_setup
    frame = candidates_from_heldout(obs).frame
    ids = set(frame.candidate_id)
    for lvl in LEVELS:
        sub = apply_sensitivity(frame, sensitivity_cutoff(lvl, frame.score, 0.008))
        assert set(sub.candidate_id) <= ids and sub.score.is_monotonic_decreasing


def test_failed_provenance_withholds_coordinates(heldout_setup):
    obs, *_ = heldout_setup
    cs = candidates_from_heldout(obs, lambda o, s: Provenance(False, 3, 0.5, "mismatch"))
    assert cs.kind == AnalysisKind.READY and not cs.has_coordinates
    assert cs.frame.t_start_s.isna().all() and cs.frame.f_center_mhz.isna().all()
    assert any("mismatch" in w for w in cs.warnings)


def test_provenance_check_that_raises_is_handled(heldout_setup):
    obs, *_ = heldout_setup

    def boom(o, s):
        raise RuntimeError("disk gone")
    cs = candidates_from_heldout(obs, boom)
    assert not cs.has_coordinates and any("disk gone" in w for w in cs.warnings)


def test_results_without_source_file_keep_scores_but_no_coordinates(heldout_setup):
    obs, *_ = heldout_setup
    gone = Observation("TGT", "TGT", "results-only", None, obs.results)
    cs = candidates_from_heldout(gone)
    assert cs.kind == AnalysisKind.SOURCE_MISSING and not cs.has_coordinates and len(cs.frame) > 0


def test_malformed_results_are_failures_not_crashes(tmp_path, heldout_setup):
    obs, *_ = heldout_setup
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1]}).to_csv(bad, index=False)
    out = candidates_from_heldout(Observation("T", "T", "bundled", obs.meta, ResultsInfo("T", {}, bad, None)))
    assert out.kind == AnalysisKind.FAILED and "missing columns" in out.message
    bad2 = tmp_path / "bad2.csv"
    pd.DataFrame({"test_tile_id": ["oops"], "reconstruction_error": [0.1]}).to_csv(bad2, index=False)
    assert candidates_from_heldout(Observation("T", "T", "bundled", obs.meta, ResultsInfo("T", {}, bad2, None))).kind == AnalysisKind.FAILED
    big = tmp_path / "big.csv"
    pd.DataFrame({"test_tile_id": ["T_test_99999"], "reconstruction_error": [0.1]}).to_csv(big, index=False)
    out = candidates_from_heldout(Observation("T", "T", "bundled", obs.meta, ResultsInfo("T", {}, big, None)))
    assert out.kind == AnalysisKind.FAILED and "partition" in out.message


def test_states_are_distinct(tmp_path, heldout_setup):
    obs, *_ = heldout_setup
    no_results = Observation("TGT", "TGT", "uploaded", obs.meta, None)
    st_not = load_candidate_set(no_results)
    assert st_not.kind == AnalysisKind.NOT_ANALYZED and st_not.frame.empty
    nothing = Observation("X", "X", "bundled", None, None, ["unreadable"])
    assert load_candidate_set(nothing).kind == AnalysisKind.FAILED
    # zero candidates is a READY state with an empty view, not an error
    cs = load_candidate_set(obs)
    cut = sensitivity_cutoff("Medium", cs.frame.score, 10.0)
    assert cs.kind == AnalysisKind.READY and apply_sensitivity(cs.frame, cut).empty


def test_scan_candidates_cover_every_tile(heldout_setup):
    obs, *_ = heldout_setup
    scan = ScanResult(np.random.RandomState(0).rand(12, 6).astype(np.float32), "m", obs.meta.fingerprint)
    cs = candidates_from_scan(obs, scan)
    assert cs.basis == "full scan" and len(cs.frame) == 72 and cs.coverage_fraction == 1.0
    assert cs.frame.candidate_id.is_unique
    wrong = candidates_from_scan(obs, ScanResult(np.zeros((3, 3), dtype=np.float32), "m", "f"))
    assert wrong.kind == AnalysisKind.FAILED
    assert load_candidate_set(obs, scan).basis == "full scan"


def test_filters(heldout_setup):
    obs, *_ = heldout_setup
    frame = candidates_from_heldout(obs).frame.head(6)
    ids = frame.candidate_id.tolist()
    reviews = {ids[0]: "retained", ids[1]: "dismissed", ids[2]: "flagged"}
    assert len(filter_candidates(frame, reviews, "Pending")) == 3
    assert len(filter_candidates(frame, reviews, "Reviewed")) == 3
    assert filter_candidates(frame, reviews, "Flagged").candidate_id.tolist() == [ids[2]]
    cat = {i: ("A" if k < 2 else "B") for k, i in enumerate(ids)}
    assert len(filter_candidates(frame, assessment=cat, category_filter=["A"])) == 2
    lo, hi = frame.f_center_mhz.min(), frame.f_center_mhz.median()
    assert filter_candidates(frame, f_range_mhz=(lo, hi)).f_center_mhz.le(hi).all()


def test_results_index_tolerates_missing_and_malformed(tmp_path):
    assert load_results_index(tmp_path) == {}
    (tmp_path / "test_summary_all_targets.json").write_text("{not json", encoding="utf-8")
    assert load_results_index(tmp_path) == {}
    (tmp_path / "test_summary_all_targets.json").write_text(json.dumps({"A": {"threshold": 0.1}, "B": 5}), encoding="utf-8")
    idx = load_results_index(tmp_path)
    assert list(idx) == ["A"] and idx["A"].threshold == 0.1 and idx["A"].rankings_path is None


def test_discovery_joins_files_to_results_and_flags_problems(tmp_path):
    raw, up, res = tmp_path / "raw", tmp_path / "up", tmp_path / "res"
    for d in (raw, up, res):
        d.mkdir()
    write_fil(raw / "a.fil", noise(128, 64), source="ALPHA")
    (raw / "broken.fil").write_bytes(b"xxxx")
    (res / "test_summary_all_targets.json").write_text(json.dumps({"ALPHA": {"threshold": 0.5}, "GHOST": {"threshold": 0.2}}), encoding="utf-8")
    found = {o.obs_id: o for o in discover_observations(raw, up, res)}
    assert found["ALPHA"].results.threshold == 0.5 and found["ALPHA"].meta is not None
    assert found["GHOST"].origin == "results-only" and found["GHOST"].meta is None and found["GHOST"].problems
    assert found["broken"].meta is None and found["broken"].problems
