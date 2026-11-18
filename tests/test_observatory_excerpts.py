import json

import numpy as np
import pandas as pd
import pytest

from src.config import SAMPLES_DATA_DIR
from src.observatory.candidates import AnalysisKind, candidates_from_heldout
from src.observatory.excerpts import choose_window, read_sidecar, sidecar_path, write_excerpt
from src.observatory.observations import Observation, ResultsInfo, discover_observations
from src.observatory.provenance import heldout_origins
from src.observatory.sources import FilterbankSource
from tests.obs_helpers import noise, write_fil

T, F = 64 * 10, 64 * 8          # parent recording: 10 x 8 tiles


@pytest.fixture
def parent(tmp_path):
    """A two-IF parent recording plus a rankings CSV over its held-out partition."""
    data = np.random.RandomState(7).rand(T, 2, F).astype(np.float32)
    path = write_fil(tmp_path / "parent.fil", data, fch1=1500.0, foff=-0.1, tsamp=0.5, source="TGT", tstart=58000.0)
    origins = heldout_origins(T, F)
    sc = np.random.RandomState(8).rand(len(origins)) * 0.01
    order = np.argsort(sc)[::-1]
    rankings = tmp_path / "test_rankings_TGT.csv"
    pd.DataFrame({
        "test_tile_id": [f"TGT_test_{i:05d}" for i in order],
        "reconstruction_error": sc[order],
    }).to_csv(rankings, index=False)
    return path, origins, sc, rankings


def test_excerpt_is_bit_identical_with_true_header(parent, tmp_path):
    path, *_ = parent
    ex = write_excerpt(path, tmp_path / "out" / "TGT.fil", t0=128, nt=256, f0=192, nf=128)
    src, cut = FilterbankSource(path), FilterbankSource(tmp_path / "out" / "TGT.fil")
    m = cut.meta
    assert (m.n_times, m.nchans, m.nifs, m.source_name) == (256, 128, 1, "TGT")
    assert m.fch1_mhz == pytest.approx(1500.0 - 192 * 0.1)
    assert m.tstart_mjd == pytest.approx(58000.0 + 128 * 0.5 / 86400)
    assert np.array_equal(cut.read_window(0, 256, 0, 128), src.read_window(128, 256, 192, 128))
    assert read_sidecar(tmp_path / "out" / "TGT.fil") == ex
    assert (ex.parent_n_times, ex.parent_nchans, ex.t_offset, ex.f_offset) == (T, F, 128, 192)


def test_write_excerpt_rejects_regions_outside_the_file(parent, tmp_path):
    path, *_ = parent
    with pytest.raises(Exception):
        write_excerpt(path, tmp_path / "x.fil", t0=T - 32, nt=64, f0=0, nf=64)


def test_sidecar_absent_or_malformed(tmp_path):
    fil = tmp_path / "a.fil"
    assert read_sidecar(fil) is None
    sidecar_path(fil).write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        read_sidecar(fil)
    sidecar_path(fil).write_text(json.dumps({"parent_file": "p", "parent_n_times": 10, "parent_nchans": 10,
                                             "t_offset": -1, "f_offset": 0}), encoding="utf-8")
    with pytest.raises(ValueError):
        read_sidecar(fil)


def test_choose_window_fits_budget_and_keeps_the_top_tile():
    origins = heldout_origins(64 * 400, 64 * 6)
    ranked = origins[np.random.RandomState(1).permutation(len(origins))]
    budget = 64 * 40 * 64 * 6 * 4
    t0, nt, f0, nf = choose_window(64 * 400, 64 * 6, ranked, budget)
    assert nt * nf * 4 <= budget and t0 + nt <= 64 * 400 and f0 + nf <= 64 * 6
    top_t, top_f = ranked[0]
    assert t0 <= top_t and top_t + 64 <= t0 + nt and f0 <= top_f and top_f + 64 <= f0 + nf


def test_heldout_candidates_map_onto_an_excerpt(parent, tmp_path):
    path, origins, sc, rankings = parent
    out = tmp_path / "s" / "TGT.fil"
    ex = write_excerpt(path, out, t0=192, nt=320, f0=128, nf=256)
    obs = Observation("TGT", "TGT", "excerpt", FilterbankSource(out).meta, ResultsInfo("TGT", {}, rankings, 0.008), excerpt=ex)
    cs = candidates_from_heldout(obs)
    assert cs.kind == AnalysisKind.READY and cs.has_coordinates
    inside = ((origins[:, 0] >= 192) & (origins[:, 0] + 64 <= 512) & (origins[:, 1] >= 128) & (origins[:, 1] + 64 <= 384))
    assert len(cs.frame) == inside.sum() > 0
    assert any("excerpt" in w for w in cs.warnings)
    # The best tile inside the excerpt sits at its original position minus the offset, with its stored score.
    best = np.flatnonzero(inside)[np.argmax(sc[inside])]
    top = cs.frame.iloc[0]
    assert top.score == pytest.approx(sc[best])
    assert (top.t0, top.f0) == (origins[best][0] - 192, origins[best][1] - 128)
    src, cut = FilterbankSource(path), FilterbankSource(out)
    assert np.array_equal(cut.read_window(int(top.t0), 64, int(top.f0), 64),
                          src.read_window(int(origins[best][0]), 64, int(origins[best][1]), 64))


def test_excerpt_with_no_stored_tiles_inside_fails_clearly(parent, tmp_path):
    path, origins, _, rankings = parent
    out = tmp_path / "s" / "TGT.fil"
    ex = write_excerpt(path, out, t0=0, nt=64, f0=0, nf=64)
    keep = ~((origins[:, 0] == 0) & (origins[:, 1] == 0))
    df = pd.read_csv(rankings)
    df = df[[keep[int(t.rsplit("_", 1)[1])] for t in df.test_tile_id]]
    df.to_csv(rankings, index=False)
    obs = Observation("TGT", "TGT", "excerpt", FilterbankSource(out).meta, ResultsInfo("TGT", {}, rankings, None), excerpt=ex)
    assert candidates_from_heldout(obs).kind == AnalysisKind.FAILED


def test_discovery_uses_excerpts_only_when_the_full_recording_is_absent(parent, tmp_path):
    path, *_ = parent
    raw, up, res, samples = (tmp_path / d for d in ("raw", "up", "res", "samples"))
    for d in (raw, up, res, samples):
        d.mkdir()
    write_excerpt(path, samples / "TGT.fil", t0=0, nt=128, f0=0, nf=128)
    write_fil(samples / "plain.fil", noise(128, 64), source="PLAIN")        # no sidecar: not an excerpt
    (samples / "bad.fil").write_bytes(b"xxxx")
    sidecar_path(samples / "bad.fil").write_text("{}", encoding="utf-8")
    (res / "test_summary_all_targets.json").write_text(json.dumps({"TGT": {"threshold": 0.5}}), encoding="utf-8")

    found = {o.obs_id: o for o in discover_observations(raw, up, res, samples)}
    assert found["TGT"].origin == "excerpt" and found["TGT"].excerpt is not None
    assert found["TGT"].results is not None and found["TGT"].meta is not None
    assert "PLAIN" not in found and found["bad"].problems
    assert not any(o.origin == "results-only" for o in found.values())

    write_fil(raw / "full.fil", noise(128, 64), source="TGT")
    found = {o.obs_id: o for o in discover_observations(raw, up, res, samples)}
    assert found["TGT"].origin == "bundled" and found["TGT"].excerpt is None


@pytest.mark.skipif(not list(SAMPLES_DATA_DIR.glob("*.excerpt.json")), reason="demo excerpts not present")
def test_committed_excerpts_reproduce_stored_scores(tmp_path):
    """The excerpts a hosted deployment ships: positions recovered on them must reproduce the stored scores."""
    from src.observatory import provenance
    from src.observatory.scoring import load_model, score_windows

    bundle = load_model()
    empty = tmp_path / "empty"
    empty.mkdir()
    checked = 0
    for obs in discover_observations(raw_dir=empty, uploads_dir=empty):
        if obs.origin != "excerpt":
            continue
        assert obs.meta is not None and obs.results is not None, obs.problems
        src = FilterbankSource(obs.meta.path)
        cs = candidates_from_heldout(obs, lambda o, s: provenance.verify_against_scores(
            o, s, lambda t0, f0: src.read_window(t0, 64, f0, 64), lambda w: score_windows(bundle, w)))
        assert cs.provenance is not None and cs.provenance.ok, cs.provenance
        assert cs.has_coordinates and len(cs.frame) > 0
        checked += 1
    assert checked >= 1
