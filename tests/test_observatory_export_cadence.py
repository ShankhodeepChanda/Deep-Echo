import numpy as np
import pandas as pd
import pytest

from src.observatory import cadence, export, scoring
from src.observatory.assessment import assess_candidate
from src.observatory.candidates import candidates_from_scan
from src.observatory.observations import Observation
from src.observatory.reviews import RETAINED, Review
from src.observatory.scoring import load_model, normalize_tiles, scan_observation, score_windows
from src.observatory.sources import FilterbankSource
from tests.obs_helpers import noise, write_fil


@pytest.fixture(scope="module")
def bundle():
    return load_model()


def make_obs(tmp_path, name="ON", n_t=64 * 4, n_f=64 * 3, seed=0, **kw):
    src = FilterbankSource(write_fil(tmp_path / f"{name}.fil", noise(n_t, n_f, seed), source=name, **kw))
    return Observation(name, name, "uploaded", src.meta), src


# ---------------------------------------------------------------- scanning
def test_normalize_tiles_matches_existing_tiler():
    from src.filterbank_reader import extract_spectrogram_tiles
    wf = np.random.RandomState(0).rand(128, 128).astype(np.float32) * 5
    ref = extract_spectrogram_tiles(wf)[:, 0]                          # (4, 64, 64)
    mine = normalize_tiles(np.stack([wf[:64, :64], wf[:64, 64:], wf[64:, :64], wf[64:, 64:]]))
    assert np.allclose(ref, mine, atol=1e-6)
    flat = normalize_tiles(np.full((1, 64, 64), 3.0))
    assert flat.max() == 0.0
    assert np.isfinite(normalize_tiles(np.full((1, 64, 64), np.nan))).all()


def test_scan_scores_every_tile_and_matches_direct_scoring(tmp_path, monkeypatch, bundle):
    monkeypatch.setattr(scoring, "CACHE_DIR", tmp_path / "cache")
    obs, src = make_obs(tmp_path)
    seen = []
    result = scan_observation(src, bundle, progress=seen.append)
    assert result.scores.shape == (4, 3) and seen[-1] == 1.0 and result.n_tiles == 12
    direct = score_windows(bundle, [src.read_window(64, 64, 128, 64)])[0]
    assert result.scores[1, 2] == pytest.approx(direct, rel=1e-5)
    cached = scoring.load_cached_scan(src.meta.fingerprint, bundle.signature)
    assert cached is not None and np.array_equal(cached.scores, result.scores)
    assert scoring.load_cached_scan("0" * 12, bundle.signature) is None


@pytest.mark.parametrize("target_tiles,max_bytes", [(4, 1 << 30), (1 << 20, 64 * 128 * 4 * 3), (3, 64 * 128 * 4)])
def test_scan_grouping_does_not_change_scores(tmp_path, monkeypatch, bundle, target_tiles, max_bytes):
    """Reading several time blocks per batch (and the memory cap) must give exactly the per-tile scores."""
    monkeypatch.setattr(scoring, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(scoring, "_TARGET_BATCH_TILES", target_tiles)
    monkeypatch.setattr(scoring, "_MAX_READ_BYTES", max_bytes)
    obs, src = make_obs(tmp_path, "grp", n_t=64 * 7 + 5, n_f=64 * 2 + 9, seed=4)
    result = scan_observation(src, bundle)
    assert result.scores.shape == (7, 2)
    for ti in range(7):
        for fi in range(2):
            direct = score_windows(bundle, [src.read_window(ti * 64, 64, fi * 64, 64)])[0]
            assert result.scores[ti, fi] == pytest.approx(direct, rel=1e-4, abs=1e-8)


def test_scan_can_be_cancelled_and_rejects_tiny_files(tmp_path, monkeypatch, bundle):
    monkeypatch.setattr(scoring, "CACHE_DIR", tmp_path / "cache")
    _, src = make_obs(tmp_path)
    with pytest.raises(InterruptedError):
        scan_observation(src, bundle, should_stop=lambda: True)
    _, tiny = make_obs(tmp_path, "tiny", n_t=30, n_f=64)
    with pytest.raises(ValueError):
        scan_observation(tiny, bundle)


def test_missing_checkpoint_is_a_clear_error(tmp_path):
    with pytest.raises(scoring.ModelUnavailableError):
        load_model(tmp_path / "nope.pt")
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"junk")
    with pytest.raises(scoring.ModelUnavailableError):
        load_model(bad)


def test_reconstruct_returns_consistent_score(bundle):
    w = np.random.RandomState(0).rand(64, 64).astype(np.float32)
    tile, recon, score = scoring.reconstruct(bundle, w)
    assert tile.shape == recon.shape == (64, 64)
    assert score == pytest.approx(score_windows(bundle, [w])[0], rel=1e-4)


# ---------------------------------------------------------------- cadence
def test_compatibility_rules(tmp_path):
    on, _ = make_obs(tmp_path, "ON", fch1=1400.0, foff=-0.01)
    same, _ = make_obs(tmp_path, "OFF1", seed=1, fch1=1400.0, foff=-0.01)
    shifted, _ = make_obs(tmp_path, "OFF2", seed=2, fch1=1400.0 - 0.01 * 64, foff=-0.01)   # grid offset by one tile
    other_res, _ = make_obs(tmp_path, "OFF3", seed=3, fch1=1400.0, foff=-0.02)
    other_band, _ = make_obs(tmp_path, "OFF4", seed=4, fch1=900.0, foff=-0.01)
    other_dt, _ = make_obs(tmp_path, "OFF5", seed=5, fch1=1400.0, foff=-0.01, tsamp=2.0)

    ok, _, f0 = cadence.compatibility(on.meta, same.meta, 64)
    assert ok and f0 == 64
    ok, _, f0 = cadence.compatibility(on.meta, shifted.meta, 128)
    assert ok and f0 == 64                                  # same frequencies, different channel index
    for bad, word in ((other_res, "Channel width"), (other_band, "cover"), (other_dt, "Time resolution")):
        ok, reason, _ = cadence.compatibility(on.meta, bad.meta, 64)
        assert not ok and word in reason


def test_cadence_run_on_real_files(tmp_path, bundle):
    on, on_src = make_obs(tmp_path, "ON", fch1=1400.0, foff=-0.01)
    off, off_src = make_obs(tmp_path, "OFF", seed=1, fch1=1400.0, foff=-0.01)
    bad, bad_src = make_obs(tmp_path, "BAD", seed=2, fch1=700.0, foff=-0.01)

    out = cadence.run_cadence("ON", on_src, 64, 64, [("OFF", off_src), ("BAD", bad_src)], bundle, threshold=1e9)
    assert out.verdict == "NORMAL_BACKGROUND"
    assert [p.pointing_type for p in out.pointings] == ["ON", "OFF"]
    assert out.skipped and out.skipped[0][0] == "BAD"
    assert out.pointings[1].window.shape == (64, 64)
    # with a threshold of zero everything is "flagged" in both pointings: a signal also seen OFF is local
    rfi = cadence.run_cadence("ON", on_src, 64, 64, [("OFF", off_src)], bundle, threshold=0.0)
    assert rfi.verdict == "LOCAL_RFI" and any("OFF" in r for r in rfi.reasons)

    with pytest.raises(ValueError, match="No OFF observation"):
        cadence.run_cadence("ON", on_src, 64, 64, [("BAD", bad_src)], bundle, threshold=0.1)


def test_most_unusual_tile_is_the_brute_force_argmax(tmp_path, bundle):
    rng = np.random.RandomState(0)
    data = (1.0 + 0.05 * rng.rand(64 * 6 + 10, 64 * 2)).astype(np.float32)
    data[64 * 3:64 * 4, 20] += 20.0
    data[64 * 4:64 * 5, 90] += 5.0
    src = FilterbankSource(write_fil(tmp_path / "p.fil", data))
    blocks = [src.read_window(i * 64, 64, 64, 64) for i in range(6)]
    expected = score_windows(bundle, blocks)
    t0, score, window = cadence.most_unusual_tile(src, bundle, 64)
    assert t0 == int(np.argmax(expected)) * 64
    assert score == pytest.approx(float(expected.max()), rel=1e-5)
    assert np.array_equal(window, blocks[int(np.argmax(expected))])


# ---------------------------------------------------------------- export
def sample_set(tmp_path, bundle, monkeypatch):
    monkeypatch.setattr(scoring, "CACHE_DIR", tmp_path / "cache")
    obs, src = make_obs(tmp_path, "EXP", fch1=1580.0, foff=-0.001, tsamp=0.5)
    cs = candidates_from_scan(obs, scan_observation(src, bundle))
    frame = cs.frame.head(3)
    ass = {}
    for r in frame.itertuples():
        w = src.read_window(int(r.t0), 64, int(r.f0), 64)
        from src.observatory.assessment import characterize
        ass[r.candidate_id] = assess_candidate(r.score, 0.0, r.f_low_mhz, r.f_high_mhz, characterize(w, obs.meta), obs.meta, int(r.f0))
    return obs, src, frame, ass


def test_export_frame_contains_real_records_only(tmp_path, bundle, monkeypatch):
    obs, src, frame, ass = sample_set(tmp_path, bundle, monkeypatch)
    first = frame.candidate_id.iloc[0]
    reviews = {first: Review(first, obs.obs_id, RETAINED, "check drift", "2026-10-23T10:00:00+00:00")}
    out = export.export_frame(frame, obs, 0.0123, "Medium", "relative", "full scan", reviews, ass, {first: "LOCAL_RFI"})
    assert list(out.columns) == export.COLUMNS and len(out) == 3
    row = out.iloc[0]
    assert row.candidate_id == first and row.review_status == "Retained" and row.review_note == "check drift"
    assert row.cadence_result == "LOCAL_RFI" and out.iloc[1].cadence_result == "not run"
    assert row.unusualness_score == pytest.approx(frame.score.iloc[0])
    assert row.freq_center_mhz == pytest.approx(frame.f_center_mhz.iloc[0], abs=1e-5)
    assert row.start_utc.endswith("Z") and row.telescope == "Green Bank Telescope (GBT)"
    assert "not evidence of extraterrestrial origin" in row.limitations
    assert out.iloc[1].review_status == "Pending"


def test_export_leaves_unavailable_values_empty(tmp_path, bundle, monkeypatch):
    obs, src, frame, ass = sample_set(tmp_path, bundle, monkeypatch)
    nocoord = frame.copy()
    nocoord[["t_start_s", "t_end_s", "f_low_mhz", "f_high_mhz", "f_center_mhz"]] = np.nan
    out = export.export_frame(nocoord, obs, 0.01, "High", "relative", "held-out tiles", {}, {})
    assert out.freq_center_mhz.isna().all() and out.elapsed_start_s.isna().all()
    assert (out.start_utc == "").all() and (out.assessment == "").all()


def test_csv_report_and_pdf_are_produced(tmp_path, bundle, monkeypatch):
    obs, src, frame, ass = sample_set(tmp_path, bundle, monkeypatch)
    out = export.export_frame(frame, obs, 0.01, "Medium", "relative", "full scan", {}, ass)
    csv = export.to_csv_bytes(out)
    back = pd.read_csv(pd.io.common.BytesIO(csv))
    assert len(back) == 3 and back.candidate_id.tolist() == out.candidate_id.tolist()

    tiles = {r.candidate_id: normalize_tiles(src.read_window(int(r.t0), 64, int(r.f0), 64)[None])[0] for r in frame.itertuples()}
    pngs = {k: export.tile_png(v) for k, v in tiles.items()}
    assert all(p[:8] == b"\x89PNG\r\n\x1a\n" for p in pngs.values())
    html_report = export.build_html_report(out, obs, pngs)
    assert frame.candidate_id.iloc[0] in html_report and "Limitations" in html_report and "data:image/png;base64," in html_report
    pdf = export.build_pdf(out, obs, tiles)
    assert pdf[:5] == b"%PDF-" and len(pdf) > 2000
