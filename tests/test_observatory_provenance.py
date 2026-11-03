import numpy as np
import pytest

from src.config import RAW_DATA_DIR
from src.observatory import provenance
from src.observatory.provenance import grid_origins, grid_shape, heldout_origins, parse_test_tile_index
from src.sequential_trainer import SequentialTrainer
from tests.obs_helpers import write_fil


def _tile_key(arr):
    return np.round(np.asarray(arr, dtype=np.float64).ravel(), 6).tobytes()


def _meta(path):
    from src.filterbank_reader import read_filterbank_header
    h, off = read_filterbank_header(path)
    return {"path": str(path), "nchans": h["nchans"], "nifs": h["nifs"], "source_name": "t",
            "telescope_name": "t", "band": "L"}


@pytest.mark.parametrize("limit", [28000, 300])
def test_heldout_origins_match_the_training_partitioning(tmp_path, limit):
    """The recovered (t0, f0) of every held-out tile must select exactly the tile the trainer put in its test split."""
    from src.observatory.sources import FilterbankSource
    rng = np.random.RandomState(7)
    data = rng.rand(64 * 20 + 10, 64 * 8 + 5).astype(np.float32) * (1 + np.arange(64 * 8 + 5) % 7)
    path = write_fil(tmp_path / "g.fil", data)
    src = FilterbankSource(path)

    trainer = SequentialTrainer(raw_dir=tmp_path, output_dir=tmp_path / "ck", figures_dir=tmp_path / "fig")
    parts = trainer.prepare_dataset_partitions(_meta(path), target_sample_limit=limit)
    origins = heldout_origins(src.meta.n_times, src.meta.nchans, limit=limit)

    assert len(origins) == len(parts["test"])
    for (t0, f0), tile in zip(origins, parts["test"][:, 0]):
        window = src.read_window(int(t0), 64, int(f0), 64)
        lo, hi = window.min(), window.max()
        expect = (window - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(window)
        assert np.allclose(expect, tile, atol=1e-6)


def test_grid_geometry():
    assert grid_shape(1025024, 352) == (16016, 5)
    assert grid_shape(1800, 64000) == (28, 1000)
    assert grid_shape(279, 1310720) == (4, 20480)
    o = grid_origins(130, 130)
    assert o.shape == (4, 2) and o[1].tolist() == [0, 64] and o[2].tolist() == [64, 0]
    assert grid_shape(10, 10) == (0, 0) and len(grid_origins(10, 10)) == 0


def test_subband_layout_for_very_wide_files():
    o = provenance.sampled_origins(279, 1310720)
    assert len(o) == 28000                       # 4 time blocks x 7000 sub-bands
    assert set(o[:, 0].tolist()) == {0, 64, 128, 192}
    assert o[:, 1].max() == 1310720 - 64


def test_parse_test_tile_index():
    assert parse_test_tile_index("ProxCen_S_test_03968") == 3968
    assert parse_test_tile_index("G12.72+6.26_test_00007") == 7
    assert parse_test_tile_index("garbage") is None
    assert parse_test_tile_index("x_test_abc") is None


@pytest.mark.skipif(not list(RAW_DATA_DIR.glob("*.fil")), reason="bundled observations not present")
def test_recovered_positions_reproduce_stored_scores_for_bundled_data():
    """Real data: rescoring the top-ranked tiles at the recovered positions gives the stored scores."""
    from src.observatory.observations import discover_observations
    from src.observatory.candidates import candidates_from_heldout
    from src.observatory.scoring import load_model, score_windows
    from src.observatory.sources import open_source

    bundle = load_model()
    checked = 0
    for obs in discover_observations():
        if obs.meta is None or obs.results is None or obs.results.rankings_path is None:
            continue
        src = open_source(obs.meta.path)
        cs = candidates_from_heldout(obs, lambda origins, scores: provenance.verify_against_scores(
            origins, scores, lambda t0, f0: src.read_window(t0, 64, f0, 64), lambda w: score_windows(bundle, w)))
        assert cs.provenance is not None and cs.provenance.ok, cs.provenance
        assert cs.has_coordinates
        checked += 1
    assert checked >= 1


def test_verify_detects_wrong_positions():
    origins = np.array([[0, 0], [64, 0]])
    stored = np.array([0.5, 0.1])
    bad = provenance.verify_against_scores(origins, stored, lambda t0, f0: np.zeros((64, 64)), lambda ws: [0.2] * len(ws), sample=2)
    assert not bad.ok and bad.max_abs_diff > 0.1
    good = provenance.verify_against_scores(origins, stored, lambda t0, f0: np.zeros((64, 64)), lambda ws: [0.5, 0.1][: len(ws)], sample=2)
    assert good.ok
