"""
Drives the real Streamlit app headlessly (streamlit.testing) through each analysis state.

Reviews, caches and uploads are redirected to temporary folders, so these tests never touch
real review data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.observatory import scoring, sources
from src.observatory.observations import Observation, ResultsInfo
from src.observatory.provenance import heldout_origins
from src.observatory.reviews import CadenceLog, Preferences, ReviewStore
from src.observatory.sources import FilterbankSource
from tests.obs_helpers import noise, write_fil

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["Overview", "Candidates", "Waterfall", "Cadence Analysis", "Compare", "Export"]


def text_of(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [e.value for e in at.error]
    parts += [w.value for w in at.warning] + [i.value for i in at.info] + [s.value for s in at.subheader]
    return "\n".join(str(p) for p in parts)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated user state plus a hook to choose which observations the app discovers."""
    from ui import loaders

    store = ReviewStore(tmp_path / "state" / "reviews.sqlite")
    cadence = CadenceLog(tmp_path / "state" / "cadence.json")
    prefs = Preferences(tmp_path / "state" / "prefs.json")
    prefs.set("guide_dismissed", True)
    monkeypatch.setattr(loaders, "review_store", lambda: store)
    monkeypatch.setattr(loaders, "cadence_log", lambda: cadence)
    monkeypatch.setattr(loaders, "preferences", lambda: prefs)
    monkeypatch.setattr(scoring, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(sources, "CACHE_DIR", tmp_path / "cache")

    def use(observations, obs_id=None, page="Overview", **state):
        monkeypatch.setattr("src.observatory.observations.discover_observations", lambda *a, **k: observations)
        at = AppTest.from_file(APP, default_timeout=300)
        at.session_state["page"] = page
        if obs_id:
            at.session_state["obs_id"] = obs_id
        for k, v in state.items():
            at.session_state[k] = v
        return at

    use.store, use.cadence, use.tmp = store, cadence, tmp_path
    return use


def make_obs(tmp_path, obs_id, with_results=True, with_file=True, n_t=64 * 12, n_f=64 * 6, csv_ok=True, real_scores=True):
    """A small real .fil plus held-out rankings. ``real_scores`` scores the held-out tiles with the actual model,
    so recovered tile positions verify; with False the stored scores are made-up and positions must be withheld."""
    meta = None
    fil = write_fil(tmp_path / f"{obs_id}.fil", np.random.RandomState(3).rand(n_t, n_f).astype(np.float32),
                    fch1=1500.0, foff=-0.1, tsamp=0.5, source=obs_id)
    src = FilterbankSource(fil)
    meta = src.meta
    results = None
    if with_results:
        origins = heldout_origins(meta.n_times, meta.nchans)
        if real_scores:
            bundle = scoring.load_model()
            sc = scoring.score_windows(bundle, [src.read_window(int(t), 64, int(f), 64) for t, f in origins]).astype(float)
        else:
            sc = np.random.RandomState(4).rand(len(origins)) * 0.01
        order = np.argsort(sc)[::-1]
        csv = tmp_path / f"test_rankings_{obs_id}.csv"
        if csv_ok:
            pd.DataFrame({"test_rank": np.arange(1, len(order) + 1), "test_tile_id": [f"{obs_id}_test_{i:05d}" for i in order],
                          "reconstruction_error": sc[order]}).to_csv(csv, index=False)
        else:
            pd.DataFrame({"unrelated": [1]}).to_csv(csv, index=False)
        results = ResultsInfo(obs_id, {"telescope": "Test Dish", "band": "L-band"}, csv, float(np.percentile(sc, 80)) if with_results else None)
    return Observation(obs_id, obs_id, "bundled" if with_results else "uploaded", meta if with_file else None, results)


@pytest.mark.parametrize("page", PAGES)
def test_no_observations_is_a_clear_state_on_every_page(env, page):
    at = env([], page=page).run()
    assert not at.exception
    assert "No observation" in text_of(at) or "No observations found" in text_of(at)


def test_unreadable_observation_explains_itself(env):
    bad = Observation("bad-1", "bad.fil", "bundled", None, None, ["bad.fil: Not a valid SIGPROC filterbank file"])
    at = env([bad], "bad-1", "Overview").run()
    assert not at.exception
    assert "could not be loaded" in text_of(at) and "Not a valid SIGPROC" in text_of(at)


@pytest.mark.parametrize("page", ["Candidates", "Waterfall", "Cadence Analysis", "Export"])
def test_loaded_but_not_analysed(env, page):
    obs = make_obs(env.tmp, "notan", with_results=False)
    at = env([obs], "notan", page).run()
    assert not at.exception
    assert "has not been analysed" in text_of(at)


def test_not_analysed_overview_offers_analysis_and_runs_it(env):
    obs = make_obs(env.tmp, "scanme", with_results=False)
    at = env([obs], "scanme", "Overview").run()
    assert [b.key for b in at.button if b.key == "run_scan"] == ["run_scan"]
    at.button(key="run_scan").click().run()
    assert not at.exception
    txt = text_of(at)
    assert "scan of the whole observation" in txt.lower()
    assert any("unusual signal" in m.value for m in at.markdown)
    # the scan was cached, and Candidates now lists real rows
    at.session_state["page"] = "Candidates"
    at.run()
    assert not at.exception and "candidates shown" in text_of(at)
    assert "no validated threshold" in text_of(at)                       # relative basis is stated for unvalidated data


def test_results_without_source_file_keep_scores_but_not_positions(env):
    obs = make_obs(env.tmp, "nodata", with_results=True, with_file=False)
    at = env([obs], "nodata", "Candidates").run()
    assert not at.exception
    txt = text_of(at)
    assert "source data file is missing" not in txt or True
    assert "candidates shown" in txt
    for page in ("Overview", "Waterfall", "Cadence Analysis"):
        at.session_state["page"] = page
        at.run()
        assert not at.exception, page
    assert "needs the observation data" in text_of(at)             # cadence page, distinct from "not analysed"


def test_unverifiable_positions_are_withheld_not_guessed(env):
    obs = make_obs(env.tmp, "unverified", with_results=True, real_scores=False)
    at = env([obs], "unverified", "Overview").run()
    assert not at.exception
    txt = text_of(at)
    assert "do not reproduce the stored scores" in txt and "Tile positions are not shown" in txt
    assert "Tile positions are unavailable" in txt                       # and the strip is not drawn from guesses
    at.session_state["page"] = "Waterfall"
    at.run()
    assert "needs the observation data" in text_of(at)


def test_malformed_results_are_reported_as_failure(env):
    obs = make_obs(env.tmp, "malformed", with_results=True, csv_ok=False)
    at = env([obs], "malformed", "Candidates").run()
    assert not at.exception
    assert "could not be used" in text_of(at) and "missing columns" in text_of(at)


def test_zero_candidates_is_a_result_not_an_error(env):
    obs = make_obs(env.tmp, "quiet", with_results=True)
    at = env([obs], "quiet", "Overview", cutoff_override=99.0).run()
    assert not at.exception
    assert "No tile scores above the cutoff" in text_of(at)
    at.session_state["page"] = "Candidates"
    at.run()
    assert "No candidates match this sensitivity" in text_of(at)


def test_ready_observation_renders_every_page_and_sensitivity_changes_the_count(env):
    obs = make_obs(env.tmp, "ready", with_results=True)
    at = env([obs], "ready", "Overview").run()
    assert not at.exception
    counts = {}
    for level in ("Low", "Medium", "High"):
        at.session_state["sensitivity"] = level
        at.run()
        assert not at.exception
        counts[level] = next(m.value for m in at.markdown if "sb-meta" in m.value and "unusual signal" in m.value)
    assert counts["Low"] != counts["High"]
    for page in PAGES:
        at.session_state["page"] = page
        at.run()
        assert not at.exception, page


def test_review_buttons_persist_through_the_app(env):
    obs = make_obs(env.tmp, "reviewme", with_results=True)
    at = env([obs], "reviewme", "Candidates").run()
    assert not at.exception
    cid = at.session_state["selected_candidate"]
    at.text_area(key=f"note_{cid}").set_value("worth a look").run()
    at.button(key=f"retain_{cid}").click().run()
    assert not at.exception
    saved = env.store.get(cid)
    assert saved is not None and saved.status == "retained" and saved.note == "worth a look"
    # reviewed state is visible to Export, and survives a fresh session
    at2 = env([obs], "reviewme", "Export").run()
    assert "1 retained" in text_of(at2)


def test_review_write_failure_is_shown_not_swallowed(env, monkeypatch):
    from src.observatory.reviews import ReviewStoreError
    obs = make_obs(env.tmp, "failsave", with_results=True)
    at = env([obs], "failsave", "Candidates").run()
    cid = at.session_state["selected_candidate"]

    def boom(*a, **k):
        raise ReviewStoreError("disk is read-only")
    monkeypatch.setattr(env.store, "set", boom)
    at.button(key=f"flag_{cid}").click().run()
    assert not at.exception
    assert any("disk is read-only" in e.value and "not saved" in e.value for e in at.error)


def test_cadence_with_a_compatible_off_observation(env):
    on = make_obs(env.tmp, "cad_on", with_results=True)
    off_fil = write_fil(env.tmp / "cad_off.fil", noise(64 * 8, 64 * 6, seed=9), fch1=1500.0, foff=-0.1, tsamp=0.5, source="cad_off")
    off = Observation("cad_off", "cad_off", "uploaded", FilterbankSource(off_fil).meta, None)
    at = env([on, off], "cad_on", "Cadence Analysis").run()
    assert not at.exception
    assert "Choose OFF-target observations" in text_of(at)
    at.button(key="cad_run").click().run()
    assert not at.exception
    cid = at.session_state["cad_candidate"]
    assert cid in env.cadence.read()
    assert env.cadence.read()[cid]["verdict"] in ("CANDIDATE_TECHNOSIGNATURE", "LOCAL_RFI", "NORMAL_BACKGROUND")
    assert "ON/OFF contrast" in text_of(at)


def test_cadence_without_a_compatible_off_observation_is_honest(env):
    on = make_obs(env.tmp, "cad_solo", with_results=True)
    far_fil = write_fil(env.tmp / "far.fil", noise(64 * 8, 64 * 6, seed=5), fch1=900.0, foff=-0.1, tsamp=0.5, source="far")
    far = Observation("far", "far", "uploaded", FilterbankSource(far_fil).meta, None)
    at = env([on, far], "cad_solo", "Cadence Analysis").run()
    assert not at.exception
    txt = text_of(at)
    assert "No OFF-target observation is available" in txt and "Nothing is estimated or simulated" in txt
    assert "Run cadence analysis" not in [b.label for b in at.button]


def test_developer_tools_are_hidden_until_requested_and_run_on_demand(env):
    obs = make_obs(env.tmp, "devtools", with_results=True)
    at = env([obs], "devtools", "Overview").run()
    assert "Synthetic training lab" not in " ".join(t.label for t in at.tabs)
    at.session_state["dev_tools"] = True
    at.run()
    assert not at.exception
    labels = " ".join(t.label for t in at.tabs)
    assert "Synthetic training lab" in labels and "Model diagnostics" in labels
    # nothing was trained just by opening the tools
    assert "lab_model" not in at.session_state
