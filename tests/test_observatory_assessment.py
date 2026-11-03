import numpy as np
import pytest

from src.observatory.assessment import (
    ARTEFACT, INSUFFICIENT, RFI, UNUSUAL, assess_candidate, characterize, known_bands_for,
)
from src.observatory.sources import FilterbankSource
from tests.obs_helpers import write_fil


def noisy(nt=64, nf=64, seed=0):
    return 1.0 + 0.05 * np.random.RandomState(seed).randn(nt, nf)


def meta_for(tmp_path, nchans=256, foff=-0.001, tsamp=1.0, fch1=1580.0):
    return FilterbankSource(write_fil(tmp_path / "m.fil", np.ones((64, nchans), dtype=np.float32), fch1=fch1, foff=foff, tsamp=tsamp)).meta


def test_persistent_narrowband_is_measured(tmp_path):
    w = noisy()
    w[:, 30] += 3.0
    c = characterize(w, meta_for(tmp_path))
    assert c.morphology == "narrowband, persistent"
    assert c.narrowband_channel == 30 and c.narrowband_snr > 6 and c.persistence > 0.9
    assert c.drift_hz_per_s == 0.0                       # steady carrier: no measurable drift
    assert c.freq_resolution_khz == pytest.approx(1.0)


def test_drift_rate_and_sign(tmp_path):
    w = noisy(seed=1)
    for t in range(64):
        w[t, 10 + t // 2] += 3.0                        # +0.5 channel per second
    c = characterize(w, meta_for(tmp_path, foff=-0.001, tsamp=1.0))
    assert c.drift_hz_per_s == pytest.approx(-500.0, rel=0.1)   # channel index up = frequency down
    c2 = characterize(w, meta_for(tmp_path, foff=+0.001, tsamp=1.0))
    assert c2.drift_hz_per_s == pytest.approx(500.0, rel=0.1)


def test_intermittent_and_impulsive_are_distinguished(tmp_path):
    w = noisy(seed=2)
    w[5:8, 40] += 3.0
    assert characterize(w, meta_for(tmp_path)).morphology == "narrowband, intermittent"
    w2 = noisy(seed=3)
    w2[30, :] += 1.5
    c = characterize(w2, meta_for(tmp_path))
    assert c.morphology == "broadband, impulsive" and c.impulsive_snr > 6


def test_featureless_and_degenerate_tiles_do_not_crash(tmp_path):
    m = meta_for(tmp_path)
    assert characterize(noisy(seed=4), m).morphology == "no clear narrowband or impulsive structure"
    assert characterize(np.full((64, 64), 5.0), m).morphology in ("no clear narrowband or impulsive structure",)
    nan = noisy()
    nan[3, 3] = np.nan
    assert np.isfinite(characterize(nan, m).narrowband_snr)
    tiny = characterize(np.ones((4, 4)), m)
    assert tiny.morphology == "unmeasured"
    assert characterize(noisy(), None).freq_resolution_khz is None


def carrier(tmp_path):
    w = noisy()
    w[:, 30] += 3.0
    return characterize(w, meta_for(tmp_path))


def test_narrowband_inside_carrier_allocation_is_likely_rfi(tmp_path):
    a = assess_candidate(0.03, 0.02, 1579.0, 1581.0, carrier(tmp_path), meta_for(tmp_path), f0_channel=64)
    assert a.category == RFI and a.preliminary
    assert any(e.kind == "rfi" for e in a.evidence) and a.bands


def test_same_signal_outside_any_allocation_stays_unusual(tmp_path):
    a = assess_candidate(0.03, 0.02, 1420.0, 1420.2, carrier(tmp_path), meta_for(tmp_path), f0_channel=64)
    assert a.category == UNUSUAL and not a.bands
    assert not any(e.kind == "rfi" for e in a.evidence)


def test_radar_allocation_needs_a_pulsed_shape(tmp_path):
    m = meta_for(tmp_path, fch1=9000.0)
    pulsed = noisy(seed=5)
    pulsed[30, :] += 1.5
    a = assess_candidate(0.03, 0.02, 9000.0, 9000.2, characterize(pulsed, m), m, f0_channel=64)
    assert a.category == RFI
    cont = assess_candidate(0.03, 0.02, 9000.0, 9000.2, carrier(tmp_path), m, f0_channel=64)
    assert cont.category == UNUSUAL                       # a steady carrier is not explained by a radar allocation


def test_band_edge_is_a_possible_artefact(tmp_path):
    m = meta_for(tmp_path, nchans=256)
    a = assess_candidate(0.03, 0.02, 1420.0, 1420.2, None, m, f0_channel=0)
    assert a.category == ARTEFACT
    last = assess_candidate(0.03, 0.02, 1420.0, 1420.2, None, m, f0_channel=256 - 64)
    assert last.category == ARTEFACT
    middle = assess_candidate(0.03, 0.02, 1420.0, 1420.2, None, m, f0_channel=64)
    assert middle.category == UNUSUAL


def test_cadence_evidence_is_used(tmp_path):
    m = meta_for(tmp_path)
    rfi = assess_candidate(0.03, 0.02, 1420.0, 1420.2, None, m, f0_channel=64, cadence_verdict="LOCAL_RFI")
    assert rfi.category == RFI
    kept = assess_candidate(0.03, 0.02, 1420.0, 1420.2, None, m, f0_channel=64, cadence_verdict="CANDIDATE_TECHNOSIGNATURE")
    assert kept.category == UNUSUAL and any("ON-target only" in e.title for e in kept.evidence)
    none = assess_candidate(0.03, 0.02, 1420.0, 1420.2, None, m, f0_channel=64)
    assert any("cadence" in x.lower() for x in none.missing)


def test_below_cutoff_without_evidence_is_insufficient():
    a = assess_candidate(0.01, 0.02, 1420.0, 1420.2, None)
    assert a.category == INSUFFICIENT
    assert any("could not be read" in x for x in a.missing)


def test_missing_position_is_reported_not_guessed():
    a = assess_candidate(0.03, 0.02, float("nan"), float("nan"), None)
    assert a.category == UNUSUAL and not a.bands
    assert any("position is unavailable" in x for x in a.missing)


def test_known_bands_overlap_rules():
    assert [b.name for b in known_bands_for(1575.0, 1576.0)][0].startswith("Radionavigation-satellite (GPS L1")
    assert known_bands_for(1420.2, 1420.6) == []         # the hydrogen line is not on the interference list
    assert known_bands_for(float("nan"), 1.0) == []


def test_no_confidence_percentages_in_output(tmp_path):
    a = assess_candidate(0.03, 0.02, 1579.0, 1581.0, carrier(tmp_path), meta_for(tmp_path), f0_channel=64)
    text = " ".join([a.headline, a.why] + [e.title + e.detail for e in a.evidence])
    assert "confidence" not in text.lower()
    assert "extraterrestrial" not in text.lower()
