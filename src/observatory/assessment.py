"""
Preliminary, rule-based source assessment for a candidate tile.

This layer is separate from the model. The autoencoder says only how unusual a
tile is; it does not identify interference, instrument effects or
technosignatures. The rules below look at what can be measured directly from
the tile and its frequency, and every label states the evidence behind it.
Nothing here is a validated classifier and no confidence percentages are given.

Labels
------
* Likely interference (RFI): morphology measured in the tile matches the kind of
  signal an overlapping service produces (a narrowband carrier inside a
  satellite or communications allocation, or a broadband impulsive event inside
  a radar allocation), or the signal also appears in OFF-target pointings.
* Possible instrument artefact: the tile sits against the edge of the recorded
  band, where the receiver bandpass rolls off.
* Unusual - needs review: scores above the applicable cutoff and the evidence
  above does not apply.
* Insufficient information: not above the cutoff and no source evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.ndimage import median_filter

from src.observatory.provenance import TILE_F
from src.observatory.sources import SourceMeta

RFI = "Likely interference (RFI)"
ARTEFACT = "Possible instrument artefact"
UNUSUAL = "Unusual — needs review"
INSUFFICIENT = "Insufficient information"
CATEGORIES = (UNUSUAL, RFI, ARTEFACT, INSUFFICIENT)

CONTINUOUS = "continuous"   # carriers: satellite, navigation, communications
PULSED = "pulsed"           # radar

# Thresholds for the morphology rules.
NARROWBAND_SNR = 6.0
NARROWBAND_MAX_CHANNELS = 3
PERSISTENT_FRACTION = 0.5
IMPULSIVE_SNR = 6.0
DRIFT_MIN_POINTS = 8
DRIFT_MIN_R2 = 0.8
DRIFT_ROW_Z = 4.0


@dataclass(frozen=True)
class BandRef:
    name: str
    low_mhz: float
    high_mhz: float
    signal_type: str


# Indicative allocations (ITU service names, rounded). They are region- and
# site-dependent and are a reference for context, not a statement about any
# particular observatory's RFI environment.
KNOWN_INTERFERENCE: Sequence[BandRef] = (
    BandRef("Radionavigation-satellite (GPS L5, Galileo E5)", 1164.0, 1215.0, CONTINUOUS),
    BandRef("Radionavigation-satellite (GPS L2, GLONASS L2, Galileo E6)", 1215.0, 1300.0, CONTINUOUS),
    BandRef("Radiolocation and aeronautical radionavigation (radar)", 1215.0, 1400.0, PULSED),
    BandRef("Mobile-satellite downlink (e.g. Inmarsat)", 1525.0, 1559.0, CONTINUOUS),
    BandRef("Radionavigation-satellite (GPS L1, Galileo E1, GLONASS L1)", 1559.0, 1610.0, CONTINUOUS),
    BandRef("Mobile-satellite (e.g. Iridium)", 1610.0, 1626.5, CONTINUOUS),
    BandRef("Space research and space operation downlink (S-band)", 2200.0, 2290.0, CONTINUOUS),
    BandRef("Deep-space research (S-band)", 2290.0, 2300.0, CONTINUOUS),
    BandRef("Mobile broadband / wireless access (e.g. LTE band 40; region-dependent)", 2300.0, 2400.0, CONTINUOUS),
    BandRef("Fixed-satellite downlink (C/X-band)", 7250.0, 7750.0, CONTINUOUS),
    BandRef("Fixed-satellite uplink (X-band)", 7900.0, 8025.0, CONTINUOUS),
    BandRef("Earth-exploration-satellite downlink (X-band)", 8025.0, 8400.0, CONTINUOUS),
    BandRef("Deep-space research downlink (X-band)", 8400.0, 8450.0, CONTINUOUS),
    BandRef("Space research downlink (X-band)", 8450.0, 8500.0, CONTINUOUS),
    BandRef("Radiolocation (maritime and weather radar)", 8500.0, 10500.0, PULSED),
    BandRef("Fixed-satellite downlink (Ku-band)", 10700.0, 11700.0, CONTINUOUS),
)

REFERENCE_NOTE = (
    "Band list is indicative (rounded ITU service allocations). Real interference depends on the region and the "
    "observatory's surroundings, so overlap is context, not proof."
)


@dataclass
class Characteristics:
    """Measurements taken directly from one tile (None where not measurable)."""

    n_time: int
    n_chan: int
    freq_resolution_khz: Optional[float]
    time_resolution_s: Optional[float]
    narrowband_snr: Optional[float] = None
    narrowband_channel: Optional[int] = None      # index within the tile
    narrowband_width_channels: Optional[int] = None
    persistence: Optional[float] = None
    drift_hz_per_s: Optional[float] = None
    drift_r2: Optional[float] = None
    impulsive_snr: Optional[float] = None
    morphology: str = "unmeasured"

    def as_dict(self) -> Dict[str, object]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class Evidence:
    kind: str       # "rfi" | "artefact" | "info" | "missing"
    title: str
    detail: str


@dataclass
class Assessment:
    category: str
    headline: str
    why: str
    evidence: List[Evidence] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    characteristics: Optional[Characteristics] = None
    bands: List[BandRef] = field(default_factory=list)
    preliminary: bool = True


def _odd(n: int) -> int:
    n = max(int(n), 1)
    return n if n % 2 else n - 1 if n > 1 else 1


def _mad_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)) + 1e-12)


def known_bands_for(f_low_mhz: float, f_high_mhz: float) -> List[BandRef]:
    """Reference allocations overlapping [f_low, f_high]."""
    if not (np.isfinite(f_low_mhz) and np.isfinite(f_high_mhz)):
        return []
    return [b for b in KNOWN_INTERFERENCE if b.low_mhz < f_high_mhz and b.high_mhz > f_low_mhz]


def characterize(window: np.ndarray, meta: Optional[SourceMeta]) -> Characteristics:
    """Measure narrowband, persistence, drift and impulsive structure in a raw (time, channel) tile."""
    w = np.nan_to_num(np.asarray(window, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    nt, nf = w.shape
    foff = abs(meta.foff_mhz) if meta else None
    tsamp = meta.tsamp_s if meta else None
    out = Characteristics(
        n_time=nt, n_chan=nf,
        freq_resolution_khz=foff * 1000 if foff else None,
        time_resolution_s=tsamp,
    )
    if nt < 8 or nf < 8:
        return out

    # Narrowband: a channel standing above the smooth bandpass of the tile.
    profile = w.mean(axis=0)
    baseline = median_filter(profile, size=_odd(min(15, nf)), mode="nearest")
    resid = profile - baseline
    sigma = _mad_sigma(resid)
    peak = int(np.argmax(resid))
    snr = float(resid[peak] / sigma)
    half = resid[peak] / 2
    lo = hi = peak
    while lo > 0 and resid[lo - 1] > half:
        lo -= 1
    while hi < nf - 1 and resid[hi + 1] > half:
        hi += 1
    width = hi - lo + 1
    out.narrowband_snr = snr
    out.narrowband_channel = peak
    out.narrowband_width_channels = width

    # Persistence of that channel through time (rows are judged against their own spread).
    row_med = np.median(w, axis=1)
    row_sigma = np.array([_mad_sigma(row) for row in w])
    z_peak = (w[:, peak] - row_med) / row_sigma
    out.persistence = float(np.mean(z_peak > 3.0))

    # Drift: track the strongest channel row by row. Judged per row, because a drifting
    # signal is smeared out of the time-averaged profile used above.
    z_rows = (w - row_med[:, None]) / row_sigma[:, None]
    best_ch = z_rows.argmax(axis=1)
    strong = z_rows.max(axis=1) > DRIFT_ROW_Z
    if strong.sum() >= DRIFT_MIN_POINTS:
        t_idx = np.arange(nt)[strong]
        c_idx = best_ch[strong].astype(float)
        if np.ptp(c_idx) == 0:
            out.drift_hz_per_s, out.drift_r2 = 0.0, 1.0       # same channel every time: steady carrier
        else:
            slope, intercept = np.polyfit(t_idx, c_idx, 1)
            pred = slope * t_idx + intercept
            ss_res = float(np.sum((c_idx - pred) ** 2))
            ss_tot = float(np.sum((c_idx - c_idx.mean()) ** 2)) + 1e-12
            r2 = 1.0 - ss_res / ss_tot
            out.drift_r2 = r2
            if r2 >= DRIFT_MIN_R2 and abs(slope) * nt >= 1.0 and foff and tsamp:
                out.drift_hz_per_s = float(slope * meta.foff_mhz * 1e6 / tsamp)   # signed: frequency change per second

    # Impulsive broadband events: a short rise in the band-averaged time series.
    tprofile = w.mean(axis=1)
    tbase = median_filter(tprofile, size=_odd(min(9, nt)), mode="nearest")
    tres = tprofile - tbase
    out.impulsive_snr = float(tres.max() / _mad_sigma(tres))

    narrow = snr >= NARROWBAND_SNR and width <= NARROWBAND_MAX_CHANNELS
    drifting = out.drift_hz_per_s is not None and out.drift_hz_per_s != 0.0
    if drifting:
        out.morphology = "narrowband, drifting"
    elif narrow and out.persistence >= PERSISTENT_FRACTION:
        out.morphology = "narrowband, persistent"
    elif narrow:
        out.morphology = "narrowband, intermittent"
    elif out.impulsive_snr >= IMPULSIVE_SNR:
        out.morphology = "broadband, impulsive"
    else:
        out.morphology = "no clear narrowband or impulsive structure"
    return out


def assess_candidate(
    score: float,
    cutoff: float,
    f_low_mhz: float,
    f_high_mhz: float,
    chars: Optional[Characteristics],
    meta: Optional[SourceMeta] = None,
    f0_channel: Optional[int] = None,
    cadence_verdict: Optional[str] = None,
) -> Assessment:
    """Combine measured evidence into a labelled, explained, preliminary assessment."""
    evidence: List[Evidence] = []
    missing: List[str] = []
    bands = known_bands_for(f_low_mhz, f_high_mhz)
    above = bool(np.isfinite(score) and score > cutoff)

    if above:
        evidence.append(Evidence("info", "Unusualness score above the cutoff", f"Score {score:.5f} exceeds {cutoff:.5f}."))

    # --- frequency context
    if not np.isfinite(f_low_mhz):
        missing.append("Tile position is unavailable, so frequency context could not be checked.")
    elif bands:
        names = "; ".join(f"{b.name} ({b.low_mhz:g}–{b.high_mhz:g} MHz)" for b in bands)
        evidence.append(Evidence("info", "Frequency overlaps a reference allocation", names + ". " + "Context only."))
    else:
        evidence.append(Evidence("info", "No overlap with the reference allocations", "The tile's frequency range is outside the indicative list."))

    # --- morphology
    morph_rfi = False
    if chars is None:
        missing.append("Tile data could not be read, so narrowband, persistence, drift and impulsive structure were not measured.")
    else:
        detail = f"{chars.morphology}"
        if chars.narrowband_snr is not None:
            detail += f" (strongest channel {chars.narrowband_snr:.1f} σ above the local bandpass, width {chars.narrowband_width_channels} ch"
            if chars.persistence is not None:
                detail += f", present in {chars.persistence * 100:.0f}% of time samples"
            detail += ")"
        evidence.append(Evidence("info", "Measured structure", detail))
        if chars.drift_hz_per_s not in (None, 0.0):
            evidence.append(Evidence("info", "Frequency drift", f"{chars.drift_hz_per_s:+.3g} Hz/s across the tile (fit R² {chars.drift_r2:.2f})."))
        elif chars.drift_hz_per_s == 0.0 and chars.morphology.startswith("narrowband"):
            evidence.append(Evidence("info", "Frequency drift", "No measurable drift at this resolution."))
        if chars.freq_resolution_khz:
            evidence.append(Evidence(
                "info", "Resolution limits",
                f"Channels are {chars.freq_resolution_khz:g} kHz wide and time samples {chars.time_resolution_s:g} s; "
                "structure finer than that cannot be seen.",
            ))
        narrowband = chars.morphology.startswith("narrowband")
        impulsive = chars.morphology == "broadband, impulsive"
        if narrowband and any(b.signal_type == CONTINUOUS for b in bands):
            morph_rfi = True
            hit = next(b for b in bands if b.signal_type == CONTINUOUS)
            evidence.append(Evidence("rfi", "Narrowband signal inside a carrier-type allocation", f"{chars.morphology} within {hit.name}."))
        if impulsive and any(b.signal_type == PULSED for b in bands):
            morph_rfi = True
            hit = next(b for b in bands if b.signal_type == PULSED)
            evidence.append(Evidence("rfi", "Broadband impulsive event inside a radar allocation", f"Short broadband rise within {hit.name}."))

    # --- cadence
    cadence_rfi = False
    if cadence_verdict == "LOCAL_RFI":
        cadence_rfi = True
        evidence.append(Evidence("rfi", "Also found in OFF-target pointings", "A signal that appears off-source is consistent with local interference."))
    elif cadence_verdict == "CANDIDATE_TECHNOSIGNATURE":
        evidence.append(Evidence("info", "Present in ON-target only", "Not seen in the OFF pointings supplied. This is a reason to look further, not evidence of origin."))
    elif cadence_verdict is None:
        missing.append("No ON/OFF cadence has been run for this candidate.")

    # --- instrument: band edge
    edge = False
    if meta is not None and f0_channel is not None:
        if f0_channel == 0 or f0_channel + TILE_F >= meta.nchans:
            edge = True
            evidence.append(Evidence("artefact", "Tile touches the edge of the recorded band", "Receiver bandpass typically rolls off at the band edge, which can look unusual to the model."))

    # --- decide
    if cadence_rfi or morph_rfi:
        category = RFI
        why = ("Seen in OFF-target pointings as well." if cadence_rfi else
               "Its measured shape matches what the overlapping service would produce.")
    elif edge:
        category = ARTEFACT
        why = "It lies at the edge of the recorded band, where bandpass roll-off is expected."
    elif above:
        category = UNUSUAL
        why = "It scores above the applicable cutoff and none of the preliminary interference or instrument rules apply."
    else:
        category = INSUFFICIENT
        why = "It does not exceed the cutoff and no source evidence was found."
    if not above and category in (RFI, ARTEFACT):
        why = "(Below the cutoff.) " + why
    headline = f"{category} (preliminary)"
    return Assessment(category, headline, why, evidence, missing, chars, bands)

