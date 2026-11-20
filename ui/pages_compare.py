"""Compare: two real candidates side by side, or two observations."""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.observatory.assessment import Characteristics
from src.observatory.candidates import AnalysisKind, apply_sensitivity, sensitivity_cutoff
from src.observatory.reviews import ReviewStoreError
from src.observatory.scoring import normalize_tiles
from ui import loaders
from ui.candidate_views import PLOT_CONFIG, SHORT_CATEGORY
from ui.components import (
    COLOR_ACCENT_HI, REVIEW_LABELS, callout, fmt_khz, fmt_range, fmt_score, fmt_seconds, fmt_utc, page_head, plot_caption,
    style_plotly_figure,
)
from ui.context import Ctx, state_panel, status_chip
from ui.plots import tile_figure, window_axes

COLOR_B = "#c9d1d6"


def _same_resolution(a, b) -> bool:
    return (
        abs(abs(a.foff_mhz) - abs(b.foff_mhz)) <= 0.01 * abs(a.foff_mhz)
        and abs(a.tsamp_s - b.tsamp_s) <= 0.01 * a.tsamp_s
    )


def _load_side(ctx: Ctx, tag: str, default_obs_id: str, default_cid: Optional[str]):
    """Pick an observation and candidate, return everything needed to show it (or None)."""
    usable = [o for o in ctx.observations if o.meta is not None and (o.results is not None or True)]
    by_id = {o.obs_id: o for o in usable}
    ids = list(by_id)
    if st.session_state.get(f"cmp_obs_{tag}") not in ids:
        st.session_state[f"cmp_obs_{tag}"] = default_obs_id if default_obs_id in ids else ids[0]
    obs_id = st.selectbox(f"Observation {tag}", ids, key=f"cmp_obs_{tag}", format_func=lambda i: by_id[i].label)
    obs = by_id[obs_id]
    cs = loaders.candidate_set(obs)
    if cs.kind not in (AnalysisKind.READY,) or cs.frame.empty or not cs.has_coordinates:
        callout("warn", f"Nothing to compare in {obs.label}", f"<p>{cs.message or 'No candidates with measured positions.'}</p>")
        return None
    cutoff = sensitivity_cutoff(ctx.level, cs.frame["score"], obs.calibrated_threshold)
    view = apply_sensitivity(cs.frame, cutoff).head(150)
    if view.empty:
        st.info(f"No candidates in {obs.label} at {ctx.level} sensitivity.")
        return None
    ids_c = view["candidate_id"].tolist()
    if st.session_state.get(f"cmp_{tag.lower()}") not in ids_c:
        st.session_state[f"cmp_{tag.lower()}"] = default_cid if default_cid in ids_c else ids_c[0]
    lookup = view.set_index("candidate_id")
    cid = st.selectbox(
        f"Candidate {tag}", ids_c, key=f"cmp_{tag.lower()}",
        format_func=lambda c: f"#{int(lookup.loc[c, 'rank'])}  ·  {lookup.loc[c, 'f_center_mhz']:,.3f} MHz  ·  {lookup.loc[c, 'score']:.5f}",
    )
    row = next(cs.frame[cs.frame["candidate_id"] == cid].itertuples(index=False))
    raw = loaders.window_for(obs, row.t0, row.f0)
    a = loaders.assessment_for(obs, row, cutoff.value, (loaders.cadence_log().read().get(cid) or {}).get("verdict"))
    return dict(obs=obs, cs=cs, cutoff=cutoff, row=row, raw=raw, tile=normalize_tiles(raw[None])[0], assessment=a, cid=cid)


def _profiles(a, b, same: bool) -> None:
    """Time-averaged spectrum and frequency-averaged time series of each (scaled) tile."""
    def freq_axis(side):
        m = side["obs"].meta
        x = (np.arange(64) - 31.5) * m.foff_mhz * 1000.0       # kHz from tile centre, signed by channel direction
        prof = side["tile"].mean(axis=0)
        order = np.argsort(x)
        return x[order], prof[order]

    def time_axis(side):
        m = side["obs"].meta
        return np.arange(64) * m.tsamp_s, side["tile"].mean(axis=1)

    for title, fn, xtitle, ytitle in (
        ("Spectrum within the tile (averaged over time)", freq_axis, "Offset from tile centre (kHz)", "Scaled level"),
        ("Time series within the tile (averaged over frequency)", time_axis, "Seconds from tile start", "Scaled level"),
    ):
        if same:
            fig = go.Figure()
            for tag, side, color, dash in (("A", a, COLOR_ACCENT_HI, "solid"), ("B", b, COLOR_B, "dot")):
                x, y = fn(side)
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"Candidate {tag}", line=dict(color=color, width=2, dash=dash)))
            style_plotly_figure(fig, height=250, title=title, xaxis_title=xtitle, yaxis_title=ytitle)
            st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, key=f"cmp_prof_{title[:8]}")
        else:
            c1, c2 = st.columns(2)
            for col, tag, side, color in ((c1, "A", a, COLOR_ACCENT_HI), (c2, "B", b, COLOR_B)):
                x, y = fn(side)
                fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=2)))
                style_plotly_figure(fig, height=250, title=f"{title.split(' (')[0]}: {tag}", xaxis_title=xtitle, yaxis_title=ytitle)
                col.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, key=f"cmp_prof_{tag}_{title[:8]}")


def _metric_rows(ctx: Ctx, side) -> dict:
    obs, row, a, m = side["obs"], side["row"], side["assessment"], side["obs"].meta
    chars: Optional[Characteristics] = a.characteristics
    scores = np.sort(side["cs"].frame["score"].to_numpy())
    pct = float(np.searchsorted(scores, row.score)) / max(len(scores), 1) * 100
    cadence = loaders.cadence_log().read().get(side["cid"])
    store_review = loaders.review_store().get(side["cid"])
    start = fmt_utc(m.start_utc + timedelta(seconds=float(row.t_start_s))) if m.start_utc else "not in file"
    return {
        "Observation": obs.label,
        "Telescope": m.telescope or "not identified in file",
        "Band": obs.band or "unknown",
        "Frequency": fmt_range(row.f_low_mhz, row.f_high_mhz, 4),
        "Elapsed time": f"{row.t_start_s:,.2f}–{row.t_end_s:,.2f} s",
        "Starts at": start,
        "Channel width": fmt_khz(m.foff_mhz),
        "Time resolution": fmt_seconds(m.tsamp_s),
        "Unusualness score": fmt_score(row.score),
        "Times the cutoff": f"{row.score / side['cutoff'].value:.2f}×",
        "Higher than": f"{pct:.1f}% of {len(scores):,} tiles",
        "Assessment (preliminary)": SHORT_CATEGORY.get(a.category, a.category),
        "Measured structure": chars.morphology if chars else "not measured",
        "Strongest channel": f"{chars.narrowband_snr:.1f} σ above bandpass" if chars and chars.narrowband_snr is not None else "n/a",
        "Present in time samples": f"{chars.persistence * 100:.0f}%" if chars and chars.persistence is not None else "n/a",
        "Frequency drift": (f"{chars.drift_hz_per_s:+.3g} Hz/s" if chars and chars.drift_hz_per_s else ("none measurable" if chars and chars.drift_hz_per_s == 0.0 else "n/a")),
        "Known allocations": "; ".join(b.name for b in a.bands) or "none in the reference list",
        "ON/OFF cadence": {"CANDIDATE_TECHNOSIGNATURE": "ON only", "LOCAL_RFI": "also OFF-source", "NORMAL_BACKGROUND": "not unusual"}.get(cadence["verdict"], cadence["verdict"]) if cadence else "not run",
        "Your review": REVIEW_LABELS[store_review.status] if store_review else "Pending",
    }


def _candidates_tab(ctx: Ctx) -> None:
    default = ctx.obs.obs_id if ctx.obs else ""
    cmp_a = st.session_state.get("cmp_a")
    ca, cb = st.columns(2, gap="large")
    with ca:
        a = _load_side(ctx, "A", default, cmp_a)
    with cb:
        second = None
        if ctx.view is not None and len(ctx.view) > 1:
            ids = ctx.view["candidate_id"].tolist()
            second = next((c for c in ids if c != (cmp_a or ids[0])), None)
        b = _load_side(ctx, "B", default, second)
    if a is None or b is None:
        return
    if a["cid"] == b["cid"]:
        st.info("A and B are the same candidate. Choose a different one for B.")

    same = _same_resolution(a["obs"].meta, b["obs"].meta)
    ta, tb = st.columns(2, gap="large")
    for col, tag, side in ((ta, "A", a), (tb, "B", b)):
        m = side["obs"].meta
        t_axis, f_axis, flipped = window_axes(m, int(side["row"].t0), int(side["row"].f0), 64, 64)
        arr = side["tile"].T[::-1] if flipped else side["tile"].T
        col.plotly_chart(
            tile_figure(arr, t_axis, f_axis, f"Candidate {tag}: observed", height=300, zrange=(0.0, 1.0), value_label="level"),
            width="stretch", config=PLOT_CONFIG, key=f"cmp_tile_{tag}",
        )
    if same:
        plot_caption("Both tiles have the same channel width and time resolution, so their shapes can be compared directly. Each is scaled 0–1 within its own tile.")
    else:
        callout(
            "warn", "These candidates have different resolutions",
            f"<p>A has {fmt_khz(a['obs'].meta.foff_mhz)} channels and {fmt_seconds(a['obs'].meta.tsamp_s)} samples; "
            f"B has {fmt_khz(b['obs'].meta.foff_mhz)} channels and {fmt_seconds(b['obs'].meta.tsamp_s)} samples. "
            "Each is drawn on its own axes and the profiles are not overlaid, because lines at different resolutions would suggest a match that does not exist.</p>",
        )
    _profiles(a, b, same)

    st.markdown("**Side by side**")
    ra, rb = _metric_rows(ctx, a), _metric_rows(ctx, b)
    table = pd.DataFrame({"": list(ra), "Candidate A": list(ra.values()), "Candidate B": list(rb.values())})
    st.dataframe(table, hide_index=True, width="stretch", height=38 * (len(table) + 1))


def _observations_tab(ctx: Ctx) -> None:
    rows = []
    for o in ctx.observations:
        cs = loaders.candidate_set(o)
        n = None
        if cs.kind == AnalysisKind.READY and not cs.frame.empty:
            cut = sensitivity_cutoff(ctx.level, cs.frame["score"], o.calibrated_threshold)
            n = int((cs.frame["score"] > cut.value).sum())
        try:
            reviews = loaders.review_store().for_observation(o.obs_id)
        except ReviewStoreError:
            reviews = {}
        m = o.meta
        rows.append({
            "Observation": o.label,
            "Telescope": (m.telescope if m else None) or (o.results.summary.get("telescope") if o.results else None) or "n/a",
            "Band": o.band or "n/a",
            "Frequency range": fmt_range(m.freq_low_mhz, m.freq_high_mhz, 1) if m else "n/a",
            "Duration": fmt_seconds(m.duration_s) if m else "n/a",
            "Analysis": cs.basis or ("not analysed" if cs.kind == AnalysisKind.NOT_ANALYZED else "unavailable"),
            "Tiles scored": cs.n_scored or None,
            f"Unusual signals ({ctx.level})": n,
            "Reviewed": sum(1 for r in reviews.values() if r.status != "pending"),
            "Retained": sum(1 for r in reviews.values() if r.status == "retained"),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Unusualness scores are only comparable within one observation: each is judged against its own cutoff, so counts "
        "across telescopes and bands show where to look, not which observation is more interesting."
    )


def render(ctx: Ctx) -> None:
    page_head("Compare", ctx.obs.label if ctx.obs else "", status_chip(ctx))
    if ctx.obs is None:
        state_panel(ctx)
        return
    if len([o for o in ctx.observations if o.meta is not None]) == 0:
        callout("warn", "Comparing needs observation data files", "<p>No observation files are available. Add one with <b>Add observation</b>.</p>")
        return
    t1, t2 = st.tabs(["Two candidates", "Two or more observations"])
    with t1:
        _candidates_tab(ctx)
    with t2:
        _observations_tab(ctx)
