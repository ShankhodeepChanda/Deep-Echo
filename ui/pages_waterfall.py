"""Waterfall: the whole observation as a time-frequency picture with candidates at their measured positions."""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.observatory.provenance import TILE_F, TILE_T
from ui import loaders
from ui.candidate_views import PLOT_CONFIG, assess, goto, select
from ui.components import (
    badge, fmt_freq, fmt_khz, fmt_range, fmt_score, fmt_seconds, kv_block, page_head, plot_caption,
)
from ui.context import Ctx, state_panel, status_chip
from ui.guide import glossary_popover
from ui.plots import zoom_figure

ZOOM_BEFORE = 64     # context shown around the tile, in samples / channels
ZOOM_AFTER = 128


def _clicked_id(event) -> str | None:
    try:
        points = event.selection.points
    except AttributeError:
        return None
    for p in points or []:
        cid = p.get("customdata")
        if isinstance(cid, (list, tuple)):
            cid = cid[0] if cid else None
        if isinstance(cid, str):
            return cid
    return None


def render(ctx: Ctx) -> None:
    h, g = st.columns([5, 1.3])
    with h:
        page_head("Waterfall", ctx.obs.label if ctx.obs else "", status_chip(ctx))
    with g:
        glossary_popover()
    if not state_panel(ctx, need_data=True, what="the waterfall"):
        return
    from ui.plots import waterfall_figure

    view = ctx.view
    c1, c2 = st.columns([2, 5])
    show_marks = c1.toggle("Show unusual signals", value=True, key="wf_marks", help="Amber diamonds mark tiles scoring above the cutoff at the current sensitivity.")
    c2.markdown('<div class="plot-caption" style="margin:0.5rem 0 0"><span class="legend-mark"></span>unusual signal. Drag to zoom, double-click to reset, click a diamond to inspect it below.</div>', unsafe_allow_html=True)

    with st.spinner("Preparing the overview of the observation (first time only)…"):
        ov = loaders.overview_for(ctx.obs)

    selected = ctx.selected_id if (ctx.selected_id and ctx.row(ctx.selected_id) is not None) else (view.iloc[0]["candidate_id"] if len(view) else None)
    partial = ctx.cset.basis == "held-out tiles" and bool(ctx.cset.n_grid_total)
    fig = waterfall_figure(ov, view if show_marks else None, selected, height=520, coverage=ctx.cset.frame if partial else None)
    event = st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
                            key="wf_chart", on_select="rerun", selection_mode="points")
    plot_caption(
        f"Each pixel averages {fmt_seconds(ov.dt_s)} of time and {fmt_khz(ov.df_mhz)} of frequency (the recording is {fmt_seconds(ov.native_dt_s)} "
        f"by {fmt_khz(ov.native_df_mhz)}). Intensity is relative and uncalibrated. "
        + (f"Faint dots are the {ctx.cset.n_scored:,} tiles analysed ({ctx.cset.coverage_fraction * 100:.0f}%); unmarked areas were not analysed, so they are unknown, not clear. " if partial else "")
        + (f"{min(len(view), 600):,} of {len(view):,} candidates drawn." if show_marks and len(view) > 600 else "")
    )
    clicked = _clicked_id(event)
    if clicked and clicked != ctx.selected_id:
        select(clicked)
        selected = clicked

    if selected is None:
        st.info("No candidates at this sensitivity. Try High sensitivity to see weaker ones.")
        return
    row = ctx.row(selected)
    if row is None or np.isnan(row.t0):
        return

    st.markdown(f"**Candidate #{int(row.rank)} at full detail**")
    meta = ctx.obs.meta
    t_lo = int(max(row.t0 - ZOOM_BEFORE, 0))
    f_lo = int(max(row.f0 - ZOOM_BEFORE, 0))
    nt = int(min(row.t0 + TILE_T + ZOOM_AFTER, meta.n_times) - t_lo)
    nf = int(min(row.f0 + TILE_F + ZOOM_AFTER, meta.nchans) - f_lo)
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown('<span class="split-marker"></span>', unsafe_allow_html=True)
        win = loaders.window_for(ctx.obs, t_lo, f_lo, nt, nf)
        st.plotly_chart(zoom_figure(win, meta, t_lo, f_lo, box=(int(row.t0), int(row.f0), TILE_T, TILE_F), height=420),
                        width="stretch", config=PLOT_CONFIG, key="wf_zoom")
        plot_caption(
            f"Native resolution ({fmt_seconds(meta.tsamp_s)} by {fmt_khz(meta.foff_mhz)}). The amber box is the 64×64 tile the model scored; "
            "the area around it is context."
        )
    with right:
        a = assess(ctx, row)
        st.markdown(badge(a.category) if a else "", unsafe_allow_html=True)
        kv_block([
            ("Frequency", fmt_range(row.f_low_mhz, row.f_high_mhz, 4), True),
            ("Centre", fmt_freq(row.f_center_mhz, 4), True),
            ("Elapsed time", f"{row.t_start_s:,.3f}–{row.t_end_s:,.3f} s", True),
            ("Unusualness", f"{fmt_score(row.score)}  ({row.score / ctx.cutoff.value:.2f}× cutoff)", True),
            ("Review", ctx.status_of(row.candidate_id).capitalize(), False),
        ])
        if a is not None:
            st.caption(a.why)
        b1, b2 = st.columns(2)
        b1.button("Open candidate", key="wf_open", on_click=goto, args=("Candidates", row.candidate_id), width="stretch", type="primary")
        b2.button("Check cadence", key="wf_cad", on_click=goto, args=("Cadence Analysis", row.candidate_id), width="stretch")
