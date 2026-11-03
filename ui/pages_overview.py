"""Overview: the observation, how many unusual signals, where they sit, and what to look at first."""

from __future__ import annotations

import re
from html import escape

import streamlit as st

from src.observatory.candidates import AnalysisKind
from src.observatory.scoring import scan_observation
from ui import loaders
from ui.candidate_views import PLOT_CONFIG, assess, candidate_table, goto
from ui.components import (
    callout, facts_grid, fmt_bytes, fmt_khz, fmt_range, fmt_seconds, fmt_utc, page_head, plot_caption,
)
from ui.context import Ctx, state_panel, status_chip
from ui.guide import first_run_guide
from ui.plots import waterfall_figure

PREVIEW_ROWS = 8
ASSESS_LIMIT = 300


def _facts(ctx: Ctx) -> None:
    m = ctx.obs.meta
    if m is None:
        s = ctx.obs.results.summary if ctx.obs.results else {}
        facts_grid([
            ("Telescope", str(s.get("telescope", "not recorded")), "" if s.get("telescope") else "na"),
            ("Band", str(s.get("band", "not recorded")), "" if s.get("band") else "na"),
            ("Source file", "not available", "na"),
        ])
        return
    facts_grid([
        ("Telescope", " \u00b7 ".join(x for x in (m.telescope, m.band) if x) or "not in file", "" if (m.telescope or m.band) else "na"),
        ("Frequency range", fmt_range(m.freq_low_mhz, m.freq_high_mhz, 2), "num"),
        ("Channel width", fmt_khz(m.foff_mhz), "num"),
        ("Time resolution", fmt_seconds(m.tsamp_s), "num"),
        ("Duration", fmt_seconds(m.duration_s), "num"),
        ("Started", fmt_utc(m.start_utc), "" if m.start_utc else "na"),
    ])
    name = re.sub(r"^[0-9a-f]{8}_", "", m.path.name) if ctx.obs.origin == "uploaded" else m.path.name
    st.markdown(f'<div class="file-line">File: {escape(name)} · {fmt_bytes(m.size_bytes)}</div>', unsafe_allow_html=True)


def _run_scan(ctx: Ctx) -> None:
    """Explicit user action: score every tile of the observation, with progress."""
    bar = st.progress(0.0, text="Analysing the observation…")
    try:
        scan_observation(ctx.source, ctx.bundle, progress=lambda p: bar.progress(min(p, 1.0), text=f"Analysing the observation… {p * 100:.0f}%"))
    except Exception as exc:
        bar.empty()
        st.session_state["scan_error"] = f"The analysis stopped: {exc}"
        return
    bar.empty()
    st.session_state["scan_error"] = None
    st.session_state.pop("selected_candidate", None)
    st.toast("Analysis complete.")
    st.rerun()


def _scan_controls(ctx: Ctx, primary: bool) -> None:
    if ctx.source is None:
        callout("warn", "The source data file is not available", "<p>Analysis needs the observation file. Add it with <b>Add observation</b> in the sidebar.</p>")
        return
    if ctx.bundle is None:
        callout("bad", "The model is unavailable", f"<p>{escape(ctx.model_error or 'No checkpoint could be loaded.')}</p>")
        return
    size = fmt_bytes(ctx.obs.meta.size_bytes)
    label = "Analyse observation" if primary else "Analyse the whole observation"
    if st.button(label, type="primary" if primary else "secondary", key="run_scan", help=f"Reads the whole file once ({size}) and scores every tile. Files of 1 to 2 GB typically take under a minute. The file is not modified."):
        _run_scan(ctx)
    if st.session_state.get("scan_error"):
        st.error(st.session_state["scan_error"])


def render(ctx: Ctx) -> None:
    first_run_guide()
    if ctx.obs is None:
        page_head("SETI-AI Observatory", "Find, inspect and review unusual radio signals.")
        state_panel(ctx)
        return
    page_head(ctx.obs.label, "", status_chip(ctx))
    _facts(ctx)

    cs = ctx.cset
    if cs is not None and cs.kind == AnalysisKind.NOT_ANALYZED:
        callout(
            "info", "This observation has not been analysed yet",
            "<p>The file is ready. Analysing scores every 64×64 tile for how unusual it looks, then shows the strongest ones here. "
            "The file is only read, never changed.</p>",
        )
        _scan_controls(ctx, primary=True)
        return
    if not state_panel(ctx):
        return

    view = ctx.view
    n = len(view)
    noun = "unusual signal" if n == 1 else "unusual signals"
    where = "" if ctx.has_coordinates else " Tile positions are unavailable, so they cannot be placed on the strip."

    # ---- strip
    partial = cs.basis == "held-out tiles" and cs.n_grid_total
    if ctx.source is not None and ctx.has_coordinates:
        with st.spinner("Preparing the overview of the observation (first time only)\u2026"):
            ov = loaders.overview_for(ctx.obs)
        st.plotly_chart(
            waterfall_figure(ov, view, ctx.selected_id, height=215, compact=True, coverage=cs.frame if partial else None),
            width="stretch", config=PLOT_CONFIG, key="ov_strip",
        )
        note = (f" Faint dots show the {cs.n_scored:,} tiles analysed ({cs.coverage_fraction * 100:.0f}% of the observation); "
                "unmarked areas were not analysed, so they are unknown, not clear.") if partial else ""
        plot_caption(
            f"Each column averages {fmt_seconds(ov.dt_s)} and each row {fmt_khz(ov.df_mhz)}. Amber diamonds mark unusual signals at their "
            f"measured time and frequency.{note}"
        )
    elif ctx.source is None:
        callout("warn", "The overview needs the source data file", "<p>Scores and rankings are shown below. Add the file to see the observation itself.</p>")

    # ---- headline and primary action
    st.markdown(f'<div class="lede"><span class="n">{n:,}</span> {noun} at {escape(ctx.level)} sensitivity</div>', unsafe_allow_html=True)
    if n:
        row = st.container(horizontal=True)
        row.button("Review top candidate", type="primary", key="ov_review", on_click=goto, args=("Candidates", view.iloc[0]["candidate_id"]), width="content")
        row.button("See all candidates", key="ov_all", on_click=goto, args=("Candidates",), width="content")
    basis = {"held-out tiles": "From the saved evaluation of held-out tiles.", "full scan": "From a scan of the whole observation."}.get(cs.basis, "")
    st.markdown(
        f'<div class="lede-sub">{basis}{where} A high score means a tile looks different from typical data; it does not identify the signal.</div>',
        unsafe_allow_html=True,
    )
    for w in cs.warnings:
        st.warning(w)

    if n == 0:
        callout(
            "info", "No tile scores above the cutoff at this sensitivity",
            "<p>That is a result, not an error: every tile looked like typical data. "
            + ("Try <b>High</b> sensitivity to see weaker candidates.</p>" if ctx.level != "High" else "No weaker candidates are available either.</p>"),
        )
    else:
        top = view.head(PREVIEW_ROWS)
        with st.spinner("Preparing candidate previews\u2026"):
            candidate_table(ctx, top, key="ov_table", selectable=False, height=min(40 + 54 * len(top), 40 + 54 * PREVIEW_ROWS))
        if n > PREVIEW_ROWS:
            st.caption(f"Showing the top {PREVIEW_ROWS} of {n:,}.")

    # ---- summaries
    left, right = st.columns(2)
    with left:
        st.markdown("**Source assessment (preliminary)**")
        if ctx.can_read_data and n:
            sample = view.head(ASSESS_LIMIT)
            counts = {}
            with st.spinner("Measuring candidates…"):
                for r in sample.itertuples(index=False):
                    a = assess(ctx, r)
                    counts[a.category] = counts.get(a.category, 0) + 1
            scope = f"the top {len(sample)} of {n:,}" if n > len(sample) else f"all {n:,}"
            rows = "".join(f"<li><b>{c}</b> {escape(k)}</li>" for k, c in sorted(counts.items(), key=lambda kv: -kv[1]))
            st.markdown(f'<div class="missing" style="margin-top:0">Rule-based checks on {scope} candidates. Not a classifier.</div><ul>{rows}</ul>', unsafe_allow_html=True)
        else:
            st.caption("Available once candidates with readable data exist.")
    with right:
        st.markdown("**Your review**")
        statuses = [ctx.status_of(c) for c in view["candidate_id"]] if n else []
        pending = statuses.count("pending")
        reviewed = n - pending
        st.markdown(
            f"<ul><li><b>{pending:,}</b> pending</li><li><b>{statuses.count('retained'):,}</b> retained</li>"
            f"<li><b>{statuses.count('flagged'):,}</b> flagged</li><li><b>{statuses.count('dismissed'):,}</b> dismissed</li></ul>",
            unsafe_allow_html=True,
        )
        if reviewed:
            st.button("Export your review", key="ov_export", on_click=goto, args=("Export",))

    st.caption("To look at another observation, use **Add observation** in the sidebar: upload a file, open one from this computer, or search the Breakthrough Listen archive.")
    if cs.basis == "held-out tiles":
        st.markdown("**Cover the whole observation**")
        st.caption(
            "These candidates come from a saved evaluation of a held-out sample of tiles. Analysing the whole observation scores every tile "
            "and may reveal more. Tiles the model was trained on may score lower than new data."
        )
        _scan_controls(ctx, primary=False)
