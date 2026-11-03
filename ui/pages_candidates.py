"""Candidates: ranked, filterable list on the left and a full investigation view on the right."""

from __future__ import annotations

import zlib

import pandas as pd
import streamlit as st

from src.observatory.assessment import CATEGORIES
from src.observatory.candidates import filter_candidates
from ui.candidate_views import assess, candidate_table, render_detail, select
from ui.components import callout, page_head
from ui.context import Ctx, state_panel, status_chip
from ui.guide import glossary_popover

PAGE_SIZE = 40
ASSESS_LIMIT = 500
SORTS = {
    "Unusualness": ("score", False),
    "Frequency": ("f_center_mhz", True),
    "Elapsed time": ("t_start_s", True),
}
STATUS_OPTIONS = ["All", "Pending", "Reviewed", "Retained", "Flagged", "Dismissed"]


def _filters(ctx: Ctx) -> pd.DataFrame:
    view = ctx.view
    c1, c2, c3, c4 = st.columns([1.1, 1.5, 1.2, 1.6])
    status = c1.selectbox("Review status", STATUS_OPTIONS, key="f_status")
    categories = []
    if ctx.can_read_data:
        categories = c2.multiselect("Source assessment", CATEGORIES, key="f_cat", placeholder="Any", help="Preliminary, rule-based. See the assessment notes on each candidate.")
    else:
        c2.selectbox("Source assessment", ["Needs readable tile data"], disabled=True, key="f_cat_off")
    sort_by = c3.selectbox("Sort by", list(SORTS), key="f_sort", disabled=False)
    f_range = None
    if ctx.has_coordinates and view["f_center_mhz"].notna().any():
        lo, hi = float(view["f_center_mhz"].min()), float(view["f_center_mhz"].max())
        if hi > lo:
            f_range = c4.slider("Frequency (MHz)", lo, hi, (lo, hi), key=f"f_freq_{ctx.obs.obs_id}", format="%.2f")
            if f_range == (lo, hi):
                f_range = None
    else:
        c4.selectbox("Frequency (MHz)", ["Needs tile positions"], disabled=True, key="f_freq_off")

    statuses = {c: ctx.status_of(c) for c in view["candidate_id"]}
    out = filter_candidates(view, statuses, status, None, None, f_range)

    if categories:
        sample = out.head(ASSESS_LIMIT)
        with st.spinner("Measuring candidates…"):
            cat_of = {r.candidate_id: assess(ctx, r).category for r in sample.itertuples(index=False)}
        out = filter_candidates(sample, statuses, "All", cat_of, categories, None)
        if len(view) > ASSESS_LIMIT:
            st.caption(f"Assessment filter applied to the top {ASSESS_LIMIT} candidates by score.")

    col, ascending = SORTS[sort_by]
    if col != "score" or ascending:
        out = out.sort_values(col, ascending=ascending, na_position="last", kind="stable")
    return out


def render(ctx: Ctx) -> None:
    sub = ctx.obs.label if ctx.obs else ""
    h, g = st.columns([5, 1.3])
    with h:
        page_head("Candidates", sub, status_chip(ctx))
    with g:
        glossary_popover()
    if not state_panel(ctx):
        return
    if ctx.view.empty:
        callout(
            "info", "No candidates match this sensitivity",
            "<p>Every tile scored below the cutoff. " + ("Try <b>High</b> sensitivity or inspect another observation.</p>" if ctx.level != "High" else "Inspect another observation.</p>"),
        )
        return

    rows = _filters(ctx)
    total = len(rows)
    pages = max((total - 1) // PAGE_SIZE + 1, 1)
    cap, pg = st.columns([5, 1]) if pages > 1 else (st.container(), None)
    with cap:
        st.caption(f"{total:,} of {len(ctx.view):,} candidates shown \u00b7 {ctx.level} sensitivity \u00b7 {ctx.cutoff.description}"
                   + (f" \u00b7 page {st.session_state.get(f'f_page_{ctx.obs.obs_id}', 1)} of {pages}" if pages > 1 else ""))
    if rows.empty:
        callout("info", "No candidates match these filters", "<p>Clear a filter to see candidates again.</p>")
        return
    page = 1
    if pg is not None:
        page = int(pg.number_input("Page", 1, pages, 1, key=f"f_page_{ctx.obs.obs_id}", help=f"Page of {pages}. {PAGE_SIZE} candidates per page.", label_visibility="collapsed"))
    window = rows.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]

    ids = window["candidate_id"].tolist()
    if st.session_state.get("selected_candidate") not in ids:
        st.session_state["selected_candidate"] = ids[0]

    left, right = st.columns([6, 5], gap="medium")
    with left:
        st.markdown('<span class="split-marker"></span>', unsafe_allow_html=True)
        with st.spinner("Preparing candidate previews…"):
            picked = candidate_table(ctx, window, key=f"cand_{zlib.crc32('|'.join(ids).encode())}", height=640, compact=True)
        if picked:
            select(picked)
        st.caption("Click a row to open it. Click a column heading to sort the rows on this page.")
    with right:
        sel_id = st.session_state["selected_candidate"]
        row = next(window[window["candidate_id"] == sel_id].itertuples(index=False))
        render_detail(ctx, row, key="cand")
