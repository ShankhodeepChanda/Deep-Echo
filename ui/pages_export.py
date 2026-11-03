"""Export: a reviewable candidate report as CSV, printable HTML and PDF."""

from __future__ import annotations

import zlib

import pandas as pd
import streamlit as st

from src.observatory import export
from src.observatory.scoring import normalize_tiles
from ui import loaders
from ui.candidate_views import assess
from ui.components import callout, page_head
from ui.context import Ctx, state_panel, status_chip

ASSESS_LIMIT = 1000
REPORT_LIMIT = 150
SCOPES = ["Retained and flagged", "Everything I have reviewed", "All candidates at this sensitivity", "The selected candidate"]


def _scope_rows(ctx: Ctx, scope: str) -> pd.DataFrame:
    frame = ctx.cset.frame
    if scope == "Retained and flagged":
        ids = {c for c, r in ctx.reviews.items() if r.status in ("retained", "flagged")}
        return frame[frame["candidate_id"].isin(ids)]
    if scope == "Everything I have reviewed":
        ids = {c for c, r in ctx.reviews.items() if r.status != "pending" or r.note}
        return frame[frame["candidate_id"].isin(ids)]
    if scope == "All candidates at this sensitivity":
        return ctx.view
    row = ctx.selected
    return frame[frame["candidate_id"] == ctx.selected_id] if row is not None else frame.iloc[0:0]


def render(ctx: Ctx) -> None:
    page_head("Export", ctx.obs.label if ctx.obs else "", status_chip(ctx))
    if not state_panel(ctx):
        return

    n_ret = sum(1 for r in ctx.reviews.values() if r.status == "retained")
    n_flag = sum(1 for r in ctx.reviews.values() if r.status == "flagged")
    n_dis = sum(1 for r in ctx.reviews.values() if r.status == "dismissed")
    st.markdown(f"**Your review of {ctx.obs.label}:** {n_ret} retained · {n_flag} flagged · {n_dis} dismissed.")

    has_kept = (n_ret + n_flag) > 0
    scope = st.radio("What to include", SCOPES, horizontal=True, key="exp_scope", index=0 if has_kept else 2)
    rows = _scope_rows(ctx, scope).sort_values("rank")
    if rows.empty:
        callout(
            "info", "Nothing to export for this choice",
            "<p>" + ("Retain or flag candidates on the <b>Candidates</b> page first." if scope in SCOPES[:2] else
                     "There are no candidates at this sensitivity." if scope == SCOPES[2] else "Select a candidate first.") + "</p>",
        )
        return

    assessments, cadence = {}, {}
    if ctx.can_read_data:
        with st.spinner("Assessing candidates for the report…"):
            for r in rows.head(ASSESS_LIMIT).itertuples(index=False):
                assessments[r.candidate_id] = assess(ctx, r)
    for c in rows["candidate_id"]:
        entry = ctx.cadence.get(c)
        if entry:
            cadence[c] = entry["verdict"]
    frame = export.export_frame(
        rows, ctx.obs, ctx.cutoff.value, ctx.level, ctx.cutoff.basis, ctx.cset.basis, ctx.reviews, assessments, cadence,
    )
    if len(rows) > ASSESS_LIMIT:
        st.caption(f"Source assessments are included for the first {ASSESS_LIMIT:,} candidates only.")

    preview_cols = [
        "candidate_id", "rank", "freq_center_mhz", "elapsed_start_s", "unusualness_score", "review_status", "review_note",
        "assessment", "cadence_result",
    ]
    st.dataframe(frame[preview_cols], hide_index=True, width="stretch", height=min(40 + 36 * len(frame), 360))
    st.caption(f"{len(frame):,} {'candidate' if len(frame) == 1 else 'candidates'}. The CSV has {len(frame.columns)} columns, including evidence, missing evidence and limitations. Empty cells mean the value is not available.")

    d1, d2 = st.columns([1, 3])
    stem = f"{ctx.obs.obs_id}_candidates".replace(" ", "_")
    d1.download_button("Download CSV", export.to_csv_bytes(frame), file_name=f"{stem}.csv", mime="text/csv", type="primary", key="dl_csv", width="stretch")

    st.markdown("**Printable report**")
    st.caption("A report with a picture of each tile, the evidence, your notes and the limitations. Open the HTML in a browser to print or save as PDF, or download the PDF directly.")
    sig = zlib.crc32(("|".join(frame["candidate_id"]) + "|".join(frame["review_status"] + frame["review_note"])).encode())
    if st.session_state.get("report_sig") != sig:
        st.session_state.pop("report_html", None)
        st.session_state.pop("report_pdf", None)
    if st.button("Prepare report", key="prep_report", ):
        shown = frame.head(REPORT_LIMIT)
        thumbs, tiles = {}, {}
        with st.spinner("Building the report…"):
            if ctx.can_read_data:
                for r in rows.sort_values("rank").head(REPORT_LIMIT).itertuples(index=False):
                    tile = normalize_tiles(loaders.window_for(ctx.obs, r.t0, r.f0)[None])[0]
                    tiles[r.candidate_id] = tile
                    thumbs[r.candidate_id] = export.tile_png(tile)
            st.session_state["report_html"] = export.build_html_report(shown, ctx.obs, thumbs).encode("utf-8")
            st.session_state["report_pdf"] = export.build_pdf(shown, ctx.obs, tiles)
            st.session_state["report_sig"] = sig
    if st.session_state.get("report_html"):
        r1, r2, _ = st.columns([1, 1, 2])
        r1.download_button("Download report (HTML)", st.session_state["report_html"], file_name=f"{stem}.html", mime="text/html", key="dl_html", width="stretch")
        r2.download_button("Download report (PDF)", st.session_state["report_pdf"], file_name=f"{stem}.pdf", mime="application/pdf", key="dl_pdf", width="stretch")
        if len(frame) > REPORT_LIMIT:
            st.caption(f"The report includes the first {REPORT_LIMIT} candidates; the CSV has all {len(frame):,}.")
