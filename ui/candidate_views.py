"""Shared candidate rendering: ranked table, assessment labels, review controls and the detail view."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.observatory.assessment import ARTEFACT, INSUFFICIENT, RFI, UNUSUAL, Assessment
from src.observatory.reviews import DISMISSED, FLAGGED, PENDING, RETAINED, ReviewStoreError
from src.observatory.scoring import reconstruct
from ui import loaders
from ui.components import (
    badge, callout, evidence_list, fmt_freq, fmt_khz, fmt_range, fmt_score, fmt_seconds, fmt_utc, kv_block, plot_caption,
    REVIEW_LABELS, review_pill,
)
from ui.context import Ctx
from ui.plots import tile_figure, window_axes

SHORT_CATEGORY = {RFI: "Likely RFI", ARTEFACT: "Possible artefact", UNUSUAL: "Unusual — review", INSUFFICIENT: "Insufficient info"}
PLOT_CONFIG = {"displayModeBar": False}
CADENCE_SHORT = {"CANDIDATE_TECHNOSIGNATURE": "ON only", "LOCAL_RFI": "Also OFF", "NORMAL_BACKGROUND": "Not unusual"}


def goto(page: str, candidate_id: Optional[str] = None) -> None:
    """Callback: switch page (and optionally candidate) at the start of the next run."""
    st.session_state["_goto"] = page
    if candidate_id:
        st.session_state["selected_candidate"] = candidate_id


def select(candidate_id: str) -> None:
    st.session_state["selected_candidate"] = candidate_id


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
def assess(ctx: Ctx, row) -> Optional[Assessment]:
    if ctx.obs is None or ctx.cutoff is None:
        return None
    return loaders.assessment_for(ctx.obs, row, ctx.cutoff.value, ctx.verdict_of(row.candidate_id))


def table_frame(ctx: Ctx, rows: pd.DataFrame, thumbs: bool = True, compact: bool = False) -> pd.DataFrame:
    """Display frame for a set of candidate rows (real values only; unavailable ones stay empty).

    ``compact`` drops the columns that the detail view repeats (elapsed time, score multiple) for narrow lists.
    """
    out: List[dict] = []
    for r in rows.itertuples(index=False):
        a = assess(ctx, r) if ctx.can_read_data or not np.isnan(r.f_low_mhz) else None
        item = {
            "Tile": loaders.thumb_uri(ctx.obs, r.t0, r.f0) if (thumbs and ctx.can_read_data) else None,
            "Rank": int(r.rank),
            "Frequency (MHz)": None if np.isnan(r.f_center_mhz) else float(r.f_center_mhz),
        }
        if not compact:
            item["Elapsed (s)"] = None if np.isnan(r.t_start_s) else float(r.t_start_s)
        item["Unusualness"] = float(r.score)
        if not compact:
            item["\u00d7 cutoff"] = float(r.score / ctx.cutoff.value) if ctx.cutoff and ctx.cutoff.value > 0 else None
        item["Assessment"] = SHORT_CATEGORY.get(a.category) if a else None
        if not compact:
            item["Cadence"] = CADENCE_SHORT.get(ctx.verdict_of(r.candidate_id), "Not run")
        item["Review"] = REVIEW_LABELS[ctx.status_of(r.candidate_id)]
        out.append(item)
    df = pd.DataFrame(out)
    if not thumbs or not ctx.can_read_data:
        df = df.drop(columns=["Tile"], errors="ignore")
    return df


def column_config(show_tile: bool) -> Dict[str, object]:
    cfg: Dict[str, object] = {
        "Rank": st.column_config.NumberColumn("#", format="%d", width=40, help="Position by unusualness score, highest first."),
        "Frequency (MHz)": st.column_config.NumberColumn("Freq. (MHz)", format="%.3f", width=100, help="Centre of the 64-channel tile."),
        "Elapsed (s)": st.column_config.NumberColumn("Elapsed (s)", format="%.1f", width=92, help="Seconds from the start of the observation to the start of the tile."),
        "Unusualness": st.column_config.NumberColumn("Score", format="%.5f", width=78, help="Unusualness: how poorly the model rebuilt this tile. Higher means more unusual."),
        "\u00d7 cutoff": st.column_config.NumberColumn("\u00d7 cutoff", format="%.2f", width=78, help="Score divided by the current cutoff."),
        "Assessment": st.column_config.TextColumn("Assessment", width=138, help="Preliminary, rule-based source assessment. See the candidate detail for the evidence."),
        "Cadence": st.column_config.TextColumn("ON/OFF cadence", width=100, help="Result of the ON/OFF cadence check, if it has been run."),
        "Review": st.column_config.TextColumn("Review", width=74),
    }
    if show_tile:
        cfg = {"Tile": st.column_config.ImageColumn("Tile", width=56, help="Time runs left to right, frequency bottom to top."), **cfg}
    return cfg


def candidate_table(ctx: Ctx, rows: pd.DataFrame, key: str, selectable: bool = True, height: Optional[int] = None, compact: bool = False) -> Optional[str]:
    """Render ranked candidates. Returns the id of the clicked row (selectable tables only)."""
    df = table_frame(ctx, rows, compact=compact)
    show_tile = "Tile" in df.columns
    kwargs = dict(
        hide_index=True, width="stretch", column_config=column_config(show_tile),
        row_height=54 if show_tile else 36, key=key,
    )
    if height:
        kwargs["height"] = height
    if not selectable:
        st.dataframe(df, **kwargs)
        return None
    event = st.dataframe(df, on_select="rerun", selection_mode="single-row", **kwargs)
    picked = event.selection.rows if event and event.selection else []
    if picked:
        return rows.iloc[picked[0]]["candidate_id"]
    return None


# ---------------------------------------------------------------------------
# Review controls
# ---------------------------------------------------------------------------
def _save_review(store, candidate_id: str, obs_id: str, status: str) -> None:
    ss = st.session_state
    try:
        store.set(candidate_id, obs_id, status, ss.get(f"note_{candidate_id}", ""))
        ss["review_error"] = None
        ss["review_saved"] = candidate_id
    except ReviewStoreError as exc:
        ss["review_error"] = str(exc)


def review_panel(ctx: Ctx, candidate_id: str) -> None:
    current = ctx.reviews.get(candidate_id)
    status = current.status if current else PENDING
    st.markdown(f"**Your review** &nbsp; {review_pill(status)}", unsafe_allow_html=True)
    if st.session_state.get("review_error"):
        st.error(st.session_state["review_error"] + " Your change was not saved.")
    c1, c2, c3, c4 = st.columns(4)
    args = (ctx.store, candidate_id, ctx.obs.obs_id)
    c1.button("Retain", key=f"retain_{candidate_id}", on_click=_save_review, args=(*args, RETAINED), width="stretch",
              type="primary" if status == RETAINED else "secondary", help="Keep for further investigation.")
    c2.button("Flag", key=f"flag_{candidate_id}", on_click=_save_review, args=(*args, FLAGGED), width="stretch",
              type="primary" if status == FLAGGED else "secondary", help="Mark for a second look or another person's opinion.")
    c3.button("Dismiss", key=f"dismiss_{candidate_id}", on_click=_save_review, args=(*args, DISMISSED), width="stretch",
              type="primary" if status == DISMISSED else "secondary", help="Not worth following up.")
    c4.button("Reset", key=f"reset_{candidate_id}", on_click=_save_review, args=(*args, PENDING), width="stretch",
              disabled=status == PENDING, help="Back to pending. Any note is kept.")
    st.text_area(
        "Notes", value=current.note if current else "", key=f"note_{candidate_id}", height=90,
        placeholder="What did you see? What would you check next?",
    )
    st.button("Save note", key=f"savenote_{candidate_id}", on_click=_save_review, args=(*args, status), width="stretch")
    if current:
        st.caption(f"Last saved {current.updated_utc.replace('T', ' ').replace('+00:00', ' UTC')}")


# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------
def _oriented(arr: np.ndarray, flipped: bool) -> np.ndarray:
    """(time, chan) -> (freq ascending, time)."""
    return arr.T[::-1] if flipped else arr.T


def tile_panel(ctx: Ctx, row, key: str, height: int = 300) -> None:
    """Observed tile / what the model expected / difference, on physical axes."""
    meta = ctx.obs.meta
    raw = loaders.window_for(ctx.obs, row.t0, row.f0)
    t_axis, f_axis, flipped = window_axes(meta, int(row.t0), int(row.f0), raw.shape[0], raw.shape[1])
    views = ["Observed"]
    if ctx.bundle is not None:
        views += ["Model expected", "Difference"]
    choice = st.segmented_control("Tile view", views, default="Observed", key=f"{key}_view", label_visibility="collapsed") or "Observed"
    if ctx.bundle is None:
        tile, recon, score = None, None, None
        from src.observatory.scoring import normalize_tiles
        tile = normalize_tiles(raw[None])[0]
    else:
        tile, recon, score = reconstruct(ctx.bundle, raw)
    if choice == "Observed":
        arr, title, z = tile, "Observed (scaled 0–1 within the tile)", (0.0, 1.0)
    elif choice == "Model expected":
        arr, title, z = recon, "What the model expected to see", (0.0, 1.0)
    else:
        arr, title, z = np.abs(tile - recon), "Where the model was wrong (absolute difference)", None
    fig = tile_figure(_oriented(arr, flipped), t_axis, f_axis, title, height=height, zrange=z, value_label="level")
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, key=f"{key}_fig_{choice}")
    plot_caption(
        f"Time runs left to right ({fmt_seconds(meta.tsamp_s)} per sample), frequency bottom to top ({fmt_khz(meta.foff_mhz)} per channel). "
        "Brighter means stronger."
    )


def percentile_text(ctx: Ctx, score: float) -> str:
    scores = ctx.cset.frame["score"].to_numpy()
    n = len(scores)
    if score >= scores.max():
        return f"the highest of the {n:,} tiles scored"
    below = float(np.searchsorted(np.sort(scores), score)) / max(n, 1) * 100
    return f"higher than {below:.1f}% of the {n:,} tiles scored"


def render_detail(ctx: Ctx, row, key: str = "detail") -> None:
    """Full investigation view for one candidate, built from its real data."""
    obs = ctx.obs
    a = assess(ctx, row)
    st.markdown(
        f'<div class="detail-head"><h2 class="detail-title">Candidate #{int(row.rank)}</h2>'
        f'<div>{badge(a.category) if a else ""}</div></div>',
        unsafe_allow_html=True,
    )
    if ctx.can_read_data and not np.isnan(row.t0):
        tile_panel(ctx, row, key)
    elif not ctx.has_coordinates:
        callout("warn", "Tile data unavailable", "<p>The tile's position in the observation is not known, so it cannot be drawn.</p>")

    ratio = row.score / ctx.cutoff.value if ctx.cutoff and ctx.cutoff.value > 0 else float("nan")
    st.markdown("**Why it was flagged**")
    st.markdown(
        f"- Unusualness score **{fmt_score(row.score)}** is {ratio:.2f}× the cutoff ({fmt_score(ctx.cutoff.value)}) and "
        f"{percentile_text(ctx, row.score)}.\n"
        "- The score measures how poorly the model rebuilt this tile. It says the tile looks different from typical data, not what caused it."
    )

    meta = obs.meta
    rows = [
        ("Observation", obs.label, False),
        ("Frequency", f"{fmt_range(row.f_low_mhz, row.f_high_mhz, 4)}  (centre {fmt_freq(row.f_center_mhz, 4)})", True),
        ("Elapsed time", "n/a" if np.isnan(row.t_start_s) else f"{row.t_start_s:,.3f}–{row.t_end_s:,.3f} s", True),
    ]
    if meta is not None and meta.start_utc is not None and not np.isnan(row.t_start_s):
        from datetime import timedelta
        rows.append(("Starts at", fmt_utc(meta.start_utc + timedelta(seconds=float(row.t_start_s))), False))
    if meta is not None:
        rows.append(("Resolution", f"{fmt_khz(meta.foff_mhz)} per channel, {fmt_seconds(meta.tsamp_s)} per sample", False))
    rows.append(("Analysis", f"{ctx.cset.basis}; tile {row.tile_ref}", False))
    kv_block(rows)

    if a is not None:
        st.markdown(f"**Source assessment** {badge(a.category)}", unsafe_allow_html=True)
        st.markdown(f'<div class="c-body" style="max-width:70ch;margin:0.1rem 0 0.3rem">{_esc(a.why)}</div>', unsafe_allow_html=True)
        evidence_list(a.evidence, a.missing)
        with st.expander("About this assessment"):
            st.markdown(
                "These are simple, transparent rules applied to what can be measured in the tile and its frequency. "
                "They are **preliminary**: not a validated classifier, with no confidence values. They are separate from the "
                "unusualness score, which comes from the model. A signal that survives these checks is a reason to look further, "
                "never evidence of extraterrestrial origin."
            )
            from src.observatory.assessment import REFERENCE_NOTE
            st.caption(REFERENCE_NOTE)

    st.divider()
    review_panel(ctx, row.candidate_id)
    st.divider()
    b1, b2, b3 = st.columns(3)
    b1.button("Waterfall", key=f"{key}_wf", on_click=goto, args=("Waterfall", row.candidate_id), width="stretch")
    b2.button("Cadence", key=f"{key}_cad", on_click=goto, args=("Cadence Analysis", row.candidate_id), width="stretch")
    b3.button("Compare", key=f"{key}_cmp", on_click=_to_compare, args=(row.candidate_id,), width="stretch")


def _to_compare(candidate_id: str) -> None:
    st.session_state["cmp_a"] = candidate_id
    goto("Compare", candidate_id)


def _esc(text: str) -> str:
    from html import escape
    return escape(str(text))
