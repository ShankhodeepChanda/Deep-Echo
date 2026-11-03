"""Cadence analysis: does a candidate also appear in OFF-target observations?"""

from __future__ import annotations

from html import escape
from typing import List

import pandas as pd
import streamlit as st

from src.observatory import cadence as cad
from src.observatory.observations import Observation
from src.observatory.provenance import TILE_F, TILE_T
from src.observatory.reviews import ReviewStoreError
from src.observatory.scoring import normalize_tiles
from ui import loaders
from ui.candidate_views import PLOT_CONFIG
from ui.components import callout, fmt_khz, fmt_range, fmt_score, fmt_seconds, page_head
from ui.context import Ctx, state_panel, status_chip
from ui.guide import glossary_popover
from ui.plots import tile_figure, window_axes

VERDICT_TEXT = {
    "CANDIDATE_TECHNOSIGNATURE": ("info", "Seen in the ON pointing only", "Worth further investigation. This is not evidence of origin."),
    "LOCAL_RFI": ("warn", "Also seen away from the target", "Consistent with local interference."),
    "NORMAL_BACKGROUND": ("", "Not unusual in the ON pointing", "The ON tile does not exceed the threshold, so there is nothing to compare."),
}


def _intro() -> None:
    with st.container(border=True):
        st.markdown("**How ON/OFF observing works**")
        st.markdown(
            '<div class="cadence-seq"><div class="step on">A<small>ON</small></div><div class="step">B<small>OFF</small></div>'
            '<div class="step on">A<small>ON</small></div><div class="step">C<small>OFF</small></div>'
            '<div class="step on">A<small>ON</small></div><div class="step">D<small>OFF</small></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "The telescope points at the target (**ON**, A) and alternates with nearby patches of sky (**OFF**, B, C, D). "
            "A signal that comes from the target should appear in ON and not in OFF. A signal that appears in both is almost "
            "certainly local interference. A signal found only in ON is a reason to investigate further. It is not proof of "
            "extraterrestrial origin."
        )


def _candidate_options(ctx: Ctx) -> List[str]:
    return ctx.view.head(200)["candidate_id"].tolist()


def _compat_table(ctx: Ctx, row, others: List[Observation]):
    rows, usable = [], []
    for o in others:
        ok, reason, f0 = cad.compatibility(ctx.obs.meta, o.meta, int(row.f0))
        rows.append({"Observation": o.label, "Can be compared": "Yes" if ok else "No",
                     "Why not": reason, "Duration": fmt_seconds(o.meta.duration_s), "Band": fmt_range(o.meta.freq_low_mhz, o.meta.freq_high_mhz, 1)})
        if ok:
            usable.append(o)
    return pd.DataFrame(rows), usable


def _pointing_thumbs(ctx: Ctx, pointings: List[dict]) -> None:
    cols = st.columns(min(len(pointings), 4) or 1)
    by_id = {o.obs_id: o for o in ctx.observations}
    for col, p in zip(cols, pointings):
        o = by_id.get(p["obs_id"])
        with col:
            if o is None or o.meta is None:
                st.caption(f"{p['label']}: data file not available")
                continue
            raw = loaders.window_for(o, p["t0"], p["f0"])
            t, f, flipped = window_axes(o.meta, p["t0"], p["f0"], TILE_T, TILE_F)
            arr = normalize_tiles(raw[None])[0]
            arr = arr.T[::-1] if flipped else arr.T
            title = f"{p['type']}: {p['label']}"
            st.plotly_chart(tile_figure(arr, t, f, title if len(title) < 34 else title[:31] + "…", height=230, show_axes_titles=False),
                            width="stretch", config=PLOT_CONFIG, key=f"cad_t_{p['obs_id']}_{p['type']}_{p['t0']}_{p['f0']}")
            st.caption(f"Score {fmt_score(p['score'])} · {'above' if p['flagged'] else 'below'} threshold")


def _show_result(ctx: Ctx, entry: dict) -> None:
    kind, title, sub = VERDICT_TEXT.get(entry["verdict"], ("", entry["verdict"], ""))
    reasons = "".join(f"<li>{escape(r)}</li>" for r in entry.get("reasons", []))
    callout(kind, title, f"<p>{escape(sub)}</p><ul>{reasons}</ul>")
    pts = entry.get("pointings", [])
    if pts:
        st.dataframe(
            pd.DataFrame([{
                "Pointing": p["label"], "Type": p["type"], "Unusualness": p["score"],
                "Above threshold": "Yes" if p["flagged"] else "No",
            } for p in pts]),
            hide_index=True, width="stretch",
            column_config={"Unusualness": st.column_config.NumberColumn(format="%.5f")},
        )
        _pointing_thumbs(ctx, pts)
    st.caption(
        f"ON/OFF contrast {entry.get('contrast', float('nan')):.1f}× (needs ≥ {cad.CONTRAST_THRESHOLD:g}×). "
        f"Threshold {fmt_score(entry.get('threshold'))}. Run {entry.get('run_utc', '').replace('T', ' ').replace('+00:00', ' UTC')}."
    )
    for obs_label, reason in entry.get("skipped", []):
        st.caption(f"Not compared: {obs_label}. {reason}")


def render(ctx: Ctx) -> None:
    h, g = st.columns([5, 1.3])
    with h:
        page_head("Cadence analysis", ctx.obs.label if ctx.obs else "", status_chip(ctx))
    with g:
        glossary_popover()
    _intro()
    if not state_panel(ctx, need_data=True, what="cadence analysis"):
        return
    if ctx.view.empty:
        st.info("No candidates at this sensitivity, so there is nothing to check. Try High sensitivity.")
        return
    if ctx.bundle is None:
        callout("bad", "The model is unavailable", f"<p>{escape(ctx.model_error or '')}</p>")
        return

    ids = _candidate_options(ctx)
    default = ctx.selected_id if ctx.selected_id in ids else ids[0]
    cid = st.selectbox(
        "Candidate to check", ids, index=ids.index(default), key="cad_candidate",
        format_func=lambda c: f"#{int(ctx.row(c).rank)}  ·  {ctx.row(c).f_center_mhz:,.3f} MHz  ·  score {ctx.row(c).score:.5f}",
    )
    st.session_state["selected_candidate"] = cid
    row = ctx.row(cid)

    others = [o for o in ctx.observations if o.obs_id != ctx.obs.obs_id and o.meta is not None]
    table, usable = _compat_table(ctx, row, others)
    needed = (
        f"covers {fmt_range(row.f_low_mhz, row.f_high_mhz, 3)}, has {fmt_khz(ctx.obs.meta.foff_mhz)} channels and "
        f"{fmt_seconds(ctx.obs.meta.tsamp_s)} time samples, matching this observation"
    )

    saved = ctx.cadence.get(cid)
    if saved:
        st.markdown("**Result**")
        _show_result(ctx, saved)

    if not usable:
        callout(
            "warn", "No OFF-target observation is available for this candidate",
            "<p>Cadence analysis compares the same frequency range in recordings made while pointing away from the target. "
            f"An OFF observation must be one that {escape(needed)}.</p>"
            "<p>Add the OFF recordings with <b>Add observation</b> in the sidebar, then return here. Nothing is estimated or simulated in their place.</p>",
        )
        if not table.empty:
            with st.expander(f"Observations checked ({len(table)})"):
                st.dataframe(table, hide_index=True, width="stretch")
        return

    st.markdown("**Choose OFF-target observations**")
    st.caption("These cover the candidate's frequency range at the same resolution.")
    by_label = {o.label: o for o in usable}
    chosen = st.multiselect("OFF observations", list(by_label), default=list(by_label), key="cad_off", label_visibility="collapsed")
    with st.expander(f"All other observations ({len(table)})"):
        st.dataframe(table, hide_index=True, width="stretch")

    if st.button("Run cadence analysis", type="primary", disabled=not chosen, key="cad_run",
                 help="Finds the most unusual tile in the same frequency range of each OFF recording and compares it with this candidate."):
        bar = st.progress(0.0, text="Comparing with OFF observations…")
        try:
            offs = [(l, loaders.source_for(by_label[l])) for l in chosen]
            out = cad.run_cadence(
                ctx.obs.label, loaders.source_for(ctx.obs), int(row.t0), int(row.f0), offs, ctx.bundle, ctx.cutoff.value,
                progress=lambda p: bar.progress(min(p, 1.0), text=f"Comparing with OFF observations… {p * 100:.0f}%"),
            )
        except Exception as exc:
            bar.empty()
            st.error(f"The cadence analysis could not run: {exc}")
            return
        bar.empty()
        label_to_obs = {o.label: o.obs_id for o in ctx.observations}
        entry = {
            "verdict": out.verdict, "contrast": out.contrast_ratio, "threshold": out.threshold, "reasons": out.reasons,
            "skipped": out.skipped,
            "pointings": [{"obs_id": label_to_obs.get(p.label, ctx.obs.obs_id), "label": p.label, "type": p.pointing_type,
                           "t0": p.t0, "f0": p.f0, "score": p.score, "flagged": p.flagged} for p in out.pointings],
        }
        try:
            loaders.cadence_log().set(cid, entry)
        except ReviewStoreError as exc:
            st.error(f"{exc} The result is shown below but was not saved.")
            _show_result(ctx, {**entry, "run_utc": ""})
            return
        st.rerun()
