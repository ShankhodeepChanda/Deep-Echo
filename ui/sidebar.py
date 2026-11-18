"""Sidebar: observation, detection sensitivity, add observation, advanced. Builds the per-run context."""

from __future__ import annotations

from html import escape
from typing import List

import pandas as pd
import streamlit as st

from src.observatory.candidates import (
    AnalysisKind, Cutoff, DEFAULT_LEVEL, LEVELS, apply_sensitivity, sensitivity_cutoff,
)
from src.observatory.observations import Observation
from src.observatory.reviews import ReviewStoreError
from ui import loaders
from ui.components import render_wordmark, sidebar_label, fmt_utc
from ui.context import Ctx
from ui.intake import render_add_observation


def _default_obs_id(observations: List[Observation]) -> str:
    for o in observations:
        if o.meta is not None and o.results is not None:
            return o.obs_id
    for o in observations:
        if o.meta is not None:
            return o.obs_id
    return observations[0].obs_id


def _option_label(o: Observation) -> str:
    bits = [o.label]
    if o.meta is not None and o.meta.band:
        bits.append(o.meta.band)
    elif o.meta is None:
        bits.append("data file missing")
    if o.excerpt is not None:
        bits.append("excerpt")
    return "  ·  ".join(bits)


def render_sidebar(observations: List[Observation]) -> Ctx:
    ss = st.session_state
    bundle, model_error = loaders.model_bundle()
    store = loaders.review_store()

    with st.sidebar:
        render_wordmark()

        # ---- observation
        sidebar_label("Observation")
        if "_goto_obs" in ss:
            ss["obs_id"] = ss.pop("_goto_obs")
        obs = None
        if observations:
            ids = [o.obs_id for o in observations]
            if ss.get("obs_id") not in ids:
                ss["obs_id"] = _default_obs_id(observations)
            by_id = {o.obs_id: o for o in observations}
            st.selectbox(
                "Observation", ids, key="obs_id", format_func=lambda i: _option_label(by_id[i]),
                label_visibility="collapsed",
            )
            obs = by_id[ss["obs_id"]]
            if obs.meta is not None:
                m = obs.meta
                tele = escape(str(m.telescope or (obs.results.summary.get("telescope") if obs.results else None) or "telescope not in file"))
                st.markdown(
                    f'<div class="sb-meta"><b>{tele}</b><br>{escape(m.band or "band unknown")} · {escape(fmt_utc(m.start_utc))}</div>',
                    unsafe_allow_html=True,
                )
                if obs.excerpt is not None:
                    st.markdown(
                        f'<div class="sb-note">Demo excerpt of the full recording: {m.duration_s:,.1f} s, '
                        f'{m.freq_low_mhz:,.2f}–{m.freq_high_mhz:,.2f} MHz.</div>',
                        unsafe_allow_html=True,
                    )
            elif obs.results is not None:
                st.markdown(
                    f'<div class="sb-meta"><b>{escape(str(obs.results.summary.get("telescope", "telescope unknown")))}</b><br>'
                    f'{escape(str(obs.results.summary.get("band", "")))}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="sb-note">No observations found yet. Add one below.</div>', unsafe_allow_html=True)

        # ---- analysis for this observation
        cset = loaders.candidate_set(obs) if obs is not None else None
        source = loaders.source_for(obs)

        # ---- sensitivity
        sidebar_label("Detection sensitivity")
        picked = st.segmented_control(
            "Detection sensitivity", list(LEVELS), default=DEFAULT_LEVEL, key="sensitivity",
            label_visibility="collapsed", selection_mode="single", width="stretch",
            help="Chooses how many unusual signals are shown for review. Low shows about half as many as Medium (the strongest); High about twice as many. Medium is the validated threshold when one exists, otherwise the top 1% of the observation's own scores.",
        )
        level = picked or DEFAULT_LEVEL

        cutoff: Cutoff | None = None
        view = pd.DataFrame()
        if cset is not None and cset.kind in (AnalysisKind.READY, AnalysisKind.SOURCE_MISSING) and not cset.frame.empty:
            cutoff = sensitivity_cutoff(level, cset.frame["score"], obs.calibrated_threshold)
            override = float(ss.get("cutoff_override", 0.0) or 0.0)
            if override > 0:
                cutoff = Cutoff(override, level, "custom", "set manually under Advanced")
            view = apply_sensitivity(cset.frame, cutoff)

        if cutoff is not None:
            n = len(view)
            noun = "unusual signal" if n == 1 else "unusual signals"
            st.markdown(f'<div class="sb-meta"><b>{n:,} {noun}</b></div>', unsafe_allow_html=True)
        elif cset is not None and cset.kind == AnalysisKind.NOT_ANALYZED:
            st.markdown('<div class="sb-note">Sensitivity applies once this observation has been analysed.</div>', unsafe_allow_html=True)

        # ---- reviews / cadence for this observation
        reviews = {}
        if obs is not None:
            try:
                reviews = store.for_observation(obs.obs_id)
            except ReviewStoreError as exc:
                st.error(str(exc))
        cadence = {}
        if obs is not None:
            cadence = {k: v for k, v in loaders.cadence_log().read().items() if k.startswith(obs.obs_id + ":")}

        # ---- add observation
        st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)
        with st.expander("Add observation"):
            render_add_observation()

        # ---- advanced
        with st.expander("Advanced / About the model"):
            st.markdown(
                '<div class="sb-note">The detector is a convolutional autoencoder trained on 64×64 tiles of typical '
                'observations. It tries to rebuild each tile; tiles it rebuilds poorly get a high <b>unusualness score</b>. '
                'It does not say what a signal is.</div>',
                unsafe_allow_html=True,
            )
            if bundle is not None:
                st.caption(f"Checkpoint: {bundle.path.name} · {bundle.signature}")
            else:
                st.error(model_error or "Model unavailable.")
            st.number_input(
                "Manual cutoff (0 = use sensitivity)", min_value=0.0, step=0.0005, format="%.6f", key="cutoff_override",
                help="Overrides the sensitivity setting with an exact unusualness-score cutoff. For checking results against a known value.",
            )
            st.toggle("Show developer tools", key="dev_tools", help="Model diagnostics, thresholds and the synthetic training lab appear at the bottom of the page.")

    return Ctx(
        observations=observations, obs=obs, bundle=bundle, model_error=model_error, cset=cset, level=level, cutoff=cutoff,
        view=view, reviews=reviews, store=store, cadence=cadence, source=source,
        selected_id=ss.get("selected_candidate"),
    )
