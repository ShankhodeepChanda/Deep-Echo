"""First-run introduction and glossary."""

from __future__ import annotations

import streamlit as st

from ui import loaders
from src.observatory.reviews import ReviewStoreError

GLOSSARY = [
    (
        "What is radio-frequency interference (RFI)?",
        "Human-made radio emission picked up by the telescope: satellites, radar, phones, vehicles, even the instrument itself. "
        "It is by far the most common reason a signal looks unusual, which is why candidates are checked against known interference "
        "and against OFF-target observations before anything else.",
    ),
    (
        "What does the unusualness score mean?",
        "The detector learns what typical 64×64 pieces of an observation look like and tries to redraw each piece. The score is how "
        "badly it redraws a piece (mean squared error). A higher score means the piece looks less like typical data. It does not say what "
        "the signal is, and scores can only be compared within the same observation.",
    ),
    (
        "What is a waterfall plot?",
        "A picture of radio power over time and frequency. Here time runs left to right and frequency bottom to top; brighter means "
        "stronger. A steady carrier is a horizontal line, and a signal whose frequency changes (for example from relative motion) "
        "is a slanted line.",
    ),
    (
        "What do ON and OFF observations mean?",
        "The telescope alternates between pointing at the target (ON) and at nearby patches of sky (OFF). A signal from the target should "
        "appear in ON only. A signal that shows up in both is almost certainly local. This is the ABACAD cadence: A is ON; B, C and D are OFF.",
    ),
    (
        "Why is a high score not proof of extraterrestrial origin?",
        "Because the model only measures how different a tile looks. Interference, instrument effects and ordinary astrophysical sources "
        "can all look different. A candidate is a reason to look closer, check the cadence, and repeat the observation. It is never a conclusion.",
    ),
]


def glossary_popover() -> None:
    with st.popover("Glossary", help="Plain-language definitions of the terms used here."):
        for question, answer in GLOSSARY:
            with st.expander(question):
                st.write(answer)


def first_run_guide() -> None:
    prefs = loaders.preferences()
    if prefs.read().get("guide_dismissed") or st.session_state.get("guide_hidden"):
        return
    with st.container(border=True):
        st.markdown("**How to use this workspace**")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**1. Look at the strip.** Each amber mark is a tile the model found unusual, placed at its measured time and frequency.")
        c2.markdown("**2. Open a candidate.** See what the model expected, why it was flagged, and what is still unknown about it.")
        c3.markdown("**3. Decide and record.** Retain, flag or dismiss, add a note, then export your review. A high score is not proof of anything on its own.")
        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button("Got it", key="guide_ok", type="primary"):
            st.session_state["guide_hidden"] = True
            try:
                prefs.set("guide_dismissed", True)
            except ReviewStoreError:
                pass  # hidden for this session even if the preference cannot be saved
            st.rerun()
        if b2.button("Hide for now", key="guide_hide"):
            st.session_state["guide_hidden"] = True
            st.rerun()
