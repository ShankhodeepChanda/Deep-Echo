"""
SETI-AI Observatory: signal investigation dashboard.

AI search for unknown technosignatures using deep anomaly detection. This file
only wires the pages together; the data layer is ``src/observatory`` and the
interface lives in ``ui/``.

Run:  streamlit run app.py
"""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="SETI-AI Observatory",
    page_icon=str(Path(__file__).parent / "ui" / "assets" / "favicon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.observatory.observations import discover_observations  # noqa: E402
from ui import (  # noqa: E402
    pages_candidates, pages_cadence, pages_compare, pages_export, pages_overview, pages_waterfall,
)
from ui.advanced import render_developer_tools  # noqa: E402
from ui.components import load_css  # noqa: E402
from ui.context import PAGES  # noqa: E402
from ui.sidebar import render_sidebar  # noqa: E402

PAGE_RENDERERS = {
    "Overview": pages_overview.render,
    "Candidates": pages_candidates.render,
    "Waterfall": pages_waterfall.render,
    "Cadence Analysis": pages_cadence.render,
    "Compare": pages_compare.render,
    "Export": pages_export.render,
}


def main() -> None:
    load_css()
    ss = st.session_state
    # A page change requested by a button is applied before the navigation widget is created.
    if "_goto" in ss:
        ss["page"] = ss.pop("_goto")
    if ss.get("page") not in PAGES:
        ss["page"] = "Overview"

    ctx = render_sidebar(discover_observations())

    st.segmented_control("Section", PAGES, key="page", label_visibility="collapsed", selection_mode="single")
    st.markdown('<div class="nav-rule"></div>', unsafe_allow_html=True)
    page = ss.get("page") or "Overview"
    PAGE_RENDERERS[page](ctx)

    if ss.get("dev_tools"):
        render_developer_tools()
    st.markdown(
        '<div class="foot">An unusual signal is a reason to look closer, not a conclusion. The unusualness score shows how different a tile looks '
        "from typical data; it does not identify what produced it.</div>",
        unsafe_allow_html=True,
    )


main()
