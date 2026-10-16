"""
UI components and theme helpers for the DEEP ECHO dashboard.

Presentation only: no backend, data or modelling logic lives here.
Colours mirror the CSS variables in styles.css, so change them in both places.
"""

from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Theme constants (names unchanged, values aligned with styles.css)
# ---------------------------------------------------------------------------
COLOR_ACCENT = "#4cc9e0"    # primary accent
COLOR_NORMAL = "#4fbf9f"    # quiet background / normal
COLOR_RFI = "#e0a458"       # interference
COLOR_ANOMALY = "#e5675f"   # anomaly / candidate
COLOR_MUTED = "#9aa7bb"     # secondary text and axes
COLOR_CARD = "#121927"      # surface
COLOR_BORDER = "rgba(255, 255, 255, 0.08)"

COLOR_TEXT = "#d8dfeb"
COLOR_TEXT_STRONG = "#f3f6fb"
COLOR_GRID = "rgba(255, 255, 255, 0.06)"

FONT_UI = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, Consolas, monospace"

# Default colour sequence for multi-series charts.
COLORWAY = [COLOR_ACCENT, COLOR_NORMAL, COLOR_RFI, COLOR_ANOMALY, "#8b9dc3", "#b48ead"]

# Spectrogram / heatmap scale: dark navy to accent, no rainbow.
HEATMAP_COLORSCALE = [
    [0.0, "#0b1018"],
    [0.35, "#16324a"],
    [0.7, "#2f8ca3"],
    [1.0, "#d6f4fa"],
]

_STYLESHEET_NAMES = ("styles.css", "style.css")


# ---------------------------------------------------------------------------
# Stylesheet and page chrome
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _read_css() -> str:
    """Read the stylesheet once per process (the file is small and static)."""
    base = Path(__file__).parent
    for name in _STYLESHEET_NAMES:
        path = base / name
        if path.exists():
            return path.read_text(encoding="utf-8")
    return ""


def load_css() -> None:
    """Inject the custom stylesheet that sits next to this file."""
    css = _read_css()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_header(status_text: Optional[str] = None) -> None:
    """Render the DEEP ECHO wordmark. A status label is shown only if given."""
    status = (
        f'<div class="de-brand-status">{escape(status_text)}</div>'
        if status_text
        else ""
    )
    st.markdown(
        f"""
        <div class="de-brand-container">
            <div class="de-brand-title">DEEP <span>ECHO</span></div>
            {status}
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_section(title: str) -> None:
    """Small section label for the sidebar (uses .sidebar-section-title)."""
    st.markdown(
        f'<div class="sidebar-section-title">{escape(title)}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Plotly styling
# ---------------------------------------------------------------------------
def style_plotly_figure(
    fig: go.Figure,
    height: int = 300,
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
) -> go.Figure:
    """Apply the shared dark theme to a Plotly figure and return it."""
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_UI, size=12, color=COLOR_MUTED),
        colorway=COLORWAY,
        height=height,
        margin=dict(l=55, r=20, t=44 if title else 20, b=45),
        hoverlabel=dict(
            bgcolor="#18212f",
            bordercolor="#232e42",
            font=dict(family=FONT_UI, size=12, color=COLOR_TEXT_STRONG),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=COLOR_TEXT),
        ),
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color="rgba(154, 167, 187, 0.6)",
            activecolor=COLOR_ACCENT,
        ),
    )
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(family=FONT_UI, size=14, color=COLOR_TEXT_STRONG),
            x=0.0,
            xanchor="left",
        )
    if xaxis_title:
        layout["xaxis_title"] = xaxis_title
    if yaxis_title:
        layout["yaxis_title"] = yaxis_title

    fig.update_layout(**layout)

    axis_style = dict(
        gridcolor=COLOR_GRID,
        zerolinecolor="rgba(255, 255, 255, 0.10)",
        linecolor="rgba(255, 255, 255, 0.12)",
        tickfont=dict(family=FONT_MONO, size=10, color=COLOR_MUTED),
        title_font=dict(family=FONT_UI, size=12, color=COLOR_MUTED),
        automargin=True,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig
