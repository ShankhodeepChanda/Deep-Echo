"""
UI components and theme helpers for the SETI-AI Observatory dashboard.

Presentation only: no backend, data or modelling logic lives here. Colours mirror
the CSS variables in styles.css and the theme in .streamlit/config.toml.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
COLOR_ACCENT = "#35b8a8"    # selection, primary action, signal intensity
COLOR_ACCENT_HI = "#63d3c4"
COLOR_MARK = "#e8a33d"      # unusual-signal marks and flags only
COLOR_NORMAL = "#35b8a8"
COLOR_RFI = "#e5675f"
COLOR_ANOMALY = "#e8a33d"
COLOR_MUTED = "#9aa4ac"
COLOR_CARD = "#15181b"
COLOR_BORDER = "#2a3035"
COLOR_TEXT = "#d9dee2"
COLOR_TEXT_STRONG = "#f2f5f7"
COLOR_GRID = "rgba(255, 255, 255, 0.06)"
COLOR_PANEL = "#15181b"

FONT_UI = "'Atkinson Hyperlegible Next', 'Segoe UI', system-ui, sans-serif"
FONT_NUM = "'JetBrains Mono', ui-monospace, Consolas, monospace"

COLORWAY = [COLOR_ACCENT, COLOR_MARK, "#8fa3b0", "#c97b84", "#7fb069", "#b48ead"]

# Dark-to-teal intensity ramp (single hue family; no rainbow).
HEATMAP_COLORSCALE = [
    [0.0, "#0e1214"],
    [0.3, "#12353a"],
    [0.62, "#1f7a78"],
    [0.85, "#6fd3c4"],
    [1.0, "#eafaf6"],
]

REVIEW_LABELS = {"pending": "Pending", "retained": "Retained", "flagged": "Flagged", "dismissed": "Dismissed"}

_STYLESHEET_NAMES = ("styles.css", "style.css")


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _read_css(path: str, mtime: float) -> str:
    """Stylesheet text; the mtime argument refreshes the cache when the file changes."""
    return Path(path).read_text(encoding="utf-8")


def load_css() -> None:
    """Inject the stylesheet that sits next to this file."""
    base = Path(__file__).parent
    for name in _STYLESHEET_NAMES:
        path = base / name
        if path.exists():
            st.markdown(f"<style>{_read_css(str(path), path.stat().st_mtime)}</style>", unsafe_allow_html=True)
            return


# ---------------------------------------------------------------------------
# Formatters (None / NaN always render as "not available", never as a number)
# ---------------------------------------------------------------------------
def _ok(x) -> bool:
    try:
        return x is not None and x == x and abs(float(x)) != float("inf")
    except (TypeError, ValueError):
        return False


def fmt_freq(mhz, digits: int = 3) -> str:
    if not _ok(mhz):
        return "n/a"
    mhz = float(mhz)
    return f"{mhz / 1000:.{max(digits, 4)}f} GHz" if abs(mhz) >= 10000 else f"{mhz:,.{digits}f} MHz"


def fmt_range(low, high, digits: int = 3) -> str:
    if not (_ok(low) and _ok(high)):
        return "n/a"
    return f"{float(low):,.{digits}f}–{float(high):,.{digits}f} MHz"


def fmt_seconds(s) -> str:
    if not _ok(s):
        return "n/a"
    s = float(s)
    if s >= 3600:
        return f"{s / 3600:.2f} h"
    if s >= 120:
        return f"{s / 60:.1f} min"
    if s >= 1:
        return f"{s:.2f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.2f} ms"
    return f"{s * 1e6:.1f} µs"


def fmt_khz(mhz) -> str:
    if not _ok(mhz):
        return "n/a"
    khz = abs(float(mhz)) * 1000
    return f"{khz / 1000:.3f} MHz" if khz >= 1000 else f"{khz:.4g} kHz"


def fmt_score(x) -> str:
    return f"{float(x):.5f}" if _ok(x) else "n/a"


def fmt_utc(dt: Optional[datetime]) -> str:
    return dt.strftime("%Y-%m-%d %H:%M UTC") if dt else "not in file"


def fmt_bytes(n) -> str:
    if not _ok(n):
        return "n/a"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# HTML building blocks
# ---------------------------------------------------------------------------
_GLYPH = (
    '<svg class="brand-glyph" viewBox="0 0 22 22" aria-hidden="true" fill="none">'
    '<rect x="1.5" y="1.5" width="19" height="19" rx="3" stroke="#35b8a8" stroke-width="1.8"/>'
    '<path d="M5 7h12M5 11h12M5 15h12" stroke="#35b8a8" stroke-width="1.6" stroke-linecap="round" opacity="0.55"/>'
    '<path d="M11 4.8l3.2 3.2L11 11.2 7.8 8z" fill="#e8a33d"/></svg>'
)


def render_wordmark() -> None:
    st.markdown(f'<div class="brand">{_GLYPH}<span class="brand-name">SETI-AI Observatory</span></div>', unsafe_allow_html=True)


def chip(text: str, kind: str = "") -> str:
    return f'<span class="chip {kind}"><i></i>{escape(text)}</span>'


def page_head(title: str, subtitle: str = "", chip_html: str = "") -> None:
    sub = f'<div class="page-sub">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="page-head"><div><h1 class="page-title">{escape(title)}</h1>{sub}</div><div>{chip_html}</div></div>',
        unsafe_allow_html=True,
    )


def facts_grid(items: Sequence[Tuple[str, str, str]]) -> None:
    """Dense definition grid. Each item is (label, value, kind) with kind in {"", "num", "na"}."""
    cells = "".join(
        f'<div class="fact"><div class="fact-label">{escape(label)}</div>'
        f'<div class="fact-value {kind}">{escape(value)}</div></div>'
        for label, value, kind in items
    )
    st.markdown(f'<div class="facts">{cells}</div>', unsafe_allow_html=True)


def callout(kind: str, title: str, body_html: str = "") -> None:
    """kind: "" | "info" | "warn" | "bad". ``body_html`` must already be escaped / trusted."""
    body = f'<div class="c-body">{body_html}</div>' if body_html else ""
    st.markdown(f'<div class="callout {kind}"><div class="c-title">{escape(title)}</div>{body}</div>', unsafe_allow_html=True)


def badge(category: str, preliminary: bool = True) -> str:
    cls = {
        "Likely interference (RFI)": "rfi",
        "Possible instrument artefact": "art",
        "Insufficient information": "insuff",
    }.get(category, "unusual")
    tail = '<span class="badge prelim">preliminary</span>' if preliminary else ""
    return f'<span class="badge {cls}">{escape(category)}</span>{tail}'


def review_pill(status: str) -> str:
    return f'<span class="review-pill {escape(status)}">{escape(REVIEW_LABELS.get(status, status))}</span>'


def kv_block(rows: Iterable[Tuple[str, str, bool]]) -> None:
    """Key/value list. Each row is (label, value, is_numeric)."""
    html = "".join(
        f'<div class="k">{escape(k)}</div><div class="v {"num" if num else ""}">{escape(v)}</div>' for k, v, num in rows
    )
    st.markdown(f'<div class="kv">{html}</div>', unsafe_allow_html=True)


def evidence_list(evidence: Sequence, missing: Sequence[str] = ()) -> None:
    items = "".join(
        f'<li><span class="k {escape(e.kind)}">{escape(e.title)}</span><span class="d">{escape(e.detail)}</span></li>'
        for e in evidence
    )
    miss = ""
    if missing:
        miss = '<div class="missing"><b>Not yet known:</b> ' + " ".join(escape(m) for m in missing) + "</div>"
    st.markdown(f'<ul class="evi">{items}</ul>{miss}', unsafe_allow_html=True)


def sidebar_label(text: str) -> None:
    st.markdown(f'<div class="sb-section">{escape(text)}</div>', unsafe_allow_html=True)


def plot_caption(text: str) -> None:
    st.markdown(f'<div class="plot-caption">{escape(text)}</div>', unsafe_allow_html=True)


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
        margin=dict(l=62, r=18, t=36 if title else 14, b=46),
        hoverlabel=dict(bgcolor="#1b1f23", bordercolor="#2a3035", font=dict(family=FONT_UI, size=12, color=COLOR_TEXT_STRONG)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=COLOR_TEXT)),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color="rgba(154, 164, 172, 0.7)", activecolor=COLOR_ACCENT_HI),
    )
    if title:
        layout["title"] = dict(text=title, font=dict(family=FONT_UI, size=14, color=COLOR_TEXT_STRONG), x=0.0, xanchor="left")
    if xaxis_title:
        layout["xaxis_title"] = xaxis_title
    if yaxis_title:
        layout["yaxis_title"] = yaxis_title
    fig.update_layout(**layout)
    axis_style = dict(
        gridcolor=COLOR_GRID,
        zerolinecolor="rgba(255, 255, 255, 0.10)",
        linecolor="rgba(255, 255, 255, 0.14)",
        tickfont=dict(family=FONT_NUM, size=10, color=COLOR_MUTED),
        title_font=dict(family=FONT_UI, size=12, color=COLOR_MUTED),
        automargin=True,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig
