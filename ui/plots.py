"""
Plotly figures for the Observatory: observation strip / waterfall, tiles and zoomed windows.

All figures are built from real arrays passed in; nothing is generated here.
Large heatmaps are quantised to uint8 before sending to the browser (about 4x
smaller); hover on those shows time and frequency only, and exact values are
available in the native-resolution views.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.observatory.sources import Overview, SourceMeta
from ui.components import (
    COLOR_ACCENT_HI, COLOR_MARK, COLOR_MUTED, HEATMAP_COLORSCALE, style_plotly_figure,
)

CLIP_LOW, CLIP_HIGH = 1.0, 99.5
MAX_MARKS = 600


def display_levels(values: np.ndarray, clip: Tuple[float, float] = (CLIP_LOW, CLIP_HIGH)) -> Tuple[np.ndarray, str]:
    """Scale intensity for display: dB when all values are positive, else linear. Returns (0-255 uint8, label)."""
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    if finite.any() and (v[finite] > 0).all():
        v = 10.0 * np.log10(np.where(finite, v, np.nan))
        label = "Relative power (dB, uncalibrated)"
    else:
        label = "Relative intensity (uncalibrated)"
    lo, hi = np.nanpercentile(v, clip) if finite.any() else (0.0, 1.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = (float(np.nanmin(v)), float(np.nanmax(v))) if finite.any() else (0.0, 1.0)
        hi = hi if hi > lo else lo + 1.0
    scaled = np.clip((np.nan_to_num(v, nan=lo) - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8), label


def _freq_tickformat(span_mhz: float) -> str:
    """Decimals that keep neighbouring ticks distinct without trailing noise."""
    if span_mhz > 40:
        return ",.0f"
    if span_mhz > 4:
        return ",.1f"
    return ",.3f"


def _colorbar() -> dict:
    return dict(
        thickness=10, len=0.9, tickvals=[0, 255], ticktext=["low", "high"],
        tickfont=dict(size=10, color=COLOR_MUTED), outlinewidth=0, x=1.005, xpad=2,
    )


def _coverage(fig: go.Figure, coverage: Optional[pd.DataFrame]) -> None:
    """Faint dots at every analysed tile, so unmarked areas read as "not analysed", not "clear"."""
    if coverage is None or coverage.empty:
        return
    c = coverage[coverage["f_center_mhz"].notna() & coverage["t_start_s"].notna()]
    fig.add_trace(go.Scattergl(
        x=(c["t_start_s"] + c["t_end_s"]) / 2, y=c["f_center_mhz"], mode="markers", name="Analysed tiles", hoverinfo="skip",
        marker=dict(symbol="circle", size=3, color="rgba(234, 250, 246, 0.35)"),
    ))


def _marks(fig: go.Figure, marks: Optional[pd.DataFrame], selected_id: Optional[str]) -> None:
    if marks is None or marks.empty:
        return
    m = marks.head(MAX_MARKS)
    m = m[m["f_center_mhz"].notna() & m["t_start_s"].notna()]
    if m.empty:
        return
    t_mid = (m["t_start_s"] + m["t_end_s"]) / 2
    text = [
        f"#{int(r)}  {f:,.3f} MHz  t={ts:,.2f} s<br>score {s:.5f}"
        for r, f, ts, s in zip(m["rank"], m["f_center_mhz"], m["t_start_s"], m["score"])
    ]
    fig.add_trace(go.Scatter(
        x=t_mid, y=m["f_center_mhz"], mode="markers", name="Unusual signals", customdata=m["candidate_id"], text=text,
        hovertemplate="%{text}<extra></extra>",
        marker=dict(symbol="diamond-open", size=11, color=COLOR_MARK, line=dict(width=2, color=COLOR_MARK)),
        selected=dict(marker=dict(color=COLOR_ACCENT_HI, opacity=1)), unselected=dict(marker=dict(opacity=1)),
    ))
    if selected_id is not None:
        sel = marks[marks["candidate_id"] == selected_id]
        sel = sel[sel["f_center_mhz"].notna()]
        if not sel.empty:
            r = sel.iloc[0]
            fig.add_trace(go.Scatter(
                x=[(r.t_start_s + r.t_end_s) / 2], y=[r.f_center_mhz], mode="markers", name="Selected", hoverinfo="skip",
                marker=dict(symbol="circle-open", size=24, color=COLOR_ACCENT_HI, line=dict(width=2.5, color=COLOR_ACCENT_HI)),
            ))


def waterfall_figure(
    ov: Overview,
    marks: Optional[pd.DataFrame] = None,
    selected_id: Optional[str] = None,
    height: int = 460,
    compact: bool = False,
    coverage: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Time (x) by frequency (y) overview with candidate marks at their measured coordinates."""
    z, label = display_levels(ov.power.T)
    fig = go.Figure(go.Heatmap(
        z=z, x=ov.time_s, y=ov.freq_mhz, colorscale=HEATMAP_COLORSCALE, zmin=0, zmax=255,
        showscale=not compact, colorbar=_colorbar(), hovertemplate="t %{x:,.2f} s<br>%{y:,.3f} MHz<extra></extra>",
        name="Intensity",
    ))
    _coverage(fig, coverage)
    _marks(fig, marks, selected_id)
    style_plotly_figure(fig, height=height, xaxis_title="Elapsed time (s)", yaxis_title="Frequency (MHz)")
    fig.update_layout(
        showlegend=False, dragmode="zoom", clickmode="event+select",
        margin=dict(l=70, r=24 if compact else 60, t=10, b=48),
    )
    fig.update_xaxes(range=[float(ov.time_s[0] - ov.dt_s / 2), float(ov.time_s[-1] + ov.dt_s / 2)], autorange=False)
    fig.update_yaxes(range=[float(ov.freq_mhz[0] - ov.df_mhz / 2), float(ov.freq_mhz[-1] + ov.df_mhz / 2)], autorange=False,
                     tickformat=_freq_tickformat(float(ov.freq_mhz.max() - ov.freq_mhz.min())))
    return fig


def window_axes(meta: SourceMeta, t0: int, f0: int, nt: int, nf: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    """(time axis s, frequency axis MHz ascending, flipped) for a native window; flipped means rows must be reversed."""
    t = (t0 + np.arange(nt)) * meta.tsamp_s
    f = meta.channel_freq_mhz(f0 + np.arange(nf))
    flipped = meta.foff_mhz < 0
    return t, (f[::-1] if flipped else f), flipped


def tile_figure(
    values: np.ndarray,
    t_axis: np.ndarray,
    f_axis: np.ndarray,
    title: str,
    height: int = 250,
    zrange: Optional[Tuple[float, float]] = (0.0, 1.0),
    show_axes_titles: bool = True,
    value_label: str = "value",
) -> go.Figure:
    """One 64 x 64 tile on physical axes. ``values`` is (freq, time) with frequency ascending."""
    kw = {} if zrange is None else dict(zmin=zrange[0], zmax=zrange[1])
    fig = go.Figure(go.Heatmap(
        z=values, x=t_axis, y=f_axis, colorscale=HEATMAP_COLORSCALE, showscale=False, **kw,
        hovertemplate=f"t %{{x:,.3f}} s<br>%{{y:,.4f}} MHz<br>{value_label} %{{z:.3f}}<extra></extra>",
    ))
    style_plotly_figure(fig, height=height, title=title,
                        xaxis_title="Elapsed time (s)" if show_axes_titles else None,
                        yaxis_title="Frequency (MHz)" if show_axes_titles else None)
    fig.update_layout(margin=dict(l=70, r=12, t=34, b=44))
    fig.update_xaxes(nticks=4)
    fig.update_yaxes(nticks=5, tickformat=",.3f")
    return fig


def zoom_figure(
    window: np.ndarray,
    meta: SourceMeta,
    t0: int,
    f0: int,
    box: Optional[Tuple[int, int, int, int]] = None,
    height: int = 420,
) -> go.Figure:
    """Native-resolution neighbourhood. ``box`` = (t0, f0, nt, nf) of the candidate tile, drawn as a rectangle."""
    nt, nf = window.shape
    t_axis, f_axis, flipped = window_axes(meta, t0, f0, nt, nf)
    arr = window.T[::-1] if flipped else window.T
    z, label = display_levels(arr, clip=(1.0, 99.8))
    fig = go.Figure(go.Heatmap(
        z=z, x=t_axis, y=f_axis, colorscale=HEATMAP_COLORSCALE, zmin=0, zmax=255, colorbar=_colorbar(),
        hovertemplate="t %{x:,.3f} s<br>%{y:,.4f} MHz<extra></extra>",
    ))
    if box is not None:
        bt0, bf0, bnt, bnf = box
        bt, bf, _ = window_axes(meta, bt0, bf0, bnt, bnf)
        half_t = meta.tsamp_s / 2
        half_f = abs(meta.foff_mhz) / 2
        fig.add_shape(type="rect", x0=bt[0] - half_t, x1=bt[-1] + half_t, y0=bf[0] - half_f, y1=bf[-1] + half_f,
                      line=dict(color=COLOR_MARK, width=2), fillcolor="rgba(0,0,0,0)")
    style_plotly_figure(fig, height=height, xaxis_title="Elapsed time (s)", yaxis_title="Frequency (MHz)")
    fig.update_layout(margin=dict(l=76, r=60, t=10, b=48))
    fig.update_yaxes(tickformat=",.3f")
    return fig
