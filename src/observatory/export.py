"""
Export of reviewed candidates: CSV records, a printable HTML report and a PDF.

Only fields that exist for a candidate are filled; unavailable values stay empty
(never a placeholder number). Every export carries the standing limitations of
the method so the file is safe to share on its own.
"""

from __future__ import annotations

import base64
import html
import io
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.observatory.assessment import Assessment
from src.observatory.observations import Observation
from src.observatory.reviews import STATUS_LABELS, Review

LIMITATIONS = (
    "The unusualness score is a reconstruction error from an autoencoder trained on typical data. It ranks how "
    "different a tile looks; it does not identify the source of a signal.",
    "Source assessments are preliminary, rule-based indicators. They are not a validated classifier and carry no "
    "confidence values.",
    "A high score is not evidence of extraterrestrial origin. Signals should be checked against ON/OFF cadence "
    "observations and known interference before any follow-up claim.",
)

COLUMNS = [
    "observation_id", "observation", "telescope", "band", "candidate_id", "rank",
    "elapsed_start_s", "elapsed_end_s", "start_utc", "freq_low_mhz", "freq_high_mhz", "freq_center_mhz",
    "unusualness_score", "cutoff", "sensitivity", "cutoff_basis", "above_cutoff", "analysis_basis",
    "review_status", "review_note", "reviewed_utc",
    "assessment", "assessment_why", "evidence_for_interference", "measured_structure", "reference_allocations",
    "cadence_result", "missing_evidence", "limitations",
]


def _joined(items: Iterable[str]) -> str:
    return " | ".join(i for i in items if i)


def export_frame(
    candidates: pd.DataFrame,
    obs: Observation,
    cutoff_value: float,
    sensitivity: str,
    cutoff_basis: str,
    analysis_basis: str,
    reviews: Dict[str, Review],
    assessments: Dict[str, Assessment],
    cadence: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """One row per candidate with review, assessment and cadence fields where they exist."""
    cadence = cadence or {}
    meta = obs.meta
    rows: List[dict] = []
    for r in candidates.itertuples(index=False):
        review = reviews.get(r.candidate_id)
        a = assessments.get(r.candidate_id)
        start_utc = ""
        if meta is not None and meta.start_utc is not None and np.isfinite(r.t_start_s):
            start_utc = (meta.start_utc + timedelta(seconds=float(r.t_start_s))).isoformat(timespec="seconds") + "Z"
        rows.append({
            "observation_id": obs.obs_id,
            "observation": obs.label,
            "telescope": (meta.telescope if meta else None) or (obs.results.summary.get("telescope") if obs.results else "") or "",
            "band": (meta.band if meta else None) or (obs.results.summary.get("band") if obs.results else "") or "",
            "candidate_id": r.candidate_id,
            "rank": int(r.rank),
            "elapsed_start_s": None if not np.isfinite(r.t_start_s) else round(float(r.t_start_s), 6),
            "elapsed_end_s": None if not np.isfinite(r.t_end_s) else round(float(r.t_end_s), 6),
            "start_utc": start_utc,
            "freq_low_mhz": None if not np.isfinite(r.f_low_mhz) else round(float(r.f_low_mhz), 6),
            "freq_high_mhz": None if not np.isfinite(r.f_high_mhz) else round(float(r.f_high_mhz), 6),
            "freq_center_mhz": None if not np.isfinite(r.f_center_mhz) else round(float(r.f_center_mhz), 6),
            "unusualness_score": float(r.score),
            "cutoff": float(cutoff_value),
            "sensitivity": sensitivity,
            "cutoff_basis": cutoff_basis,
            "above_cutoff": bool(r.score > cutoff_value),
            "analysis_basis": analysis_basis,
            "review_status": STATUS_LABELS[review.status] if review else STATUS_LABELS["pending"],
            "review_note": review.note if review else "",
            "reviewed_utc": review.updated_utc if review else "",
            "assessment": a.category if a else "",
            "assessment_why": a.why if a else "",
            "evidence_for_interference": _joined(e.detail for e in a.evidence if e.kind in ("rfi", "artefact")) if a else "",
            "measured_structure": a.characteristics.morphology if a and a.characteristics else "",
            "reference_allocations": _joined(b.name for b in a.bands) if a else "",
            "cadence_result": cadence.get(r.candidate_id, "not run"),
            "missing_evidence": _joined(a.missing) if a else "",
            "limitations": " ".join(LIMITATIONS),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


# ---------------------------------------------------------------------------
# Thumbnails and reports
# ---------------------------------------------------------------------------
def tile_png(tile: np.ndarray, size: int = 160) -> bytes:
    """Small PNG of a normalised 64 x 64 tile using the app's dark-to-teal ramp."""
    from matplotlib.colors import LinearSegmentedColormap
    from PIL import Image

    ramp = LinearSegmentedColormap.from_list("obs", ["#0e1214", "#12353a", "#1f7a78", "#6fd3c4", "#eafaf6"])
    t = np.asarray(tile, dtype=float)
    span = t.max() - t.min()
    t = (t - t.min()) / span if span > 1e-12 else np.zeros_like(t)
    rgba = (ramp(np.flipud(t.T)) * 255).astype(np.uint8)   # frequency up, time right
    img = Image.fromarray(rgba).convert("RGB").resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def build_html_report(frame: pd.DataFrame, obs: Observation, thumbs: Optional[Dict[str, bytes]] = None) -> str:
    """Self-contained printable report (open in a browser; Print > Save as PDF also works)."""
    thumbs = thumbs or {}
    esc = html.escape
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>SETI-AI Observatory candidate report</title>",
        "<style>body{font:14px/1.5 system-ui,sans-serif;color:#1b1f23;max-width:960px;margin:2rem auto;padding:0 1rem}"
        "h1{font-size:1.4rem}h2{font-size:1.05rem;margin-top:1.6rem}table{border-collapse:collapse;width:100%}"
        "td,th{border-bottom:1px solid #d7dde2;padding:.35rem .5rem;text-align:left;vertical-align:top;font-size:13px}"
        ".c{display:flex;gap:1rem;margin:1rem 0;padding-bottom:1rem;border-bottom:1px solid #d7dde2}"
        ".c img{width:140px;height:140px;image-rendering:pixelated}.k{color:#566069}.lim{background:#f1f4f6;padding:.6rem .9rem;border-radius:4px}"
        "</style></head><body>",
        "<h1>SETI-AI Observatory — candidate report</h1>",
        f"<p class='k'>{esc(obs.label)} · generated {generated} · {len(frame)} candidate(s)</p>",
        "<div class='lim'><b>Limitations</b><ul>" + "".join(f"<li>{esc(t)}</li>" for t in LIMITATIONS) + "</ul></div>",
        "<h2>Candidates</h2>",
    ]
    for r in frame.itertuples(index=False):
        img = f"<img alt='tile' src='{data_uri(thumbs[r.candidate_id])}'>" if r.candidate_id in thumbs else ""
        freq = "" if pd.isna(r.freq_center_mhz) else f"{r.freq_center_mhz:.4f} MHz ({r.freq_low_mhz:.4f}–{r.freq_high_mhz:.4f})"
        when = "" if pd.isna(r.elapsed_start_s) else f"{r.elapsed_start_s:.3f}–{r.elapsed_end_s:.3f} s elapsed"
        parts.append(
            f"<div class='c'>{img}<div><b>{esc(r.candidate_id)}</b> — {esc(r.assessment or 'not assessed')} (preliminary)<br>"
            f"<span class='k'>Review:</span> {esc(r.review_status)}"
            + (f" — {esc(r.review_note)}" if r.review_note else "") + "<br>"
            f"<span class='k'>Score:</span> {r.unusualness_score:.5f} (cutoff {r.cutoff:.5f}, {esc(r.sensitivity)} sensitivity)<br>"
            f"<span class='k'>Frequency:</span> {esc(freq) or 'unavailable'}<br><span class='k'>Time:</span> {esc(when) or 'unavailable'}<br>"
            f"<span class='k'>Cadence:</span> {esc(r.cadence_result)}<br>"
            + (f"<span class='k'>Why:</span> {esc(r.assessment_why)}<br>" if r.assessment_why else "")
            + (f"<span class='k'>Missing evidence:</span> {esc(r.missing_evidence)}" if r.missing_evidence else "")
            + "</div></div>"
        )
    parts.append("</body></html>")
    return "".join(parts)


def build_pdf(frame: pd.DataFrame, obs: Observation, thumbs: Optional[Dict[str, np.ndarray]] = None) -> bytes:
    """PDF report via matplotlib (no extra dependency): one summary page, then candidates, four per page."""
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    thumbs = thumbs or {}
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "SETI-AI Observatory — candidate report", fontsize=16, weight="bold")
        fig.text(0.08, 0.915, f"{obs.label} · {len(frame)} candidate(s) · "
                              f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", fontsize=9, color="#555")
        y = 0.87
        fig.text(0.08, y, "Limitations", fontsize=11, weight="bold")
        for t in LIMITATIONS:
            for i, line in enumerate(textwrap.wrap(t, 100)):
                y -= 0.022
                fig.text(0.09 if i else 0.08, y, ("• " if i == 0 else "  ") + line, fontsize=8.5)
        pdf.savefig(fig)
        plt.close(fig)

        per_page = 4
        rows = list(frame.itertuples(index=False))
        for start in range(0, len(rows), per_page):
            fig = plt.figure(figsize=(8.27, 11.69))
            for slot, r in enumerate(rows[start:start + per_page]):
                top = 0.95 - slot * 0.24
                if r.candidate_id in thumbs:
                    ax = fig.add_axes([0.08, top - 0.19, 0.2, 0.19 * 8.27 / 11.69 * 1.0])
                    ax.imshow(np.flipud(np.asarray(thumbs[r.candidate_id]).T), cmap="viridis", aspect="auto")
                    ax.set_xticks([]); ax.set_yticks([])
                lines = [
                    f"{r.candidate_id}  —  {r.assessment or 'not assessed'} (preliminary)",
                    f"Review: {r.review_status}" + (f" — {r.review_note}" if r.review_note else ""),
                    f"Score {r.unusualness_score:.5f} (cutoff {r.cutoff:.5f}, {r.sensitivity} sensitivity)",
                    "Frequency: " + ("unavailable" if pd.isna(r.freq_center_mhz) else f"{r.freq_center_mhz:.4f} MHz ({r.freq_low_mhz:.4f}–{r.freq_high_mhz:.4f})"),
                    "Time: " + ("unavailable" if pd.isna(r.elapsed_start_s) else f"{r.elapsed_start_s:.3f}–{r.elapsed_end_s:.3f} s elapsed"),
                    f"Cadence: {r.cadence_result}",
                ]
                if r.assessment_why:
                    lines += textwrap.wrap("Why: " + r.assessment_why, 78)
                if r.missing_evidence:
                    lines += textwrap.wrap("Missing evidence: " + r.missing_evidence, 78)
                for i, line in enumerate(lines[:11]):
                    fig.text(0.32, top - 0.012 - i * 0.0175, line, fontsize=8, weight="bold" if i == 0 else "normal")
            pdf.savefig(fig)
            plt.close(fig)
    return buf.getvalue()
