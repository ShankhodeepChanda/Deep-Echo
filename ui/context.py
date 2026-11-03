"""Per-run context shared by every page, and the analysis-state messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from src.observatory.candidates import AnalysisKind, CandidateSet, Cutoff
from src.observatory.observations import Observation
from src.observatory.reviews import Review, ReviewStore
from src.observatory.scoring import ModelBundle
from src.observatory.sources import ObservationSource
from ui.components import callout, chip

PAGES = ["Overview", "Candidates", "Waterfall", "Cadence Analysis", "Compare", "Export"]


@dataclass
class Ctx:
    observations: List[Observation]
    obs: Optional[Observation]
    bundle: Optional[ModelBundle]
    model_error: Optional[str]
    cset: Optional[CandidateSet]
    level: str
    cutoff: Optional[Cutoff]
    view: pd.DataFrame                       # candidates above the cutoff, best first
    reviews: Dict[str, Review]
    store: ReviewStore
    cadence: Dict[str, dict]
    source: Optional[ObservationSource]
    selected_id: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.cset is not None and self.cset.kind in (AnalysisKind.READY, AnalysisKind.SOURCE_MISSING)

    @property
    def has_coordinates(self) -> bool:
        return bool(self.cset and self.cset.has_coordinates)

    @property
    def can_read_data(self) -> bool:
        return self.source is not None and self.has_coordinates

    def status_of(self, candidate_id: str) -> str:
        r = self.reviews.get(candidate_id)
        return r.status if r else "pending"

    def verdict_of(self, candidate_id: str) -> Optional[str]:
        entry = self.cadence.get(candidate_id)
        return entry.get("verdict") if entry else None

    def row(self, candidate_id: Optional[str]):
        """Row of the full candidate frame for an id, or None."""
        if not candidate_id or self.cset is None or self.cset.frame.empty:
            return None
        hit = self.cset.frame[self.cset.frame["candidate_id"] == candidate_id]
        return None if hit.empty else next(hit.itertuples(index=False))

    @property
    def selected(self):
        return self.row(self.selected_id)


def status_chip(ctx: Ctx) -> str:
    """Short status for the header and sidebar."""
    if ctx.obs is None:
        return chip("No observation", "warn")
    cs = ctx.cset
    if cs is None:
        return chip("Not loaded", "warn")
    if cs.kind == AnalysisKind.READY:
        return chip("Full scan" if cs.basis == "full scan" else "Held-out tiles analysed", "ok")
    if cs.kind == AnalysisKind.NOT_ANALYZED:
        return chip("Not analysed yet", "warn")
    if cs.kind == AnalysisKind.SOURCE_MISSING:
        return chip("Source file missing", "warn")
    return chip("Analysis unavailable", "bad")


def state_panel(ctx: Ctx, need_data: bool = False, what: str = "this view") -> bool:
    """Explain why a page cannot show content. Returns True when the page may proceed.

    The states are kept distinct on purpose: no observation, not yet analysed,
    stored results without the data file, results that failed to load, and a
    missing model are different situations with different remedies.
    """
    if ctx.obs is None:
        callout(
            "info", "No observation selected",
            "<p>Choose an observation in the sidebar, or add one with <b>Add observation</b>. "
            "Supported files are SIGPROC filterbank (<code>.fil</code>) and HDF5 (<code>.h5</code>, <code>.hdf5</code>).</p>",
        )
        return False
    cs = ctx.cset
    if ctx.obs.problems and ctx.obs.meta is None and (cs is None or cs.kind == AnalysisKind.FAILED):
        items = "".join(f"<li>{_esc(p)}</li>" for p in ctx.obs.problems)
        callout("bad", "This observation could not be loaded", f"<ul>{items}</ul>")
        return False
    if cs is None:
        callout("bad", "Analysis could not start", "<p>No candidate data was produced for this observation.</p>")
        return False
    if cs.kind == AnalysisKind.FAILED:
        callout("bad", "Analysis results could not be used", f"<p>{_esc(cs.message)}</p>")
        return False
    if cs.kind == AnalysisKind.NOT_ANALYZED:
        callout(
            "warn", "This observation has not been analysed yet",
            "<p>The file is loaded, but no unusualness scores exist for it. Open <b>Overview</b> and choose "
            "<b>Analyse observation</b> to score every tile.</p>",
        )
        return False
    if need_data and not ctx.can_read_data:
        why = "the source data file is not available" if ctx.source is None else "tile positions could not be verified"
        callout("warn", f"{what.capitalize()} needs the observation data", f"<p>It is unavailable because {why}. Scores and rankings are still shown elsewhere.</p>")
        return False
    return True


def _esc(text: str) -> str:
    from html import escape
    return escape(str(text))


def selected_from_state() -> Optional[str]:
    return st.session_state.get("selected_candidate")
