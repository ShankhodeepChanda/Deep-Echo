---
version: 1
slug: "app-py"
primary_target: "app.py"
related_targets: []
---

# Surface brief: SETI-AI Observatory dashboard (app.py)

Scope and visitor mode: Operate. A researcher or student inspects one radio observation at a time, triages unusual candidates, compares, and exports a review.

Audience, job, task: find which few candidates in an observation deserve a second look, understand why, and record a decision that survives restarts. Proof/content: only real files (rankings, headers, waterfall windows, model scores). Constraints: Streamlit stack, owner-pinned dark charcoal with restrained teal, no fabricated numbers, model internals behind Advanced.

Unresolved: PDF export depends on a clean matplotlib path; archive download is optional.

## Direction contract

THESIS: The observation is a strip and candidates are marks pinned to it. One thing is memorable: a full-band time-frequency strip, always present at the top of the working pages, with candidate marks at their measured coordinates. It refuses the category default of a KPI-tile grid plus an ML-metrics tab.

OWN-WORLD: Charcoal neutrals (near-black ground, one step lighter panels, hairline borders), a single teal for selection, primary action and signal intensity, amber reserved for unusual-signal marks and flags only. Workhorse humanist sans (Atkinson Hyperlegible Next) for everything; tabular monospace only for frequency, time and score values. Spectrograms use a dark-to-teal-to-pale ramp, never a rainbow. Radii 4-6px, no shadows beyond one popover shadow, no gradients on surfaces.

STORY: Visitor learns what was observed and how many signals are unusual, sees where they sit on the strip, opens one, reads why it was flagged and what is still unknown, decides, annotates, exports. They are never told a high score means extraterrestrial.

FIRST VIEWPORT: Left sidebar (observation, sensitivity, upload, advanced). Main: page title row with observation name and status; segmented navigation (Overview, Candidates, Waterfall, Cadence, Compare, Export); under it the strip at full width (about 190px) with amber marks; below, one dense line of observation facts (telescope, band, frequency range, duration, date, candidates at this sensitivity) and a ranked candidate preview table with thumbnails. Primary action: Review top candidate.

FORM: Standard Operate shell (sidebar plus segmented page tabs plus work area); the signature move is the observation strip, not any catalog world. Brief-pinned direction overrides the concept-seed roll. seed: none (unavailable)

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
