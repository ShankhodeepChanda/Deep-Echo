# Product

<!-- impeccable:product-schema 1 -->

<!-- Provenance: written from the owner's detailed redesign brief (2026-10-23), not from an interview. Every fact below is stated in that brief or visible in the repository; nothing is invented. -->

## Platform

web

## Stack
Existing: Streamlit (1.66 installed), Plotly, NumPy/SciPy/pandas, PyTorch. The brief says keep the stack; no framework migration.

## Users
Researchers and students who want to find, inspect, compare and review unusual radio signals in telescope observations. They are not machine-learning developers and do not want to configure model internals. They work at a laptop, often for long sessions, reviewing many candidates and deciding which deserve follow-up.

## Product Purpose
SETI-AI Observatory is a signal-investigation workspace built on a deep anomaly detector (CNN autoencoder) for the project "AI Search for Unknown Technosignatures Using Deep Anomaly Detection". It answers: what observation am I looking at, how many unusual candidates were found, which deserve attention first, what do they look like, why were they flagged, are they consistent with known interference, do they persist in ON/OFF pointings, what should I do next, and can I export my review. Success is a researcher moving from an observation to a short, defensible list of reviewed candidates.

## Positioning
The model emits an unusualness (reconstruction-error) score, not a validated source class. The product keeps that score separate from a transparent, evidence-based, explicitly preliminary assessment, and ties every candidate to its real time and frequency coordinates in the real observation.

## Operating Context
Inputs: SIGPROC filterbank (.fil) and HDF5 (.h5/.hdf5) observations; three bundled Breakthrough Listen observations in `data/raw` (Parkes L-band G12.72+6.26, Parkes S-band Proxima Cen, GBT X-band HIP26137); ranked held-out-tile results in `artifacts/results`; a Breakthrough Listen Open Data API client (`src/bl_api_client.py`); an ABACAD ON/OFF cadence filter (`src/cadence_filter.py`). Review decisions must survive restarts.

## Capabilities and Constraints
- A high score is never proof of extraterrestrial origin; copy must say so.
- No fabricated statistics, candidates, coordinates, classifications, cadence results or confidence percentages.
- Missing metadata, unavailable analysis, zero candidates and processing failure are four different states.
- Do not silently substitute synthetic data for real data. Do not interpret frequency-domain CSV as a waveform.
- Preserve model checkpoints, backend APIs, training functionality (kept under Advanced), and uploaded source data. Do not retrain on reruns.
- Large files (1.4 to 1.8 GB each) must be handled memory-consciously.
- Known: the pre-existing rankings CSVs carry `time_offset_sec` / `center_freq_mhz` columns derived from the tile index, not measured; true tile coordinates are recovered from the seeded partitioning and verified against the stored scores.
- Undecided: PDF export (CSV only unless a clean path exists).

## Brand Commitments
Application title: SETI-AI Observatory. Visual direction pinned by the owner: dark charcoal / near-black scientific workspace, restrained cyan or teal accent, compact information-dense candidate views, no generic Streamlit template look, no gaming/flashy treatment, no emoji navigation.

## Evidence on Hand
Real: three observations in `data/raw`; per-target held-out rankings (4,200 tiles each) and thresholds; trained checkpoints; top-candidate triplet figures; injection-recovery benchmark. Absent (must not be fabricated): ON/OFF cadence sets for these targets, a validated source classifier, RFI percentages from an implemented method, observation-date metadata beyond the filterbank `tstart`.

## Product Principles
1. The observation and its candidates lead; model internals sit behind Advanced.
2. Every number on screen traces to a real file or computation.
3. Say what is known, what is unknown, and what would settle it.
4. Reviewing is the core loop: inspect, decide, annotate, export.
5. Familiar controls and dense, scannable layout over expressive chrome.

## Accessibility & Inclusion
Readable contrast (WCAG AA), visible keyboard focus, no colour-only status, laptop-width layouts.
