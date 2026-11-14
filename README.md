# SETI-AI Observatory

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deep-echo-72qjxkrmphscusgmhgfnla.streamlit.app/)

**Live demo: [deep-echo-72qjxkrmphscusgmhgfnla.streamlit.app](https://deep-echo-72qjxkrmphscusgmhgfnla.streamlit.app/)**

> The hosted demo shows the saved held-out evaluation (scores and rankings) for the three bundled targets. The raw telescope recordings (1.4–1.8 GB each) are not in the repository, so waterfall views need an observation uploaded (up to 200 MB) or the app run locally with the files in `data/raw/`.

**SETI-AI Observatory** is a signal-investigation dashboard for finding, inspecting, comparing and reviewing unusual radio signals. It is built on **DEEP ECHO**, an unsupervised deep learning pipeline for technosignature candidate detection in radio astronomy. It analyzes high-resolution time-frequency dynamic spectra recorded by major radio telescopes (including the Parkes 64m dish and the Green Bank 100m Telescope) using convolutional autoencoders trained to identify spectral anomalies against terrestrial and astronomical background noise.

---

## Architecture Overview

```
Raw Filterbank Stream (.fil)
            │
            ▼
 Dynamic Spectrum Tiling (64 × 64 patches)
            │
            ▼
 CNN Autoencoder (Latent Dimension: 48)
 ├── Conv2d (1 → 16, stride 2) + ReLU
 ├── Conv2d (16 → 32, stride 2) + ReLU
 ├── Conv2d (32 → 64, stride 2) + ReLU
 ├── Linear Latent Bottleneck (48-d)
 ├── ConvTranspose2d (64 → 32, stride 2) + ReLU
 ├── ConvTranspose2d (32 → 16, stride 2) + ReLU
 └── ConvTranspose2d (16 → 1, stride 2) + Sigmoid
            │
            ▼
 Mean Squared Reconstruction Error (MSE)
            │
            ▼
 Anomaly Classification & Triplet Visual Inspection
 [Original Observation vs. Autoencoder Reconstruction vs. Residual Error]
```

---

## Observation Targets & Telescopes

DEEP ECHO supports multi-dataset sequential evaluation across diverse astronomical receivers:

| Target | Telescope | Band | Frequency range (from header) | Tiles in the 64×64 grid | Held-out evaluation tiles |
|---|---|---|---|---|---|
| **G12.72+6.26** | Parkes 64m Dish | L-band | 1,205.97–1,359.97 MHz | 80,080 | 4,200 (of a 28,000-tile sample) |
| **ProxCen_S** | Parkes 64m Dish | S-band (4 IFs) | 2,240.00–2,368.00 MHz | 28,000 | 4,200 |
| **HIP 26137** | Green Bank Telescope | X-band | 7,501.46–11,251.46 MHz | 81,920 | 4,200 (of a 28,000-tile sample) |

These values are read from the observation files themselves; the dashboard never hard-codes them.

### Continual Sequential Training & Replay Buffer
To support observations across multiple telescopes without catastrophic forgetting, the model is sequentially trained using an experience replay buffer (10% retained exemplar memory from preceding telescope stages).

- **Global Recalibrated Threshold:** `0.022265` (calibrated on 12,600 unseen validation tiles across all three targets at the 99th percentile).
- **Target-Adaptive Baseline Thresholds:** e.g., `0.006484` for quiet L-band observations.

---

## Synthetic Signal Injection & Recovery Benchmark

Because verified extraterrestrial technosignatures do not exist in training archives, pipeline sensitivity and completeness limits are empirically established by injecting synthetic drifting narrowband carriers ($0.8$ channel width, drift rates $\pm 0.6$ bins/step) into pristine telescope background noise tiles across signal-to-noise ratios from **-5 dB to +35 dB**.

### Empirical Performance Summary

| SNR (dB) | Adaptive Recall (`0.0065`) | Global Recall (`0.0223`) | Mean Reconstruction MSE | Status |
|---|---|---|---|---|
| **-5.0 dB** | 2.00% | 0.00% | 0.004862 | Noise Floor |
| **0.0 dB** | 9.33% | 0.00% | 0.005146 | Sub-threshold |
| **+5.0 dB** | 52.33% | 0.00% | 0.006000 | **50% Sensitivity Limit (~7.5 dB)** |
| **+10.0 dB** | 80.67% | 0.00% | 0.008163 | Linear response regime |
| **+15.0 dB** | 98.67% | 0.00% | 0.011353 | **90% Completeness Limit (~12.5 dB)** |
| **+20.0 dB** | 100.00% | 0.00% | 0.013879 | 100% Recovery |
| **+25.0 dB** | 100.00% | 1.00% | 0.015047 | 100% Recovery |
| **+30.0 dB** | 100.00% | 4.33% | 0.016027 | 100% Recovery |
| **+35.0 dB** | 100.00% | 11.67% | 0.017120 | 100% Recovery |

- **Baseline False Positive Rate (FPR):** $0.00\%$ on clean telescope noise.
- **Figures:** `artifacts/figures/injection_recovery_curve.png`
- **Benchmark Data:** `artifacts/results/injection_recovery_benchmark.json`

---

## The Dashboard

`app.py` is organised around the investigation workflow, not the model lifecycle:

1. **Overview** — the observation, its metadata, a time-frequency strip with unusual signals marked at their measured positions, how many were found at the current sensitivity, and a ranked preview.
2. **Candidates** — a filterable, sortable list with tile thumbnails and a detail view: what was observed, what the model expected, why it was flagged, a preliminary source assessment with its evidence, and review controls (retain, flag, dismiss, notes).
3. **Waterfall** — the whole observation as an interactive time-frequency plot with candidate marks; click a mark to see the tile at native resolution.
4. **Cadence Analysis** — ON/OFF comparison using `src/cadence_filter.py`. It needs OFF-target observations covering the same band at the same resolution; without them the page says so and nothing is simulated.
5. **Compare** — two real candidates side by side (tiles, spectra, time series, measurements), or a summary across observations.
6. **Export** — CSV, printable HTML and PDF of retained/flagged/reviewed candidates, including evidence, missing evidence and limitations.

The sidebar holds the observation, **detection sensitivity** (Low / Medium / High), **Add observation**, and **Advanced / About the model**. Training diagnostics, thresholds and the synthetic training lab live under Advanced (toggle *Show developer tools*) and never run unless you press their buttons.

### What the numbers mean

* **Unusualness score** — the autoencoder's mean squared reconstruction error for a 64×64 tile. It ranks how different a tile looks from typical data. It does **not** identify the source, and scores are only comparable within one observation.
* **Source assessment** — a separate, *preliminary* rule-based layer (`src/observatory/assessment.py`). It looks at what can be measured in the tile (narrowband structure, persistence, drift, impulsive structure), the tile's frequency against an indicative list of satellite/radar allocations, band-edge position, and cadence results. Labels: *Likely interference (RFI)*, *Possible instrument artefact*, *Unusual — needs review*, *Insufficient information*. It gives no confidence percentages and is not a validated classifier.
* **A high score is never evidence of extraterrestrial origin.** Every page and export says so.

### Detection sensitivity (how Low / Medium / High are defined)

Sensitivity changes only which tiles are *shown*; it does not retrain or recalibrate the model (`src/observatory/candidates.py`).

| Level | Observation with a validated threshold | Observation without one (e.g. an upload) |
|---|---|---|
| **Medium** | exactly the validated threshold (99th percentile of validation scores) | the top 1% of the observation's own scores (a ranking rule, not a calibration) |
| **Low** | about half as many candidates as Medium — the strongest | same rule, relative to Medium |
| **High** | about twice as many as Medium, never fewer than 10 tiles | same rule, relative to Medium |

Counts are recomputed from the loaded scores. A manual numeric cutoff is available under Advanced.

### Where candidate positions come from

The stored rankings CSVs (`artifacts/results/test_rankings_*.csv`) cover a held-out *sample* of tiles, and their `time_offset_sec` / `center_freq_mhz` columns are derived from the tile index rather than measured. The dashboard therefore recovers each tile's true (time, channel) origin from the seeded partitioning used at evaluation time (`src/observatory/provenance.py`) and **verifies** it by re-scoring tiles from the source file and comparing with the stored scores. If that check fails, positions are withheld rather than shown. Use **Analyse observation** on the Overview to score every tile of an observation (about 10–30 seconds for the bundled 1.4–1.8 GB files).

### Your data

* **Add observation** accepts `.fil` (SIGPROC filterbank) and `.h5`/`.hdf5` (Breakthrough Listen style; needs `pip install h5py`). Files are validated before use; frequency-domain CSV tables are not observations and are refused.
* *Upload* keeps a copy in `data/uploads/`; *Open from this computer* reads a (large) file in place. Originals are never modified, and files are read through memory maps so multi-gigabyte observations stay memory-friendly.
* Reviews, uploads and caches live under `data/reviews`, `data/uploads`, `data/cache` (git-ignored). Set `OBSERVATORY_STATE_DIR` to keep them elsewhere.

---

## Project Structure

```text
SETI/
├── app.py                                   # Streamlit entry point (wires the pages together)
├── PRODUCT.md                               # Product context for design work
├── requirements.txt                         # Python dependencies
├── .streamlit/config.toml                   # Dark theme tokens (also themes the data grids)
├── ui/
│   ├── styles.css                           # Observatory stylesheet
│   ├── components.py                        # Theme, formatters, HTML building blocks
│   ├── plots.py                             # Waterfall, tile and zoom figures
│   ├── sidebar.py  context.py  loaders.py   # Sidebar, per-run context, cached data access
│   ├── pages_*.py                           # Overview, Candidates, Waterfall, Cadence, Compare, Export
│   ├── candidate_views.py                   # Ranked table, detail view, review controls
│   ├── intake.py                            # Upload / open in place / archive
│   ├── guide.py                             # First-run guide and glossary
│   ├── advanced.py  dev_lab.py              # Model diagnostics and the synthetic training lab
│   └── assets/favicon.png
├── src/
│   ├── observatory/                         # Dashboard data layer (no Streamlit imports)
│   │   ├── sources.py                       # Validation, memory-mapped reads, overview
│   │   ├── observations.py  provenance.py   # Catalogue, upload registry, tile positions
│   │   ├── scoring.py  candidates.py        # Scoring, full scans, sensitivity, filters
│   │   ├── assessment.py  cadence.py        # Preliminary assessment, ON/OFF adapter
│   │   ├── reviews.py  export.py  archive.py
│   ├── models/cnn_autoencoder.py            # PyTorch autoencoder architecture
│   ├── filterbank_reader.py                 # Multi-receiver .fil parser & tiler
│   ├── sequential_trainer.py                # Multi-dataset trainer with replay
│   ├── cadence_filter.py                    # ON-OFF telescope pointing filter
│   ├── signal_generator.py                  # Synthetic RF waveform generator
│   └── training.py                          # Training loops and error calculators
├── scripts/                                 # Evaluation, training and benchmark runners
├── artifacts/                               # Checkpoints, figures, results
└── tests/                                   # Unit, integration and app-state tests (pytest)
```

---

## Quickstart

### 1. Environment Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Test Suite
```powershell
.\venv\Scripts\python.exe -m pytest tests/
```

### 3. Generate Evaluation & Benchmark Figures
```powershell
# Evaluate all 3 telescope targets and write rankings and candidate figures
.\venv\Scripts\python.exe scripts/evaluate_all_targets.py

# Run synthetic signal injection & recovery sensitivity benchmark
.\venv\Scripts\python.exe scripts/injection_recovery_benchmark.py
```

### 4. Launch the dashboard
```powershell
.\venv\Scripts\streamlit.exe run app.py
```
Open `http://localhost:8501` in your browser. For HDF5 observations also run `pip install h5py` (optional).
