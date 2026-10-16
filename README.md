# DEEP ECHO

**DEEP ECHO** is an unsupervised deep learning pipeline and instrument dashboard for technosignature candidate detection in radio astronomy. It analyzes high-resolution time-frequency dynamic spectra recorded by major radio telescopes (including the Parkes 64m dish and the Green Bank 100m Telescope) using convolutional autoencoders trained to identify spectral anomalies against terrestrial and astronomical background noise.

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

| Target | Telescope | Receiver Band | Center Frequency | Matrix Patches | Split (Train / Val / Test) |
|---|---|---|---|---|---|
| **G12.72+6.26** | Parkes 64m Dish | GUPPI L-band | 1359.75 MHz | 80,080 | 19,600 / 4,200 / 4,200 (subsampled) |
| **ProxCen_S** | Parkes / MeerKAT | GUPPI S-band (4 IFs) | 2600.00 MHz | 28,000 | 19,600 / 4,200 / 4,200 |
| **HIP 26137** | Green Bank Telescope | BL Receiver X-band | 8410.00 MHz | 28,000 | 19,600 / 4,200 / 4,200 |

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

## Dashboard Navigation

The DEEP ECHO dashboard (`app.py`) provides a four-tab operational interface:

1. **Explorer**: Dynamic spectrum stream telemetry, frequency band specifications, tile dimensions, and raw metadata for any selected observation target.
2. **Train**: Real-time training progress curves, checkpoint manifests, and partition summaries.
3. **Results**: Holdout partition anomaly statistics, error histograms, coordinate tables, visual 3-panel dynamic spectrum triplets (Original, Reconstruction, Residual Error), and synthetic signal injection sensitivity curves.
4. **Compare**: Side-by-side dynamic spectrum triplet comparisons of top candidates for the active telescope pointing.

---

## Project Structure

```text
SETI/
├── app.py                                   # DEEP ECHO Streamlit dashboard
├── requirements.txt                         # Python dependencies
├── ui/
│   ├── styles.css                           # Instrument dark theme CSS
│   └── components.py                        # Reusable UI widgets and styles
├── src/
│   ├── models/cnn_autoencoder.py            # PyTorch autoencoder architecture
│   ├── filterbank_reader.py                 # Multi-receiver .fil parser & tiler
│   ├── sequential_trainer.py                # Multi-dataset trainer with replay
│   ├── cadence_filter.py                    # ON-OFF telescope pointing filter
│   ├── signal_generator.py                  # Synthetic RF waveform generator
│   └── training.py                          # Training loops and error calculators
├── scripts/
│   ├── evaluate_all_targets.py              # Evaluates holdout sets for all targets
│   ├── injection_recovery_benchmark.py      # SNR sensitivity & recall benchmark
│   ├── train_sequential.py                  # Sequential training across datasets
│   └── batch_scan.py                        # Batch evaluation runner
├── artifacts/
│   ├── checkpoints/sequential_training/     # Model weights and manifests
│   ├── figures/                             # Triplet anomaly plots & recall curves
│   └── results/                             # Rankings CSVs and evaluation JSONs
└── tests/                                   # 24 unit and integration tests
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
# Evaluate all 3 telescope targets and produce candidate triplet images
.\venv\Scripts\python.exe scripts/evaluate_all_targets.py

# Run synthetic signal injection & recovery sensitivity benchmark
.\venv\Scripts\python.exe scripts/injection_recovery_benchmark.py
```

### 4. Launch DEEP ECHO Dashboard
```powershell
.\venv\Scripts\streamlit.exe run app.py
```
Open `http://localhost:8501` in your browser.
