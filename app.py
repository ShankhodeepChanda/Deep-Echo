"""
DEEP ECHO — Instrument Anomaly Detection Dashboard
Deep Learning Pipeline for Radio-Frequency Observations
"""

from pathlib import Path
import json
import time

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import signal as scipy_signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.signal_generator import (
    generate_normal_signal,
    generate_rfi_signal,
    generate_anomaly_signal,
    generate_dataset,
)
from src.config import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_DURATION,
    DEFAULT_N_FFT,
    DEFAULT_HOP_LENGTH,
    SPEC_HEIGHT,
    SPEC_WIDTH,
    LATENT_DIM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_EPOCHS,
    DEFAULT_ANOMALY_PERCENTILE,
    SYNTHETIC_DATA_DIR,
    ARTIFACTS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
)
from src.cadence_filter import analyze_cadence_sequence
from src.filterbank_reader import load_filterbank_waterfall, extract_spectrogram_tiles
from ui.components import (
    load_css,
    render_header,
    sidebar_section,
    style_plotly_figure,
    COLOR_ACCENT,
    COLOR_NORMAL,
    COLOR_RFI,
    COLOR_ANOMALY,
    COLOR_MUTED,
    HEATMAP_COLORSCALE,
)

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="DEEP ECHO",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Signal processing helpers
# ──────────────────────────────────────────────
def compute_spectrogram(
    sig: np.ndarray,
    sample_rate: float,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute power spectrogram via STFT and return (frequencies, times, power_dB)."""
    frequencies, times, Zxx = scipy_signal.stft(
        sig, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length
    )
    power = np.abs(Zxx) ** 2
    power_dB = 10.0 * np.log10(power + 1e-12)
    return frequencies, times, power_dB


def resize_spectrogram(spec: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize 2-D spectrogram to (target_h, target_w) via linear interpolation."""
    from scipy.ndimage import zoom

    zoom_h = target_h / spec.shape[0]
    zoom_w = target_w / spec.shape[1]
    return zoom(spec, (zoom_h, zoom_w), order=1)


def signals_to_spectrograms(
    signals: np.ndarray,
    sample_rate: float,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    target_h: int = SPEC_HEIGHT,
    target_w: int = SPEC_WIDTH,
) -> np.ndarray:
    """Convert array of time-domain signals into batch of normalized spectrograms."""
    specs = []
    for sig in signals:
        _, _, power_dB = compute_spectrogram(sig, sample_rate, n_fft, hop_length)
        resized = resize_spectrogram(power_dB, target_h, target_w)
        specs.append(resized)
    specs = np.array(specs, dtype=np.float32)
    # Per-sample normalization to [0, 1]
    for i in range(len(specs)):
        smin, smax = specs[i].min(), specs[i].max()
        if smax - smin > 1e-8:
            specs[i] = (specs[i] - smin) / (smax - smin)
    return specs


# ──────────────────────────────────────────────
# CNN Autoencoder
# ──────────────────────────────────────────────
class CNNAutoencoder(nn.Module):
    """Convolutional autoencoder for spectrogram anomaly detection."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def train_autoencoder(
    specs: np.ndarray,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LEARNING_RATE,
    latent_dim: int = LATENT_DIM,
    progress_bar=None,
    status_text=None,
) -> tuple[CNNAutoencoder, list[float]]:
    """Train the CNN Autoencoder and return (model, loss_history)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor(specs, dtype=torch.float32).unsqueeze(1).to(device)
    dataset = TensorDataset(tensor, tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CNNAutoencoder(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history: list[float] = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, _ in loader:
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        avg_loss = epoch_loss / len(tensor)
        loss_history.append(avg_loss)
        if progress_bar is not None:
            progress_bar.progress((epoch + 1) / epochs)
        if status_text is not None:
            status_text.text(f"Epoch {epoch + 1}/{epochs} · MSE: {avg_loss:.6f}")

    model.eval()
    return model, loss_history


def compute_reconstruction_errors(
    model: CNNAutoencoder, specs: np.ndarray
) -> np.ndarray:
    """Compute per-sample mean squared reconstruction error."""
    device = next(model.parameters()).device
    tensor = torch.tensor(specs, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        reconstructed = model(tensor)
    mse = ((tensor - reconstructed) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
    return mse


# ──────────────────────────────────────────────
# Plotting helpers (Styled for DEEP ECHO)
# ──────────────────────────────────────────────
def plot_signal(sig: np.ndarray, sample_rate: float, title: str = "Signal") -> go.Figure:
    t = np.arange(len(sig)) / sample_rate
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=sig, mode="lines",
        line=dict(width=0.9, color=COLOR_ACCENT),
    ))
    return style_plotly_figure(fig, height=260, title=title, xaxis_title="Time (s)", yaxis_title="Amplitude")


def plot_spectrogram(
    frequencies: np.ndarray, times: np.ndarray, power_dB: np.ndarray,
    title: str = "Spectrogram",
) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=power_dB, x=times, y=frequencies,
        colorscale=HEATMAP_COLORSCALE, colorbar=dict(title="dB", thickness=12, len=0.85),
    ))
    return style_plotly_figure(fig, height=290, title=title, xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")


def plot_loss_curve(loss_history: list[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=loss_history, mode="lines+markers",
        marker=dict(size=4, color=COLOR_ACCENT),
        line=dict(color=COLOR_ACCENT, width=1.5),
    ))
    return style_plotly_figure(fig, height=270, title="Training Loss Convergence", xaxis_title="Epoch", yaxis_title="MSE Loss")


def plot_error_distribution(
    errors: np.ndarray, labels: np.ndarray, threshold: float
) -> go.Figure:
    df = pd.DataFrame({"Error": errors, "Category": labels})
    fig = px.histogram(
        df, x="Error", color="Category", nbins=60,
        opacity=0.75, barmode="overlay",
        color_discrete_map={
            "normal_like": COLOR_NORMAL,
            "synthetic_rfi": COLOR_RFI,
            "synthetic_anomaly": COLOR_ANOMALY,
        },
    )
    fig.add_vline(
        x=threshold, line_dash="dash", line_color=COLOR_RFI,
        annotation_text=f"Threshold ({threshold:.4f})",
        annotation_font=dict(size=10, color=COLOR_RFI),
    )
    return style_plotly_figure(fig, height=320, title="Reconstruction Error Distribution", xaxis_title="MSE Loss", yaxis_title="Count")


def plot_real_error_distribution(errors: np.ndarray, threshold: float) -> go.Figure:
    status_col = [
        "Anomaly (> 99th %tile)" if e > threshold else "Normal Radio Background"
        for e in errors
    ]
    df = pd.DataFrame({
        "Error": errors,
        "Classification": status_col,
    })
    fig = px.histogram(
        df,
        x="Error",
        color="Classification",
        nbins=65,
        opacity=0.75,
        barmode="overlay",
        color_discrete_map={
            "Normal Radio Background": COLOR_ACCENT,
            "Anomaly (> 99th %tile)": COLOR_ANOMALY,
        },
    )
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color=COLOR_RFI,
        annotation_text=f"Threshold ({threshold:.6f})",
        annotation_position="top right",
        annotation_font=dict(size=10, color=COLOR_RFI),
    )
    return style_plotly_figure(
        fig, height=330,
        title=f"Holdout Test Error Distribution ({len(errors):,} Tiles)",
        xaxis_title="Reconstruction Error (MSE)",
        yaxis_title="Tile Count",
    )


def plot_comparison(
    sig_a: np.ndarray, sig_b: np.ndarray, sample_rate: float,
    label_a: str = "Signal A", label_b: str = "Signal B",
) -> go.Figure:
    t = np.arange(max(len(sig_a), len(sig_b))) / sample_rate
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(label_a, label_b))
    fig.add_trace(go.Scatter(x=t[:len(sig_a)], y=sig_a, mode="lines", line=dict(width=0.8, color=COLOR_ACCENT)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t[:len(sig_b)], y=sig_b, mode="lines", line=dict(width=0.8, color=COLOR_ANOMALY)), row=2, col=1)
    return style_plotly_figure(fig, height=360, xaxis_title="Time (s)")


# ──────────────────────────────────────────────
# Dashboard layout
# ──────────────────────────────────────────────
def main():
    # Inject instrument stylesheet
    load_css()

    # ── Multi-Target Configuration ──
    targets_metadata = {
        "G12.72+6.26 (Parkes L-band)": {
            "key": "G12.72+6.26",
            "name": "G12.72+6.26",
            "telescope": "Parkes 64m Dish",
            "telescope_short": "Parkes 64m",
            "backend": "GUPPI L-band",
            "freq_range": "1317 – 1360 MHz",
            "center_freq": "1359.75 MHz",
            "duration": "299.9 s",
            "samples": "1,025,024",
            "tiles": "80,080 (64×64)",
            "split": "56,056 / 24,024",
            "source_file": "guppi_58212_..._010485_G12.72+6.26_...fil",
            "dims": "1,025,024 samples × 352 channels",
            "fig_prefix": "test_anomaly",
            "calibrated_threshold": 0.006484,
        },
        "ProxCen_S (Parkes S-band)": {
            "key": "ProxCen_S",
            "name": "ProxCen_S",
            "telescope": "Parkes / MeerKAT (BL Multibeam)",
            "telescope_short": "Parkes 64m",
            "backend": "GUPPI S-band",
            "freq_range": "2300 – 2900 MHz",
            "center_freq": "2600.00 MHz",
            "duration": "299.9 s",
            "samples": "1,025,024",
            "tiles": "28,000 (64×64)",
            "split": "19,600 / 8,400",
            "source_file": "guppi_59335_..._ProxCen_S_...fil",
            "dims": "1,025,024 samples × 4 IFs × 512 channels",
            "fig_prefix": "test_anomaly_proxcen",
            "calibrated_threshold": 0.016391,
        },
        "HIP 26137 (GBT X-band)": {
            "key": "HIP26137",
            "name": "HIP 26137",
            "telescope": "Green Bank Telescope (GBT)",
            "telescope_short": "GBT 100m",
            "backend": "BL Digital Receiver X-band",
            "freq_range": "8000 – 9000 MHz",
            "center_freq": "8410.00 MHz",
            "duration": "299.9 s",
            "samples": "16,384",
            "tiles": "28,000 (64×64)",
            "split": "19,600 / 8,400",
            "source_file": "spliced_blc..._HIP26137_...fil",
            "dims": "16,384 time bins × 1,310,720 channels",
            "fig_prefix": "test_anomaly_gbt",
            "calibrated_threshold": 0.022265,
        },
    }

    # ── Sidebar ──
    sidebar_section("Data Source")
    data_mode = st.sidebar.radio(
        "Mode",
        [
            "Real Observation (Parkes / GBT)",
            "Synthetic Prototype",
        ],
        index=0,
        label_visibility="collapsed",
        help="Switch between Breakthrough Listen telescope data and synthetic RF generator.",
    )

    if "Real" in data_mode:
        target_choice = st.sidebar.selectbox(
            "Observation Target",
            list(targets_metadata.keys()),
            index=0,
            help="Select Breakthrough Listen radio telescope target dataset.",
        )
        target_info = targets_metadata[target_choice]
        status_text = f"{target_info['telescope_short'].upper()} · ACTIVE"
    else:
        target_choice = "G12.72+6.26 (Parkes L-band)"
        target_info = targets_metadata[target_choice]
        status_text = "SYNTHETIC PROTOTYPE"

    # ── Top Wordmark Header (No subtitle) ──
    render_header(status_text=status_text)

    # ── Sidebar Parameters ──
    if "Real" in data_mode:
        sidebar_section("Target Specifications")
        st.sidebar.markdown(
            f"""
            <div style='font-size: 0.8rem; line-height: 1.6; color: #94a3b8;'>
                <div><b>Target:</b> {target_info['name']}</div>
                <div><b>Telescope:</b> {target_info['telescope_short']}</div>
                <div><b>Backend:</b> {target_info['backend']}</div>
                <div><b>Center:</b> {target_info['center_freq']}</div>
                <div><b>Samples:</b> {target_info['samples']}</div>
                <div><b>Tiles:</b> {target_info['tiles']}</div>
                <div><b>Split:</b> {target_info['split']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sidebar_section("Active Model")
        active_thresh = target_info.get("calibrated_threshold", 0.006484)
        st.sidebar.markdown(
            f"""
            <div style='font-size: 0.8rem; line-height: 1.6; color: #94a3b8;'>
                <div><b>Model:</b> Continual Multi-Dataset</div>
                <div><b>Stages:</b> 3 / 3 Complete</div>
                <div><b>Threshold:</b> <code>{active_thresh:.6f}</code></div>
                <div><b>Telescopes:</b> Parkes & GBT</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Baseline defaults
        sample_rate = DEFAULT_SAMPLE_RATE
        duration = DEFAULT_DURATION
        n_fft = DEFAULT_N_FFT
        hop_length = DEFAULT_HOP_LENGTH
        num_normal = 200
        num_rfi = 50
        num_anomaly = 50
        latent_dim = LATENT_DIM
        epochs = DEFAULT_EPOCHS
        batch_size = DEFAULT_BATCH_SIZE
        learning_rate = DEFAULT_LEARNING_RATE
        anomaly_pct = DEFAULT_ANOMALY_PERCENTILE

    else:
        sidebar_section("Signal Settings")
        sample_rate = st.sidebar.slider("Sample Rate (Hz)", 500.0, 5000.0, DEFAULT_SAMPLE_RATE, step=100.0, help="Digital sampling rate")
        duration = st.sidebar.slider("Duration (s)", 0.5, 3.0, DEFAULT_DURATION, step=0.25, help="Observation duration in seconds")
        n_fft = st.sidebar.select_slider("FFT Size", options=[64, 128, 256, 512], value=DEFAULT_N_FFT, help="STFT window size")
        hop_length = st.sidebar.select_slider("Hop Length", options=[16, 32, 64, 128], value=DEFAULT_HOP_LENGTH, help="STFT stride length")

        sidebar_section("Dataset Size")
        num_normal = st.sidebar.number_input("Normal", 20, 500, 200, step=10)
        num_rfi = st.sidebar.number_input("RFI", 5, 200, 50, step=5)
        num_anomaly = st.sidebar.number_input("Anomaly", 5, 200, 50, step=5)

        sidebar_section("Hyperparameters")
        latent_dim = st.sidebar.slider("Latent Dim", 8, 128, LATENT_DIM, step=8)
        epochs = st.sidebar.slider("Epochs", 5, 100, DEFAULT_EPOCHS)
        batch_size = st.sidebar.select_slider("Batch Size", options=[8, 16, 32, 64, 128], value=DEFAULT_BATCH_SIZE)
        learning_rate = st.sidebar.select_slider(
            "Learning Rate", options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=DEFAULT_LEARNING_RATE, format_func=lambda v: f"{v:.0e}"
        )
        anomaly_pct = st.sidebar.slider("Threshold Percentile", 90.0, 99.9, DEFAULT_ANOMALY_PERCENTILE, step=0.5)

    # ── Clean Navigation Tabs ──
    tab_explore, tab_train, tab_results, tab_compare = st.tabs(
        ["Explorer", "Train", "Results", "Compare"]
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Tab 1 — Explorer
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_explore:
        if "Real" in data_mode:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Telescope", target_info["telescope_short"])
            col_m2.metric("Target", target_info["name"])
            col_m3.metric("Frequency Band", target_info["freq_range"])
            col_m4.metric("Duration", target_info["duration"])

            st.markdown(
                f"""
                <div class='de-card' style='margin-top: 1rem;'>
                    <div class='de-card-header'>Observation Stream Specification</div>
                    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; font-size: 0.85rem;'>
                        <div><b>Source File:</b> <code>{target_info['source_file']}</code></div>
                        <div><b>Dimensions:</b> {target_info['dims']}</div>
                        <div><b>Tiled Matrix:</b> {target_info['tiles']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            col_type, col_seed = st.columns([2, 1])
            with col_type:
                sig_type = st.selectbox(
                    "Signal Type",
                    ["Normal-like", "RFI (Interference)", "Anomaly (Unusual)"],
                    help="Synthetic waveform family to generate.",
                )
            with col_seed:
                seed = st.number_input("Random Seed", 0, 999999, 42)

            if st.button("Generate Signal", use_container_width=True):
                gen_fn = {
                    "Normal-like": generate_normal_signal,
                    "RFI (Interference)": generate_rfi_signal,
                    "Anomaly (Unusual)": generate_anomaly_signal,
                }[sig_type]

                sig, meta = gen_fn(sample_rate, duration, seed=seed)

                col_sig, col_spec = st.columns(2)
                with col_sig:
                    st.plotly_chart(
                        plot_signal(sig, sample_rate, f"{sig_type} — Time Domain"),
                        use_container_width=True,
                    )
                with col_spec:
                    freqs, times, power_dB = compute_spectrogram(sig, sample_rate, n_fft, hop_length)
                    st.plotly_chart(
                        plot_spectrogram(freqs, times, power_dB, f"{sig_type} — Spectrogram"),
                        use_container_width=True,
                    )

                with st.expander("Signal Metadata"):
                    st.json(meta)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Tab 2 — Train
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_train:
        if "Real" in data_mode:
            summary_path = RESULTS_DIR / "cnn_autoencoder_summary.json"
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Train Partition", f"{summary.get('n_train', 56056):,} tiles")
                c2.metric("Holdout Partition", f"{summary.get('n_val', 24024):,} tiles")
                c3.metric("Final Loss (MSE)", f"{summary.get('final_loss', 0.004513):.6f}")
                c4.metric("Calibrated Threshold", f"{summary.get('calibrated_threshold', 0.006484):.6f}")

                st.plotly_chart(plot_loss_curve(summary.get("loss_history", [])), use_container_width=True)

                seq_manifest_path = ARTIFACTS_DIR / "checkpoints" / "sequential_training" / "experiment_manifest.json"
                seq_csv_path = ARTIFACTS_DIR / "checkpoints" / "sequential_training" / "sequential_stages_summary.csv"
                if seq_manifest_path.exists() and seq_csv_path.exists():
                    try:
                        with open(seq_manifest_path, "r", encoding="utf-8") as f_seq:
                            seq_m = json.load(f_seq)

                        st.markdown("<div class='de-card-header' style='margin-top: 1.5rem;'>Sequential Continual Learning</div>", unsafe_allow_html=True)
                        sc1, sc2, sc3, sc4 = st.columns(4)
                        sc1.metric("Sequential Stages", "3 / 3")
                        sc2.metric("Telescope Facilities", "Parkes & GBT")
                        sc3.metric("Recalibrated Threshold", f"{seq_m.get('final_recalibrated_threshold', 0.022265):.6f}")
                        sc4.metric("Replay Ratio", f"{seq_m.get('replay_ratio', 0.1)*100:.0f}%")

                        df_seq = pd.read_csv(seq_csv_path)
                        cols_show = ["stage", "dataset", "telescope", "band", "train_samples", "val_samples", "test_samples", "final_train_loss", "val_mean_error", "stage_threshold"]
                        st.dataframe(df_seq[[c for c in cols_show if c in df_seq.columns]], use_container_width=True)

                        c_img1, c_img2 = st.columns(2)
                        img_curves = FIGURES_DIR / "sequential_loss_curves.png"
                        img_ret = FIGURES_DIR / "sequential_retention_matrix.png"
                        if img_curves.exists():
                            c_img1.image(str(img_curves), caption="Sequential Loss Convergence", use_container_width=True)
                        if img_ret.exists():
                            c_img2.image(str(img_ret), caption="Cross-Dataset Retention Matrix", use_container_width=True)

                        test_rows = []
                        for target_k, test_v in seq_m.get("final_test_evaluation", {}).items():
                            test_rows.append({
                                "Target": target_k,
                                "Telescope": test_v.get("telescope", "N/A"),
                                "Band": test_v.get("band", "N/A"),
                                "Test Tiles": test_v.get("n_test_tiles", 0),
                                "Mean MSE": f"{test_v.get('test_mean_mse', 0.0):.6f}",
                                "Std MSE": f"{test_v.get('test_std_mse', 0.0):.6f}",
                                "Flagged": f"{test_v.get('flagged_anomalies', 0)} / {test_v.get('n_test_tiles', 0)} ({test_v.get('anomaly_rate_pct', 0.0):.2f}%)",
                            })
                        if test_rows:
                            st.markdown("<div class='de-card-header' style='margin-top: 1rem;'>Final Untouched Test Partitions</div>", unsafe_allow_html=True)
                            st.dataframe(pd.DataFrame(test_rows), use_container_width=True)
                    except Exception:
                        pass
            else:
                st.info("Run training to see results.")
        else:
            if st.button("Generate Data & Train", use_container_width=True):
                with st.spinner("Generating synthetic signals..."):
                    signals, labels, metadata_list = generate_dataset(
                        num_normal=num_normal,
                        num_rfi=num_rfi,
                        num_anomaly=num_anomaly,
                        sample_rate=sample_rate,
                        duration=duration,
                        base_seed=42,
                    )

                with st.spinner("Computing spectrograms..."):
                    specs = signals_to_spectrograms(
                        signals, sample_rate, n_fft, hop_length, SPEC_HEIGHT, SPEC_WIDTH
                    )

                example_cols = st.columns(4)
                indices = [0, num_normal, num_normal + num_rfi, len(signals) - 1]
                for i, idx in enumerate(indices):
                    with example_cols[i]:
                        fig = go.Figure(data=go.Heatmap(
                            z=specs[idx], colorscale=HEATMAP_COLORSCALE, showscale=False,
                        ))
                        fig.update_layout(
                            title=dict(text=labels[idx], font=dict(size=10, color=COLOR_MUTED)),
                            template="plotly_dark",
                            height=170, margin=dict(l=5, r=5, t=25, b=5),
                            xaxis=dict(visible=False), yaxis=dict(visible=False),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                normal_mask = labels == "normal_like"
                normal_specs = specs[normal_mask]

                progress = st.progress(0)
                status = st.empty()
                model, loss_history = train_autoencoder(
                    normal_specs, epochs=epochs, batch_size=batch_size,
                    lr=learning_rate, latent_dim=latent_dim,
                    progress_bar=progress, status_text=status,
                )

                st.plotly_chart(plot_loss_curve(loss_history), use_container_width=True)

                with st.spinner("Computing reconstruction errors..."):
                    errors = compute_reconstruction_errors(model, specs)

                threshold = float(np.percentile(errors[normal_mask], anomaly_pct))

                st.session_state["model"] = model
                st.session_state["specs"] = specs
                st.session_state["signals"] = signals
                st.session_state["labels"] = labels
                st.session_state["errors"] = errors
                st.session_state["threshold"] = threshold
                st.session_state["loss_history"] = loss_history
                st.session_state["metadata_list"] = metadata_list
                st.session_state["sample_rate"] = sample_rate

                st.toast("Training complete. Switch to Results tab.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Tab 3 — Results
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_results:
        if "Real" in data_mode:
            target_k = target_info["key"]
            summary_path = RESULTS_DIR / "test_summary_all_targets.json"
            rankings_path = RESULTS_DIR / f"test_rankings_{target_k}.csv"

            if not rankings_path.exists():
                rankings_path = RESULTS_DIR / "test_set_anomaly_rankings.csv"

            if not rankings_path.exists():
                st.info("Run training to see results.")
            else:
                test_summary = {}
                if summary_path.exists():
                    try:
                        with open(summary_path, "r", encoding="utf-8") as f:
                            all_summaries = json.load(f)
                        test_summary = all_summaries.get(target_k, {})
                    except Exception:
                        pass
                if not test_summary:
                    prim_summary = RESULTS_DIR / "test_set_evaluation_summary.json"
                    if prim_summary.exists():
                        try:
                            with open(prim_summary, "r", encoding="utf-8") as f:
                                test_summary = json.load(f)
                        except Exception:
                            pass

                df_rankings = pd.read_csv(rankings_path)

                base_thresh = float(test_summary.get("threshold", target_info.get("calibrated_threshold", 0.006484)))
                errors_arr = df_rankings["reconstruction_error"].values
                n_test = len(errors_arr)

                col_th1, col_th2 = st.columns([3, 1])
                with col_th1:
                    thresh = st.slider(
                        "Decision Threshold",
                        min_value=float(np.min(errors_arr)),
                        max_value=float(np.max(errors_arr)),
                        value=float(base_thresh),
                        step=0.0001,
                        format="%.6f",
                        help="MSE threshold separating normal radio background from candidate anomalies.",
                    )
                with col_th2:
                    if st.button("Reset Threshold", use_container_width=True):
                        thresh = base_thresh

                flagged_mask = errors_arr > thresh
                n_anom = int(np.sum(flagged_mask))
                max_err = float(np.max(errors_arr))
                rate_pct = float(n_anom / n_test * 100.0) if n_test > 0 else 0.0

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Holdout Tiles", f"{n_test:,}")
                col2.metric("Flagged Anomalies", f"{n_anom} ({rate_pct:.2f}%)")
                col3.metric("Threshold (99%)", f"{thresh:.6f}")
                col4.metric("Peak MSE", f"{max_err:.6f}")

                st.plotly_chart(
                    plot_real_error_distribution(errors_arr, thresh),
                    use_container_width=True,
                )

                df_rankings["is_anomaly"] = flagged_mask
                df_rankings["threshold"] = thresh

                st.markdown("<div class='de-card-header'>Top Ranked Anomaly Coordinates</div>", unsafe_allow_html=True)
                display_cols = [
                    "test_rank",
                    "reconstruction_error",
                    "time_offset_sec",
                    "center_freq_mhz",
                    "is_anomaly",
                    "t_index",
                    "f_index",
                ]
                avail_cols = [c for c in display_cols if c in df_rankings.columns]
                st.dataframe(df_rankings[avail_cols].head(50), use_container_width=True, height=260)

                st.markdown("<div class='de-card-header' style='margin-top: 1rem;'>Visual Inspection: Top Candidate Triplets</div>", unsafe_allow_html=True)
                anom_tabs = st.tabs([f"Candidate {i}" for i in range(1, 6)])
                prefix = target_info["fig_prefix"]
                for i, atab in enumerate(anom_tabs):
                    rank_num = i + 1
                    with atab:
                        fig_path = FIGURES_DIR / f"{prefix}_top_{rank_num}.png"
                        if fig_path.exists():
                            row = df_rankings.iloc[i]
                            st.image(str(fig_path), use_container_width=True)
                            st.caption(
                                f"Rank #{rank_num} | MSE: {row['reconstruction_error']:.6f} | "
                                f"Time: {row['time_offset_sec']:.2f}s | Center Freq: {row['center_freq_mhz']:.2f} MHz"
                            )
                        else:
                            st.info(f"Figure {fig_path.name} not found.")

                # Detection Sensitivity Benchmark Card
                inj_curve_path = FIGURES_DIR / "injection_recovery_curve.png"
                if inj_curve_path.exists():
                    st.markdown("<div class='de-card-header' style='margin-top: 1.5rem;'>Synthetic Signal Injection & Recovery Benchmark</div>", unsafe_allow_html=True)
                    st.image(str(inj_curve_path), use_container_width=True)
                    st.caption("Drifting carrier recovery rate across -5 dB to +35 dB SNR against Adaptive and Global thresholds.")

        else:
            if "errors" not in st.session_state:
                st.info("Run training to see results.")
            else:
                errors = st.session_state["errors"]
                labels = st.session_state["labels"]
                threshold = st.session_state["threshold"]
                signals = st.session_state["signals"]
                specs = st.session_state["specs"]
                sr = st.session_state["sample_rate"]

                flagged = errors > threshold
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Signals", len(errors))
                col2.metric("Flagged", int(flagged.sum()))
                col3.metric("Threshold", f"{threshold:.5f}")
                col4.metric("Max Error", f"{errors.max():.5f}")

                st.plotly_chart(
                    plot_error_distribution(errors, labels, threshold),
                    use_container_width=True,
                )

                st.markdown("<div class='de-card-header'>Category Statistics</div>", unsafe_allow_html=True)
                categories = np.unique(labels)
                rows = []
                for cat in categories:
                    mask = labels == cat
                    cat_errs = errors[mask]
                    rows.append({
                        "Category": cat,
                        "Count": int(mask.sum()),
                        "Mean Error": f"{cat_errs.mean():.6f}",
                        "Std Error": f"{cat_errs.std():.6f}",
                        "Min": f"{cat_errs.min():.6f}",
                        "Max": f"{cat_errs.max():.6f}",
                        "Flagged": int((cat_errs > threshold).sum()),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                top_k = st.slider("Top anomalies to inspect", 3, 20, 5)
                ranked = np.argsort(errors)[::-1][:top_k]

                for rank, idx in enumerate(ranked):
                    with st.expander(f"#{rank + 1} — {labels[idx]} | Error: {errors[idx]:.6f}"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.plotly_chart(plot_signal(signals[idx], sr, "Time Domain"), use_container_width=True)
                        with c2:
                            fig = go.Figure(data=go.Heatmap(z=specs[idx], colorscale=HEATMAP_COLORSCALE, showscale=False))
                            fig.update_layout(template="plotly_dark", height=240, margin=dict(l=10, r=10, t=10, b=10))
                            st.plotly_chart(fig, use_container_width=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Tab 4 — Compare
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_compare:
        if "Real" in data_mode:
            prefix = target_info["fig_prefix"]
            fig_top1 = FIGURES_DIR / f"{prefix}_top_1.png"
            fig_top2 = FIGURES_DIR / f"{prefix}_top_2.png"

            if fig_top1.exists():
                st.markdown(f"<div class='de-card-header'>{target_info['name']} — Candidate #1 Triplet Dynamic Spectrum</div>", unsafe_allow_html=True)
                st.image(str(fig_top1), use_container_width=True)

            if fig_top2.exists():
                st.markdown(f"<div class='de-card-header' style='margin-top: 1rem;'>{target_info['name']} — Candidate #2 Triplet Dynamic Spectrum</div>", unsafe_allow_html=True)
                st.image(str(fig_top2), use_container_width=True)

        else:
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                type_a = st.selectbox("Signal A Type", ["Normal-like", "RFI", "Anomaly"], key="cmp_a")
                seed_a = st.number_input("Seed A", 0, 999999, 10, key="seed_a")
            with comp_col2:
                type_b = st.selectbox("Signal B Type", ["Normal-like", "RFI", "Anomaly"], key="cmp_b")
                seed_b = st.number_input("Seed B", 0, 999999, 99, key="seed_b")

            if st.button("Compare", use_container_width=True):
                fn_map = {
                    "Normal-like": generate_normal_signal,
                    "RFI": generate_rfi_signal,
                    "Anomaly": generate_anomaly_signal,
                }
                sig_a, meta_a = fn_map[type_a](sample_rate, duration, seed=seed_a)
                sig_b, meta_b = fn_map[type_b](sample_rate, duration, seed=seed_b)

                st.plotly_chart(
                    plot_comparison(sig_a, sig_b, sample_rate, f"Signal A ({type_a})", f"Signal B ({type_b})"),
                    use_container_width=True,
                )

                col_sa, col_sb = st.columns(2)
                with col_sa:
                    fa, ta, pa = compute_spectrogram(sig_a, sample_rate, n_fft, hop_length)
                    st.plotly_chart(plot_spectrogram(fa, ta, pa, f"Signal A — {type_a}"), use_container_width=True)
                with col_sb:
                    fb, tb, pb = compute_spectrogram(sig_b, sample_rate, n_fft, hop_length)
                    st.plotly_chart(plot_spectrogram(fb, tb, pb, f"Signal B — {type_b}"), use_container_width=True)

                with st.expander("Metadata Comparison"):
                    mc1, mc2 = st.columns(2)
                    with mc1:
                        st.markdown(f"**Signal A ({type_a})**")
                        st.json(meta_a)
                    with mc2:
                        st.markdown(f"**Signal B ({type_b})**")
                        st.json(meta_b)

    # ── Subtle Instrument Footer ──
    st.markdown(
        "<div style='text-align: center; color: #334155; font-size: 0.75rem; letter-spacing: 0.08em; margin-top: 2rem; padding: 1rem 0; border-top: 1px solid rgba(255, 255, 255, 0.05);'>"
        "DEEP ECHO · INSTRUMENT ANOMALY DETECTION PIPELINE"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
