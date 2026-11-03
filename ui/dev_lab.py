"""
Developer lab: the original synthetic-signal workflow (generate, train, score, compare).

Moved here unchanged from the first version of the dashboard so training stays
available under Advanced. Nothing in this module runs unless the user opens it
and presses an explicit button; the main investigation workflow never uses it.
"""


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
)
from ui.components import (
    style_plotly_figure,
    COLOR_ACCENT,
    COLOR_NORMAL,
    COLOR_RFI,
    COLOR_ANOMALY,
    HEATMAP_COLORSCALE,
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
# Lab layout (controls inline; nothing runs until a button is pressed)
# ──────────────────────────────────────────────
def render_synthetic_lab() -> None:
    st.caption(
        "Generates synthetic waveforms and trains a small autoencoder on them. This is a development tool for the model; "
        "it never touches your observations, reviews or the saved checkpoints."
    )
    c1, c2, c3, c4 = st.columns(4)
    sample_rate = c1.slider("Sample rate (Hz)", 500.0, 5000.0, DEFAULT_SAMPLE_RATE, step=100.0, key="lab_sr")
    duration = c2.slider("Duration (s)", 0.5, 3.0, DEFAULT_DURATION, step=0.25, key="lab_dur")
    n_fft = c3.select_slider("FFT size", options=[64, 128, 256, 512], value=DEFAULT_N_FFT, key="lab_fft")
    hop_length = c4.select_slider("Hop length", options=[16, 32, 64, 128], value=DEFAULT_HOP_LENGTH, key="lab_hop")
    c1, c2, c3 = st.columns(3)
    num_normal = c1.number_input("Normal signals", 20, 500, 200, step=10, key="lab_nn")
    num_rfi = c2.number_input("RFI signals", 5, 200, 50, step=5, key="lab_nr")
    num_anomaly = c3.number_input("Anomaly signals", 5, 200, 50, step=5, key="lab_na")
    c1, c2, c3, c4, c5 = st.columns(5)
    latent_dim = c1.slider("Latent dim", 8, 128, LATENT_DIM, step=8, key="lab_ld")
    epochs = c2.slider("Epochs", 5, 100, DEFAULT_EPOCHS, key="lab_ep")
    batch_size = c3.select_slider("Batch size", options=[8, 16, 32, 64, 128], value=DEFAULT_BATCH_SIZE, key="lab_bs")
    learning_rate = c4.select_slider("Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=DEFAULT_LEARNING_RATE, key="lab_lr")
    anomaly_pct = c5.slider("Threshold percentile", 90.0, 99.9, DEFAULT_ANOMALY_PERCENTILE, step=0.5, key="lab_pct")

    tab_gen, tab_train, tab_res, tab_cmp = st.tabs(["Generate", "Train", "Results", "Compare"])

    with tab_gen:
        col_type, col_seed = st.columns([2, 1])
        sig_type = col_type.selectbox("Signal type", ["Normal-like", "RFI (Interference)", "Anomaly (Unusual)"], key="lab_sigtype")
        seed = col_seed.number_input("Random seed", 0, 999999, 42, key="lab_seed")
        if st.button("Generate signal", key="lab_gen"):
            gen_fn = {"Normal-like": generate_normal_signal, "RFI (Interference)": generate_rfi_signal, "Anomaly (Unusual)": generate_anomaly_signal}[sig_type]
            sig, meta = gen_fn(sample_rate, duration, seed=seed)
            col_sig, col_spec = st.columns(2)
            col_sig.plotly_chart(plot_signal(sig, sample_rate, f"{sig_type}: time domain"), width="stretch")
            freqs, times, power_dB = compute_spectrogram(sig, sample_rate, n_fft, hop_length)
            col_spec.plotly_chart(plot_spectrogram(freqs, times, power_dB, f"{sig_type}: spectrogram"), width="stretch")
            with st.expander("Signal metadata"):
                st.json(meta)

    with tab_train:
        if st.button("Generate data and train", key="lab_train"):
            with st.spinner("Generating synthetic signals…"):
                signals, labels, metadata_list = generate_dataset(
                    num_normal=num_normal, num_rfi=num_rfi, num_anomaly=num_anomaly,
                    sample_rate=sample_rate, duration=duration, base_seed=42,
                )
            with st.spinner("Computing spectrograms…"):
                specs = signals_to_spectrograms(signals, sample_rate, n_fft, hop_length, SPEC_HEIGHT, SPEC_WIDTH)
            normal_specs = specs[labels == "normal_like"]
            progress = st.progress(0)
            status = st.empty()
            model, loss_history = train_autoencoder(
                normal_specs, epochs=epochs, batch_size=batch_size, lr=learning_rate, latent_dim=latent_dim,
                progress_bar=progress, status_text=status,
            )
            st.plotly_chart(plot_loss_curve(loss_history), width="stretch")
            with st.spinner("Computing reconstruction errors…"):
                errors = compute_reconstruction_errors(model, specs)
            threshold = float(np.percentile(errors[labels == "normal_like"], anomaly_pct))
            st.session_state.update({
                "lab_model": model, "lab_specs": specs, "lab_signals": signals, "lab_labels": labels,
                "lab_errors": errors, "lab_threshold": threshold, "lab_sr_used": sample_rate,
            })
            st.toast("Training complete. See the Results tab.")

    with tab_res:
        if "lab_errors" not in st.session_state:
            st.info("Train a model first.")
        else:
            errors, labels = st.session_state["lab_errors"], st.session_state["lab_labels"]
            threshold, signals = st.session_state["lab_threshold"], st.session_state["lab_signals"]
            specs, sr = st.session_state["lab_specs"], st.session_state["lab_sr_used"]
            flagged = errors > threshold
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total signals", len(errors))
            m2.metric("Flagged", int(flagged.sum()))
            m3.metric("Threshold", f"{threshold:.5f}")
            m4.metric("Max error", f"{errors.max():.5f}")
            st.plotly_chart(plot_error_distribution(errors, labels, threshold), width="stretch")
            rows = []
            for cat in np.unique(labels):
                mask = labels == cat
                e = errors[mask]
                rows.append({"Category": cat, "Count": int(mask.sum()), "Mean error": f"{e.mean():.6f}", "Std": f"{e.std():.6f}",
                             "Min": f"{e.min():.6f}", "Max": f"{e.max():.6f}", "Flagged": int((e > threshold).sum())})
            st.dataframe(pd.DataFrame(rows), width="stretch")
            top_k = st.slider("Top signals to inspect", 3, 20, 5, key="lab_topk")
            for rank, idx in enumerate(np.argsort(errors)[::-1][:top_k]):
                with st.expander(f"#{rank + 1}: {labels[idx]} | error {errors[idx]:.6f}"):
                    a, b = st.columns(2)
                    a.plotly_chart(plot_signal(signals[idx], sr, "Time domain"), width="stretch")
                    fig = go.Figure(data=go.Heatmap(z=specs[idx], colorscale=HEATMAP_COLORSCALE, showscale=False))
                    fig.update_layout(template="plotly_dark", height=240, margin=dict(l=10, r=10, t=10, b=10))
                    b.plotly_chart(fig, width="stretch")

    with tab_cmp:
        ca, cb = st.columns(2)
        type_a = ca.selectbox("Signal A type", ["Normal-like", "RFI", "Anomaly"], key="lab_cmp_a")
        seed_a = ca.number_input("Seed A", 0, 999999, 10, key="lab_seed_a")
        type_b = cb.selectbox("Signal B type", ["Normal-like", "RFI", "Anomaly"], key="lab_cmp_b")
        seed_b = cb.number_input("Seed B", 0, 999999, 99, key="lab_seed_b")
        if st.button("Compare synthetic signals", key="lab_compare"):
            fn_map = {"Normal-like": generate_normal_signal, "RFI": generate_rfi_signal, "Anomaly": generate_anomaly_signal}
            sig_a, _ = fn_map[type_a](sample_rate, duration, seed=seed_a)
            sig_b, _ = fn_map[type_b](sample_rate, duration, seed=seed_b)
            st.plotly_chart(plot_comparison(sig_a, sig_b, sample_rate, f"Signal A ({type_a})", f"Signal B ({type_b})"), width="stretch")
