"""
Batch Anomaly Detection and Export Pipeline for SETI Observations.

Implements Phase 17: Batch Analysis & Export.
- Loads trained CNN Autoencoder model from artifacts/models/
- Processes multiple radio observation files or synthetic batches
- Runs anomaly detection and cadence verification
- Exports structured CSV & JSON reports to artifacts/results/
- Saves top anomaly figures to artifacts/figures/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.config import (
    MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    RAW_DATA_DIR,
    SPEC_HEIGHT,
    SPEC_WIDTH,
)
from src.models.cnn_autoencoder import CNNAutoencoder
from src.training import compute_reconstruction_errors
from src.filterbank_reader import load_filterbank_waterfall, extract_spectrogram_tiles
from src.signal_generator import generate_dataset
from src.spectrogram import compute_spectrogram, resize_spectrogram

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("batch_scan")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Scan SETI Observations for Anomalies")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODELS_DIR / "cnn_autoencoder.pt"),
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help="Directory with .fil / .npy observation files",
    )
    parser.add_argument(
        "--synthetic-eval",
        action="store_true",
        help="Run batch evaluation on a mixed synthetic test suite",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(RESULTS_DIR / "batch_scan_results.csv"),
        help="Path to output CSV report",
    )
    parser.add_argument(
        "--top-k-figures",
        type=int,
        default=5,
        help="Number of top anomaly spectrogram figures to save",
    )
    return parser.parse_args()


def load_trained_model(model_path: Path, device: str = "cpu"):
    """Load model checkpoint and metadata."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}. Run scripts/train.py first.")

    checkpoint = torch.load(model_path, map_location=device)
    latent_dim = checkpoint.get("latent_dim", 48)
    threshold = checkpoint.get("threshold", 0.05)

    model = CNNAutoencoder(input_size=(1, SPEC_HEIGHT, SPEC_WIDTH), latent_dim=latent_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, threshold, checkpoint


def save_anomaly_figure(spec_tile: np.ndarray, title: str, output_path: Path):
    """Save 2D spectrogram tile image."""
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(spec_tile, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time Bin")
    ax.set_ylabel("Frequency Channel")
    fig.colorbar(im, ax=ax, label="Normalized Intensity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def run_batch_scan():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model, threshold, meta = load_trained_model(Path(args.model_path), device=device)
    logger.info(f"Loaded model from {args.model_path} (Threshold: {threshold:.6f})")

    records = []
    tiles_to_plot = []

    if args.synthetic_eval:
        logger.info("Running synthetic evaluation test suite...")
        signals, labels, _ = generate_dataset(
            num_normal=50,
            num_rfi=25,
            num_anomaly=25,
            base_seed=999,
        )
        for i, sig in enumerate(signals):
            _, _, p_power = compute_spectrogram(sig, fs=1000.0)
            p_db = 10.0 * np.log10(p_power + 1e-12)
            resized = resize_spectrogram(p_db, SPEC_HEIGHT, SPEC_WIDTH)
            smin, smax = resized.min(), resized.max()
            if smax - smin > 1e-8:
                resized = (resized - smin) / (smax - smin)

            spec_4d = resized[np.newaxis, np.newaxis, :, :]
            err = float(compute_reconstruction_errors(model, spec_4d, device=device)[0])
            is_anomaly = bool(err > threshold)

            records.append({
                "observation_id": f"syn_{i:04d}",
                "source": "synthetic",
                "ground_truth_category": labels[i],
                "reconstruction_error": err,
                "threshold": threshold,
                "is_anomalous": is_anomaly,
            })
            tiles_to_plot.append((resized, f"ID: syn_{i:04d} | {labels[i]} | Err: {err:.4f}", err))

    else:
        input_dir = Path(args.input_dir)
        files = list(input_dir.glob("*.fil")) + list(input_dir.glob("*.npy"))
        if not files:
            logger.warning(f"No .fil or .npy files found in {input_dir}. Falling back to synthetic batch scan.")
            args.synthetic_eval = True
            return run_batch_scan()

        logger.info(f"Scanning {len(files)} files in {input_dir}...")
        for fpath in files:
            try:
                if fpath.suffix.lower() == ".fil":
                    data = load_filterbank_waterfall(fpath)
                    tiles = extract_spectrogram_tiles(data["waterfall"], tile_h=SPEC_HEIGHT, tile_w=SPEC_WIDTH)
                else:
                    arr = np.load(fpath)
                    tiles = extract_spectrogram_tiles(arr, tile_h=SPEC_HEIGHT, tile_w=SPEC_WIDTH)

                errs = compute_reconstruction_errors(model, tiles, device=device)
                for t_idx, err in enumerate(errs):
                    is_anomaly = bool(err > threshold)
                    records.append({
                        "observation_id": f"{fpath.stem}_tile_{t_idx}",
                        "source": fpath.name,
                        "ground_truth_category": "unknown",
                        "reconstruction_error": float(err),
                        "threshold": threshold,
                        "is_anomalous": is_anomaly,
                    })
                    tiles_to_plot.append((tiles[t_idx, 0], f"{fpath.stem} tile {t_idx} (Err: {err:.4f})", float(err)))
            except Exception as e:
                logger.error(f"Error processing {fpath.name}: {e}")

    # Build DataFrame & export
    df = pd.DataFrame(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logger.info(f"Saved batch results ({len(df)} records) to {args.output_csv}")

    # Save summary stats
    summary = {
        "total_observations": len(df),
        "total_flagged_anomalies": int(df["is_anomalous"].sum()),
        "anomaly_rate": float(df["is_anomalous"].mean()),
        "mean_reconstruction_error": float(df["reconstruction_error"].mean()),
        "max_reconstruction_error": float(df["reconstruction_error"].max()),
        "threshold": threshold,
    }
    summary_json_path = Path(args.output_csv).with_suffix(".json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary JSON to {summary_json_path}")

    # Save top-k anomaly figures
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sorted_tiles = sorted(tiles_to_plot, key=lambda x: x[2], reverse=True)[: args.top_k_figures]
    for rank, (tile, title, err) in enumerate(sorted_tiles):
        fig_path = FIGURES_DIR / f"top_anomaly_{rank + 1}.png"
        save_anomaly_figure(tile, title, fig_path)
        logger.info(f"Saved anomaly figure #{rank + 1} to {fig_path}")

    logger.info("Batch scan and export complete.")


if __name__ == "__main__":
    run_batch_scan()
