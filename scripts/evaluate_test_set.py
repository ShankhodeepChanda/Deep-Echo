"""
Evaluate Trained CNN Autoencoder on the 30% Holdout Test Set.

- Reconstructs the exact 70/30 train/test split from seed=42.
- Evaluates reconstruction errors on all 24,024 unseen test tiles.
- Flags tiles exceeding the calibrated anomaly threshold (0.006484).
- Saves a 3-panel figure (Original, Reconstructed, Residual Error) for the top K anomalies.
- Computes exact observation time (seconds) and center frequency (MHz) for each anomaly.
- Exports CSV and JSON reports to artifacts/results/.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

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
    SPEC_HEIGHT,
    SPEC_WIDTH,
)
from src.models.cnn_autoencoder import CNNAutoencoder
from src.training import compute_reconstruction_errors
from src.filterbank_reader import load_filterbank_waterfall, read_filterbank_header

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_test")


def extract_tiles_with_coords(waterfall: np.ndarray, tile_h: int = 64, tile_w: int = 64):
    """Extract 64x64 tiles while retaining (time_idx, freq_idx) coordinates."""
    T, F = waterfall.shape
    tiles = []
    coords = []

    for t_idx in range(0, T - tile_h + 1, tile_h):
        for f_idx in range(0, F - tile_w + 1, tile_w):
            tile = waterfall[t_idx : t_idx + tile_h, f_idx : f_idx + tile_w].copy()
            t_min, t_max = tile.min(), tile.max()
            if t_max - t_min > 1e-8:
                tile = (tile - t_min) / (t_max - t_min)
            else:
                tile = np.zeros_like(tile)
            tiles.append(tile)
            coords.append((t_idx, f_idx))

    tiles_arr = np.expand_dims(np.array(tiles, dtype=np.float32), 1)
    return tiles_arr, coords


def plot_triplet_anomaly(original: np.ndarray, reconstructed: np.ndarray, residual: np.ndarray, title: str, output_path: Path):
    """Save 3-panel comparison: Original, Reconstruction, Residual Error."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    im0 = axes[0].imshow(original, origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_title("Original Observation", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Frequency Channel")
    axes[0].set_ylabel("Time Bin")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(reconstructed, origin="lower", aspect="auto", cmap="viridis")
    axes[1].set_title("Autoencoder Reconstruction", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Frequency Channel")
    axes[1].set_ylabel("Time Bin")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(residual, origin="lower", aspect="auto", cmap="inferno")
    axes[2].set_title("Residual Error (|Orig - Recon|)", fontsize=11, fontweight="bold", color="darkred")
    axes[2].set_xlabel("Frequency Channel")
    axes[2].set_ylabel("Time Bin")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate 30% Test Set for Top Anomalies")
    parser.add_argument(
        "--source",
        type=str,
        default="data/raw/guppi_58212_57774_010485_G12.72+6.26_0001.0001.fil",
        help="Path to observation file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODELS_DIR / "cnn_autoencoder.pt"),
        help="Path to trained model",
    )
    parser.add_argument("--train-split", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5, help="Number of top anomalies to plot")
    args = parser.parse_args()

    # Load file & header
    fpath = Path(args.source)
    if not fpath.exists():
        raise FileNotFoundError(f"Source file not found: {fpath}")

    logger.info(f"Reading observation: {fpath.name}...")
    header, _ = read_filterbank_header(fpath)
    fch1 = float(header.get("fch1", 1359.75))
    foff = float(header.get("foff", -0.4375))
    tsamp = float(header.get("tsamp", 0.00029257))

    data = load_filterbank_waterfall(fpath)
    all_tiles, all_coords = extract_tiles_with_coords(data["waterfall"], tile_h=SPEC_HEIGHT, tile_w=SPEC_WIDTH)
    n_samples = len(all_tiles)

    # Recreate the exact 70/30 split
    indices = np.random.RandomState(args.seed).permutation(n_samples)
    split_idx = max(int(n_samples * args.train_split), 1)
    val_indices = indices[split_idx:]

    test_tiles = all_tiles[val_indices]
    test_coords = [all_coords[idx] for idx in val_indices]
    n_test = len(test_tiles)
    logger.info(f"Loaded {n_test} holdout test tiles (unseen 30% split).")

    # Load model
    device = "cpu"
    ckpt = torch.load(args.model_path, map_location=device)
    latent_dim = ckpt.get("latent_dim", 48)
    threshold = float(ckpt.get("threshold", 0.006484))

    model = CNNAutoencoder(input_size=(1, SPEC_HEIGHT, SPEC_WIDTH), latent_dim=latent_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Compute errors on test set
    logger.info("Computing reconstruction errors across test set...")
    val_errors = compute_reconstruction_errors(model, test_tiles, batch_size=128, device=device)

    # Sort anomalies descending
    ranked_order = np.argsort(val_errors)[::-1]
    anomalous_mask = val_errors > threshold
    total_anomalies = int(np.sum(anomalous_mask))

    logger.info(f"Evaluation complete. Found {total_anomalies} / {n_test} anomalous tiles above threshold {threshold:.6f} ({total_anomalies / n_test * 100:.2f}%)")

    # Export structured CSV
    records = []
    for rank, orig_pos in enumerate(ranked_order):
        err = float(val_errors[orig_pos])
        t_idx, f_idx = test_coords[orig_pos]
        time_sec = t_idx * tsamp
        center_freq_mhz = fch1 + (f_idx + SPEC_WIDTH / 2.0) * foff

        records.append({
            "test_rank": rank + 1,
            "test_tile_id": f"test_{orig_pos:05d}",
            "reconstruction_error": err,
            "is_anomaly": bool(err > threshold),
            "threshold": threshold,
            "time_offset_sec": round(time_sec, 3),
            "center_freq_mhz": round(center_freq_mhz, 3),
            "t_index": t_idx,
            "f_index": f_idx,
        })

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "test_set_anomaly_rankings.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved rankings table to {csv_path}")

    # Generate Top K Figures
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top_k_indices = ranked_order[: args.top_k]

    with torch.no_grad():
        for rank, pos in enumerate(top_k_indices):
            tile_4d = torch.from_numpy(test_tiles[pos : pos + 1]).to(device)
            recon_4d = model(tile_4d)
            
            orig_2d = tile_4d.squeeze().cpu().numpy()
            recon_2d = recon_4d.squeeze().cpu().numpy()
            residual_2d = np.abs(orig_2d - recon_2d)

            err = float(val_errors[pos])
            t_idx, f_idx = test_coords[pos]
            t_sec = t_idx * tsamp
            freq_mhz = fch1 + (f_idx + SPEC_WIDTH / 2.0) * foff

            title = (
                f"Test Anomaly #{rank + 1} | MSE: {err:.6f} (Threshold: {threshold:.6f})\n"
                f"Time: {t_sec:.2f}s | Center Freq: {freq_mhz:.2f} MHz | Source: Parkes L-Band"
            )
            out_img = FIGURES_DIR / f"test_anomaly_top_{rank + 1}.png"
            plot_triplet_anomaly(orig_2d, recon_2d, residual_2d, title, out_img)
            logger.info(f"Saved top anomaly #{rank + 1} figure: {out_img.name}")

    # Save summary JSON
    summary = {
        "dataset": fpath.name,
        "n_test_tiles": n_test,
        "threshold": threshold,
        "anomalies_detected": total_anomalies,
        "anomaly_rate": float(total_anomalies / n_test),
        "mean_error": float(np.mean(val_errors)),
        "std_error": float(np.std(val_errors)),
        "min_error": float(np.min(val_errors)),
        "max_error": float(np.max(val_errors)),
        "top_5_errors": [float(val_errors[p]) for p in top_k_indices],
    }
    summary_path = RESULTS_DIR / "test_set_evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved evaluation summary to {summary_path}")


if __name__ == "__main__":
    main()
