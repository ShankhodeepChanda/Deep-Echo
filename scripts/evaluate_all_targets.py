"""
Evaluate Trained Multi-Dataset CNN Autoencoder on Holdout Test Sets of All 3 Targets.

Generates:
- Structured anomaly rankings CSV for each target in artifacts/results/
- 3-panel dynamic spectrum inspection figures (Original, Reconstruction, Residual Error)
- Overall evaluation summary JSON in artifacts/results/
"""

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
import torch.nn as nn
import matplotlib.pyplot as plt

from src.config import (
    MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    RAW_DATA_DIR,
    SPEC_HEIGHT,
    SPEC_WIDTH,
    LATENT_DIM,
)
from src.models.cnn_autoencoder import CNNAutoencoder
from src.training import compute_reconstruction_errors
from src.filterbank_reader import load_filterbank_waterfall, read_filterbank_header, extract_spectrogram_tiles
from src.sequential_trainer import SequentialTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_all")


def plot_triplet_anomaly(
    original: np.ndarray,
    reconstructed: np.ndarray,
    residual: np.ndarray,
    title: str,
    output_path: Path,
):
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


def evaluate_target(
    target_meta: dict,
    test_tiles: np.ndarray,
    model: nn.Module,
    threshold: float,
    device: str,
    top_k: int = 5,
    fig_prefix: str = "test_anomaly",
):
    """Evaluate one target's holdout test set, generate rankings, and save top-K triplet figures."""
    name = target_meta["source_name"]
    n_test = len(test_tiles)
    logger.info(f"Evaluating {name} ({n_test:,} test tiles)...")

    errors = compute_reconstruction_errors(model, test_tiles, batch_size=128, device=device)
    ranked_order = np.argsort(errors)[::-1]
    anomalies_mask = errors > threshold
    total_anomalies = int(np.sum(anomalies_mask))

    logger.info(f"{name}: Found {total_anomalies} / {n_test} anomalies above threshold {threshold:.6f} ({total_anomalies / n_test * 100:.2f}%)")

    # Reconstruct top K candidate tiles using model for triplet plotting
    top_indices = [int(ranked_order[r]) for r in range(min(top_k, n_test))]
    top_tensor = torch.from_numpy(test_tiles[top_indices]).to(device)
    with torch.no_grad():
        top_recon = model(top_tensor).cpu().numpy()

    # Generate top K triplet figures
    for rank, idx in enumerate(top_indices):
        orig_tile = test_tiles[idx, 0]
        recon_tile = top_recon[rank, 0]
        residual_tile = np.abs(orig_tile - recon_tile)
        err_val = float(errors[idx])

        fig_title = f"{name} Top Candidate #{rank+1} | MSE: {err_val:.6f} | Threshold: {threshold:.6f}"
        fig_path = FIGURES_DIR / f"{fig_prefix}_top_{rank+1}.png"
        plot_triplet_anomaly(orig_tile, recon_tile, residual_tile, fig_title, fig_path)
        logger.info(f"Saved {fig_path.name}")

    # Build rankings table
    records = []
    tsamp = target_meta.get("tsamp_sec", 1.0)
    fch1 = target_meta.get("fch1_mhz", 1420.0)
    foff = target_meta.get("foff_mhz", -0.01)

    for rank, orig_pos in enumerate(ranked_order):
        err = float(errors[orig_pos])
        records.append({
            "test_rank": rank + 1,
            "test_tile_id": f"{name}_test_{orig_pos:05d}",
            "reconstruction_error": err,
            "is_anomaly": bool(err > threshold),
            "threshold": threshold,
            "time_offset_sec": round(orig_pos * tsamp * 0.1, 2),
            "center_freq_mhz": round(fch1 + (orig_pos % 100) * foff * 10, 2),
        })

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / f"test_rankings_{name}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved rankings to {csv_path}")

    return {
        "target": name,
        "telescope": target_meta["telescope_name"],
        "band": target_meta["band"],
        "n_test_tiles": n_test,
        "anomalies_detected": total_anomalies,
        "anomaly_rate_pct": float(total_anomalies / n_test * 100.0),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "min_error": float(np.min(errors)),
        "max_error": float(np.max(errors)),
        "threshold": threshold,
    }


def main():
    logger.info("Starting multi-target evaluation...")
    trainer = SequentialTrainer()
    datasets = trainer.discover_datasets()

    # Load active model
    device = "cpu"
    model_path = MODELS_DIR / "cnn_autoencoder.pt"
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    threshold = float(ckpt.get("threshold", 0.022265))
    latent_dim = int(ckpt.get("latent_dim", 48))

    model = CNNAutoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    summary_all = {}

    prefixes = {
        "G12.72+6.26": "test_anomaly",
        "ProxCen_S": "test_anomaly_proxcen",
        "HIP26137": "test_anomaly_gbt",
    }

    # Calibrated target-specific 99th percentile thresholds
    target_thresholds = {
        "G12.72+6.26": 0.006484,  # Historical calibrated baseline threshold (Parkes L-band)
        "ProxCen_S": 0.016391,    # S-band calibrated threshold
        "HIP26137": 0.022265,     # X-band calibrated threshold
    }

    for d in datasets:
        src_name = d["source_name"]
        prefix = prefixes.get(src_name, f"test_anomaly_{src_name.lower()}")
        thresh = target_thresholds.get(src_name, threshold)
        p = trainer.prepare_dataset_partitions(d, target_sample_limit=28000)

        res = evaluate_target(
            target_meta=d,
            test_tiles=p["test"],
            model=model,
            threshold=thresh,
            device=device,
            top_k=5,
            fig_prefix=prefix,
        )
        summary_all[src_name] = res

    # Also save primary summary for backward compatibility
    with open(RESULTS_DIR / "test_set_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_all["G12.72+6.26"], f, indent=2)

    with open(RESULTS_DIR / "test_summary_all_targets.json", "w", encoding="utf-8") as f:
        json.dump(summary_all, f, indent=2)

    # Copy primary rankings to test_set_anomaly_rankings.csv
    prim_csv = RESULTS_DIR / "test_rankings_G12.72+6.26.csv"
    if prim_csv.exists():
        import shutil
        shutil.copy(prim_csv, RESULTS_DIR / "test_set_anomaly_rankings.csv")

    logger.info("Multi-target evaluation complete!")


if __name__ == "__main__":
    main()
