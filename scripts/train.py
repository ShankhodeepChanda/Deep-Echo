"""
Command-line training script for SETI CNN Autoencoder Anomaly Detection.

Usage:
    python scripts/train.py --source synthetic --epochs 20 --batch-size 32
    python scripts/train.py --source data/raw/obs.fil --epochs 30
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
import torch

from src.config import (
    MODELS_DIR,
    RESULTS_DIR,
    LATENT_DIM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_EPOCHS,
    DEFAULT_ANOMALY_PERCENTILE,
    SPEC_HEIGHT,
    SPEC_WIDTH,
)
from src.signal_generator import generate_dataset
from src.spectrogram import compute_spectrogram, resize_spectrogram
from src.training import (
    train_autoencoder,
    compute_reconstruction_errors,
    calibrate_threshold,
)
from src.filterbank_reader import load_filterbank_waterfall, extract_spectrogram_tiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SETI CNN Autoencoder")
    parser.add_argument(
        "--source",
        type=str,
        default="synthetic",
        help="Data source: 'synthetic' or path to a .fil / .npy / .npz observation file",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM, help="Latent bottleneck size")
    parser.add_argument("--percentile", type=float, default=DEFAULT_ANOMALY_PERCENTILE, help="Threshold percentile")
    parser.add_argument("--train-split", type=float, default=0.70, help="Train/test split ratio (default: 0.70 for 70/30)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-name", type=str, default="cnn_autoencoder", help="Base name for saved model")
    return parser.parse_args()


def get_training_data(source: str, seed: int = 42) -> np.ndarray:
    """Prepare training spectrogram tiles."""
    if source.lower() == "synthetic":
        logger.info("Generating synthetic normal training signals...")
        signals, labels, _ = generate_dataset(
            num_normal=250,
            num_rfi=0,
            num_anomaly=0,
            base_seed=seed,
        )
        specs = []
        for sig in signals:
            _, _, p_power = compute_spectrogram(sig, fs=1000.0)
            p_db = 10.0 * np.log10(p_power + 1e-12)
            resized = resize_spectrogram(p_db, SPEC_HEIGHT, SPEC_WIDTH)
            smin, smax = resized.min(), resized.max()
            if smax - smin > 1e-8:
                resized = (resized - smin) / (smax - smin)
            specs.append(resized)
        return np.expand_dims(np.array(specs, dtype=np.float32), 1)

    file_path = Path(source)
    if not file_path.exists():
        raise FileNotFoundError(f"Training source file not found: {file_path}")

    logger.info(f"Loading training data from {file_path}...")
    if file_path.suffix.lower() == ".fil":
        data = load_filterbank_waterfall(file_path)
        tiles = extract_spectrogram_tiles(data["waterfall"], tile_h=SPEC_HEIGHT, tile_w=SPEC_WIDTH)
        return tiles
    elif file_path.suffix.lower() == ".npy":
        arr = np.load(file_path)
        if arr.ndim == 2:
            return extract_spectrogram_tiles(arr, tile_h=SPEC_HEIGHT, tile_w=SPEC_WIDTH)
        elif arr.ndim == 3:
            return np.expand_dims(arr.astype(np.float32), 1)
        elif arr.ndim == 4:
            return arr.astype(np.float32)
        else:
            raise ValueError(f"Unexpected array shape in {file_path}: {arr.shape}")
    else:
        raise ValueError(f"Unsupported file format for direct training: {file_path.suffix}")


def main():
    args = parse_args()

    # Load data
    train_specs = get_training_data(args.source, seed=args.seed)
    n_samples = len(train_specs)
    logger.info(f"Loaded {n_samples} training tiles with shape {train_specs.shape}")

    # Train-val split (e.g. 70/30)
    indices = np.random.RandomState(args.seed).permutation(n_samples)
    split_idx = max(int(n_samples * args.train_split), 1)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    train_data = train_specs[train_idx]
    val_data = train_specs[val_idx] if len(val_idx) > 0 else train_data
    logger.info(
        f"Split {n_samples} tiles into {len(train_data)} train ({args.train_split*100:.0f}%) "
        f"and {len(val_data)} test/val ({(1 - args.train_split)*100:.0f}%)"
    )

    # Train model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting training on {device} ({len(train_data)} train, {len(val_data)} val)...")

    model, loss_history = train_autoencoder(
        train_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        device=device,
        seed=args.seed,
    )

    # Compute validation reconstruction errors and calibrate threshold
    val_errors = compute_reconstruction_errors(model, val_data, batch_size=args.batch_size, device=device)
    threshold = calibrate_threshold(val_errors, percentile=args.percentile)

    logger.info(f"Validation Mean Error: {float(np.mean(val_errors)):.6f}")
    logger.info(f"Calibrated Threshold ({args.percentile}th percentile): {threshold:.6f}")

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{args.output_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "latent_dim": args.latent_dim,
            "threshold": threshold,
            "config": vars(args),
        },
        model_path,
    )
    logger.info(f"Model checkpoint saved to {model_path}")

    summary = {
        "source": args.source,
        "n_samples": n_samples,
        "n_train": len(train_data),
        "n_val": len(val_data),
        "train_split": args.train_split,
        "epochs": args.epochs,
        "latent_dim": args.latent_dim,
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "calibrated_threshold": threshold,
        "val_mean_error": float(np.mean(val_errors)),
        "val_std_error": float(np.std(val_errors)),
        "loss_history": loss_history,
    }
    summary_path = RESULTS_DIR / f"{args.output_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training summary saved to {summary_path}")


if __name__ == "__main__":
    main()
