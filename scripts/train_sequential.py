"""
Command-Line Interface for Sequential Multi-Dataset Fine-Tuning.

Usage:
    python scripts/train_sequential.py --epochs 10 --batch-size 128
    python scripts/train_sequential.py --epochs 5 --retrain-stage-1
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sequential_trainer import SequentialTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_sequential")


def parse_args():
    parser = argparse.ArgumentParser(description="Sequential Multi-Dataset Fine-Tuning for SETI CNN Autoencoder")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs per dataset stage")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for fine-tuning")
    parser.add_argument("--replay-ratio", type=float, default=0.10, help="Fraction of previous training samples to replay")
    parser.add_argument("--retrain-stage-1", action="store_true", help="Force retraining Stage 1 rather than reusing C1")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "artifacts" / "checkpoints" / "sequential_training"))
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible partitions")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing Sequential Trainer...")

    trainer = SequentialTrainer(
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )

    manifest = trainer.run(
        epochs_per_stage=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        replay_ratio=args.replay_ratio,
        reuse_stage_1_checkpoint=not args.retrain_stage_1,
    )

    print("\n" + "=" * 70)
    print("SEQUENTIAL MULTI-DATASET FINE-TUNING COMPLETE")
    print("=" * 70)
    print(f"Total Stages Completed: {len(manifest['stage_summaries'])}")
    print(f"Final Recalibrated Anomaly Threshold: {manifest['final_recalibrated_threshold']:.6f} (99th %tile)")
    print(f"Historical Parkes Single-Dataset Threshold was: {manifest['historical_threshold']:.6f}")
    print(f"Manifest written to: {args.output_dir}/experiment_manifest.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
