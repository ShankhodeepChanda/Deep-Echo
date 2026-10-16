"""
Generate and save demo synthetic datasets for the SETI Anomaly Detection project.
"""

import numpy as np
from pathlib import Path
from src import signal_generator
from src.config import SYNTHETIC_DATA_DIR

def main():
    print("Generating demo synthetic dataset...")
    signals, labels, metadata = signal_generator.generate_dataset(
        num_normal=200,
        num_rfi=50,
        num_anomaly=50,
        sample_rate=1000.0,
        duration=1.0,
        base_seed=42
    )

    output_path = SYNTHETIC_DATA_DIR / "demo_dataset.npz"
    np.savez(
        output_path,
        signals=signals,
        labels=labels
    )
    print(f"Saved demo dataset with {len(signals)} observations to {output_path}")

if __name__ == "__main__":
    main()
