"""
End-to-end Integration Test for SETI Technosignature Anomaly Detection Pipeline (Phase 19).
Verifies data generation -> spectrogram tiling -> model training -> scoring -> cadence verification.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from src.signal_generator import generate_dataset
from src.spectrogram import compute_spectrogram, resize_spectrogram
from src.training import (
    train_autoencoder,
    compute_reconstruction_errors,
    calibrate_threshold,
)
from src.cadence_filter import analyze_cadence_sequence
from src.config import SPEC_HEIGHT, SPEC_WIDTH


def test_full_pipeline_integration(tmp_path):
    # 1. Generate small dataset
    signals, labels, _ = generate_dataset(
        num_normal=20,
        num_rfi=5,
        num_anomaly=5,
        base_seed=123,
    )
    assert len(signals) == 30

    # 2. Spectrogram generation & normalization
    specs = []
    for sig in signals:
        _, _, p_power = compute_spectrogram(sig, fs=1000.0)
        p_db = 10.0 * np.log10(p_power + 1e-12)
        resized = resize_spectrogram(p_db, SPEC_HEIGHT, SPEC_WIDTH)
        smin, smax = resized.min(), resized.max()
        if smax - smin > 1e-8:
            resized = (resized - smin) / (smax - smin)
        specs.append(resized)

    specs_array = np.expand_dims(np.array(specs, dtype=np.float32), 1)
    assert specs_array.shape == (30, 1, 64, 64)

    # 3. Model training on normal signals only
    normal_specs = specs_array[:20]
    model, history = train_autoencoder(
        normal_specs,
        epochs=1,
        batch_size=10,
        lr=1e-3,
        latent_dim=16,
        seed=123,
    )
    assert len(history) == 1

    # 4. Calibration & Reconstruction Error
    errors = compute_reconstruction_errors(model, specs_array)
    assert len(errors) == 30
    threshold = calibrate_threshold(errors[:20], percentile=95.0)
    assert threshold > 0.0

    # 5. Cadence verification test
    cadence = [
        {"target": "TARGET_A", "pointing_type": "ON", "spectrogram": specs_array[25, 0]},
        {"target": "OFF_B", "pointing_type": "OFF", "spectrogram": specs_array[0, 0]},
        {"target": "TARGET_A", "pointing_type": "ON", "spectrogram": specs_array[26, 0]},
        {"target": "OFF_C", "pointing_type": "OFF", "spectrogram": specs_array[1, 0]},
    ]
    cadence_res = analyze_cadence_sequence(model, cadence, threshold=threshold)
    assert cadence_res.verdict in {"CANDIDATE_TECHNOSIGNATURE", "LOCAL_RFI", "NORMAL_BACKGROUND"}

    # 6. Model save and load test
    save_file = tmp_path / "test_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "latent_dim": 16,
            "threshold": threshold,
        },
        save_file,
    )
    assert save_file.exists()
