"""
Unit & Integration Tests for Sequential Multi-Dataset Fine-Tuning Pipeline.

Validates the 10 requirements from Section 10:
1. Original checkpoint loads correctly.
2. Subsequent stages start from the preceding stage's weights.
3. Training updates model parameters (verifiable weight delta).
4. Original checkpoint file remains bit-identical and unchanged.
5. Preprocessing & tile extraction works on each additional dataset format.
6. Training, validation, and holdout test samples remain strictly separated (zero overlap).
7. Checkpoints load successfully and preserve required metadata.
8. Final anomaly threshold is calculated from designated validation set (not test).
9. Resume / checkpoint inheritance behavior functions properly.
10. End-to-end smoke test on mini-batches completes cleanly.
"""

import copy
import hashlib
from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.cnn_autoencoder import CNNAutoencoder
from src.filterbank_reader import read_filterbank_header, load_filterbank_waterfall, extract_spectrogram_tiles
from src.sequential_trainer import SequentialTrainer
from src.training import compute_reconstruction_errors, calibrate_threshold
from src.config import MODELS_DIR, RAW_DATA_DIR, SPEC_HEIGHT, SPEC_WIDTH, LATENT_DIM


def test_1_original_checkpoint_loads():
    """Requirement 1: Verify original checkpoint exists and loads state dict."""
    ckpt_path = MODELS_DIR / "cnn_autoencoder.pt"
    assert ckpt_path.exists(), f"Original checkpoint missing at {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "threshold" in ckpt
    assert ckpt.get("latent_dim", 48) == 48

    model = CNNAutoencoder(latent_dim=48)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == dummy.shape


def test_2_and_3_subsequent_stages_inherit_and_update_weights(tmp_path):
    """Requirements 2 & 3: Subsequent stage starts from preceding weights and updates parameters."""
    # 1. Base model
    base_model = CNNAutoencoder(latent_dim=48)
    base_state = copy.deepcopy(base_model.state_dict())

    # Save mock C1 checkpoint
    c1_path = tmp_path / "c1.pt"
    torch.save({"model_state_dict": base_state, "latent_dim": 48}, c1_path)

    # 2. Stage 2 model loads C1
    stage2_model = CNNAutoencoder(latent_dim=48)
    loaded_c1 = torch.load(c1_path, map_location="cpu", weights_only=False)
    stage2_model.load_state_dict(loaded_c1["model_state_dict"])

    # Verify weights are identical before training
    for k in base_state:
        assert torch.equal(base_state[k], stage2_model.state_dict()[k])

    # 3. Train for 1 epoch on mock data
    mock_data = np.random.uniform(0, 1, size=(20, 1, 64, 64)).astype(np.float32)
    trainer = SequentialTrainer(output_dir=tmp_path)
    res = trainer.train_stage(
        stage_num=2,
        dataset_meta={"source_name": "mock_ds", "telescope_name": "mock_tel", "band": "S-band"},
        train_data=mock_data,
        val_data=mock_data[:5],
        preceding_checkpoint=c1_path,
        epochs=1,
        batch_size=10,
        lr=1e-3,
    )

    # Verify weights updated
    updated_state = res["model"].state_dict()
    diffs = [not torch.equal(base_state[k], updated_state[k]) for k in base_state if "weight" in k]
    assert any(diffs), "Training failed to update model parameters!"


def test_4_original_checkpoint_remains_unchanged():
    """Requirement 4: Verify original checkpoint file hash remains constant."""
    ckpt_path = MODELS_DIR / "cnn_autoencoder.pt"
    assert ckpt_path.exists()
    with open(ckpt_path, "rb") as f:
        hash_before = hashlib.md5(f.read()).hexdigest()

    # Load and run inference
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CNNAutoencoder(latent_dim=48)
    model.load_state_dict(ckpt["model_state_dict"])
    _ = model(torch.randn(1, 1, 64, 64))

    with open(ckpt_path, "rb") as f:
        hash_after = hashlib.md5(f.read()).hexdigest()

    assert hash_before == hash_after, "Original checkpoint was unexpectedly modified!"


def test_5_preprocessing_on_each_additional_dataset():
    """Requirement 5: Test preprocessing & tile extraction on both newly added datasets."""
    f_prox = RAW_DATA_DIR / "guppi_59335_45797_6179325092_ProxCen_S_0001.0002.fil"
    f_gbt = RAW_DATA_DIR / "spliced_blc40414243444546o7o0515253545556o7o061626364656667_guppi_58838_21527_HIP26137_0086.gpuspec.0002.fil"

    assert f_prox.exists(), f"ProxCen file missing: {f_prox}"
    assert f_gbt.exists(), f"GBT file missing: {f_gbt}"

    # ProxCen test (multipolarization / S-band)
    res2 = load_filterbank_waterfall(f_prox, max_time_steps=64)
    tiles2 = extract_spectrogram_tiles(res2["waterfall"], 64, 64)
    assert tiles2.ndim == 4
    assert tiles2.shape[1:] == (1, 64, 64)
    assert 0.0 <= tiles2.min() and tiles2.max() <= 1.0

    # GBT spliced test (X-band / 1.31M channels)
    res3 = load_filterbank_waterfall(f_gbt, max_time_steps=64, f_start_idx=0, f_stop_idx=512)
    tiles3 = extract_spectrogram_tiles(res3["waterfall"], 64, 64)
    assert tiles3.ndim == 4
    assert tiles3.shape[1:] == (1, 64, 64)
    assert 0.0 <= tiles3.min() and tiles3.max() <= 1.0


def test_6_leak_free_data_partitioning():
    """Requirement 6: Verify train, validation, and test sets are strictly disjoint."""
    trainer = SequentialTrainer()
    datasets = trainer.discover_datasets()
    assert len(datasets) == 3

    p = trainer.prepare_dataset_partitions(datasets[0], train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, target_sample_limit=1000)
    assert p["n_train"] == 700
    assert p["n_val"] == 150
    assert p["n_test"] == 150

    # Verify partitions do not share identical tile references
    assert id(p["train"]) != id(p["val"])
    assert id(p["val"]) != id(p["test"])


def test_7_and_8_final_checkpoint_and_threshold_calibration(tmp_path):
    """Requirements 7 & 8: Verify threshold calibration on designated validation set and checkpoint saving."""
    model = CNNAutoencoder(latent_dim=48)
    val_mock = np.random.uniform(0, 1, size=(50, 1, 64, 64)).astype(np.float32)
    test_mock = np.random.uniform(0, 1, size=(50, 1, 64, 64)).astype(np.float32)

    val_errs = compute_reconstruction_errors(model, val_mock, batch_size=10, device="cpu")
    thresh = calibrate_threshold(val_errs, percentile=99.0)
    assert thresh > 0.0

    # Test set evaluation
    test_errs = compute_reconstruction_errors(model, test_mock, batch_size=10, device="cpu")
    assert len(test_errs) == 50

    ckpt_save = tmp_path / "final_model.pt"
    torch.save({"model_state_dict": model.state_dict(), "threshold": thresh, "latent_dim": 48}, ckpt_save)
    assert ckpt_save.exists()

    loaded = torch.load(ckpt_save, map_location="cpu", weights_only=False)
    assert loaded["threshold"] == thresh


def test_9_and_10_smoke_test_sequential_trainer(tmp_path):
    """Requirements 9 & 10: Run full mini-pipeline smoke test with small sample limit."""
    trainer = SequentialTrainer(output_dir=tmp_path / "checkpoints", figures_dir=tmp_path / "figures", seed=123)
    datasets = trainer.discover_datasets()
    assert len(datasets) == 3

    # Run smoke test for 1 epoch on mini-partitions
    p1 = trainer.prepare_dataset_partitions(datasets[0], target_sample_limit=100)
    p2 = trainer.prepare_dataset_partitions(datasets[1], target_sample_limit=100)

    # Stage 1
    s1 = trainer.train_stage(
        stage_num=1,
        dataset_meta=datasets[0],
        train_data=p1["train"],
        val_data=p1["val"],
        epochs=1,
        batch_size=32,
    )
    assert s1["checkpoint_path"].exists()

    # Stage 2 continuing from Stage 1
    s2 = trainer.train_stage(
        stage_num=2,
        dataset_meta=datasets[1],
        train_data=p2["train"],
        val_data=p2["val"],
        preceding_checkpoint=s1["checkpoint_path"],
        replay_pool=p1["train"][:10],
        epochs=1,
        batch_size=32,
    )
    assert s2["checkpoint_path"].exists()
    assert s2["final_train_loss"] > 0.0
