import numpy as np
import pytest
from src import signal_generator

def test_generate_time_axis():
    t = signal_generator.generate_time_axis(100.0, 2.0)
    assert len(t) == 200
    assert t[0] == 0.0
    assert t[-1] < 2.0

def test_generate_normal_signal():
    sig, meta = signal_generator.generate_normal_signal(1000.0, 1.0, seed=42)
    assert isinstance(sig, np.ndarray)
    assert len(sig) == 1000
    assert meta["category"] == "normal_like"
    assert np.isfinite(sig).all()

def test_generate_rfi_signal():
    sig, meta = signal_generator.generate_rfi_signal(1000.0, 1.0, seed=42)
    assert isinstance(sig, np.ndarray)
    assert len(sig) == 1000
    assert meta["category"] == "synthetic_rfi"
    assert np.isfinite(sig).all()

def test_generate_anomaly_signal():
    sig, meta = signal_generator.generate_anomaly_signal(1000.0, 1.0, seed=42)
    assert isinstance(sig, np.ndarray)
    assert len(sig) == 1000
    assert meta["category"] == "synthetic_anomaly"
    assert np.isfinite(sig).all()

def test_generate_dataset():
    signals, labels, metadata = signal_generator.generate_dataset(
        num_normal=10, num_rfi=5, num_anomaly=5, sample_rate=500.0, duration=0.5, base_seed=123
    )
    assert signals.shape == (20, 250)
    assert labels.shape == (20,)
    assert len(metadata) == 20
    assert np.sum(labels == "normal_like") == 10
    assert np.sum(labels == "synthetic_rfi") == 5
    assert np.sum(labels == "synthetic_anomaly") == 5
