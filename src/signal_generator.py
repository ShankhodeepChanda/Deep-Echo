"""
Synthetic Radio Signal Generator for SETI Deep Anomaly Detection.

This module generates synthetic time-domain radio frequency (RF) signals across three distinct categories:
1. Normal-like observations (background noise, mild narrowband carriers, simple linear drifts).
2. RFI-like observations (terrestrial radio frequency interference: persistent carriers, broadband bursts, pulses).
3. Unusual synthetic observations (non-linear drifts, multiple intersecting tracks, complex anomaly patterns).
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Union

def generate_time_axis(sample_rate: float, duration: float) -> np.ndarray:
    """Generate a time axis array for signal generation."""
    num_samples = int(sample_rate * duration)
    return np.linspace(0, duration, num_samples, endpoint=False)

def generate_gaussian_noise(num_samples: int, mean: float = 0.0, std: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
    """Generate additive white Gaussian noise."""
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, num_samples)

def generate_chirp_signal(
    t: np.ndarray,
    f_start: float,
    f_end: float,
    amplitude: float = 1.0,
    phase: float = 0.0
) -> np.ndarray:
    """
    Generate a linear frequency modulated (LFM) chirp signal (drifting tone).

    Formula: s(t) = A * cos(2 * pi * (f_start * t + 0.5 * (f_end - f_start) / T * t^2) + phase)
    """
    duration = t[-1] - t[0] if len(t) > 1 else 1.0
    # Instantaneous frequency: f(t) = f_start + (f_end - f_start) * t / duration
    # Phase integral: 2 * pi * (f_start * t + 0.5 * (f_end - f_start) * t^2 / duration)
    if duration == 0:
        phase_inst = 2 * np.pi * f_start * t + phase
    else:
        phase_inst = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * (t ** 2) / duration) + phase
    return amplitude * np.cos(phase_inst)

def generate_normal_signal(
    sample_rate: float = 1000.0,
    duration: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Generate a normal-like observation consisting of background noise
    plus zero or one mild linear drifting carrier or steady narrowband tone.
    """
    rng = np.random.default_rng(seed)
    t = generate_time_axis(sample_rate, duration)

    # Base noise
    noise_std = rng.uniform(0.8, 1.2)
    signal = generate_gaussian_noise(len(t), mean=0.0, std=noise_std, seed=rng)

    has_carrier = rng.random() > 0.3
    metadata = {
        "category": "normal_like",
        "sample_rate": sample_rate,
        "duration": duration,
        "noise_std": noise_std,
        "has_carrier": has_carrier
    }

    if has_carrier:
        f_start = rng.uniform(50.0, sample_rate * 0.4)
        drift = rng.uniform(-20.0, 20.0)
        f_end = f_start + drift
        amplitude = rng.uniform(0.5, 2.0)
        phase = rng.uniform(0, 2 * np.pi)

        carrier = generate_chirp_signal(t, f_start, f_end, amplitude, phase)
        signal += carrier
        metadata.update({
            "f_start": f_start,
            "f_end": f_end,
            "drift": drift,
            "amplitude": amplitude
        })

    return signal, metadata

def generate_rfi_signal(
    sample_rate: float = 1000.0,
    duration: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Generate an RFI-like observation (terrestrial interference) such as
    persistent strong carriers, vertical broadband bursts, or pulsing interference.
    """
    rng = np.random.default_rng(seed)
    t = generate_time_axis(sample_rate, duration)

    # Background noise
    signal = generate_gaussian_noise(len(t), mean=0.0, std=1.0, seed=rng)

    rfi_type = rng.choice(["persistent_multi", "broadband_burst", "pulsed_carrier"])
    metadata = {
        "category": "synthetic_rfi",
        "sample_rate": sample_rate,
        "duration": duration,
        "rfi_type": rfi_type
    }

    if rfi_type == "persistent_multi":
        # Multiple strong steady carriers
        num_carriers = rng.integers(2, 5)
        for _ in range(num_carriers):
            freq = rng.uniform(20.0, sample_rate * 0.45)
            amp = rng.uniform(2.5, 6.0)
            signal += amp * np.cos(2 * np.pi * freq * t)
        metadata["num_carriers"] = num_carriers

    elif rfi_type == "broadband_burst":
        # Transient broadband impulse (vertical spike in spectrogram)
        burst_start = rng.integers(int(len(t) * 0.3), int(len(t) * 0.7))
        burst_len = rng.integers(int(sample_rate * 0.02), int(sample_rate * 0.08))
        burst_end = min(burst_start + burst_len, len(t))
        amplitude = rng.uniform(8.0, 15.0)
        signal[burst_start:burst_end] += amplitude * rng.normal(0, 1, burst_end - burst_start)
        metadata["burst_start"] = int(burst_start)
        metadata["burst_end"] = int(burst_end)
        metadata["amplitude"] = float(amplitude)

    else:  # pulsed_carrier
        freq = rng.uniform(100.0, sample_rate * 0.4)
        pulse_freq_hz = rng.uniform(5.0, 20.0) # pulse repetition rate
        envelope = 0.5 * (1 + np.cos(2 * np.pi * pulse_freq_hz * t))
        signal += rng.uniform(3.0, 7.0) * envelope * np.cos(2 * np.pi * freq * t)
        metadata["carrier_freq"] = freq
        metadata["pulse_rate"] = pulse_freq_hz

    return signal, metadata

def generate_anomaly_signal(
    sample_rate: float = 1000.0,
    duration: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Generate an unusual synthetic observation (potential technosignature candidate pattern)
    such as nonlinear frequency drifting, complex multi-component tracks, or unusual modulations.
    """
    rng = np.random.default_rng(seed)
    t = generate_time_axis(sample_rate, duration)

    signal = generate_gaussian_noise(len(t), mean=0.0, std=1.0, seed=rng)

    anomaly_type = rng.choice(["nonlinear_drift", "accelerated_chirp", "dual_crossing"])
    metadata = {
        "category": "synthetic_anomaly",
        "sample_rate": sample_rate,
        "duration": duration,
        "anomaly_type": anomaly_type
    }

    if anomaly_type == "nonlinear_drift":
        # Quadratic frequency modulation
        f_base = rng.uniform(100.0, 300.0)
        quad_coeff = rng.uniform(-200.0, 200.0)
        phase = 2 * np.pi * (f_base * t + quad_coeff * (t ** 3) / 3.0)
        amplitude = rng.uniform(1.5, 3.5)
        signal += amplitude * np.cos(phase)
        metadata["f_base"] = f_base
        metadata["quad_coeff"] = quad_coeff

    elif anomaly_type == "accelerated_chirp":
        # Strong high-acceleration chirp
        f_start = rng.uniform(50.0, 200.0)
        f_end = f_start + rng.choice([-1.0, 1.0]) * rng.uniform(150.0, 300.0)
        amplitude = rng.uniform(2.0, 4.0)
        # Exponential-like frequency shift simulation via squared term
        phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * (t ** 1.5) / (duration ** 0.5))
        signal += amplitude * np.cos(phase)
        metadata["f_start"] = f_start
        metadata["f_end"] = f_end

    else:  # dual_crossing
        # Two simultaneous drifting tones crossing each other
        f1_start = rng.uniform(100.0, 200.0)
        f1_end = f1_start + 100.0
        f2_start = rng.uniform(300.0, 400.0)
        f2_end = f2_start - 100.0

        sig1 = 1.8 * generate_chirp_signal(t, f1_start, f1_end)
        sig2 = 1.8 * generate_chirp_signal(t, f2_start, f2_end)
        signal += sig1 + sig2
        metadata["f1_range"] = [f1_start, f1_end]
        metadata["f2_range"] = [f2_start, f2_end]

    return signal, metadata

def generate_dataset(
    num_normal: int = 100,
    num_rfi: int = 30,
    num_anomaly: int = 20,
    sample_rate: float = 1000.0,
    duration: float = 1.0,
    base_seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Generate a complete mixed dataset of signals, their category labels, and metadata.

    Returns:
        signals: Array of shape (N, num_samples)
        labels: Array of shape (N,) with category strings
        metadata_list: List of metadata dictionaries
    """
    rng = np.random.default_rng(base_seed)
    seeds = rng.integers(0, 1_000_000, size=(num_normal + num_rfi + num_anomaly))

    signals = []
    labels = []
    metadata_list = []

    idx = 0
    # 1. Normal-like
    for i in range(num_normal):
        sig, meta = generate_normal_signal(sample_rate, duration, seed=int(seeds[idx]))
        signals.append(sig)
        labels.append("normal_like")
        metadata_list.append(meta)
        idx += 1

    # 2. RFI-like
    for i in range(num_rfi):
        sig, meta = generate_rfi_signal(sample_rate, duration, seed=int(seeds[idx]))
        signals.append(sig)
        labels.append("synthetic_rfi")
        metadata_list.append(meta)
        idx += 1

    # 3. Anomaly
    for i in range(num_anomaly):
        sig, meta = generate_anomaly_signal(sample_rate, duration, seed=int(seeds[idx]))
        signals.append(sig)
        labels.append("synthetic_anomaly")
        metadata_list.append(meta)
        idx += 1

    return np.array(signals), np.array(labels), metadata_list
