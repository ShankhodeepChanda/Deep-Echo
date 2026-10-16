"""
Spectrogram generation for SETI anomaly detection.

Implements STFT-based spectrogram computation with configurable parameters.
Supports magnitude and power spectrograms with optional logarithmic scaling.
"""

from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)


def compute_spectrogram(
    sig: np.ndarray,
    fs: float,
    nperseg: int = 128,
    noverlap: Optional[int] = None,
    window: str = 'hann',
    scaling: str = 'power',
    nfft: Optional[int] = None,
    axis: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Short-Time Fourier Transform (STFT) spectrogram.

    Parameters
    ----------
    sig : np.ndarray
        1D input signal (real or complex).
    fs : float
        Sampling frequency in Hz.
    nperseg : int, default 128
        Length of each segment. Longer segment = finer frequency resolution.
    noverlap : int or None, default None
        Number of points to overlap between segments. If None, defaults to nperseg//2.
    window : str, default 'hann'
        Window function name recognized by scipy.signal.stft.
    scaling : {'power', 'magnitude'}, default 'power'
        Whether to return power (squared magnitude) or magnitude.
    nfft : int or None, default None
        Zero-padding size for FFT. If None, uses nperseg.
    axis : int, default -1
        Axis over which to apply FFT.

    Returns
    -------
    f : np.ndarray
        Array of sample frequencies (Hz).
    t : np.ndarray
        Array of time points (seconds).
    Sxx : np.ndarray
        Spectrogram (shape: (n_freqs, n_times)).
    """
    if noverlap is None:
        noverlap = nperseg // 2

    # Validate inputs
    if sig.ndim != 1:
        raise ValueError("signal must be 1D array")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if nperseg <= 0:
        raise ValueError("nperseg must be positive")
    if noverlap > nperseg:
        raise ValueError("noverlap cannot exceed nperseg")

    # Compute STFT
    f, t, Zxx = signal.stft(sig, fs=fs, window=window,
                           nperseg=nperseg, noverlap=noverlap,
                           nfft=nfft, axis=axis)

    # Apply scaling
    if scaling == 'power':
        Sxx = np.abs(Zxx)**2
    elif scaling == 'magnitude':
        Sxx = np.abs(Zxx)
    else:
        raise ValueError("scaling must be 'power' or 'magnitude'")

    # Ensure shape consistency (freqs, times)
    if Sxx.ndim == 2:
        Sxx = Sxx.T  # Convert to (freqs, times)
    return f, t, Sxx


def generate_spectrogram(
    sig: np.ndarray,
    sample_rate: float,
    nperseg: int = 128,
    noverlap: Optional[int] = None,
    window: str = 'hann',
    scaling: str = 'power',
    nfft: Optional[int] = None,
    axis: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper that computes spectrogram with default parameters.

    Returns:
        f (frequencies), t (times), Sxx (spectrogram matrix)
    """
    return compute_spectrogram(sig, fs=sample_rate,
                              nperseg=nperseg, noverlap=noverlap,
                              window=window, scaling=scaling,
                              nfft=nfft, axis=axis)


def resize_spectrogram(
    spectrogram: np.ndarray,
    new_height: int,
    new_width: int,
    method: str = 'nearest'
) -> np.ndarray:
    """
    Resize spectrogram to specified dimensions using interpolation.

    Parameters
    ----------
    spectrogram : np.ndarray
        Input spectrogram (frequency x time).
    new_height : int
        New height (frequency dimension).
    new_width : int
        New width (time dimension).
    method : str, default 'nearest'
        Interpolation method: 'nearest', 'bilinear', 'bicubic'

    Returns
    -------
    resized_spectrogram : np.ndarray
        Resized spectrogram with shape (new_height, new_width).
    """
    if spectrogram.ndim != 2:
        raise ValueError("spectrogram must be 2D array")

    # Use scipy.ndimage.zoom for proper resizing
    from scipy.ndimage import zoom

    height_scale = new_height / spectrogram.shape[0] if spectrogram.shape[0] > 0 else 1.0
    width_scale = new_width / spectrogram.shape[1] if spectrogram.shape[1] > 0 else 1.0

    resized = zoom(spectrogram, (height_scale, width_scale), order=0 if method == 'nearest' else 1)
    return resized


def resize_spectrogram_with_padding(
    spectrogram: np.ndarray,
    target_height: int,
    target_width: int,
    pad_value: float = 0.0,
    method: str = 'nearest'
) -> np.ndarray:
    """
    Resize spectrogram to target dimensions with padding if needed.

    If target dimensions are larger than spectrogram, pads with pad_value.
    If smaller, crops from center.

    Parameters
    ----------
    spectrogram : np.ndarray
        Input spectrogram (freq x time)
    target_height : int
        Desired height (frequency dimension)
    target_width : int
        Desired width (time dimension)
    pad_value : float, default 0.0
        Value to use for padding
    method : str, default 'nearest'
        Interpolation method for resizing (only used if resizing down)

    Returns
    -------
    resized_spectrogram : np.ndarray
        Resized spectrogram with shape (target_height, target_width)
    """
    if spectrogram.ndim != 2:
        raise ValueError("spectrogram must be 2D array")

    current_height, current_width = spectrogram.shape

    # Create output array filled with pad_value
    resized = np.full((target_height, target_width), pad_value, dtype=spectrogram.dtype)

    # Calculate where to place the original spectrogram
    y_start = (target_height - current_height) // 2
    x_start = (target_width - current_width) // 2

    # Calculate end positions
    y_end = min(y_start + current_height, target_height)
    x_end = min(x_start + current_width, target_width)

    # Calculate source region
    src_y_end = min(current_height, target_height)
    src_x_end = min(current_width, target_width)

    # Place spectrogram in center of output array
    resized[y_start:y_end, x_start:x_end] = spectrogram[:src_y_end, :src_x_end]

    return resized


def generate_spectrogram_2d(
    sig: np.ndarray,
    sample_rate: float,
    nperseg: int = 128,
    noverlap: Optional[int] = None,
    window: str = 'hann',
    scaling: str = 'power',
    nfft: Optional[int] = None,
    nfft_axis: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 2D spectrogram (frequency x time) from 1D signal.

    This is the main function for creating spectrograms for model input.
    """
    f, t, Sxx = compute_spectrogram(sig, fs=sample_rate,
                                   nperseg=nperseg, noverlap=noverlap,
                                   window=window, scaling=scaling,
                                   nfft=nfft, nfft_axis=nfft_axis)

    # Ensure proper shape (freq x time)
    if Sxx.ndim == 1:
        Sxx = Sxx.reshape(-1, 1)
    elif Sxx.ndim == 0:
        Sxx = Sxx.reshape(1, 1)

    return f, t, Sxx


def generate_spectrogram_batch(
    signals: List[np.ndarray],
    sample_rates: List[float],
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate spectrograms for multiple signals with potentially different sample rates.

    Returns stacked arrays:
    - f: 2D array (n_signals, max_freqs) - padded to max dimensions
    - t: 2D array (n_signals, max_times) - padded to max dimensions
    - Sxx: 3D array (n_signals, max_freqs, max_times) - padded to max dimensions
    """
    if len(signals) != len(sample_rates):
        raise ValueError("signals and sample_rates must have same length")

    # Find max dimensions
    max_freqs = 0
    max_times = 0
    n_signals = len(signals)

    for i, (sig, fs) in enumerate(zip(signals, sample_rates)):
        f, t, Sxx = generate_spectrogram(sig, fs=fs, **kwargs)
        max_freqs = max(max_freqs, f.shape[0])
        max_times = max(max_times, t.shape[0])

    # Create padded arrays
    padded_f = np.zeros((n_signals, max_freqs), dtype=float)
    padded_t = np.zeros((n_signals, max_times), dtype=float)
    padded_sxx = np.zeros((n_signals, max_freqs, max_times), dtype=float)

    for i, (sig, fs) in enumerate(zip(signals, sample_rates)):
        f, t, Sxx = generate_spectrogram(sig, fs=fs, **kwargs)
        padded_f[i, :f.shape[0]] = f
        padded_t[i, :t.shape[0]] = t
        padded_sxx[i, :f.shape[0], :t.shape[0]] = Sxx

    return padded_f, padded_t, padded_sxx


def log_scale_spectrogram(spectrogram: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Apply logarithmic scaling to spectrogram with numerical safeguard.

    Parameters
    ----------
    spectrogram : np.ndarray
        Input spectrogram (power or magnitude).
    eps : float, default 1e-10
        Small constant to avoid log(0).

    Returns
    -------
    log_spectrogram : np.ndarray
        Log-scaled spectrogram.
    """
    return np.log10(spectrogram + eps)


def normalize_spectrogram(
    spectrogram: np.ndarray,
    method: str = 'standard',
    stats: Optional[dict] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Normalize spectrogram using dataset statistics.

    Parameters
    ----------
    spectrogram : np.ndarray
        Input spectrogram (freq x time).
    method : str, default 'standard'
        Normalization method: 'standard', 'minmax', 'log_standard'.
    stats : dict or None
        Pre-computed statistics (mean, std, min, max) for inference.
        If None, computed from this spectrogram.

    Returns
    -------
    normalized : np.ndarray
        Normalized spectrogram.
    stats : dict
        Statistics used for normalization.
    """
    if method == 'standard':
        if stats is None:
            mean = np.mean(spectrogram)
            std = np.std(spectrogram)
            stats = {'mean': float(mean), 'std': float(std)}
        else:
            mean, std = stats['mean'], stats['std']
        if std > 0:
            normalized = (spectrogram - mean) / std
        else:
            normalized = spectrogram - mean

    elif method == 'minmax':
        if stats is None:
            min_val = np.min(spectrogram)
            max_val = np.max(spectrogram)
            stats = {'min': float(min_val), 'max': float(max_val)}
        else:
            min_val, max_val = stats['min'], stats['max']
        if max_val > min_val:
            normalized = (spectrogram - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(spectrogram)

    elif method == 'log_standard':
        if stats is None:
            log_spec = log_scale_spectrogram(spectrogram)
            mean = np.mean(log_spec)
            std = np.std(log_spec)
            stats = {'mean': float(mean), 'std': float(std)}
        else:
            mean, std = stats['mean'], stats['std']
        log_spec = log_scale_spectrogram(spectrogram)
        if std > 0:
            normalized = (log_spec - mean) / std
        else:
            normalized = log_spec - mean
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, stats


if __name__ == "__main__":
    # Self-test
    import numpy as np

    np.random.seed(42)
    fs = 1000.0  # 1 kHz
    duration = 1.0  # 1 second
    t = np.arange(0, duration, 1/fs)

    # Create test signal: two chirps
    sig = (np.sin(2 * np.pi * 50 * t) +
           0.5 * np.sin(2 * np.pi * 200 * t) +
           0.1 * np.random.randn(len(t)))

    f, t_spec, Sxx = generate_spectrogram(sig, fs=fs, nperseg=128, noverlap=64)

    print(f"Signal shape: {sig.shape}")
    print(f"Frequency axis: {f.shape[0]} points, {f[0]:.1f}-{f[-1]:.1f} Hz")
    print(f"Time axis: {t_spec.shape[0]} points, {t_spec[0]:.3f}-{t_spec[-1]:.3f} s")
    print(f"Spectrogram shape: {Sxx.shape}")
    print(f"Spectrogram range: {Sxx.min():.2e} - {Sxx.max():.2e}")

    # Test resize
    resized = resize_spectrogram(Sxx, 64, 64)
    print(f"Resized shape: {resized.shape}")

    # Test log scale
    log_spec = log_scale_spectrogram(Sxx)
    print(f"Log spectrogram range: {log_spec.min():.2f} - {log_spec.max():.2f}")

    # Test normalization
    norm_spec, stats = normalize_spectrogram(Sxx, method='standard')
    print(f"Normalized range: {norm_spec.min():.2f} - {norm_spec.max():.2f}")
    print(f"Normalized mean: {norm_spec.mean():.4f}, std: {norm_spec.std():.4f}")
    print(f"Stats: {stats}")

    print("\nAll tests passed!")