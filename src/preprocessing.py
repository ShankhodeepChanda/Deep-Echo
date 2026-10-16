"""
Signal preprocessing for SETI anomaly detection.

Provides functions to clean and prepare signals for spectrogram generation
and model input while preserving potential signal structure.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


def validate_and_clean_signal(
    signal: np.ndarray,
    *,
    remove_mean: bool = False,
    detrend: bool = False,
    remove_dc: bool = False,
    scale: Optional[str] = None,
    clip_percentile: Optional[Tuple[float, float]] = None,
    replace_nan: Optional[float] = None,
    replace_inf: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Validate and preprocess a 1D signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (real or complex). Will be converted to float64 for processing.
    remove_mean : bool, default False
        If True, subtract the mean from the signal.
    detrend : bool, default False
        If True, remove a linear trend using scipy.signal.detrend.
    remove_dc : bool, default False
        If True, remove the DC offset (same as remove_mean for real signals).
    scale : {'minmax', 'standard', 'unit', None}, default None
        Normalization method:
        - 'minmax': scale to [0, 1]
        - 'standard': zero mean, unit variance
        - 'unit': scale to unit L2 norm
        - None: no scaling
    clip_percentile : tuple (low, high) in [0,100], default None
        If provided, clip values to the given percentiles (e.g., (1, 99)).
    replace_nan : float, default None
        Value to replace NaNs with. If None, NaNs are not replaced.
    replace_inf : float, default None
        Value to replace +/-inf with. If None, infs are not replaced.

    Returns
    -------
    processed_signal : np.ndarray
        The preprocessed signal (float64, same shape as input).
    metadata : dict
        Dictionary containing:
        - original_shape
        - original_dtype
        - steps_applied (list of strings)
        - nan_count, inf_count
        - min, max, mean, std (after processing)
        - any parameters used (e.g., scale method, clip bounds)
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("signal must be a numpy ndarray")

    # Work on a copy to avoid side effects
    proc = signal.astype(np.float64, copy=True)
    meta = {
        "original_shape": signal.shape,
        "original_dtype": str(signal.dtype),
        "steps_applied": [],
    }

    # Count non-finite values before any processing
    nan_mask = np.isnan(proc)
    inf_mask = np.isinf(proc)
    meta["nan_count"] = int(np.sum(nan_mask))
    meta["inf_count"] = int(np.sum(inf_mask))

    # Replace non-finite values if requested
    if replace_nan is not None and meta["nan_count"] > 0:
        proc = np.where(nan_mask, replace_nan, proc)
        meta["steps_applied"].append(f"replace_nan_with_{replace_nan}")
    if replace_inf is not None and meta["inf_count"] > 0:
        proc = np.where(inf_mask, replace_inf, proc)
        meta["steps_applied"].append(f"replace_inf_with_{replace_inf}")

    # Optional processing steps
    if remove_mean or remove_dc:
        mean_val = np.mean(proc)
        proc -= mean_val
        meta["steps_applied"].append("remove_mean")
        meta["removed_mean"] = float(mean_val)

    if detrend:
        try:
            from scipy import signal
            proc = signal.detrend(proc, type='linear')
            meta["steps_applied"].append("detrend_linear")
        except ImportError:
            logger.warning("scipy not available; skipping detrend")
            # Fallback: subtract linear fit
            x = np.arange(len(proc))
            coeffs = np.polyfit(x, proc, 1)
            trend = np.polyval(coeffs, x)
            proc -= trend
            meta["steps_applied"].append("detrend_linear_fallback")

    # Clip outliers by percentile
    if clip_percentile is not None:
        low, high = clip_percentile
        if not (0 <= low < high <= 100):
            raise ValueError("clip_percentile must be (low, high) with 0<=low<high<=100")
        low_val = np.percentile(proc, low)
        high_val = np.percentile(proc, high)
        proc = np.clip(proc, low_val, high_val)
        meta["steps_applied"].append(f"clip_percentile_{low}_{high}")
        meta["clip_bounds"] = (float(low_val), float(high_val))

    # Scaling
    if scale == "minmax":
        min_val = np.min(proc)
        max_val = np.max(proc)
        if max_val > min_val:
            proc = (proc - min_val) / (max_val - min_val)
        else:
            proc = np.zeros_like(proc)
        meta["steps_applied"].append("scale_minmax")
        meta["scale_minmax"] = (float(min_val), float(max_val))
    elif scale == "standard":
        mean_val = np.mean(proc)
        std_val = np.std(proc)
        if std_val > 0:
            proc = (proc - mean_val) / std_val
        else:
            proc = np.zeros_like(proc)
        meta["steps_applied"].append("scale_standard")
        meta["scale_std"] = (float(mean_val), float(std_val))
    elif scale == "unit":
        norm = np.linalg.norm(proc)
        if norm > 0:
            proc = proc / norm
        else:
            proc = np.zeros_like(proc)
        meta["steps_applied"].append("scale_unit")
        meta["scale_unit_norm"] = float(norm)

    # Final statistics
    meta.update({
        "min": float(np.min(proc)),
        "max": float(np.max(proc)),
        "mean": float(np.mean(proc)),
        "std": float(np.std(proc)),
    })

    return proc, meta


def compute_noise_floor(signal: np.ndarray, method: str = "median") -> float:
    """
    Estimate the noise floor of a signal.

    Parameters
    ----------
    signal : np.ndarray
        Preprocessed signal (real).
    method : {'median', 'mean', 'rmse'}, default 'median'
        Method to estimate noise floor.

    Returns
    -------
    noise_floor : float
        Estimated noise level.
    """
    if method == "median":
        return float(np.median(np.abs(signal)))
    if method == "mean":
        return float(np.mean(np.abs(signal)))
    if method == "rmse":
        return float(np.sqrt(np.mean(signal**2)))
    raise ValueError(f"Unknown method: {method}")


def signal_to_analytic(signal: np.ndarray) -> np.ndarray:
    """
    Convert a real signal to its analytic representation using the Hilbert transform.

    Parameters
    ----------
    signal : np.ndarray
        Real-valued signal.

    Returns
    -------
    analytic_signal : np.ndarray
        Complex analytic signal (real part = original, imaginary part = Hilbert transform).
    """
    try:
        from scipy import signal
        return signal.hilbert(signal)
    except ImportError:
        logger.warning("scipy not available; returning original signal as analytic (imag=0)")
        return signal + 0j


def extract_envelope(signal: np.ndarray) -> np.ndarray:
    """
    Extract the envelope of a signal using the Hilbert transform.

    Parameters
    ----------
    signal : np.ndarray
        Real-valued signal.

    Returns
    -------
    envelope : np.ndarray
        Envelope of the signal (magnitude of analytic signal).
    """
    analytic = signal_to_analytic(signal)
    return np.abs(analytic)


def extract_phase(signal: np.ndarray) -> np.ndarray:
    """
    Extract the instantaneous phase of a signal using the Hilbert transform.

    Parameters
    ----------
    signal : np.ndarray
        Real-valued signal.

    Returns
    -------
    phase : np.ndarray
        Instantaneous phase in radians.
    """
    analytic = signal_to_analytic(signal)
    return np.angle(analytic)


def extract_instantaneous_frequency(signal: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    Extract the instantaneous frequency of a signal.

    Parameters
    ----------
    signal : np.ndarray
        Real-valued signal.
    sample_rate : float
        Sampling rate in Hz.

    Returns
    -------
    instantaneous_frequency : np.ndarray
        Instantaneous frequency in Hz (same length as signal).
    """
    phase = extract_phase(signal)
    # Differentiate phase and convert to frequency
    instant_freq = np.diff(unwrap_phase(phase)) * (sample_rate / (2 * np.pi))
    # Prepend first value to maintain length
    instant_freq = np.concatenate(([instant_freq[0]] if len(instant_freq) > 0 else [0], instant_freq))
    return instant_freq


def unwrap_phase(phase: np.ndarray) -> np.ndarray:
    """
    Unwrap phase to avoid jumps of 2*pi.

    Parameters
    ----------
    phase : np.ndarray
        Wrapped phase in radians.

    Returns
    -------
    unwrapped_phase : np.ndarray
        Unwrapped phase.
    """
    return np.unwrap(phase)


# ──────────────────────────────────────────────────────────────────────
# Quality checks
# ──────────────────────────────────────────────────────────────────────
def check_signal_quality(signal: np.ndarray, *, min_len: int = 10, max_std: Optional[float] = None) -> Dict[str, Any]:
    """
    Perform basic quality checks on a signal.

    Returns a dict with boolean flags and any issues.
    """
    issues = []
    if signal.size < min_len:
        issues.append(f"Signal too short: {signal.size} < {min_len}")
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        issues.append("Signal contains NaN or Inf values")
    if max_std is not None:
        std_val = np.std(signal)
        if std_val > max_std:
            issues.append(f"Signal std ({std_val}) exceeds max_std ({max_std})")
    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "length": int(signal.size),
        "has_nan": bool(np.any(np.isnan(signal))),
        "has_inf": bool(np.any(np.isinf(signal))),
    }


# ──────────────────────────────────────────────────────────────────────
# Example usage and self-test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simple self-test
    np.random.seed(42)
    test_sig = np.random.randn(1000) + np.linspace(0, 1, 1000)  # noisy ramp
    proc, meta = validate_and_clean_signal(
        test_sig,
        remove_mean=True,
        detrend=True,
        scale="standard",
        clip_percentile=(1, 99),
    )
    print("Preprocessing test:")
    print(f"  Original shape: {meta['original_shape']}")
    print(f"  Steps applied: {meta['steps_applied']}")
    print(f"  Processed mean: {meta['mean']:.6f} (should be ~0)")
    print(f"  Processed std:  {meta['std']:.6f} (should be ~1)")

    # Noise floor
    nf = compute_noise_floor(proc, method="median")
    print(f"  Noise floor (median abs): {nf:.6f}")

    # Quality
    quality = check_signal_quality(proc, min_len=50)
    print(f"  Quality check passed: {quality['passed']}")