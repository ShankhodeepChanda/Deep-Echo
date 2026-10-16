"""
Data loading and validation for SETI anomaly detection project.

Supports CSV, NPY, NPZ formats for radio signal data.
Validates file integrity, dimensions, data types, and metadata.
Provides dataset summaries for dashboard display.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Supported format detectors
# ──────────────────────────────────────────────────────────────────────

def detect_format(file_path: Path) -> str:
    """
    Detect the format of a data file by extension.

    Returns one of: 'csv', 'npy', 'npz', 'fil', 'h5'
    Raises ValueError for unsupported extensions.
    """
    ext = file_path.suffix.lower()
    if ext == ".csv":
        return "csv"
    if ext in { ".npy", ".npz" }:
        return "npy"
    if ext == ".fil":
        return "fil"
    if ext in { ".h5", ".hdf5" }:
        return "h5"
    raise ValueError(f"Unsupported file extension: {ext}. Supported: .csv, .npy, .npz, .fil, .h5")


# ──────────────────────────────────────────────────────────────────────
# CSV loaders
# ──────────────────────────────────────────────────────────────────────

def load_csv_as_waveform(file_path: Path, expected_dims: int | None = None) -> dict:
    """
    Load a CSV file as a sampled waveform.

    Expected CSV layout:
    - Column 0: time (seconds) or index
    - Column 1: amplitude / voltage value

    Returns dict with keys: 'signal', 'time', 'sample_rate' (estimated), 'source_format'.
    """
    df = pd.read_csv(file_path, header=None)
    values = df.iloc[:, 1].values.astype(float)
    time_col = df.iloc[:, 0].values

    # Try to infer sample rate from time column
    if len(time_col) > 1 and np.all(np.diff(time_col) != 0):
        dt = np.median(np.diff(time_col))
        sample_rate = 1.0 / dt if dt > 0 else None
    else:
        sample_rate = None
        time_col = np.arange(len(values))

    signal = values.reshape(-1)

    if expected_dims and signal.size != expected_dims:
        logger.warning(
            f"CSV signal size {signal.size} does not match expected {expected_dims}"
        )

    return {
        "signal": signal,
        "time": time_col.astype(float),
        "sample_rate": sample_rate,
        "source_format": "csv_waveform",
        "num_samples": len(signal),
        "duration_sec": (time_col[-1] - time_col[0]) if len(time_col) > 1 else float(len(values)),
    }


def load_csv_as_iq(file_path: Path) -> dict:
    """
    Load a CSV file as I/Q (complex) samples.

    Expected CSV layout:
    - Column 0: I component
    - Column 1: Q component

    Returns dict with keys: 'iq_samples', 'sample_rate' (estimated), 'source_format'.
    """
    df = pd.read_csv(file_path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"I/Q CSV must have at least 2 columns, got {df.shape[1]}")

    i_component = df.iloc[:, 0].values.astype(float)
    q_component = df.iloc[:, 1].values.astype(float)
    iq_samples = (i_component + 1j * q_component).astype(complex)

    dt = np.mediandiff(df.iloc[:, 0]) if df.shape[0] > 1 else None
    sample_rate = 1.0 / dt if dt and dt > 0 else None

    return {
        "iq_samples": iq_samples,
        "sample_rate": sample_rate,
        "source_format": "csv_iq",
        "num_samples": len(iq_samples),
    }


def load_csv_as_spectrogram(file_path: Path) -> dict:
    """
    Load a CSV file as a precomputed spectrogram.

    Expected CSV layout:
    - Rows: time frames
    - Columns: frequency bins
    - Values: power or magnitude

    Returns dict with keys: 'spectrogram', 'shape', 'source_format'.
    """
    df = pd.read_csv(file_path, header=None)
    matrix = df.values.astype(float)

    # Replace non-finite values with small positive floor
    matrix = np.where(np.isnan(matrix) | np.isinf(matrix), 1e-10, matrix)

    return {
        "spectrogram": matrix,
        "shape": matrix.shape,
        "source_format": "csv_spectrogram",
        "num_rows": matrix.shape[0],
        "num_cols": matrix.shape[1],
    }


# ──────────────────────────────────────────────────────────────────────
# NPY/NPZ loaders
# ──────────────────────────────────────────────────────────────────────

def load_npy(file_path: Path) -> dict:
    """
    Load a .npy file containing a real-valued 1D signal.

    Returns dict with keys: 'signal', 'shape', 'source_format', 'dtype'.
    """
    arr = np.load(file_path)
    if arr.ndim != 1:
        logger.warning(f".npy file has {arr.ndim} dimensions, expected 1D")

    return {
        "signal": arr.astype(float),
        "shape": arr.shape,
        "source_format": "npy_1d",
        "dtype": str(arr.dtype),
        "num_samples": arr.size,
    }


def load_npz_signals(file_path: Path, key: str = "signals") -> dict:
    """
    Load a .npz file containing signal arrays.

    Expected NPZ structure:
    - Key: array name (default: "signals")
    - Value: 2D array of shape (n_samples, n_timesteps) or dict of arrays

    Returns dict with loaded arrays keyed by their NPZ keys.
    """
    data = np.load(file_path)
    result = {}

    if isinstance(data, dict):
        for k in data.keys():
            arr = data[k]
            if arr.ndim == 1:
                result[f"{k}_1d"] = {
                    "signal": arr.astype(float),
                    "shape": arr.shape,
                    "source_format": f"npz_1d_{k}",
                    "dtype": str(arr.dtype),
                    "num_samples": arr.size,
                }
            elif arr.ndim == 2:
                result[f"{k}_2d"] = {
                    "signal": arr.astype(float),
                    "shape": arr.shape,
                    "source_format": f"npz_2d_{k}",
                    "dtype": str(arr.dtype),
                    "num_samples": arr.shape[0],
                }
            else:
                result[f"{k}_n"] = {
                    "signal": arr.astype(float).ravel(),
                    "shape": arr.shape,
                    "source_format": f"npz_n_{k}",
                    "dtype": str(arr.dtype),
                    "num_samples": arr.size,
                }
    else:
        raise KeyError(f"NPZ file has no data dictionary; got {type(data)}")

    return result


# ──────────────────────────────────────────────────────────────────────
# Generic loader dispatch
# ──────────────────────────────────────────────────────────────────────

def load_data(file_path: Path, *args, **kwargs) -> dict:
    """
    Dispatch to the appropriate loader based on file extension.

    Parameters passed through to the specific loader.

    Returns the dict returned by the chosen loader.
    """
    fmt = detect_format(file_path)
    if fmt == "fil":
        from src.filterbank_reader import load_filterbank_waterfall
        return load_filterbank_waterfall(file_path, *args, **kwargs)
    elif fmt == "npy":
        if file_path.suffix.lower() == ".npz":
            return load_npz_signals(file_path, *args, **kwargs)
        return load_npy(file_path, *args, **kwargs)
    elif fmt == "csv":
        return load_csv_as_waveform(file_path, *args, **kwargs)
    else:
        raise ValueError(f"No loader available for format: {fmt}")


# ──────────────────────────────────────────────────────────────────────
# Data validation
# ──────────────────────────────────────────────────────────────────────

def validate_signal(signal: np.ndarray, allow_nan: bool = False, allow_inf: bool = False) -> dict:
    """
    Validate a signal array.

    Checks:
    - Is numpy array.
    - Has valid (finite) values unless allow_nan/allow_inf set.
    - Has sensible length (> 1 sample).
    - Optional: NaN/infinity counts.

    Returns dict with keys: 'valid', 'nan_count', 'inf_count', 'issues'.
    """
    issues = []

    if not isinstance(signal, np.ndarray):
        return {
            "valid": False,
            "nan_count": 0,
            "inf_count": 0,
            "issues": ["signal must be a numpy ndarray"],
        }

    nan_count = int(np.sum(np.isnan(signal)))
    inf_count = int(np.sum(np.isinf(signal)))

    if not allow_nan and nan_count > 0:
        issues.append(f"Found {nan_count} NaN values")
        return {"valid": False, "nan_count": nan_count, "inf_count": inf_count, "issues": issues}

    if not allow_inf and inf_count > 0:
        issues.append(f"Found {inf_count} infinity values")
        return {"valid": False, "nan_count": nan_count, "inf_count": inf_count, "issues": issues}

    if signal.size < 2:
        issues.append("Signal must have at least 2 samples")
        return {"valid": False, "nan_count": nan_count, "inf_count": inf_count, "issues": issues}

    return {
        "valid": True,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "issues": issues,
    }


def validate_file(file_path: Path) -> dict:
    """
    Validate a data file without loading full contents.

    Checks:
    - File exists.
    - Has supported extension.
    - File size is reasonable (not 0 bytes, not unreasonably large).

    Returns dict with keys: 'exists', 'format', 'size_bytes', 'valid', 'issues'.
    """
    issues = []

    if not file_path.exists():
        return {
            "exists": False,
            "format": None,
            "size_bytes": 0,
            "valid": False,
            "issues": [f"File does not exist: {file_path}"],
        }

    size_bytes = file_path.stat().st_size
    if size_bytes == 0:
        issues.append("File is 0 bytes")
    if size_bytes > 500 * 1024 * 1024:  # 500 MB
        issues.append(f"File is very large ({size_bytes / (1024*1024):.1f} MB); consider downsampling")

    try:
        fmt = detect_format(file_path)
    except ValueError as e:
        fmt = None
        issues.append(str(e))

    return {
        "exists": file_path.exists(),
        "format": fmt,
        "size_bytes": size_bytes,
        "valid": len(issues) == 0,
        "issues": issues,
    }


# ──────────────────────────────────────────────────────────────────────
# Dataset summary
# ──────────────────────────────────────────────────────────────────────

def summarize_dataset(entries: list[dict]) -> dict:
    """
    Produce a summary statistic dict from a list of loaded data entries.

    Each entry should be a dict returned by one of the loaders,
    expected to contain at least 'num_samples' and 'source_format'.

    Returns dict with keys:
    - total_samples, num_files, format_breakdown, duration_range, amplitude_stats
    """
    if not entries:
        return {
            "total_samples": 0,
            "num_files": 0,
            "format_breakdown": {},
            "duration_range": (0, 0),
            "amplitude_stats": {"mean": 0, "std": 0, "min": 0, "max": 0},

        }

    total_samples = sum(e.get("num_samples", 0) for e in entries)
    num_files = len(entries)

    # Format breakdown
    format_breakdown = {}
    for e in entries:
        fmt = e.get("source_format", "unknown")
        format_breakdown[fmt] = format_breakdown.get(fmt, 0) + 1

    # Duration range (if available)
    durations = []
    for e in entries:
        d = e.get("duration_sec")
        if d is not None and isinstance(d, (int, float)) and d > 0:
            durations.append(d)
    duration_range = (min(durations), max(durations)) if durations else (0, 0)

    # Amplitude stats (if signal data available)
    all_amplitudes = []
    for e in entries:
        sig = e.get("signal")
        if sig is not None and isinstance(sig, np.ndarray):
            all_amplitudes.extend(sig.abs().tolist())

    if all_amplitudes:
        import numpy as np
        amplitude_stats = {
            "mean": float(np.mean(all_amplitudes)),
            "std": float(np.std(all_amplitudes)),
            "min": float(np.min(all_amplitudes)),
            "max": float(np.max(all_amplitudes)),
        }
    else:
        amplitude_stats = {"mean": 0, "std": 0, "min": 0, "max": 0}

    return {
        "total_samples": total_samples,
        "num_files": num_files,
        "format_breakdown": format_breakdown,
        "duration_range": duration_range,
        "amplitude_stats": amplitude_stats,
    }


# ──────────────────────────────────────────────────────────────────────
# Main CLI entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print(f"Usage: python -m src.data_loader <file_path> [key]")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    result = load_data(file_path, *sys.argv[2:])
    print(json.dumps(result, default=str, indent=2))