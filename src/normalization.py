"""
Spectrogram normalization for SETI anomaly detection.

Implements configurable normalization consistent with model training
and inference pipelines. Fits dataset-level statistics on training data
only and reuses them during validation, testing, and inference.
"""

from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SpectrogramNormalizer:
    """
    Normalizer that fits on training data and reuses during inference.

    Purpose:
        Prepare spectrograms consistently for model training and inference.
        The normalizer stores statistics after fitting and applies the same
        transformation during validation, testing, and inference.
    """

    def __init__(self, method: str = "standard", clip_percentile: Optional[float] = None):
        """
        Initialize normalizer.

        Parameters
        ----------
        method : {'standard', 'minmax', 'log_standard'}, default 'standard'
            Normalization method.
        clip_percentile : float or None, default None
            Percentile for clipping outliers before normalization (e.g., 99.0).
        """
        self.method = method
        self.clip_percentile = clip_percentile
        self.fitted_ = None
        self._fitted = False

    def fit(self, spectrograms: List[np.ndarray]):
        """
        Fit normalizer on a list of spectrograms (training data only).

        Parameters
        ----------
        spectrograms : List[np.ndarray]
            List of 2D spectrogram arrays (frequency x time).
        """
        if self.clip_percentile is not None:
            # Clip all spectrograms to percentile bounds first
            clipped_arrays = []
            for sg in spectrograms:
                low_val = np.percentile(sg, self.clip_percentile / 2)
                high_val = np.percentile(sg, 100 - self.clip_percentile / 2)
                clipped = np.clip(sg, low_val, high_val)
                clipped_arrays.append(clipped)
        else:
            clipped_arrays = [np.array(sg) for sg in spectrograms]

        if self.method == "standard":
            # Compute global mean and std across all spectrograms
            all_values = np.concatenate([s.flatten() for s in clipped_arrays])
            self.fitted_ = {"mean": float(np.mean(all_values)), "std": float(np.std(all_values))}
        elif self.method == "minmax":
            all_values = np.concatenate([s.flatten() for s in clipped_arrays])
            self.fitted_ = {"min": float(np.min(all_values)), "max": float(np.max(all_values))}
        elif self.method == "log_standard":
            # First apply log, then compute stats
            log_arrays = [np.log10(sg + 1e-10) for sg in clipped_arrays]
            all_values = np.concatenate([l.flatten() for l in log_arrays])
            self.fitted_ = {"mean": float(np.mean(all_values)), "std": float(np.std(all_values))}
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        self._fitted = True
        logger.info(f"SpectrogramNormalizer fitted: method={self.method}, stats={self.fitted_}")

    def transform(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Normalize a single spectrogram using fitted statistics.

        Parameters
        ----------
        spectrogram : np.ndarray
            2D spectrogram array (frequency x time).

        Returns
        -------
        normalized : np.ndarray
            Normalized spectrogram.
        """
        if not self._fitted:
            raise RuntimeError("Normalizer must be fitted before transform()")

        sg = np.array(spectrogram, dtype=float)

        # Apply clipping if configured and not already done
        if self.clip_percentile is not None:
            low_val = np.percentile(sg, self.clip_percentile / 2)
            high_val = np.percentile(sg, 100 - self.clip_percentile / 2)
            sg = np.clip(sg, low_val, high_val)

        if self.method == "standard":
            mean = self.fitted_["mean"]
            std = self.fitted_["std"]
            if std > 0:
                return (sg - mean) / std
            return sg - mean

        elif self.method == "minmax":
            mn = self.fitted_["min"]
            mx = self.fitted_["max"]
            if mx > mn:
                return (sg - mn) / (mx - mn)
            return np.zeros_like(sg)

        elif self.method == "log_standard":
            log_sg = np.log10(sg + 1e-10)
            mean = self.fitted_["mean"]
            std = self.fitted_["std"]
            if std > 0:
                return (log_sg - mean) / std
            return log_sg - mean

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def fit_transform(self, spectrograms: List[np.ndarray]):
        """
        Fit then transform in one step.

        Parameters
        ----------
        spectrograms : List[np.ndarray]
            List of 2D spectrogram arrays (training data).

        Returns
        -------
        normalized : List[np.ndarray]
            List of normalized spectrograms.
        """
        self.fit(spectrograms)
        return [self.transform(sg) for sg in spectrograms]

    def save(self, path: Path) -> None:
        """
        Save fitted statistics to file.

        Parameters
        ----------
        path : Path
            File path to save (numpy .npz format).
        """
        if not self._fitted:
            raise RuntimeError("Normalizer must be fitted before save()")
        np.savez(path, method=self.method, **self.fitted_)
        logger.info(f"Normalizer saved to {path}")

    def load(self, path: Path) -> None:
        """
        Load fitted statistics from file.

        Parameters
        ----------
        path : Path
            File path to load from (numpy .npz format).
        """
        data = np.load(path, allow_pickle=True)
        self.method = data["method"].item()
        self.fitted_ = {"mean": float(data["mean"]), "std": float(data["std"])}
        self._fitted = True
        logger.info(f"Normalizer loaded from {path} (method={self.method})")


def normalize_spectrogram_array(
    spectrograms: List[np.ndarray],
    method: str = "standard",
    clip_percentile: Optional[float] = None,
    return_stats: False,
) -> Tuple[List[np.ndarray], dict]:
    """
    Normalize an array of spectrograms using the specified method.

    Parameters
    ----------
    spectrograms : List[np.ndarray]
        List of 2D spectrogram arrays (frequency x time).
    method : str, default 'standard'
        Normalization method.
    clip_percentile : float or None
        Percentile for clipping outliers before normalization.
    return_stats : bool, default False
        If True, return fitted statistics dict alongside normalized arrays.

    Returns
    -------
    normalized : List[np.ndarray]
        List of normalized spectrograms.
    stats : dict
        Fitted statistics (only if return_stats=True).
    """
    normalizer = SpectrogramNormalizer(method=method, clip_percentile=clip_percentile)
    normalizer.fit(spectrograms)
    normalized = [normalizer.transform(sg) for sg in spectrograms]

    stats = {
        "method": method,
        "clip_percentile": clip_percentile if clip_percentile is not None else "none",
        "fitted": normalizer._fitted,
    }
    if normalizer._fitted:
        stats.update(normalizer.fitted_)

    if return_stats:
        return normalized, stats
    return normalized, {}


def normalize_for_autoencoder_input(
    spectrogram: np.ndarray,
    *,
    normalization: str = "standard",
    mean: Optional[float] = None,
    std: Optional[float] = None,
    min_: Optional[float] = None,
    max_: Optional[float] = None,
) -> np.ndarray:
    """
    Normalize spectrogram for CNN autoencoder input.

    The autoencoder expects normalized input typically in range [0, 1]
    or standardized (zero mean, unit variance).

    Parameters
    ----------
    spectrogram : np.ndarray
        2D spectrogram (frequency x time).
    normalization : str, default 'standard'
        One of: 'standard', 'minmax'.
    mean : float or None
        Pre-computed mean (for inference with fitted stats).
    std : float or None
        Pre-computed std (for inference with fitted stats).
    min_ : float or None
        Pre-computed min (for minmax inference).
    max_ : float or None
        Pre-computed max (for minmax inference).

    Returns
    -------
    normalized : np.ndarray
        Spectrogram normalized for model input.
    """
    sg = np.array(spectrogram, dtype=float)

    if normalization == "standard":
        if mean is not None and std is not None:
            # Use pre-computed stats (inference)
            if std > 0:
                return (sg - mean) / std
            return sg - mean
        # Fallback: compute from this spectrogram (training)
        return (sg - np.mean(sg)) / np.std(sg)

    elif normalization == "minmax":
        if min_ is not None and max_ is not None:
            # Use pre-computed stats (inference)
            if max_ > min_:
                return (sg - min_) / (max_ - min_)
            return np.zeros_like(sg)
        # Fallback: compute from this spectrogram (training)
        if sg.max > sg.min:
            return (sg - sg.min) / (sg.max - sg.min)
        return np.zeros_like(sg)

    else:
        raise ValueError(f"Unknown normalization: {normalization}")


# ──────────────────────────────────────────────────────────────────────
# Example/self-test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)

    # Generate test spectrograms
    n_sg = 5
    sg_list = []
    for i in range(n_sg):
        # Create varying spectrograms
        duration = 1.0
        fs = 1000.0
        t = np.arange(0, duration, 1 / fs)

        # Signal: chirp + noise
        sig = np.sin(2 * np.pi * (50 + i * 20) * t ** 1.5) + 0.1 * np.random.randn(len(t))

        # Compute spectrogram
        from scipy import signal
        f, t_spec, Sxx = signal.stft(sig, fs=fs, nperseg=128, noverlap=64)
        Sxx = np.abs(Sxx)**2  # Power spectrogram
        sg_list.append(Sxx)

    print(f"Generated {n_sg} spectrograms")
    for i, sg in enumerate(sg_list):
        print(f"  Spectrogram {i}: shape={sg.shape}, range={sg.min():.4e}-{sg.max():.4e}")

    # Standard normalization
    normalized, stats = normalize_spectrogram_array(sg_list, method="standard", return_stats=True)
    print(f"\nStandard normalization stats: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
    for i, ng in enumerate(normalized):
        print(f"  Normalized {i}: mean={ng.mean():.4f:.4f}, std={ng.std():.4f:.4f}")

    # MinMax normalization
    normalized_mm, stats_mm = normalize_spectrogram_array(sg_list, method="minmax", return_stats=True)
    print(f"\nMinMax normalization: min={stats_mm['min']:.6f}, max={stats_mm['max']:.6f}")
    for i, ng in enumerate(normalized_mm):
        print(f"  MinMax {i}: range={ng.min():.4f}-{ng.max():.4f}")

    # Inference with pre-computed stats
    new_sg = sg_list[0]
    normalized_inf, _ = normalize_spectrogram_array([new_sg], method="standard",
                                                    return_stats=False)
    print(f"\nInference normalization (first spectrogram): mean={normalized_inf[0].mean():.4f:.4f}, "
          f"std={normalized_inf[0].std():.4f:.4f}")

    # Autoencoder input normalization
    autoencoder_normalized = normalize_for_autoencoder_input(sg_list[0], normalization="minmax")
    print(f"\nAutoencoder minmax input: range={autoencoder_normalized.min():.4f}-{autoencoder_normalized.max():.4f}")

    print("\nPhase 6 Spectrogram Normalization complete!")