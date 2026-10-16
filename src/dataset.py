"""
Dataset utilities for SETI anomaly detection.

Provides PyTorch Dataset classes for signal and spectrogram data,
with support for normal-only training and train/validation splits.
"""

from typing import Optional, Tuple, List, Any, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import logging

logger = logging.getLogger(__name__)


class SignalDataset(Dataset):
    """Dataset for raw signal waveforms."""

    def __init__(self, signals: List[np.ndarray], labels: Optional[List[int]] = None, transform=None):
        """
        Initialize signal dataset.

        Parameters
        ----------
        signals : List[np.ndarray]
            List of 1D signal arrays.
        labels : List[int] or None, default None
            Optional labels for each signal (e.g., 0=normal, 1=RFI, 2=anomaly).
        transform : callable or None, default None
            Optional transform to apply to each signal.
        """
        self.signals = [np.array(s, dtype=np.float32) for s in signals]
        self.labels = labels
        self.transform = transform
        if labels is not None:
            assert len(self.signals) == len(self.labels), "signals and labels must have same length"

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[int]]:
        signal = self.signals[idx]
        if self.transform:
            signal = self.transform(signal)
        signal_tensor = torch.from_numpy(signal)
        if self.labels is not None:
            label = self.labels[idx]
            return signal_tensor, label
        return signal_tensor, None


class SpectrogramDataset(Dataset):
    """Dataset for precomputed spectrograms."""

    def __init__(self, spectrograms: List[np.ndarray], labels: Optional[List[int]] = None,
                 transform=None, normalize: bool = False):
        """
        Initialize spectrogram dataset.

        Parameters
        ----------
        spectrograms : List[np.ndarray]
            List of 2D spectrogram arrays (frequency x time).
        labels : List[int] or None, default None
            Optional labels for each spectrogram.
        transform : callable or None, default None
            Optional transform to apply to each spectrogram.
        normalize : bool, default False
            If True, apply z-score normalization per spectrogram.
        """
        self.spectrograms = [np.array(s, dtype=np.float32) for s in spectrograms]
        self.labels = labels
        self.transform = transform
        self.normalize = normalize
        if labels is not None:
            assert len(self.spectrograms) == len(self.labels), "spectrograms and labels must have same length"

    def __len__(self) -> int:
        return len(self.spectrograms)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[int]]:
        spec = self.spectrograms[idx]
        if self.normalize:
            # Per-spectrogram z-score normalization
            mean = np.mean(spec)
            std = np.std(spec)
            if std > 0:
                spec = (spec - mean) / std
            else:
                spec = spec - mean
        if self.transform:
            spec = self.transform(spec)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0)  # Add channel dimension: (1, H, W)
        if self.labels is not None:
            label = self.labels[idx]
            return spec_tensor, label
        return spec_tensor, None


def create_dataloader_from_signals(
    signals: List[np.ndarray],
    batch_size: int = 32,
    shuffle: bool = True,
    labels: Optional[List[int]] = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a PyTorch DataLoader from raw signals.

    Parameters
    ----------
    signals : List[np.ndarray]
        List of 1D signal arrays.
    batch_size : int, default 32
        Batch size.
    shuffle : bool, default True
        Whether to shuffle data.
    labels : List[int] or None, default None
        Optional labels.
    num_workers : int, default 0
        Number of worker threads for data loading.

    Returns
    -------
    DataLoader
        PyTorch DataLoader yielding (signal_tensor, label_or_None).
    """
    dataset = SignalDataset(signals, labels=labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def create_dataloader_from_spectrograms(
    spectrograms: List[np.ndarray],
    batch_size: int = 32,
    shuffle: bool = True,
    labels: Optional[List[int]] = None,
    num_workers: int = 0,
    normalize: bool = False,
) -> DataLoader:
    """
    Create a PyTorch DataLoader from spectrograms.

    Parameters
    ----------
    spectrograms : List[np.ndarray]
        List of 2D spectrogram arrays.
    batch_size : int, default 32
        Batch size.
    shuffle : bool, default True
        Whether to shuffle data.
    labels : List[int] or None, default None
        Optional labels.
    num_workers : int, default 0
        Number of worker threads for data loading.
    normalize : bool, default False
        If True, apply per-spectrogram normalization.

    Returns
    -------
    DataLoader
        PyTorch DataLoader yielding (spec_tensor, label_or_None).
    """
    dataset = SpectrogramDataset(spectrograms, labels=labels, normalize=normalize)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: Optional[int] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split a dataset into train, validation, and test subsets.

    Parameters
    ----------
    dataset : Dataset
        PyTorch Dataset to split.
    train_ratio : float, default 0.8
        Fraction for training set.
    val_ratio : float, default 0.1
        Fraction for validation set.
    test_ratio : float, default 0.1
        Fraction for test set.
    random_seed : int or None, default None
        Random seed for reproducibility.

    Returns
    -------
    train_dataset, val_dataset, test_dataset : Dataset
        Split datasets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size  # remainder

    generator = torch.Generator().manual_seed(random_seed) if random_seed is not None else None
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    return train_dataset, val_dataset, test_dataset


def create_normal_only_dataloader(
    signals: List[np.ndarray],
    labels: List[int],
    normal_label: int = 0,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader containing only normal-labeled signals.

    Parameters
    ----------
    signals : List[np.ndarray]
        List of signal arrays.
    labels : List[int]
        Corresponding labels.
    normal_label : int, default 0
        Label indicating normal class.
    batch_size : int, default 32
        Batch size.
    shuffle : bool, default True
        Whether to shuffle.
    num_workers : int, default 0
        Number of worker threads.

    Returns
    -------
    DataLoader
        DataLoader for normal signals only.
    """
    normal_signals = [sig for sig, lbl in zip(signals, labels) if lbl == normal_label]
    logger.info(f"Filtered {len(normal_signals)} normal signals from {len(signals)} total")
    return create_dataloader_from_signals(
        normal_signals,
        batch_size=batch_size,
        shuffle=shuffle,
        labels=None,  # No labels needed for normal-only training
        num_workers=num_workers,
    )


def get_dataset_statistics(dataset: Dataset) -> Dict[str, Any]:
    """
    Compute basic statistics for a dataset.

    Parameters
    ----------
    dataset : Dataset
        PyTorch Dataset (SignalDataset or SpectrogramDataset).

    Returns
    -------
    stats : dict
        Dictionary with mean, std, min, max, shape info.
    """
    # We'll compute on a sample to avoid memory issues
    sample_size = min(100, len(dataset))
    indices = np.random.choice(len(dataset), size=sample_size, replace=False)
    samples = [dataset[i][0].numpy() for i in indices]  # Get tensor, convert to numpy

    # Stack along new axis
    stacked = np.stack(samples, axis=0)
    return {
        "mean": float(np.mean(stacked)),
        "std": float(np.std(stacked)),
        "min": float(np.min(stacked)),
        "max": float(np.max(stacked)),
        "shape": list(stacked.shape[1:]),  # Exclude batch dimension
        "num_samples": len(dataset),
    }


if __name__ == "__main__":
    # Self-test
    import numpy as np

    np.random.seed(42)
    # Create dummy signals
    signals = [np.random.randn(1000) for _ in range(20)]
    labels = [0 if i % 5 != 0 else 1 for i in range(20)]  # 80% normal, 20% RFI

    # Test SignalDataset
    sig_dataset = SignalDataset(signals, labels=labels)
    print(f"SignalDataset length: {len(sig_dataset)}")
    sample_signal, sample_label = sig_dataset[0]
    print(f"Sample signal shape: {sample_signal.shape}, label: {sample_label}")

    # Test SpectrogramDataset (using dummy 2D arrays)
    specs = [np.random.randn(64, 64) for _ in range(20)]
    spec_dataset = SpectrogramDataset(specs, labels=labels, normalize=True)
    print(f"SpectrogramDataset length: {len(spec_dataset)}")
    sample_spec, sample_label_spec = spec_dataset[0]
    print(f"Sample spectrogram shape: {sample_spec.shape}, label: {sample_label_spec}")

    # Test split
    train_set, val_set, test_set = split_dataset(sig_dataset, 0.7, 0.2, 0.1, random_seed=42)
    print(f"Split sizes: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    # Test normal-only loader
    normal_loader = create_normal_only_dataloader(signals, labels, normal_label=0, batch_size=4)
    for batch_x, batch_y in normal_loader:
        print(f"Normal batch shape: {batch_x.shape}, labels: {batch_y}")
        break

    print("\nPhase 8 dataset utilities test complete!")