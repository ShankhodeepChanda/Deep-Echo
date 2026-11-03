"""
Model access, tile scoring and full-observation scans.

Tiles are normalised exactly as ``extract_spectrogram_tiles`` does (per-tile
min-max to [0, 1]) so scores here match the scores in the stored rankings.
Scores are the autoencoder's mean squared reconstruction error: an
"unusualness" measure, not a source classification.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch

from src.config import CACHE_DIR, MODELS_DIR
from src.models.cnn_autoencoder import CNNAutoencoder
from src.observatory.provenance import TILE_F, TILE_T, grid_shape
from src.observatory.sources import ObservationSource, ProgressCallback
from src.training import compute_reconstruction_errors

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = MODELS_DIR / "cnn_autoencoder.pt"
_DEVICE = "cpu"
_BATCH = 512
_FREQ_CHUNK_BLOCKS = 2048
_TARGET_BATCH_TILES = 4096              # tiles gathered per read when the grid is narrow in frequency
_MAX_READ_BYTES = 256 * 1024 * 1024    # cap on raw data held in memory per read


class ModelUnavailableError(Exception):
    """The model checkpoint is missing or unreadable."""


@dataclass
class ModelBundle:
    model: CNNAutoencoder
    path: Path
    latent_dim: int
    checkpoint_threshold: Optional[float]
    signature: str


def load_model(path: Path = DEFAULT_MODEL_PATH) -> ModelBundle:
    """Load a trained autoencoder checkpoint (read-only; the file is never modified)."""
    path = Path(path)
    if not path.exists():
        raise ModelUnavailableError(f"Model checkpoint not found: {path.name}")
    try:
        ckpt = torch.load(path, map_location=_DEVICE, weights_only=False)
        latent = int(ckpt.get("latent_dim", 48))
        model = CNNAutoencoder(latent_dim=latent)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
    except Exception as exc:  # corrupt file, shape mismatch, missing key
        raise ModelUnavailableError(f"Could not load {path.name}: {exc}") from exc
    stat = path.stat()
    signature = hashlib.sha1(f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:10]
    thr = ckpt.get("threshold")
    return ModelBundle(model, path, latent, float(thr) if thr is not None else None, signature)


def normalize_tiles(tiles: np.ndarray) -> np.ndarray:
    """Per-tile min-max to [0, 1] for an (N, H, W) stack; flat tiles become zeros. NaN/inf are zeroed first."""
    tiles = np.nan_to_num(np.asarray(tiles, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = tiles.min(axis=(1, 2), keepdims=True)
    span = tiles.max(axis=(1, 2), keepdims=True) - lo
    ok = span > 1e-8
    return np.where(ok, (tiles - lo) / np.where(ok, span, 1.0), 0.0).astype(np.float32)


def score_windows(bundle: ModelBundle, windows: Sequence[np.ndarray]) -> np.ndarray:
    """Unusualness scores for raw (64, 64) windows."""
    stack = normalize_tiles(np.stack([np.asarray(w, dtype=np.float32) for w in windows]))
    return compute_reconstruction_errors(bundle.model, stack, batch_size=_BATCH, device=_DEVICE)


def reconstruct(bundle: ModelBundle, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """(normalised tile, reconstruction, score) for one raw window, all 64 x 64."""
    tile = normalize_tiles(np.asarray(window, dtype=np.float32)[None])
    with torch.no_grad():
        recon = bundle.model(torch.from_numpy(tile[:, None]).to(_DEVICE)).cpu().numpy()[0, 0]
    score = float(np.mean((recon - tile[0]) ** 2))
    return tile[0], recon, score


# ---------------------------------------------------------------------------
# Full-observation scan
# ---------------------------------------------------------------------------
@dataclass
class ScanResult:
    """Scores for every non-overlapping 64 x 64 tile of an observation."""

    scores: np.ndarray            # (tiles in time, tiles in frequency)
    model_signature: str
    source_fingerprint: str
    seconds: float = 0.0

    @property
    def n_tiles(self) -> int:
        return int(self.scores.size)


def scan_cache_path(source_fingerprint: str, model_signature: str) -> Path:
    return CACHE_DIR / "scans" / f"{source_fingerprint}_{model_signature}.npz"


def load_cached_scan(source_fingerprint: str, model_signature: str) -> Optional[ScanResult]:
    path = scan_cache_path(source_fingerprint, model_signature)
    if not path.exists():
        return None
    try:
        with np.load(path) as z:
            return ScanResult(z["scores"], model_signature, source_fingerprint, float(z["seconds"]))
    except Exception:
        logger.warning("Ignoring unreadable scan cache %s", path)
        return None


def scan_observation(
    source: ObservationSource,
    bundle: ModelBundle,
    progress: Optional[ProgressCallback] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> ScanResult:
    """Score every non-overlapping tile of the observation and cache the result.

    Streams 64 time samples at a time and scores frequency chunks, so memory
    stays bounded for multi-gigabyte files. The source file is only read.
    """
    meta = source.meta
    n_t, n_f = grid_shape(meta.n_times, meta.nchans)
    if n_t < 1 or n_f < 1:
        raise ValueError(
            f"Observation is smaller than one {TILE_T} x {TILE_F} tile ({meta.n_times} samples x {meta.nchans} channels)."
        )
    started = time.time()
    scores = np.zeros((n_t, n_f), dtype=np.float32)

    # Read several 64-sample time blocks at once so the model sees large batches, but cap the raw bytes held in memory.
    block_bytes = TILE_T * meta.nchans * max(meta.nifs, 1) * 4
    group = int(max(1, min(n_t, _TARGET_BATCH_TILES // n_f, _MAX_READ_BYTES // block_bytes)))
    for t_blk in range(0, n_t, group):
        if should_stop and should_stop():
            raise InterruptedError("Scan cancelled")
        k = min(group, n_t - t_blk)
        rows = source.read_rows(t_blk * TILE_T, (t_blk + k) * TILE_T)          # (k * 64, nchans)
        for fb0 in range(0, n_f, _FREQ_CHUNK_BLOCKS):
            fb1 = min(fb0 + _FREQ_CHUNK_BLOCKS, n_f)
            nb = fb1 - fb0
            block = rows[:, fb0 * TILE_F:fb1 * TILE_F].reshape(k, TILE_T, nb, TILE_F).transpose(0, 2, 1, 3).reshape(k * nb, TILE_T, TILE_F)
            scores[t_blk:t_blk + k, fb0:fb1] = compute_reconstruction_errors(
                bundle.model, normalize_tiles(block), batch_size=_BATCH, device=_DEVICE
            ).reshape(k, nb)
        if progress:
            progress((t_blk + k) / n_t)
    result = ScanResult(scores, bundle.signature, meta.fingerprint, time.time() - started)

    path = scan_cache_path(meta.fingerprint, bundle.signature)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, scores=scores, seconds=result.seconds)
        tmp.replace(path)
    except OSError:
        logger.warning("Could not cache scan at %s", path)
    return result
