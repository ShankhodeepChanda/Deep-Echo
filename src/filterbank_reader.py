"""
Radio Astronomy Dynamic Spectrum & Filterbank Data Reader.

Supports:
- SIGPROC filterbank (.fil) binary files via pure Python stream parser.
- HDF5 (.h5) Breakthrough Listen dynamic spectrum files (if h5py available).
- Extraction of 2D waterfall dynamic spectra (time x frequency).
- Sub-band channel frequency slicing and patch extraction for CNN tiling.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import struct
import numpy as np
import logging

logger = logging.getLogger(__name__)


def read_filterbank_header(file_path: Union[str, Path]) -> Tuple[Dict[str, Any], int]:
    """
    Parse SIGPROC filterbank (.fil) binary header.
    
    Returns
    -------
    header : dict
        Header keywords and metadata.
    header_bytes : int
        Byte offset where the raw spectral data begins.
    """
    file_path = Path(file_path)
    header: Dict[str, Any] = {}

    with open(file_path, "rb") as f:
        # Helper to read next string
        def read_string() -> str:
            size_bytes = f.read(4)
            if not size_bytes or len(size_bytes) < 4:
                return ""
            size = struct.unpack("<I", size_bytes)[0]
            if size > 1000:  # Safety check
                return ""
            raw = f.read(size)
            return raw.decode("utf-8", errors="replace")

        first_key = read_string()
        if first_key != "HEADER_START":
            raise ValueError(f"Not a valid SIGPROC filterbank file (missing HEADER_START in {file_path.name})")

        while True:
            key = read_string()
            if not key or key == "HEADER_END":
                break

            # Handle known header value types
            if key in {"source_name", "rawdatafile"}:
                header[key] = read_string()
            elif key in {"telescope_id", "machine_id", "data_type", "nchans", "nbits", "nifs", "barycentric", "pulsarcentric", "nbeams", "ibeam"}:
                bytes_val = f.read(4)
                if len(bytes_val) == 4:
                    header[key] = struct.unpack("<I", bytes_val)[0]
            elif key in {"tstart", "tsamp", "fch1", "foff", "refdm", "period"}:
                bytes_val = f.read(8)
                if len(bytes_val) == 8:
                    header[key] = struct.unpack("<d", bytes_val)[0]
            elif key in {"src_raj", "src_dej", "az_start", "za_start"}:
                bytes_val = f.read(8)
                if len(bytes_val) == 8:
                    header[key] = struct.unpack("<d", bytes_val)[0]
            else:
                # Unknown key, attempt to skip or break if corrupted
                logger.debug(f"Skipping unknown header key: {key}")
                break

        header_bytes = f.tell()

    return header, header_bytes


def load_filterbank_waterfall(
    file_path: Union[str, Path],
    max_time_steps: Optional[int] = None,
    f_start_idx: Optional[int] = None,
    f_stop_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Load dynamic spectrum waterfall (time x frequency) from a SIGPROC .fil file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to .fil file.
    max_time_steps : int, optional
        Maximum time samples to read (to avoid memory explosion).
    f_start_idx : int, optional
        Starting frequency channel index.
    f_stop_idx : int, optional
        Ending frequency channel index.

    Returns
    -------
    dict with keys:
        'waterfall': 2D np.ndarray (n_times, n_freqs)
        'header': header metadata
        'time_axis': 1D array of time stamps (seconds from start)
        'freq_axis': 1D array of channel center frequencies (MHz)
    """
    file_path = Path(file_path)
    header, data_offset = read_filterbank_header(file_path)

    nchans = int(header.get("nchans", 1))
    nbits = int(header.get("nbits", 32))
    nifs = int(header.get("nifs", 1))
    tsamp = float(header.get("tsamp", 1.0))
    fch1 = float(header.get("fch1", 0.0))
    foff = float(header.get("foff", 1.0))

    dtype_map = {8: np.uint8, 16: np.uint16, 32: np.float32}
    dtype = dtype_map.get(nbits, np.float32)

    f_start = f_start_idx or 0
    f_stop = f_stop_idx or nchans

    file_size = file_path.stat().st_size
    data_size = file_size - data_offset
    bytes_per_sample = nbits // 8
    samples_per_timestep = nchans * nifs * bytes_per_sample

    total_time_steps = data_size // samples_per_timestep
    if max_time_steps:
        time_steps_to_read = min(total_time_steps, max_time_steps)
    else:
        time_steps_to_read = total_time_steps

    with open(file_path, "rb") as f:
        f.seek(data_offset)
        count = time_steps_to_read * nchans * nifs
        raw_data = np.fromfile(f, dtype=dtype, count=count)

    if nifs > 1:
        # Multipolarization / multi-IF: extract Stokes I total intensity
        shaped = raw_data.reshape(time_steps_to_read, nifs, nchans)
        if nifs >= 2:
            waterfall_full = (shaped[:, 0, :] + shaped[:, 1, :]) / 2.0
        else:
            waterfall_full = shaped[:, 0, :]
        waterfall = waterfall_full[:, f_start:f_stop]
    else:
        waterfall = raw_data.reshape(time_steps_to_read, nchans)[:, f_start:f_stop]

    freq_axis = fch1 + np.arange(f_start, f_stop) * foff
    time_axis = np.arange(time_steps_to_read) * tsamp

    return {
        "waterfall": waterfall.astype(np.float32),
        "header": header,
        "time_axis": time_axis,
        "freq_axis": freq_axis,
        "n_times": time_steps_to_read,
        "n_chans": f_stop - f_start,
    }


def extract_spectrogram_tiles(
    waterfall: np.ndarray,
    tile_h: int = 64,
    tile_w: int = 64,
    step_h: Optional[int] = None,
    step_w: Optional[int] = None,
) -> np.ndarray:
    """
    Tile a large 2D dynamic spectrum (time x frequency) into fixed-size
    patches ready for the CNN autoencoder.
    
    Parameters
    ----------
    waterfall : np.ndarray
        2D array of shape (T, F).
    tile_h : int, default 64
        Time dimension height for each tile.
    tile_w : int, default 64
        Frequency dimension width for each tile.
    step_h : int, optional
        Vertical stride (default tile_h).
    step_w : int, optional
        Horizontal stride (default tile_w).

    Returns
    -------
    tiles : np.ndarray
        Array of shape (N, 1, tile_h, tile_w) normalized to [0, 1].
    """
    T, F = waterfall.shape
    sh = step_h or tile_h
    sw = step_w or tile_w

    tiles = []
    for t_idx in range(0, T - tile_h + 1, sh):
        for f_idx in range(0, F - tile_w + 1, sw):
            tile = waterfall[t_idx : t_idx + tile_h, f_idx : f_idx + tile_w].copy()
            # Normalize tile to [0, 1]
            t_min, t_max = tile.min(), tile.max()
            if t_max - t_min > 1e-8:
                tile = (tile - t_min) / (t_max - t_min)
            else:
                tile = np.zeros_like(tile)
            tiles.append(tile)

    if not tiles:
        # If waterfall is smaller than tile dimensions, resize directly
        from scipy.ndimage import zoom
        zh = tile_h / max(T, 1)
        zw = tile_w / max(F, 1)
        resized = zoom(waterfall, (zh, zw), order=1)
        t_min, t_max = resized.min(), resized.max()
        if t_max - t_min > 1e-8:
            resized = (resized - t_min) / (t_max - t_min)
        tiles.append(resized)

    tiles_arr = np.array(tiles, dtype=np.float32)
    return np.expand_dims(tiles_arr, 1)  # (N, 1, H, W)
