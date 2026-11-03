"""
Observation sources: validation, metadata and memory-conscious data access.

Filterbank (.fil) files are read through ``numpy.memmap`` so a window or a
block of rows is read without loading the whole file. HDF5 (.h5/.hdf5)
dynamic spectra are read through ``h5py`` when it is installed; if it is not,
callers get a clear ``OptionalDependencyError`` and the rest of the app keeps
working.

Header parsing reuses ``src.filterbank_reader.read_filterbank_header``. Total
intensity follows the existing loader (``load_filterbank_waterfall``): with two
or more IFs the first two are averaged, otherwise the single IF is used.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np

from src.config import CACHE_DIR
from src.filterbank_reader import read_filterbank_header

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".fil", ".h5", ".hdf5")
_FIL_DTYPES = {8: np.uint8, 16: np.uint16, 32: np.float32}
_MJD_EPOCH = datetime(1858, 11, 17)

ProgressCallback = Callable[[float], None]


class SourceError(Exception):
    """The file cannot be read as an observation."""


class OptionalDependencyError(SourceError):
    """A package needed for this format is not installed."""


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def band_name(center_mhz: float) -> Optional[str]:
    """IEEE radar-band letter for a centre frequency, or None if outside 1-18 GHz."""
    for low, high, name in (
        (1000, 2000, "L-band"),
        (2000, 4000, "S-band"),
        (4000, 8000, "C-band"),
        (8000, 12000, "X-band"),
        (12000, 18000, "Ku-band"),
    ):
        if low <= center_mhz < high:
            return name
    return None


def telescope_label(telescope_id: Optional[int]) -> Optional[str]:
    """Telescope name for a SIGPROC id, or None when the id is not known."""
    if telescope_id is None:
        return None
    from src.sequential_trainer import TELESCOPE_NAMES

    return TELESCOPE_NAMES.get(int(telescope_id))


@dataclass(frozen=True)
class SourceMeta:
    """Header-level facts about one observation file. Missing values are None."""

    path: Path
    fmt: str                     # "fil" or "h5"
    source_name: str
    nchans: int
    nifs: int
    n_times: int
    tsamp_s: float
    fch1_mhz: float
    foff_mhz: float
    size_bytes: int
    nbits: int = 32
    telescope_id: Optional[int] = None
    tstart_mjd: Optional[float] = None
    header_bytes: int = 0
    h5_dataset: str = "data"

    @property
    def freq_low_mhz(self) -> float:
        """Lower edge of the covered band (channel edges, not centres)."""
        return min(self.fch1_mhz, self.fch1_mhz + (self.nchans - 1) * self.foff_mhz) - abs(self.foff_mhz) / 2

    @property
    def freq_high_mhz(self) -> float:
        return max(self.fch1_mhz, self.fch1_mhz + (self.nchans - 1) * self.foff_mhz) + abs(self.foff_mhz) / 2

    @property
    def center_mhz(self) -> float:
        return (self.freq_low_mhz + self.freq_high_mhz) / 2

    @property
    def duration_s(self) -> float:
        return self.n_times * self.tsamp_s

    @property
    def band(self) -> Optional[str]:
        return band_name(self.center_mhz)

    @property
    def telescope(self) -> Optional[str]:
        return telescope_label(self.telescope_id)

    @property
    def start_utc(self) -> Optional[datetime]:
        if self.tstart_mjd is None or not math.isfinite(self.tstart_mjd) or self.tstart_mjd <= 0:
            return None
        try:
            return _MJD_EPOCH + timedelta(days=float(self.tstart_mjd))
        except OverflowError:
            return None

    def channel_freq_mhz(self, channel: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Centre frequency of a channel index."""
        return self.fch1_mhz + np.asarray(channel) * self.foff_mhz

    def freq_to_channel(self, freq_mhz: float) -> int:
        """Nearest channel index for a frequency; raises if outside the band."""
        idx = int(round((freq_mhz - self.fch1_mhz) / self.foff_mhz))
        if idx < 0 or idx >= self.nchans:
            raise ValueError(f"{freq_mhz:.3f} MHz is outside {self.freq_low_mhz:.1f}-{self.freq_high_mhz:.1f} MHz")
        return idx

    @property
    def fingerprint(self) -> str:
        """Short id that changes if the file is replaced (path, size, mtime)."""
        try:
            stat = self.path.stat()
            raw = f"{self.path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        except OSError:
            raw = str(self.path)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


@dataclass
class ValidationReport:
    """Outcome of checking a candidate observation file."""

    path: Path
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    meta: Optional[SourceMeta] = None
    optional_dependency_missing: Optional[str] = None


# ---------------------------------------------------------------------------
# Filterbank
# ---------------------------------------------------------------------------
def _stokes_i(block: np.ndarray) -> np.ndarray:
    """Collapse a (rows, nifs, chans) block to total intensity (rows, chans)."""
    if block.shape[1] >= 2:
        return (block[:, 0, :].astype(np.float32) + block[:, 1, :].astype(np.float32)) * 0.5
    return np.array(block[:, 0, :], dtype=np.float32)  # always a copy, never a view of the memory map


class FilterbankSource:
    """Memory-mapped SIGPROC filterbank."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.meta = self._read_meta(self.path)

    @staticmethod
    def _read_meta(path: Path) -> SourceMeta:
        try:
            header, offset = read_filterbank_header(path)
        except ValueError as exc:
            raise SourceError(str(exc)) from exc
        except OSError as exc:
            raise SourceError(f"Cannot open {path.name}: {exc}") from exc

        missing = [k for k in ("nchans", "nbits", "tsamp", "fch1", "foff") if k not in header]
        if missing:
            raise SourceError("Header is missing required fields: " + ", ".join(missing))
        nbits = int(header["nbits"])
        if nbits not in _FIL_DTYPES:
            raise SourceError(f"Unsupported sample size: {nbits} bits (supported: 8, 16, 32)")
        nchans = int(header["nchans"])
        nifs = max(int(header.get("nifs", 1)), 1)
        if nchans <= 0:
            raise SourceError("Header reports zero channels")
        if float(header["tsamp"]) <= 0 or float(header["foff"]) == 0:
            raise SourceError("Header has a non-positive sample time or zero channel width")

        with open(path, "rb") as fh:
            fh.seek(max(offset - 16, 0))
            tail = fh.read(16)
        if b"HEADER_END" not in tail:
            raise SourceError("Header contains fields this reader does not recognise, so the data start cannot be located")

        size = path.stat().st_size
        row_bytes = nchans * nifs * (nbits // 8)
        n_times = (size - offset) // row_bytes
        if n_times < 1:
            raise SourceError("File contains no complete time sample after the header")

        return SourceMeta(
            path=path, fmt="fil", source_name=str(header.get("source_name") or path.stem),
            nchans=nchans, nifs=nifs, n_times=int(n_times), tsamp_s=float(header["tsamp"]),
            fch1_mhz=float(header["fch1"]), foff_mhz=float(header["foff"]), size_bytes=size,
            nbits=nbits, telescope_id=header.get("telescope_id"),
            tstart_mjd=header.get("tstart"), header_bytes=int(offset),
        )

    def _memmap(self) -> np.memmap:
        m = self.meta
        return np.memmap(
            self.path, dtype=_FIL_DTYPES[m.nbits], mode="r", offset=m.header_bytes,
            shape=(m.n_times, m.nifs, m.nchans),
        )

    def read_rows(self, t0: int, t1: int) -> np.ndarray:
        """Total intensity for time samples [t0, t1): float32 (rows, nchans)."""
        t0, t1 = int(t0), int(min(t1, self.meta.n_times))
        if t0 < 0 or t1 <= t0:
            raise ValueError(f"Invalid time range [{t0}, {t1})")
        mm = self._memmap()
        try:
            return _stokes_i(np.asarray(mm[t0:t1]))
        finally:
            del mm

    def read_window(self, t0: int, nt: int, f0: int, nf: int) -> np.ndarray:
        """Total intensity for a time/channel window: float32 (nt, nf)."""
        m = self.meta
        if t0 < 0 or f0 < 0 or t0 + nt > m.n_times or f0 + nf > m.nchans or nt <= 0 or nf <= 0:
            raise ValueError(
                f"Window t[{t0}:{t0 + nt}] f[{f0}:{f0 + nf}] is outside the data ({m.n_times} x {m.nchans})"
            )
        mm = self._memmap()
        try:
            return _stokes_i(np.asarray(mm[t0:t0 + nt, :, f0:f0 + nf]))
        finally:
            del mm


# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------
def _import_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise OptionalDependencyError(
            "Reading .h5/.hdf5 observations needs the optional package h5py (pip install h5py)."
        ) from exc
    return h5py


def _h5_value(obj, key: str):
    if key not in obj.attrs:
        return None
    value = obj.attrs[key]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, np.generic):
        return value.item()
    return value


class HDF5Source:
    """Breakthrough Listen style HDF5: a 3-D ``data`` dataset (time, IF, frequency)."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.meta = self._read_meta(self.path)

    @staticmethod
    def _read_meta(path: Path) -> SourceMeta:
        h5py = _import_h5py()
        try:
            with h5py.File(path, "r") as fh:
                if "data" not in fh:
                    raise SourceError("HDF5 file has no 'data' dataset (expected a Breakthrough Listen dynamic spectrum)")
                ds = fh["data"]
                if ds.ndim != 3:
                    raise SourceError(f"'data' must be 3-D (time, IF, frequency); found {ds.ndim}-D")
                n_times, nifs, nchans = (int(x) for x in ds.shape)

                def attr(key):
                    value = _h5_value(ds, key)
                    return value if value is not None else _h5_value(fh, key)

                missing = [k for k in ("fch1", "foff", "tsamp") if attr(k) is None]
                if missing:
                    raise SourceError("HDF5 attributes missing: " + ", ".join(missing))
                return SourceMeta(
                    path=path, fmt="h5", source_name=str(attr("source_name") or path.stem),
                    nchans=nchans, nifs=nifs, n_times=n_times, tsamp_s=float(attr("tsamp")),
                    fch1_mhz=float(attr("fch1")), foff_mhz=float(attr("foff")),
                    size_bytes=path.stat().st_size, nbits=int(attr("nbits") or 32),
                    telescope_id=(int(attr("telescope_id")) if attr("telescope_id") is not None else None),
                    tstart_mjd=(float(attr("tstart")) if attr("tstart") is not None else None),
                )
        except SourceError:
            raise
        except OSError as exc:
            raise SourceError(f"Cannot open HDF5 file: {exc}") from exc

    def _read(self, t0: int, t1: int, f0: int, f1: int) -> np.ndarray:
        h5py = _import_h5py()
        try:
            with h5py.File(self.path, "r") as fh:
                return _stokes_i(np.asarray(fh["data"][t0:t1, :, f0:f1]))
        except OSError as exc:
            raise SourceError(
                f"Could not read data from {self.path.name}: {exc}. "
                "Files with bitshuffle compression also need the hdf5plugin package."
            ) from exc

    def read_rows(self, t0: int, t1: int) -> np.ndarray:
        t1 = int(min(t1, self.meta.n_times))
        if t0 < 0 or t1 <= t0:
            raise ValueError(f"Invalid time range [{t0}, {t1})")
        return self._read(int(t0), t1, 0, self.meta.nchans)

    def read_window(self, t0: int, nt: int, f0: int, nf: int) -> np.ndarray:
        m = self.meta
        if t0 < 0 or f0 < 0 or t0 + nt > m.n_times or f0 + nf > m.nchans or nt <= 0 or nf <= 0:
            raise ValueError("Window is outside the data")
        return self._read(int(t0), int(t0 + nt), int(f0), int(f0 + nf))


ObservationSource = Union[FilterbankSource, HDF5Source]


def open_source(path: Union[str, Path]) -> ObservationSource:
    """Open an observation file by extension. Raises ``SourceError`` subclasses."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".fil":
        return FilterbankSource(path)
    if ext in (".h5", ".hdf5"):
        return HDF5Source(path)
    raise SourceError(f"Unsupported file type '{ext or path.name}'. Supported: .fil, .h5, .hdf5")


def validate_observation_file(path: Union[str, Path]) -> ValidationReport:
    """Check a file without loading its data; report errors, warnings and metadata."""
    path = Path(path)
    report = ValidationReport(path=path, ok=False)
    if not path.exists() or not path.is_file():
        report.errors.append("File not found.")
        return report
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        hint = " Frequency-domain CSV tables are not time-frequency observations and are not supported." if ext == ".csv" else ""
        report.errors.append(f"Unsupported file type '{ext or '(none)'}'. Supported: .fil, .h5, .hdf5." + hint)
        return report
    if path.stat().st_size == 0:
        report.errors.append("File is empty.")
        return report
    try:
        source = open_source(path)
    except OptionalDependencyError as exc:
        report.optional_dependency_missing = "h5py"
        report.errors.append(str(exc))
        return report
    except SourceError as exc:
        report.errors.append(str(exc))
        return report

    meta = source.meta
    report.meta = meta
    if meta.fmt == "fil":
        row_bytes = meta.nchans * meta.nifs * (meta.nbits // 8)
        if (meta.size_bytes - meta.header_bytes) % row_bytes:
            report.warnings.append("The file ends part-way through a time sample; the incomplete sample is ignored.")
    if meta.n_times < 64:
        report.warnings.append(f"Only {meta.n_times} time samples: fewer than the 64 needed for one analysis tile, so it cannot be scanned.")
    if meta.nchans < 64:
        report.warnings.append(f"Only {meta.nchans} frequency channels: fewer than the 64 needed for one analysis tile, so it cannot be scanned.")
    if meta.tstart_mjd is None:
        report.warnings.append("No observation start time in the header; times are shown as elapsed seconds only.")
    if meta.telescope is None:
        report.warnings.append("Telescope is not identified in the header.")
    report.ok = True
    return report


# ---------------------------------------------------------------------------
# Overview (decimated waterfall)
# ---------------------------------------------------------------------------
@dataclass
class Overview:
    """Block-averaged waterfall for display. ``power`` is (time bins, freq bins), frequency ascending."""

    power: np.ndarray
    time_s: np.ndarray           # bin centres, seconds from start
    freq_mhz: np.ndarray         # bin centres, ascending
    dt_s: float                  # width of one time bin
    df_mhz: float                # width of one frequency bin
    native_dt_s: float
    native_df_mhz: float


def _overview_cache_path(meta: SourceMeta, max_t: int, max_f: int) -> Path:
    return CACHE_DIR / "overview" / f"{meta.fingerprint}_{max_t}x{max_f}.npz"


def build_overview(
    source: ObservationSource,
    max_t: int = 1200,
    max_f: int = 1600,
    progress: Optional[ProgressCallback] = None,
    use_cache: bool = True,
) -> Overview:
    """Block-mean the observation to at most ``max_t`` x ``max_f`` bins.

    Reads one output time bin at a time, so peak memory is a few time samples
    of the full band regardless of file size. The result is cached on disk.
    """
    meta = source.meta
    cache = _overview_cache_path(meta, max_t, max_f)
    if use_cache and cache.exists():
        try:
            with np.load(cache) as z:
                return Overview(z["power"], z["time_s"], z["freq_mhz"], float(z["dt_s"]), float(z["df_mhz"]),
                                meta.tsamp_s, abs(meta.foff_mhz))
        except Exception:
            logger.warning("Ignoring unreadable overview cache %s", cache)

    bt = max(1, math.ceil(meta.n_times / max_t))
    bf = max(1, math.ceil(meta.nchans / max_f))
    n_tb = math.ceil(meta.n_times / bt)
    starts = np.arange(0, meta.nchans, bf)
    counts = np.diff(np.append(starts, meta.nchans)).astype(np.float64)
    power = np.empty((n_tb, len(starts)), dtype=np.float32)

    for i in range(n_tb):
        rows = source.read_rows(i * bt, min((i + 1) * bt, meta.n_times))
        mean_row = rows.mean(axis=0, dtype=np.float64)
        power[i] = np.add.reduceat(mean_row, starts) / counts
        if progress and (i % 16 == 0 or i == n_tb - 1):
            progress((i + 1) / n_tb)

    freq_native = meta.fch1_mhz + (starts + counts / 2 - 0.5) * meta.foff_mhz
    time_s = (np.arange(n_tb) * bt + np.minimum(bt, meta.n_times - np.arange(n_tb) * bt) / 2) * meta.tsamp_s
    if meta.foff_mhz < 0:  # present with frequency increasing
        power = power[:, ::-1]
        freq_native = freq_native[::-1]
    result = Overview(power, time_s, freq_native, bt * meta.tsamp_s, bf * abs(meta.foff_mhz),
                      meta.tsamp_s, abs(meta.foff_mhz))
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp.npz")
        np.savez(tmp, power=power, time_s=time_s, freq_mhz=freq_native, dt_s=result.dt_s, df_mhz=result.df_mhz)
        tmp.replace(cache)
    except OSError:
        logger.warning("Could not write overview cache %s", cache)
    return result
