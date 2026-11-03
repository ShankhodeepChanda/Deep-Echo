"""Helpers that write small, valid observation files for tests."""
import struct
from pathlib import Path

import numpy as np


def _s(text: str) -> bytes:
    raw = text.encode("ascii")
    return struct.pack("<I", len(raw)) + raw


def write_fil(path, data, fch1=1400.0, foff=-0.01, tsamp=1.0, source="TESTSRC", telescope_id=6, tstart=58000.5, nifs=1):
    """Write float32 data of shape (time, chans) or (time, nifs, chans) as SIGPROC filterbank."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data[:, None, :]
    n_t, n_if, n_ch = data.shape
    head = _s("HEADER_START")
    head += _s("source_name") + _s(source)
    head += _s("telescope_id") + struct.pack("<I", telescope_id)
    head += _s("nchans") + struct.pack("<I", n_ch)
    head += _s("nbits") + struct.pack("<I", 32)
    head += _s("nifs") + struct.pack("<I", n_if)
    head += _s("tstart") + struct.pack("<d", tstart)
    head += _s("tsamp") + struct.pack("<d", tsamp)
    head += _s("fch1") + struct.pack("<d", fch1)
    head += _s("foff") + struct.pack("<d", foff)
    head += _s("HEADER_END")
    with open(path, "wb") as fh:
        fh.write(head)
        fh.write(data.tobytes())
    return Path(path)


def noise(n_t, n_ch, seed=0, level=1.0):
    return (np.random.RandomState(seed).rand(n_t, n_ch).astype(np.float32) * 0.2 + 1.0) * level
