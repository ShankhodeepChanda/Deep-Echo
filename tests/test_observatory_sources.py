import sys
import numpy as np
import pytest

from src.observatory import sources
from src.observatory.sources import (
    FilterbankSource, HDF5Source, OptionalDependencyError, SourceError,
    build_overview, open_source, validate_observation_file,
)
from tests.obs_helpers import noise, write_fil


def test_window_and_rows_match_array(tmp_path):
    data = np.arange(200 * 128, dtype=np.float32).reshape(200, 128)
    path = write_fil(tmp_path / "a.fil", data)
    src = FilterbankSource(path)
    assert (src.meta.n_times, src.meta.nchans) == (200, 128)
    assert np.array_equal(src.read_window(10, 64, 32, 64), data[10:74, 32:96])
    assert np.array_equal(src.read_rows(5, 9), data[5:9])


def test_two_ifs_are_averaged_like_the_existing_loader(tmp_path):
    rng = np.random.RandomState(1)
    data = rng.rand(70, 2, 64).astype(np.float32)
    src = FilterbankSource(write_fil(tmp_path / "b.fil", data))
    assert np.allclose(src.read_window(0, 64, 0, 64), (data[:64, 0] + data[:64, 1]) / 2)


def test_metadata_frequency_helpers(tmp_path):
    src = FilterbankSource(write_fil(tmp_path / "c.fil", noise(70, 100), fch1=1400.0, foff=-0.5, tsamp=2.0))
    m = src.meta
    assert m.freq_high_mhz == pytest.approx(1400.25)
    assert m.freq_low_mhz == pytest.approx(1400.0 - 99 * 0.5 - 0.25)
    assert m.duration_s == pytest.approx(140.0)
    assert m.band == "L-band" and m.telescope == "Green Bank Telescope (GBT)"
    assert m.freq_to_channel(1390.0) == 20
    with pytest.raises(ValueError):
        m.freq_to_channel(1500.0)
    assert m.start_utc is not None


def test_out_of_range_window_is_rejected(tmp_path):
    src = FilterbankSource(write_fil(tmp_path / "d.fil", noise(70, 64)))
    with pytest.raises(ValueError):
        src.read_window(10, 64, 0, 64)


@pytest.mark.parametrize("name,payload,fragment", [
    ("empty.fil", b"", "empty"),
    ("junk.fil", b"not a filterbank file at all", "HEADER_START"),
    ("table.csv", b"a,b\n1,2\n", "CSV"),
    ("x.txt", b"hello", "Unsupported"),
])
def test_validation_rejects_bad_files(tmp_path, name, payload, fragment):
    path = tmp_path / name
    path.write_bytes(payload)
    report = validate_observation_file(path)
    assert not report.ok
    assert any(fragment in e for e in report.errors), report.errors


def test_validation_reports_missing_file(tmp_path):
    assert not validate_observation_file(tmp_path / "nope.fil").ok


def test_validation_warns_on_truncated_and_small_files(tmp_path):
    path = write_fil(tmp_path / "t.fil", noise(30, 64))
    with open(path, "ab") as fh:
        fh.write(b"\x00\x00")  # partial sample
    report = validate_observation_file(path)
    assert report.ok
    assert any("part-way" in w for w in report.warnings)
    assert any("fewer than the 64" in w for w in report.warnings)


def test_validation_ok_has_metadata(tmp_path):
    report = validate_observation_file(write_fil(tmp_path / "ok.fil", noise(128, 128)))
    assert report.ok and report.meta.nchans == 128 and not report.errors


def test_overview_block_means_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "CACHE_DIR", tmp_path / "cache")
    data = noise(400, 256)
    src = FilterbankSource(write_fil(tmp_path / "o.fil", data, foff=-1.0))
    ov = build_overview(src, max_t=100, max_f=64)
    assert ov.power.shape == (100, 64)
    # frequency axis ascends even though foff is negative; time-bin of 4 samples, freq-bin of 4 channels
    assert np.all(np.diff(ov.freq_mhz) > 0)
    assert ov.power[0, -1] == pytest.approx(data[:4, :4].mean(), rel=1e-5)
    again = build_overview(src, max_t=100, max_f=64)
    assert np.array_equal(again.power, ov.power)


def test_overview_handles_partial_last_bins(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "CACHE_DIR", tmp_path / "cache")
    data = noise(103, 70)
    ov = build_overview(FilterbankSource(write_fil(tmp_path / "p.fil", data)), max_t=10, max_f=7, use_cache=False)
    assert ov.power.shape == (10, 7)          # bins of 11 samples; the last holds only the remaining 4
    assert np.isfinite(ov.power).all()
    assert ov.power[-1, :].mean() == pytest.approx(data[99:, :].mean(), rel=1e-4)


def test_open_source_dispatch_and_errors(tmp_path):
    with pytest.raises(SourceError):
        open_source(tmp_path / "a.xyz")


def test_hdf5_roundtrip(tmp_path):
    h5py = pytest.importorskip("h5py")
    data = np.random.RandomState(2).rand(80, 1, 128).astype(np.float32)
    path = tmp_path / "obs.h5"
    with h5py.File(path, "w") as fh:
        ds = fh.create_dataset("data", data=data)
        ds.attrs.update({"fch1": 8000.0, "foff": -0.001, "tsamp": 0.5, "source_name": "H5SRC", "tstart": 58100.0})
    src = open_source(path)
    assert isinstance(src, HDF5Source)
    assert src.meta.source_name == "H5SRC" and src.meta.nchans == 128 and src.meta.n_times == 80
    assert np.allclose(src.read_window(4, 64, 8, 64), data[4:68, 0, 8:72])
    assert validate_observation_file(path).ok


def test_hdf5_schema_errors(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("other", data=np.zeros((4, 4)))
    report = validate_observation_file(path)
    assert not report.ok and "data" in report.errors[0]
    path2 = tmp_path / "noattrs.h5"
    with h5py.File(path2, "w") as fh:
        fh.create_dataset("data", data=np.zeros((10, 1, 8), dtype=np.float32))
    assert "fch1" in validate_observation_file(path2).errors[0]


def test_hdf5_reports_missing_dependency(tmp_path, monkeypatch):
    path = tmp_path / "x.h5"
    path.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)
    monkeypatch.setitem(sys.modules, "h5py", None)  # makes `import h5py` raise ImportError
    report = validate_observation_file(path)
    assert not report.ok and report.optional_dependency_missing == "h5py"
    with pytest.raises(OptionalDependencyError):
        open_source(path)
