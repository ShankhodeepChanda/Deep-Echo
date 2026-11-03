import json

import pytest

from src.observatory import archive
from src.observatory.observations import (
    content_key, discover_observations, read_registry, register_observation,
)
from tests.obs_helpers import noise, write_fil


def test_same_content_registers_once(tmp_path):
    up = tmp_path / "up"
    up.mkdir()
    a = write_fil(tmp_path / "a.fil", noise(128, 64, seed=1), source="S")
    copy = tmp_path / "copy.fil"
    copy.write_bytes(a.read_bytes())                       # same bytes under another name
    different = write_fil(tmp_path / "d.fil", noise(128, 64, seed=2), source="S")

    first, new1 = register_observation(a, "a.fil", external=False, uploads_dir=up)
    again, new2 = register_observation(copy, "copy.fil", external=False, uploads_dir=up)
    other, new3 = register_observation(different, "d.fil", external=False, uploads_dir=up)
    same_path, new4 = register_observation(a, "a.fil", external=False, uploads_dir=up)

    assert new1 and not new2 and again == first
    assert new3 and other != first
    assert not new4 and same_path == first
    assert [e["id"] for e in read_registry(up)] == [first, other]


def test_content_key_sees_changes_at_the_end_of_large_files(tmp_path):
    big = tmp_path / "big.bin"
    big.write_bytes(b"a" * (3 << 20))
    k1 = content_key(big)
    with open(big, "r+b") as fh:
        fh.seek((3 << 20) - 1)
        fh.write(b"b")
    assert content_key(big) != k1
    assert content_key(big) == content_key(big)


def test_registered_files_are_discovered_and_missing_ones_are_reported(tmp_path):
    raw, up, res = tmp_path / "raw", tmp_path / "up", tmp_path / "res"
    for d in (raw, up, res):
        d.mkdir()
    keep = write_fil(up / "keep.fil", noise(128, 64), source="KEEP")
    gone = write_fil(up / "gone.fil", noise(128, 64, seed=3), source="GONE")
    k_id, _ = register_observation(keep, "keep.fil", external=False, uploads_dir=up)
    g_id, _ = register_observation(gone, "gone.fil", external=True, uploads_dir=up)
    gone.unlink()
    found = {o.obs_id: o for o in discover_observations(raw, up, res)}
    assert found[k_id].meta is not None and found[k_id].origin == "uploaded"
    assert found[g_id].meta is None and found[g_id].origin == "external" and "no longer" in found[g_id].problems[0]


def test_corrupt_registry_is_ignored(tmp_path):
    (tmp_path / "registry.json").write_text("{nope", encoding="utf-8")
    assert read_registry(tmp_path) == []
    (tmp_path / "registry.json").write_text(json.dumps([{"id": "x"}, {"id": "y", "path": "p"}]), encoding="utf-8")
    assert [e["id"] for e in read_registry(tmp_path)] == ["y"]


# ---------------------------------------------------------------- archive
class FakeClient:
    def __init__(self, records=None, fail=False):
        self.records, self.fail, self.downloaded = records or [], fail, []

    def query_files(self, target="", limit=None, **kw):
        if self.fail:
            raise ConnectionError("offline")
        return self.records

    def download_file(self, info, max_size_mb, progress_callback=None):
        self.downloaded.append((info, max_size_mb))
        return "downloaded.fil"


def test_archive_search_describes_records():
    client = FakeClient([
        {"target": "HIP26137", "telescope": "GBT", "file_type": "Filterbank", "size": "1.4 GB", "url": "http://x/y/file.fil"},
        {"target": "HIP26137", "size": 0.1, "url": None},
        "not a record",
    ])
    items = archive.search(client, " HIP26137 ")
    assert len(items) == 2
    assert items[0].label.startswith("HIP26137 · GBT · Filterbank") and "file.fil" in items[0].label
    assert items[0].size_mb == pytest.approx(1.4 * 1024)
    assert items[1].size_mb == pytest.approx(0.1 * 1024) and items[1].url is None


def test_archive_requires_a_target_and_surfaces_network_errors():
    with pytest.raises(ValueError):
        archive.search(FakeClient(), "   ")
    with pytest.raises(ConnectionError):
        archive.search(FakeClient(fail=True), "x")


def test_archive_download_is_capped_and_explicit():
    client = FakeClient()
    big = archive.describe({"url": "u", "size": 5.0, "target": "T"})            # 5 GB
    with pytest.raises(ValueError, match="above the 200 MB limit"):
        archive.download(client, big, 200.0)
    assert client.downloaded == []                                              # nothing requested when refused
    small = archive.describe({"url": "u", "size": "50 MB"})
    assert archive.download(client, small, 200.0) == "downloaded.fil"
    assert client.downloaded[0][1] == 200.0
    with pytest.raises(ValueError, match="no download link"):
        archive.download(client, archive.describe({"size": 1}), 200.0)


@pytest.mark.parametrize("raw,expected", [("2 GB", 2048.0), ("300 MB", 300.0), (1, 1024.0), (None, None), ("big", None), ("5 XB", None)])
def test_archive_size_parsing(raw, expected):
    assert archive._size_mb(raw) == expected
