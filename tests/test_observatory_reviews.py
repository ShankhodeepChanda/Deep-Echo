import pytest

from src.observatory.reviews import (
    DISMISSED, FLAGGED, PENDING, RETAINED, Preferences, ReviewStore, ReviewStoreError,
)


def test_reviews_persist_across_store_instances(tmp_path):
    path = tmp_path / "r.sqlite"
    ReviewStore(path).set("OBS:t1:f2", "OBS", RETAINED, "  looks drifting  ")
    got = ReviewStore(path).for_observation("OBS")
    assert got["OBS:t1:f2"].status == RETAINED and got["OBS:t1:f2"].note == "looks drifting"


def test_update_replaces_and_pending_without_note_clears(tmp_path):
    store = ReviewStore(tmp_path / "r.sqlite")
    store.set("c1", "O", FLAGGED, "x")
    store.set("c1", "O", DISMISSED, "y")
    assert store.get("c1").status == DISMISSED and store.get("c1").note == "y"
    store.set("c1", "O", PENDING, "")
    assert store.get("c1") is None
    store.set("c2", "O", PENDING, "note only")      # a note alone is kept
    assert store.get("c2").status == PENDING and store.get("c2").note == "note only"


def test_reviews_are_scoped_by_observation(tmp_path):
    store = ReviewStore(tmp_path / "r.sqlite")
    store.set("a", "O1", RETAINED)
    store.set("b", "O2", FLAGGED)
    assert set(store.for_observation("O1")) == {"a"}
    assert {r.candidate_id for r in store.all()} == {"a", "b"}


def test_invalid_status_is_an_error(tmp_path):
    with pytest.raises(ReviewStoreError):
        ReviewStore(tmp_path / "r.sqlite").set("c", "O", "maybe")


def test_write_failure_raises_instead_of_dropping(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    store = ReviewStore(blocker / "r.sqlite")        # parent is a file: cannot create
    with pytest.raises(ReviewStoreError):
        store.set("c", "O", RETAINED)
    with pytest.raises(ReviewStoreError):
        store.for_observation("O")


def test_corrupt_database_raises(tmp_path):
    path = tmp_path / "r.sqlite"
    path.write_bytes(b"this is not sqlite" * 100)
    with pytest.raises(ReviewStoreError):
        ReviewStore(path).set("c", "O", RETAINED)


def test_preferences_roundtrip_and_recovery(tmp_path):
    prefs = Preferences(tmp_path / "p.json")
    assert prefs.read() == {}
    prefs.set("guide_dismissed", True)
    assert Preferences(tmp_path / "p.json").read() == {"guide_dismissed": True}
    (tmp_path / "p.json").write_text("{broken", encoding="utf-8")
    assert prefs.read() == {}
