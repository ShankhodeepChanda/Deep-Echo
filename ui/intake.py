"""Adding observations: upload, open in place from disk, or (optionally) fetch from the Breakthrough Listen archive."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import streamlit as st

from src.config import BL_CACHE_DIR, DEFAULT_BL_MAX_DOWNLOAD_MB, UPLOADS_DIR
from src.observatory import archive
from src.observatory.observations import register_observation
from src.observatory.sources import SUPPORTED_EXTENSIONS, ValidationReport, validate_observation_file
from ui.components import fmt_bytes, fmt_khz, fmt_range, fmt_seconds, fmt_utc, kv_block

_CHUNK = 8 * 1024 * 1024
_INCOMING = UPLOADS_DIR / ".incoming"


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", name).strip("._") or "observation"


def _preview(report: ValidationReport) -> None:
    """Metadata preview, warnings and errors for a validated file."""
    if report.meta:
        m = report.meta
        kv_block([
            ("Source", m.source_name, False),
            ("Telescope", m.telescope or "not identified in file", False),
            ("Frequency", f"{fmt_range(m.freq_low_mhz, m.freq_high_mhz, 2)}  ({m.nchans:,} channels, {fmt_khz(abs(m.foff_mhz))} each)", False),
            ("Duration", f"{fmt_seconds(m.duration_s)}  ({m.n_times:,} samples of {fmt_seconds(m.tsamp_s)})", False),
            ("Started", fmt_utc(m.start_utc), False),
            ("Size", f"{fmt_bytes(m.size_bytes)} \u00b7 {m.fmt.upper()}", False),
        ])
    for w in report.warnings:
        st.caption(f"Note: {w}")
    for e in report.errors:
        st.error(e)
    if report.optional_dependency_missing:
        st.info(f"Install the optional package with: pip install {report.optional_dependency_missing}")


def _use(path: Path, original_name: str, external: bool) -> None:
    obs_id, is_new = register_observation(path, original_name, external)
    if not is_new and not external:
        path.unlink(missing_ok=True)          # an identical file is already registered; keep a single copy
    st.session_state["_goto_obs"] = obs_id
    st.session_state["_goto"] = "Overview"
    st.session_state.pop("selected_candidate", None)
    st.toast("Observation added." if is_new else "That file is already added. Opened the existing observation.")
    st.rerun()


# ---------------------------------------------------------------------------
def _upload_tab() -> None:
    n = st.session_state.setdefault("uploader_n", 0)
    up = st.file_uploader(
        "Observation file", type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS], key=f"uploader_{n}",
        help="SIGPROC filterbank (.fil) or Breakthrough Listen HDF5 (.h5, .hdf5). Your original file is not modified.",
    )
    st.caption("Accepts .fil, .h5 and .hdf5, up to 500 MB. For larger files use **Local file**, which reads in place.")
    staged = st.session_state.get("staged")
    if up is None:
        if staged:
            Path(staged["path"]).unlink(missing_ok=True)
            st.session_state.pop("staged", None)
        return

    if not staged or staged["id"] != up.file_id:
        if staged:
            Path(staged["path"]).unlink(missing_ok=True)
        _INCOMING.mkdir(parents=True, exist_ok=True)
        dest = _INCOMING / f"{uuid.uuid4().hex[:8]}_{_safe_name(up.name)}"
        bar = st.progress(0.0, text="Saving file for checking\u2026")
        total = max(up.size, 1)
        written = 0
        try:
            up.seek(0)
            with open(dest, "wb") as fh:
                while True:
                    chunk = up.read(_CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    written += len(chunk)
                    bar.progress(min(written / total, 1.0), text="Saving file for checking\u2026")
        except OSError as exc:
            bar.empty()
            dest.unlink(missing_ok=True)
            st.error(f"Could not save the upload: {exc}")
            return
        bar.empty()
        staged = {"id": up.file_id, "path": str(dest), "name": up.name}
        st.session_state["staged"] = staged

    with st.spinner("Checking the file\u2026"):
        report = validate_observation_file(staged["path"])
    _preview(report)
    if report.ok:
        if st.button("Use this observation", type="primary", key="use_upload", width="stretch"):
            final = UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}_{_safe_name(staged['name'])}"
            Path(staged["path"]).replace(final)
            st.session_state.pop("staged", None)
            st.session_state["uploader_n"] = n + 1
            _use(final, staged["name"], external=False)
    elif st.button("Discard this file", key="discard_upload", width="stretch"):
        Path(staged["path"]).unlink(missing_ok=True)
        st.session_state.pop("staged", None)
        st.session_state["uploader_n"] = n + 1
        st.rerun()


def _disk_tab() -> None:
    raw = st.text_input(
        "Path to a file", key="disk_path", placeholder=r"C:\data\observation.fil",
        help="The file is read where it is: nothing is copied and it is never modified. Best for multi-gigabyte observations.",
    )
    path_text = raw.strip().strip('"').strip("'")
    if not path_text:
        st.caption("Paste the full path to a .fil, .h5 or .hdf5 file.")
        return
    path = Path(path_text)
    if path.is_dir():
        st.error("That is a folder. Enter the path of a file.")
        return
    report = validate_observation_file(path)
    _preview(report)
    if report.ok and st.button("Use this observation", type="primary", key="use_disk", width="stretch"):
        _use(path, path.name, external=True)


def _archive_tab() -> None:
    st.caption("Optional. Nothing is requested until you search, and local files never need this.")
    target = st.text_input("Target name", key="bl_target", placeholder="e.g. HIP26137")
    if st.button("Search archive", key="bl_search", width="stretch"):
        try:
            from src.bl_api_client import BreakthroughListenClient
            with st.spinner("Searching the archive\u2026"):
                client = BreakthroughListenClient()
                st.session_state["bl_results"] = archive.search(client, target)
            st.session_state["bl_error"] = None
        except ValueError as exc:
            st.session_state["bl_error"] = str(exc)
        except Exception as exc:  # network, HTTP, malformed response
            st.session_state["bl_error"] = f"The archive could not be reached: {exc}"
    if st.session_state.get("bl_error"):
        st.error(st.session_state["bl_error"])
    results = st.session_state.get("bl_results")
    if results is None:
        return
    if not results:
        st.info("No files matched that target.")
        return
    labels = [f"{r.label}" + (f"  \u2014  {r.size_mb:,.0f} MB" if r.size_mb else "") for r in results]
    pick = st.selectbox("Files found", range(len(results)), format_func=lambda i: labels[i], key="bl_pick")
    limit = st.number_input("Largest download allowed (MB)", 10.0, 20000.0, float(DEFAULT_BL_MAX_DOWNLOAD_MB), step=50.0, key="bl_limit")
    item = results[pick]
    if item.size_mb and item.size_mb > limit:
        st.warning(f"This file is {item.size_mb:,.0f} MB, above your limit of {limit:,.0f} MB.")
    if st.button("Download and add", key="bl_download", disabled=bool(item.size_mb and item.size_mb > limit), width="stretch"):
        bar = st.progress(0.0, text="Downloading\u2026")
        try:
            from src.bl_api_client import BreakthroughListenClient
            client = BreakthroughListenClient(cache_dir=BL_CACHE_DIR)
            path = archive.download(client, item, limit, lambda done, total: bar.progress(min(done / total, 1.0) if total else 0.0, text="Downloading\u2026"))
        except Exception as exc:
            bar.empty()
            st.error(f"Download failed: {exc}")
            return
        bar.empty()
        report = validate_observation_file(path)
        _preview(report)
        if report.ok:
            _use(path, path.name, external=True)


def render_add_observation() -> None:
    mode = st.radio(
        "How to add", ["Upload", "Local file", "Archive"],
        label_visibility="collapsed", key="intake_mode",
    )
    if mode == "Upload":
        _upload_tab()
    elif mode == "Local file":
        _disk_tab()
    else:
        _archive_tab()
