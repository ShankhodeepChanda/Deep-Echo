"""
Breakthrough Listen Open Data API Client.

Provides a clean Python interface to the Breakthrough Listen Open Data API
(http://seti.berkeley.edu/opendata and dev server http://35.236.84.6:5001).

Implements:
- list_targets
- list_telescopes
- list_file_types
- query_files
- get_cadence_url
- get_cadence
- get_diagnostic_sources
- download_file (with caching, md5 validation, and size limits)
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable
from urllib.parse import urljoin
import requests
from dotenv import load_dotenv

from src.config import (
    BL_PROD_URL,
    BL_DEV_URL,
    BL_CACHE_DIR,
    DEFAULT_BL_MAX_DOWNLOAD_MB,
)

# Load environment variables from .env if present
load_dotenv()

logger = logging.getLogger(__name__)


class BreakthroughListenAPIError(Exception):
    """Exception raised when Breakthrough Listen API returns an error."""
    pass


class BreakthroughListenClient:
    """
    Client for interacting with the Breakthrough Listen Open Data API.
    
    Parameters
    ----------
    base_url : str, optional
        API root URL. Defaults to production (http://seti.berkeley.edu/opendata)
        or can be set to dev (http://35.236.84.6:5001).
    api_key : str, optional
        API Key or Token if required by host/proxy.
        Falls back to environment variable SETI_API_KEY or BL_API_KEY.
    cache_dir : Path, optional
        Local directory for caching downloaded observation files.
    timeout : int, default 30
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        timeout: int = 30,
    ):
        self.base_url = (
            base_url
            or os.getenv("SETI_BASE_URL")
            or os.getenv("BL_BASE_URL")
            or BL_PROD_URL
        ).rstrip("/")

        self.api_key = (
            api_key
            or os.getenv("SETI_API_KEY")
            or os.getenv("BL_API_KEY")
        )

        self.cache_dir = Path(cache_dir or BL_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "SETI-AnomalyDetection/1.0 (BreakthroughListenClient)",
            "Accept": "application/json",
        })

        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "X-API-Key": self.api_key,
            })

    def _build_url(self, endpoint: str) -> str:
        """Construct full URL from base_url and endpoint."""
        clean_endpoint = endpoint.lstrip("/")
        return f"{self.base_url}/{clean_endpoint}"

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute GET request with standardized error handling."""
        url = self._build_url(endpoint)
        try:
            logger.debug(f"GET {url} with params {params}")
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            # Handle API responses with explicit 'result' status
            if isinstance(data, dict):
                if data.get("result") == "error":
                    msg = data.get("message", "Unknown API error")
                    raise BreakthroughListenAPIError(f"API returned error: {msg}")

            return data
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e}")
            raise BreakthroughListenAPIError(f"HTTP error connecting to {url}: {e}") from e
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection error for {url}: {e}")
            raise BreakthroughListenAPIError(f"Failed to connect to {url}: {e}") from e

    # ──────────────────────────────────────────────────────────────────
    # Metadata Listing Endpoints
    # ──────────────────────────────────────────────────────────────────

    def list_targets(self) -> List[str]:
        """
        List all targets currently available in the database.
        
        Returns
        -------
        list of str
            Target names.
        """
        data = self._get("api/list-targets")
        if isinstance(data, dict):
            return data.get("targets", data.get("data", []))
        return data

    def list_telescopes(self) -> List[str]:
        """
        List all telescopes in the database (e.g. GBT, Parkes, APF).
        
        Returns
        -------
        list of str
            Telescope identifiers.
        """
        data = self._get("api/list-telescopes")
        if isinstance(data, dict):
            return data.get("telescopes", data.get("data", []))
        return data

    def list_file_types(self) -> List[str]:
        """
        List all file type names in the database (e.g. HDF5, Filterbank).
        
        Returns
        -------
        list of str
            Supported file type names.
        """
        data = self._get("api/list-file-types")
        if isinstance(data, dict):
            return data.get("file_types", data.get("data", []))
        return data

    # ──────────────────────────────────────────────────────────────────
    # File & Cadence Query Endpoints
    # ──────────────────────────────────────────────────────────────────

    def query_files(
        self,
        target: str = "",
        telescopes: Optional[Union[str, List[str]]] = None,
        file_types: Optional[Union[str, List[str]]] = None,
        pos_ra: Optional[float] = None,
        pos_dec: Optional[float] = None,
        pos_rad: Optional[float] = None,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
        freq_start: Optional[float] = None,
        freq_end: Optional[float] = None,
        limit: Optional[int] = 50,
        cadence: bool = False,
        primary_target: Optional[bool] = None,
        grades: Optional[Union[str, List[str]]] = None,
        quality: Optional[Union[str, List[str]]] = None,
        min_size_gb: Optional[float] = None,
        max_size_gb: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query files in the Breakthrough Listen database with filters.
        
        Parameters
        ----------
        target : str
            Target name filter (matches substring; "" returns all targets).
        telescopes : str or list of str, optional
            Telescope names (comma-separated or list).
        file_types : str or list of str, optional
            File types (e.g. 'HDF5', 'Filterbank').
        pos_ra : float, optional
            Right ascension in degrees.
        pos_dec : float, optional
            Declination in degrees.
        pos_rad : float, optional
            Search radius in degrees around (pos_ra, pos_dec).
        time_start : float, optional
            MJD start time.
        time_end : float, optional
            MJD end time.
        freq_start : float, optional
            Start frequency in MHz.
        freq_end : float, optional
            End frequency in MHz.
        limit : int, optional
            Maximum number of entries to return.
        cadence : bool, default False
            If True, groups observations into cadences (ABACAD sequence).
        primary_target : bool, optional
            If True and cadence=True, returns only cadences where target is 'A'.
        grades : str or list of str, optional
            Data quality grades (A, B, C, F).
        quality : str or list of str, optional
            Quality filters.
        min_size_gb : float, optional
            Minimum file size in GB.
        max_size_gb : float, optional
            Maximum file size in GB.

        Returns
        -------
        list of dict
            List of observation records matching criteria.
        """
        params: Dict[str, Any] = {"target": target}

        if telescopes:
            params["telescopes"] = ",".join(telescopes) if isinstance(telescopes, list) else telescopes
        if file_types:
            params["file-types"] = ",".join(file_types) if isinstance(file_types, list) else file_types
        if pos_ra is not None:
            params["pos-ra"] = pos_ra
        if pos_dec is not None:
            params["pos-dec"] = pos_dec
        if pos_rad is not None:
            params["pos-rad"] = pos_rad
        if time_start is not None:
            params["time-start"] = time_start
        if time_end is not None:
            params["time-end"] = time_end
        if freq_start is not None:
            params["freq-start"] = freq_start
        if freq_end is not None:
            params["freq-end"] = freq_end
        if limit is not None:
            params["limit"] = limit
        if cadence:
            params["cadence"] = "true"
        if primary_target is not None:
            params["primary-target"] = "true" if primary_target else "false"
        if grades:
            params["grades"] = ",".join(grades) if isinstance(grades, list) else grades
        if quality:
            params["quality"] = ",".join(quality) if isinstance(quality, list) else quality
        if min_size_gb is not None:
            params["minSize"] = min_size_gb
        if max_size_gb is not None:
            params["maxSize"] = max_size_gb

        res = self._get("api/query-files", params=params)
        if isinstance(res, dict):
            return res.get("data", [])
        return res

    def get_cadence_url(self, file_url: str) -> Optional[str]:
        """
        Retrieve cadence identifier URL for a given observation file URL.
        """
        data = self._get("api/get-cadence-url", params={"url": file_url})
        if isinstance(data, dict):
            return data.get("url")
        return str(data)

    def get_cadence(self, cadence_url_or_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all observation entries belonging to a specific cadence.
        
        Parameters
        ----------
        cadence_url_or_id : str
            The cadence identifier or relative path returned by get_cadence_url
            or in query_files(..., cadence=True).
        """
        endpoint = f"api/get-cadence/{cadence_url_or_id.strip('/')}"
        data = self._get(endpoint)
        if isinstance(data, dict):
            return data.get("data", [])
        return data

    def get_diagnostic_sources(self, diagnostic_type: str, observation_id: Union[int, str]) -> Dict[str, Any]:
        """
        Retrieve calibrators or pulsars targeted within 12 hours of an observation.
        
        Parameters
        ----------
        diagnostic_type : str
            Either 'Pulsar' or 'Calibrator'.
        observation_id : int or str
            Observation ID number.
        """
        if diagnostic_type not in {"Pulsar", "Calibrator"}:
            raise ValueError(f"diagnostic_type must be 'Pulsar' or 'Calibrator', got {diagnostic_type}")
        endpoint = f"api/get-diagnostic-sources/{diagnostic_type}/{observation_id}"
        return self._get(endpoint)

    # ──────────────────────────────────────────────────────────────────
    # Download & Cache Management
    # ──────────────────────────────────────────────────────────────────

    def download_file(
        self,
        file_info_or_url: Union[str, Dict[str, Any]],
        output_dir: Optional[Union[str, Path]] = None,
        max_size_mb: float = DEFAULT_BL_MAX_DOWNLOAD_MB,
        force_download: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """
        Download an observation data file to local storage with caching and safety checks.
        
        Parameters
        ----------
        file_info_or_url : str or dict
            Either direct URL string or observation dictionary returned by query_files.
        output_dir : Path, optional
            Target directory. Defaults to self.cache_dir.
        max_size_mb : float, default 200.0
            Maximum allowable download size in MB. Protects disk space.
        force_download : bool, default False
            If True, re-download even if cached file exists.
        progress_callback : callable, optional
            Function called with (bytes_downloaded, total_bytes).

        Returns
        -------
        Path
            Path to downloaded file on disk.
        """
        if isinstance(file_info_or_url, dict):
            file_url = file_info_or_url.get("url")
            expected_md5 = file_info_or_url.get("md5sum")
            reported_size = file_info_or_url.get("size")  # size string or float in GB
        else:
            file_url = file_info_or_url
            expected_md5 = None
            reported_size = None

        if not file_url:
            raise ValueError("No download URL provided in file_info")

        # Resolve relative URLs
        if not file_url.startswith("http://") and not file_url.startswith("https://"):
            full_url = self._build_url(file_url)
        else:
            full_url = file_url

        target_dir = Path(output_dir or self.cache_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        filename = Path(file_url).name
        if not filename:
            # Fallback hash filename if URL doesn't have filename
            filename = f"bl_obs_{hashlib.md5(file_url.encode()).hexdigest()[:12]}.bin"

        local_file_path = target_dir / filename

        # Check existing file in cache
        if local_file_path.exists() and not force_download:
            if expected_md5:
                if self._verify_md5(local_file_path, expected_md5):
                    logger.info(f"Using cached file with verified MD5: {local_file_path}")
                    return local_file_path
                logger.warning(f"Cached file MD5 mismatch. Re-downloading {filename}...")
            else:
                logger.info(f"Using cached file: {local_file_path}")
                return local_file_path

        # Check header size before streaming
        head_resp = self.session.head(full_url, timeout=self.timeout, allow_redirects=True)
        content_length = head_resp.headers.get("Content-Length")
        if content_length:
            total_bytes = int(content_length)
            total_mb = total_bytes / (1024 * 1024)
            if total_mb > max_size_mb:
                raise ValueError(
                    f"File size ({total_mb:.1f} MB) exceeds maximum allowed download "
                    f"limit ({max_size_mb:.1f} MB). Adjust max_size_mb if you wish to proceed."
                )
        else:
            total_bytes = 0

        # Stream download
        logger.info(f"Downloading {filename} from {full_url}...")
        temp_file = local_file_path.with_suffix(local_file_path.suffix + ".part")
        downloaded = 0

        with self.session.get(full_url, stream=True, timeout=self.timeout) as stream_resp:
            stream_resp.raise_for_status()
            with open(temp_file, "wb") as f:
                for chunk in stream_resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, total_bytes)

        # Rename temp to final
        if temp_file.exists():
            if local_file_path.exists():
                local_file_path.unlink()
            temp_file.rename(local_file_path)

        # Verify MD5 if available
        if expected_md5 and not self._verify_md5(local_file_path, expected_md5):
            local_file_path.unlink()
            raise ValueError(f"Downloaded file {filename} failed MD5 verification!")

        logger.info(f"Download complete: {local_file_path} ({downloaded / (1024*1024):.2f} MB)")
        return local_file_path

    @staticmethod
    def _verify_md5(file_path: Path, expected_md5: str) -> bool:
        """Verify md5 checksum of a local file."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                hasher.update(block)
        return hasher.hexdigest().lower() == expected_md5.lower()
