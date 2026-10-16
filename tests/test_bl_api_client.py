"""
Unit and integration tests for Breakthrough Listen API client and filterbank reader.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import struct

from src.bl_api_client import BreakthroughListenClient, BreakthroughListenAPIError
from src.filterbank_reader import (
    read_filterbank_header,
    load_filterbank_waterfall,
    extract_spectrogram_tiles,
)


class TestBreakthroughListenClient:
    def test_client_init_and_headers(self):
        client = BreakthroughListenClient(
            base_url="http://mock-seti.org/api",
            api_key="secret-token-123",
        )
        assert client.base_url == "http://mock-seti.org/api"
        assert client.session.headers["X-API-Key"] == "secret-token-123"
        assert client.session.headers["Authorization"] == "Bearer secret-token-123"

    @patch("requests.Session.get")
    def test_list_targets(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"result": "success", "data": ["HIP11048", "VOYAGER1"]}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        client = BreakthroughListenClient(base_url="http://mock.seti.org")
        targets = client.list_targets()

        assert targets == ["HIP11048", "VOYAGER1"]
        mock_get.assert_called_once()
        assert "api/list-targets" in mock_get.call_args[0][0]

    @patch("requests.Session.get")
    def test_query_files_params(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "result": "success",
            "data": [
                {
                    "target": "HIP11048",
                    "telescope": "GBT",
                    "url": "http://data.seti.org/sample.fil",
                    "size": "0.1",
                }
            ],
        }
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        client = BreakthroughListenClient()
        results = client.query_files(
            target="HIP11048",
            telescopes=["GBT", "Parkes"],
            cadence=True,
            limit=10,
        )

        assert len(results) == 1
        assert results[0]["target"] == "HIP11048"

        call_params = mock_get.call_args[1]["params"]
        assert call_params["target"] == "HIP11048"
        assert call_params["telescopes"] == "GBT,Parkes"
        assert call_params["cadence"] == "true"
        assert call_params["limit"] == 10

    @patch("requests.Session.get")
    def test_api_error_handling(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"result": "error", "message": "Invalid query parameters"}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        client = BreakthroughListenClient()
        with pytest.raises(BreakthroughListenAPIError) as exc_info:
            client.query_files(target="Nonexistent")
        assert "Invalid query parameters" in str(exc_info.value)


class TestFilterbankReader:
    def _create_mock_fil_file(self, tmp_path: Path, nchans=32, n_times=10) -> Path:
        """Create a synthetic binary SIGPROC filterbank file."""
        fil_path = tmp_path / "mock_obs.fil"
        with open(fil_path, "wb") as f:
            def write_str(s: str):
                f.write(struct.pack("<I", len(s)))
                f.write(s.encode("utf-8"))

            write_str("HEADER_START")
            write_str("source_name")
            write_str("HIP11048")
            write_str("telescope_id")
            f.write(struct.pack("<I", 6))  # GBT
            write_str("nchans")
            f.write(struct.pack("<I", nchans))
            write_str("nbits")
            f.write(struct.pack("<I", 32))  # float32
            write_str("tsamp")
            f.write(struct.pack("<d", 0.1))  # 100ms
            write_str("fch1")
            f.write(struct.pack("<d", 1420.0))  # 1420 MHz
            write_str("foff")
            f.write(struct.pack("<d", -0.05))  # -50 kHz
            write_str("HEADER_END")

            # Write spectral data: n_times x nchans float32
            data = np.random.randn(n_times, nchans).astype(np.float32)
            data.tofile(f)

        return fil_path

    def test_read_filterbank_header(self, tmp_path):
        fil_path = self._create_mock_fil_file(tmp_path, nchans=64, n_times=16)
        header, offset = read_filterbank_header(fil_path)

        assert header["source_name"] == "HIP11048"
        assert header["nchans"] == 64
        assert header["nbits"] == 32
        assert offset > 0

    def test_load_filterbank_waterfall(self, tmp_path):
        fil_path = self._create_mock_fil_file(tmp_path, nchans=64, n_times=20)
        wf_data = load_filterbank_waterfall(fil_path, max_time_steps=15)

        assert wf_data["waterfall"].shape == (15, 64)
        assert len(wf_data["time_axis"]) == 15
        assert len(wf_data["freq_axis"]) == 64

    def test_extract_spectrogram_tiles(self):
        waterfall = np.random.rand(128, 128).astype(np.float32)
        tiles = extract_spectrogram_tiles(waterfall, tile_h=64, tile_w=64)

        # 128x128 with non-overlapping 64x64 tiles produces 4 tiles
        assert tiles.shape == (4, 1, 64, 64)
        assert tiles.min() >= 0.0
        assert tiles.max() <= 1.0
