"""
Unit tests for Cadence Verification and Reference Comparison (Phase 11).
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch

from src.models.cnn_autoencoder import CNNAutoencoder
from src.cadence_filter import analyze_cadence_sequence


class TestCadenceFilter:
    @pytest.fixture
    def mock_model(self):
        torch.manual_seed(42)
        model = CNNAutoencoder(input_size=(1, 64, 64), latent_dim=16)
        model.eval()
        return model

    def test_candidate_technosignature_detection(self, mock_model):
        """Test that anomaly present ONLY in ON-target pointings flags as CANDIDATE."""
        # 3 ON-target (high error 0.08), 3 OFF-target (low error 0.01)
        cadence = [
            {"target": "HIP11048", "pointing_type": "ON", "spectrogram": np.zeros((1, 64, 64))},
            {"target": "OFF_1", "pointing_type": "OFF", "spectrogram": np.zeros((1, 64, 64))},
            {"target": "HIP11048", "pointing_type": "ON", "spectrogram": np.zeros((1, 64, 64))},
            {"target": "OFF_2", "pointing_type": "OFF", "spectrogram": np.zeros((1, 64, 64))},
            {"target": "HIP11048", "pointing_type": "ON", "spectrogram": np.zeros((1, 64, 64))},
            {"target": "OFF_3", "pointing_type": "OFF", "spectrogram": np.zeros((1, 64, 64))},
        ]

        # Returns: 0.08, 0.01, 0.08, 0.01, 0.08, 0.01
        with patch("src.cadence_filter.compute_reconstruction_errors", side_effect=[
            np.array([0.08]), np.array([0.01]),
            np.array([0.08]), np.array([0.01]),
            np.array([0.08]), np.array([0.01]),
        ]):
            result = analyze_cadence_sequence(
                mock_model,
                cadence,
                threshold=0.04,
                contrast_threshold=1.5,
            )

        assert result.verdict == "CANDIDATE_TECHNOSIGNATURE"
        assert result.contrast_ratio > 1.5
        assert result.details["has_on_anomaly"] is True
        assert result.details["has_off_anomaly"] is False

    def test_local_rfi_rejection(self, mock_model):
        """Test that anomaly present in both ON and OFF pointings is rejected as LOCAL_RFI."""
        cadence = [
            {"target": "HIP11048", "pointing_type": "ON", "spectrogram": np.zeros((1, 64, 64))},
            {"target": "OFF_1", "pointing_type": "OFF", "spectrogram": np.zeros((1, 64, 64))},
            {"target": "HIP11048", "pointing_type": "ON", "spectrogram": np.zeros((1, 64, 64))},
            {"target": "OFF_2", "pointing_type": "OFF", "spectrogram": np.zeros((1, 64, 64))},
        ]

        with patch("src.cadence_filter.compute_reconstruction_errors", side_effect=[
            np.array([0.08]), np.array([0.09]),
            np.array([0.07]), np.array([0.08]),
        ]):
            result = analyze_cadence_sequence(
                mock_model,
                cadence,
                threshold=0.04,
            )

        assert result.verdict == "LOCAL_RFI"
        assert result.details["has_on_anomaly"] is True
        assert result.details["has_off_anomaly"] is True

    def test_normal_background(self, mock_model):
        """Test clean observations are classified as NORMAL_BACKGROUND."""
        cadence = [
            {"target": "HIP11048", "pointing_type": "ON", "spectrogram": np.zeros((1, 64, 64))},
            {"target": "OFF_1", "pointing_type": "OFF", "spectrogram": np.zeros((1, 64, 64))},
        ]

        with patch("src.cadence_filter.compute_reconstruction_errors", side_effect=[
            np.array([0.01]), np.array([0.02]),
        ]):
            result = analyze_cadence_sequence(
                mock_model,
                cadence,
                threshold=0.04,
            )

        assert result.verdict == "NORMAL_BACKGROUND"
