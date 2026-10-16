"""
Cadence Verification and Reference Comparison for SETI Observations.

Implements Phase 11: Feature Extraction & Reference Comparison using the
standard radio astronomy ABACAD cadence technique:
- A: ON-target pointing (target star)
- B, C, D: OFF-target pointings (reference background sky)

Distinguishes:
1. CANDIDATE_TECHNOSIGNATURE: Signal detected in ON-target (A) but absent in OFF-targets (B, C, D).
2. LOCAL_RFI: Signal detected in both ON-target and OFF-target pointings (originating from Earth/satellites).
3. NORMAL_BACKGROUND: No anomalous signal detected in any pointing.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import logging

from src.training import compute_reconstruction_errors

logger = logging.getLogger(__name__)


class CadenceAnalysisResult:
    """Stores result of a multi-pointing cadence comparison."""

    def __init__(
        self,
        target_name: str,
        observations: List[Dict[str, Any]],
        verdict: str,
        on_target_error: float,
        off_target_error: float,
        contrast_ratio: float,
        details: Dict[str, Any],
    ):
        self.target_name = target_name
        self.observations = observations
        self.verdict = verdict
        self.on_target_error = on_target_error
        self.off_target_error = off_target_error
        self.contrast_ratio = contrast_ratio
        self.details = details

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_name": self.target_name,
            "verdict": self.verdict,
            "on_target_error": self.on_target_error,
            "off_target_error": self.off_target_error,
            "contrast_ratio": self.contrast_ratio,
            "details": self.details,
            "num_pointings": len(self.observations),
        }


def analyze_cadence_sequence(
    model: Any,
    cadence_observations: List[Dict[str, Any]],
    threshold: float,
    contrast_threshold: float = 2.0,
    device: Optional[str] = None,
) -> CadenceAnalysisResult:
    """
    Analyze a sequence of observations forming an ABACAD cadence.
    
    Parameters
    ----------
    model : CNNAutoencoder
        Trained autoencoder.
    cadence_observations : list of dict
        List of observation dictionaries, each containing:
        - 'target': str (e.g. 'HIP11048' or 'OFF_POINTING_1')
        - 'pointing_type': str ('ON' or 'OFF', or inferred from primary target)
        - 'spectrogram': np.ndarray of shape (1, H, W) or (H, W)
    threshold : float
        Anomaly detection threshold (e.g. from calibrate_threshold).
    contrast_threshold : float, default 2.0
        Minimum ratio between ON-target error and OFF-target error to be
        considered candidate technosignature.
    device : str, optional
        Computation device ('cpu' or 'cuda').

    Returns
    -------
    CadenceAnalysisResult
    """
    if not cadence_observations:
        raise ValueError("cadence_observations list cannot be empty")

    primary_target = cadence_observations[0].get("target", "TARGET_A")

    on_errors: List[float] = []
    off_errors: List[float] = []
    obs_details: List[Dict[str, Any]] = []

    for i, obs in enumerate(cadence_observations):
        spec = obs.get("spectrogram")
        if spec is None:
            raise KeyError(f"Observation index {i} missing 'spectrogram' array")

        # Standardize shape to (1, 1, H, W)
        spec_arr = np.array(spec, dtype=np.float32)
        if spec_arr.ndim == 2:
            spec_arr = spec_arr[np.newaxis, np.newaxis, :, :]
        elif spec_arr.ndim == 3:
            spec_arr = spec_arr[np.newaxis, :, :, :]

        # Compute error
        err = float(compute_reconstruction_errors(model, spec_arr, device=device)[0])

        is_flagged = bool(err > threshold)

        # Determine pointing type (ON if matches primary target or marked ON)
        pointing = obs.get("pointing_type")
        if not pointing:
            target_name = obs.get("target", "")
            pointing = "ON" if target_name == primary_target else "OFF"

        if pointing.upper() == "ON":
            on_errors.append(err)
        else:
            off_errors.append(err)

        obs_details.append({
            "index": i,
            "target": obs.get("target", f"Pointing_{i}"),
            "pointing_type": pointing.upper(),
            "reconstruction_error": err,
            "flagged_anomalous": is_flagged,
        })

    avg_on_error = float(np.mean(on_errors)) if on_errors else 0.0
    avg_off_error = float(np.mean(off_errors)) if off_errors else 1e-6
    contrast_ratio = avg_on_error / max(avg_off_error, 1e-6)

    has_on_anomaly = any(err > threshold for err in on_errors)
    has_off_anomaly = any(err > threshold for err in off_errors)

    if has_on_anomaly:
        if not has_off_anomaly and contrast_ratio >= contrast_threshold:
            verdict = "CANDIDATE_TECHNOSIGNATURE"
        else:
            verdict = "LOCAL_RFI"
    else:
        verdict = "NORMAL_BACKGROUND"

    return CadenceAnalysisResult(
        target_name=primary_target,
        observations=cadence_observations,
        verdict=verdict,
        on_target_error=avg_on_error,
        off_target_error=avg_off_error,
        contrast_ratio=contrast_ratio,
        details={
            "pointings_summary": obs_details,
            "has_on_anomaly": has_on_anomaly,
            "has_off_anomaly": has_off_anomaly,
            "anomaly_threshold": threshold,
            "contrast_threshold": contrast_threshold,
        },
    )
