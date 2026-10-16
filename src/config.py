"""
Project configuration and constants for AI Search for Unknown Technosignatures
Using Deep Anomaly Detection.
"""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
SAMPLES_DATA_DIR = DATA_DIR / "samples"

RAW_DATA_DIR = DATA_DIR / "raw"
BL_CACHE_DIR = DATA_DIR / "bl_cache"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

# Ensure directories exist
for d in [SYNTHETIC_DATA_DIR, SAMPLES_DATA_DIR, RAW_DATA_DIR, BL_CACHE_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Breakthrough Listen Open Data API defaults
BL_PROD_URL = "http://seti.berkeley.edu/opendata"
BL_DEV_URL = "http://35.236.84.6:5001"
DEFAULT_BL_MAX_DOWNLOAD_MB = 200.0  # Max download limit per file to guard disk space

# Signal processing & Spectrogram parameters
DEFAULT_SAMPLE_RATE = 1000.0  # Hz (synthetic base)
DEFAULT_DURATION = 1.0        # seconds
DEFAULT_N_FFT = 128
DEFAULT_HOP_LENGTH = 32
SPEC_HEIGHT = 64
SPEC_WIDTH = 64

# Model parameters
LATENT_DIM = 48
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 20

# Anomaly thresholding
DEFAULT_ANOMALY_PERCENTILE = 99.0
