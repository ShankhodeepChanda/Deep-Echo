import pytest
from src import config

def test_config_paths():
    assert config.BASE_DIR.exists()
    assert config.SYNTHETIC_DATA_DIR.exists()
    assert config.MODELS_DIR.exists()
    assert config.SPEC_HEIGHT == 64
    assert config.SPEC_WIDTH == 64
    assert config.DEFAULT_ANOMALY_PERCENTILE == 99.0
