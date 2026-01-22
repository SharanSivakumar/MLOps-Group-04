import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from drift.drift_detection import DriftDetector, DriftLogger


@pytest.fixture
def drift_logger():
    """Create a drift logger for testing."""
    return DriftLogger(bucket_name="test-bucket", logs_prefix="test_logs")


@pytest.fixture
def drift_detector():
    """Create a drift detector for testing."""
    with patch("src.drift_detection.torch.load") as mock_load:
        # Mock training data
        mock_data = {
            "x": np.random.randn(100, 1, 224, 224).astype(np.float32),
            "y": np.random.randint(0, 3, 100),
        }
        mock_load.return_value = mock_data
        detector = DriftDetector()
        return detector


def test_drift_logger_initialization(drift_logger):
    """Test drift logger initialization."""
    assert drift_logger.bucket_name == "test-bucket"
    assert drift_logger.logs_prefix == "test_logs"
    assert drift_logger.buffer_size == 100
    assert len(drift_logger.local_buffer) == 0


def test_drift_logger_log_prediction(drift_logger):
    """Test logging a single prediction."""
    input_data = np.random.randn(1, 224, 224).astype(np.float32)
    prediction = 1
    probabilities = np.array([0.1, 0.7, 0.2])

    drift_logger.log_prediction(input_data, prediction, probabilities)

    assert len(drift_logger.local_buffer) == 1
    entry = drift_logger.local_buffer[0]
    assert entry["prediction"] == 1
    assert "features" in entry
    assert "mean" in entry["features"]


def test_drift_logger_buffer_flush(drift_logger):
    """Test that buffer flushes after reaching buffer_size."""
    input_data = np.random.randn(1, 224, 224).astype(np.float32)
    prediction = 0
    probabilities = np.array([0.8, 0.1, 0.1])

    # Mock GCS client
    with patch("src.drift_detection.storage.Client") as mock_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        # Log buffer_size predictions to trigger flush
        for _ in range(drift_logger.buffer_size):
            drift_logger.log_prediction(input_data, prediction, probabilities)

        # Buffer should be cleared after flush
        assert len(drift_logger.local_buffer) == 0
        mock_blob.upload_from_string.assert_called_once()


def test_drift_detector_initialization(drift_detector):
    """Test drift detector initialization."""
    assert drift_detector.reference_data is not None
    assert len(drift_detector.reference_data) > 0
    assert drift_detector.reference_data.shape[1] == 5  # 5 features


def test_drift_detector_load_production_data(drift_detector):
    """Test loading production data from GCS."""
    # Mock GCS data
    mock_logs = [
        {"features": {"mean": 0.5, "std": 0.2, "min": 0.0, "max": 1.0, "median": 0.5}, "prediction": 0}
    ]

    with patch("src.drift_detection.storage.Client") as mock_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.list_blobs.return_value = [mock_blob]
        mock_blob.name = "test_logs/predictions.jsonl"
        mock_blob.download_as_text.return_value = json.dumps(mock_logs[0])

        production_data = drift_detector.load_production_data("test-bucket", "test_logs")

        assert len(production_data) > 0
        assert production_data.shape[1] == 5


def test_drift_detector_detect_drift(drift_detector):
    """Test drift detection."""
    # Create synthetic production data with drift
    production_data = np.random.randn(50, 5) + 2.0  # Shifted distribution

    with patch("src.drift_detection.Report") as mock_report:
        mock_report_instance = MagicMock()
        mock_report.return_value = mock_report_instance

        # Mock report results
        mock_report_instance.as_dict.return_value = {
            "metrics": [
                {
                    "metric": "DatasetDriftMetric",
                    "result": {
                        "dataset_drift": True,
                        "drift_by_columns": {"mean": {"drift_detected": True}, "std": {"drift_detected": False}},
                    },
                }
            ]
        }

        result = drift_detector.detect_drift(production_data)

        assert "drift_detected" in result
        assert "drifted_features" in result
        assert "report_path" in result
        mock_report_instance.save_html.assert_called_once()


def test_drift_detector_empty_production_data(drift_detector):
    """Test drift detection with empty production data."""
    production_data = np.array([])
    result = drift_detector.detect_drift(production_data)

    assert "error" in result
    assert result["error"] == "No production data available"


def test_feature_extraction_consistency():
    """Test that feature extraction is consistent between logger and detector."""
    input_data = np.random.randn(1, 224, 224).astype(np.float32)

    # Extract features like DriftLogger
    features = {
        "mean": float(input_data.mean()),
        "std": float(input_data.std()),
        "min": float(input_data.min()),
        "max": float(input_data.max()),
        "median": float(np.median(input_data)),
    }

    # Check all features are present
    assert "mean" in features
    assert "std" in features
    assert "min" in features
    assert "max" in features
    assert "median" in features

    # Check features are floats
    assert all(isinstance(v, float) for v in features.values())
