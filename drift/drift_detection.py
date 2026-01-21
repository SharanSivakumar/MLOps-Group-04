"""Data drift detection module for ECG classification."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from evidently import Report
from evidently.metrics import ValueDrift
from google.cloud import storage


class DriftLogger:
    """Logs predictions and inputs to GCS for drift monitoring."""

    def __init__(self, bucket_name: str = "psychic-iridium-484208-c3-mlops-data", logs_prefix: str = "drift_logs"):
        self.bucket_name = bucket_name
        self.logs_prefix = logs_prefix
        self.local_buffer: List[Dict] = []
        self.buffer_size = 100  # Flush to GCS after 100 predictions

    def log_prediction(
        self,
        input_data: np.ndarray,
        prediction: int,
        probabilities: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Log a single prediction with input data."""
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Extract features from input (e.g., statistical summaries)
        features = {
            "mean": float(input_data.mean()),
            "std": float(input_data.std()),
            "min": float(input_data.min()),
            "max": float(input_data.max()),
            "median": float(np.median(input_data)),
        }

        log_entry = {
            "timestamp": timestamp.isoformat(),
            "prediction": int(prediction),
            "probabilities": probabilities.tolist(),
            "features": features,
        }

        self.local_buffer.append(log_entry)

        # Flush to GCS if buffer is full
        if len(self.local_buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Write buffered logs to GCS."""
        if not self.local_buffer:
            return

        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)

            # Create filename with timestamp
            filename = f"{self.logs_prefix}/predictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
            blob = bucket.blob(filename)

            # Write as JSON lines
            content = "\n".join(json.dumps(entry) for entry in self.local_buffer)
            blob.upload_from_string(content, content_type="application/x-ndjson")

            print(f"Flushed {len(self.local_buffer)} predictions to gs://{self.bucket_name}/{filename}")
            self.local_buffer.clear()
        except Exception as e:
            print(f"Error flushing logs to GCS: {e}")


class DriftDetector:
    """Detects data drift using Evidently AI."""

    def __init__(self, reference_data_path: str = "data/processed/train.pt"):
        self.reference_data_path = reference_data_path
        self.reference_data: Optional[np.ndarray] = None
        self.class_names = ["AF", "Noise", "NSR"]

    def _load_reference_data(self) -> np.ndarray:
        """Load and prepare reference data from training set (lazy-loaded)."""
        if self.reference_data is not None:
            return self.reference_data
            
        try:
            data = torch.load(self.reference_data_path)
            X = data["x"].numpy()

            # Extract statistical features similar to DriftLogger
            n_samples = min(1000, X.shape[0])  # Use subset for efficiency
            features = []
            for i in range(n_samples):
                sample = X[i]
                features.append(
                    [
                        sample.mean(),
                        sample.std(),
                        sample.min(),
                        sample.max(),
                        np.median(sample),
                    ]
                )

            self.reference_data = np.array(features)
            return self.reference_data
        except FileNotFoundError:
            print(f"Warning: Reference data file not found at {self.reference_data_path}")
            # Return dummy data if file not found
            return np.array([[0.0, 0.1, -0.5, 0.5, 0.0]] * 100)
            )

        return np.array(features)

    def load_production_data(self, bucket_name: str, prefix: str = "drift_logs") -> np.ndarray:
        """Load production data from GCS logs."""
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=prefix))

            production_features = []
            for blob in blobs:
                if not blob.name.endswith(".jsonl"):
                    continue

                content = blob.download_as_text()
                for line in content.strip().split("\n"):
                    if not line:
                        continue
                    entry = json.loads(line)
                    features = entry["features"]
                    production_features.append(
                        [
                            features["mean"],
                            features["std"],
                            features["min"],
                            features["max"],
                            features["median"],
                        ]
                    )

            return np.array(production_features) if production_features else np.array([])
        except Exception as e:
            print(f"Error loading production data: {e}")
            return np.array([])

    def detect_drift(self, production_data: np.ndarray) -> Dict:
        """Detect drift using Evidently ValueDrift metrics."""
        if len(production_data) == 0:
            return {"error": "No production data available"}

        # Lazy-load reference data
        reference_data = self._load_reference_data()        feature_names = ["mean", "std", "min", "max", "median"]
        reference_df = pd.DataFrame(self.reference_data, columns=feature_names)
        production_df = pd.DataFrame(production_data, columns=feature_names)

        # Create ValueDrift metrics for each feature
        metrics = [ValueDrift(column=col) for col in feature_names]
        
        report = Report(metrics=metrics)
        report.run(reference_data=reference_df, current_data=production_df)

        # Save report as HTML
        report_path = f"reports/drift_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
        Path("reports").mkdir(exist_ok=True)
        report.save_html(report_path)

        # Extract drift results
        report_dict = report.as_dict()
        drift_detected = False
        drifted_features = []

        # Check each metric result for drift
        if "metrics" in report_dict:
            for metric in report_dict["metrics"]:
                result = metric.get("result", {})
                if result.get("drift_detected", False):
                    column_name = result.get("column_name", "unknown")
                    drifted_features.append(column_name)
                    drift_detected = True

        return {
            "drift_detected": drift_detected,
            "drifted_features": drifted_features,
            "report_path": report_path,
            "n_reference_samples": len(reference_df),
            "n_production_samples": len(production_df),
            "timestamp": datetime.utcnow().isoformat(),
        }


if __name__ == "__main__":
    # Test drift detection
    detector = DriftDetector()
    production_data = detector.load_production_data("psychic-iridium-484208-c3-mlops-data")

    if len(production_data) > 0:
        result = detector.detect_drift(production_data)
        print("Drift Detection Results:")
        print(f"  Drift Detected: {result['drift_detected']}")
        print(f"  Drifted Features: {result['drifted_features']}")
        print(f"  Report: {result['report_path']}")
    else:
        print("No production data available yet")
