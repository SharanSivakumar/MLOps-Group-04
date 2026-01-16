import random

from locust import HttpUser, between, task


class ECGInferenceUser(HttpUser):
    """Locust user class for testing the ECG Classification API."""

    wait_time = between(1, 3)

    @task(1)
    def get_root(self) -> None:
        """Test the root endpoint."""
        self.client.get("/")

    @task(1)
    def health_check(self) -> None:
        """Test the health check endpoint."""
        self.client.get("/health")

    @task(5)
    def predict_ecg(self) -> None:
        """Test the prediction endpoint with random ECG data."""
        # Generate random 224x224 ECG data
        ecg_data = [[random.uniform(-1, 1) for _ in range(224)] for _ in range(224)]
        
        payload = {"data": ecg_data}
        
        self.client.post(
            "/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )