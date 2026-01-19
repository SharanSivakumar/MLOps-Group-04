import io
import random

import numpy as np
from locust import HttpUser, between, task


class ECGInferenceUser(HttpUser):
    
    wait_time = between(1, 3)

    @task(1)
    def get_root(self) -> None:
        self.client.get("/")

    @task(5)
    def predict_ecg(self) -> None:
        ecg_data = np.random.uniform(-1, 1, size=(224, 224)).astype(np.float32)
        
        bio = io.BytesIO()
        np.save(bio, ecg_data)
        bio.seek(0)
        
        files = {"file": ("test_ecg.npy", bio, "application/octet-stream")}
        
        self.client.post(
            "/predict",
            files=files,
        )