from fastapi.testclient import TestClient
from src.api import app
import numpy as np
import io
import pytest

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_healthcheck(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_valid_shape(client):
    # Create a dummy .npy file in memory
    # Shape (1, 224, 224)
    data = np.random.randn(1, 224, 224).astype(np.float32)
    
    with io.BytesIO() as bio:
        np.save(bio, data)
        bio.seek(0)
        files = {"file": ("test.npy", bio, "application/octet-stream")}
        response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    json_response = response.json()
    assert "predicted_class_id" in json_response
    assert "predicted_label" in json_response
    assert "probabilities" in json_response
    assert json_response["predicted_label"] in ["AF", "Noise", "NSR"]

def test_predict_incorrect_shape(client):
    # Only 2D array, but API might handle it. Let's send something totally wrong.
    data = np.random.randn(10, 10).astype(np.float32)
    
    with io.BytesIO() as bio:
        np.save(bio, data)
        bio.seek(0)
        files = {"file": ("test.npy", bio, "application/octet-stream")}
        response = client.post("/predict", files=files)
        
    # The API code tries to fix some shapes, but (10,10) should probably fail or error out
    # If it fails, it returns 500 or 400 depending on logic.
    # Current logic: re-raises internal errors as 500.
    # Logic checks: if tensor_data.shape[1:] != (1, 224, 224): raise ValueError
    
    assert response.status_code in [400, 500] 

def test_predict_invalid_extension(client):
    with io.BytesIO(b"dummy data") as bio:
        files = {"file": ("test.txt", bio, "text/plain")}
        response = client.post("/predict", files=files)
    
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]
