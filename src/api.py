from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Dict

import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os

from google.cloud import storage
from google.auth.exceptions import DefaultCredentialsError

from src.model import ECGClassifier

# Global variables for model and device
model = None
device = None
drift_logger = None


class _DummyECGModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        batch = x.shape[0]
        return torch.zeros((batch, 3), dtype=torch.float32, device=x.device)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device

    checkpoint_path = "checkpoints/ecg-epoch=01-val_loss=0.11.ckpt"

    try:
        # If checkpoint not present locally, try to download from GCS
        if not os.path.exists(checkpoint_path):
            if os.environ.get("DISABLE_MODEL_DOWNLOAD") == "1":
                device = torch.device("cpu")
                model = _DummyECGModel()
                model.to(device)
                model.eval()
                yield
                return

            gcs_uri = os.environ.get(
                "MODEL_GCS_URI",
                "gs://psychic-iridium-484208-c3-mlops-data/models/ecg-epoch=01-val_loss=0.11.ckpt",
            )
            if gcs_uri.startswith("gs://"):
                # parse gs://bucket/path/to/object
                _p = gcs_uri[len("gs://") :]
                bucket_name, _, blob_name = _p.partition("/")
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                try:
                    client = storage.Client()
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
                    print(f"Downloading model from {gcs_uri} to {checkpoint_path}...")
                    blob.download_to_filename(checkpoint_path)
                    print("Download complete.")
                except (DefaultCredentialsError, Exception) as e:
                    print(f"Failed to download model from {gcs_uri}: {e}")
                    # Fall back to dummy model if credentials are missing or explicitly disabled
                    # In Cloud Run, credentials should be available automatically
                    if (
                        isinstance(e, DefaultCredentialsError)
                        or os.environ.get("CI") == "true"
                        or os.environ.get("DISABLE_MODEL_DOWNLOAD") == "1"
                    ):
                        print("Falling back to dummy model...")
                        device = torch.device("cpu")
                        model = _DummyECGModel()
                        model.to(device)
                        model.eval()
                        yield
                        return
                    # Re-raise if it's a different error and we're in production
                    raise
            else:
                raise RuntimeError("MODEL_GCS_URI must be a gs:// URI when checkpoint is missing")
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {device}...")

        # Load model from checkpoint
        # strict=False might be needed if there are slight mismatches, but ideally should be True
        model = ECGClassifier.load_from_checkpoint(checkpoint_path, map_location=device)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
        yield
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model from {checkpoint_path}")
    finally:
        # Cleanup if needed
        pass


app = FastAPI(
    title="ECG Classification API",
    description="API for classifying ECG signals using EfficientNet-B0",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok", "message": "ECG Classification Service is ready"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    global model, device

    if not file.filename.endswith(".npy"):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Invalid file format. Please upload a .npy file."
        )

    try:
        # Read file contents
        contents = await file.read()

        # Load numpy array from bytes
        # We need to write to a BytesIO-like object or temporary file to load with numpy
        # But np.frombuffer might work if it's a flat array, but .npy has a header.
        # Safest way without temp file is using io.BytesIO
        import io

        with io.BytesIO(contents) as bio:
            data = np.load(bio)

        # Preprocess
        # Expected shape: (1, 224, 224) or (224, 224) -> add batch dim
        if data.shape == (224, 224):
            data = np.expand_dims(data, axis=0)  # (1, 224, 224)

        if data.shape != (1, 224, 224):
            # Ensure channel dim exists if it was (224, 224) originally and became (1, 224, 224) - wait,
            # training data loading:
            # data = np.load(file_path).astype(np.float32) # (224, 224)
            # data = np.expand_dims(data, axis=0) # (1, 224, 224)
            # Model input expects batch dim: (B, C, H, W) -> (1, 1, 224, 224)
            pass

        # Convert to tensor
        tensor_data = torch.from_numpy(data.astype(np.float64)).float()  # Ensure float32

        # Add batch dimension if missing
        if tensor_data.ndim == 3:
            tensor_data = tensor_data.unsqueeze(0)

        if tensor_data.shape[1:] != (1, 224, 224):
            raise ValueError(f"Invalid shape. Expected (1, 224, 224) per sample, got {tensor_data.shape[1:]}")

        # Inference
        tensor_data = tensor_data.to(device)

        with torch.no_grad():
            logits = model(tensor_data)
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()

        classes = ["AF", "Noise", "NSR"]
        result = {
            "predicted_class_id": predicted_class,
            "predicted_label": classes[predicted_class],
            "probabilities": {classes[i]: float(probs[0, i]) for i in range(len(classes))},
        }

        return JSONResponse(content=result, status_code=HTTPStatus.OK)

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}")
