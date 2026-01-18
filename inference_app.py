from pathlib import Path
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import ECGClassifier

app = FastAPI(
    title="ECG Classification API",
    description="API for ECG signal classification using a trained EfficientNet model",
    version="1.0.0",
)

# Model and configuration
CHECKPOINT_PATH = Path("checkpoints/ecg-epoch=01-val_loss=0.16.ckpt")
CLASS_NAMES = ["AF", "Noise", "NSR"]
MODEL = None


class ECGInput(BaseModel):
    """Input schema for ECG data."""

    data: List[List[float]] = Field(
        ..., description="2D ECG signal array of shape (224, 224)", min_items=224, max_items=224
    )


class PredictionOutput(BaseModel):
    """Output schema for predictions."""

    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: dict[str, float]


def load_model() -> ECGClassifier:
    """Load the trained model from checkpoint."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            f"Available checkpoints: {list(Path('checkpoints').glob('*.ckpt'))}"
        )

    model = ECGClassifier.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()
    model.freeze()
    return model


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global MODEL
    MODEL = load_model()
    print(f"Model loaded successfully from {CHECKPOINT_PATH}")


@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "name": "ECG Classification API",
        "version": "1.0.0",
        "model_checkpoint": str(CHECKPOINT_PATH),
        "classes": CLASS_NAMES,
        "endpoints": {"/predict": "POST - Make predictions on ECG data", "/health": "GET - Check API health status"},
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": MODEL is not None, "checkpoint": str(CHECKPOINT_PATH)}


@app.post("/predict", response_model=PredictionOutput)
def predict(ecg_input: ECGInput):
    """
    Perform inference on ECG signal data.

    Args:
        ecg_input: ECG signal data as a 224x224 2D array

    Returns:
        Prediction with class name, confidence, and all class probabilities
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to numpy array and validate shape
        data = np.array(ecg_input.data, dtype=np.float32)

        if data.shape != (224, 224):
            raise HTTPException(status_code=400, detail=f"Invalid input shape {data.shape}. Expected (224, 224)")

        # Prepare tensor: add batch and channel dimensions (1, 1, 224, 224)
        tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            logits = MODEL(tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_idx].item()

        # Prepare response
        prob_dict = {class_name: float(probabilities[0, i].item()) for i, class_name in enumerate(CLASS_NAMES)}

        return PredictionOutput(
            predicted_class=CLASS_NAMES[predicted_idx],
            predicted_index=predicted_idx,
            confidence=confidence,
            probabilities=prob_dict,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
