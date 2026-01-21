"""Drift Detection API - Deploy as separate Cloud Run service."""

from datetime import datetime
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.drift_detection import DriftDetector

app = FastAPI(
    title="ECG Drift Detection API",
    description="API for monitoring data drift in ECG classification model",
    version="1.0.0",
)

detector = DriftDetector()


class DriftCheckResponse(BaseModel):
    """Response schema for drift detection."""

    drift_detected: bool
    drifted_features: list[str]
    report_path: str
    n_reference_samples: int
    n_production_samples: int
    timestamp: str


@app.get("/")
def healthcheck() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "message": "Drift Detection Service is ready"}


@app.post("/check_drift", response_model=DriftCheckResponse)
async def check_drift(bucket_name: str = "psychic-iridium-484208-c3-mlops-data") -> DriftCheckResponse:
    """
    Check for data drift between training data and production data.

    Args:
        bucket_name: GCS bucket containing production logs

    Returns:
        DriftCheckResponse with drift detection results
    """
    try:
        # Load production data from GCS
        production_data = detector.load_production_data(bucket_name)

        if len(production_data) == 0:
            raise HTTPException(status_code=404, detail="No production data found in GCS")

        # Detect drift
        result = detector.detect_drift(production_data)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return DriftCheckResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")


@app.get("/drift_status")
async def drift_status() -> Dict:
    """Get current drift monitoring status."""
    return {
        "service": "drift_detection",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "reference_data": "data/processed/train.pt",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
