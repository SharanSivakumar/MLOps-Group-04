"""Drift Detection API - Deploy as separate Cloud Run service."""

import anyio
from datetime import datetime
from typing import Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from evidently import Report
from evidently.presets import DataDriftPreset

from drift_detection import DriftDetector

app = FastAPI(
    title="ECG Drift Detection API",
    description="API for monitoring data drift in ECG classification model",
    version="1.0.0",
)

detector = DriftDetector()


def load_training_data() -> pd.DataFrame:
    """Load training data as DataFrame for drift analysis."""
    reference_data = detector._load_reference_data()
    feature_names = ["mean", "std", "min", "max", "median"]
    return pd.DataFrame(reference_data, columns=feature_names)


def load_latest_files(bucket_name: str, n: int = 5) -> pd.DataFrame:
    """Load the latest n files from GCS and return as DataFrame."""
    production_data = detector.load_production_data(bucket_name)
    if len(production_data) == 0:
        return pd.DataFrame(columns=["mean", "std", "min", "max", "median"])
    
    feature_names = ["mean", "std", "min", "max", "median"]
    production_df = pd.DataFrame(production_data, columns=feature_names)
    
    if len(production_df) > n * 100:
        production_df = production_df.tail(n * 100)
    
    return production_df


def run_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> None:
    """Run the analysis and return the report."""
    text_overview_report = Report(metrics=[DataDriftPreset()])
    snapshot = text_overview_report.run(reference_data=reference_data, current_data=current_data)
    snapshot.save_html("monitoring.html")


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


@app.get("/report", response_class=HTMLResponse)
async def get_report(n: int = 5, bucket_name: str = "psychic-iridium-484208-c3-mlops-data"):
    """Generate and return the report."""
    try:
        training_data = load_training_data()
        prediction_data = load_latest_files(bucket_name, n=n)
        
        if len(prediction_data) == 0:
            raise HTTPException(status_code=404, detail="No production data found in GCS")
        
        run_analysis(training_data, prediction_data)
        
        async with await anyio.open_file("monitoring.html", encoding="utf-8") as f:
            html_content = await f.read()
        
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Report file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
