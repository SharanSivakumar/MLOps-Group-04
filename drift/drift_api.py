import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
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
    return detector.get_reference_dataframe()


def load_latest_files(bucket_name: str, n: int = 5) -> pd.DataFrame:
    """Load the latest n files from GCS and return as DataFrame."""
    production_df = detector.load_production_data(bucket_name, include_predictions=True)
    if isinstance(production_df, pd.DataFrame) and not production_df.empty:
        if len(production_df) > n * 100:
            production_df = production_df.tail(n * 100)
        return production_df

    return pd.DataFrame(columns=["mean", "std", "min", "max", "median", "label"])


def run_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> str:
    """Run the analysis and return the HTML content as a string with labels shown alongside numeric features."""
    numeric_features = ["mean", "std", "min", "max", "median"]
    
    if "label" in reference_data.columns and "label" in current_data.columns:
        reference_data = reference_data.copy()
        current_data = current_data.copy()
        
        reference_data["label"] = reference_data["label"].astype(str)
        current_data["label"] = current_data["label"].astype(str)
    
    text_overview_report = Report(metrics=[DataDriftPreset()])
    snapshot = text_overview_report.run(reference_data=reference_data, current_data=current_data)

    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as tmp:
            tmp_file = tmp.name
            snapshot.save_html(tmp.name)
            tmp.flush()
            html_content = Path(tmp.name).read_text(encoding="utf-8")
            
            if "label" in reference_data.columns and "label" in current_data.columns:
                html_content = enhance_html_with_label_stats(html_content, reference_data, current_data, numeric_features)
        
        return html_content
    finally:
        if tmp_file and Path(tmp_file).exists():
            Path(tmp_file).unlink()


def enhance_html_with_label_stats(html_content: str, reference_data: pd.DataFrame, current_data: pd.DataFrame, numeric_features: list) -> str:
    """Enhance HTML report with label statistics shown alongside numeric features."""
    label_stats_html = generate_label_stats_table(reference_data, current_data, numeric_features)
    
    insertion_point = html_content.find("</body>")
    if insertion_point != -1:
        html_content = html_content[:insertion_point] + label_stats_html + html_content[insertion_point:]
    
    return html_content


def generate_label_stats_table(reference_data: pd.DataFrame, current_data: pd.DataFrame, numeric_features: list) -> str:
    """Generate HTML table showing numeric statistics grouped by label."""
    html = """
    <div style="margin: 20px; padding: 20px; background-color: #f5f5f5; border-radius: 8px;">
        <h2 style="color: #2c3e50; margin-bottom: 20px;">Numeric Features Statistics by Label</h2>
    """
    
    for dataset_name, dataset in [("Reference (Training)", reference_data), ("Current (Production)", current_data)]:
        html += f"""
        <div style="margin-bottom: 30px;">
            <h3 style="color: #34495e; margin-bottom: 15px;">{dataset_name}</h3>
            <table style="width: 100%; border-collapse: collapse; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <thead>
                    <tr style="background-color: #3498db; color: white;">
                        <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Label</th>
        """
        
        for feature in numeric_features:
            html += f'<th style="padding: 12px; text-align: center; border: 1px solid #ddd;">{feature.capitalize()}</th>'
        
        html += """
                    </tr>
                </thead>
                <tbody>
        """
        
        if "label" in dataset.columns:
            for label in sorted(dataset["label"].unique()):
                label_data = dataset[dataset["label"] == label]
                html += f"""
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 12px; font-weight: bold; border: 1px solid #ddd; background-color: #ecf0f1; color: #2c3e50;">{label}</td>
                """
                
                for feature in numeric_features:
                    if feature in label_data.columns:
                        mean_val = label_data[feature].mean()
                        median_val = label_data[feature].median()
                        std_val = label_data[feature].std()
                        html += f"""
                        <td style="padding: 12px; text-align: center; border: 1px solid #ddd; color: #2c3e50; background-color: white;">
                            <div style="font-size: 0.9em; color: #2c3e50;">
                                <div style="color: #2c3e50;"><strong style="color: #2c3e50;">Mean:</strong> <span style="color: #2c3e50;">{mean_val:.4f}</span></div>
                                <div style="color: #2c3e50;"><strong style="color: #2c3e50;">Median:</strong> <span style="color: #2c3e50;">{median_val:.4f}</span></div>
                                <div style="color: #2c3e50;"><strong style="color: #2c3e50;">Std:</strong> <span style="color: #2c3e50;">{std_val:.4f}</span></div>
                            </div>
                        </td>
                        """
                    else:
                        html += '<td style="padding: 12px; text-align: center; border: 1px solid #ddd; color: #2c3e50; background-color: white;">-</td>'
                
                html += "</tr>"
        
        html += """
                </tbody>
            </table>
        </div>
        """
    
    html += """
    </div>
    """
    
    return html


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


@app.get("/report")
async def get_report(n: int = 5, bucket_name: str = "psychic-iridium-484208-c3-mlops-data"):
    """Generate and return the report as HTML that displays in browser."""
    try:
        training_data = load_training_data()
        prediction_data = load_latest_files(bucket_name, n=n)
        
        if len(prediction_data) == 0:
            raise HTTPException(status_code=404, detail="No production data found in GCS")
        
        html_content = run_analysis(training_data, prediction_data)

        if not html_content or len(html_content.strip()) < 20:
            raise HTTPException(status_code=500, detail="Generated report is empty")

        return Response(
            content=html_content,
            media_type="text/html",
            headers={"Content-Disposition": "inline"}
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Report file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
