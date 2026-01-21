# Data Drift Detection Setup Guide

This guide explains how to set up and use data drift detection for your ECG classification model in GCP.

## Overview

Data drift detection monitors your production model by comparing incoming data distributions against your training data. This helps detect when model performance may degrade due to changes in input data patterns.

## Architecture

```
┌─────────────────┐
│  Cloud Run API  │
│  (Inference)    │──┐
└─────────────────┘  │
                     │ Logs predictions
                     ▼
              ┌──────────────┐
              │  GCS Bucket  │
              │  drift_logs/ │
              └──────────────┘
                     │
                     │ Daily check
                     ▼
         ┌────────────────────┐
         │  Cloud Run API     │
         │  (Drift Detection) │
         └────────────────────┘
                     │
                     ▼
         ┌────────────────────┐
         │ Cloud Scheduler    │
         │ (Automated checks) │
         └────────────────────┘
```

## Components

### 1. **DriftLogger** (src/drift_detection.py)
- Logs all predictions with statistical features
- Buffers logs locally and flushes to GCS
- Integrated into main inference API

### 2. **DriftDetector** (src/drift_detection.py)
- Loads reference data from training set
- Compares production data against reference
- Uses Evidently AI for statistical drift tests
- Generates HTML reports

### 3. **Drift Detection API** (src/drift_api.py)
- Separate Cloud Run service
- Endpoint: `/check_drift`
- Runs drift detection on demand

### 4. **Cloud Scheduler**
- Triggers drift checks daily at 2 AM UTC
- Calls drift detection API automatically

## Setup Instructions

### Step 1: Deploy Updated Inference API

The main API now logs all predictions to GCS:

```bash
# Build and deploy with updated logging
gcloud builds submit --config=cloudbuild.yaml
```

### Step 2: Deploy Drift Detection API

```bash
# Build and deploy drift detection service
gcloud builds submit --config=drift_cloudbuild.yaml
```

### Step 3: Setup Cloud Scheduler

```bash
# Run setup script (replace with your project ID)
uv run python scripts/setup_drift_scheduler.py psychic-iridium-484208-c3
```

### Step 4: Configure Monitoring & Alerts

1. Go to [Cloud Monitoring](https://console.cloud.google.com/monitoring)
2. Create an alerting policy:
   - **Metric**: Cloud Run request count
   - **Filter**: `service_name="drift-detection-api"`
   - **Condition**: Success rate < 100% (indicates drift detected)
   - **Notification**: Email or Slack

3. Create log-based alerts:
   ```
   resource.type="cloud_run_revision"
   resource.labels.service_name="drift-detection-api"
   jsonPayload.drift_detected=true
   ```

## Usage

### Manual Drift Check

```bash
# Get drift API URL
DRIFT_API_URL=$(gcloud run services describe drift-detection-api \
  --region=europe-north1 --format="value(status.url)")

# Trigger drift check
curl -X POST "${DRIFT_API_URL}/check_drift"
```

### View Drift Reports

Reports are saved to `reports/drift_report_TIMESTAMP.html`

```bash
# List reports in GCS (if uploaded)
gsutil ls gs://psychic-iridium-484208-c3-mlops-data/drift_reports/

# Download latest report
gsutil cp gs://psychic-iridium-484208-c3-mlops-data/drift_reports/drift_report_latest.html .
```

### Check Logs

```bash
# View prediction logs
gcloud logging read "resource.type=cloud_run_revision AND \
  resource.labels.service_name=ecg-classification-api" \
  --limit 50 --format json

# View drift detection logs
gcloud logging read "resource.type=cloud_run_revision AND \
  resource.labels.service_name=drift-detection-api" \
  --limit 50 --format json
```

## Features Monitored

The system tracks these statistical features for drift:
- **Mean**: Average pixel intensity
- **Standard Deviation**: Spread of values
- **Min/Max**: Range of values
- **Median**: Middle value

## Drift Detection Methods

Evidently AI uses multiple statistical tests:
- **Kolmogorov-Smirnov test**: Distribution comparison
- **Chi-squared test**: Categorical drift
- **Population Stability Index (PSI)**: Numerical drift
- **Jensen-Shannon divergence**: Distance between distributions

## Interpreting Results

### No Drift Detected
```json
{
  "drift_detected": false,
  "drifted_features": [],
  "n_production_samples": 1000
}
```
✅ Model is operating normally

### Drift Detected
```json
{
  "drift_detected": true,
  "drifted_features": ["mean", "std"],
  "n_production_samples": 1000
}
```
⚠️ **Actions to take:**
1. Investigate data quality issues
2. Check if data preprocessing changed
3. Consider retraining with recent data
4. Review model performance metrics

## Testing

Run drift detection tests:

```bash
# Run all drift tests
uv run pytest tests/test_drift_detection.py -v

# Run specific test
uv run pytest tests/test_drift_detection.py::test_drift_detector_detect_drift -v
```

## Cost Optimization

### Storage Costs
- Prediction logs: ~1KB per prediction
- 10,000 predictions/day ≈ 10MB/day ≈ 300MB/month
- GCS Standard: ~$0.02/GB → ~$0.01/month

### Compute Costs
- Drift API: On-demand, runs once daily
- Typical run: 2-3 minutes
- Cloud Run: 2 vCPU, 2GB RAM
- Cost: ~$0.001 per run → ~$0.03/month

### Total: ~$0.04/month

## Troubleshooting

### No production data found
**Issue**: `No production data found in GCS`

**Solution**:
1. Ensure inference API is deployed and receiving requests
2. Check GCS bucket permissions
3. Verify drift_logger is initialized in API

### Drift detection fails
**Issue**: `Drift detection failed: ...`

**Solution**:
1. Check reference data exists: `data/processed/train.pt`
2. Ensure Evidently is installed: `uv add evidently`
3. Review logs for detailed error

### High memory usage
**Issue**: Drift detection runs out of memory

**Solution**:
1. Reduce sample size in `DriftDetector._load_reference_data()`
2. Increase Cloud Run memory limit in `drift_cloudbuild.yaml`
3. Process data in batches

## Best Practices

1. **Monitor daily**: Catch drift early before major issues
2. **Set thresholds**: Define acceptable drift levels
3. **Keep reference data updated**: Retrain and update reference periodically
4. **Log metadata**: Include timestamps, versions, and context
5. **Automate responses**: Trigger retraining when drift exceeds thresholds

## Course Alignment (DTU MLOps Week 3)

This setup covers:
- ✅ M27: Check model robustness to data drifting
- ✅ M27: Setup collection of input-output data from deployed app
- ✅ M27: Deploy drift detection API to cloud
- ✅ M28: Instrument API with metrics (via logging)
- ✅ M28: Setup cloud monitoring
- ✅ M28: Create alert systems in GCP

## References

- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [GCP Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud Scheduler Documentation](https://cloud.google.com/scheduler/docs)
- [DTU MLOps Course Material](https://github.com/SkafteNicki/dtu_mlops)
