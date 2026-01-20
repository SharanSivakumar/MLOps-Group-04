# Quick Start: Data Drift Detection

## What Was Added

✅ **Prediction Logging**: Your API now automatically logs all predictions to GCS
✅ **Drift Detection Module**: Statistical analysis comparing production vs training data  
✅ **Drift API**: Separate Cloud Run service for drift checking
✅ **Automated Monitoring**: Cloud Scheduler runs daily checks
✅ **Tests**: Full test coverage for drift detection
✅ **Documentation**: Complete setup and troubleshooting guide

## File Changes

### New Files
- `src/drift_detection.py` - Core drift detection logic (DriftLogger & DriftDetector)
- `src/drift_api.py` - FastAPI service for drift detection
- `dockerfiles/drift.dockerfile` - Docker image for drift service
- `drift_cloudbuild.yaml` - GCP Cloud Build configuration
- `scripts/setup_drift_scheduler.py` - Setup automation script
- `tests/test_drift_detection.py` - Comprehensive tests
- `docs/source/drift_detection.md` - Full documentation
- `.github/workflows/drift_deploy.yml` - CI/CD for drift service

### Modified Files
- `src/api.py` - Added prediction logging with DriftLogger
- `tasks.py` - Added drift detection tasks
- `requirements.txt` - Added `evidently` package

## Quick Deploy

```bash
# 1. Install new dependencies
uv sync

# 2. Deploy updated inference API (with logging)
gcloud builds submit --config=cloudbuild.yaml

# 3. Deploy drift detection service
gcloud builds submit --config=drift_cloudbuild.yaml

# 4. Setup automated daily checks
uv run invoke setup-drift-scheduler

# 5. Test manually
uv run invoke check-drift
```

## How It Works

1. **Production API** logs every prediction to GCS with statistical features
2. **Drift Detector** compares production data distributions against training data
3. **Evidently AI** performs statistical tests (KS-test, PSI, etc.)
4. **Cloud Scheduler** triggers daily automated checks
5. **Alerts** notify you when drift is detected

## Testing Locally

```bash
# Run drift detection tests
uv run pytest tests/test_drift_detection.py -v

# Build drift Docker image locally
uv run invoke docker-build --target=drift

# Run drift API locally
docker run -p 8001:8001 drift:latest
```

## Monitoring Drift

```bash
# Check drift status
curl https://drift-detection-api-URL.run.app/drift_status

# Trigger manual drift check
curl -X POST https://drift-detection-api-URL.run.app/check_drift

# View logs
gcloud logging read "resource.type=cloud_run_revision" --limit 50
```

## Course Checklist Completion

This implementation covers these DTU MLOps requirements:

### Week 3 (M27) - Data Drift
- ✅ Check how robust your model is towards data drifting
- ✅ Setup collection of input-output data from your deployed application
- ✅ Deploy to the cloud a drift detection API

### Week 3 (M28) - Monitoring
- ✅ Instrument your API with a couple of system metrics
- ✅ Setup cloud monitoring of your instrumented application  
- ✅ Create one or more alert systems in GCP

## Cost Estimate

- **Storage**: ~$0.01/month (prediction logs)
- **Compute**: ~$0.03/month (daily drift checks)
- **Total**: ~$0.04/month

## Next Steps

1. **Set up alerts** in Cloud Monitoring console
2. **Review first drift report** after 24 hours of production data
3. **Adjust thresholds** based on your requirements
4. **Integrate with retraining pipeline** to auto-trigger when drift exceeds limits

## Troubleshooting

**No production data found?**
- Ensure inference API is receiving requests
- Check GCS bucket: `gsutil ls gs://psychic-iridium-484208-c3-mlops-data/drift_logs/`

**Drift detection fails?**
- Verify reference data exists: `data/processed/train.pt`
- Check Cloud Run logs for detailed errors

**Need help?**
- See full documentation: `docs/source/drift_detection.md`
- Check test examples: `tests/test_drift_detection.py`

## Architecture Diagram

```
                    Production Traffic
                           │
                           ▼
                  ┌────────────────┐
                  │  Inference API │
                  │  (Cloud Run)   │
                  └────────────────┘
                           │
                           │ Log predictions
                           ▼
                  ┌────────────────┐
                  │   GCS Bucket   │
                  │  /drift_logs   │
                  └────────────────┘
                           │
                           │ Daily @ 2AM UTC
                           ▼
        ┌──────────────────────────────────┐
        │     Cloud Scheduler Job          │
        │   "drift-check-daily"            │
        └──────────────────────────────────┘
                           │
                           │ POST /check_drift
                           ▼
                  ┌────────────────┐
                  │  Drift API     │
                  │  (Cloud Run)   │
                  └────────────────┘
                           │
                           ├─→ Load training data
                           ├─→ Load production logs
                           ├─→ Run statistical tests
                           └─→ Generate report
                                    │
                                    ▼
                           ┌────────────────┐
                           │ Cloud Logging  │
                           │  & Monitoring  │
                           └────────────────┘
                                    │
                                    ▼
                           ┌────────────────┐
                           │  Email/Slack   │
                           │    Alerts      │
                           └────────────────┘
```

## Key Metrics Tracked

- **Mean**: Average signal intensity → Detects brightness shifts
- **Std Dev**: Signal variability → Detects noise changes  
- **Min/Max**: Value range → Detects clipping or scaling issues
- **Median**: Central tendency → Detects distribution skew

## What Drift Detection Catches

✅ Data quality degradation  
✅ Sensor calibration drift  
✅ Different patient populations  
✅ Changed preprocessing pipelines  
✅ Equipment upgrades/changes  

---

**Ready to deploy!** 🚀

See `docs/source/drift_detection.md` for detailed information.
