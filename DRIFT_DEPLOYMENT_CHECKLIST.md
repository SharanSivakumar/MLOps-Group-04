# Drift Detection Deployment Checklist

## Pre-Deployment
- [ ] Read `DRIFT_QUICKSTART.md` to understand the architecture
- [ ] Review `docs/source/drift_detection.md` for detailed documentation
- [ ] Ensure you have GCP credentials configured: `gcloud auth login`
- [ ] Verify GCP project: `gcloud config get-value project`

## Local Setup
- [ ] Sync dependencies: `uv sync`
- [ ] Verify evidently installed: `uv run python -c "import evidently; print(evidently.__version__)"`
- [ ] Run tests locally: `uv run pytest tests/test_drift_detection.py -v`
- [ ] Build drift Docker image locally: `uv run invoke docker-build --target=drift`

## Deployment

### Phase 1: Deploy Inference API with Logging
- [ ] Check current API is working: `curl https://ecg-classification-api-URL/health`
- [ ] Deploy updated API: `gcloud builds submit --config=cloudbuild.yaml`
- [ ] Verify deployment: `gcloud run services describe ecg-classification-api --region=europe-north1`
- [ ] Check Cloud Run logs show drift logging: `gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ecg-classification-api" --limit 5`

### Phase 2: Deploy Drift Detection API
- [ ] Deploy drift service: `gcloud builds submit --config=drift_cloudbuild.yaml`
- [ ] Verify deployment: `gcloud run services describe drift-detection-api --region=europe-north1`
- [ ] Health check: `curl https://drift-detection-api-URL/`
- [ ] Check service logs: `gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=drift-detection-api" --limit 5`

### Phase 3: Setup Automation
- [ ] Get your GCP project ID: `gcloud config get-value project`
- [ ] Setup Cloud Scheduler: `uv run invoke setup-drift-scheduler`
- [ ] Verify scheduler job: `gcloud scheduler jobs list`
- [ ] Check job details: `gcloud scheduler jobs describe drift-check-daily --location=europe-north1`

### Phase 4: Setup Monitoring & Alerts
- [ ] Go to [Cloud Monitoring](https://console.cloud.google.com/monitoring)
- [ ] Create alerting policy:
  - [ ] Metric: Cloud Run request count
  - [ ] Filter: `service_name="drift-detection-api"`
  - [ ] Condition: Success rate threshold
  - [ ] Notification: Email/Slack channel
- [ ] Create log-based alerts in Cloud Logging:
  - [ ] Filter: `jsonPayload.drift_detected=true`
  - [ ] Action: Send notification

## Validation

### Test Prediction Logging
- [ ] Make inference: Send a test ECG to your API
- [ ] Wait 1-2 minutes
- [ ] Check GCS logs: `gsutil ls -r gs://psychic-iridium-484208-c3-mlops-data/drift_logs/`
- [ ] Verify log format: `gsutil cat gs://psychic-iridium-484208-c3-mlops-data/drift_logs/predictions_*.jsonl | head -5`

### Test Drift Detection Manually
- [ ] Get drift API URL: 
  ```bash
  DRIFT_API=$(gcloud run services describe drift-detection-api \
    --region=europe-north1 --format="value(status.url)")
  ```
- [ ] Trigger check:
  ```bash
  curl -X POST "${DRIFT_API}/check_drift" \
    -H "Content-Type: application/json"
  ```
- [ ] Check response includes `drift_detected`, `drifted_features`, `report_path`
- [ ] Review generated report in GCS (if any)

### Test Scheduler
- [ ] Wait until tomorrow 2 AM UTC, OR
- [ ] Manually trigger: `gcloud scheduler jobs run drift-check-daily --location=europe-north1`
- [ ] Monitor logs: `gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=drift-detection-api"`
- [ ] Verify drift report generated

## Performance Checks

- [ ] Inference API latency: Ensure logging doesn't impact response time (<100ms)
- [ ] Drift API memory: Check Cloud Run metrics dashboard
- [ ] GCS costs: Monitor in Cloud Billing
- [ ] Log volume: `gsutil du -s gs://psychic-iridium-484208-c3-mlops-data/drift_logs/`

## Documentation

- [ ] Update project `README.md` with drift detection info
- [ ] Update `reports/README.md` - check off M27, M28 items
- [ ] Add drift detection section to your architecture diagram
- [ ] Document any custom thresholds or configurations

## Ongoing Maintenance

### Daily
- [ ] Check Cloud Monitoring dashboard for alerts
- [ ] Review any drift detection results

### Weekly
- [ ] Review drift detection logs for patterns
- [ ] Monitor storage costs
- [ ] Check if model retraining needed

### Monthly
- [ ] Archive old logs to Cloud Storage (lifecycle rules)
- [ ] Review and update drift thresholds
- [ ] Analyze model performance metrics

## Troubleshooting Guide

If something goes wrong, check:

1. **Inference API not logging**
   - [ ] Check API logs: `gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ecg-classification-api"`
   - [ ] Verify `drift_logger` initialization in `src/api.py`
   - [ ] Ensure GCS credentials are correct

2. **Drift API returns error**
   - [ ] Check service logs: `gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=drift-detection-api"`
   - [ ] Verify reference data exists: `gsutil cat gs://psychic-iridium-484208-c3-mlops-data/models/train.pt > /dev/null`
   - [ ] Check if production logs have data: `gsutil ls gs://psychic-iridium-484208-c3-mlops-data/drift_logs/`

3. **Scheduler job not running**
   - [ ] Verify job exists: `gcloud scheduler jobs describe drift-check-daily --location=europe-north1`
   - [ ] Check job history: `gcloud scheduler jobs list --location=europe-north1`
   - [ ] Manual trigger: `gcloud scheduler jobs run drift-check-daily --location=europe-north1`
   - [ ] Check service account permissions

4. **High costs**
   - [ ] Reduce drift check frequency in Cloud Scheduler
   - [ ] Implement log retention/lifecycle in GCS
   - [ ] Reduce buffer size in DriftLogger if needed

## Support & Resources

- **Code**: See implementation in `src/drift_detection.py`, `src/drift_api.py`
- **Tests**: `tests/test_drift_detection.py`
- **Docs**: `docs/source/drift_detection.md`
- **Tasks**: `uv run invoke --list` (shows drift-related tasks)

## Sign-Off

- [ ] All tests pass locally
- [ ] All phases deployed successfully
- [ ] Monitoring and alerts configured
- [ ] Team members trained on drift detection
- [ ] Documentation complete
- [ ] Ready for production use

---

**Date Completed**: _______________
**Team Members**: _______________
**Notes**: 
```
[Add any custom configurations or observations here]
```
