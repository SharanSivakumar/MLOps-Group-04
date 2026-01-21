To Deploy Drift Detection API follwow the steps below:

```bash
./mlops-group-04/drift && docker build -t drift-detection:latest -f Dockerfile . && docker tag drift-detection:latest europe-north1-docker.pkg.dev/¤/ml-images/drift-api:latest && docker push europe-north1-docker.pkg.dev/$PROJECT_ID/ml-images/drift-api:latest 2>&1 | tail -20
```

```bash
 gcloud run deploy drift-detection-api \
   --image=europe-north1-docker.pkg.dev/$PROJECT_ID/ml-images/drift-api:latest \
   --region=europe-north1 \
   --platform=managed \
   --allow-unauthenticated \
   --memory=2Gi \
   --cpu=2 \
   --timeout=600 \
   --max-instances=5 \
   --project=$PROJECT_ID \
   --port=8080
```