To Deploy Drift Detection API follwow the steps below:

```bash
./mlops-group-04/drift && docker build -t drift-detection:test1 -f Dockerfile . && docker tag drift-detection:test1 europe-north1-docker.pkg.dev/psychic-iridium-484208-c3/ml-images/drift-api:test1 && docker push europe-north1-docker.pkg.dev/psychic-iridium-484208-c3/ml-images/drift-api:test1 2>&1 | tail -20
```

```bash
 gcloud run deploy drift-detection-api \
   --image=europe-north1-docker.pkg.dev/psychic-iridium-484208-c3/ml-images/drift-api:test1 \
   --region=europe-north1 \
   --platform=managed \
   --allow-unauthenticated \
   --memory=2Gi \
   --cpu=2 \
   --timeout=600 \
   --max-instances=5 \
   --project=psychic-iridium-484208-c3 \
   --port=8080
```

docker push europe-north1-docker.pkg.dev/psychic-iridium-484208-c3/ml-images/drift-api:^Cst 2>&1 | tail -20