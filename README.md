# MLOps Group04 2026 Project

Students:
- Julian S194077
- Xiaopeng S194408
- Sharan S242656
- David S250806

## End‑to‑end cloud setup for this branch

This section explains how to go from a fresh clone to running training on Vertex AI using DVC‑managed data and Artifact Registry.

### Prerequisites

- **Local tools**
  - `git`
  - `python >= 3.12`
  - [`uv`](https://github.com/astral-sh/uv) installed on your machine
  - [`gcloud` CLI](https://cloud.google.com/sdk/docs/install) installed and initialized
- **GCP resources**
  - A Google Cloud project (project ID referenced as `PROJECT_ID` below)
  - Billing enabled on the project
  - APIs enabled:
    - Vertex AI API
    - Cloud Build API
    - Artifact Registry API
    - Cloud Storage API

You can set the project once with:

```bash
gcloud config set project PROJECT_ID
```

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/<your-org-or-user>/mlops-group-04.git
cd mlops-group-04

# Install dependencies via uv (creates/uses .venv)
uv sync
```

### 2. Configure DVC and data remote

This branch uses **DVC with a GCS remote**. The current `.dvc/config` points to:

- `gs://psychic-iridium-484208-c3-mlops-data/dvc`

If you want to use **your own** bucket, create one and update DVC:

```bash
# Create a bucket for DVC data (adjust name/region)
gsutil mb -l europe-north1 gs://PROJECT_ID-mlops-data

# Point DVC remote to your bucket
uv run dvc remote modify storage url gs://PROJECT_ID-mlops-data/dvc
```

#### 2.1 Initial data upload (only needed once per dataset)

If the remote is empty and you have the raw `.npy` data locally:

```bash
# Place data under data/time_series/AF, data/time_series/Noise, data/time_series/NSR, data/time_series/Other

# Track the directory with DVC
uv run dvc add data/time_series

# Commit the .dvc file to git
git add data/time_series.dvc
git commit -m "Track time_series data with DVC"

# Push data to the remote
uv run dvc push data/time_series.dvc
```

After this, other users (and Vertex AI) will be able to `dvc pull` the data.

#### 2.2 Pulling data (most users)

If the data is already in the remote:

```bash
uv run dvc pull
```

This will populate `data/time_series/...` locally.

### 3. Build and push the training image to Artifact Registry

This branch uses **Cloud Build** and a training image in Artifact Registry.

- **Artifact Registry repo**: `ml-images`
- **Image URI** (as used in `config.yaml` and `cloudbuild.yaml`):
  - `europe-north1-docker.pkg.dev/PROJECT_ID/ml-images/train-image:latest`

If you do not already have the `ml-images` repository, create it once:

```bash
gcloud artifacts repositories create ml-images \
  --repository-format=docker \
  --location=europe-north1 \
  --description="Training images for ECG project"
```

Then build and push the image (from the repo root):

```bash
# Ensure gcloud is set to the correct project
gcloud config set project PROJECT_ID

# Build and push using Cloud Build
gcloud builds submit --config cloudbuild.yaml .
```

This will:
- Build the training image using `dockerfiles/train.dockerfile`
- Push it to `europe-north1-docker.pkg.dev/PROJECT_ID/ml-images/train-image:latest`

If you change the project ID or region, update:
- `cloudbuild.yaml` `image` / `-t` / `push` URIs
- `config.yaml` `containerSpec.imageUri`

### 4. Run training on Vertex AI

With the image pushed and DVC data remote populated, submit a custom training job:

```bash
gcloud ai custom-jobs create \
  --region=europe-north1 \
  --display-name=ecg-training-$(date +%Y%m%d-%H%M%S) \
  --config=config.yaml
```

To stream logs:

```bash
gcloud ai custom-jobs stream-logs projects/PROJECT_NUM/locations/europe-north1/customJobs/JOB_ID \
  --region=europe-north1
```

You can obtain `JOB_ID` from the output of the `custom-jobs create` command or via:

```bash
gcloud ai custom-jobs list --region=europe-north1
```

### 5. Cancel or clean up jobs

- **List running or pending jobs**:

```bash
gcloud ai custom-jobs list \
  --region=europe-north1 \
  --filter="state:JOB_STATE_RUNNING OR state:JOB_STATE_PENDING_QUEUE OR state:JOB_STATE_PENDING"
```

- **Cancel all running/pending jobs**:

```bash
gcloud ai custom-jobs list \
  --region=europe-north1 \
  --filter="state:JOB_STATE_RUNNING OR state:JOB_STATE_PENDING_QUEUE OR state:JOB_STATE_PENDING" \
  --format="value(name)" | \
  xargs -I {} gcloud ai custom-jobs cancel {} --region=europe-north1 2>/dev/null || true
```

### 6. Quick reference (happy path)

- **One-time (per project)**:
  - **Create GCS bucket** for DVC data and update `.dvc/config` (or use existing)
  - **Create Artifact Registry** repo `ml-images`
  - **Enable** Vertex AI, Cloud Build, Artifact Registry, and Storage APIs
- **Per dataset**:
  - Put data under `data/time_series/...`
  - `uv run dvc add data/time_series`
  - `uv run dvc push data/time_series.dvc`
- **Per code change**:
  - `gcloud builds submit --config cloudbuild.yaml .`
  - `gcloud ai custom-jobs create --region=europe-north1 --display-name=... --config=config.yaml`
  - Use `gcloud ai custom-jobs stream-logs ...` to monitor


## Aim

This project aims to classify 2D-transformed and preprocessed electrocardiograms (ECG) signals into four different categories, including normal sinus rhythm (NSR), atrial fibrillation (AF), other, and noisy ECG. The preproccsing involves bandpass-filtering between 0.5 and 50 Hz. SUbsequently, the 2D-transforms are generated using the so-called continuous wavelet transformation (CWT), so the ECG is represented in the time-frequency domain. Users should be able to upload the 2D-transformed ECG data and receive a classification result indicating the type of signal. This project will utilize the CACHET-CADB (Copenhagen Center for Health Technology - Contextualized Arrhythmia Database) provided by DTU Health. We are using a subset of the dataset that contains only the ECG recordings. It contains 1602 ten-second long ECG samples of AF, NSR, noise, and other rhythm classes, which are manually annotated by two cardiologists. The ECG is sampled at 1024 Hz and a 12-bit resolution. The dataset can be found in Kumar, Devender, et al. "CACHET-CADB: A contextualized ambulatory electrocardiography arrhythmia dataset." Frontiers in Cardiovascular Medicine 9 (2022): 893090.

The dataset is split into a training, validation, and test set in a stratified manner, so the data distribution is preserved in each set. The goal is to develope a consistent, reproducible, and efficient framework, so it can be used by other researchers within the healthcare or machine learning community.


## Model

The chosen architecture is EfficientNet-B7, known for its efficiency and performance in image classification tasks. It is a convolutional neural network (CNN) from 2019. A pretrained model was downloaded from Pytorch and fine-tuned. See the original publication in Tan, Mingxing, and Quoc Le. "Efficientnet: Rethinking model scaling for convolutional neural networks." International conference on machine learning. PMLR, 2019. We are using PyTorch Lightning for model training and evaluation easier.


## Frameworks and Libraries

The project utilizes the following frameworks and libraries:
- Hydra for configuration management
- PyTorch Lightning for model training and evaluation
- Docker for containerization
- UV for python package management



## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```
