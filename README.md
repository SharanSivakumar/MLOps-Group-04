# MLOps Group 04 Project

**Students:**
- Julian S194077
- Xiaopeng S194408
- Sharan S242656
- David S250806

## Aim

This project aims to classify 2D-transformed and preprocessed electrocardiogram (ECG) signals into four different categories, including normal sinus rhythm (NSR), atrial fibrillation (AF), other, and noisy ECG. The preprocessing involves bandpass-filtering between 0.5 and 50 Hz. Subsequently, the 2D-transforms are generated using the so-called continuous wavelet transformation (CWT), so the ECG is represented in the time-frequency domain. Users should be able to upload the 2D-transformed ECG data and receive a classification result indicating the type of signal. This project will utilize the CACHET-CADB (Copenhagen Center for Health Technology - Contextualized Arrhythmia Database) provided by DTU Health. We are using a subset of the dataset that contains only the ECG recordings. It contains 1602 ten-second long ECG samples of AF, NSR, noise, and other rhythm classes, which are manually annotated by two cardiologists. The ECG is sampled at 1024 Hz and a 12-bit resolution. The dataset can be found in Kumar, Devender, et al. "CACHET-CADB: A contextualized ambulatory electrocardiography arrhythmia dataset." Frontiers in Cardiovascular Medicine 9 (2022): 893090.

The dataset is split into a training, validation, and test set in a stratified manner, so the data distribution is preserved in each set. The goal is to develop a consistent, reproducible, and efficient framework, so it can be used by other researchers within the healthcare or machine learning community.


![](./reports/figures/sys-overview.png)

## Model

The chosen architecture is EfficientNet-B4, known for its efficiency and performance in image classification tasks. It is a convolutional neural network (CNN) from 2019. A pretrained model was downloaded from PyTorch and fine-tuned. See the original publication in Tan, Mingxing, and Quoc Le. "Efficientnet: Rethinking model scaling for convolutional neural networks." International conference on machine learning. PMLR, 2019. We are using PyTorch Lightning for model training and evaluation.

The model architecture has been modified to accept single-channel grayscale images (1 channel) instead of the standard RGB input (3 channels), as the CWT-transformed ECG signals are represented as 2D grayscale spectrograms. The first convolutional layer has been adapted accordingly, and the final classification layer outputs predictions for three classes: AF, NSR, and Noise.

## Frameworks and Libraries

The project utilizes the following frameworks and libraries:

- Hydra for configuration management
- PyTorch Lightning for model training and evaluation
- Docker for containerization
- UV for python package management
- FastAPI for REST API development
- Streamlit for frontend interface
- DVC (Data Version Control) for data versioning and management
- Google Cloud Platform services (Cloud Run, Cloud Build, Vertex AI, Cloud Storage, Artifact Registry)
- Evidently AI for drift detection
- Ruff for code linting and formatting
- Pytest for testing
- Pre-commit hooks for code quality assurance


## Project Structure

```
├── .github/                      # CI/CD workflows and GitHub Actions
│   ├── workflows/               # Automated testing, linting, and deployment workflows
│   ├── agents/                  # Agent configuration files
│   └── prompts/                 # Prompt templates for development
├── .dvc/                        # DVC configuration and cache
├── checkpoints/                 # Model checkpoint storage
├── configs/                     # Hydra configuration files
├── data/                        # Data directory (managed by DVC)
│   └── time_series/            # Raw time-series ECG data
├── dockerfiles/                 # Docker configuration files
│   ├── api.dockerfile          # API service container definition
│   ├── api_entrypoint.sh       # API startup script
│   ├── train.dockerfile        # Training container definition
│   └── entrypoint.sh           # Training startup script
├── drift/                       # Drift detection service
│   ├── drift_detection.py      # Core drift detection logic
│   ├── drift_api.py            # Drift detection API endpoints
│   ├── Dockerfile              # Drift service container definition
│   └── entrypoint.sh           # Drift service startup script
├── docs/                        # Documentation sources (MkDocs)
│   └── source/                 # Documentation source files
├── frontend/                    # Streamlit frontend application
│   ├── streamlit_app.py        # Main frontend application
│   ├── frontend.dockerfile     # Frontend container definition
│   └── requirements_frontend.txt # Frontend dependencies
├── lightning_logs/              # PyTorch Lightning training logs
├── models/                      # Local model storage directory
├── notebooks/                   # Jupyter notebooks for exploration
├── profiler/                    # Performance profiling logs
├── reports/                     # Generated reports and figures
│   └── figures/                # Visualization outputs
├── scripts/                     # Utility scripts
│   └── setup_drift_scheduler.py # Drift detection scheduler setup
├── src/                         # Core source code
│   ├── api.py                  # FastAPI inference service
│   ├── data.py                 # Data loading and preprocessing
│   ├── evaluate.py             # Model evaluation scripts
│   ├── model.py                # EfficientNet model definition
│   ├── train.py                # Training loop and logic
│   ├── visualize.py            # Visualization utilities
│   └── random_input.py         # Random input generation for testing
├── tests/                       # Test suite
│   ├── test_api.py             # API endpoint tests
│   ├── test_data.py            # Data pipeline tests
│   ├── test_drift_detection.py # Drift detection tests
│   ├── test_model.py           # Model architecture tests
│   ├── test_training.py        # Training pipeline tests
│   ├── test_load.py            # Load testing utilities
│   └── performancetests/       # Performance testing with Locust
│       └── locustfile.py       # Load testing configuration
├── .dockerignore                # Docker build exclusions
├── .dvcignore                   # DVC tracking exclusions
├── .gcloudignore                # Google Cloud deployment exclusions
├── .gitignore                   # Git exclusions
├── .pre-commit-config.yaml      # Pre-commit hook configuration
├── .python-version             # Python version specification
├── AGENTS.md                    # Development agent guidelines
├── cloudbuild.yaml              # Google Cloud Build configuration
├── config.yaml                  # Vertex AI training configuration
├── pyproject.toml               # Python project configuration and dependencies
├── requirements.txt             # Python package dependencies
├── uv.lock                      # UV package lock file
└── README.md                    # Project documentation
```

## Core Components

### Source Code Modules

**src/api.py**: Implements the FastAPI inference service. The service handles model loading from Google Cloud Storage, processes incoming ECG data, and returns classification predictions. The API includes health check endpoints and error handling for various failure scenarios.

**src/model.py**: Defines the ECGClassifier class, which extends PyTorch Lightning's LightningModule. The class implements the EfficientNet-B4 architecture with modifications for single-channel input and custom classification head. It includes training, validation, and testing logic with comprehensive metrics tracking.

**src/data.py**: Contains data loading and preprocessing utilities. This module handles dataset loading, data transformations, and preparation of data loaders for training and evaluation.

**src/train.py**: Implements the training pipeline using PyTorch Lightning. It configures trainers, callbacks, and training loops, integrating with Hydra for configuration management.

**src/evaluate.py**: Provides evaluation scripts for model performance assessment on test datasets, including metric computation and result reporting.

**src/visualize.py**: Contains visualization utilities for model outputs, training curves, and data exploration.

### Drift Detection Module

The drift detection system (located in `drift/`) monitors production data for distribution shifts:

**drift/drift_detection.py**: Implements the DriftDetector and DriftLogger classes. The DriftLogger collects prediction data and features from production inference requests, buffering them locally before uploading to Google Cloud Storage. The DriftDetector loads reference training data and production logs, then uses Evidently AI to perform statistical drift analysis.

**drift/drift_api.py**: Provides REST API endpoints for triggering drift detection analyses and retrieving drift reports. The API can generate HTML reports showing drift metrics and feature-level comparisons.

## Setup and Installation

### Prerequisites

- Python 3.12 or higher
- UV package manager
- Google Cloud SDK (gcloud CLI)
- Docker (for containerization)
- Git

### Google Cloud Platform Requirements

A Google Cloud project with the following APIs enabled:
- Vertex AI API
- Cloud Build API
- Artifact Registry API
- Cloud Storage API
- Cloud Run API

The project requires appropriate IAM permissions for service accounts to access Cloud Storage, deploy to Cloud Run, and execute Vertex AI jobs.

### Local Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd mlops-group-04
   ```

2. **Install Dependencies**:
   ```bash
   uv sync
   ```

3. **Configure DVC Remote** (if using custom bucket):
   ```bash
   uv run dvc remote modify storage url gs://PROJECT_ID-mlops-data/dvc
   ```

4. **Pull Data**:
   ```bash
   uv run dvc pull
   ```

5. **Authenticate with Google Cloud**:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

## Usage

### Training

Model training can be executed either locally or on Google Cloud Vertex AI:

**Local Training**:
```bash
uv run python src/train.py
```

**Cloud Training via Vertex AI**:
```bash
# Build and push training image
gcloud builds submit --config cloudbuild.yaml .

# Submit training job
gcloud ai custom-jobs create \
  --region=europe-north1 \
  --display-name=ecg-training-$(date +%Y%m%d-%H%M%S) \
  --config=config.yaml
```

### API Deployment

The API service is automatically built and deployed through Google Cloud Build when changes are pushed to the repository. Manual deployment can be performed using:

```bash
gcloud builds submit --config cloudbuild.yaml
```

The Cloud Build configuration builds Docker images for the API, frontend, and drift detection services, pushes them to Artifact Registry, and deploys them to Cloud Run with appropriate resource allocations.

### Running Services Locally

**API Service**:
```bash
uv run uvicorn src.api:app --host 0.0.0.0 --port 8080
```

**Frontend**:
```bash
uv run streamlit run frontend/streamlit_app.py
```

**Drift Detection API**:
```bash
cd drift
uv run uvicorn drift_api:app --host 0.0.0.0 --port 8080
```

### API Usage

The inference API accepts POST requests to `/predict` with a `.npy` file containing 2D-transformed ECG data. The expected input shape is (1, 224, 224) or (224, 224), representing a single grayscale spectrogram image.

**Example Request**:
```bash
curl -X POST "https://ecg-api-<project-id>.europe-north1.run.app/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@ecg_data.npy"
```

**Example Response**:
```json
{
  "predicted_class_id": 0,
  "predicted_label": "AF",
  "probabilities": {
    "AF": 0.85,
    "Noise": 0.10,
    "NSR": 0.05
  }
}
```

## Monitoring and Maintenance

### Drift Detection

The drift detection service continuously monitors production data. To trigger a drift analysis:

```bash
curl -X POST "https://drift-detection-api-<project-id>.europe-north1.run.app/detect-drift" \
  -H "Content-Type: application/json" \
  -d '{
    "bucket_name": "your-bucket",
    "logs_prefix": "drift_logs"
  }'
```

### Logging and Debugging

All services integrate with Google Cloud Logging. Logs can be accessed through:
- Google Cloud Console Logs Viewer
- gcloud CLI: `gcloud logging read`
- Service-specific log URLs provided in deployment outputs

## Testing

Run the test suite locally:

```bash
uv run pytest tests/
```

Run specific test categories:
```bash
uv run pytest tests/test_api.py          # API tests
uv run pytest tests/test_model.py        # Model tests
uv run pytest tests/test_drift_detection.py  # Drift detection tests
```

Run load tests:
```bash
cd tests/performancetests
locust -f locustfile.py
```

## Configuration

The project uses Hydra for configuration management. Configuration files are located in `configs/` and can be customized for different experiments. Key configuration parameters include:

- Model hyperparameters (learning rate, batch size, epochs)
- Data paths and preprocessing settings
- Training device configuration (CPU/GPU)
- Evaluation metrics and thresholds



## References

- Kumar, Devender, et al. "CACHET-CADB: A contextualized ambulatory electrocardiography arrhythmia dataset." Frontiers in Cardiovascular Medicine 9 (2022): 893090.
- Tan, Mingxing, and Quoc Le. "Efficientnet: Rethinking model scaling for convolutional neural networks." International conference on machine learning. PMLR, 2019.
