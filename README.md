# MLOps Group04 2026 Project

Students:
- Julian S194077
- Xiaopeng S194408
- Sharan S242656
- David S250806

## Guide to run instance
Update the region based on your config

gcloud builds submit --config cloudbuild.yaml .


```
gcloud ai custom-jobs create \
  --region=europe-north1 \
  --display-name=ecg-training-$(date +%Y%m%d-%H%M%S) \
  --config=config.yaml
```
Note: Adjust cloudbuild and config with the artifact location and project details, setting the project id before executing should resolve isses

For now also upload the data to DVC before running the build and execution of the project


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

