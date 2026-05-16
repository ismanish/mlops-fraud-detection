# MLOps Fraud Detection — Production Pipeline

End-to-end MLOps pipeline for credit card fraud detection, built as a learning project to demonstrate production-grade ML engineering practices.

## Architecture

```
┌─────────────────── GitHub ───────────────────────┐
│  CI (lint/test) → Train (pipeline) → Deploy (CD) │
└──────────┬────────────┬──────────────┬───────────┘
           │            │              │
           ▼            ▼              ▼
┌──────────────────── AWS ─────────────────────────┐
│                                                   │
│   S3 Bucket         ECR              Lambda       │
│   (data/models)     (Docker images)  (serving)    │
│                                          │        │
│                                    API Gateway    │
│                                    (HTTP API)     │
│                                                   │
│   CloudWatch                                      │
│   (metrics, dashboards, alarms)                   │
└───────────────────────────────────────────────────┘

Local / CI:
   DVC (data versioning) + MLflow (experiment tracking)
```

## Tech Stack

| Component | Tool |
|---|---|
| ML Framework | scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| Data Versioning | DVC + S3 |
| CI/CD | GitHub Actions (3 workflows) |
| Serving | FastAPI + AWS Lambda + API Gateway |
| Containerization | Docker + ECR |
| Monitoring | Custom drift detection + CloudWatch |
| Infrastructure | Terraform |
| Data Validation | Pandera |
| Code Quality | Ruff, Black, Mypy, pre-commit |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
bash scripts/run_pipeline.sh

# Or step by step:
python -m src.data.ingest
python -m src.data.validate
python -m src.data.preprocess
python -m src.models.train
python -m src.models.evaluate

# View MLflow UI
mlflow ui --port 5000

# Run tests
pytest tests/ -v

# Serve locally
uvicorn src.serving.app:app --port 8000
```

## Project Structure

```
├── .github/workflows/      # CI/CD pipelines
│   ├── ci.yml              # Lint + test on every push
│   ├── train.yml           # Train + evaluate + quality gates
│   └── deploy.yml          # Build Docker → ECR → Lambda
├── configs/                # Configuration files
├── data/                   # DVC-tracked data
├── guide/                  # Detailed MLOps learning guides
│   ├── 01_mlops_overview.md
│   ├── 02_data_versioning_dvc.md
│   ├── 03_experiment_tracking_mlflow.md
│   ├── 04_ci_cd_pipelines.md
│   ├── 05_containerization_docker.md
│   ├── 06_model_serving_aws.md
│   ├── 07_monitoring_drift_detection.md
│   ├── 08_infrastructure_as_code.md
│   └── 09_interview_questions.md
├── infrastructure/terraform/  # IaC for AWS resources
├── metrics/                # Model metrics (git-tracked)
├── models/                 # Trained model artifacts
├── scripts/                # Setup and deployment scripts
├── src/
│   ├── data/               # Ingestion, validation, preprocessing
│   ├── models/             # Training, evaluation, prediction
│   ├── monitoring/         # Drift detection, performance tracking
│   ├── serving/            # FastAPI app + Lambda handler
│   └── utils/              # Config, logging
├── tests/                  # Unit, integration, model quality tests
├── dvc.yaml                # DVC pipeline definition
├── params.yaml             # Hyperparameters and config
├── Dockerfile              # Lambda container image
└── Makefile                # Common commands
```

## MLOps Components Demonstrated

1. **Data Versioning** — DVC tracks data files in S3, ensuring reproducibility
2. **Experiment Tracking** — MLflow logs params, metrics, and model artifacts
3. **Automated Pipeline** — DVC DAG: ingest → validate → preprocess → train → evaluate
4. **CI/CD** — GitHub Actions: lint → test → train → quality gates → deploy
5. **Quality Gates** — Model must pass AUC-ROC, precision, recall thresholds before deployment
6. **Containerization** — Docker image pushed to ECR for Lambda deployment
7. **Model Serving** — FastAPI behind API Gateway with Lambda
8. **Monitoring** — Statistical drift detection (KS test, PSI) + CloudWatch metrics/alarms
9. **Infrastructure as Code** — Terraform manages all AWS resources
10. **Code Quality** — Ruff, Black, Mypy, pre-commit hooks, Gitleaks

## Learning Guide

The `guide/` folder contains 9 detailed chapters covering every MLOps concept in this project, with architecture diagrams, code walkthroughs, and 60+ interview questions with answers. Start with `guide/01_mlops_overview.md`.

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": -0.34,
    "V25": -0.07, "V26": -0.06, "V27": -0.03, "V28": -0.01,
    "Amount": 149.62
  }'
```

## AWS Deployment

```bash
# Set up AWS resources
bash scripts/setup_aws.sh

# Deploy infrastructure with Terraform
cd infrastructure/terraform && terraform init && terraform plan

# Build and push Docker image
bash scripts/deploy.sh push

# Update Lambda
bash scripts/deploy.sh deploy
```
