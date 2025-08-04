# 🍷 Wine Quality Prediction MLOps

A complete MLOps pipeline for predicting wine quality using physicochemical features. This project demonstrates best practices in machine learning operations including experiment tracking, model deployment, monitoring, and automated workflows.

## 🎯 Business Problem
Predict physicochemical factors that drive higher-scoring Vinho Verde wines so producers can optimize fermentation and sulphite usage for quality improvement.

## 🤖 Machine Learning Problem
- **Type**: Multi-class classification (quality scores 3-9)
- **Input**: 11 numeric physicochemical features per wine batch
- **Output**: Predicted quality score (integer 3-9)
- **Success Metric**: Weighted F1-score ≥ 0.68 on hold-out test set

## 📊 Dataset Features
1. Fixed acidity        2. Volatile acidity     3. Citric acid
4. Residual sugar      5. Chlorides            6. Free sulfur dioxide
7. Total sulfur dioxide 8. Density             9. pH
10. Sulphates          11. Alcohol


## 📋 Table of Contents

- Problem Description  
- Project Overview  
- Technology Stack  
- Architecture & Flow  
- Installation & Setup  
- Usage  
- Testing  
- Deployment  
- Best Practices  
- Troubleshooting  
- Service Directory  

## 🎯 Problem Description

**Wine quality** plays a crucial role in ensuring economic value and consumer satisfaction in the wine industry. Predicting wine quality based on physicochemical properties helps producers optimize fermentation processes and make informed decisions about production quality control.

Challenges include:

- Accurately modeling multi-class classification of wine quality scores (3-9).  
- Automating the full machine learning lifecycle from raw data ingestion to deployment and continuous monitoring.  
- Providing actionable insights to stakeholders for improving wine quality.

**Our Solution** is a fully automated MLOps pipeline that predicts wine quality using physicochemical features, integrated with modern tools for reproducibility, deployment, monitoring, and workflow orchestration.

## 🚀 Project Overview

This project is designed as a **production-ready MLOps pipeline** including:  

- Automated data ingestion, preprocessing, and feature engineering using DVC and custom scripts.  
- Model training with hyperparameter tuning and experiment tracking using MLflow.  
- FastAPI for model serving and a user-friendly Streamlit UI for interactive inference.  
- Prefect for workflow orchestration and scheduling of training and monitoring.  
- Comprehensive monitoring setup with Evidently for data drift detection and Grafana for visualization dashboards.  
- Infrastructure as Code (IaC) ready for cloud or local deployment with Docker and Terraform.  

## 🛠️ Technology Stack

| Component            | Technology              | Purpose                             |
|----------------------|-------------------------|-----------------------------------|
| Programming          | Python 3.11+            | Core development                  |
| Machine Learning     | Scikit-learn, XGBoost   | Model training and prediction    |
| Hyperparameter Tuning| Optuna                  | Optimization of model parameters |
| Experiment Tracking  | MLflow                  | Logging, model registry          |
| Workflow Orchestration| Prefect                 | Pipeline automation and scheduling|
| API & UI             | FastAPI, Streamlit      | Serving models and user inference|
| Monitoring           | Evidently, Prometheus, Grafana, PostgreSQL | Data & model performance monitoring |
| Containerization     | Docker, Docker Compose  | Reproducible environment         |
| Infrastructure as Code| Terraform               | Cloud infrastructure provisioning  |

## 🏗️ Architecture & Flow

```mermaid
graph TB

subgraph "Data Layer"
A[Raw Wine Data] --> B[Data Processing and Feature Engineering]
end

subgraph "ML Pipeline"
B --> C[Model Training + Optuna]
C --> D[Model Evaluation]
D --> E[Model Registry (MLflow)]
end

subgraph "Orchestration"
F[Prefect Server] --> G[Pipeline Scheduling]
G --> H[Training & Monitoring Deployments]
end

subgraph "Deployment"
E --> I[Model Deployment]
I --> J[FastAPI Service]
J --> K[Streamlit UI for Inference]
end

subgraph "Monitoring"
J --> L[Evidently Data Drift Detection]
L --> M[Prometheus & Grafana Dashboards]
M --> N[Alerts and Notifications]
end

subgraph "Infrastructure"
O[Docker Containers] --> P[Local & Cloud Deployment]
end

F --> G
G --> L
```

## 📦 Installation & Setup

### Prerequisites

- Python 3.11+  
- Docker & Docker Compose  
- Git  
- Prefect CLI  
- MLflow server  
- Node.js (optional, for Grafana customizations)  

### Quick Start

```bash
git clone https://github.com/your-username/wine-quality-prediction-mlops
cd wine-quality-prediction-mlops

# Install python dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

## 🚦 Usage

### Run complete pipeline via DVC

```bash
dvc repro
```

### Start Monitoring Stack

```bash
cd monitoring
docker-compose up -d
```

### Start MLflow Server

```bash
mlflow server --host 0.0.0.0 --port 5000
```

### Start FastAPI Service

```bash
uvicorn src.wine_quality.api.app:app --host 0.0.0.0 --port 8000
```

### Start Prefect Server and Deploy Flows

```bash
prefect server start --host 0.0.0.0 --port 4200

cd prefect_flows
python deployment.py

prefect agent start -q default
```

## 🧪 Testing

Run linting, formatting, and tests:

```bash
make lint
pytest -v
```

## 🚀 Deployment

### Docker Compose

```bash
docker-compose up --build
```


4. Deploy Docker images to cloud infrastructure.

## ✅ Best Practices

- Automated testing with `pytest`.  
- Code quality enforced via `black`, `flake8`, `isort`, and pre-commit hooks.  
- Model versioning and registry using MLflow.  
- Orchestration and scheduling with Prefect.  
- Alerting and monitoring using Prometheus, Grafana, and Evidently.  
- Reproducible pipelines leveraging DVC and containerization.  

## 🩺 Troubleshooting

### Common Issues

- **Port Conflicts**: Check ports 3000, 4200, 5000, 8000, 8501.  
- **Docker Cleanup**: Use `docker system prune -a` and restart containers.  
- **Prefect/Mlflow Connectivity**: Use `curl` to check health of services.  

### Check Running Services

```bash
docker ps -a
docker logs <container_name>
```

## 🌐 Service Directory

| Service       | URL                     | Credentials  |
|---------------|-------------------------|--------------|
| Streamlit UI  | http://localhost:8501   |              |
| FastAPI Docs  | http://localhost:8000/docs |            |
| MLflow UI     | http://localhost:5000   |              |
| Prefect UI    | http://localhost:4200   |              |
| Grafana       | http://localhost:3000   | admin/admin  |
| Prometheus    | http://localhost:9090   |              |

## ✨ Final Thoughts

This professional-grade wine quality MLOps project empowers winemakers and data scientists alike with an automated, monitored, and reproducible ML lifecycle - from raw data to real-time insights.



