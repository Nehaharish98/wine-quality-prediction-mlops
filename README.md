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

## 🏗️ Architecture
── wine_quality-prediction-mlops/ # Core package
│ ├── data/ # Data processing
│ ├── models/ # ML models
│ ├── api/ # FastAPI application
│ └── tracking/ # MLflow integration
├── tests/ # Test suites
├── scripts/ # Pipeline scripts
├── notebooks/ # Jupyter notebooks
├── terraform/ # Infrastructure as Code
├── monitoring/ # Prometheus & Grafana
├── prefect_flows/ # Workflow orchestration
└── deployment/ # Deployment configs
