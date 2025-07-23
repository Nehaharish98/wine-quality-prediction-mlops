FROM python:3.10-slim

# Install dependencies
RUN pip install mlflow scikit-learn pandas xgboost gunicorn

# Expose MLflow model server port
Expose 

# Set environment variables
ENV MLFLOW_TRACKING_URI=http://mlflow-tracking.cloud:5000

CMD ["mlflow", "models", "serve", "-m", "models:/RandomForest-red@production", "--host", "0.0.0.0", "--no-conda"]