# Use an official Python image
FROM python:3.10-slim

WORKDIR /app

RUN pip install fastapi uvicorn gunicorn mlflow scikit-learn pandas xgboost

EXPOSE 8000

COPY deployment/api.py .

ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5001

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "api:app", "--bind", "0.0.0.0:8000"]