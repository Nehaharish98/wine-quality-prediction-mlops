"""Prefect training pipeline."""
from prefect import flow, task
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
import subprocess
import pandas as pd
from pathlib import Path
import yaml

@task
def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from params.yaml."""
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

@task
def download_data(output_path: str) -> str:
    """Download wine quality dataset."""
    result = subprocess.run([
        "python", "scripts/download_data.py", 
        "--out", output_path
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Data download failed: {result.stderr}")
    
    print(result.stdout)
    return output_path

@task
def preprocess_data(input_path: str, output_path: str) -> str:
    """Preprocess wine quality data."""
    result = subprocess.run([
        "python", "scripts/preprocess.py",
        "--in", input_path,
        "--out", output_path
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Data preprocessing failed: {result.stderr}")
    
    print(result.stdout)
    return output_path

@task
def train_model(data_path: str, model_path: str) -> dict:
    """Train wine quality model."""
    result = subprocess.run([
        "python", "scripts/train_model.py",
        "--data", data_path,
        "--model", model_path
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Model training failed: {result.stderr}")
    
    print(result.stdout)
    
    # Read metrics
    metrics_path = Path(model_path).parent / "metrics.json"
    if metrics_path.exists():
        import json
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    return {}

@task
def validate_model(metrics: dict, min_f1_score: float = 0.68) -> bool:
    """Validate model performance."""
    f1_score = metrics.get('f1_weighted', 0.0)
    
    if f1_score >= min_f1_score:
        print(f"Model validation passed: F1 score {f1_score:.3f} >= {min_f1_score}")
        return True
    else:
        print(f"Model validation failed: F1 score {f1_score:.3f} < {min_f1_score}")
        return False

@task
def deploy_model(model_path: str) -> str:
    """Deploy model to production."""
    # This would trigger your deployment script
    result = subprocess.run([
        "bash", "scripts/deploy_model.sh"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Deployment warning: {result.stderr}")
    
    print(result.stdout)
    return "Deployment completed"

@flow(name="wine-quality-training-pipeline")
def training_pipeline():
    """Complete training pipeline for wine quality model."""
    
    # Load parameters
    params = load_params()
    
    # Extract paths
    raw_data_path = params['paths']['data']['raw']
    processed_data_path = params['paths']['data']['processed']
    model_path = params['paths']['models']['model']
    
    # Execute pipeline steps
    print("Starting wine quality training pipeline...")
    
    # Step 1: Download data
    downloaded_data = download_data(raw_data_path)
    
    # Step 2: Preprocess data
    processed_data = preprocess_data(downloaded_data, processed_data_path)
    
    # Step 3: Train model
    metrics = train_model(processed_data, model_path)
    
    # Step 4: Validate model
    is_valid = validate_model(metrics)
    
    # Step 5: Deploy if valid
    if is_valid:
        deployment_result = deploy_model(model_path)
        print(deployment_result)
    else:
        print("Model validation failed. Skipping deployment.")
    
    return {
        "metrics": metrics,
        "validation_passed": is_valid,
        "deployed": is_valid
    }

if __name__ == "__main__":
    # Run the flow
    result = training_pipeline()
    print(f"Pipeline completed: {result}")