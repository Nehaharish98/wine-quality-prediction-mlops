"""Train wine quality model with DVC integration."""
import pandas as pd
import argparse
import os
import yaml
from pathlib import Path
from models.trainer import WineQualityTrainer
from data.loader import split_features_target
from tracking.mlflow_logger import MLflowLogger

def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from params.yaml."""
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(
    data_path: str = None,
    model_output: str = None, 
    params_path: str = "params.yaml"
):
    """Train and save model with DVC and MLflow integration."""
    
    # Load parameters
    params = load_params(params_path)
    
    # Use paths from params if not provided
    data_path = data_path or params['paths']['data']['processed']
    model_output = model_output or params['paths']['models']['model']
    
    # Initialize MLflow logger
    mlflow_config = params['mlflow']
    logger = MLflowLogger(
        tracking_uri=mlflow_config['tracking_uri'],
        experiment_name=mlflow_config['experiment_name']
    )
    
    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    X, y = split_features_target(df, params['base']['target_col'])
    
    # Get training parameters
    train_params = params['train']
    
    # Train model
    print("Training model...")
    trainer = WineQualityTrainer(**train_params)
    metrics = trainer.train(X, y)
    
    # Save model locally
    Path(model_output).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(model_output)
    
    # Save metrics for DVC
    metrics_path = params['paths']['models']['metrics']
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(metrics_path, 'w') as f:
        json.dump({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, f, indent=2)
    
    # Log to MLflow
    run_id = logger.log_training_run(
        model=trainer.model,
        params=train_params,
        metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        artifacts_dir=str(Path(model_output).parent),
        registered_model_name=mlflow_config['registered_model_name']
    )
    
    print(f"Model saved to {model_output}")
    print(f"Metrics saved to {metrics_path}")
    print(f"F1 Score: {metrics['f1_weighted']:.3f}")
    print(f"MLflow Run ID: {run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Training data path")
    parser.add_argument("--model", help="Model output path")
    parser.add_argument("--params", default="params.yaml", help="Parameters file")
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data,
        model_output=args.model,
        params_path=args.params
    )