"""Train wine quality model with MLflow tracking."""
import pandas as pd
import argparse
import os
from pathlib import Path
from models.trainer import WineQualityTrainer
from data.loader import split_features_target
from tracking.mlflow_logger import MLflowLogger

def train_model(
    data_path: str, 
    model_output: str,
    tracking_uri: str = None,
    experiment_name: str = "wine-quality",
    register_as: str = None
):
    """Train and save model with MLflow tracking."""
    # Initialize MLflow logger
    logger = MLflowLogger(tracking_uri, experiment_name)
    
    # Load data
    df = pd.read_parquet(data_path)
    X, y = split_features_target(df)
    
    # Model parameters
    model_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    }
    
    # Train model
    trainer = WineQualityTrainer(**model_params)
    metrics = trainer.train(X, y)
    
    # Save model locally
    Path(model_output).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(model_output)
    
    # Save metrics locally
    metrics_path = Path(model_output).parent / "metrics.json"
    import json
    with open(metrics_path, 'w') as f:
        json.dump({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, f, indent=2)
    
    # Log to MLflow
    run_id = logger.log_training_run(
        model=trainer.model,
        params=model_params,
        metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        artifacts_dir=str(Path(model_output).parent),
        registered_model_name=register_as
    )
    
    print(f"Model saved to {model_output}")
    print(f"F1 Score: {metrics['f1_weighted']:.3f}")
    print(f"MLflow Run ID: {run_id}")
    
    if register_as:
        print(f"Model registered as: {register_as}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="models/model.pkl")
    parser.add_argument("--tracking_uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment_name", default="wine-quality")
    parser.add_argument("--register_as", help="Register model with this name")
    
    args = parser.parse_args()
    train_model(
        data_path=args.data,
        model_output=args.model,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        register_as=args.register_as
    )