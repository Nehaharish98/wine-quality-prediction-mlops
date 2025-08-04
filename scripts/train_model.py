"""Train wine quality model with DVC integration - CORRECTED VERSION."""
import pandas as pd
import argparse
import os
import yaml
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append('src')

from wine_quality.data.loader import split_features_target
from wine_quality.tracking.mlflow_logger import MLflowLogger

# Import trainer from models directory (create if needed)
try:
    from models.trainer import WineQualityTrainer
except ImportError:
    # Fallback implementation if models/trainer.py doesn't exist
    print("Warning: models/trainer.py not found. Using fallback implementation.")
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

    class WineQualityTrainer:
        def __init__(self, **params):
            self.params = params
            self.model = RandomForestClassifier(**params)
            self.is_trained = False

        def train(self, X, y):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            self.model.fit(X_train, y_train)
            self.is_trained = True

            y_pred = self.model.predict(X_test)
            metrics = {
                'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'accuracy': accuracy_score(y_test, y_pred),
                'precision_macro': precision_score(y_test, y_pred, average='macro'),
                'recall_macro': recall_score(y_test, y_pred, average='macro')
            }
            return metrics

        def save_model(self, filepath):
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, filepath)

def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from params.yaml."""
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(
    data_path: str = None,
    model_output: str = None,
    params_path: str = "params.yaml",
    register_as: str = None
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

    # Handle both parquet and CSV files
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

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
    registered_model_name = register_as or mlflow_config.get('registered_model_name')
    run_id = logger.log_training_run(
        model=trainer.model,
        params=train_params,
        metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        artifacts_dir=str(Path(model_output).parent),
        registered_model_name=registered_model_name
    )

    print(f"Model saved to {model_output}")
    print(f"Metrics saved to {metrics_path}")
    print(f"F1 Score: {metrics['f1_weighted']:.3f}")
    print(f"MLflow Run ID: {run_id}")
    print(f"Target achieved (â‰¥0.68): {'âœ…' if metrics['f1_weighted'] >= 0.68 else 'âŒ'}")

def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Training data path")
    parser.add_argument("--model", help="Model output path")
    parser.add_argument("--params", default="params.yaml", help="Parameters file")
    parser.add_argument("--register_as", help="Register model as this name in MLflow")

    args = parser.parse_args()

    train_model(
        data_path=args.data,
        model_output=args.model,
        params_path=args.params,
        register_as=args.register_as
    )

if __name__ == "__main__":
    main()
