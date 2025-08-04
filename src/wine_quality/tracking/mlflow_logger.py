"""MLflow logging utilities."""
import mlflow
import mlflow.sklearn
from typing import Dict, Any
from pathlib import Path
import json

class MLflowLogger:
    def __init__(self, tracking_uri: str = None, experiment_name: str = "wine-quality"):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)
        self.experiment_id = experiment_id

    def log_training_run(
        self,
        model,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts_dir: str = None,
        registered_model_name: str = None
    ) -> str:
        """Log a complete training run."""
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            if registered_model_name:
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=registered_model_name
                )
            else:
                mlflow.sklearn.log_model(model, "model")

            # Log artifacts if directory provided
            if artifacts_dir and Path(artifacts_dir).exists():
                mlflow.log_artifacts(artifacts_dir)

            return run.info.run_id

    def register_model(self, model_uri: str, model_name: str) -> None:
        """Register a model in MLflow registry."""
        result = mlflow.register_model(model_uri, model_name)
        print(f"Model {model_name} version {result.version} registered")
        return result

    def get_latest_model_version(self, model_name: str) -> str:
        """Get the latest version of a registered model."""
        client = mlflow.MlflowClient()
        latest_versions = client.get_latest_versions(
            model_name, stages=["Production", "Staging", "None"]
        )
        if latest_versions:
            return latest_versions[0].version
        return None

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str
    ) -> None:
        """Transition model to a specific stage."""
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        print(f"Model {model_name} version {version} transitioned to {stage}")
