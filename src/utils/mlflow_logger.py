import mlflow, os, numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models import infer_signature
import matplotlib.pyplot as plt, seaborn as sns, logging, requests, time

log = logging.getLogger(__name__)

def _server_up(uri):
    for _ in range(10):
        try:
            if requests.get(f"{uri}/health", timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            time.sleep(1)
    return False

def setup(cfg):
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    if not _server_up(cfg["tracking_uri"]):
        log.error("MLflow server unreachable at %s", cfg["tracking_uri"])
        return False
    mlflow.set_experiment(cfg["experiment_name"])
    return True

def log_run(model, name, X_test, y_test, params, features, cfg):
    if not setup(cfg):
        return
    with mlflow.start_run(run_name=name):
        y_pred = model.predict(X_test)
        mlflow.log_params(params)
        mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, y_pred)))
        mlflow.log_metric("r2",   r2_score(y_test, y_pred))

        # scatter plot
        plt.figure(figsize=(5,5))
        plt.scatter(y_test, y_pred, alpha=.6)
        plt.plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], "r--")
        plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Predictions")
        path = "mlflow_artifacts/scatter.png"; os.makedirs("mlflow_artifacts", exist_ok=True)
        plt.savefig(path, dpi=120); plt.close()
        mlflow.log_artifact(path, "plots")

        mlflow.sklearn.log_model(
            model, artifact_path="model",
            input_example=X_test.iloc[:3], signature=infer_signature(X_test, y_pred)
        )