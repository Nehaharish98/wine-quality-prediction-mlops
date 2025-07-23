import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from mlflow.models import infer_signature

def log_with_mlflow(
    model, model_name, dataset_name, params,
    X_test, y_test, feature_names, mlflow_config
):
    """
    Log params, metrics, confusion matrix, feature importance plots, and register model with MLflow.
    """
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])
    run_name = f"{dataset_name.capitalize()}Wine-{model_name}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(params)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        mlflow.log_metric("accuracy", report["accuracy"])
        for class_label in ["0", "1"]:
            if class_label in report:
                mlflow.log_metric(f"precision_{class_label}", report[class_label]["precision"])
                mlflow.log_metric(f"recall_{class_label}", report[class_label]["recall"])
                mlflow.log_metric(f"f1_{class_label}", report[class_label]["f1-score"])
        # Confusion matrix plot as artifact
        cm_path = _plot_and_save_confusion_matrix(y_test, y_pred, labels=[0,1], run_name=run_name)
        mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")
        # Feature importance as artifact (if available)
        if hasattr(model, "feature_importances_"):
            fi_path = _plot_and_save_feat_importance(model.feature_importances_, feature_names, run_name)
            mlflow.log_artifact(fi_path, artifact_path="feature_importance")
        # Log and register the model in MLflow Model Registry
        # Log and register the model in MLflow Model Registry
        # Log and register the model in MLflow Model Registry
        # Compute input_example for clear schema documentation
        if isinstance(X_test, pd.DataFrame):
            input_example = X_test.iloc[:3]
        else:
            input_example = pd.DataFrame(X_test, columns=feature_names).iloc[:3]

        # Compute model signature
        signature = infer_signature(X_test, y_pred)

        # Log and register the model with input_example and signature
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=f"{model_name}-{dataset_name}",
            input_example=input_example,
            signature=signature
        )
        mlflow.set_tag("context", f"{model_name} on {dataset_name} wine. Check artifacts for confusion matrix and feature importance.")

def _plot_and_save_confusion_matrix(y_true, y_pred, labels, run_name, out_dir="mlflow_artifacts"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"{run_name} Confusion Matrix")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"cm_{run_name}.png")
    plt.savefig(fname)
    plt.close()
    return fname

def _plot_and_save_feat_importance(feature_importances, feature_names, run_name, out_dir="mlflow_artifacts"):
    sorted_idx = np.argsort(feature_importances)[::-1]
    plt.figure(figsize=(8,4))
    plt.bar(np.array(feature_names)[sorted_idx], np.array(feature_importances)[sorted_idx])
    plt.title(f"{run_name} Feature Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"fi_{run_name}.png")
    plt.savefig(fname)
    plt.close()
    return fname