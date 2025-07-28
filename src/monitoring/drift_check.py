import os
import pandas as pd
import mlflow
from evidently import Report, DataDriftPreset
from prefect import flow
import ast

REFERENCE_PATH = "data/raw/wine-quality-red.csv"
CURRENT_LOG_PATH = "logs/prediction_log.csv"
REPORT_DIR = "reports"
REPORT_PATH = os.path.join(REPORT_DIR, "drift_report.html")

def load_current_features(log_csv_path):
    predlog = pd.read_csv(log_csv_path)
    features_df = predlog["features"].apply(ast.literal_eval).apply(pd.Series)
    return features_df

def run_drift_check():
    ref = pd.read_csv(REFERENCE_PATH, delimiter=";")
    current = load_current_features(CURRENT_LOG_PATH)
    ref_features = ref.drop(columns=["quality"])
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_features, current_data=current)
    os.makedirs(REPORT_DIR, exist_ok=True)
    report.save_html(REPORT_PATH)
    print(f"[INFO] Drift report saved to {REPORT_PATH}")
    with mlflow.start_run(run_name="drift_monitoring", nested=True):
        mlflow.log_artifact(REPORT_PATH, artifact_path="monitoring")
        print("[INFO] Drift report logged to MLflow.")

@flow
def prefect_flow():
    run_drift_check()

if __name__ == "__main__":
    # For scheduling remove, or keep for immediate manual testing
    prefect_flow.serve(cron="0 2 * * *")