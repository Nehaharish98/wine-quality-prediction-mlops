import os
import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from prefect import flow

#CONFIGURATION 
REFERENCE_PATH = "data/raw/wine-quality-red.csv"         # Adjust if needed
CURRENT_LOG_PATH = "logs/prediction_log.csv"             # Where you log production/batch prediction inputs
REPORT_DIR = "reports"
REPORT_PATH = os.path.join(REPORT_DIR, "drift_report.html")

#UTILITY: Load Current Data
def load_current_features(log_csv_path):
    predlog = pd.read_csv(log_csv_path)
    features_df = predlog["features"].apply(eval).apply(pd.Series)
    return features_df

#MAIN DRIFT CHECK LOGIC 
def run_drift_check():
    # 1. Load reference/training and current/production feature data
    ref = pd.read_csv(REFERENCE_PATH, delimiter=";")  # assumes original CSV semi-colon
    current = load_current_features(CURRENT_LOG_PATH)
    ref_features = ref.drop(columns=["quality"])
    
    # 2. Run Evidently drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_features, current_data=current)
    
    # 3. Save report as HTML
    os.makedirs(REPORT_DIR, exist_ok=True)
    report.save_html(REPORT_PATH)
    print(f"[INFO] Drift report saved to {REPORT_PATH}")
    
    # 4. Optionally log as MLflow artifact
    with mlflow.start_run(run_name="drift_monitoring", nested=True):
        mlflow.log_artifact(REPORT_PATH, artifact_path="monitoring")
        print("[INFO] Drift report logged to MLflow.")

# ==========Prefect Scheduling ==========
if __name__ == "__main__":
    @flow
    def prefect_flow(): run_drift_check()
    prefect_flow.serve(cron="0 2 * * *") # for daily at 2am
    run_drift_check()
