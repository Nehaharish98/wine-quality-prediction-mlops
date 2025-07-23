from prefect import flow, task
import mlflow.pyfunc
import pandas as pd

# Task: load new batch of input data
@task
def load_data(path):
    return pd.read_csv(path)

# Task: load the production (or staging) model from MLflow Registry
@task
def load_model(model_name, alias="production"):
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")  # use your MLflow server URI here
    model_uri = f"models:/{model_name}@{alias}"
    return mlflow.pyfunc.load_model(model_uri)


# Task: run predictions using loaded model and data
@task
def predict(model, data):
    return model.predict(data)

# Task: save the predictions to disk (or could be a downstream system)
@task
def save_output(preds, path):
    pd.DataFrame({"predictions": preds}).to_csv(path, index=False)

# Orchestrated Flow: runs the workflow in order, and Prefect tracks every step
@flow
def batch_inference_flow(input_csv, model_name, output_csv):
    data = load_data(input_csv)
    model = load_model(model_name)
    preds = predict(model, data)
    save_output(preds, output_csv)

# CLI/Script entrypoint for ad-hoc run; will also work if scheduled with Prefect
if __name__ == "__main__":
    batch_inference_flow("data/test/test_wine.csv", "RandomForest-red", "outputs/predictions.csv")