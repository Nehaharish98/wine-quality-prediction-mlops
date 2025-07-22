from src.data.make_dataset import load_wine_data, save_dataset
from src.features.build_features import build_features
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model
from prefect import flow, task

@task
def load_and_process():
    df = load_wine_data()
    save_dataset(df)
    return df

@task
def split_data(df):
    return build_features(df)

@task
def train(X_train, y_train, model_type="random_forest"):
    return train_model(X_train, y_train, model_type=model_type)

@task
def evaluate(model, X_test, y_test):
    return evaluate_model(model, X_test, y_test)

@flow(name="Wine Quality Prediction Pipeline")
def main_flow(model_type="random_forest"):
    df = load_and_process()
    X_train, X_test, y_train, y_test = split_data(df)
    model = train(X_train, y_train, model_type=model_type)
    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main_flow(model_type="xgboost")