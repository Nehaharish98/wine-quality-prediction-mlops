import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def train_models(df, config, params):
    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["train"]["test_size"],
        random_state=params["train"]["random_state"]
    )

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="RandomForest"):
        rf = RandomForestRegressor(**params["model"]["random_forest"])
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)

        mlflow.sklearn.log_model(rf, "random_forest_model")
        mlflow.log_metric("rmse", rmse)

    with mlflow.start_run(run_name="XGBoost"):
        xgb = XGBRegressor(**params["model"]["xgboost"])
        xgb.fit(X_train, y_train)
        preds = xgb.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)

        mlflow.xgboost.log_model(xgb, "xgboost_model")
        mlflow.log_metric("rmse", rmse)