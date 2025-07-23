import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.build_features import load_data, preprocess
from src.models.train_evaluate import train_random_forest, train_xgboost
from src.utils.mlflow_logger import log_with_mlflow

# ------- Load config --------
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)
data_conf = config["data"]
mlflow_conf = config["mlflow"]

# ------- Load & preprocess data ---------
red_df, white_df = load_data(data_conf["raw_data_red"], data_conf["raw_data_white"])
X_red, y_red, scaler_red = preprocess(red_df)
X_white, y_white, scaler_white = preprocess(white_df)

# ------- Split data --------
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
    X_red, y_red, stratify=y_red, test_size=0.2, random_state=42
)
X_train_white, X_test_white, y_train_white, y_test_white = train_test_split(
    X_white, y_white, stratify=y_white, test_size=0.2, random_state=42
)

# ------- Feature names ---------
feature_names = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

# ------- Train and log all models --------
# Random Forest on Red Wine
rf_red, rf_red_params = train_random_forest(X_train_red, y_train_red)
log_with_mlflow(rf_red, "RandomForest", "red", rf_red_params, X_test_red, y_test_red, feature_names, mlflow_conf)

# XGBoost on Red Wine
xgb_red, xgb_red_params = train_xgboost(X_train_red, y_train_red)
log_with_mlflow(xgb_red, "XGBoost", "red", xgb_red_params, X_test_red, y_test_red, feature_names, mlflow_conf)

# Random Forest on White Wine
rf_white, rf_white_params = train_random_forest(X_train_white, y_train_white)
log_with_mlflow(rf_white, "RandomForest", "white", rf_white_params, X_test_white, y_test_white, feature_names, mlflow_conf)

# XGBoost on White Wine
xgb_white, xgb_white_params = train_xgboost(X_train_white, y_train_white)
log_with_mlflow(xgb_white, "XGBoost", "white", xgb_white_params, X_test_white, y_test_white, feature_names, mlflow_conf)