from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import mlflow.xgboost


def train_model(X_train, y_train, model_type="random_forest"):
    mlflow.set_experiment("wine-quality")

    with mlflow.start_run():
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            mlflow.sklearn.autolog()
        else:
            model = XGBClassifier(n_estimators=100, max_depth=10, use_label_encoder=False, eval_metric='mlogloss')
            mlflow.xgboost.autolog()

        model.fit(X_train, y_train)
        return model