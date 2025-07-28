from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb

def _choose_model(problem, algo):
    if algo == "rf":
        return (RandomForestRegressor if problem == "regression" else RandomForestClassifier)(random_state=42)
    if algo == "xgb":
        return (xgb.XGBRegressor if problem == "regression" else xgb.XGBClassifier)(random_state=42)
    raise ValueError("Unknown algo")

def train(algo, X, y, problem):
    model = _choose_model(problem, algo)
    grid = {
        "rf":  {"n_estimators": [100, 200], "max_depth": [None, 10]},
        "xgb": {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.1, 0.3]},
    }[algo]

    scorer = "neg_mean_squared_error" if problem == "regression" else "accuracy"
    search = GridSearchCV(model, grid, cv=3, scoring=scorer, n_jobs=-1, verbose=0)
    search.fit(X, y)
    print(f"[{algo.upper()}] best params:", search.best_params_)
    return search.best_estimator_, search.best_params_