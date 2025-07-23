import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import mlflow
import mlflow.sklearn

def split_data(X, y, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_random_forest(X_train, y_train, n_jobs=-1):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None]
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=n_jobs)
    grid_rf = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=n_jobs, verbose=1)
    grid_rf.fit(X_train, y_train)
    print(f"[RandomForest] Best params: {grid_rf.best_params_}")
    return grid_rf.best_estimator_, grid_rf.best_params_

def train_xgboost(X_train, y_train, n_jobs=-1):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.3]
    }
    xgb = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        n_jobs=n_jobs
    )
    grid_xgb = GridSearchCV(xgb, param_grid, cv=3, scoring='f1', n_jobs=n_jobs, verbose=1)
    grid_xgb.fit(X_train, y_train)
    print(f"[XGBoost] Best params: {grid_xgb.best_params_}")
    return grid_xgb.best_estimator_, grid_xgb.best_params_

# Evaluation 
def evaluate_model(model, X_test, y_test, model_name, output_dir="outputs"):
    """Evaluate a fitted model, print classification report, save confusion matrix plot."""
    y_pred = model.predict(X_test)
    print(f"\n--- {model_name} CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f} | F1: {f1_score(y_test, y_pred):.3f}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()

#Feature Importance 
def plot_feature_importance(model, feature_names, model_name, output_dir="outputs"):
    """Plot and save feature importances for tree-based models."""
    importances = getattr(model.best_estimator_, 'feature_importances_', None)
    if importances is not None:
        series = pd.Series(importances, index=feature_names)
        series.sort_values(ascending=False).plot(kind='bar', figsize=(12,6))
        plt.title(f"{model_name} Feature Importance")
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{model_name}_feature_importance.png"))
        plt.close()
        print(series.sort_values(ascending=False).head())
    else:
        print(f"[WARN] No feature_importances_ found for {model_name}")

#Model Saving 
def save_model(model, save_path):
    """Persist the given model via joblib."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model.best_estimator_, save_path)
    print(f"[INFO] Model saved to {save_path}")


if __name__ == "__main__":
    # Example usage: import & preprocess data in your pipeline, then:
    # from src.features.build_features import preprocess, load_data
    # red_df, white_df = load_data(...)
    # X, y, scaler = preprocess(red_df)
    # X_train, X_test, y_train, y_test = split_data(X, y)
    # rf_model = train_random_forest(X_train, y_train)
    # xgb_model = train_xgboost(X_train, y_train)
    # evaluate_model(rf_model, X_test, y_test, "RandomForest")
    # evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    # plot_feature_importance(rf_model, feature_names, "RandomForest")
    # plot_feature_importance(xgb_model, feature_names, "XGBoost")
    # save_model(rf_model, "models/random_forest.pkl")
    # save_model(xgb_model, "models/xgboost.pkl")
    print("Use the functions in this module from your pipeline or notebook.")