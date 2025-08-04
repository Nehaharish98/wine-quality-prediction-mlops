"""XGBoost model - FIXED AND OPTIMIZED"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import json
from pathlib import Path

def train_xgboost_fixed(data_path: str, model_output: str):
    """Train XGBoost with proper configuration."""
    # Load data
    df = pd.read_parquet(data_path)
    feature_cols = [col for col in df.columns if col != 'quality']
    X = df[feature_cols]
    y = df['quality']

    print(f"Dataset shape: {X.shape}")
    print(f"Quality distribution:\n{y.value_counts().sort_index()}")

    # Encode labels for XGBoost
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Compute sample weights for imbalanced classes
    sample_weights = compute_sample_weight('balanced', y_train)

    # PROPERLY TUNED XGBOOST
    print("Training XGBoost model...")
    model = XGBClassifier(
        objective='multi:softprob',
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=0.01,
        random_state=42,
        n_jobs=-1
    )

    # Fit with sample weights
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n=== XGBOOST PERFORMANCE ===")
    print(f"F1 Score (weighted): {f1_weighted:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Target (≥0.68): {'✅ ACHIEVED' if f1_weighted >= 0.68 else '❌ BELOW'}")

    # Convert back to original labels for report
    y_test_orig = label_encoder.inverse_transform(y_test)
    y_pred_orig = label_encoder.inverse_transform(y_pred)

    print(f"\n=== DETAILED REPORT ===")
    print(classification_report(y_test_orig, y_pred_orig, zero_division=0))

    # Save model
    Path(model_output).parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        'model': model,
        'label_encoder': label_encoder,
        'feature_cols': feature_cols
    }
    joblib.dump(model_data, model_output)

    # Save metrics (fix JSON serialization)
    metrics = {
        'algorithm': 'XGBoost',
        'f1_weighted': float(f1_weighted),
        'f1_macro': float(f1_macro),
        'accuracy': float(accuracy),
        'target_achieved': bool(f1_weighted >= 0.68)  # Convert to native Python bool
    }

    metrics_path = model_output.replace('.pkl', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {model_output}")
    return f1_weighted

if __name__ == "__main__":
    train_xgboost_fixed('data/processed/train.parquet', 'models/xgboost_fixed.pkl')
