"""Simple ensemble - MAXIMUM PERFORMANCE"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import joblib
from pathlib import Path

def train_simple_ensemble(data_path: str, model_output: str):
    """Train simple but powerful ensemble."""
    # Load data
    df = pd.read_parquet(data_path)
    feature_cols = [col for col in df.columns if col != 'quality']
    X, y = df[feature_cols], df['quality']

    print(f"Dataset shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training ensemble model...")

    # Three different tree-based models
    rf1 = RandomForestClassifier(
        n_estimators=300, max_depth=15, class_weight='balanced',
        random_state=42, n_jobs=-1
    )

    rf2 = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=5,
        class_weight='balanced', random_state=123, n_jobs=-1
    )

    et = ExtraTreesClassifier(
        n_estimators=300, max_depth=18, class_weight='balanced',
        random_state=456, n_jobs=-1
    )

    # Voting ensemble
    ensemble = VotingClassifier(
        estimators=[('rf1', rf1), ('rf2', rf2), ('et', et)],
        voting='soft'
    )

    ensemble.fit(X_train, y_train)

    # Evaluate
    y_pred = ensemble.predict(X_test)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    print(f"\n=== ENSEMBLE PERFORMANCE ===")
    print(f"F1 Score (weighted): {f1_weighted:.4f}")
    print(f"Target (≥0.68): {'✅ ACHIEVED' if f1_weighted >= 0.68 else '❌ BELOW'}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save model
    Path(model_output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, model_output)
    print(f"Ensemble model saved to {model_output}")

    return f1_weighted

if __name__ == "__main__":
    train_simple_ensemble('data/processed/train.parquet', 'models/ensemble.pkl')
