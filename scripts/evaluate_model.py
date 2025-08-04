"""Evaluate trained wine quality model."""
import pandas as pd
import numpy as np
import argparse
import yaml
import json
from pathlib import Path
import joblib
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from params.yaml."""
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(
    model_path: str,
    data_path: str,
    output_path: str,
    params_path: str = "params.yaml"
):
    """Evaluate trained model on test dataset."""

    # Load parameters
    params = load_params(params_path)
    target_col = params['base']['target_col']

    # Load model
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    # Load test data
    print(f"Loading test data from {data_path}")
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    # Split features and target
    feature_cols = [col for col in df.columns if col != target_col]
    X_test = df[feature_cols]
    y_test = df[target_col]

    print(f"Evaluating on {len(X_test)} test samples")

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    evaluation_metrics = {
        'f1_weighted': float(f1_score(y_test, y_pred, average='weighted')),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision_macro': float(precision_score(y_test, y_pred, average='macro')),
        'recall_macro': float(recall_score(y_test, y_pred, average='macro')),
        'test_samples': len(X_test),
        'n_features': len(feature_cols),
        'target_achieved': float(f1_score(y_test, y_pred, average='weighted')) >= 0.68
    }

    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    evaluation_metrics['classification_report'] = class_report

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    evaluation_metrics['confusion_matrix'] = cm.tolist()

    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(feature_cols, model.feature_importances_))
        evaluation_metrics['feature_importance'] = {
            k: float(v) for k, v in feature_importance.items()
        }

    # Save evaluation results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)

    # Print results
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"F1 Score (weighted): {evaluation_metrics['f1_weighted']:.4f}")
    print(f"F1 Score (macro): {evaluation_metrics['f1_macro']:.4f}")
    print(f"Accuracy: {evaluation_metrics['accuracy']:.4f}")
    print(f"Precision (macro): {evaluation_metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {evaluation_metrics['recall_macro']:.4f}")
    print(f"Target achieved (â‰¥0.68): {'âœ…' if evaluation_metrics['target_achieved'] else 'âŒ'}")
    print(f"\nDetailed classification report:")
    print(classification_report(y_test, y_pred))

    # Create visualizations if matplotlib available
    try:
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        cm_plot_path = output_path.replace('.json', '_confusion_matrix.png')
        plt.savefig(cm_plot_path)
        plt.close()
        print(f"Confusion matrix saved to {cm_plot_path}")

        # Plot feature importance if available
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)

            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.title('Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()

            importance_plot_path = output_path.replace('.json', '_feature_importance.png')
            plt.savefig(importance_plot_path)
            plt.close()
            print(f"Feature importance plot saved to {importance_plot_path}")

    except ImportError:
        print("Matplotlib not available, skipping visualizations")

    print(f"\nEvaluation results saved to {output_path}")
    return evaluation_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--data", required=True, help="Path to test data")
    parser.add_argument("--output", required=True, help="Output path for evaluation results")
    parser.add_argument("--params", default="params.yaml", help="Parameters file")

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        data_path=args.data,
        output_path=args.output,
        params_path=args.params
    )
