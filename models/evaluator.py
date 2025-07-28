"""Model evaluation utilities."""
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pandas as pd

def evaluate_model(y_true, y_pred) -> dict:
    """Evaluate model performance."""
    return {
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'classification_report': classification_report(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
