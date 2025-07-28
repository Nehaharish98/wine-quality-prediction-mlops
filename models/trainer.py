"""Model training utilities."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib
import pandas as pd

class WineQualityTrainer:
    def __init__(self, **model_params):
        self.model = RandomForestClassifier(**model_params)
        self.is_trained = False
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train the model and return metrics."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        y_pred = self.model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        return {
            'f1_weighted': f1,
            'classification_report': classification_report(y_test, y_pred)
        }
    
    def save_model(self, path: str):
        """Save trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        joblib.dump(self.model, path)