"""Model prediction utilities."""
import joblib
import pandas as pd
import numpy as np

class WineQualityPredictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)