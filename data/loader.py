"""Data loading utilities."""
import pandas as pd
from pathlib import Path
from typing import Tuple

def load_wine_data(data_path: str) -> pd.DataFrame:
    """Load wine quality dataset."""
    return pd.read_csv(data_path)

def split_features_target(df: pd.DataFrame, target_col: str = 'quality') -> Tuple[pd.DataFrame, pd.Series]:
    """Split features and target."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y