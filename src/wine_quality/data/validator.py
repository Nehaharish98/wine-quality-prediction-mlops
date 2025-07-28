"""Data validation utilities."""
import pandas as pd
from typing import List

def validate_features(df: pd.DataFrame, expected_columns: List[str]) -> bool:
    """Validate that DataFrame contains expected columns."""
    return set(expected_columns).issubset(set(df.columns))

def check_data_quality(df: pd.DataFrame) -> dict:
    """Check basic data quality metrics."""
    return {
        'null_count': df.isnull().sum().sum(),
        'duplicate_count': df.duplicated().sum(),
        'shape': df.shape
    }