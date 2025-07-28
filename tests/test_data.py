"""Tests for data modules."""
import pytest
import pandas as pd
from data.loader import load_wine_data, split_features_target

def test_split_features_target():
    """Test feature-target splitting."""
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'quality': [5, 6, 7]
    })
    X, y = split_features_target(df)
    assert list(X.columns) == ['feature1', 'feature2']
    assert list(y) == [5, 6, 7]