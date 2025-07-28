"""Tests for model modules."""
import pytest
import pandas as pd
from models.trainer import WineQualityTrainer

def test_trainer_initialization():
    """Test trainer initialization."""
    trainer = WineQualityTrainer(n_estimators=10)
    assert not trainer.is_trained
    assert trainer.model.n_estimators == 10