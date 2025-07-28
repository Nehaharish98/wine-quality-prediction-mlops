"""API request/response schemas."""
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

class WineFeatures(BaseModel):
    """Wine features for prediction."""
    fixed_acidity: float = Field(..., description="Fixed acidity", ge=0)
    volatile_acidity: float = Field(..., description="Volatile acidity", ge=0)
    citric_acid: float = Field(..., description="Citric acid", ge=0)
    residual_sugar: float = Field(..., description="Residual sugar", ge=0)
    chlorides: float = Field(..., description="Chlorides", ge=0)
    free_sulfur_dioxide: float = Field(..., description="Free sulfur dioxide", ge=0)
    total_sulfur_dioxide: float = Field(..., description="Total sulfur dioxide", ge=0)
    density: float = Field(..., description="Density", gt=0)
    pH: float = Field(..., description="pH", ge=0, le=14)
    sulphates: float = Field(..., description="Sulphates", ge=0)
    alcohol: float = Field(..., description="Alcohol", ge=0)
    wine_type: Optional[str] = Field(default="red", description="Wine type (red/white)")
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for prediction."""
        return np.array([[
            self.fixed_acidity,
            self.volatile_acidity,
            self.citric_acid,
            self.residual_sugar,
            self.chlorides,
            self.free_sulfur_dioxide,
            self.total_sulfur_dioxide,
            self.density,
            self.pH,
            self.sulphates,
            self.alcohol
        ]])
    
    class Config:
        schema_extra = {
            "example": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.7,
                "citric_acid": 0.0,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4,
                "wine_type": "red"
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response schema."""
    quality: int = Field(..., description="Predicted wine quality (3-9)")
    confidence: float = Field(..., description="Prediction confidence")
    probabilities: dict = Field(..., description="Class probabilities")

class BatchWineFeatures(BaseModel):
    """Batch prediction request."""
    wines: List[WineFeatures]

class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: Optional[str] = None

class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    model_version: str
    features: List[str]
    target_classes: List[int]
    last_updated: str