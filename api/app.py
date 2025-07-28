"""FastAPI application for wine quality prediction with monitoring."""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import joblib
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import os
import uuid

from .schemas import (
    WineFeatures, 
    PredictionResponse, 
    BatchWineFeatures, 
    BatchPredictionResponse,
    HealthResponse,
    ModelInfo
)
from .metrics import (
    metrics_collector,
    track_requests,
    initialize_metrics,
    drift_detector
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Wine Quality Predictor API",
    description="Machine Learning API for predicting wine quality with monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
model_version = None
model_loaded = False
feature_names = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", 
    "density", "pH", "sulphates", "alcohol"
]
target_classes = list(range(3, 10))

def load_model():
    """Load the trained model."""
    global model, model_version, model_loaded
    
    model_path = os.getenv("MODEL_PATH", "models/model.pkl")
    
    try:
        if Path(model_path).exists():
            model = joblib.load(model_path)
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_loaded = True
            logger.info(f"Model loaded successfully from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}")
            # Load a dummy model for testing
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            X_dummy = np.random.rand(100, 11)
            y_dummy = np.random.randint(3, 10, 100)
            model.fit(X_dummy, y_dummy)
            model_version = "dummy_model"
            model_loaded = True
            logger.info("Loaded dummy model for testing")
        
        # Update metrics
        metrics_collector.update_health_status(model_loaded)
        if model_loaded:
            metrics_collector.update_model_info({
                'model_name': 'wine_quality_predictor',
                'version': model_version,
                'framework': 'scikit-learn',
                'features': ','.join(feature_names)
            })
            
        # Set baseline for drift detection (you would normally compute this from training data)
        baseline_features = {
            'fixed_acidity': 8.32,
            'volatile_acidity': 0.53,
            'citric_acid': 0.27,
            'residual_sugar': 2.54,
            'chlorides': 0.09,
            'free_sulfur_dioxide': 15.87,
            'total_sulfur_dioxide': 46.47,
            'density': 0.997,
            'pH': 3.31,
            'sulphates': 0.66,
            'alcohol': 10.42
        }
        drift_detector.set_baseline(baseline_features)
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_loaded = False
        metrics_collector.update_health_status(False)

# Load model on startup
load_model()

@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    logger.info("Starting Wine Quality Predictor API with monitoring")
    initialize_metrics()
    if not model_loaded:
        load_model()

@app.get("/metrics", response_class=PlainTextResponse, tags=["Monitoring"])
async def get_metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

@app.get("/", tags=["Root"])
@track_requests
async def read_root():
    """Root endpoint."""
    return {
        "message": "Wine Quality Predictor API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "metrics": "/metrics"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
@track_requests
async def health_check():
    """Health check endpoint."""
    is_healthy = model_loaded
    metrics_collector.update_health_status(is_healthy)
    
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=model_loaded,
        model_version=model_version
    )

@app.get("/model-info", response_model=ModelInfo, tags=["Model"])
@track_requests
async def get_model_info():
    """Get model information."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_name="Wine Quality Predictor",
        model_version=model_version,
        features=feature_names,
        target_classes=target_classes,
        last_updated=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
@track_requests
async def predict_wine_quality(wine: WineFeatures, request: Request):
    """Predict wine quality for a single sample."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input data
        input_data = wine.to_array()
        
        # Update drift detection
        feature_dict = {
            'fixed_acidity': wine.fixed_acidity,
            'volatile_acidity': wine.volatile_acidity,
            'citric_acid': wine.citric_acid,
            'residual_sugar': wine.residual_sugar,
            'chlorides': wine.chlorides,
            'free_sulfur_dioxide': wine.free_sulfur_dioxide,
            'total_sulfur_dioxide': wine.total_sulfur_dioxide,
            'density': wine.density,
            'pH': wine.pH,
            'sulphates': wine.sulphates,
            'alcohol': wine.alcohol
        }
        drift_detector.update_current_stats(feature_dict)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Calculate confidence
        confidence = float(np.max(probabilities))
        
        # Create probability dictionary
        prob_dict = {
            str(class_): float(prob) 
            for class_, prob in zip(target_classes, probabilities)
        }
        
        # Record prediction metrics
        metrics_collector.record_prediction(int(prediction))
        
        logger.info(f"Prediction made: quality={prediction}, confidence={confidence:.3f}")
        
        return PredictionResponse(
            quality=int(prediction),
            confidence=confidence,
            probabilities=prob_dict
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Prediction"])
@track_requests
async def predict_batch_wine_quality(batch: BatchWineFeatures):
    """Predict wine quality for multiple samples."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        
        for wine in batch.wines:
            # Prepare input data
            input_data = wine.to_array()
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            
            # Calculate confidence
            confidence = float(np.max(probabilities))
            
            # Create probability dictionary
            prob_dict = {
                str(class_): float(prob) 
                for class_, prob in zip(target_classes, probabilities)
            }
            
            # Record prediction metrics
            metrics_collector.record_prediction(int(prediction))
            
            predictions.append(PredictionResponse(
                quality=int(prediction),
                confidence=confidence,
                probabilities=prob_dict
            ))
        
        logger.info(f"Batch prediction completed: {len(predictions)} samples")
        
        return BatchPredictionResponse(predictions=predictions)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/drift-report", tags=["Monitoring"])
@track_requests
async def get_drift_report():
    """Get current drift report."""
    drift_scores = drift_detector.calculate_drift()
    
    return {
        "drift_scores": drift_scores,
        "sample_count": drift_detector.sample_count,
        "drift_threshold": drift_detector.drift_threshold,
        "features_with_drift": [
            feature for feature, score in drift_scores.items()
            if score > drift_detector.drift_threshold
        ]
    }

@app.post("/reload-model", tags=["Model"])
@track_requests
async def reload_model(background_tasks: BackgroundTasks):
    """Reload the model."""
    background_tasks.add_task(load_model)
    return {"message": "Model reload initiated"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)