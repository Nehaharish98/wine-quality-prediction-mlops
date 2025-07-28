"""Prometheus metrics for wine quality API."""
from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Define metrics
REQUEST_COUNT = Counter(
    'wine_requests_total',
    'Total number of requests to wine quality API',
    ['method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'wine_latency_seconds',
    'Request latency for wine quality predictions',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'wine_predictions_total',
    'Total number of predictions made',
    ['predicted_quality']
)

ACTIVE_REQUESTS = Gauge(
    'wine_active_requests',
    'Number of active requests being processed'
)

MODEL_INFO = Info(
    'wine_model_info',
    'Information about the wine quality model'
)

HEALTH_STATUS = Gauge(
    'wine_api_health',
    'Health status of wine quality API (1=healthy, 0=unhealthy)'
)

# Model performance metrics
MODEL_ACCURACY = Gauge(
    'wine_model_accuracy',
    'Current model accuracy score'
)

FEATURE_DRIFT = Gauge(
    'wine_feature_drift',
    'Feature drift score for wine quality model',
    ['feature_name']
)

class MetricsCollector:
    """Collect and manage metrics for the wine quality API."""
    
    def __init__(self):
        self.request_start_times: Dict[str, float] = {}
        
    def record_request_start(self, request_id: str):
        """Record the start time of a request."""
        self.request_start_times[request_id] = time.time()
        ACTIVE_REQUESTS.inc()
    
    def record_request_end(
        self, 
        request_id: str, 
        method: str, 
        endpoint: str, 
        status_code: int
    ):
        """Record the end of a request and calculate metrics."""
        if request_id in self.request_start_times:
            duration = time.time() - self.request_start_times[request_id]
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
            del self.request_start_times[request_id]
        
        REQUEST_COUNT.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=status_code
        ).inc()
        
        ACTIVE_REQUESTS.dec()
    
    def record_prediction(self, predicted_quality: int):
        """Record a prediction."""
        PREDICTION_COUNT.labels(predicted_quality=str(predicted_quality)).inc()
    
    def update_model_info(self, model_info: Dict[str, Any]):
        """Update model information."""
        MODEL_INFO.info(model_info)
    
    def update_health_status(self, is_healthy: bool):
        """Update health status."""
        HEALTH_STATUS.set(1 if is_healthy else 0)
    
    def update_model_accuracy(self, accuracy: float):
        """Update model accuracy metric."""
        MODEL_ACCURACY.set(accuracy)
    
    def update_feature_drift(self, feature_name: str, drift_score: float):
        """Update feature drift metric."""
        FEATURE_DRIFT.labels(feature_name=feature_name).set(drift_score)

# Global metrics collector instance
metrics_collector = MetricsCollector()

def track_requests(func):
    """Decorator to track request metrics."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request_id = f"{time.time()}_{id(args)}"
        method = "POST"  # Most of our endpoints are POST
        endpoint = func.__name__
        
        try:
            metrics_collector.record_request_start(request_id)
            result = await func(*args, **kwargs)
            metrics_collector.record_request_end(request_id, method, endpoint, 200)
            return result
        except Exception as e:
            metrics_collector.record_request_end(request_id, method, endpoint, 500)
            raise e
    
    return wrapper

def initialize_metrics():
    """Initialize metrics with default values."""
    metrics_collector.update_health_status(False)
    MODEL_ACCURACY.set(0.0)
    
    # Initialize model info
    metrics_collector.update_model_info({
        'model_name': 'wine_quality_predictor',
        'version': '1.0.0',
        'framework': 'scikit-learn'
    })
    
    logger.info("Metrics initialized")

# Data drift detection utilities
class DriftDetector:
    """Simple drift detection for wine quality features."""
    
    def __init__(self):
        self.baseline_stats = {}
        self.current_stats = {}
        self.sample_count = 0
        self.drift_threshold = 0.1
    
    def set_baseline(self, features: Dict[str, float]):
        """Set baseline statistics."""
        self.baseline_stats = features.copy()
    
    def update_current_stats(self, features: Dict[str, float]):
        """Update current statistics with new sample."""
        self.sample_count += 1
        
        for feature, value in features.items():
            if feature not in self.current_stats:
                self.current_stats[feature] = []
            
            self.current_stats[feature].append(value)
            
            # Keep only last 1000 samples
            if len(self.current_stats[feature]) > 1000:
                self.current_stats[feature].pop(0)
    
    def calculate_drift(self) -> Dict[str, float]:
        """Calculate drift scores for all features."""
        drift_scores = {}
        
        if not self.baseline_stats or self.sample_count < 10:
            return drift_scores
        
        for feature in self.baseline_stats:
            if feature in self.current_stats and self.current_stats[feature]:
                # Simple drift calculation using mean difference
                baseline_mean = self.baseline_stats[feature]
                current_mean = sum(self.current_stats[feature]) / len(self.current_stats[feature])
                
                drift_score = abs(current_mean - baseline_mean) / abs(baseline_mean + 1e-8)
                drift_scores[feature] = drift_score
                
                # Update Prometheus metric
                metrics_collector.update_feature_drift(feature, drift_score)
        
        return drift_scores

# Global drift detector
drift_detector = DriftDetector()