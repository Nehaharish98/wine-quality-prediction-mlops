FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Create directories for models and data
RUN mkdir -p models data/raw data/processed

# Copy pre-trained model if available
#COPY models/ models/ 2>/dev/null || true
COPY models/ models/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/model.pkl

# Command to run the API - CORRECTED PATH
CMD ["uvicorn", "wine_quality.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
