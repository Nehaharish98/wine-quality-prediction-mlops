FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Expose port
EXPOSE 8000

# Set environment variables
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000
ENV PYTHONPATH=/app

# Command to run the API
CMD ["uvicorn", "src.wine_quality.api.app:app", "--host", "0.0.0.0", "--port", "8000"]