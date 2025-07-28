"""FastAPI application."""
from fastapi import FastAPI

app = FastAPI(title="Wine Quality Predictor", version="0.1.0")

@app.get("/")
def read_root():
    return {"message": "Wine Quality Predictor API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
