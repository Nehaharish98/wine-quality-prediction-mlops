from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Load MLflow model from the registry (ensure env MLflow URI is set)
model = mlflow.pyfunc.load_model("models:/RandomForest-red@production")

@app.post("/predict")
def predict(features: WineFeatures):
    try:
        data = pd.DataFrame([features.dict()])
        prediction = model.predict(data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Define data schema
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float