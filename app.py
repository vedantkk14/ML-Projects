from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import json

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all models
models = {
    'linear_regression': joblib.load('linear_regression_model.pkl'),
    'random_forest': joblib.load('random_forest_model.pkl'),
    'xgboost': joblib.load('xgboost_model.pkl')
}
scaler = joblib.load('scaler.pkl')

# Load model results
with open('model_results.json', 'r') as f:
    model_results = json.load(f)

@app.get("/")
def root():
    return FileResponse("index.html")

@app.get("/model-results")
def get_model_results():
    """Get comparison results for all models"""
    return {"results": model_results}

class InputData(BaseModel):
    Hours_Studied: float
    Previous_Scores: float
    Extracurricular_Activities: int
    Sleep_Hours: float
    Sample_Question_Papers_Practiced: float
    model: str = "xgboost"

@app.post("/predict")
def predict(data: InputData):
    """Predict using selected model (default: xgboost)"""
    # Validate model selection
    selected_model = data.model.lower()
    if selected_model not in models:
        return {"error": f"Model '{data.model}' not found. Available: {list(models.keys())}"}
    
    input_array = np.array([[data.Hours_Studied, data.Previous_Scores,
                            data.Extracurricular_Activities, data.Sleep_Hours,
                            data.Sample_Question_Papers_Practiced]])
    scaled = scaler.transform(input_array)
    prediction = models[selected_model].predict(scaled)[0]
    return {
        "model": data.model,
        "Predicted Performance Index": round(float(prediction), 2)
    }