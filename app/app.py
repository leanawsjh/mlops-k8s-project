from fastapi import FastAPI
import numpy as np
import logging
from model_loader import load_model
from drift import detect_drift

app = FastAPI()

logging.basicConfig(level=logging.INFO)

model, model_name = load_model()

@app.get("/")
def home():
    return {"message": "ML API running", "model": model_name}

@app.get("/health")
def health():
    return {"status": "healthy", "model": model_name}

@app.post("/predict")
def predict(features: list):
    features_array = np.array([features])
    prediction = model.predict(features_array)

    drift = detect_drift(features)

    logging.info(f"Input: {features}, Prediction: {prediction}, Drift: {drift}")

    return {
        "prediction": float(prediction[0]),
        "drift_detected": drift,
        "model": model_name
    }
