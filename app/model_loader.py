import joblib
import os

def load_model():
    model_name = os.getenv("MODEL_NAME", "models/saved/model_v1.pkl")
    print(f"Loading model: {model_name}")
    return joblib.load(model_name), model_name
