from fastapi import FastAPI
import joblib
import numpy as np
import logging
from datetime import datetime


app = FastAPI(title="Heart Disease Prediction API")

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

@app.post("/predict")
def predict(data: dict):
    logger.info(f"Received request at {datetime.now()}")
    logger.info(f"Input data: {data}")

    features = np.array(list(data.values())).reshape(1, -1)
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    logger.info(f"Prediction: {prediction}, Confidence: {probability}")

    return {
        "prediction": int(prediction),
        "confidence": round(float(probability), 3)
    }
