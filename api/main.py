from fastapi import FastAPI
import joblib
import numpy as np
import logging
from datetime import datetime

# 🔹 NEW: Prometheus metrics instrumentation
from prometheus_fastapi_instrumentator import Instrumentator


# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(title="Heart Disease Prediction API")

# 🔹 NEW: Enable Prometheus-compatible metrics
Instrumentator().instrument(app).expose(app)


# -------------------------------
# Load model and scaler
# -------------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# -------------------------------
# Logging configuration
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# -------------------------------
# Health check endpoint
# -------------------------------
@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}


# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
def predict(data: dict):
    # Log request timestamp
    logger.info(f"Received request at {datetime.now()}")

    # Log input payload
    logger.info(f"Input data: {data}")

    # Preprocess input
    features = np.array(list(data.values())).reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Model inference
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    # Log prediction and confidence
    logger.info(f"Prediction: {prediction}, Confidence: {probability}")

    return {
        "prediction": int(prediction),
        "confidence": round(float(probability), 3)
    }