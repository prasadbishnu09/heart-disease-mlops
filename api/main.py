from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="Heart Disease Prediction API")

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    """
    Expected input: JSON with 13 features
    """
    features = np.array(list(data.values())).reshape(1, -1)

    # Scale input
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "confidence": round(float(probability), 3)
    }