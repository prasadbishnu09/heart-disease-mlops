import joblib
import numpy as np

def test_model_loads():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    assert model is not None
    assert scaler is not None

def test_model_prediction_shape():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")

    sample = np.array([[63,1,1,145,233,1,2,150,0,2.3,3,0,6]])
    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)

    assert len(pred) == 1