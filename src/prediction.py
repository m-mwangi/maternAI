import numpy as np
import pandas as pd
import joblib

def load_model(model_path="random_forest_maternal_health.pkl", scaler_path="scaler.pkl"):
    """Loads the trained model and scaler."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully!")
    return model, scaler

def make_prediction(model, scaler, sample):
    """Takes a raw input sample, applies scaling, and predicts the risk level."""
    # Feature names (ensure these match the model's training data)
    feature_names = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]  

    # Convert input list to DataFrame
    sample_df = pd.DataFrame([sample], columns=feature_names)

    # Apply scaling
    sample_scaled = scaler.transform(sample_df)

    # Make prediction
    predicted_risk_numeric = model.predict(sample_scaled)[0]

    # Convert numerical prediction to label
    risk_mapping = {0: "low risk", 1: "mid risk", 2: "high risk"}
    predicted_risk_label = risk_mapping[predicted_risk_numeric]

    return predicted_risk_label

if __name__ == "__main__":
    # Load model and scaler
    model, scaler = load_model()

    # 🔹 **Manually input values** (Replace these with actual values)
    sample = [25, 120, 80, 6.5, 37.5, 75]  # Example: Age 25, BP 120/80, BS 6.5, BodyTemp 37.5, HeartRate 75

    # Make prediction
    predicted_risk = make_prediction(model, scaler, sample)
    print(f"Predicted Risk Level: {predicted_risk}")
