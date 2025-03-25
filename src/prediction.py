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
    # Convert to DataFrame with feature names
    feature_names = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]  # Update with correct feature names
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

    # Example: Use the first test sample for prediction
    dataset_path = "Maternal Health Risk Data Set.csv"
    df = pd.read_csv(dataset_path)
    df_cleaned = df.drop(columns=["RiskLevel"])  # Remove target column

    sample = df_cleaned.iloc[0].values  # Select the first row as a test sample

    # Make prediction
    predicted_risk = make_prediction(model, scaler, sample)
    print(f"Predicted Risk Level: {predicted_risk}")