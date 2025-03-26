from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import joblib
import numpy as np
import os

app = FastAPI()

# Define model directory
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_maternal_health.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Define expected columns for retraining
EXPECTED_COLUMNS = ["Age", "SystolicBp", "DiastolicBp", "BS", "BodyTemp", "HeartRate", "RiskLevel"]

# Load model and scaler
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    raise FileNotFoundError(f"Model or Scaler not found in {MODEL_DIR}!")

@app.get("/")
def home():
    return {"message": "Maternal Health Risk Prediction API is Running!"}

@app.post("/predict/")
def predict(Age: float, SystolicBp: float, DiastolicBp: float, BS: float, BodyTemp: float, HeartRate: float):
    """Make a prediction using input features."""
    features = np.array([[Age, SystolicBp, DiastolicBp, BS, BodyTemp, HeartRate]])
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)[0]
    
    risk_mapping = {0: "low risk", 1: "mid risk", 2: "high risk"}
    return {"Predicted Risk Level": risk_mapping[prediction]}

@app.post("/upload/")
async def upload_data(file: UploadFile = File(...)):
    """Upload a new dataset for retraining."""
    try:
        df = pd.read_csv(file.file)
        
        # Check if the uploaded file has the expected columns
        if set(df.columns) != set(EXPECTED_COLUMNS):
            raise HTTPException(status_code=400, detail="CSV file does not match the expected format.")
        
        df.to_csv("new_data.csv", index=False)
        return {"message": "File uploaded successfully!", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain/")
def retrain_model():
    """Retrain the model when new data is uploaded."""
    try:
        df = pd.read_csv("new_data.csv")
        
        # Check if the columns match the expected ones before retraining
        if set(df.columns) != set(EXPECTED_COLUMNS):
            raise HTTPException(status_code=400, detail="Uploaded CSV file does not match the required format.")
        
        X_new = df.drop(columns=["RiskLevel"])
        y_new = df["RiskLevel"]

        # Retrain model
        from sklearn.ensemble import RandomForestClassifier
        model_new = RandomForestClassifier(n_estimators=100, random_state=42)
        model_new.fit(X_new, y_new)

        # Save new model
        joblib.dump(model_new, MODEL_PATH)
        
        return {"message": "Model retrained and updated successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

