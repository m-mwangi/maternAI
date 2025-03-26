from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import os
import numpy as np
from src.model import train_model, load_model
from src.preprocessing import preprocess_data, load_scaler
from src.prediction import make_prediction

app = FastAPI()

# Define model directory
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_maternal_health.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Load existing model and scaler
try:
    model = load_model(MODEL_PATH)
    scaler = load_scaler(SCALER_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Model or scaler not found in {MODEL_DIR}! Ensure they exist.")

@app.get("/")
def home():
    return {"message": "Maternal Health Risk Prediction API is Running!"}

@app.post("/predict/")
def predict(Age: float, SystolicBP: float, DiastolicBP: float, BS: float, BodyTemp: float, HeartRate: float):
    """Make a prediction using input features."""
    input_features = np.array([[Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate]])
    risk_level = make_prediction(model, scaler, input_features)
    return {"Predicted Risk Level": risk_level}

@app.post("/upload/")
async def upload_data(file: UploadFile = File(...)):
    """Upload a new dataset for retraining."""
    try:
        df = pd.read_csv(file.file)

        # Check if the uploaded file has the expected columns
        expected_columns = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "RiskLevel"]
        if set(df.columns) != set(expected_columns):
            raise HTTPException(status_code=400, detail="CSV file does not match the expected format.")
        
        # Save uploaded data
        upload_file_path = "new_data.csv"
        df.to_csv(upload_file_path, index=False)

        return JSONResponse(content={"message": "File uploaded successfully for retraining."})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain/")
def retrain_model(file_path: str = "new_data.csv"):
    """Retrain the model using uploaded data and evaluate it."""
    try:
        new_model, accuracy = train_model(file_path)
        joblib.dump(new_model, MODEL_PATH)
        return {
            "message": "Model retrained successfully!",
            "accuracy": accuracy,
            "new_model_version": MODEL_PATH,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retraining the model: {str(e)}")
