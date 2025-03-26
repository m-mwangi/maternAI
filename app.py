from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import joblib
import numpy as np
import os
import time

app = FastAPI()

# Define model directory
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_maternal_health.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Define expected columns for retraining
EXPECTED_COLUMNS = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "RiskLevel"]

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
        
        # Log the filename and a sample of the data (first few rows)
        print(f"File {file.filename} uploaded successfully!")
        print(f"Sample data: \n{df.head()}")

        # Save the uploaded file
        df.to_csv("new_data.csv", index=False)

        # Log file timestamp
        upload_timestamp = time.ctime(os.path.getmtime("new_data.csv"))
        print(f"File {file.filename} uploaded at {upload_timestamp}")
        
        return {"message": "File uploaded successfully!", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain/")
def retrain_model():
    """Retrain the model when new data is uploaded."""
    try:
        # Read the uploaded data used for retraining
        df = pd.read_csv("new_data.csv")

        # Log information about the new data being used for retraining
        retrain_timestamp = time.ctime(os.path.getmtime("new_data.csv"))
        print(f"Retraining with the data from new_data.csv. Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")
        print(f"Retraining using data from new_data.csv, file last modified at {retrain_timestamp}")
        print(f"Sample data used for retraining: \n{df.head()}")

        # Check if the columns match the expected ones before retraining
        if set(df.columns) != set(EXPECTED_COLUMNS):
            raise HTTPException(status_code=400, detail="Uploaded CSV file does not match the required format.")
        
        X_new = df.drop(columns=["RiskLevel"])
        y_new = df["RiskLevel"]

        # Retrain model
        from sklearn.ensemble import RandomForestClassifier
        model_new = RandomForestClassifier(n_estimators=100, random_state=42)
        model_new.fit(X_new, y_new)

        # Save the new model
        joblib.dump(model_new, MODEL_PATH)

        return {
            "message": f"Model retrained and updated successfully using {df.shape[0]} records."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
