from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import os
import numpy as np  # Import NumPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
    scaler = joblib.load(SCALER_PATH)  # Load scaler
else:
    raise FileNotFoundError(f"Model or Scaler not found in {MODEL_DIR}!")

@app.get("/")
def home():
    return {"message": "Maternal Health Risk Prediction API is Running!"}

@app.post("/predict/")
def predict(Age: float, SystolicBP: float, DiastolicBP: float, BS: float, BodyTemp: float, HeartRate: float):
    """Make a prediction using input features."""
    
    # Create input feature array
    features = np.array([[Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate]])
    
    # Scale the input features
    features_scaled = scaler.transform(features)
    
    # Make a prediction
    prediction = model.predict(features_scaled)[0]
    
    # Map numerical prediction to risk level
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
        
        # Save the uploaded file for retraining
        upload_file_path = "new_data.csv"
        df.to_csv(upload_file_path, index=False)

        return JSONResponse(content={"message": "File uploaded successfully for retraining."})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain/")
def retrain_model(file_path: str = "new_data.csv"):
    """Retrain the model using the uploaded data and evaluate it."""
    try:
        # Load the uploaded data for retraining
        df = pd.read_csv(file_path)
        
        # Check if the columns match the expected ones before retraining
        if set(df.columns) != set(EXPECTED_COLUMNS):
            raise HTTPException(status_code=400, detail="Uploaded CSV file does not match the required format.")
        
        X_new = df.drop(columns=["RiskLevel"])
        y_new = df["RiskLevel"]

        # Split the data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

        # Retrain the model
        model_new = RandomForestClassifier(n_estimators=100, random_state=42)
        model_new.fit(X_train, y_train)

        # Evaluate the new model
        y_pred = model_new.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Save the retrained model
        joblib.dump(model_new, MODEL_PATH)

        return {
            "message": "Model retrained successfully!",
            "new_model_version": MODEL_PATH,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retraining the model: {str(e)}")
