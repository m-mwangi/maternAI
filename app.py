from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

app = FastAPI()

# Define directories
MODEL_DIR = "models"
UPLOAD_DIR = "uploads"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Model paths
MODEL_VERSION = 1  # Initialize model version
MODEL_PATH_TEMPLATE = os.path.join(MODEL_DIR, "random_forest_model_v{}.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Expected columns for retraining
EXPECTED_COLUMNS = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "RiskLevel"]

# Load model
model_path = MODEL_PATH_TEMPLATE.format(MODEL_VERSION)
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model not found at {model_path}!")

@app.get("/")
def home():
    return {"message": "Maternal Health Risk Prediction API is Running!"}

@app.post("/predict/")
def predict(Age: float, SystolicBP: float, DiastolicBP: float, BS: float, BodyTemp: float, HeartRate: float):
    """Make a prediction using input features."""
    features = np.array([[Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate]])
    
    try:
        # Load scaler if available
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features  # Use raw features if no scaler is available

        prediction = model.predict(features_scaled)[0]
        risk_mapping = {0: "low risk", 1: "mid risk", 2: "high risk"}
        return {"Predicted Risk Level": risk_mapping[prediction]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/upload/")
async def upload_data(file: UploadFile = File(...)):
    """Upload a new dataset for retraining."""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        return JSONResponse(content={"message": "File uploaded successfully for retraining.", "file_path": file_path})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain/")
def retrain_model(file_path: str):
    """Retrain the model using the uploaded data and evaluate it."""
    global MODEL_VERSION
    
    try:
        # Load the uploaded data for retraining
        df = pd.read_csv(file_path)

        # Validate columns
        if set(df.columns) != set(EXPECTED_COLUMNS):
            raise HTTPException(status_code=400, detail="Uploaded CSV file does not match the required format.")

        X_new = df.drop(columns=["RiskLevel"])
        y_new = df["RiskLevel"]

        # Split the data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

        # Train a new model
        model_new = RandomForestClassifier(n_estimators=100, random_state=42)
        model_new.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model_new.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        class_report = classification_report(y_test, y_pred, target_names=["Low Risk", "Mid Risk", "High Risk"], output_dict=True)

        # Update model version
        MODEL_VERSION += 1
        new_model_path = MODEL_PATH_TEMPLATE.format(MODEL_VERSION)

        # Save the retrained model
        joblib.dump(model_new, new_model_path)

        return {
            "message": "Model retrained successfully!",
            "model_version": MODEL_VERSION,
            "model_path": new_model_path,
            "evaluation_metrics": {
                "test_metrics": {
                    "accuracy": accuracy,
                    "confusion_matrix": conf_matrix,
                    "classification_report": class_report
                }
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retraining the model: {str(e)}")
