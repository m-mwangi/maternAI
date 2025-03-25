import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from preprocessing import load_data, preprocess_data, split_and_balance_data

def train_model(X_train, y_train):
    """Trains a RandomForest model and returns the trained model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test):
    """Evaluates the model's accuracy and displays a confusion matrix."""
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    val_accuracy = accuracy_score(y_val, model.predict(X_val))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    print("\nModel Performance Summary:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("\nClassification Report:\n", classification_report(y_test, model.predict(X_test), zero_division=1))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, model.predict(X_test))
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def save_model(model, scaler, model_dir="models"):
    """Saves the trained model and scaler inside the matern_ai/model directory."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "random_forest_maternal_health.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model and scaler saved successfully in {model_dir}!")

if __name__ == "__main__":
    dataset_path = "Maternal Health Risk Data Set.csv"
    df = load_data(dataset_path)
    df_cleaned = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_balance_data(df_cleaned)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test)
    save_model(model, scaler)
