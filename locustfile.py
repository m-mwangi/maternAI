from locust import HttpUser, task, between
import random
import os

class APIUser(HttpUser):
    wait_time = between(1, 3)  # Users wait between 1-3 seconds before making a new request
    file_uploaded = False  # Track if file upload was successful

    @task(3)  # This task will run 3x more often than others
    def predict(self):
        """Send a prediction request with random test data."""
        data = {
            "Age": random.randint(18, 45),
            "SystolicBp": random.randint(90, 180),
            "DiastolicBp": random.randint(60, 120),
            "BS": round(random.uniform(3.0, 10.0), 1),
            "BodyTemp": round(random.uniform(36.0, 39.0), 1),
            "HeartRate": random.randint(60, 130)
        }
        response = self.client.post("/predict/", params=data)  # Use params, NOT json
        print("Predict Response:", response.text)

    @task(1)
    def upload_data(self):
        """Upload the dataset before retraining."""
        if os.path.exists("Maternal Health Risk Data Set.csv"):
            with open("Maternal Health Risk Data Set.csv", "rb") as file:
                files = {"file": file}
                response = self.client.post("/upload/", files=files)
                print("Upload Response:", response.text)
                if response.status_code == 200:
                    APIUser.file_uploaded = True  # Mark file as uploaded
        else:
            print("Error: Dataset file not found!")

    @task(1)
    def retrain(self):
        """Retrain model only if the file was uploaded successfully."""
        if APIUser.file_uploaded:
            response = self.client.post("/retrain/")
            print("Retrain Response:", response.text)
        else:
            print("Skipping retrain: File not uploaded yet.")
