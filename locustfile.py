from locust import HttpUser, task, between
import random
import os

class APIUser(HttpUser):
    wait_time = between(1, 3)  # Users wait between 1-3 seconds before making a new request

    @task(3)  # This task will run 3x more often than others
    def predict(self):
        """Send a prediction request with random test data."""
        data = {
            "Age": random.randint(18, 45),
            "SystolicBP": random.randint(90, 180),  # Fixed key name
            "DiastolicBP": random.randint(60, 120),  # Fixed key name
            "BS": round(random.uniform(3.0, 10.0), 1),
            "BodyTemp": round(random.uniform(36.0, 39.0), 1),
            "HeartRate": random.randint(60, 130)
        }
        response = self.client.post("/predict/", json=data)  # Use JSON instead of params
        print("Predict Response:", response.text)

    @task(1)
    def upload_data(self):
        """Upload the dataset before retraining."""
        if os.path.exists("Maternal Health Risk Data Set.csv"):
            with open("Maternal Health Risk Data Set.csv", "rb") as file:
                files = {"file": file}
                response = self.client.post("/upload/", files=files)
                print("Upload Response:", response.text)
        else:
            print("Error: Dataset file not found!")

    @task(1)
    def retrain(self):
        """Check if file is uploaded and retrain model if available."""
        check_response = self.client.get("/check_upload/")  # New check endpoint
        if check_response.status_code == 200 and check_response.json().get("file_uploaded"):
            response = self.client.post("/retrain/")
            print("Retrain Response:", response.text)
        else:
            print("Skipping retrain: File not uploaded yet.")
