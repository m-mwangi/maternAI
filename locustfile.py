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
            "SystolicBP": random.randint(90, 180),
            "DiastolicBP": random.randint(60, 120),
            "BS": round(random.uniform(3.0, 10.0), 1),
            "BodyTemp": round(random.uniform(36.0, 39.0), 1),
            "HeartRate": random.randint(60, 130)
        }
        response = self.client.post("/predict/", params=data)  # Using query parameters
        print("Predict Response:", response.status_code, response.text)

    @task(1)
    def upload_data(self):
        """Upload the dataset before retraining."""
        file_path = "Maternal Health Risk Data Set.csv"
        
        if os.path.exists(file_path):
            with open(file_path, "rb") as file:
                files = {"file": file}
                response = self.client.post("/upload/", files=files)
                
                if response.status_code == 200:
                    response_json = response.json()
                    if response_json.get("message") == "File uploaded successfully for retraining.":
                        APIUser.file_uploaded = True
                        print("Upload Success:", response.text)
                    else:
                        print("Upload Failed:", response.text)
                else:
                    print("Upload Error:", response.status_code, response.text)
        else:
            print("Error: Dataset file not found!")

    @task(1)
    def retrain(self):
        """Retrain model only if the file was uploaded successfully."""
        if APIUser.file_uploaded:
            response = self.client.post("/retrain/")
            print("Retrain Response:", response.status_code, response.text)
        else:
            print("Skipping retrain: File not uploaded yet.")
