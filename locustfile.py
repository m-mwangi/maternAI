from locust import HttpUser, task, between
import random

class APIUser(HttpUser):
    wait_time = between(1, 3)  # Users wait between 1-3 seconds before making a new request

    @task(3)  # This task will run 3x more often than others
    def predict(self):
        # Example input data (modify according to your API's expected input)
        data = {
            "Age": random.randint(18, 45),
            "SystolicBP": random.randint(90, 180),
            "DiastolicBP": random.randint(60, 120),
            "BS": round(random.uniform(3.0, 10.0), 1),
            "BodyTemp": round(random.uniform(36.0, 39.0), 1),
            "HeartRate": random.randint(60, 130)
        }
        self.client.post("/predict/", json=data)  # Adjust URL if necessary

    @task(1)
    def upload_data(self):
        # Simulating file upload (ensure you have a valid test.csv file)
        files = {'file': open("Maternal Health Risk Data Set.csv", "rb")}
        self.client.post("/upload/", files=files)

    @task(1)
    def retrain(self):
        self.client.post("/retrain/")  # Calls retraining endpoint

