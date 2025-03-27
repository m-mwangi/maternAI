# MaternAI

## Project Description
MaternAI is a machine learning-powered application designed to predict maternal health risks. The app classifies patients into different risk levels: High Risk, Mid Risk, and Low Risk. It features a Flask-based web application where users can input patient data for risk predictions, upload new data for model retraining, and explore various dataset visualizations.

## Features
- **Prediction Page**: Allows users to input patient data and receive risk predictions.
- **Dashboard**: Visualizations showcasing insights from the dataset.
- **Data Upload**: Users can upload new data to the system for model retraining.
- **Retrain Model**: A trigger to retrain the machine learning model based on new data.
- **Flood Simulation**: Tests system performance under a high load of prediction requests using Locust.
- **Dockerized Deployment**: The app is fully containerized for easy deployment and scaling.

## Video Demo and Live Link to App
- **Video Demo**: [YouTube Link]
- **Live App**: [Deployed URL]
- **Docker Image**: [Docker Hub Link]
- **Deployed API (Swagger UI)**: [Swagger UI Link]

## Navigating the Deployed App
The deployed app consists of five key sections:

1. **Home**: An overview of the project, including its purpose and features.
2. **Preprocess**: Details the preprocessing steps applied to the dataset before training the model.
3. **Visualizations**: Provides insights into the dataset's features and their impact on maternal health risk predictions.
4. **Retrain**: Users can upload a CSV file to retrain the model. The file should contain all necessary feature columns. Navigate to `data/uploads` to download sample files, rename them, and upload them. After uploading, click the **Retrain** button. Upon retraining, a success message with the new model version will appear. 
   - Evaluation metrics can be viewed via the **Swagger UI** link above.
5. **Predict**: Input feature values and click **Predict** to receive risk level predictions. This page also explains the meaning of each prediction and includes usage instructions.

🚨 **Note:** Due to server inactivity, requests for retraining and prediction may take some time. We appreciate your patience.

## Docker Image Deployment
MaternAI is deployed on a cloud platform, but you can also run it locally using Docker:

1. Pull the Docker image:
   ```bash
   docker pull [your_dockerhub_username]/maternai:latest
   ```
2. Run the container:
   ```bash
   docker run -p 5000:5000 maternai:latest
   ```
3. Access the app locally at:
   ```
   http://localhost:5000
   ```

## Setting Up Locally
To run the project locally:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/[your_github_username]/maternai.git
   cd maternai
   ```
2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Preprocessing
The preprocessing script is located at `src/preprocessing.py`.
To run:
```bash
python src/preprocessing.py
```

## Model Training
The model training script is located at `src/model.py`.
To train:
```bash
python src/model.py
```

## Prediction
The prediction script is located at `src/prediction.py`.
To make predictions:
```bash
python src/prediction.py
```

## Notebook
The Jupyter notebook contains all preprocessing, model training, and visualization functions.
To use:
```bash
pip install jupyter
jupyter notebook
```
Open the notebook from the Jupyter interface.

## Results from Flood Request Simulation
To test system performance under heavy load using Locust:
```bash
locust -f locustfile.py --host=[your_deployed_url]
```
Access the Locust interface via the link in your terminal, configure the user load, and analyze the response times.

## Frontend Repository
[Frontend GitHub Repository Link]

## Contribution Guidelines
Feel free to fork the repository and submit pull requests for improvements or bug fixes. 🚀
