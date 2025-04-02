# MaternAI

## Project Description
Maternal health complications are key challenges in many developing African countries, contributing to high mortality rates from preventable conditions. This project will develop an ML pipeline, scale it effectively, and monitor its performance on a cloud platform of your choice.

MaternAI is an ML-powered application designed to predict maternal health risks. It classifies patients into different risk levels: High Risk, Mid Risk, and Low Risk. 
It contains a web application based on Flask where users can input different values to predict risks. It also has a section for model retraining and visualization insights that can help one understand the dataset fully.

More information about my proposal project can be found here: https://docs.google.com/document/d/1xnnj8wq3rHsqiU6VgBP0DfMDTO1EZ9WXXSz1G3YgdLc/edit?usp=sharing

## About the Dataset
The dataset used for this model is publicly available on Kaggle:
🔗 [Maternal Health Risk Dataset](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data/data)  

### Key Variables:
- Age - This is the age in years when a woman is pregnant.
- SystolicBP - The upper value of Blood Pressure in mmHg.
- DiastolicBP - Lower value of Blood Pressure in mmHg.
- BS (Blood Sugar Level) - Blood sugar levels in terms of molar concentration, mmol/L.
- HeartRate - A normal resting heart rate in beats per minute.
- Risk Level - Predicted Risk Intensity Level during pregnancy considering the previous attributes. Categorized as either 'High Risk', 'Low Risk', or 'Mid Risk'.

## Features/Navigating My Deployed Web Application
The web app consists of five key sections:
1. **Home** - An overview of the project, including its purpose and features.
2. **Preprocess** - Details the preprocessing steps I applied before model training and building.
3. **Visualizations** - Provides detailed insights into some of the dataset's features and their impact on maternal health risk predictions.
4. **Retrain** - This section allows users to upload any new CSV file to enable model retraining. The uploaded file should match the feature columns in my original dataset. To test this feature, navigate to `data/uploads` to download sample files, rename them, and upload them. After uploading, click the **Retrain** button. Upon retraining, a success message with the new model version and accuracy will appear. 
5. **Predict** - Input feature values and click **Predict** to receive risk level predictions. The page also explains the meaning of each prediction, range of values that can be inputted as well as usage instructions.

## Demo Video Presentation:
https://youtu.be/Uqx17yIzydI

## Live Web App:
https://matern-ai-front-end.onrender.com

## Deployed API (SwaggerUI):
https://matern-ai-1.onrender.com

## Frontend Repository
[https://github.com/m-mwangi/maternAI_front-end]

## Docker Image Deployment
**Docker Image** : mmwangi2/maternai:latest

MaternAI is deployed on a cloud platform, but you can also run it locally using Docker:

1. Pull the Docker image:
   ```bash
   docker pull mmwangi2/maternai:latest
   ```
2. Run the container:
   ```bash
   docker run -p 5000:5000 maternai:latest
   ```
3. Access the app locally at:
   ```
   http://localhost:5000
   ```

## Setting Up Project Locally
To run the project locally:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/m-mwangi/maternAI.git
   cd maternAI
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

### Preprocessing
The preprocessing script is located at `src/preprocessing.py`.
To run:
```bash
python src/preprocessing.py
```

### Model Training
The model training script is located at `src/model.py`.
To train:
```bash
python src/model.py
```

### Prediction
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
locust -f locustfile.py --host=https://matern-ai-1.onrender.com
```
Access the Locust interface via the link in your terminal, configure the user load, and analyze the response times.

![Screenshot (18)](https://github.com/user-attachments/assets/8f33a894-71c5-4d6a-8745-0e59f5af6109)
