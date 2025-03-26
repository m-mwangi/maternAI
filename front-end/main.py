from flask import Flask, render_template, request
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Configure upload & model folders
UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MODEL_FOLDER"] = MODEL_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Home Page
@app.route('/')
def home():
    return render_template('home.html')

# Preprocessing Page
@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('preprocessing.html', message="No file selected.")

        file = request.files['file']
        if file.filename == '':
            return render_template('preprocessing.html', message="No file selected.")

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        df = pd.read_csv(filepath)
        num_rows, num_cols = df.shape
        columns = df.columns.tolist()
        missing_values = df.isnull().sum().to_dict()

        return render_template(
            'preprocessing.html',
            message="File uploaded successfully!",
            num_rows=num_rows,
            num_cols=num_cols,
            columns=columns,
            missing_values=missing_values
        )
    return render_template('preprocessing.html')

# Load latest dataset
def load_latest_data():
    files = os.listdir(app.config["UPLOAD_FOLDER"])
    if not files:
        return None
    latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(app.config["UPLOAD_FOLDER"], f)))
    return pd.read_csv(os.path.join(app.config["UPLOAD_FOLDER"], latest_file))

# Generate plots
def generate_plots():
    df = load_latest_data()
    if df is None:
        return None, "No dataset uploaded yet."

    numeric_df = df.select_dtypes(include=['number'])

    fig, ax = plt.subplots()
    numeric_columns = numeric_df.columns.tolist()
    if numeric_columns:
        sns.histplot(numeric_df[numeric_columns[0]], kde=True, bins=30, ax=ax)
        ax.set_title(f"Distribution of {numeric_columns[0]}")
    else:
        ax.set_title("No Numeric Columns Found")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot1_url = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot2_url = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()
    else:
        plot2_url = None

    fig, ax = plt.subplots()
    if numeric_columns:
        sns.boxplot(x=numeric_df[numeric_columns[0]], ax=ax)
        ax.set_title(f"Boxplot of {numeric_columns[0]}")
    else:
        ax.set_title("No Numeric Columns Found")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot3_url = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    return [plot1_url, plot2_url, plot3_url], None

# Visualizations Page
@app.route('/visualizations')
def visualizations():
    plots, error = generate_plots()
    return render_template('visualizations.html', plots=plots, error=error)

# Retraining Page
@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('retrain.html', message="No file selected.")

        file = request.files['file']
        if file.filename == '':
            return render_template('retrain.html', message="No file selected.")

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)

            # Ensure correct feature names
            expected_columns = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate', 'RiskLevel']
            if not all(col in df.columns for col in expected_columns):
                return render_template('retrain.html', message="Invalid dataset format. Ensure correct column names.")

            # Encode target variable
            label_encoder = LabelEncoder()
            df['RiskLevel'] = label_encoder.fit_transform(df['RiskLevel'])

            # Split data
            X = df.drop(columns=['RiskLevel'])
            y = df['RiskLevel']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Save model
            model_path = os.path.join(app.config["MODEL_FOLDER"], "model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            return render_template('retrain.html', message="Model retrained successfully!", model_saved=True)

        except Exception as e:
            return render_template('retrain.html', message=f"Error: {str(e)}")

    return render_template('retrain.html', message=None)

# Prediction Page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    model_path = os.path.join(app.config["MODEL_FOLDER"], "model.pkl")

    if not os.path.exists(model_path):
        return render_template('predict.html', result="Model not found. Please retrain first.")

    # Load the trained model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    if request.method == 'POST':
        try:
            # Extract input values
            age = float(request.form['age'])
            systolic_bp = float(request.form['systolic_bp'])
            diastolic_bp = float(request.form['diastolic_bp'])
            bs = float(request.form['bs'])
            body_temp = float(request.form['body_temp'])
            heart_rate = float(request.form['heart_rate'])

            # Create DataFrame for prediction
            input_data = pd.DataFrame([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]],
                                      columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'])

            # Make prediction
            prediction = model.predict(input_data)[0]

            # Convert prediction to label
            risk_labels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
            prediction_text = risk_labels.get(prediction, "Unknown Risk Level")

            return render_template('predict.html', result=f"Predicted Maternal Risk: {prediction_text}")

        except Exception as e:
            return render_template('predict.html', result=f"Error: {str(e)}")

    return render_template('predict.html', result=None)

# Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)
