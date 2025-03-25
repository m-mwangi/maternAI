import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    """Loads dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Handles missing values, encodes labels, and removes outliers."""
    # Encode the target variable
    df["RiskLevel"] = df["RiskLevel"].replace({"high risk": 2, "mid risk": 1, "low risk": 0})

    # Handle outliers using the IQR method
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    df_no_outliers = df.copy()
    for column in df.columns:
        lower_bound = Q1[column] - 1.5 * IQR[column]
        upper_bound = Q3[column] + 1.5 * IQR[column]
        df_no_outliers[column] = np.where(df[column] < lower_bound, lower_bound, df_no_outliers[column])
        df_no_outliers[column] = np.where(df[column] > upper_bound, upper_bound, df_no_outliers[column])

    return df_no_outliers

def split_and_balance_data(df):
    """Splits dataset into training, validation, and test sets, and applies SMOTE."""
    X = df.drop('RiskLevel', axis=1)
    y = df['RiskLevel']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Apply SMOTE for class balancing
    smote = SMOTE(sampling_strategy={0: 406}, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Scale the features
    scaler = StandardScaler()
    X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_resampled_scaled, X_val_scaled, X_test_scaled, y_train_resampled, y_val, y_test, scaler

if __name__ == "__main__":
    dataset_path = "Maternal Health Risk Data Set.csv"
    df = load_data(dataset_path)
    df_cleaned = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_balance_data(df_cleaned)

    print("Preprocessing complete. Data is ready for training.")