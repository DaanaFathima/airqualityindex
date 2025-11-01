import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


def calculate_accuracy(y_true, y_pred, tolerance=0.10):
    """Custom accuracy within a given percentage tolerance"""
    percentage_error = np.abs((y_true - y_pred) / y_true)
    correct_predictions = percentage_error <= tolerance
    return np.mean(correct_predictions) * 100


def train_model(data_path='AirDataset.csv', model_path='random_forest_model.joblib'):
    """Load data, preprocess, train RandomForest model, evaluate, and save"""
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The file '{data_path}' was not found.")
        return None, None

    # --- Preprocessing ---
    df.drop_duplicates(inplace=True)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Handle outliers with IQR method
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)

    # Features & target
    if 'AQI Value' not in df.columns:
        raise ValueError("Dataset must contain 'AQI Value' column as target")
    X = df.drop('AQI Value', axis=1)
    y = df['AQI Value']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest
    model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True, max_features='sqrt')
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = calculate_accuracy(y_test.values, y_pred)
    print("Model Evaluation Metrics:")
    print(f"  MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"  MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"  R2: {r2_score(y_test, y_pred):.2f}")
    print(f"  OOB Score: {model.oob_score_:.2f}")
    print(f"  Accuracy (within 10%): {accuracy:.2f}%")

    # Save model and feature columns
    joblib.dump({'model': model, 'feature_columns': X.columns.tolist()}, model_path)
    print(f"\nModel saved to {model_path}")

    return model, X.columns.tolist()


def predict_sample(model_path='random_forest_model.joblib', sample_values=None):
    """
    Load the trained model and predict AQI for a sample input.
    `sample_values` should be a dictionary with keys matching the feature columns.
    """
    try:
        saved_data = joblib.load(model_path)
        model = saved_data['model']
        feature_columns = saved_data['feature_columns']
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        return
    except KeyError:
        print(f"Error: Model file '{model_path}' is corrupted.")
        return

    # If no sample values provided, use default zeros
    if sample_values is None:
        sample_values = {col: 0 for col in feature_columns}

    # Make sure the sample input has all required columns
    sample_input = pd.DataFrame([{col: sample_values.get(col, 0) for col in feature_columns}])

    predicted_aqi = model.predict(sample_input)[0]
    print("\nSample Input Prediction:")
    print(sample_input)
    print(f"\nPredicted AQI: {predicted_aqi:.2f}")


if __name__ == '__main__':
    # Train model
    train_model()

    # Example: Predict using a sample input
    sample_values = {
        'PM2.5 AQI Value': 85,
        'PM10 AQI Value': 140,
        'NO2 AQI Value': 55,
        'Ozone AQI Value': 50,
        'CO AQI Value': 1.0,
        'Temperature': 30.0,
        'Humidity': 70,
        'WindSpeed': 3.0,
        'Pressure': 1010,
        'lat': 18.5,
        'lon': 73.9
    }
    predict_sample(sample_values=sample_values)
