import joblib
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


def predict_aqi(input_data, model_path='random_forest_model.joblib'):
    """
    Loads a pre-trained model and makes a prediction on new data.

    Args:
        input_data (dict or pd.DataFrame): The new data for which you want a prediction.
        model_path (str): The path to the saved model file.

    Returns:
        float: The predicted AQI value, or None if an error occurs.
    """
    # Load the saved model dictionary
    try:
        saved_data = joblib.load(model_path)
        model = saved_data['model']               # Extract the model
        feature_columns = saved_data['feature_columns']  # Extract feature columns
    except FileNotFoundError:
        print(f"Error: The model file '{model_path}' was not found.")
        print("Please run train_model.py first to create the model.")
        return None

    # Ensure input data has all required columns
    input_df = pd.DataFrame([{col: input_data.get(col, 0) for col in feature_columns}])

    # Make a prediction
    prediction = model.predict(input_df)
    return prediction[0]


if __name__ == '__main__':
    # 35 new test data samples
    test_data = [
        {'CO AQI Value': 1, 'Ozone AQI Value': 10, 'NO2 AQI Value': 5, 'PM2.5 AQI Value': 20, 'lat': 35.6895, 'lng': 139.6917},
        {'CO AQI Value': 2, 'Ozone AQI Value': 15, 'NO2 AQI Value': 8, 'PM2.5 AQI Value': 25, 'lat': 55.7558, 'lng': 37.6173},
        {'CO AQI Value': 0, 'Ozone AQI Value': 5, 'NO2 AQI Value': 2, 'PM2.5 AQI Value': 10, 'lat': -33.8688, 'lng': 151.2093},
        {'CO AQI Value': 3, 'Ozone AQI Value': 20, 'NO2 AQI Value': 12, 'PM2.5 AQI Value': 35, 'lat': 48.8566, 'lng': 2.3522},
        {'CO AQI Value': 5, 'Ozone AQI Value': 25, 'NO2 AQI Value': 15, 'PM2.5 AQI Value': 50, 'lat': 52.5200, 'lng': 13.4050},
        {'CO AQI Value': 7, 'Ozone AQI Value': 30, 'NO2 AQI Value': 18, 'PM2.5 AQI Value': 75, 'lat': 19.0760, 'lng': 72.8777},
        {'CO AQI Value': 4, 'Ozone AQI Value': 22, 'NO2 AQI Value': 14, 'PM2.5 AQI Value': 60, 'lat': 31.2304, 'lng': 121.4737},
        {'CO AQI Value': 1, 'Ozone AQI Value': 12, 'NO2 AQI Value': 6, 'PM2.5 AQI Value': 30, 'lat': 41.9028, 'lng': 12.4964},
        {'CO AQI Value': 2, 'Ozone AQI Value': 18, 'NO2 AQI Value': 10, 'PM2.5 AQI Value': 40, 'lat': 43.6532, 'lng': -79.3832},
        {'CO AQI Value': 6, 'Ozone AQI Value': 35, 'NO2 AQI Value': 20, 'PM2.5 AQI Value': 90, 'lat': 23.1291, 'lng': 113.2644},
        {'CO AQI Value': 8, 'Ozone AQI Value': 40, 'NO2 AQI Value': 22, 'PM2.5 AQI Value': 110, 'lat': 39.9042, 'lng': 116.4074},
        {'CO AQI Value': 0, 'Ozone AQI Value': 5, 'NO2 AQI Value': 1, 'PM2.5 AQI Value': 5, 'lat': 59.3293, 'lng': 18.0686},
        {'CO AQI Value': 3, 'Ozone AQI Value': 25, 'NO2 AQI Value': 12, 'PM2.5 AQI Value': 55, 'lat': 34.6937, 'lng': 135.5023},
        {'CO AQI Value': 2, 'Ozone AQI Value': 14, 'NO2 AQI Value': 7, 'PM2.5 AQI Value': 28, 'lat': -23.5505, 'lng': -46.6333},
        {'CO AQI Value': 1, 'Ozone AQI Value': 8, 'NO2 AQI Value': 3, 'PM2.5 AQI Value': 15, 'lat': 37.5665, 'lng': 126.9780},
        {'CO AQI Value': 5, 'Ozone AQI Value': 30, 'NO2 AQI Value': 18, 'PM2.5 AQI Value': 70, 'lat': 13.7563, 'lng': 100.5018},
        {'CO AQI Value': 6, 'Ozone AQI Value': 35, 'NO2 AQI Value': 20, 'PM2.5 AQI Value': 95, 'lat': 22.3964, 'lng': 114.1095},
        {'CO AQI Value': 0, 'Ozone AQI Value': 3, 'NO2 AQI Value': 0, 'PM2.5 AQI Value': 5, 'lat': 64.8378, 'lng': -147.7164},
        {'CO AQI Value': 2, 'Ozone AQI Value': 12, 'NO2 AQI Value': 6, 'PM2.5 AQI Value': 25, 'lat': 45.4215, 'lng': -75.6972},
        {'CO AQI Value': 3, 'Ozone AQI Value': 18, 'NO2 AQI Value': 10, 'PM2.5 AQI Value': 40, 'lat': 50.1109, 'lng': 8.6821},
        {'CO AQI Value': 4, 'Ozone AQI Value': 20, 'NO2 AQI Value': 12, 'PM2.5 AQI Value': 45, 'lat': 25.2048, 'lng': 55.2708},
        {'CO AQI Value': 1, 'Ozone AQI Value': 6, 'NO2 AQI Value': 2, 'PM2.5 AQI Value': 10, 'lat': 55.9533, 'lng': -3.1883},
        {'CO AQI Value': 2, 'Ozone AQI Value': 15, 'NO2 AQI Value': 5, 'PM2.5 AQI Value': 30, 'lat': 37.7749, 'lng': -122.4194},
        {'CO AQI Value': 5, 'Ozone AQI Value': 28, 'NO2 AQI Value': 15, 'PM2.5 AQI Value': 60, 'lat': 41.3851, 'lng': 2.1734},
        {'CO AQI Value': 7, 'Ozone AQI Value': 32, 'NO2 AQI Value': 20, 'PM2.5 AQI Value': 85, 'lat': 31.7683, 'lng': 35.2137},
        {'CO AQI Value': 3, 'Ozone AQI Value': 18, 'NO2 AQI Value': 10, 'PM2.5 AQI Value': 50, 'lat': 59.9139, 'lng': 10.7522},
        {'CO AQI Value': 6, 'Ozone AQI Value': 35, 'NO2 AQI Value': 22, 'PM2.5 AQI Value': 100, 'lat': 19.4326, 'lng': -99.1332},
        {'CO AQI Value': 1, 'Ozone AQI Value': 5, 'NO2 AQI Value': 2, 'PM2.5 AQI Value': 15, 'lat': 1.3521, 'lng': 103.8198},
        {'CO AQI Value': 2, 'Ozone AQI Value': 12, 'NO2 AQI Value': 8, 'PM2.5 AQI Value': 35, 'lat': 52.3676, 'lng': 4.9041},
        {'CO AQI Value': 4, 'Ozone AQI Value': 25, 'NO2 AQI Value': 15, 'PM2.5 AQI Value': 55, 'lat': 55.7558, 'lng': 37.6173},
        {'CO AQI Value': 0, 'Ozone AQI Value': 4, 'NO2 AQI Value': 1, 'PM2.5 AQI Value': 8, 'lat': -36.8485, 'lng': 174.7633},
        {'CO AQI Value': 3, 'Ozone AQI Value': 22, 'NO2 AQI Value': 12, 'PM2.5 AQI Value': 60, 'lat': 21.0285, 'lng': 105.8542},
        {'CO AQI Value': 5, 'Ozone AQI Value': 30, 'NO2 AQI Value': 20, 'PM2.5 AQI Value': 80, 'lat': 30.0444, 'lng': 31.2357},
        {'CO AQI Value': 6, 'Ozone AQI Value': 35, 'NO2 AQI Value': 25, 'PM2.5 AQI Value': 95, 'lat': 28.6139, 'lng': 77.2090},
        {'CO AQI Value': 2, 'Ozone AQI Value': 10, 'NO2 AQI Value': 5, 'PM2.5 AQI Value': 25, 'lat': 43.7696, 'lng': 11.2558}
    ]

    print("\n--- AQI Prediction Batch Test (35 samples) ---")

    # Loop through each data sample and print the prediction
    for i, data_point in enumerate(test_data):
        predicted_value = predict_aqi(data_point)
        if predicted_value is not None:
            print(f"\nTest Sample #{i+1}")
            print(f"Input Data: {data_point}")
            print(f"Predicted AQI Value: {predicted_value:.2f}")
            print("---------------------------------")
