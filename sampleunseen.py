import joblib
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


def predict_aqi(input_data, model_path='random_forest_model.joblib'):
    """Predict AQI using saved Random Forest model."""
    try:
        saved_data = joblib.load(model_path)
        model = saved_data['model']
        feature_columns = saved_data['feature_columns']
    except FileNotFoundError:
        print(f"Model file '{model_path}' not found.")
        return None

    input_df = pd.DataFrame([{col: input_data.get(col, 0) for col in feature_columns}])
    prediction = model.predict(input_df)
    return prediction[0]


if __name__ == '__main__':
  
    unseen_test_data = [
        {'CO AQI Value': 0, 'Ozone AQI Value': 5, 'NO2 AQI Value': 0, 'PM2.5 AQI Value': 12, 'lat': 60.1699, 'lng': 24.9384},   # Helsinki
        {'CO AQI Value': 1, 'Ozone AQI Value': 18, 'NO2 AQI Value': 7, 'PM2.5 AQI Value': 35, 'lat': 45.4642, 'lng': 9.1900},    # Milan
        {'CO AQI Value': 3, 'Ozone AQI Value': 20, 'NO2 AQI Value': 10, 'PM2.5 AQI Value': 50, 'lat': 33.8688, 'lng': 151.2093}, # Sydney
        {'CO AQI Value': 2, 'Ozone AQI Value': 10, 'NO2 AQI Value': 5, 'PM2.5 AQI Value': 28, 'lat': 49.2827, 'lng': -123.1207}, # Vancouver
        {'CO AQI Value': 4, 'Ozone AQI Value': 25, 'NO2 AQI Value': 15, 'PM2.5 AQI Value': 65, 'lat': 25.276987, 'lng': 55.296249}, # Dubai
        {'CO AQI Value': 6, 'Ozone AQI Value': 35, 'NO2 AQI Value': 20, 'PM2.5 AQI Value': 90, 'lat': 31.5204, 'lng': 74.3587},    # Lahore
        {'CO AQI Value': 0, 'Ozone AQI Value': 3, 'NO2 AQI Value': 0, 'PM2.5 AQI Value': 8, 'lat': 64.8378, 'lng': -147.7164},   # Fairbanks
        {'CO AQI Value': 2, 'Ozone AQI Value': 12, 'NO2 AQI Value': 6, 'PM2.5 AQI Value': 30, 'lat': 55.9533, 'lng': -3.1883},    # Edinburgh
        {'CO AQI Value': 5, 'Ozone AQI Value': 28, 'NO2 AQI Value': 18, 'PM2.5 AQI Value': 80, 'lat': 19.4326, 'lng': -99.1332},   # Mexico City
        {'CO AQI Value': 1, 'Ozone AQI Value': 7, 'NO2 AQI Value': 3, 'PM2.5 AQI Value': 15, 'lat': 52.5200, 'lng': 13.4050},     # Berlin
        {'CO AQI Value': 3, 'Ozone AQI Value': 22, 'NO2 AQI Value': 12, 'PM2.5 AQI Value': 55, 'lat': 39.9042, 'lng': 116.4074},   # Beijing
        {'CO AQI Value': 2, 'Ozone AQI Value': 14, 'NO2 AQI Value': 8, 'PM2.5 AQI Value': 35, 'lat': -23.5505, 'lng': -46.6333},  # Sao Paulo
        {'CO AQI Value': 4, 'Ozone AQI Value': 30, 'NO2 AQI Value': 18, 'PM2.5 AQI Value': 70, 'lat': 13.7563, 'lng': 100.5018},   # Bangkok
        {'CO AQI Value': 6, 'Ozone AQI Value': 40, 'NO2 AQI Value': 22, 'PM2.5 AQI Value': 100, 'lat': 19.0759, 'lng': 72.8777},   # Mumbai
        {'CO AQI Value': 0, 'Ozone AQI Value': 2, 'NO2 AQI Value': 0, 'PM2.5 AQI Value': 5, 'lat': 59.3293, 'lng': 18.0686},      # Stockholm
        {'CO AQI Value': 1, 'Ozone AQI Value': 8, 'NO2 AQI Value': 4, 'PM2.5 AQI Value': 18, 'lat': 48.1351, 'lng': 11.5820},      # Munich
        {'CO AQI Value': 3, 'Ozone AQI Value': 20, 'NO2 AQI Value': 10, 'PM2.5 AQI Value': 50, 'lat': 31.2304, 'lng': 121.4737},   # Shanghai
        {'CO AQI Value': 2, 'Ozone AQI Value': 12, 'NO2 AQI Value': 5, 'PM2.5 AQI Value': 30, 'lat': 37.7749, 'lng': -122.4194},   # San Francisco
        {'CO AQI Value': 5, 'Ozone AQI Value': 28, 'NO2 AQI Value': 15, 'PM2.5 AQI Value': 75, 'lat': 41.8781, 'lng': -87.6298},    # Chicago
        {'CO AQI Value': 7, 'Ozone AQI Value': 38, 'NO2 AQI Value': 20, 'PM2.5 AQI Value': 110, 'lat': 28.7041, 'lng': 77.1025}     # Delhi
    ]

    print("\n--- AQI Prediction Batch Test (20 unseen samples) ---")

    # Loop through each sample and predict
    for i, data_point in enumerate(unseen_test_data):
        predicted_value = predict_aqi(data_point)
        if predicted_value is not None:
            print(f"\nTest Sample #{i+1}")
            print(f"Input Data: {data_point}")
            print(f"Predicted AQI Value: {predicted_value:.2f}")
            print("---------------------------------")
