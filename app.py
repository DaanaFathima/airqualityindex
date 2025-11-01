from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Path to your saved joblib file
MODEL_PATH = "random_forest_model.joblib"

model = None
feature_columns = None
scaler = None

# Try to load the saved model. The file may contain either:
# 1) a dict with keys: 'model', 'feature_columns', optional 'scaler'
# 2) or the raw estimator object itself
if os.path.exists(MODEL_PATH):
    saved = joblib.load(MODEL_PATH)
    if isinstance(saved, dict) and 'model' in saved:
        model = saved['model']
        feature_columns = saved.get('feature_columns')
        scaler = saved.get('scaler')
    else:
        model = saved

# Default feature order to use if the joblib didn't include 'feature_columns'
if feature_columns is None:
    feature_columns = [
        "CO AQI Value",
        "Ozone AQI Value",
        "NO2 AQI Value",
        "PM2.5 AQI Value",
        "lat",
        "lng"
    ]

# Accept many possible form / JSON field names and map them to model column names
INPUT_ALIASES = {
    # HTML names you used in your template (some include dots/underscores)
    "PM2.5_AQI_Value": "PM2.5 AQI Value",
    "PM2.5 AQI Value": "PM2.5 AQI Value",
    "pm25": "PM2.5 AQI Value",

    "NO2_AQI_Value": "NO2 AQI Value",
    "NO2 AQI Value": "NO2 AQI Value",
    "no2": "NO2 AQI Value",

    "Ozone_AQI_Value": "Ozone AQI Value",
    "Ozone AQI Value": "Ozone AQI Value",
    "o3": "Ozone AQI Value",
    "ozone": "Ozone AQI Value",

    "CO_AQI_Value": "CO AQI Value",
    "CO AQI Value": "CO AQI Value",
    "co": "CO AQI Value",

    # lat / lon variations
    "lat": "lat",
    "latitude": "lat",
    "lng": "lng",
    "lon": "lng",
    "longitude": "lng",
}


def get_aqi_status(aqi: float) -> str:
    """Return human-friendly AQI category for a numeric AQI value."""
    try:
        aqi = float(aqi)
    except Exception:
        return "Unknown"

    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def build_input_dataframe(raw: dict) -> pd.DataFrame:
    """Convert a raw dict (from form or JSON) into a single-row DataFrame
    with the columns in the same order as feature_columns. Missing values
    default to 0.0 (this mirrors your current approach but can be changed).
    """
    row = {}
    for col in feature_columns:
        row[col] = 0.0

    # Normalize input keys using INPUT_ALIASES mapping
    for k, v in raw.items():
        if k in INPUT_ALIASES:
            mapped = INPUT_ALIASES[k]
        else:
            # try to match ignoring case and punctuation
            lookup = k.replace('.', '').replace('_', '').replace(' ', '').lower()
            mapped = None
            for alias, target in INPUT_ALIASES.items():
                normal = alias.replace('.', '').replace('_', '').replace(' ', '').lower()
                if normal == lookup:
                    mapped = target
                    break
            if mapped is None:
                # maybe the user already sent the exact model column name
                if k in feature_columns:
                    mapped = k

        if mapped and mapped in row:
            try:
                row[mapped] = float(raw[k])
            except Exception:
                # leave default 0.0 if conversion fails
                pass

    df = pd.DataFrame([row], columns=feature_columns)
    return df


@app.route("/")
def home():
    # Renders your provided index.html form; keep the same filename in templates folder
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", aqi=None, status="Error: model file not found or failed to load.")

    # Build a raw dict from the form. Flask's request.form contains strings.
    raw = {}
    for key in request.form.keys():
        raw[key] = request.form.get(key)

    # Also support common JSON body names if someone sends JSON
    if request.is_json:
        try:
            raw_json = request.get_json()
            if isinstance(raw_json, dict):
                raw.update(raw_json)
        except Exception:
            pass

    try:
        input_df = build_input_dataframe(raw)

        # apply scaler if present
        if scaler is not None:
            X = scaler.transform(input_df)
        else:
            X = input_df

        pred = float(model.predict(X)[0])
        status = get_aqi_status(pred)

        # if any of the original form fields were missing, show a note
        missing_fields = [k for k in feature_columns if float(input_df.loc[0, k]) == 0.0]
        note = ""
        if missing_fields:
            note = f" (note: some model inputs were 0.0 — check your form values)"

        return render_template("index.html", aqi=round(pred, 2), status=status + note)

    except Exception as e:
        return render_template("index.html", aqi=None, status=f"Error: {str(e)}")


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Accepts a JSON list of input dictionaries and returns JSON predictions.
    This is useful for your batch test data (the 35-sample list). Each item in
    the incoming list can be in the same shape as your training columns (e.g.
    {'CO AQI Value': 1, 'Ozone AQI Value': 10, ...}) or use the HTML field names.
    """
    if model is None:
        return jsonify({'error': 'model not loaded'}), 500

    if not request.is_json:
        return jsonify({'error': 'send a JSON list of input dicts'}), 400

    payload = request.get_json()
    if not isinstance(payload, list):
        return jsonify({'error': 'expected a JSON list of dicts'}), 400

    results = []
    for item in payload:
        try:
            df = build_input_dataframe(item)
            X = scaler.transform(df) if scaler is not None else df
            pred = float(model.predict(X)[0])
            results.append({'input': item, 'predicted_aqi': round(pred, 2), 'status': get_aqi_status(pred)})
        except Exception as e:
            results.append({'input': item, 'error': str(e)})

    return jsonify(results)


if __name__ == '__main__':
    # set debug=False for production
    app.run(debug=True)
