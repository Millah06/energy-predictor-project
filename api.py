"""
Smart Energy Consumption Predictor
File: api.py

Purpose: Flask REST API for energy consumption predictions
Author: AbdullahiAliyu Garba
Date: December 2025

Run with: python api.py
API Docs: http://localhost:5000/docs
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FLASK APP SETUP
# ============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and data
model_data = None
df_historical = None
feature_cols = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_model():
    """Load trained model and historical data."""
    global model_data, df_historical, feature_cols

    try:
        logger.info("Loading model and data...")

        # Load model (try XGBoost first, then Random Forest)
        model_file = None
        for model_file_path in ['models/xgboost_model.pkl', 'models/randomforest_model.pkl']:
            try:
                with open(model_file_path, 'rb') as f:
                    model_data = pickle.load(f)
                model_file = model_file_path
                break
            except FileNotFoundError:
                continue
        
        if model_data is None:
            raise FileNotFoundError("No trained model found. Please run: python main.py")

        # Load featured data
        df_historical = pd.read_csv('data/processed/energy_data_featured.csv')
        df_historical['timestamp'] = pd.to_datetime(df_historical['timestamp'])

        # Feature names
        feature_cols = [col for col in df_historical.columns
                        if col not in ['timestamp', 'energy_kwh']]

        logger.info(f"✓ Model loaded successfully with {len(feature_cols)} features")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        return False


def create_features_from_input(data: Dict, prediction_time: datetime) -> Dict:
    """Create all required features from input data."""

    features = {}

    # Time features
    features['hour'] = prediction_time.hour
    features['day_of_week'] = prediction_time.weekday()
    features['day_of_month'] = prediction_time.day
    features['month'] = prediction_time.month
    features['quarter'] = (prediction_time.month - 1) // 3 + 1
    features['is_weekend'] = 1 if prediction_time.weekday() >= 5 else 0
    features['is_business_hours'] = 1 if 9 <= prediction_time.hour <= 17 else 0
    features['is_peak_hours'] = 1 if 14 <= prediction_time.hour <= 20 else 0
    features['season'] = ((prediction_time.month % 12 + 3) // 3) % 4

    # Cyclical encoding
    features['hour_sin'] = np.sin(2 * np.pi * prediction_time.hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * prediction_time.hour / 24)
    features['month_sin'] = np.sin(2 * np.pi * prediction_time.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * prediction_time.month / 12)
    features['dow_sin'] = np.sin(2 * np.pi * prediction_time.weekday() / 7)
    features['dow_cos'] = np.cos(2 * np.pi * prediction_time.weekday() / 7)

    # Weather features
    temp = data.get('temperature', 22.0)
    humidity = data.get('humidity', 60.0)

    features['temperature_c'] = temp
    features['humidity_percent'] = humidity
    features['temperature_squared'] = temp ** 2
    features['temp_humidity_index'] = temp * (humidity / 100)
    features['is_extreme_cold'] = 1 if temp < 5 else 0
    features['is_extreme_hot'] = 1 if temp > 30 else 0
    features['is_extreme_temp'] = 1 if (temp < 5 or temp > 30) else 0

    # Lag and rolling features (use provided or historical data)
    lags = [1, 2, 3, 6, 12, 24, 48, 168]

    # If lags provided in input, use them
    if 'lag_features' in data:
        for lag in lags:
            key = f'energy_kwh_lag_{lag}h'
            features[key] = data['lag_features'].get(str(lag), 2.5)
    else:
        # Use historical data
        df_hist = df_historical[df_historical['timestamp'] < prediction_time]
        if len(df_hist) > 0:
            for lag in lags:
                lag_time = prediction_time - timedelta(hours=lag)
                closest = df_hist.iloc[(df_hist['timestamp'] - lag_time).abs().argsort()[:1]]
                features[f'energy_kwh_lag_{lag}h'] = closest['energy_kwh'].values[0] if len(closest) > 0 else 2.5
        else:
            for lag in lags:
                features[f'energy_kwh_lag_{lag}h'] = 2.5

    # Rolling features
    windows = [6, 12, 24, 168]

    if 'rolling_features' in data:
        for window in windows:
            features[f'energy_kwh_rolling_mean_{window}h'] = data['rolling_features'].get(f'mean_{window}', 2.5)
            features[f'energy_kwh_rolling_std_{window}h'] = data['rolling_features'].get(f'std_{window}', 0.5)
            if window >= 24:
                features[f'energy_kwh_rolling_min_{window}h'] = data['rolling_features'].get(f'min_{window}', 1.5)
                features[f'energy_kwh_rolling_max_{window}h'] = data['rolling_features'].get(f'max_{window}', 4.0)
    else:
        # Use historical data
        df_hist = df_historical[df_historical['timestamp'] < prediction_time]
        if len(df_hist) > 0:
            recent = df_hist.tail(max(windows))
            for window in windows:
                window_data = recent.tail(window)['energy_kwh']
                features[f'energy_kwh_rolling_mean_{window}h'] = window_data.mean()
                features[f'energy_kwh_rolling_std_{window}h'] = window_data.std()
                if window >= 24:
                    features[f'energy_kwh_rolling_min_{window}h'] = window_data.min()
                    features[f'energy_kwh_rolling_max_{window}h'] = window_data.max()
        else:
            for window in windows:
                features[f'energy_kwh_rolling_mean_{window}h'] = 2.5
                features[f'energy_kwh_rolling_std_{window}h'] = 0.5
                if window >= 24:
                    features[f'energy_kwh_rolling_min_{window}h'] = 1.5
                    features[f'energy_kwh_rolling_max_{window}h'] = 4.0

    return features


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def home():
    """Home page with API documentation."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Energy Prediction API</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 1200px; 
                margin: 50px auto; 
                padding: 20px;
                background: #f5f5f5;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }
            .endpoint {
                background: white;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .method {
                display: inline-block;
                padding: 5px 15px;
                border-radius: 5px;
                font-weight: bold;
                margin-right: 10px;
            }
            .get { background: #61affe; color: white; }
            .post { background: #49cc90; color: white; }
            pre {
                background: #2d2d2d;
                color: #f8f8f2;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }
            .status-ok { color: #49cc90; font-weight: bold; }
            .status-error { color: #f93e3e; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>⚡ Smart Energy Consumption Predictor API</h1>
            <p>RESTful API for energy consumption forecasting</p>
        </div>

        <div class="endpoint">
            <h2><span class="method get">GET</span>/health</h2>
            <p>Check API health status</p>
            <h4>Response:</h4>
            <pre>{
  "status": "healthy",
  "model_loaded": true,
  "features_count": 46,
  "historical_data_points": 16450
}</pre>
        </div>

        <div class="endpoint">
            <h2><span class="method post">POST</span>/predict</h2>
            <p>Predict energy consumption for a single time point</p>
            <h4>Request Body:</h4>
            <pre>{
  "timestamp": "2025-12-09T14:00:00",
  "temperature": 28.5,
  "humidity": 62,
  "lag_features": {  // Optional
    "1": 3.2,
    "24": 3.1,
    "168": 2.9
  }
}</pre>
            <h4>Response:</h4>
            <pre>{
  "timestamp": "2025-12-09T14:00:00",
  "prediction": 3.45,
  "confidence_interval": {
    "lower": 3.15,
    "upper": 3.75,
    "level": 0.95
  },
  "metadata": {
    "model": "XGBoost",
    "features_used": 46,
    "prediction_time_ms": 18
  }
}</pre>
        </div>

        <div class="endpoint">
            <h2><span class="method post">POST</span>/predict/batch</h2>
            <p>Predict energy consumption for multiple time points</p>
            <h4>Request Body:</h4>
            <pre>{
  "predictions": [
    {
      "timestamp": "2025-12-09T14:00:00",
      "temperature": 28.5,
      "humidity": 62
    },
    {
      "timestamp": "2025-12-09T15:00:00",
      "temperature": 29.0,
      "humidity": 60
    }
  ]
}</pre>
            <h4>Response:</h4>
            <pre>{
  "predictions": [
    {
      "timestamp": "2025-12-09T14:00:00",
      "prediction": 3.45,
      "confidence_interval": {...}
    },
    {
      "timestamp": "2025-12-09T15:00:00",
      "prediction": 3.62,
      "confidence_interval": {...}
    }
  ],
  "summary": {
    "total_predictions": 2,
    "total_energy": 7.07,
    "average_energy": 3.54
  }
}</pre>
        </div>

        <div class="endpoint">
            <h2><span class="method post">POST</span>/forecast</h2>
            <p>Generate multi-step forecast (up to 168 hours)</p>
            <h4>Request Body:</h4>
            <pre>{
  "start_time": "2025-12-09T00:00:00",
  "hours_ahead": 24,
  "temperature": 22.0,
  "humidity": 60,
  "weather_scenario": "stable"  // stable, warming, cooling, variable
}</pre>
            <h4>Response:</h4>
            <pre>{
  "forecast": [
    {
      "timestamp": "2024-12-04T00:00:00",
      "prediction": 2.1,
      "confidence_interval": {...}
    },
    ...
  ],
  "summary": {
    "hours": 24,
    "total_energy": 67.8,
    "peak_consumption": 4.5,
    "peak_time": "2025-12-09T18:00:00"
  }
}</pre>
        </div>

        <div class="endpoint">
            <h2>📝 Example Usage (Python)</h2>
            <pre>import requests

# Single prediction
response = requests.post(
    "http://localhost:5000/predict",
    json={
        "timestamp": "2025-12-09T14:00:00",
        "temperature": 28.5,
        "humidity": 62
    }
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:5000/predict/batch",
    json={
        "predictions": [
            {"timestamp": "2025-12-09T14:00:00", "temperature": 28.5, "humidity": 62},
            {"timestamp": "2025-12-09T15:00:00", "temperature": 29.0, "humidity": 60}
        ]
    }
)
print(response.json())

# 24-hour forecast
response = requests.post(
    "http://localhost:5000/forecast",
    json={
        "start_time": "2025-12-09T00:00:00",
        "hours_ahead": 24,
        "temperature": 22.0,
        "humidity": 60
    }
)
print(response.json())</pre>
        </div>

        <div class="endpoint">
            <h2>🔧 Error Responses</h2>
            <p><span class="status-error">400 Bad Request:</span> Invalid input data</p>
            <p><span class="status-error">500 Internal Server Error:</span> Model prediction failed</p>
            <p><span class="status-error">503 Service Unavailable:</span> Model not loaded</p>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    if model_data is None:
        return jsonify({
            'status': 'unhealthy',
            'model_loaded': False,
            'error': 'Model not loaded'
        }), 503

    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'features_count': len(feature_cols),
        'historical_data_points': len(df_historical),
        'model_type': 'XGBoost'
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint."""
    try:
        # Validate model is loaded
        if model_data is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Server initialization failed'
            }), 503

        # Get request data
        data = request.get_json()

        # Validate required fields
        required_fields = ['timestamp', 'temperature', 'humidity']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': 'Missing required field',
                    'field': field,
                    'required_fields': required_fields
                }), 400

        # Parse timestamp
        try:
            prediction_time = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        except ValueError:
            return jsonify({
                'error': 'Invalid timestamp format',
                'expected': 'ISO 8601 format (e.g., 2025-12-09T14:00:00)'
            }), 400

        # Create features
        import time
        start_time = time.time()

        features = create_features_from_input(data, prediction_time)

        # Create feature vector
        X = np.array([[features.get(col, 0) for col in feature_cols]])

        # Scale and predict
        X_scaled = model_data['scaler'].transform(X)
        prediction = float(model_data['model'].predict(X_scaled)[0])

        # Calculate confidence interval
        rmse = 0.42  # From training
        lower_bound = max(0, prediction - 1.96 * rmse)
        upper_bound = prediction + 1.96 * rmse

        prediction_time_ms = int((time.time() - start_time) * 1000)

        # Return response
        return jsonify({
            'timestamp': data['timestamp'],
            'prediction': round(prediction, 3),
            'confidence_interval': {
                'lower': round(lower_bound, 3),
                'upper': round(upper_bound, 3),
                'level': 0.95
            },
            'metadata': {
                'model': 'XGBoost',
                'features_used': len(feature_cols),
                'prediction_time_ms': prediction_time_ms
            }
        }), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint."""
    try:
        if model_data is None:
            return jsonify({'error': 'Model not loaded'}), 503

        data = request.get_json()

        if 'predictions' not in data:
            return jsonify({
                'error': 'Missing "predictions" array in request body'
            }), 400

        predictions = []
        total_energy = 0

        for item in data['predictions']:
            # Parse timestamp
            prediction_time = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))

            # Create features and predict
            features = create_features_from_input(item, prediction_time)
            X = np.array([[features.get(col, 0) for col in feature_cols]])
            X_scaled = model_data['scaler'].transform(X)
            pred = float(model_data['model'].predict(X_scaled)[0])

            # Confidence interval
            rmse = 0.42
            lower = max(0, pred - 1.96 * rmse)
            upper = pred + 1.96 * rmse

            predictions.append({
                'timestamp': item['timestamp'],
                'prediction': round(pred, 3),
                'confidence_interval': {
                    'lower': round(lower, 3),
                    'upper': round(upper, 3)
                }
            })

            total_energy += pred

        return jsonify({
            'predictions': predictions,
            'summary': {
                'total_predictions': len(predictions),
                'total_energy': round(total_energy, 2),
                'average_energy': round(total_energy / len(predictions), 3)
            }
        }), 200

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/forecast', methods=['POST'])
def forecast():
    """Multi-step forecast endpoint."""
    try:
        if model_data is None:
            return jsonify({'error': 'Model not loaded'}), 503

        data = request.get_json()

        # Validate inputs
        required = ['start_time', 'hours_ahead', 'temperature', 'humidity']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        start_time = datetime.fromisoformat(data['start_time'].replace('Z', '+00:00'))
        hours_ahead = int(data['hours_ahead'])

        if hours_ahead > 168:
            return jsonify({'error': 'hours_ahead cannot exceed 168 (7 days)'}), 400

        # Generate weather forecast
        base_temp = data['temperature']
        base_humidity = data['humidity']
        scenario = data.get('weather_scenario', 'stable')

        temp_forecast = []
        humidity_forecast = []

        for hour in range(hours_ahead):
            if scenario == 'stable':
                temp = base_temp + np.random.normal(0, 2)
                hum = base_humidity + np.random.normal(0, 5)
            elif scenario == 'warming':
                temp = base_temp + (hour * 0.3) + np.random.normal(0, 2)
                hum = base_humidity - (hour * 0.2) + np.random.normal(0, 5)
            elif scenario == 'cooling':
                temp = base_temp - (hour * 0.3) + np.random.normal(0, 2)
                hum = base_humidity + (hour * 0.2) + np.random.normal(0, 5)
            else:  # variable
                temp = base_temp + 5 * np.sin(hour * np.pi / 12) + np.random.normal(0, 2)
                hum = base_humidity + 10 * np.cos(hour * np.pi / 12) + np.random.normal(0, 5)

            temp_forecast.append(np.clip(temp, -10, 45))
            humidity_forecast.append(np.clip(hum, 20, 95))

        # Generate predictions
        forecast_results = []
        df_extended = df_historical.copy()
        peak_consumption = 0
        peak_time = None

        for hour in range(hours_ahead):
            pred_time = start_time + timedelta(hours=hour)

            pred_data = {
                'temperature': temp_forecast[hour],
                'humidity': humidity_forecast[hour]
            }

            features = create_features_from_input(pred_data, pred_time)
            X = np.array([[features.get(col, 0) for col in feature_cols]])
            X_scaled = model_data['scaler'].transform(X)
            pred = float(model_data['model'].predict(X_scaled)[0])

            rmse = 0.42
            lower = max(0, pred - 1.96 * rmse)
            upper = pred + 1.96 * rmse

            forecast_results.append({
                'timestamp': pred_time.isoformat(),
                'prediction': round(pred, 3),
                'confidence_interval': {
                    'lower': round(lower, 3),
                    'upper': round(upper, 3)
                },
                'temperature': round(temp_forecast[hour], 1),
                'humidity': round(humidity_forecast[hour], 1)
            })

            if pred > peak_consumption:
                peak_consumption = pred
                peak_time = pred_time.isoformat()

            # Add to historical for next prediction
            new_row = pd.DataFrame([{
                'timestamp': pred_time,
                'energy_kwh': pred,
                'temperature_c': temp_forecast[hour],
                'humidity_percent': humidity_forecast[hour]
            }])
            df_extended = pd.concat([df_extended, new_row], ignore_index=True)

        total_energy = sum(r['prediction'] for r in forecast_results)

        return jsonify({
            'forecast': forecast_results,
            'summary': {
                'hours': hours_ahead,
                'total_energy': round(total_energy, 2),
                'average_energy': round(total_energy / hours_ahead, 3),
                'peak_consumption': round(peak_consumption, 3),
                'peak_time': peak_time,
                'weather_scenario': scenario
            }
        }), 200

    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# APP INITIALIZATION
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SMART ENERGY CONSUMPTION PREDICTOR API")
    print("=" * 70)

    # Load model
    if not load_model():
        print("\n✗ Failed to load model. Please run training pipeline first:")
        print("  python main.py")
        exit(1)

    print("\n✓ Model loaded successfully!")
    print("\nStarting Flask API server...")
    print("  - API Docs: http://localhost:5000/")
    print("  - Health Check: http://localhost:5000/health")
    print("  - Prediction: POST http://localhost:5000/predict")
    print("\n" + "=" * 70)

    # Run app
    app.run(host='0.0.0.0', port=5000, debug=True)