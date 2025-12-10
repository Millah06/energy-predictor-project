"""
Smart Energy Consumption Predictor
File: dashboard.py

Purpose: Interactive Streamlit dashboard for energy predictions
Author: AbdullahiAliyu Garba
Date: December 2025

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    
    /* Fix text color for specific classes */
    .st-emotion-cache-1q82h82,
    .exsm2dv3,
    .st-emotion-cache-1q82h82.exsm2dv3 {
        color: #000000 !important;
    }
    
    .prediction-box {
        background-color: #e1f5ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        color: #262730
    }
    .info-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        color: #262730

    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model_and_data():
    """Load trained model and historical data."""
    try:
        # Load model (try XGBoost first, then Random Forest)
        model_data = None
        model_file = None
        model_name = None
        for model_file_path in ['models/xgboost_model.pkl', 'models/randomforest_model.pkl']:
            try:
                with open(model_file_path, 'rb') as f:
                    model_data = pickle.load(f)
                model_file = model_file_path
                model_name = 'XGBoost' if 'xgboost' in model_file_path else 'Random Forest'
                break
            except FileNotFoundError:
                continue
        
        if model_data is None:
            raise FileNotFoundError("No trained model found. Please run: python main.py")

        # Load featured data for historical reference
        df_featured = pd.read_csv('data/processed/energy_data_featured.csv')
        df_featured['timestamp'] = pd.to_datetime(df_featured['timestamp'])

        # Load feature names
        feature_cols = [col for col in df_featured.columns
                        if col not in ['timestamp', 'energy_kwh']]

        return model_data, df_featured, feature_cols

    except FileNotFoundError as e:
        st.error("⚠ Model files not found! Please run the training pipeline first.")
        st.code("python main.py")
        st.stop()


def create_time_features(timestamp: datetime) -> Dict:
    """Create time-based features from timestamp."""
    features = {
        'hour': timestamp.hour,
        'day_of_week': timestamp.weekday(),
        'day_of_month': timestamp.day,
        'month': timestamp.month,
        'quarter': (timestamp.month - 1) // 3 + 1,
        'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
        'is_business_hours': 1 if 9 <= timestamp.hour <= 17 else 0,
        'is_peak_hours': 1 if 14 <= timestamp.hour <= 20 else 0,
        'season': ((timestamp.month % 12 + 3) // 3) % 4,
        'hour_sin': np.sin(2 * np.pi * timestamp.hour / 24),
        'hour_cos': np.cos(2 * np.pi * timestamp.hour / 24),
        'month_sin': np.sin(2 * np.pi * timestamp.month / 12),
        'month_cos': np.cos(2 * np.pi * timestamp.month / 12),
        'dow_sin': np.sin(2 * np.pi * timestamp.weekday() / 7),
        'dow_cos': np.cos(2 * np.pi * timestamp.weekday() / 7)
    }
    return features


def create_lag_rolling_features(
        df_historical: pd.DataFrame,
        prediction_time: datetime,
        lags: List[int] = [1, 2, 3, 6, 12, 24, 48, 168]
) -> Dict:
    """Create lag and rolling features from historical data."""

    # Get historical values before prediction time
    df_hist = df_historical[df_historical['timestamp'] < prediction_time].copy()

    if len(df_hist) == 0:
        # Return default values if no history
        return {f'energy_kwh_lag_{lag}h': 2.5 for lag in lags}

    features = {}

    # Lag features
    for lag in lags:
        lag_time = prediction_time - timedelta(hours=lag)
        closest = df_hist.iloc[(df_hist['timestamp'] - lag_time).abs().argsort()[:1]]

        if len(closest) > 0:
            features[f'energy_kwh_lag_{lag}h'] = closest['energy_kwh'].values[0]
        else:
            features[f'energy_kwh_lag_{lag}h'] = df_hist['energy_kwh'].mean()

    # Rolling features
    windows = [6, 12, 24, 168]
    recent_data = df_hist.tail(max(windows))

    for window in windows:
        window_data = recent_data.tail(window)['energy_kwh']

        features[f'energy_kwh_rolling_mean_{window}h'] = window_data.mean()
        features[f'energy_kwh_rolling_std_{window}h'] = window_data.std()

        if window >= 24:
            features[f'energy_kwh_rolling_min_{window}h'] = window_data.min()
            features[f'energy_kwh_rolling_max_{window}h'] = window_data.max()

    return features


def create_weather_features(temperature: float, humidity: float) -> Dict:
    """Create weather interaction features."""
    return {
        'temperature_c': temperature,
        'humidity_percent': humidity,
        'temperature_squared': temperature ** 2,
        'temp_humidity_index': temperature * (humidity / 100),
        'is_extreme_cold': 1 if temperature < 5 else 0,
        'is_extreme_hot': 1 if temperature > 30 else 0,
        'is_extreme_temp': 1 if (temperature < 5 or temperature > 30) else 0
    }


def predict_single(
        model_data: Dict,
        timestamp: datetime,
        temperature: float,
        humidity: float,
        df_historical: pd.DataFrame,
        feature_cols: List[str]
) -> Tuple[float, float, float]:
    """
    Predict energy consumption for a single timestamp.

    Returns:
        Tuple of (prediction, lower_bound, upper_bound)
    """

    # Create all features
    features = {}
    features.update(create_time_features(timestamp))
    features.update(create_weather_features(temperature, humidity))
    features.update(create_lag_rolling_features(df_historical, timestamp))

    # Create feature vector in correct order
    X = np.array([[features.get(col, 0) for col in feature_cols]])

    # Scale features
    X_scaled = model_data['scaler'].transform(X)

    # Predict
    prediction = model_data['model'].predict(X_scaled)[0]

    # Calculate confidence intervals (using prediction std)
    # Approximate 95% CI as ±2 * historical RMSE
    rmse_estimate = 0.42  # From training results
    lower_bound = max(0, prediction - 1.96 * rmse_estimate)
    upper_bound = prediction + 1.96 * rmse_estimate

    return prediction, lower_bound, upper_bound


def predict_multi_step(
        model_data: Dict,
        start_time: datetime,
        hours_ahead: int,
        temperature_forecast: List[float],
        humidity_forecast: List[float],
        df_historical: pd.DataFrame,
        feature_cols: List[str]
) -> pd.DataFrame:
    """
    Predict multiple time steps ahead.

    Uses recursive forecasting: prediction at time t becomes lag feature for t+1.
    """

    predictions = []
    df_extended = df_historical.copy()

    for hour in range(hours_ahead):
        pred_time = start_time + timedelta(hours=hour)
        temp = temperature_forecast[hour] if hour < len(temperature_forecast) else temperature_forecast[-1]
        humidity = humidity_forecast[hour] if hour < len(humidity_forecast) else humidity_forecast[-1]

        # Predict
        pred, lower, upper = predict_single(
            model_data, pred_time, temp, humidity, df_extended, feature_cols
        )

        predictions.append({
            'timestamp': pred_time,
            'energy_kwh': pred,
            'lower_bound': lower,
            'upper_bound': upper,
            'temperature_c': temp,
            'humidity_percent': humidity
        })

        # Add prediction to historical data for next iteration
        new_row = pd.DataFrame([{
            'timestamp': pred_time,
            'energy_kwh': pred,
            'temperature_c': temp,
            'humidity_percent': humidity
        }])
        df_extended = pd.concat([df_extended, new_row], ignore_index=True)

    return pd.DataFrame(predictions)


def generate_weather_forecast(
        base_temp: float,
        base_humidity: float,
        hours: int,
        variation: str = "stable"
) -> Tuple[List[float], List[float]]:
    """Generate synthetic weather forecast."""

    temperatures = []
    humidities = []

    for hour in range(hours):
        if variation == "stable":
            temp = base_temp + np.random.normal(0, 2)
            humidity = base_humidity + np.random.normal(0, 5)
        elif variation == "warming":
            temp = base_temp + (hour * 0.3) + np.random.normal(0, 2)
            humidity = base_humidity - (hour * 0.2) + np.random.normal(0, 5)
        elif variation == "cooling":
            temp = base_temp - (hour * 0.3) + np.random.normal(0, 2)
            humidity = base_humidity + (hour * 0.2) + np.random.normal(0, 5)
        elif variation == "variable":
            temp = base_temp + 5 * np.sin(hour * np.pi / 12) + np.random.normal(0, 2)
            humidity = base_humidity + 10 * np.cos(hour * np.pi / 12) + np.random.normal(0, 5)

        temperatures.append(np.clip(temp, -10, 45))
        humidities.append(np.clip(humidity, 20, 95))

    return temperatures, humidities


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    """Main dashboard function."""

    # Header
    st.title("⚡ Smart Energy Consumption Predictor")
    st.markdown("### AI-Powered Energy Forecasting Dashboard")
    st.markdown("---")

    # Load model and data
    with st.spinner("Loading model and data..."):
        model_data, df_historical, feature_cols = load_model_and_data()

    # Sidebar - Input Controls
    with st.sidebar:
        st.header("🎛 Prediction Controls")

        # Date and time selection
        st.subheader("📅 Date & Time")
        prediction_date = st.date_input(
            "Select Date",
            value=datetime.now().date(),
            min_value=datetime.now().date(),
            max_value=datetime.now().date() + timedelta(days=30)
        )

        prediction_hour = st.slider(
            "Select Hour",
            min_value=0,
            max_value=23,
            value=datetime.now().hour,
            help="Hour of day (0-23)"
        )

        # Combine date and hour
        prediction_time = datetime.combine(prediction_date, datetime.min.time())
        prediction_time = prediction_time.replace(hour=prediction_hour)

        st.markdown("---")

        # Weather inputs
        st.subheader("🌡 Weather Conditions")
        temperature = st.slider(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=45.0,
            value=22.0,
            step=0.5,
            help="Current outdoor temperature"
        )

        humidity = st.slider(
            "Humidity (%)",
            min_value=20,
            max_value=95,
            value=60,
            step=1,
            help="Relative humidity percentage"
        )

        st.markdown("---")

        # Forecast horizon
        st.subheader("🔮 Forecast Horizon")
        forecast_type = st.radio(
            "Select Prediction Type",
            options=["Next Hour", "Next 24 Hours", "Next 7 Days"],
            help="Choose forecast duration"
        )

        # Weather scenario for multi-step
        if forecast_type != "Next Hour":
            weather_scenario = st.selectbox(
                "Weather Scenario",
                options=["Stable", "Warming Trend", "Cooling Trend", "Variable"],
                help="Expected weather pattern"
            )

        st.markdown("---")

        # Usage scenario
        st.subheader("🏠 Usage Scenario")
        usage_scenario = st.selectbox(
            "Building Activity Level",
            options=["Normal", "High Activity", "Low Activity", "Vacation Mode"],
            help="Expected occupancy and activity"
        )

        st.markdown("---")

        # Predict button
        predict_button = st.button("🚀 Generate Prediction", type="primary", use_container_width=True)

    # Main content area
    if predict_button:

        # Adjust predictions based on usage scenario
        scenario_multipliers = {
            "Normal": 1.0,
            "High Activity": 1.2,
            "Low Activity": 0.8,
            "Vacation Mode": 0.5
        }
        multiplier = scenario_multipliers[usage_scenario]

        # ====================================================================
        # SINGLE HOUR PREDICTION
        # ====================================================================
        if forecast_type == "Next Hour":

            with st.spinner("Generating prediction..."):
                prediction, lower, upper = predict_single(
                    model_data, prediction_time, temperature, humidity,
                    df_historical, feature_cols
                )

                # Apply scenario multiplier
                prediction *= multiplier
                lower *= multiplier
                upper *= multiplier

            # Display results
            st.success("✅ Prediction Generated Successfully!")

            # Prediction metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="Predicted Consumption",
                    value=f"{prediction:.2f} kWh",
                    help="Expected energy consumption"
                )

            with col2:
                st.metric(
                    label="Confidence Range",
                    value=f"±{(upper - prediction):.2f} kWh",
                    help="95% confidence interval"
                )

            with col3:
                estimated_cost = prediction * 0.12  # $0.12/kWh average rate
                st.metric(
                    label="Estimated Cost",
                    value=f"${estimated_cost:.2f}",
                    help="At $0.12/kWh"
                )

            with col4:
                # Compare to average
                avg_consumption = df_historical['energy_kwh'].mean()
                delta = ((prediction - avg_consumption) / avg_consumption) * 100
                st.metric(
                    label="vs Average",
                    value=f"{abs(delta):.1f}%",
                    delta=f"{'Higher' if delta > 0 else 'Lower'}",
                    delta_color="inverse"
                )

            # Detailed prediction box
            st.markdown(f"""
            <div class="prediction-box">
                <h3>📊 Prediction Details</h3>
                <p><strong>Time:</strong> {prediction_time.strftime('%Y-%m-%d %H:%M')}</p>
                <p><strong>Expected Consumption:</strong> {prediction:.2f} kWh</p>
                <p><strong>Confidence Interval:</strong> {lower:.2f} - {upper:.2f} kWh (95%)</p>
                <p><strong>Scenario Adjustment:</strong> {usage_scenario} ({multiplier:.1f}x)</p>
            </div>
            """, unsafe_allow_html=True)

            # Comparison chart
            st.subheader("📈 Hourly Comparison")

            # Get same hour in past week for comparison
            comparison_times = [prediction_time - timedelta(days=i) for i in range(1, 8)]
            comparison_data = []

            for comp_time in comparison_times:
                closest = df_historical[
                    (df_historical['timestamp'].dt.date == comp_time.date()) &
                    (df_historical['timestamp'].dt.hour == comp_time.hour)
                    ]
                if len(closest) > 0:
                    comparison_data.append({
                        'Time': comp_time.strftime('%a %m/%d'),
                        'Energy (kWh)': closest['energy_kwh'].values[0]
                    })

            comparison_data.append({
                'Time': 'Predicted',
                'Energy (kWh)': prediction
            })

            comp_df = pd.DataFrame(comparison_data)

            fig_comp = px.bar(
                comp_df,
                x='Time',
                y='Energy (kWh)',
                title="Predicted vs Same Hour Past Week",
                color='Time',
                color_discrete_map={'Predicted': '#ff6b6b'}
            )
            fig_comp.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_comp, use_container_width=True)

        # ====================================================================
        # MULTI-STEP PREDICTIONS (24H or 7D)
        # ====================================================================
        else:
            hours_ahead = 24 if forecast_type == "Next 24 Hours" else 168

            # Generate weather forecast
            weather_map = {
                "Stable": "stable",
                "Warming Trend": "warming",
                "Cooling Trend": "cooling",
                "Variable": "variable"
            }

            with st.spinner(f"Generating {hours_ahead}-hour forecast..."):
                temp_forecast, humidity_forecast = generate_weather_forecast(
                    temperature, humidity, hours_ahead,
                    variation=weather_map[weather_scenario]
                )

                # Generate predictions
                forecast_df = predict_multi_step(
                    model_data, prediction_time, hours_ahead,
                    temp_forecast, humidity_forecast,
                    df_historical, feature_cols
                )

                # Apply scenario multiplier
                forecast_df['energy_kwh'] *= multiplier
                forecast_df['lower_bound'] *= multiplier
                forecast_df['upper_bound'] *= multiplier

            st.success(f"✅ {hours_ahead}-Hour Forecast Generated Successfully!")

            # Summary metrics
            total_energy = forecast_df['energy_kwh'].sum()
            peak_energy = forecast_df['energy_kwh'].max()
            peak_time = forecast_df.loc[forecast_df['energy_kwh'].idxmax(), 'timestamp']
            avg_energy = forecast_df['energy_kwh'].mean()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="Total Energy",
                    value=f"{total_energy:.1f} kWh",
                    help=f"Sum over {hours_ahead} hours"
                )

            with col2:
                st.metric(
                    label="Peak Consumption",
                    value=f"{peak_energy:.2f} kWh",
                    help="Maximum predicted value"
                )

            with col3:
                st.metric(
                    label="Average",
                    value=f"{avg_energy:.2f} kWh",
                    help="Mean hourly consumption"
                )

            with col4:
                total_cost = total_energy * 0.12
                st.metric(
                    label="Estimated Cost",
                    value=f"${total_cost:.2f}",
                    help="At $0.12/kWh"
                )

            # Peak time info
            st.markdown(f"""
            <div class="info-box">
                <strong>⚡ Peak Demand Alert:</strong> 
                Expected peak consumption of {peak_energy:.2f} kWh at 
                {peak_time.strftime('%Y-%m-%d %H:%M')}
            </div>
            """, unsafe_allow_html=True)

            # Forecast visualization
            st.subheader("📈 Energy Consumption Forecast")

            fig_forecast = go.Figure()

            # Add confidence interval
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['timestamp'],
                y=forecast_df['upper_bound'],
                fill=None,
                mode='lines',
                line_color='rgba(0,100,255,0)',
                showlegend=False,
                name='Upper Bound'
            ))

            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['timestamp'],
                y=forecast_df['lower_bound'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,100,255,0)',
                fillcolor='rgba(0,100,255,0.2)',
                name='95% Confidence'
            ))

            # Add prediction line
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['timestamp'],
                y=forecast_df['energy_kwh'],
                mode='lines+markers',
                name='Predicted Consumption',
                line=dict(color='#2196F3', width=3),
                marker=dict(size=6)
            ))

            # Highlight peak
            fig_forecast.add_trace(go.Scatter(
                x=[peak_time],
                y=[peak_energy],
                mode='markers',
                name='Peak Demand',
                marker=dict(size=15, color='red', symbol='star')
            ))

            fig_forecast.update_layout(
                title=f"{hours_ahead}-Hour Energy Consumption Forecast",
                xaxis_title="Time",
                yaxis_title="Energy Consumption (kWh)",
                hovermode='x unified',
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig_forecast, use_container_width=True)

            # Daily/weekly pattern analysis
            if forecast_type == "Next 24 Hours":
                st.subheader("🕐 Hourly Pattern Analysis")

                hourly_avg = forecast_df.groupby(
                    forecast_df['timestamp'].dt.hour
                )['energy_kwh'].mean().reset_index()
                hourly_avg.columns = ['Hour', 'Average kWh']

                fig_hourly = px.bar(
                    hourly_avg,
                    x='Hour',
                    y='Average kWh',
                    title="Average Consumption by Hour of Day",
                    color='Average kWh',
                    color_continuous_scale='Blues'
                )
                fig_hourly.update_layout(height=400)
                st.plotly_chart(fig_hourly, use_container_width=True)

            else:  # 7-day forecast
                st.subheader("📆 Daily Pattern Analysis")

                forecast_df['date'] = forecast_df['timestamp'].dt.date
                daily_total = forecast_df.groupby('date')['energy_kwh'].sum().reset_index()
                daily_total.columns = ['Date', 'Total kWh']

                fig_daily = px.bar(
                    daily_total,
                    x='Date',
                    y='Total kWh',
                    title="Daily Total Energy Consumption",
                    color='Total kWh',
                    color_continuous_scale='Oranges'
                )
                fig_daily.update_layout(height=400)
                st.plotly_chart(fig_daily, use_container_width=True)

            # Weather correlation
            st.subheader("🌡 Temperature Impact")

            fig_temp = go.Figure()

            fig_temp.add_trace(go.Scatter(
                x=forecast_df['timestamp'],
                y=forecast_df['temperature_c'],
                mode='lines',
                name='Temperature (°C)',
                yaxis='y',
                line=dict(color='orange', width=2)
            ))

            fig_temp.add_trace(go.Scatter(
                x=forecast_df['timestamp'],
                y=forecast_df['energy_kwh'],
                mode='lines',
                name='Energy (kWh)',
                yaxis='y2',
                line=dict(color='blue', width=2)
            ))

            fig_temp.update_layout(
                title="Temperature vs Energy Consumption",
                xaxis_title="Time",
                yaxis=dict(title="Temperature (°C)", side='left'),
                yaxis2=dict(title="Energy (kWh)", overlaying='y', side='right'),
                hovermode='x unified',
                height=400
            )

            st.plotly_chart(fig_temp, use_container_width=True)

            # Download forecast
            st.subheader("💾 Export Forecast")

            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast as CSV",
                data=csv,
                file_name=f"energy_forecast_{prediction_time.strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

    else:
        # Show welcome message and instructions
        st.info("👈 Configure your prediction parameters in the sidebar and click '🚀 Generate Prediction'")

        # Show historical data overview
        st.subheader("📊 Historical Data Overview")

        # Last 7 days chart
        last_week = df_historical.tail(168)  # Last 7 days

        fig_hist = px.line(
            last_week,
            x='timestamp',
            y='energy_kwh',
            title="Energy Consumption - Last 7 Days",
            labels={'energy_kwh': 'Energy (kWh)', 'timestamp': 'Time'}
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Average (7d)", f"{last_week['energy_kwh'].mean():.2f} kWh")
        with col2:
            st.metric("Peak (7d)", f"{last_week['energy_kwh'].max():.2f} kWh")
        with col3:
            st.metric("Min (7d)", f"{last_week['energy_kwh'].min():.2f} kWh")
        with col4:
            st.metric("Std Dev (7d)", f"{last_week['energy_kwh'].std():.2f} kWh")


# ============================================================================
# RUN DASHBOARD
# ============================================================================

if __name__ == "__main__":
    main()