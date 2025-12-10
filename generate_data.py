"""
Smart Energy Consumption Predictor
File: 01_generate_data.py

Purpose: Generate realistic synthetic energy consumption data
Author: AbdullahiAliyu Garba
Date: December 2025
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

def generate_synthetic_energy_data(start_date: str = "2024-01-01",
                                   end_date: str = "2025-12-01",
                                   freq: str = "H") -> pd.DataFrame:
    """
    Generate realistic synthetic energy consumption data.

    '''
    This simulates a residential household with:
    - Daily patterns (higher usage during day, lower at night)
    - Weekly patterns (lower usage on weekends)
    - Seasonal patterns (higher in summer for AC, winter for heating)
    - Weather correlation (temperature affects usage)
    - Random noise and occasional anomalies

    Args:
        start_date: Start date for data generation (YYYY-MM-DD)
        end_date: End date for data generation (YYYY-MM-DD)
        freq: Frequency ('H' for hourly, 'D' for daily)

    Returns:
        DataFrame with timestamp, energy_kwh, temperature, humidity
    """

    print("=" * 70)
    print("GENERATING SYNTHETIC ENERGY CONSUMPTION DATA")
    print("=" * 70)

    # Create timestamp range
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_samples = len(timestamps)

    print(f"\nDataset Info:")
    print(f"  Start Date: {start_date}")
    print(f"  End Date: {end_date}")
    print(f"  Frequency: {freq} (Hourly)")
    print(f"  Total Records: {n_samples:,}")
    print(f"  Duration: {(timestamps[-1] - timestamps[0]).days} days")

    # Extract time features for pattern generation
    # Convert to numpy arrays to avoid pandas Index issues
    hours = np.asarray(timestamps.hour)
    days_of_week = np.asarray(timestamps.dayofweek)
    months = np.asarray(timestamps.month)
    day_of_year = np.asarray(timestamps.dayofyear)

    # ====================================================================
    # COMPONENT 1: Base Load (always-on appliances)
    # ====================================================================
    # Refrigerator, standby devices, etc.
    base_load = 0.5 + np.random.normal(0, 0.05, n_samples)

    # ====================================================================
    # COMPONENT 2: Daily Pattern (circadian rhythm of household)
    # ====================================================================
    # Morning peak (6-9am), evening peak (5-10pm), low at night
    daily_pattern = (
        1.5 * np.sin((hours - 6) * np.pi / 12) +  # Morning peak
        2.0 * np.sin((hours - 18) * np.pi / 6) +   # Evening peak (stronger)
        1.0  # Offset
    )
    daily_pattern = np.maximum(daily_pattern, 0)  # No negative values

    # ====================================================================
    # COMPONENT 3: Weekly Pattern (weekday vs weekend behavior)
    # ====================================================================
    # Weekend: lower usage (people out, different schedule)
    weekly_pattern = np.where(
        (days_of_week == 5) | (days_of_week == 6),  # Saturday/Sunday
        0.85,  # 15% lower on weekends
        1.0
    )

    # ====================================================================
    # COMPONENT 4: Seasonal Pattern (heating/cooling needs)
    # ====================================================================
    # Summer (Jun-Aug) and Winter (Dec-Feb) have higher usage
    seasonal_pattern = 1.0 + 0.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # ====================================================================
    # COMPONENT 5: Temperature Generation
    # ====================================================================
    # Seasonal temperature with daily variation
    base_temp = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    daily_temp_variation = 5 * np.sin(2 * np.pi * hours / 24)
    temperature = base_temp + daily_temp_variation + np.random.normal(0, 2, n_samples)

    # ====================================================================
    # COMPONENT 6: Temperature Impact on Energy Usage
    # ====================================================================
    # Extreme temperatures (hot or cold) increase energy usage
    temp_impact = np.where(
        temperature > 25,  # Hot days (AC usage)
        0.05 * (temperature - 25),
        np.where(
            temperature < 10,  # Cold days (heating usage)
            0.08 * (10 - temperature),
            0
        )
    )

    # ====================================================================
    # COMPONENT 7: Humidity Generation
    # ====================================================================
    # Correlated with temperature (hot days often more humid)
    humidity = 50 + 20 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    humidity += np.random.normal(0, 10, n_samples)
    humidity = np.clip(humidity, 20, 95)  # Physical limits

    # ====================================================================
    # COMPONENT 8: Combine All Components
    # ====================================================================
    energy_kwh = (
        base_load +
        daily_pattern * weekly_pattern * seasonal_pattern +
        temp_impact +
        np.random.normal(0, 0.2, n_samples)  # Random noise
    )

    # Ensure no negative values
    energy_kwh = np.maximum(energy_kwh, 0.1)

    # ====================================================================
    # COMPONENT 9: Add Realistic Anomalies (2% of data)
    # ====================================================================
    # Simulate occasional unusual events (parties, vacations, etc.)
    anomaly_indices = np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)
    anomaly_multipliers = np.random.uniform(0.3, 2.5, len(anomaly_indices))
    # Ensure energy_kwh is a numpy array and use direct assignment
    energy_kwh = np.asarray(energy_kwh).copy()
    energy_kwh[anomaly_indices] = energy_kwh[anomaly_indices] * anomaly_multipliers

    # ====================================================================
    # COMPONENT 10: Add Missing Values (realistic sensor failures)
    # ====================================================================
    # 1% missing data in random chunks
    # Ensure all arrays are proper numpy arrays
    temperature = np.asarray(temperature).copy()
    humidity = np.asarray(humidity).copy()
    missing_chunks = 5  # Number of outage events
    for _ in range(missing_chunks):
        start_idx = np.random.randint(100, n_samples - 100)
        chunk_size = np.random.randint(1, 10)  # 1-10 hour outages
        energy_kwh[start_idx:start_idx + chunk_size] = np.nan
        temperature[start_idx:start_idx + chunk_size] = np.nan
        humidity[start_idx:start_idx + chunk_size] = np.nan

    # ====================================================================
    # CREATE FINAL DATAFRAME
    # ====================================================================
    df = pd.DataFrame({
        'timestamp': timestamps,
        'energy_kwh': energy_kwh,
        'temperature_c': temperature,
        'humidity_percent': humidity
    })

    # Add basic time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # ====================================================================
    # PRINT DATASET STATISTICS
    # ====================================================================
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"\nEnergy Consumption (kWh):")
    print(f"  Mean:    {df['energy_kwh'].mean():.2f} kWh")
    print(f"  Median:  {df['energy_kwh'].median():.2f} kWh")
    print(f"  Std Dev: {df['energy_kwh'].std():.2f} kWh")
    print(f"  Min:     {df['energy_kwh'].min():.2f} kWh")
    print(f"  Max:     {df['energy_kwh'].max():.2f} kWh")

    print(f"\nTemperature (°C):")
    print(f"  Mean:    {df['temperature_c'].mean():.2f}°C")
    print(f"  Range:   {df['temperature_c'].min():.2f}°C to {df['temperature_c'].max():.2f}°C")

    print(f"\nHumidity (%):")
    print(f"  Mean:    {df['humidity_percent'].mean():.1f}%")
    print(f"  Range:   {df['humidity_percent'].min():.1f}% to {df['humidity_percent'].max():.1f}%")

    print(f"\nData Quality:")
    print(f"  Missing Values: {df['energy_kwh'].isna().sum()} ({df['energy_kwh'].isna().sum()/len(df)*100:.2f}%)")
    print(f"  Complete Cases: {df.dropna().shape[0]:,}")

    print("\n" + "=" * 70)
    print("DATA GENERATION COMPLETE")
    print("=" * 70)

    return df


if __name__ == "__main__":
    # Generate 2 years of hourly data
    df = generate_synthetic_energy_data(
    start_date="2024-01-01",
    end_date="2025-12-01",
    freq="H"
    )


    # Save to CSV
    output_file = "data/raw/energy_data.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Data saved to: {output_file}")

    # Display sample
    print("\nFirst 10 rows:")
    print(df.head(10).to_string())

    print("\nLast 10 rows:")
    print(df.tail(10).to_string())
