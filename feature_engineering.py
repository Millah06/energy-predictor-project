"""
Smart Energy Consumption Predictor
File: feature_engineering.py

Purpose: Generate realistic synthetic energy consumption data
Author: Abdullahi Aliyu Garba
Date: December 2025
"""

import numpy as np
import pandas as pd
from typing import List
import warnings

warnings.filterwarnings('ignore')


class EnergyFeatureEngineer:
    """
    Feature engineering pipeline for energy consumption prediction.

    Creates:
    - Time-based features (hour, day, month, cyclical encoding)
    - Lag features (historical usage)
    - Rolling statistics (moving averages, std dev)
    - Weather interaction features
    """

    def __init__(self, verbose: bool = True):
        """Initialize feature engineer."""
        self.verbose = verbose
        self.feature_names = []

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from timestamp.

        Features created:
        - hour (0-23)
        - day_of_week (0=Monday, 6=Sunday)
        - day_of_month (1-31)
        - month (1-12)
        - quarter (1-4)
        - is_weekend (binary)
        - is_business_hours (9am-5pm)
        - is_peak_hours (2pm-8pm)
        - season (encoded as 0-3)
        - Cyclical encodings for hour and month

        Args:
            df: Input dataframe with 'timestamp' column

        Returns:
            DataFrame with additional time features
        """
        if self.verbose:
            print("=" * 70)
            print("CREATING TIME-BASED FEATURES")
            print("=" * 70)

        df = df.copy()

        # Basic time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter

        # Binary indicators
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_peak_hours'] = ((df['hour'] >= 14) & (df['hour'] <= 20)).astype(int)

        # Season encoding (0=Winter, 1=Spring, 2=Summer, 3=Fall)
        df['season'] = ((df['month'] % 12 + 3) // 3) % 4

        # Cyclical encoding for hour (captures that 23h and 0h are close)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Cyclical encoding for month (captures that Dec and Jan are close)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Day of week cyclical encoding
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        time_features = [
            'hour', 'day_of_week', 'day_of_month', 'month', 'quarter',
            'is_weekend', 'is_business_hours', 'is_peak_hours', 'season',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos'
        ]

        if self.verbose:
            print(f"\n✓ Created {len(time_features)} time-based features:")
            for feat in time_features:
                print(f"  - {feat}")

        return df

    def create_lag_features(
            self,
            df: pd.DataFrame,
            target_col: str = 'energy_kwh',
            lags: List[int] = [1, 2, 3, 6, 12, 24, 48, 168]
    ) -> pd.DataFrame:
        """
        Create lag features (historical values).

        Lag features capture recent history:
        - lag_1: 1 hour ago
        - lag_24: Same hour yesterday
        - lag_168: Same hour last week

        Args:
            df: Input dataframe
            target_col: Column to create lags from
            lags: List of lag periods (in hours)

        Returns:
            DataFrame with lag features
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("CREATING LAG FEATURES")
            print("=" * 70)

        df = df.copy()

        for lag in lags:
            col_name = f'{target_col}_lag_{lag}h'
            df[col_name] = df[target_col].shift(lag)

            if self.verbose:
                print(f"  - {col_name}: {lag} hours ago")

        # Drop rows with NaN lag values (first N rows)
        initial_count = len(df)
        df = df.dropna(subset=[f'{target_col}_lag_{lag}h' for lag in lags]).reset_index(drop=True)

        if self.verbose:
            print(f"\n✓ Created {len(lags)} lag features")
            print(f"  Records after dropping NaN lags: {len(df):,} (dropped {initial_count - len(df):,})")

        return df

    def create_rolling_features(
            self,
            df: pd.DataFrame,
            target_col: str = 'energy_kwh',
            windows: List[int] = [6, 12, 24, 168]
    ) -> pd.DataFrame:
        """
        Create rolling window statistics.

        Rolling features capture trends and variability:
        - rolling_mean_24h: Average of last 24 hours
        - rolling_std_24h: Variability in last 24 hours
        - rolling_min/max: Extremes in recent period

        Args:
            df: Input dataframe
            target_col: Column to calculate rolling stats from
            windows: List of window sizes (in hours)

        Returns:
            DataFrame with rolling features
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("CREATING ROLLING WINDOW FEATURES")
            print("=" * 70)

        df = df.copy()
        feature_count = 0

        for window in windows:
            # Rolling mean
            col_name = f'{target_col}_rolling_mean_{window}h'
            df[col_name] = df[target_col].rolling(window=window, min_periods=1).mean()
            feature_count += 1

            # Rolling standard deviation (captures variability)
            col_name = f'{target_col}_rolling_std_{window}h'
            df[col_name] = df[target_col].rolling(window=window, min_periods=1).std()
            feature_count += 1

            # Rolling min and max (for longer windows only)
            if window >= 24:
                col_name = f'{target_col}_rolling_min_{window}h'
                df[col_name] = df[target_col].rolling(window=window, min_periods=1).min()
                feature_count += 1

                col_name = f'{target_col}_rolling_max_{window}h'
                df[col_name] = df[target_col].rolling(window=window, min_periods=1).max()
                feature_count += 1

            if self.verbose:
                print(f"  - Window {window}h: mean, std" + (", min, max" if window >= 24 else ""))

        if self.verbose:
            print(f"\n✓ Created {feature_count} rolling window features")

        return df

    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather interaction features.

        Features:
        - temperature_squared: Captures extreme temperature effects
        - temp_humidity_index: "Feels like" temperature
        - is_extreme_temp: Binary flag for very hot/cold days

        Args:
            df: Input dataframe with temperature and humidity

        Returns:
            DataFrame with weather interaction features
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("CREATING WEATHER INTERACTION FEATURES")
            print("=" * 70)

        df = df.copy()

        # Temperature squared (non-linear relationship with energy usage)
        df['temperature_squared'] = df['temperature_c'] ** 2

        # Temperature-humidity interaction (heat index proxy)
        df['temp_humidity_index'] = df['temperature_c'] * (df['humidity_percent'] / 100)

        # Binary flags for extreme temperatures
        df['is_extreme_cold'] = (df['temperature_c'] < 5).astype(int)
        df['is_extreme_hot'] = (df['temperature_c'] > 30).astype(int)
        df['is_extreme_temp'] = (df['is_extreme_cold'] | df['is_extreme_hot']).astype(int)

        weather_features = [
            'temperature_squared', 'temp_humidity_index',
            'is_extreme_cold', 'is_extreme_hot', 'is_extreme_temp'
        ]

        if self.verbose:
            print(f"\n✓ Created {len(weather_features)} weather features:")
            for feat in weather_features:
                print(f"  - {feat}")

        return df

    def full_pipeline(
            self,
            df: pd.DataFrame,
            lags: List[int] = [1, 2, 3, 6, 12, 24, 48, 168],
            windows: List[int] = [6, 12, 24, 168]
    ) -> pd.DataFrame:
        """
        Execute complete feature engineering pipeline.

        Args:
            df: Clean input dataframe
            lags: Lag periods for lag features
            windows: Window sizes for rolling features

        Returns:
            DataFrame with all engineered features
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("STARTING FEATURE ENGINEERING PIPELINE")
            print("=" * 70)
            print(f"Initial shape: {df.shape}")

        # Create all features
        df = self.create_time_features(df)
        df = self.create_lag_features(df, lags=lags)
        df = self.create_rolling_features(df, windows=windows)
        df = self.create_weather_features(df)

        # Store feature names (excluding timestamp and target)
        self.feature_names = [col for col in df.columns if col not in ['timestamp', 'energy_kwh']]

        if self.verbose:
            print("\n" + "=" * 70)
            print("FEATURE ENGINEERING COMPLETE")
            print("=" * 70)
            print(f"Final shape: {df.shape}")
            print(f"Total features created: {len(self.feature_names)}")
            print(f"\nFeature categories:")

            time_feats = [f for f in self.feature_names if any(
                x in f for x in ['hour', 'day', 'month', 'season', 'weekend', 'business', 'peak', 'sin', 'cos'])]
            lag_feats = [f for f in self.feature_names if 'lag' in f]
            rolling_feats = [f for f in self.feature_names if 'rolling' in f]
            weather_feats = [f for f in self.feature_names if
                             any(x in f for x in ['temperature', 'humidity', 'extreme'])]

            print(f"  Time features: {len(time_feats)}")
            print(f"  Lag features: {len(lag_feats)}")
            print(f"  Rolling features: {len(rolling_feats)}")
            print(f"  Weather features: {len(weather_feats)}")

        return df


if __name__ == "__main__":
    # Load cleaned data
    df_clean = pd.read_csv("data/processed/energy_data_clean.csv")
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])

    print(f"Loaded clean data: {df_clean.shape}")

    # Initialize feature engineer
    engineer = EnergyFeatureEngineer(verbose=True)

    # Run full pipeline
    df_featured = engineer.full_pipeline(df_clean)

    # Save featured data
    output_file = "data/processed/energy_data_featured.csv"
    df_featured.to_csv(output_file, index=False)
    print(f"\n✓ Featured data saved to: {output_file}")

    # Display sample
    print("\n" + "=" * 70)
    print("SAMPLE OF FEATURED DATA")
    print("=" * 70)
    print(df_featured.head(5))

    print("\n" + "=" * 70)
    print("FEATURE LIST")
    print("=" * 70)
    for i, feat in enumerate(engineer.feature_names, 1):
        print(f"{i:2d}. {feat}")