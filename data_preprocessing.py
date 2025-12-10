"""
Smart Energy Consumption Predictor
File: data_preprocessing.py

Purpose: Generate realistic synthetic energy consumption data
Author: Abdullahi Aliyu Garba
Date: December 2025
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class EnergyDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for energy consumption data.

    Handles:
    - Missing value imputation
    - Outlier detection and treatment
    - Data validation
    - Feature scaling
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize preprocessor.

        Args:
            verbose: Print detailed processing information
        """
        self.verbose = verbose
        self.stats = {}  # Store preprocessing statistics

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        if self.verbose:
            print("=" * 70)
            print("LOADING DATA")
            print("=" * 70)

        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        if self.verbose:
            print(f"\n✓ Loaded {len(df):,} records")
            print(f"  Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"  Columns: {list(df.columns)}")

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using intelligent imputation strategies.

        Strategy:
        - For gaps < 3 hours: Linear interpolation
        - For gaps 3-6 hours: Forward/backward fill average
        - For gaps > 6 hours: Drop records (too risky to impute)

        Args:
            df: Input dataframe

        Returns:
            DataFrame with missing values handled
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("HANDLING MISSING VALUES")
            print("=" * 70)

        df = df.copy()
        initial_missing = df['energy_kwh'].isna().sum()

        if initial_missing == 0:
            if self.verbose:
                print("\n✓ No missing values found!")
            return df

        if self.verbose:
            print(f"\nInitial missing values: {initial_missing} ({initial_missing / len(df) * 100:.2f}%)")

        # Identify missing value gaps
        df['missing_flag'] = df['energy_kwh'].isna().astype(int)
        df['gap_group'] = (df['missing_flag'] != df['missing_flag'].shift()).cumsum()

        # Calculate gap sizes
        gap_sizes = df[df['missing_flag'] == 1].groupby('gap_group').size()

        if self.verbose:
            print(f"\nMissing value gaps found: {len(gap_sizes)}")
            print(f"  Small gaps (1-2 hours): {(gap_sizes <= 2).sum()}")
            print(f"  Medium gaps (3-6 hours): {((gap_sizes >= 3) & (gap_sizes <= 6)).sum()}")
            print(f"  Large gaps (>6 hours): {(gap_sizes > 6).sum()}")

        # Strategy 1: Interpolate small gaps (1-2 hours)
        small_gap_mask = df['gap_group'].isin(gap_sizes[gap_sizes <= 2].index)
        small_gap_count = (small_gap_mask & df['missing_flag'].astype(bool)).sum()

        df.loc[small_gap_mask, 'energy_kwh'] = df.loc[small_gap_mask, 'energy_kwh'].interpolate(
            method='linear', limit_direction='both'
        )

        # Strategy 2: Forward/backward fill for medium gaps (3-6 hours)
        medium_gap_mask = df['gap_group'].isin(gap_sizes[(gap_sizes >= 3) & (gap_sizes <= 6)].index)
        medium_gap_count = (medium_gap_mask & df['missing_flag'].astype(bool)).sum()

        df.loc[medium_gap_mask, 'energy_kwh'] = df.loc[medium_gap_mask, 'energy_kwh'].fillna(
            method='ffill'
        ).fillna(method='bfill')

        # Strategy 3: Drop large gaps (>6 hours)
        large_gap_mask = df['gap_group'].isin(gap_sizes[gap_sizes > 6].index)
        large_gap_count = (large_gap_mask & df['missing_flag'].astype(bool)).sum()

        df = df[~large_gap_mask].reset_index(drop=True)

        # Handle missing weather data (simpler strategy - interpolate)
        df['temperature_c'] = df['temperature_c'].interpolate(method='linear', limit=10)
        df['humidity_percent'] = df['humidity_percent'].interpolate(method='linear', limit=10)

        # Drop any remaining missing values
        remaining_missing = df['energy_kwh'].isna().sum()
        if remaining_missing > 0:
            df = df.dropna(subset=['energy_kwh']).reset_index(drop=True)

        # Clean up temporary columns
        df = df.drop(['missing_flag', 'gap_group'], axis=1)

        if self.verbose:
            print("\nImputation Summary:")
            print(f"  Small gaps interpolated: {small_gap_count}")
            print(f"  Medium gaps filled: {medium_gap_count}")
            print(f"  Large gaps dropped: {large_gap_count}")
            print(f"  Final missing values: {df['energy_kwh'].isna().sum()}")
            print(f"  Records remaining: {len(df):,}")

        self.stats['missing_handled'] = initial_missing - df['energy_kwh'].isna().sum()

        return df

    def detect_and_treat_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and treat outliers using IQR method.

        Outliers are replaced with rolling median to maintain time-series continuity.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with outliers treated
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("DETECTING AND TREATING OUTLIERS")
            print("=" * 70)

        df = df.copy()

        # Calculate IQR bounds
        Q1 = df['energy_kwh'].quantile(0.25)
        Q3 = df['energy_kwh'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Ensure physical constraints
        lower_bound = max(lower_bound, 0)  # Energy can't be negative

        # Identify outliers
        outliers = (df['energy_kwh'] < lower_bound) | (df['energy_kwh'] > upper_bound)
        outlier_count = outliers.sum()

        if self.verbose:
            print(f"\nIQR Analysis:")
            print(f"  Q1 (25th percentile): {Q1:.2f} kWh")
            print(f"  Q3 (75th percentile): {Q3:.2f} kWh")
            print(f"  IQR: {IQR:.2f} kWh")
            print(f"  Lower Fence: {lower_bound:.2f} kWh")
            print(f"  Upper Fence: {upper_bound:.2f} kWh")
            print(f"\nOutliers detected: {outlier_count} ({outlier_count / len(df) * 100:.2f}%)")

        if outlier_count > 0:
            # Replace outliers with rolling median (6-hour window)
            rolling_median = df['energy_kwh'].rolling(window=6, center=True, min_periods=1).median()
            df.loc[outliers, 'energy_kwh'] = rolling_median[outliers]

            if self.verbose:
                print(f"✓ Outliers replaced with rolling median (6-hour window)")

        self.stats['outliers_treated'] = outlier_count

        return df

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data quality and apply business rules.

        Checks:
        - Energy values are non-negative
        - Temperature within reasonable range
        - Humidity within 0-100%
        - No duplicate timestamps

        Args:
            df: Input dataframe

        Returns:
            Validated DataFrame
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("VALIDATING DATA QUALITY")
            print("=" * 70)

        df = df.copy()
        initial_count = len(df)

        # Check 1: Remove negative energy values
        negative_energy = (df['energy_kwh'] < 0).sum()
        df = df[df['energy_kwh'] >= 0].reset_index(drop=True)

        # Check 2: Temperature range (-30°C to 50°C)
        invalid_temp = ((df['temperature_c'] < -30) | (df['temperature_c'] > 50)).sum()
        df = df[(df['temperature_c'] >= -30) & (df['temperature_c'] <= 50)].reset_index(drop=True)

        # Check 3: Humidity range (0-100%)
        invalid_humidity = ((df['humidity_percent'] < 0) | (df['humidity_percent'] > 100)).sum()
        df = df[(df['humidity_percent'] >= 0) & (df['humidity_percent'] <= 100)].reset_index(drop=True)

        # Check 4: Remove duplicate timestamps
        duplicates = df.duplicated(subset=['timestamp']).sum()
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

        # Check 5: Ensure chronological order
        df = df.sort_values('timestamp').reset_index(drop=True)

        removed = initial_count - len(df)

        if self.verbose:
            print("\nValidation Results:")
            print(f"  Negative energy values: {negative_energy}")
            print(f"  Invalid temperatures: {invalid_temp}")
            print(f"  Invalid humidity values: {invalid_humidity}")
            print(f"  Duplicate timestamps: {duplicates}")
            print(f"  Total records removed: {removed}")
            print(f"  Final record count: {len(df):,}")

            # Calculate data completeness
            completeness = len(df) / initial_count * 100
            print(f"\n✓ Data completeness: {completeness:.2f}%")

        self.stats['records_removed'] = removed

        return df

    def full_pipeline(self, filepath: str) -> pd.DataFrame:
        """
        Execute complete preprocessing pipeline.

        Args:
            filepath: Path to raw data CSV

        Returns:
            Clean, preprocessed DataFrame ready for feature engineering
        """
        # Load data
        df = self.load_data(filepath)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Detect and treat outliers
        df = self.detect_and_treat_outliers(df)

        # Validate data
        df = self.validate_data(df)

        if self.verbose:
            print("\n" + "=" * 70)
            print("PREPROCESSING COMPLETE")
            print("=" * 70)
            print(f"\nFinal Dataset:")
            print(f"  Records: {len(df):,}")
            print(f"  Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"  Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
            print(f"  Missing Values: {df.isna().sum().sum()}")
            print("\nPreprocessing Statistics:")
            for key, value in self.stats.items():
                print(f"  {key}: {value}")

        return df


if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = EnergyDataPreprocessor(verbose=True)

    # Run full pipeline
    df_clean = preprocessor.full_pipeline("data/raw/energy_data.csv")

    # Save cleaned data
    output_file = "data/processed/energy_data_clean.csv"
    df_clean.to_csv(output_file, index=False)
    print(f"\n✓ Clean data saved to: {output_file}")

    # Display summary statistics
    print("\n" + "=" * 70)
    print("CLEAN DATA SUMMARY")
    print("=" * 70)
    print(df_clean.describe())