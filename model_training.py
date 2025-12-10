"""
Smart Energy Consumption Predictor
File: model_training.py

Purpose: Train and evaluate multiple forecasting models
Author: ML Engineering Team
Date: 2024-12-03
"""

import numpy as np
import pandas as pd
import pickle
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')

# Optional imports - models will be skipped if packages aren't available
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARNING] scikit-learn not available. Random Forest model will be skipped.")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARNING] xgboost not available. XGBoost model will be skipped.")

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[WARNING] statsmodels not available. ARIMA model will be skipped.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("[WARNING] tensorflow not available. LSTM model will be skipped.")


class EnergyPredictionModels:
    """
    Comprehensive model training and evaluation for energy prediction.

    Implements:
    - ARIMA (time-series baseline)
    - Random Forest Regressor
    - XGBoost
    - LSTM Neural Network
    """

    def __init__(self, random_state: int = 42):
        """Initialize models with random seed for reproducibility."""
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.results = {}

        # Set random seeds
        np.random.seed(random_state)
        if HAS_TENSORFLOW:
            tf.random.set_seed(random_state)

    def prepare_data(
            self,
            df: pd.DataFrame,
            target_col: str = 'energy_kwh',
            test_size: float = 0.15,
            val_size: float = 0.15
    ) -> Tuple:
        """
        Prepare data for training with time-series split.

        IMPORTANT: No shuffling! Time order must be preserved.

        Args:
            df: Featured dataframe
            target_col: Target variable column name
            test_size: Proportion for test set
            val_size: Proportion for validation set

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
        """
        print("=" * 70)
        print("PREPARING DATA FOR TRAINING")
        print("=" * 70)

        # Remove timestamp and target from features
        feature_cols = [col for col in df.columns if col not in ['timestamp', target_col]]

        X = df[feature_cols].values
        y = df[target_col].values

        # Time-series split (chronological order)
        n_samples = len(X)
        test_start = int(n_samples * (1 - test_size))
        val_start = int(n_samples * (1 - test_size - val_size))

        X_train = X[:val_start]
        X_val = X[val_start:test_start]
        X_test = X[test_start:]

        y_train = y[:val_start]
        y_val = y[val_start:test_start]
        y_test = y[test_start:]

        print(f"\nData split (chronological):")
        print(f"  Training:   {X_train.shape[0]:,} samples ({X_train.shape[0] / n_samples * 100:.1f}%)")
        print(f"  Validation: {X_val.shape[0]:,} samples ({X_val.shape[0] / n_samples * 100:.1f}%)")
        print(f"  Test:       {X_test.shape[0]:,} samples ({X_test.shape[0] / n_samples * 100:.1f}%)")
        print(f"  Features:   {X_train.shape[1]}")

        return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.

        Metrics:
        - MAE: Mean Absolute Error (kWh)
        - RMSE: Root Mean Square Error (kWh)
        - MAPE: Mean Absolute Percentage Error (%)
        - R²: Coefficient of determination

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        if HAS_SKLEARN:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
        else:
            # Manual calculation if sklearn not available
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # MAPE (avoid division by zero)
        mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 0.1, None))) * 100

        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }

    def train_arima(
            self,
            y_train: np.ndarray,
            y_val: np.ndarray,
            order: Tuple[int, int, int] = (24, 1, 2)
    ) -> Dict:
        """
        Train ARIMA model.

        Note: ARIMA only uses past target values, not external features.
        This serves as a univariate baseline.

        Args:
            y_train: Training target values
            y_val: Validation target values
            order: ARIMA order (p, d, q)

        Returns:
            Dictionary with model and metrics
        """
        print("\n" + "=" * 70)
        print("TRAINING ARIMA MODEL")
        print("=" * 70)
        print(f"Order: ARIMA{order}")

        try:
            # Train ARIMA on training data
            model = ARIMA(y_train, order=order)
            fitted_model = model.fit()

            # Predictions (forecast validation period)
            y_pred = fitted_model.forecast(steps=len(y_val))

            # Calculate metrics
            metrics = self.calculate_metrics(y_val, y_pred)

            print(f"\nARIMA Results:")
            print(f"  MAE:  {metrics['MAE']:.3f} kWh")
            print(f"  RMSE: {metrics['RMSE']:.3f} kWh")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
            print(f"  R²:   {metrics['R2']:.3f}")

            return {
                'model': fitted_model,
                'metrics': metrics,
                'predictions': y_pred
            }

        except Exception as e:
            print(f"✗ ARIMA training failed: {e}")
            return None

    def train_random_forest(
            self,
            X_train: np.ndarray,
            X_val: np.ndarray,
            y_train: np.ndarray,
            y_val: np.ndarray
    ) -> Dict:
        """
        Train Random Forest Regressor.

        Args:
            X_train, X_val: Feature matrices
            y_train, y_val: Target values

        Returns:
            Dictionary with model, metrics, and feature importance
        """
        print("\n" + "=" * 70)
        print("TRAINING RANDOM FOREST MODEL")
        print("=" * 70)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )

        print("Training...")
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_val_scaled)

        # Calculate metrics
        metrics = self.calculate_metrics(y_val, y_pred)

        print(f"\nRandom Forest Results:")
        print(f"  MAE:  {metrics['MAE']:.3f} kWh")
        print(f"  RMSE: {metrics['RMSE']:.3f} kWh")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  R²:   {metrics['R2']:.3f}")
        print(f"  Trees: {model.n_estimators}")

        return {
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'predictions': y_pred,
            'feature_importance': model.feature_importances_
        }

    def train_xgboost(
            self,
            X_train: np.ndarray,
            X_val: np.ndarray,
            y_train: np.ndarray,
            y_val: np.ndarray
    ) -> Dict:
        """
        Train XGBoost model.

        Args:
            X_train, X_val: Feature matrices
            y_train, y_val: Target values

        Returns:
            Dictionary with model, metrics, and feature importance
        """
        print("\n" + "=" * 70)
        print("TRAINING XGBOOST MODEL")
        print("=" * 70)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train model
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )

        print("Training...")
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )

        # Predictions
        y_pred = model.predict(X_val_scaled)

        # Calculate metrics
        metrics = self.calculate_metrics(y_val, y_pred)

        print(f"\nXGBoost Results:")
        print(f"  MAE:  {metrics['MAE']:.3f} kWh")
        print(f"  RMSE: {metrics['RMSE']:.3f} kWh")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  R²:   {metrics['R2']:.3f}")
        print(f"  Best Iteration: {model.best_iteration}")

        return {
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'predictions': y_pred,
            'feature_importance': model.feature_importances_
        }

    def train_lstm(
            self,
            X_train: np.ndarray,
            X_val: np.ndarray,
            y_train: np.ndarray,
            y_val: np.ndarray,
            lookback: int = 24,
            epochs: int = 50,
            batch_size: int = 64
    ) -> Dict:
        """
        Train LSTM neural network.

        LSTM requires 3D input: (samples, timesteps, features)
        We reshape data to use last 'lookback' hours as sequence.

        Args:
            X_train, X_val: Feature matrices
            y_train, y_val: Target values
            lookback: Number of past timesteps to use
            epochs: Training epochs
            batch_size: Batch size for training

        Returns:
            Dictionary with model, metrics, and predictions
        """
        print("\n" + "=" * 70)
        print("TRAINING LSTM MODEL")
        print("=" * 70)
        print(f"Lookback window: {lookback} hours")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Reshape for LSTM (samples, timesteps, features)
        def create_sequences(X, y, lookback):
            Xs, ys = [], []
            for i in range(lookback, len(X)):
                Xs.append(X[i - lookback:i])
                ys.append(y[i])
            return np.array(Xs), np.array(ys)

        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, lookback)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, lookback)

        print(f"Sequence shapes:")
        print(f"  X_train: {X_train_seq.shape}")
        print(f"  X_val:   {X_val_seq.shape}")

        # Build LSTM model
        model = Sequential([
            LSTM(128, activation='relu', return_sequences=True,
                 input_shape=(lookback, X_train_seq.shape[2])),
            Dropout(0.2),
            LSTM(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        print("\nModel Architecture:")
        model.summary()

        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )

        print("\nTraining...")
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )

        # Predictions
        y_pred = model.predict(X_val_seq, verbose=0).flatten()

        # Calculate metrics
        metrics = self.calculate_metrics(y_val_seq, y_pred)

        print(f"\nLSTM Results:")
        print(f"  MAE:  {metrics['MAE']:.3f} kWh")
        print(f"  RMSE: {metrics['RMSE']:.3f} kWh")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  R²:   {metrics['R2']:.3f}")
        print(f"  Epochs trained: {len(history.history['loss'])}")

        return {
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'predictions': y_pred,
            'history': history.history,
            'lookback': lookback
        }

    def train_all_models(
            self,
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            feature_names: List[str]
    ) -> Dict:
        """
        Train all models and store results.

        Args:
            X_train, X_val, X_test: Feature matrices
            y_train, y_val, y_test: Target values
            feature_names: List of feature names

        Returns:
            Dictionary of all model results
        """
        print("\n" + "=" * 70)
        print("TRAINING ALL MODELS")
        print("=" * 70)

        results = {}

        # 1. ARIMA (baseline)
        if HAS_STATSMODELS:
            arima_result = self.train_arima(y_train, y_val)
            if arima_result:
                results['ARIMA'] = arima_result
        else:
            print("\n[Skipping] ARIMA - statsmodels not available")

        # 2. Random Forest
        if HAS_SKLEARN:
            rf_result = self.train_random_forest(X_train, X_val, y_train, y_val)
            results['RandomForest'] = rf_result
        else:
            print("\n[Skipping] Random Forest - scikit-learn not available")

        # 3. XGBoost
        if HAS_XGBOOST:
            xgb_result = self.train_xgboost(X_train, X_val, y_train, y_val)
            results['XGBoost'] = xgb_result
        else:
            print("\n[Skipping] XGBoost - xgboost not available")

        # 4. LSTM
        if HAS_TENSORFLOW:
            lstm_result = self.train_lstm(X_train, X_val, y_train, y_val)
            results['LSTM'] = lstm_result
        else:
            print("\n[Skipping] LSTM - tensorflow not available")

        # Store results
        self.results = results
        self.feature_names = feature_names

        return results

    def compare_models(self) -> pd.DataFrame:
        """
        Create comparison table of all models.

        Returns:
            DataFrame with model comparison
        """
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)

        comparison_data = []

        for model_name, result in self.results.items():
            if result and 'metrics' in result:
                metrics = result['metrics']
                comparison_data.append({
                    'Model': model_name,
                    'MAE (kWh)': f"{metrics['MAE']:.3f}",
                    'RMSE (kWh)': f"{metrics['RMSE']:.3f}",
                    'MAPE (%)': f"{metrics['MAPE']:.2f}",
                    'R² Score': f"{metrics['R2']:.3f}"
                })

        comparison_df = pd.DataFrame(comparison_data)
        print("\n", comparison_df.to_string(index=False))

        # Determine best model (lowest MAPE)
        best_idx = comparison_df['MAPE (%)'].astype(float).idxmin()
        best_model = comparison_df.loc[best_idx, 'Model']

        print(f"\n✓ Best Model: {best_model} (lowest MAPE)")

        return comparison_df

    def save_models(self, output_dir: str = "models/"):
        """Save all trained models to disk."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print("SAVING MODELS")
        print("=" * 70)

        for model_name, result in self.results.items():
            if result is None:
                continue

            try:
                if model_name == 'LSTM':
                    # Save Keras model
                    result['model'].save(f"{output_dir}lstm_model.h5")
                    # Save scaler separately
                    with open(f"{output_dir}lstm_scaler.pkl", 'wb') as f:
                        pickle.dump(result['scaler'], f)
                    print(f"✓ Saved {model_name} model")
                else:
                    # Save sklearn/statsmodels models
                    with open(f"{output_dir}{model_name.lower()}_model.pkl", 'wb') as f:
                        pickle.dump(result, f)
                    print(f"✓ Saved {model_name} model")
            except Exception as e:
                print(f"✗ Failed to save {model_name}: {e}")


if __name__ == "__main__":
    # Load featured data
    df = pd.read_csv("data/processed/energy_data_featured.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Loaded data: {df.shape}")

    # Initialize model trainer
    trainer = EnergyPredictionModels(random_state=42)

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = trainer.prepare_data(df)

    # Train all models
    results = trainer.train_all_models(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        feature_names
    )

    # Compare models
    comparison_df = trainer.compare_models()

    # Save comparison
    comparison_df.to_csv("results/model_comparison.csv", index=False)
    print("\n✓ Comparison saved to: results/model_comparison.csv")

    # Save models
    trainer.save_models()