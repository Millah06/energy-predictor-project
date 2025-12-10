"""
Smart Energy Consumption Predictor
File: visualizations.py

Purpose: Flask REST API for energy consumption predictions
Author: Abdullahi Aliyu Garba
Date: December 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

# Optional imports for visualization
try:
    import matplotlib
    # Use non-interactive backend (Agg) to avoid GUI/Tkinter issues
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
    # Set style
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    sns.set_palette("husl")
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib/seaborn not available. Visualization features will be disabled.")


class EnergyVisualizationSuite:
    """
    Comprehensive visualization suite for energy prediction analysis.

    Creates:
    - Actual vs Predicted plots
    - Residual analysis
    - Feature importance charts
    - Error distribution
    - Time-series decomposition
    - Model comparison charts
    """

    def __init__(self, figsize: tuple = (15, 8)):
        """Initialize visualization suite."""
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib seaborn")
        self.figsize = figsize
        self.colors = {
            'actual': '#2E86DE',
            'predicted': '#EE5A6F',
            'residual': '#26DE81',
            'error': '#FC5C65'
        }

    def plot_predictions_vs_actual(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            timestamps: pd.Series = None,
            model_name: str = "Model",
            save_path: str = None,
            show_n_points: int = 500
    ):
        """
        Plot actual vs predicted values over time.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            timestamps: Optional timestamps for x-axis
            model_name: Name of the model for title
            save_path: Path to save figure
            show_n_points: Number of points to display (for readability)
        """
        # Use only last N points for readability
        if len(y_true) > show_n_points:
            y_true = y_true[-show_n_points:]
            y_pred = y_pred[-show_n_points:]
            if timestamps is not None:
                timestamps = timestamps[-show_n_points:]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[3, 1])

        # Create x-axis
        if timestamps is not None:
            x = timestamps
            xlabel = 'Time'
        else:
            x = np.arange(len(y_true))
            xlabel = 'Sample Index'

        # Top plot: Actual vs Predicted
        ax1.plot(x, y_true, label='Actual', color=self.colors['actual'],
                 linewidth=1.5, alpha=0.8)
        ax1.plot(x, y_pred, label='Predicted', color=self.colors['predicted'],
                 linewidth=1.5, linestyle='--', alpha=0.8)

        # Highlight large errors (>15% MAPE)
        errors = np.abs((y_true - y_pred) / np.clip(y_true, 0.1, None))
        large_errors = errors > 0.15
        if large_errors.any():
            ax1.scatter(np.array(x)[large_errors], y_true[large_errors],
                        color=self.colors['error'], s=50, zorder=5,
                        label='Large Error (>15%)', alpha=0.6)

        ax1.set_ylabel('Energy Consumption (kWh)', fontsize=12, fontweight='bold')
        ax1.set_title(f'{model_name}: Actual vs Predicted Energy Consumption',
                      fontsize=14, fontweight='bold', pad=20)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Bottom plot: Residuals
        residuals = y_true - y_pred
        ax2.bar(x, residuals, color=self.colors['residual'], alpha=0.6, width=1)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax2.set_ylabel('Residual (kWh)', fontsize=12, fontweight='bold')
        ax2.set_title('Prediction Residuals (Actual - Predicted)',
                      fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.show()

    def plot_scatter_actual_vs_predicted(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            model_name: str = "Model",
            save_path: str = None
    ):
        """
        Create scatter plot of actual vs predicted values.

        Perfect predictions fall on the diagonal line.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color=self.colors['actual'])

        # Perfect prediction line (45-degree)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, label='Perfect Prediction')

        # Calculate metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))

        # Add metrics as text
        textstr = f'MAE: {mae:.3f} kWh\nRMSE: {rmse:.3f} kWh\nR²: {r2:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

        ax.set_xlabel('Actual Energy Consumption (kWh)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Energy Consumption (kWh)', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name}: Actual vs Predicted Scatter Plot',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.show()

    def plot_residual_analysis(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            model_name: str = "Model",
            save_path: str = None
    ):
        """
        Create comprehensive residual analysis plots.

        Includes:
        - Residual distribution (histogram)
        - Residual vs predicted (heteroscedasticity check)
        - Q-Q plot (normality check)

        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save figure
        """
        residuals = y_true - y_pred

        fig = plt.figure(figsize=(18, 5))

        # 1. Residual Distribution
        ax1 = plt.subplot(131)
        ax1.hist(residuals, bins=50, color=self.colors['residual'],
                 alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax1.axvline(x=residuals.mean(), color='blue', linestyle='--',
                    linewidth=2, label=f'Mean: {residuals.mean():.3f}')
        ax1.set_xlabel('Residual (kWh)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Residuals vs Predicted
        ax2 = plt.subplot(132)
        ax2.scatter(y_pred, residuals, alpha=0.5, s=20, color=self.colors['actual'])
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)

        # Add smoothed trend line
        from scipy.ndimage import uniform_filter1d
        sorted_idx = np.argsort(y_pred)
        smoothed = uniform_filter1d(residuals[sorted_idx], size=50)
        ax2.plot(y_pred[sorted_idx], smoothed, color='orange',
                 linewidth=2, label='Smoothed Trend')

        ax2.set_xlabel('Predicted Value (kWh)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Residual (kWh)', fontsize=11, fontweight='bold')
        ax2.set_title('Residuals vs Predicted (Heteroscedasticity Check)',
                      fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Q-Q Plot (Normality Check)
        ax3 = plt.subplot(133)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        plt.suptitle(f'{model_name}: Residual Analysis',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.show()

    def plot_feature_importance(
            self,
            feature_importance: np.ndarray,
            feature_names: List[str],
            model_name: str = "Model",
            top_n: int = 20,
            save_path: str = None
    ):
        """
        Plot feature importance bar chart.

        Args:
            feature_importance: Array of importance scores
            feature_names: List of feature names
            model_name: Name of the model
            top_n: Number of top features to display
            save_path: Path to save figure
        """
        # Sort features by importance
        indices = np.argsort(feature_importance)[-top_n:]

        fig, ax = plt.subplots(figsize=(12, 8))

        # Create horizontal bar chart
        bars = ax.barh(range(len(indices)), feature_importance[indices],
                       color=self.colors['actual'], alpha=0.7)

        # Color top 3 features differently
        for i, bar in enumerate(bars[-3:]):
            bar.set_color(self.colors['predicted'])

        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name}: Top {top_n} Feature Importance',
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')

        # Add importance values on bars
        for i, (idx, imp) in enumerate(zip(indices, feature_importance[indices])):
            ax.text(imp, i, f' {imp:.4f}', va='center', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.show()

    def plot_model_comparison(
            self,
            comparison_df: pd.DataFrame,
            save_path: str = None
    ):
        """
        Create model comparison visualization.

        Args:
            comparison_df: DataFrame with model comparison metrics
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        models = comparison_df['Model'].values
        metrics = ['MAE (kWh)', 'RMSE (kWh)', 'MAPE (%)', 'R² Score']

        for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
            values = comparison_df[metric].astype(float).values

            # Create bar plot
            bars = ax.bar(models, values, color=self.colors['actual'], alpha=0.7)

            # Highlight best model (depends on metric)
            if metric == 'R² Score':
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)

            bars[best_idx].set_color(self.colors['predicted'])
            bars[best_idx].set_alpha(1.0)

            ax.set_ylabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(f'Model Comparison: {metric}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Rotate x-axis labels
            ax.set_xticklabels(models, rotation=45, ha='right')

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{value:.3f}' if metric != 'MAPE (%)' else f'{value:.2f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.show()

    def plot_error_distribution(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            model_name: str = "Model",
            save_path: str = None
    ):
        """
        Plot error distribution analysis.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save figure
        """
        # Calculate percentage errors
        percentage_errors = np.abs((y_true - y_pred) / np.clip(y_true, 0.1, None)) * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Absolute error distribution
        absolute_errors = np.abs(y_true - y_pred)
        ax1.hist(absolute_errors, bins=50, color=self.colors['residual'],
                 alpha=0.7, edgecolor='black')
        ax1.axvline(x=absolute_errors.mean(), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {absolute_errors.mean():.3f} kWh')
        ax1.axvline(x=np.median(absolute_errors), color='blue', linestyle='--',
                    linewidth=2, label=f'Median: {np.median(absolute_errors):.3f} kWh')
        ax1.set_xlabel('Absolute Error (kWh)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Percentage error distribution
        ax2.hist(percentage_errors, bins=50, color=self.colors['predicted'],
                 alpha=0.7, edgecolor='black')
        ax2.axvline(x=percentage_errors.mean(), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {percentage_errors.mean():.2f}%')
        ax2.axvline(x=np.median(percentage_errors), color='blue', linestyle='--',
                    linewidth=2, label=f'Median: {np.median(percentage_errors):.2f}%')
        ax2.set_xlabel('Percentage Error (%)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Percentage Error Distribution (MAPE)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f'{model_name}: Error Analysis', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")

        plt.show()


if __name__ == "_main_":
    import pickle

    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Load data
    df = pd.read_csv("data/processed/energy_data_featured.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Load models
    with open("models/xgboost_model.pkl", 'rb') as f:
        xgb_result = pickle.load(f)

    # Get test set predictions
    target_col = 'energy_kwh'
    feature_cols = [col for col in df.columns if col not in ['timestamp', target_col]]

    # Time-series split (last 15% for test)
    test_size = 0.15
    n_samples = len(df)
    test_start = int(n_samples * (1 - test_size))

    X_test = df[feature_cols].iloc[test_start:].values
    y_test = df[target_col].iloc[test_start:].values
    timestamps_test = df['timestamp'].iloc[test_start:]

    # Scale and predict
    X_test_scaled = xgb_result['scaler'].transform(X_test)
    y_pred = xgb_result['model'].predict(X_test_scaled)

    # Initialize visualizer
    viz = EnergyVisualizationSuite()

    # Create visualizations
    print("\n1. Actual vs Predicted Time Series...")
    viz.plot_predictions_vs_actual(
        y_test, y_pred, timestamps_test,
        model_name="XGBoost",
        save_path="results/predictions_vs_actual.png"
    )

    print("\n2. Scatter Plot...")
    viz.plot_scatter_actual_vs_predicted(
        y_test, y_pred,
        model_name="XGBoost",
        save_path="results/scatter_plot.png"
    )

    print("\n3. Residual Analysis...")
    viz.plot_residual_analysis(
        y_test, y_pred,
        model_name="XGBoost",
        save_path="results/residual_analysis.png"
    )

    print("\n4. Feature Importance...")
    viz.plot_feature_importance(
        xgb_result['feature_importance'],
        feature_cols,
        model_name="XGBoost",
        save_path="results/feature_importance.png"
    )

    print("\n5. Error Distribution...")
    viz.plot_error_distribution(
        y_test, y_pred,
        model_name="XGBoost",
        save_path="results/error_distribution.png"
    )

    print("\n6. Model Comparison...")
    comparison_df = pd.read_csv("results/model_comparison.csv")
    viz.plot_model_comparison(
        comparison_df,
        save_path="results/model_comparison.png"
    )

    print("\n" + "=" * 70)
    print("ALL VISUALIZATIONS COMPLETE")
    print("=" * 70)
    print("Check 'results/' folder for saved plots")