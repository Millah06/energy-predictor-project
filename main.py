"""
Smart Energy Consumption Predictor
File: main.py

Purpose: Main execution script - runs entire pipeline
Author: AbdullahiAliyu Garba
Date: December 2025

This script orchestrates the complete pipeline:
1. Data generation
2. Data preprocessing
3. Feature engineering
4. Model training
5. Visualization
"""

import os
import sys
from datetime import datetime


def create_directories():
    """Create necessary project directories."""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'results',
        'logs'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("[OK] Created project directories")


def run_pipeline(skip_data_generation=False):
    """
    Run the complete ML pipeline.

    Args:
        skip_data_generation: If True, uses existing data instead of generating new
    """

    print("\n" + "=" * 70)
    print("SMART ENERGY CONSUMPTION PREDICTOR")
    print("Complete ML Pipeline Execution")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Create directories
    create_directories()

    try:
        # ====================================================================
        # STEP 1: DATA GENERATION
        # ====================================================================
        if not skip_data_generation:
            print("\n" + "=" * 70)
            print("STEP 1/5: DATA GENERATION")
            print("=" * 70)

            import importlib
            gen_module = importlib.import_module('generate_data')
            generate_synthetic_energy_data = gen_module.generate_synthetic_energy_data

            df_raw = generate_synthetic_energy_data(
                start_date="2024-01-01",
                end_date="2025-12-01",
                freq="H"
            )

            df_raw.to_csv("data/raw/energy_data.csv", index=False)
            print("\n[OK] STEP 1 COMPLETE: Data generated and saved")
        else:
            print("\n[SKIP] STEP 1 SKIPPED: Using existing data")

        # ====================================================================
        # STEP 2: DATA PREPROCESSING
        # ====================================================================
        print("\n" + "=" * 70)
        print("STEP 2/5: DATA PREPROCESSING")
        print("=" * 70)

        import importlib
        prep_module = importlib.import_module('data_preprocessing')
        EnergyDataPreprocessor = prep_module.EnergyDataPreprocessor

        preprocessor = EnergyDataPreprocessor(verbose=True)
        df_clean = preprocessor.full_pipeline("data/raw/energy_data.csv")
        df_clean.to_csv("data/processed/energy_data_clean.csv", index=False)

        print("\n[OK] STEP 2 COMPLETE: Data cleaned and validated")

        # ====================================================================
        # STEP 3: FEATURE ENGINEERING
        # ====================================================================
        print("\n" + "=" * 70)
        print("STEP 3/5: FEATURE ENGINEERING")
        print("=" * 70)

        import importlib
        feat_module = importlib.import_module('feature_engineering')
        EnergyFeatureEngineer = feat_module.EnergyFeatureEngineer

        engineer = EnergyFeatureEngineer(verbose=True)
        df_featured = engineer.full_pipeline(df_clean)
        df_featured.to_csv("data/processed/energy_data_featured.csv", index=False)

        print(f"\n[OK] STEP 3 COMPLETE: {len(engineer.feature_names)} features created")

        # ====================================================================
        # STEP 4: MODEL TRAINING
        # ====================================================================
        print("\n" + "=" * 70)
        print("STEP 4/5: MODEL TRAINING")
        print("=" * 70)

        import importlib
        train_module = importlib.import_module('model_training')
        EnergyPredictionModels = train_module.EnergyPredictionModels

        trainer = EnergyPredictionModels(random_state=42)

        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = \
            trainer.prepare_data(df_featured)

        # Train all models
        results = trainer.train_all_models(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            feature_names
        )

        # Compare models
        comparison_df = trainer.compare_models()
        comparison_df.to_csv("results/model_comparison.csv", index=False)

        # Save models
        trainer.save_models()

        print("\n[OK] STEP 4 COMPLETE: All models trained and saved")

        # ====================================================================
        # STEP 5: VISUALIZATION
        # ====================================================================
        print("\n" + "=" * 70)
        print("STEP 5/5: VISUALIZATION")
        print("=" * 70)

        import importlib
        viz_module = importlib.import_module('visualizations')
        EnergyVisualizationSuite = viz_module.EnergyVisualizationSuite
        HAS_MATPLOTLIB = viz_module.HAS_MATPLOTLIB
        import pickle
        import os

        # Find the best available model for visualization
        model_files = {
            'xgboost': 'models/xgboost_model.pkl',
            'randomforest': 'models/randomforest_model.pkl',
            'arima': 'models/arima_model.pkl'
        }
        
        best_model_name = None
        best_model_result = None
        
        # Try to load models in order of preference
        for name, filepath in model_files.items():
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        model_result = pickle.load(f)
                    best_model_name = name
                    best_model_result = model_result
                    print(f"\nUsing {name} model for visualizations")
                    break
                except Exception as e:
                    print(f"[WARNING] Could not load {name} model: {e}")
                    continue
        
        if best_model_result is None:
            print("\n[WARNING] No trained models found for visualization. Skipping visualization step.")
            print("[OK] STEP 5 SKIPPED: No models available for visualization")
            return
        
        # Get test predictions
        if 'scaler' in best_model_result and best_model_result['scaler'] is not None:
            X_test_scaled = best_model_result['scaler'].transform(X_test)
            y_pred_test = best_model_result['model'].predict(X_test_scaled)
        else:
            # Model doesn't need scaling (e.g., ARIMA)
            y_pred_test = best_model_result['model'].predict(X_test) if hasattr(best_model_result['model'], 'predict') else None
            if y_pred_test is None:
                print("[WARNING] Could not generate predictions. Skipping visualization.")
                return

        # Check if matplotlib is available
        try:
            # HAS_MATPLOTLIB already imported above
            if not HAS_MATPLOTLIB:
                print("\n[WARNING] matplotlib/seaborn not available. Skipping visualization step.")
                print("[OK] STEP 5 SKIPPED: Install matplotlib and seaborn for visualizations")
                print("  Run: pip install matplotlib seaborn")
                return
        except ImportError:
            print("\n[WARNING] matplotlib/seaborn not available. Skipping visualization step.")
            print("[OK] STEP 5 SKIPPED: Install matplotlib and seaborn for visualizations")
            print("  Run: pip install matplotlib seaborn")
            return

        # Get timestamps for test set
        timestamps_test = df_featured['timestamp'].iloc[-len(y_test):]

        # Create visualizations
        viz = EnergyVisualizationSuite()

        print("\nGenerating visualizations...")

        model_display_name = best_model_name.replace('randomforest', 'Random Forest').replace('xgboost', 'XGBoost').replace('arima', 'ARIMA').title()
        
        viz.plot_predictions_vs_actual(
            y_test, y_pred_test, timestamps_test,
            model_name=model_display_name,
            save_path="results/predictions_vs_actual.png",
            show_n_points=500
        )

        viz.plot_scatter_actual_vs_predicted(
            y_test, y_pred_test,
            model_name=model_display_name,
            save_path="results/scatter_plot.png"
        )

        viz.plot_residual_analysis(
            y_test, y_pred_test,
            model_name=model_display_name,
            save_path="results/residual_analysis.png"
        )

        # Feature importance only available for tree-based models
        if 'feature_importance' in best_model_result and best_model_result['feature_importance'] is not None:
            viz.plot_feature_importance(
                best_model_result['feature_importance'],
                feature_names,
                model_name=model_display_name,
                top_n=20,
                save_path="results/feature_importance.png"
            )

        viz.plot_error_distribution(
            y_test, y_pred_test,
            model_name=model_display_name,
            save_path="results/error_distribution.png"
        )

        viz.plot_model_comparison(
            comparison_df,
            save_path="results/model_comparison.png"
        )

        print("\n[OK] STEP 5 COMPLETE: All visualizations saved to results/")

        # ====================================================================
        # PIPELINE COMPLETE
        # ====================================================================
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION COMPLETE")
        print("=" * 70)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nGenerated Outputs:")
        print("  Data:")
        print("    - data/raw/energy_data.csv")
        print("    - data/processed/energy_data_clean.csv")
        print("    - data/processed/energy_data_featured.csv")
        print("  Models:")
        print("    - models/arima_model.pkl")
        print("    - models/randomforest_model.pkl")
        print("    - models/xgboost_model.pkl")
        print("    - models/lstm_model.h5")
        print("  Results:")
        print("    - results/model_comparison.csv")
        print("    - results/*.png (6 visualization plots)")
        print("\n" + "=" * 70)
        print("SUCCESS: All steps completed successfully!")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR IN PIPELINE EXECUTION")
        print("=" * 70)
        print(f"Error: {str(e)}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()
        print("=" * 70)
        sys.exit(1)


def show_help():
    """Display help information."""
    help_text = """
    SMART ENERGY CONSUMPTION PREDICTOR
    ===================================

    Usage:
        python main.py [OPTIONS]

    Options:
        --skip-data-gen    Use existing data instead of generating new data
        --help             Show this help message

    Description:
        This script runs the complete ML pipeline for energy consumption prediction:
        1. Generate synthetic energy data (or use existing)
        2. Clean and preprocess data
        3. Engineer time-series features
        4. Train multiple models (ARIMA, Random Forest, XGBoost, LSTM)
        5. Generate comprehensive visualizations

    Examples:
        # Run complete pipeline (generates new data)
        python main.py

        # Run pipeline with existing data
        python main.py --skip-data-gen

    Requirements:
        - Python 3.8+
        - See requirements.txt for package dependencies
        - Install: pip install -r requirements.txt

    Output:
        - Cleaned datasets in data/processed/
        - Trained models in models/
        - Visualizations and reports in results/
    """
    print(help_text)


if __name__  == "__main__":
    # Parse command line arguments
    skip_data_gen = "--skip-data-gen" in sys.argv
    show_help_flag = "--help" in sys.argv or "-h" in sys.argv

    if show_help_flag:
        show_help()
    else:
        run_pipeline(skip_data_generation=skip_data_gen)
