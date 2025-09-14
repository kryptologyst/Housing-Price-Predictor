#!/usr/bin/env python3
"""
Quick start script for Housing Price Predictor.

This script provides a simple way to get started with the housing price predictor
without needing to understand the full API.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from housing_predictor import HousingPredictor


def quick_start():
    """Run a quick demonstration of the housing price predictor."""
    print("🏠 Housing Price Predictor - Quick Start")
    print("=" * 50)
    
    # Initialize predictor
    print("Initializing predictor...")
    predictor = HousingPredictor()
    
    # Run full pipeline with California housing data
    print("Running full pipeline with California housing data...")
    results = predictor.run_full_pipeline(
        dataset_name='california',
        model_names=['linear_regression', 'random_forest']
    )
    
    # Display results
    print("\n📊 Results:")
    print("-" * 30)
    
    for model_name, metrics in results['evaluation_results'].items():
        print(f"{model_name}:")
        print(f"  RMSE: ${metrics['RMSE']:,.2f}")
        print(f"  MAE:  ${metrics['MAE']:,.2f}")
        print(f"  R²:   {metrics['R2']:.4f}")
        print()
    
    print(f"🏆 Best Model: {results['best_model']}")
    
    # Make a sample prediction
    print("\n🔮 Sample Prediction:")
    print("-" * 30)
    
    # Use first test sample
    X_test, y_test = predictor.test_data
    prediction = predictor.predict(X_test[:1])
    actual = y_test[0]
    
    print(f"Actual Price:    ${actual * 100000:,.2f}")
    print(f"Predicted Price: ${prediction[0] * 100000:,.2f}")
    print(f"Error:           ${abs(actual - prediction[0]) * 100000:,.2f}")
    
    print("\n✅ Quick start completed!")
    print("\nNext steps:")
    print("- Run 'streamlit run web_app/app.py' for web interface")
    print("- Run 'python web_app/cli.py --help' for CLI options")
    print("- Check tests/ directory for more examples")


if __name__ == "__main__":
    quick_start()
