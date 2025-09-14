#!/usr/bin/env python3
"""
Command Line Interface for Housing Price Predictor.

This module provides a command-line interface for the housing price predictor,
allowing users to train models and make predictions from the terminal.
"""

import argparse
import sys
from pathlib import Path
import json
from typing import List, Optional, Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from housing_predictor import HousingPredictor
from housing_predictor.config.settings import config


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Housing Price Predictor CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models on California housing dataset
  housing-predictor train --dataset california --models linear_regression random_forest

  # Make prediction with specific features
  housing-predictor predict --model random_forest --features 8.3 41.0 6.9 1.0 322.0 2.5 37.9 -122.2

  # Evaluate model performance
  housing-predictor evaluate --dataset synthetic --models all

  # Show feature importance
  housing-predictor importance --model random_forest
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--dataset', choices=['california', 'synthetic', 'zillow'],
                            default='california', help='Dataset to use')
    train_parser.add_argument('--models', nargs='+', 
                            choices=['linear_regression', 'random_forest', 'gradient_boosting', 
                                   'neural_network', 'xgboost', 'lightgbm', 'all'],
                            default=['linear_regression', 'random_forest'],
                            help='Models to train')
    train_parser.add_argument('--test-size', type=float, default=0.2,
                            help='Test set size (default: 0.2)')
    train_parser.add_argument('--random-state', type=int, default=42,
                            help='Random state for reproducibility')
    train_parser.add_argument('--output', type=str, help='Output file for results')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model', required=True,
                              choices=['linear_regression', 'random_forest', 'gradient_boosting',
                                     'neural_network', 'xgboost', 'lightgbm'],
                              help='Model to use for prediction')
    predict_parser.add_argument('--features', nargs='+', type=float, required=True,
                              help='Feature values for prediction')
    predict_parser.add_argument('--model-file', type=str,
                              help='Path to saved model file')
    predict_parser.add_argument('--output', type=str, help='Output file for prediction')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--dataset', choices=['california', 'synthetic', 'zillow'],
                           default='california', help='Dataset to use')
    eval_parser.add_argument('--models', nargs='+',
                           choices=['linear_regression', 'random_forest', 'gradient_boosting',
                                  'neural_network', 'xgboost', 'lightgbm', 'all'],
                           default=['all'], help='Models to evaluate')
    eval_parser.add_argument('--output', type=str, help='Output file for results')
    
    # Feature importance command
    importance_parser = subparsers.add_parser('importance', help='Show feature importance')
    importance_parser.add_argument('--model', required=True,
                                 choices=['linear_regression', 'random_forest', 'gradient_boosting',
                                        'neural_network', 'xgboost', 'lightgbm'],
                                 help='Model to analyze')
    importance_parser.add_argument('--dataset', choices=['california', 'synthetic', 'zillow'],
                                 default='california', help='Dataset to use')
    importance_parser.add_argument('--top-n', type=int, default=10,
                                 help='Number of top features to show')
    importance_parser.add_argument('--output', type=str, help='Output file for results')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show configuration')
    config_parser.add_argument('--key', type=str, help='Specific config key to show')
    
    return parser


def train_models(args: argparse.Namespace) -> None:
    """Train models based on command line arguments."""
    print(f"🏠 Training models on {args.dataset} dataset...")
    
    # Initialize predictor
    predictor = HousingPredictor()
    
    # Determine models to train
    if 'all' in args.models:
        models_to_train = None  # Use all default models
    else:
        models_to_train = args.models
    
    try:
        # Run full pipeline
        results = predictor.run_full_pipeline(
            dataset_name=args.dataset,
            model_names=models_to_train
        )
        
        print("✅ Training completed successfully!")
        print(f"📊 Best model: {results['best_model']}")
        print(f"📈 R² Score: {results['evaluation_results'][results['best_model']]['R2']:.4f}")
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to {args.output}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        sys.exit(1)


def make_prediction(args: argparse.Namespace) -> None:
    """Make prediction based on command line arguments."""
    print(f"🔮 Making prediction with {args.model} model...")
    
    # Initialize predictor
    predictor = HousingPredictor()
    
    try:
        # Load model if file specified
        if args.model_file:
            predictor.load_model(args.model, args.model_file)
        else:
            # Train model first
            print("Training model first...")
            results = predictor.run_full_pipeline(
                dataset_name='california',  # Default dataset
                model_names=[args.model]
            )
        
        # Make prediction
        features_array = np.array(args.features).reshape(1, -1)
        prediction = predictor.predict(features_array, args.model)
        
        print(f"🏠 Predicted Price: ${prediction[0]:,.2f}")
        
        # Save prediction if output file specified
        if args.output:
            result = {
                'model': args.model,
                'features': args.features,
                'prediction': float(prediction[0])
            }
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Prediction saved to {args.output}")
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        sys.exit(1)


def evaluate_models(args: argparse.Namespace) -> None:
    """Evaluate model performance."""
    print(f"📊 Evaluating models on {args.dataset} dataset...")
    
    # Initialize predictor
    predictor = HousingPredictor()
    
    # Determine models to evaluate
    if 'all' in args.models:
        models_to_evaluate = None
    else:
        models_to_evaluate = args.models
    
    try:
        # Run evaluation
        results = predictor.run_full_pipeline(
            dataset_name=args.dataset,
            model_names=models_to_evaluate
        )
        
        print("📈 Model Performance:")
        print("-" * 50)
        
        for model_name, metrics in results['evaluation_results'].items():
            print(f"{model_name}:")
            print(f"  RMSE: ${metrics['RMSE']:,.2f}")
            print(f"  MAE:  ${metrics['MAE']:,.2f}")
            print(f"  R²:   {metrics['R2']:.4f}")
            print()
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to {args.output}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        sys.exit(1)


def show_feature_importance(args: argparse.Namespace) -> None:
    """Show feature importance for a model."""
    print(f"🎯 Analyzing feature importance for {args.model} model...")
    
    # Initialize predictor
    predictor = HousingPredictor()
    
    try:
        # Train model first
        results = predictor.run_full_pipeline(
            dataset_name=args.dataset,
            model_names=[args.model]
        )
        
        # Get feature importance
        importance = predictor.get_feature_importance(args.model)
        
        if not importance:
            print("❌ Feature importance not available for this model.")
            return
        
        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"📊 Top {args.top_n} Most Important Features:")
        print("-" * 50)
        
        for i, (feature, score) in enumerate(sorted_features[:args.top_n], 1):
            print(f"{i:2d}. {feature:20s}: {score:.4f}")
        
        # Save results if output file specified
        if args.output:
            result = {
                'model': args.model,
                'dataset': args.dataset,
                'feature_importance': dict(sorted_features)
            }
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Results saved to {args.output}")
        
    except Exception as e:
        print(f"❌ Error analyzing feature importance: {e}")
        sys.exit(1)


def show_config(args: argparse.Namespace) -> None:
    """Show configuration."""
    if args.key:
        value = config.get(args.key)
        print(f"{args.key}: {value}")
    else:
        print("📋 Current Configuration:")
        print("-" * 30)
        print(f"Data paths: {config.data_paths}")
        print(f"Model configs: {list(config.model_configs.keys())}")
        print(f"Training config: {config.training_config}")
        print(f"Logging config: {config.logging_config}")


def main():
    """Main CLI function."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Import numpy here to avoid import issues
    import numpy as np
    
    # Execute command
    if args.command == 'train':
        train_models(args)
    elif args.command == 'predict':
        make_prediction(args)
    elif args.command == 'evaluate':
        evaluate_models(args)
    elif args.command == 'importance':
        show_feature_importance(args)
    elif args.command == 'config':
        show_config(args)


if __name__ == "__main__":
    main()
