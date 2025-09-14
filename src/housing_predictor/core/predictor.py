"""
Main Housing Price Predictor class.

This module provides the main interface for housing price prediction,
combining data loading, preprocessing, model training, and evaluation.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from .data_loader import DataLoader
from .models import ModelFactory
from ..utils.visualization import VisualizationUtils
from ..utils.preprocessing import PreprocessingUtils
from ..config.settings import config


class HousingPredictor:
    """
    Main housing price predictor class.
    
    This class provides a comprehensive interface for housing price prediction
    using various machine learning models and datasets.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the housing price predictor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = config
        self.data_loader = DataLoader()
        self.model_factory = ModelFactory()
        self.visualizer = VisualizationUtils()
        self.preprocessor = PreprocessingUtils()
        
        self.models: Dict[str, Any] = {}
        self.training_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.feature_names: Optional[List[str]] = None
        self.evaluation_results: Dict[str, Dict[str, float]] = {}
        
        logger.info("Housing Price Predictor initialized")
    
    def load_dataset(self, dataset_name: str = 'california', **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a dataset for training.
        
        Args:
            dataset_name: Name of dataset to load ('california', 'zillow', 'synthetic')
            **kwargs: Additional parameters for dataset loading
            
        Returns:
            Tuple of (X, y) features and targets
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        if dataset_name.lower() == 'california':
            X, y = self.data_loader.load_california_housing()
            self.feature_names = [
                'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                'Population', 'AveOccup', 'Latitude', 'Longitude'
            ]
            
        elif dataset_name.lower() == 'synthetic':
            n_samples = kwargs.get('n_samples', 1000)
            X, y = self.data_loader.create_synthetic_dataset(n_samples=n_samples)
            self.feature_names = list(X.columns)
            
        elif dataset_name.lower() == 'zillow':
            datasets = self.data_loader.load_zillow_data()
            if not datasets:
                raise ValueError("No Zillow data found. Please check data directory.")
            
            dataset_key = kwargs.get('dataset_key', 'zhvi')
            target_metros = kwargs.get('target_metros', None)
            min_data_points = kwargs.get('min_data_points', 24)
            
            X, y, metro_info = self.data_loader.prepare_time_series_data(
                datasets, dataset_key, target_metros, min_data_points
            )
            
            if X is None:
                raise ValueError("Failed to prepare Zillow time series data")
            
            self.feature_names = [f"month_{i}" for i in range(12)] + ['size_rank', 'data_length']
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def preprocess_data(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray,
                       dataset_name: str = 'default') -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for machine learning.
        
        Args:
            X: Feature data
            y: Target data
            dataset_name: Name for caching preprocessing steps
            
        Returns:
            Tuple of preprocessed (X, y)
        """
        logger.info("Preprocessing data...")
        
        if isinstance(X, pd.DataFrame):
            X_processed, y_processed, _ = self.data_loader.preprocess_data(
                X, y, dataset_name=dataset_name
            )
        else:
            X_processed = X
            y_processed = y
        
        logger.info(f"Data preprocessed: {X_processed.shape}")
        return X_processed, y_processed
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Split data into training and test sets.
        
        Args:
            X: Feature data
            y: Target data
            test_size: Proportion for test set
            random_state: Random state for reproducibility
        """
        X_train, X_test, y_train, y_test = self.data_loader.split_data(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.training_data = (X_train, y_train)
        self.test_data = (X_test, y_test)
        
        logger.info(f"Data split: {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    def train_models(self, model_names: Optional[List[str]] = None) -> None:
        """
        Train multiple models.
        
        Args:
            model_names: List of model names to train. If None, trains all default models.
        """
        if self.training_data is None:
            raise ValueError("No training data available. Call split_data() first.")
        
        X_train, y_train = self.training_data
        
        if model_names is None:
            models_to_train = self.model_factory.get_default_models()
        else:
            models_to_train = {}
            for name in model_names:
                try:
                    models_to_train[name] = self.model_factory.create_model(name)
                except Exception as e:
                    logger.error(f"Failed to create model {name}: {e}")
        
        logger.info(f"Training {len(models_to_train)} models...")
        
        for name, model in models_to_train.items():
            try:
                trained_model = self.model_factory.train_model(model, X_train, y_train, model_name=name)
                self.models[name] = trained_model
                logger.info(f"✅ {name} trained successfully")
            except Exception as e:
                logger.error(f"❌ Failed to train {name}: {e}")
        
        logger.info(f"Training completed: {len(self.models)} models trained")
    
    def evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models.
        
        Returns:
            Dictionary mapping model names to evaluation metrics
        """
        if not self.models:
            raise ValueError("No trained models available. Call train_models() first.")
        
        if self.test_data is None:
            raise ValueError("No test data available. Call split_data() first.")
        
        X_test, y_test = self.test_data
        
        logger.info("Evaluating models...")
        
        for name, model in self.models.items():
            try:
                metrics = self.model_factory.evaluate_model(model, X_test, y_test, model_name=name)
                self.evaluation_results[name] = metrics
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")
        
        logger.info("Model evaluation completed")
        return self.evaluation_results
    
    def predict(self, X: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            X: Features for prediction
            model_name: Name of model to use. If None, uses best model.
            
        Returns:
            Predictions
        """
        if not self.models:
            raise ValueError("No trained models available.")
        
        if model_name is None:
            # Use best model based on R² score
            if self.evaluation_results:
                best_model = max(self.evaluation_results.keys(), 
                               key=lambda x: self.evaluation_results[x].get('R2', 0))
                model_name = best_model
            else:
                model_name = list(self.models.keys())[0]
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        return self.model_factory.predict(self.models[model_name], X)
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> Dict[str, float]:
        """
        Get feature importance from a model.
        
        Args:
            model_name: Name of model. If None, uses best model.
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.models:
            raise ValueError("No trained models available.")
        
        if model_name is None:
            if self.evaluation_results:
                best_model = max(self.evaluation_results.keys(), 
                               key=lambda x: self.evaluation_results[x].get('R2', 0))
                model_name = best_model
            else:
                model_name = list(self.models.keys())[0]
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        return self.model_factory.get_feature_importance(
            self.models[model_name], self.feature_names
        )
    
    def plot_feature_importance(self, model_name: Optional[str] = None, 
                               top_n: int = 10) -> None:
        """
        Plot feature importance.
        
        Args:
            model_name: Name of model. If None, uses best model.
            top_n: Number of top features to show
        """
        importance = self.get_feature_importance(model_name)
        
        if not importance:
            logger.warning("No feature importance available")
            return
        
        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        self.visualizer.plot_feature_importance(sorted_features, model_name or "Model")
    
    def plot_model_comparison(self) -> None:
        """Plot comparison of all trained models."""
        if not self.evaluation_results:
            logger.warning("No evaluation results available")
            return
        
        self.visualizer.plot_model_comparison(self.evaluation_results)
    
    def plot_predictions(self, model_name: Optional[str] = None) -> None:
        """
        Plot actual vs predicted values.
        
        Args:
            model_name: Name of model. If None, uses best model.
        """
        if self.test_data is None:
            logger.warning("No test data available")
            return
        
        X_test, y_test = self.test_data
        
        if model_name is None:
            if self.evaluation_results:
                best_model = max(self.evaluation_results.keys(), 
                               key=lambda x: self.evaluation_results[x].get('R2', 0))
                model_name = best_model
            else:
                model_name = list(self.models.keys())[0]
        
        y_pred = self.predict(X_test, model_name)
        self.visualizer.plot_predictions(y_test, y_pred, model_name)
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of model to save
            filepath: Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        self.model_factory.save_model(self.models[model_name], filepath)
    
    def load_model(self, model_name: str, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name for the loaded model
            filepath: Path to the saved model
        """
        model = self.model_factory.load_model(filepath)
        self.models[model_name] = model
        logger.info(f"Model {model_name} loaded from {filepath}")
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get summary of all trained models.
        
        Returns:
            DataFrame with model performance metrics
        """
        if not self.evaluation_results:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, metrics in self.evaluation_results.items():
            summary_data.append({
                'Model': model_name,
                'RMSE': metrics.get('RMSE', 0),
                'MAE': metrics.get('MAE', 0),
                'R²': metrics.get('R2', 0)
            })
        
        return pd.DataFrame(summary_data).sort_values('R²', ascending=False)
    
    def run_full_pipeline(self, dataset_name: str = 'california', 
                         model_names: Optional[List[str]] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        Run the complete prediction pipeline.
        
        Args:
            dataset_name: Name of dataset to use
            model_names: List of models to train
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with results and metrics
        """
        logger.info("Starting full prediction pipeline...")
        
        # Load data
        X, y = self.load_dataset(dataset_name, **kwargs)
        
        # Preprocess data
        X_processed, y_processed = self.preprocess_data(X, y, dataset_name)
        
        # Split data
        self.split_data(X_processed, y_processed)
        
        # Train models
        self.train_models(model_names)
        
        # Evaluate models
        evaluation_results = self.evaluate_models()
        
        # Get best model
        best_model = max(evaluation_results.keys(), 
                        key=lambda x: evaluation_results[x].get('R2', 0))
        
        logger.info(f"Pipeline completed. Best model: {best_model}")
        
        return {
            'evaluation_results': evaluation_results,
            'best_model': best_model,
            'feature_names': self.feature_names,
            'data_shape': X_processed.shape
        }
