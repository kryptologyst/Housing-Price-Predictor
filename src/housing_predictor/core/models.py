"""
Model factory and management for the Housing Price Predictor.

This module provides a factory pattern for creating and managing various
machine learning models with consistent interfaces.
"""

from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from loguru import logger

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

from ..config.settings import config


class ModelFactory:
    """Factory for creating and managing ML models."""
    
    def __init__(self):
        """Initialize model factory."""
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.model_configs = config.model_configs
        
    def create_model(self, model_name: str, **kwargs) -> Any:
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Additional parameters for model creation
            
        Returns:
            Model instance
        """
        model_name = model_name.lower().replace(' ', '_')
        
        if model_name == 'linear_regression':
            return self._create_linear_regression(**kwargs)
        elif model_name == 'random_forest':
            return self._create_random_forest(**kwargs)
        elif model_name == 'gradient_boosting':
            return self._create_gradient_boosting(**kwargs)
        elif model_name == 'neural_network':
            return self._create_neural_network(**kwargs)
        elif model_name == 'xgboost':
            return self._create_xgboost(**kwargs)
        elif model_name == 'lightgbm':
            return self._create_lightgbm(**kwargs)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _create_linear_regression(self, **kwargs) -> LinearRegression:
        """Create Linear Regression model."""
        params = self.model_configs.get('linear_regression', {})
        params.update(kwargs)
        return LinearRegression(**params)
    
    def _create_random_forest(self, **kwargs) -> RandomForestRegressor:
        """Create Random Forest model."""
        params = self.model_configs.get('random_forest', {})
        params.update(kwargs)
        return RandomForestRegressor(**params)
    
    def _create_gradient_boosting(self, **kwargs) -> GradientBoostingRegressor:
        """Create Gradient Boosting model."""
        params = self.model_configs.get('gradient_boosting', {})
        params.update(kwargs)
        return GradientBoostingRegressor(**params)
    
    def _create_neural_network(self, **kwargs) -> MLPRegressor:
        """Create Neural Network model."""
        params = self.model_configs.get('neural_network', {})
        params.update(kwargs)
        return MLPRegressor(**params)
    
    def _create_xgboost(self, **kwargs) -> Any:
        """Create XGBoost model."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        
        params = self.model_configs.get('xgboost', {})
        params.update(kwargs)
        return xgb.XGBRegressor(**params)
    
    def _create_lightgbm(self, **kwargs) -> Any:
        """Create LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")
        
        params = self.model_configs.get('lightgbm', {})
        params.update(kwargs)
        return lgb.LGBMRegressor(**params)
    
    def get_default_models(self) -> Dict[str, Any]:
        """
        Get dictionary of default models.
        
        Returns:
            Dictionary mapping model names to instances
        """
        models = {
            'Linear Regression': self.create_model('linear_regression'),
            'Random Forest': self.create_model('random_forest'),
            'Gradient Boosting': self.create_model('gradient_boosting'),
            'Neural Network': self.create_model('neural_network')
        }
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = self.create_model('xgboost')
        
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = self.create_model('lightgbm')
        
        return models
    
    def train_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                   model_name: str = 'model') -> Any:
        """
        Train a model with optional validation.
        
        Args:
            model: Model instance to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            model_name: Name of the model for logging
            
        Returns:
            Trained model
        """
        logger.info(f"Training {model_name}...")
        
        # Handle different model types
        if hasattr(model, 'fit'):
            if X_val is not None and y_val is not None and hasattr(model, 'eval_set'):
                # For models that support early stopping
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_train, y_train)
        else:
            raise ValueError(f"Model {model_name} does not have fit method")
        
        logger.info(f"{model_name} training completed")
        return model
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = 'model') -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        logger.info(f"{model_name} evaluation:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE:  {mae:.4f}")
        logger.info(f"  R²:   {r2:.4f}")
        
        return metrics
    
    def get_feature_importance(self, model: Any, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get feature importance from a model.
        
        Args:
            model: Trained model
            feature_names: Names of features
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            logger.warning("Model does not support feature importance")
            return {}
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        return dict(zip(feature_names, importance))
    
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model: Trained model
            X: Features for prediction
            
        Returns:
            Predictions
        """
        return model.predict(X)
    
    def save_model(self, model: Any, filepath: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model to save
            filepath: Path to save the model
        """
        import joblib
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        import joblib
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
