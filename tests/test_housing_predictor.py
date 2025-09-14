"""
Tests for the Housing Price Predictor.

This module contains comprehensive tests for all components of the
housing price predictor system.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from housing_predictor import HousingPredictor, DataLoader, ModelFactory
from housing_predictor.core.predictor import HousingPredictor as CorePredictor
from housing_predictor.core.data_loader import DataLoader as CoreDataLoader
from housing_predictor.core.models import ModelFactory as CoreModelFactory
from housing_predictor.config.settings import Config


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.data_loader = CoreDataLoader()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_california_housing_load(self):
        """Test loading California housing dataset."""
        X, y = self.data_loader.load_california_housing()
        
        assert X is not None
        assert y is not None
        assert X.shape[0] > 0
        assert len(y) == X.shape[0]
        assert X.shape[1] == 8  # California housing has 8 features
    
    def test_synthetic_dataset_creation(self):
        """Test synthetic dataset creation."""
        X, y = self.data_loader.create_synthetic_dataset(n_samples=100)
        
        assert X is not None
        assert y is not None
        assert X.shape[0] == 100
        assert len(y) == 100
        assert X.shape[1] > 0
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, np.ndarray)
    
    def test_data_preprocessing(self):
        """Test data preprocessing."""
        # Create test data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, np.nan, 5],
            'feature2': ['A', 'B', 'A', 'C', 'B'],
            'feature3': [10, 20, 30, 40, 50]
        })
        y = np.array([100, 200, 300, 400, 500])
        
        X_processed, y_processed, info = self.data_loader.preprocess_data(X, y)
        
        assert X_processed is not None
        assert y_processed is not None
        assert not np.isnan(X_processed).any()  # No NaN values
        assert X_processed.shape[0] == X.shape[0]
    
    def test_data_splitting(self):
        """Test data splitting functionality."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        X_train, X_test, y_train, y_test = self.data_loader.split_data(X, y, test_size=0.2)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_test) == 20  # 20% of 100
        assert len(X_train) == 80  # 80% of 100


class TestModelFactory:
    """Test cases for ModelFactory class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model_factory = CoreModelFactory()
        self.X_train = np.random.randn(100, 5)
        self.y_train = np.random.randn(100)
        self.X_test = np.random.randn(20, 5)
        self.y_test = np.random.randn(20)
    
    def test_model_creation(self):
        """Test model creation."""
        models = ['linear_regression', 'random_forest', 'gradient_boosting', 'neural_network']
        
        for model_name in models:
            model = self.model_factory.create_model(model_name)
            assert model is not None
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')
    
    def test_model_training(self):
        """Test model training."""
        model = self.model_factory.create_model('linear_regression')
        trained_model = self.model_factory.train_model(model, self.X_train, self.y_train)
        
        assert trained_model is not None
        assert hasattr(trained_model, 'predict')
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        model = self.model_factory.create_model('linear_regression')
        trained_model = self.model_factory.train_model(model, self.X_train, self.y_train)
        
        metrics = self.model_factory.evaluate_model(trained_model, self.X_test, self.y_test)
        
        assert 'MSE' in metrics
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        assert 'R2' in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        model = self.model_factory.create_model('random_forest')
        trained_model = self.model_factory.train_model(model, self.X_train, self.y_train)
        
        importance = self.model_factory.get_feature_importance(trained_model, ['f1', 'f2', 'f3', 'f4', 'f5'])
        
        assert isinstance(importance, dict)
        assert len(importance) == 5
        assert all(isinstance(v, (int, float)) for v in importance.values())


class TestHousingPredictor:
    """Test cases for HousingPredictor class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.predictor = CorePredictor()
    
    def test_dataset_loading(self):
        """Test dataset loading."""
        X, y = self.predictor.load_dataset('california')
        
        assert X is not None
        assert y is not None
        assert X.shape[0] > 0
        assert len(y) == X.shape[0]
    
    def test_data_preprocessing(self):
        """Test data preprocessing."""
        X, y = self.predictor.load_dataset('california')
        X_processed, y_processed = self.predictor.preprocess_data(X, y)
        
        assert X_processed is not None
        assert y_processed is not None
        assert X_processed.shape[0] == X.shape[0]
    
    def test_data_splitting(self):
        """Test data splitting."""
        X, y = self.predictor.load_dataset('california')
        X_processed, y_processed = self.predictor.preprocess_data(X, y)
        self.predictor.split_data(X_processed, y_processed)
        
        assert self.predictor.training_data is not None
        assert self.predictor.test_data is not None
        assert len(self.predictor.training_data[0]) + len(self.predictor.test_data[0]) == len(X_processed)
    
    def test_model_training(self):
        """Test model training."""
        X, y = self.predictor.load_dataset('california')
        X_processed, y_processed = self.predictor.preprocess_data(X, y)
        self.predictor.split_data(X_processed, y_processed)
        
        self.predictor.train_models(['linear_regression', 'random_forest'])
        
        assert len(self.predictor.models) == 2
        assert 'Linear Regression' in self.predictor.models
        assert 'Random Forest' in self.predictor.models
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        X, y = self.predictor.load_dataset('california')
        X_processed, y_processed = self.predictor.preprocess_data(X, y)
        self.predictor.split_data(X_processed, y_processed)
        self.predictor.train_models(['linear_regression'])
        
        results = self.predictor.evaluate_models()
        
        assert len(results) == 1
        assert 'Linear Regression' in results
        assert 'RMSE' in results['Linear Regression']
        assert 'R2' in results['Linear Regression']
    
    def test_prediction(self):
        """Test making predictions."""
        X, y = self.predictor.load_dataset('california')
        X_processed, y_processed = self.predictor.preprocess_data(X, y)
        self.predictor.split_data(X_processed, y_processed)
        self.predictor.train_models(['linear_regression'])
        
        # Use first test sample for prediction
        X_test, y_test = self.predictor.test_data
        prediction = self.predictor.predict(X_test[:1])
        
        assert prediction is not None
        assert len(prediction) == 1
        assert isinstance(prediction[0], (int, float))
    
    def test_full_pipeline(self):
        """Test complete prediction pipeline."""
        results = self.predictor.run_full_pipeline('california', ['linear_regression'])
        
        assert 'evaluation_results' in results
        assert 'best_model' in results
        assert 'feature_names' in results
        assert 'data_shape' in results
        assert len(results['evaluation_results']) == 1


class TestConfig:
    """Test cases for Config class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yaml"
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        
        assert config.get('data.zillow_path') is not None
        assert config.get('models.random_forest') is not None
        assert config.get('training.test_size') is not None
    
    def test_config_get_set(self):
        """Test configuration get/set operations."""
        config = Config()
        
        # Test setting and getting values
        config.set('test.key', 'test_value')
        assert config.get('test.key') == 'test_value'
        
        # Test default value
        assert config.get('nonexistent.key', 'default') == 'default'
    
    def test_config_properties(self):
        """Test configuration properties."""
        config = Config()
        
        assert isinstance(config.data_paths, dict)
        assert isinstance(config.model_configs, dict)
        assert isinstance(config.training_config, dict)
        assert isinstance(config.logging_config, dict)


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_california_housing(self):
        """Test complete end-to-end workflow with California housing."""
        predictor = CorePredictor()
        
        # Run full pipeline
        results = predictor.run_full_pipeline(
            dataset_name='california',
            model_names=['linear_regression', 'random_forest']
        )
        
        # Verify results
        assert len(results['evaluation_results']) == 2
        assert results['best_model'] in results['evaluation_results']
        assert results['data_shape'][0] > 0
        assert results['data_shape'][1] == 8  # California housing features
    
    def test_end_to_end_synthetic_data(self):
        """Test complete end-to-end workflow with synthetic data."""
        predictor = CorePredictor()
        
        # Run full pipeline
        results = predictor.run_full_pipeline(
            dataset_name='synthetic',
            model_names=['linear_regression']
        )
        
        # Verify results
        assert len(results['evaluation_results']) == 1
        assert results['best_model'] in results['evaluation_results']
        assert results['data_shape'][0] > 0
    
    def test_model_performance_reasonable(self):
        """Test that model performance is reasonable."""
        predictor = CorePredictor()
        
        results = predictor.run_full_pipeline(
            dataset_name='california',
            model_names=['linear_regression']
        )
        
        # Check that R² is reasonable (not too negative)
        r2_score = results['evaluation_results']['Linear Regression']['R2']
        assert r2_score > -1.0  # Should not be extremely negative
        
        # Check that RMSE is reasonable
        rmse = results['evaluation_results']['Linear Regression']['RMSE']
        assert rmse > 0  # Should be positive


# Pytest fixtures
@pytest.fixture
def sample_data():
    """Sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    return X, y


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    np.random.seed(42)
    data = {
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randn(100)
    }
    return pd.DataFrame(data)


# Performance tests
@pytest.mark.slow
def test_large_dataset_performance():
    """Test performance with larger dataset."""
    predictor = CorePredictor()
    
    # Create larger synthetic dataset
    X, y = predictor.load_dataset('synthetic', n_samples=5000)
    
    # Should complete without errors
    assert X.shape[0] == 5000
    assert len(y) == 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
