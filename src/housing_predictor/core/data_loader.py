"""
Data loading utilities for the Housing Price Predictor.

This module provides comprehensive data loading capabilities for various
housing datasets including Zillow data, California housing, and synthetic datasets.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from loguru import logger

from ..config.settings import config


class DataLoader:
    """Comprehensive data loader for housing datasets."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            cache_dir: Directory for caching processed data
        """
        self.cache_dir = Path(cache_dir or config.get('data.cache_dir', 'data/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
    def load_california_housing(self, return_X_y: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.DataFrame, pd.Series]]:
        """
        Load California housing dataset.
        
        Args:
            return_X_y: If True, return X, y as numpy arrays. If False, return DataFrames.
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        logger.info("Loading California Housing Dataset...")
        
        try:
            housing = fetch_california_housing()
            X = pd.DataFrame(housing.data, columns=housing.feature_names)
            y = housing.target
            
            logger.info(f"Loaded California Housing: {X.shape[0]} samples, {X.shape[1]} features")
            
            if return_X_y:
                return X.values, y
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading California housing data: {e}")
            raise
    
    def load_zillow_data(self, data_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load Zillow datasets from directory.
        
        Args:
            data_dir: Directory containing Zillow data files
            
        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        data_dir = Path(data_dir or config.get('data.zillow_path', 'data/zillow_data'))
        
        if not data_dir.exists():
            logger.warning(f"Zillow data directory not found: {data_dir}")
            return {}
        
        datasets = {}
        dataset_mapping = {
            'zhvi': 'Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
            'zori': 'Metro_zori_uc_sfrcondomfr_sm_month.csv',
            'inventory': 'Metro_invt_fs_uc_sfrcondo_sm_month.csv',
            'sales_count': 'Metro_sales_count_now_uc_sfrcondo_month.csv',
            'market_temp': 'Metro_market_temp_index_uc_sfrcondo_month.csv',
            'income_needed': 'Metro_new_homeowner_income_needed_downpayment_0.20_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv'
        }
        
        for key, filename in dataset_mapping.items():
            filepath = data_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    datasets[key] = df
                    logger.info(f"Loaded {key}: {df.shape[0]} metros, {df.shape[1]} columns")
                except Exception as e:
                    logger.error(f"Error loading {key}: {e}")
            else:
                logger.warning(f"File not found: {filepath}")
        
        return datasets
    
    def create_synthetic_dataset(self, n_samples: int = 1000, random_state: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Create a realistic synthetic housing dataset.
        
        Args:
            n_samples: Number of samples to generate
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X, y) where X is features DataFrame and y is target array
        """
        logger.info(f"Creating synthetic dataset with {n_samples} samples...")
        
        np.random.seed(random_state)
        
        # Generate realistic features
        data = {
            'sqft': np.random.normal(2000, 500, n_samples),
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.uniform(1, 4, n_samples),
            'age': np.random.randint(0, 50, n_samples),
            'garage_size': np.random.randint(0, 4, n_samples),
            'lot_size': np.random.normal(8000, 2000, n_samples),
            'neighborhood_score': np.random.uniform(1, 10, n_samples),
            'school_rating': np.random.uniform(1, 10, n_samples),
            'crime_rate': np.random.uniform(0, 10, n_samples),
            'distance_to_city': np.random.uniform(1, 50, n_samples),
            'has_pool': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'has_fireplace': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'condition': np.random.randint(1, 6, n_samples),
            'year_built': np.random.randint(1950, 2023, n_samples)
        }
        
        # Ensure positive values for size features
        for key in ['sqft', 'lot_size']:
            data[key] = np.abs(data[key])
        
        # Create realistic price based on features
        price = (
            data['sqft'] * 150 +
            data['bedrooms'] * 5000 +
            data['bathrooms'] * 8000 +
            (50 - data['age']) * 1000 +
            data['garage_size'] * 3000 +
            data['lot_size'] * 5 +
            data['neighborhood_score'] * 10000 +
            data['school_rating'] * 8000 +
            (10 - data['crime_rate']) * 5000 +
            (50 - data['distance_to_city']) * 2000 +
            data['has_pool'] * 25000 +
            data['has_fireplace'] * 5000 +
            data['condition'] * 10000 +
            (data['year_built'] - 1950) * 200 +
            np.random.normal(0, 20000, n_samples)  # Add noise
        )
        
        # Ensure reasonable price range
        price = np.clip(price, 100000, 2000000)
        
        X = pd.DataFrame(data)
        y = price
        
        logger.info(f"Synthetic dataset created: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Price range: ${y.min():,.0f} - ${y.max():,.0f}")
        
        return X, y
    
    def prepare_time_series_data(self, datasets: Dict[str, pd.DataFrame], 
                               dataset_key: str = 'zhvi',
                               target_metros: Optional[List[str]] = None,
                               min_data_points: int = 24) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[Dict[str, Any]]]:
        """
        Prepare time series data for machine learning.
        
        Args:
            datasets: Dictionary of loaded datasets
            dataset_key: Key of dataset to use
            target_metros: List of metro areas to focus on
            min_data_points: Minimum data points required per metro
            
        Returns:
            Tuple of (X, y, metro_info) or (None, None, []) if preparation fails
        """
        if dataset_key not in datasets:
            logger.error(f"Dataset {dataset_key} not found in loaded datasets")
            return None, None, []
        
        df = datasets[dataset_key].copy()
        
        # Get date columns (skip metadata columns)
        metadata_cols = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']
        date_cols = [col for col in df.columns if col not in metadata_cols]
        
        # Filter metros if specified
        if target_metros:
            df = df[df['RegionName'].isin(target_metros)]
        
        features = []
        targets = []
        metro_info = []
        
        for idx, row in df.iterrows():
            metro_name = row['RegionName']
            state_name = row['StateName']
            size_rank = row['SizeRank']
            
            # Get time series values (skip NaN values)
            values = [row[col] for col in date_cols if pd.notna(row[col])]
            
            if len(values) < min_data_points:
                continue
            
            # Create sliding window features
            window_size = 12  # Use 12 months of history
            for i in range(window_size, len(values)):
                # Features: past 12 months + metro characteristics
                feature_vector = values[i-window_size:i] + [size_rank, len(values)]
                target_value = values[i]
                
                features.append(feature_vector)
                targets.append(target_value)
                metro_info.append({
                    'metro': metro_name,
                    'state': state_name,
                    'month_index': i,
                    'date': date_cols[i] if i < len(date_cols) else 'unknown'
                })
        
        if not features:
            logger.warning("No valid time series data found")
            return None, None, []
        
        X = np.array(features)
        y = np.array(targets)
        
        logger.info(f"Prepared {len(X)} time series samples from {len(df)} metros")
        logger.info(f"Feature dimensions: {X.shape[1]} (12 months history + metro features)")
        
        return X, y, metro_info
    
    def preprocess_data(self, X: pd.DataFrame, y: Optional[np.ndarray] = None,
                       handle_categorical: bool = True,
                       scale_features: bool = True,
                       dataset_name: str = 'default') -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        Preprocess data for machine learning.
        
        Args:
            X: Feature DataFrame
            y: Target array (optional)
            handle_categorical: Whether to encode categorical variables
            scale_features: Whether to scale numerical features
            dataset_name: Name for caching encoders/scalers
            
        Returns:
            Tuple of (X_processed, y, preprocessing_info)
        """
        logger.info(f"Preprocessing data: {X.shape[0]} samples, {X.shape[1]} features")
        
        X_processed = X.copy()
        preprocessing_info = {}
        
        # Handle categorical variables
        if handle_categorical:
            categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
            
            for col in categorical_cols:
                encoder_key = f"{dataset_name}_{col}"
                if encoder_key not in self.label_encoders:
                    self.label_encoders[encoder_key] = LabelEncoder()
                    X_processed[col] = self.label_encoders[encoder_key].fit_transform(X_processed[col].astype(str))
                else:
                    X_processed[col] = self.label_encoders[encoder_key].transform(X_processed[col].astype(str))
                
                preprocessing_info[f'categorical_{col}'] = len(self.label_encoders[encoder_key].classes_)
        
        # Handle missing values
        missing_counts = X_processed.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Found missing values: {missing_counts[missing_counts > 0].to_dict()}")
            X_processed = X_processed.fillna(X_processed.median())
        
        # Scale features
        if scale_features:
            scaler_key = f"{dataset_name}_scaler"
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()
                X_processed = self.scalers[scaler_key].fit_transform(X_processed)
            else:
                X_processed = self.scalers[scaler_key].transform(X_processed)
            
            preprocessing_info['scaled'] = True
        
        logger.info(f"Preprocessing completed: {X_processed.shape}")
        return X_processed, y, preprocessing_info
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.
        
        Args:
            X: Feature array
            y: Target array
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def cache_data(self, data: Any, filename: str) -> None:
        """
        Cache processed data to disk.
        
        Args:
            data: Data to cache
            filename: Filename for cached data
        """
        cache_path = self.cache_dir / f"{filename}.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Data cached to {cache_path}")
    
    def load_cached_data(self, filename: str) -> Optional[Any]:
        """
        Load cached data from disk.
        
        Args:
            filename: Filename of cached data
            
        Returns:
            Cached data or None if not found
        """
        cache_path = self.cache_dir / f"{filename}.pkl"
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Data loaded from cache: {cache_path}")
            return data
        return None
