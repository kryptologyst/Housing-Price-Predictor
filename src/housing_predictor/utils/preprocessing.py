"""
Preprocessing utilities for the Housing Price Predictor.

This module provides comprehensive data preprocessing capabilities
including feature engineering, scaling, and data cleaning.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from loguru import logger

from ..config.settings import config


class PreprocessingUtils:
    """Utility class for data preprocessing operations."""
    
    def __init__(self):
        """Initialize preprocessing utilities."""
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.imputers: Dict[str, Any] = {}
        self.feature_selectors: Dict[str, Any] = {}
        self.pca_transformers: Dict[str, Any] = {}
    
    def clean_data(self, df: pd.DataFrame, 
                   remove_duplicates: bool = True,
                   remove_outliers: bool = False,
                   outlier_method: str = 'iqr') -> pd.DataFrame:
        """
        Clean the dataset by removing duplicates and outliers.
        
        Args:
            df: Input DataFrame
            remove_duplicates: Whether to remove duplicate rows
            remove_outliers: Whether to remove outliers
            outlier_method: Method for outlier detection ('iqr', 'zscore')
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove duplicates
        if remove_duplicates:
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            removed_rows = initial_rows - len(df_clean)
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} duplicate rows")
        
        # Remove outliers
        if remove_outliers:
            initial_rows = len(df_clean)
            df_clean = self._remove_outliers(df_clean, method=outlier_method)
            removed_rows = initial_rows - len(df_clean)
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} outlier rows")
        
        return df_clean
    
    def _remove_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers using specified method.
        
        Args:
            df: Input DataFrame
            method: Method for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == 'zscore':
            for col in numeric_cols:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < 3]
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, 
                             strategy: str = 'median',
                             categorical_strategy: str = 'most_frequent') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for numerical columns ('mean', 'median', 'mode', 'knn')
            categorical_strategy: Strategy for categorical columns
            
        Returns:
            DataFrame with missing values handled
        """
        df_processed = df.copy()
        
        # Separate numerical and categorical columns
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        
        # Handle numerical columns
        if len(numeric_cols) > 0:
            if strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
                df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
            else:
                imputer = SimpleImputer(strategy=strategy)
                df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
        
        # Handle categorical columns
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy=categorical_strategy)
            df_processed[categorical_cols] = imputer.fit_transform(df_processed[categorical_cols])
        
        return df_processed
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                  method: str = 'label',
                                  max_categories: int = 10) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input DataFrame
            method: Encoding method ('label', 'onehot')
            max_categories: Maximum number of categories for one-hot encoding
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if method == 'label':
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                self.encoders[f"{col}_label"] = encoder
            
            elif method == 'onehot':
                unique_cats = df_encoded[col].nunique()
                if unique_cats <= max_categories:
                    dummies = pd.get_dummies(df_encoded[col], prefix=col)
                    df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
                else:
                    # Use label encoding for high cardinality
                    encoder = LabelEncoder()
                    df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                    self.encoders[f"{col}_label"] = encoder
        
        return df_encoded
    
    def scale_features(self, X: np.ndarray, method: str = 'standard',
                      scaler_name: str = 'default') -> Tuple[np.ndarray, Any]:
        """
        Scale features using specified method.
        
        Args:
            X: Feature array
            method: Scaling method ('standard', 'minmax', 'robust')
            scaler_name: Name for caching the scaler
            
        Returns:
            Tuple of (scaled_features, scaler)
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        X_scaled = scaler.fit_transform(X)
        self.scalers[scaler_name] = scaler
        
        logger.info(f"Features scaled using {method} scaling")
        return X_scaled, scaler
    
    def inverse_scale_features(self, X_scaled: np.ndarray, scaler_name: str = 'default') -> np.ndarray:
        """
        Inverse transform scaled features.
        
        Args:
            X_scaled: Scaled feature array
            scaler_name: Name of the scaler to use
            
        Returns:
            Original scale features
        """
        if scaler_name not in self.scalers:
            raise ValueError(f"Scaler {scaler_name} not found")
        
        return self.scalers[scaler_name].inverse_transform(X_scaled)
    
    def select_features(self, X: np.ndarray, y: np.ndarray, 
                       k: int = 10, method: str = 'f_regression') -> Tuple[np.ndarray, Any]:
        """
        Select top k features using specified method.
        
        Args:
            X: Feature array
            y: Target array
            k: Number of features to select
            method: Selection method ('f_regression', 'mutual_info')
            
        Returns:
            Tuple of (selected_features, selector)
        """
        if method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=k)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        X_selected = selector.fit_transform(X, y)
        self.feature_selectors[f"{method}_{k}"] = selector
        
        logger.info(f"Selected {k} features using {method}")
        return X_selected, selector
    
    def apply_pca(self, X: np.ndarray, n_components: Optional[int] = None,
                 variance_threshold: float = 0.95) -> Tuple[np.ndarray, Any]:
        """
        Apply Principal Component Analysis.
        
        Args:
            X: Feature array
            n_components: Number of components. If None, determined by variance_threshold
            variance_threshold: Minimum variance to retain
            
        Returns:
            Tuple of (transformed_features, pca_transformer)
        """
        if n_components is None:
            # Determine number of components to retain variance_threshold variance
            pca_temp = PCA()
            pca_temp.fit(X)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_threshold) + 1
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        self.pca_transformers[f"pca_{n_components}"] = pca
        
        explained_variance = pca.explained_variance_ratio_.sum()
        logger.info(f"PCA applied: {n_components} components, {explained_variance:.3f} variance explained")
        
        return X_pca, pca
    
    def create_polynomial_features(self, X: np.ndarray, degree: int = 2) -> np.ndarray:
        """
        Create polynomial features.
        
        Args:
            X: Feature array
            degree: Degree of polynomial features
            
        Returns:
            Array with polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        logger.info(f"Created polynomial features: degree {degree}, {X_poly.shape[1]} features")
        return X_poly
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                   feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features between specified feature pairs.
        
        Args:
            df: Input DataFrame
            feature_pairs: List of (feature1, feature2) tuples
            
        Returns:
            DataFrame with interaction features
        """
        df_interactions = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                df_interactions[interaction_name] = df[feat1] * df[feat2]
        
        logger.info(f"Created {len(feature_pairs)} interaction features")
        return df_interactions
    
    def create_time_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Create time-based features from a date column.
        
        Args:
            df: Input DataFrame
            date_column: Name of the date column
            
        Returns:
            DataFrame with time features
        """
        df_time = df.copy()
        
        if date_column in df.columns:
            df_time[date_column] = pd.to_datetime(df_time[date_column])
            
            # Extract time features
            df_time['year'] = df_time[date_column].dt.year
            df_time['month'] = df_time[date_column].dt.month
            df_time['day'] = df_time[date_column].dt.day
            df_time['dayofweek'] = df_time[date_column].dt.dayofweek
            df_time['quarter'] = df_time[date_column].dt.quarter
            
            # Cyclical encoding for time features
            df_time['month_sin'] = np.sin(2 * np.pi * df_time['month'] / 12)
            df_time['month_cos'] = np.cos(2 * np.pi * df_time['month'] / 12)
            df_time['dayofweek_sin'] = np.sin(2 * np.pi * df_time['dayofweek'] / 7)
            df_time['dayofweek_cos'] = np.cos(2 * np.pi * df_time['dayofweek'] / 7)
        
        logger.info("Created time-based features")
        return df_time
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of applied preprocessing steps.
        
        Returns:
            Dictionary with preprocessing summary
        """
        return {
            'scalers': list(self.scalers.keys()),
            'encoders': list(self.encoders.keys()),
            'imputers': list(self.imputers.keys()),
            'feature_selectors': list(self.feature_selectors.keys()),
            'pca_transformers': list(self.pca_transformers.keys())
        }
    
    def save_preprocessing_objects(self, filepath: str) -> None:
        """
        Save preprocessing objects to disk.
        
        Args:
            filepath: Path to save the objects
        """
        import joblib
        
        preprocessing_objects = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'imputers': self.imputers,
            'feature_selectors': self.feature_selectors,
            'pca_transformers': self.pca_transformers
        }
        
        joblib.dump(preprocessing_objects, filepath)
        logger.info(f"Preprocessing objects saved to {filepath}")
    
    def load_preprocessing_objects(self, filepath: str) -> None:
        """
        Load preprocessing objects from disk.
        
        Args:
            filepath: Path to load the objects from
        """
        import joblib
        
        preprocessing_objects = joblib.load(filepath)
        
        self.scalers = preprocessing_objects.get('scalers', {})
        self.encoders = preprocessing_objects.get('encoders', {})
        self.imputers = preprocessing_objects.get('imputers', {})
        self.feature_selectors = preprocessing_objects.get('feature_selectors', {})
        self.pca_transformers = preprocessing_objects.get('pca_transformers', {})
        
        logger.info(f"Preprocessing objects loaded from {filepath}")
