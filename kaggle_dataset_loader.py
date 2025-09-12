"""
Kaggle Dataset Loader for Housing Price Prediction

This module provides utilities to load and work with popular Kaggle housing datasets
like the Ames Housing Dataset and Boston Housing prices.

Author: AI Projects Series
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class KaggleHousingLoader:
    """Load and preprocess Kaggle housing datasets."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_ames_housing(self, file_path):
        """
        Load Ames Housing Dataset from Kaggle.
        
        Download from: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
        """
        try:
            df = pd.read_csv(file_path)
            print(f"🏠 Loaded Ames Housing Dataset: {df.shape}")
            return self.preprocess_ames_data(df)
        except FileNotFoundError:
            print("❌ Ames dataset not found. Download from Kaggle:")
            print("https://www.kaggle.com/c/house-prices-advanced-regression-techniques")
            return None, None
    
    def preprocess_ames_data(self, df):
        """Preprocess the Ames housing dataset."""
        print("🔧 Preprocessing Ames Housing Data...")
        
        # Select key features for prediction
        key_features = [
            'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
            'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath',
            'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
            'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF'
        ]
        
        # Add categorical features
        categorical_features = ['Neighborhood', 'BldgType', 'HouseStyle']
        
        # Filter available features
        available_features = [f for f in key_features if f in df.columns]
        available_categorical = [f for f in categorical_features if f in df.columns]
        
        # Create working dataframe
        X = df[available_features + available_categorical].copy()
        y = df['SalePrice'] if 'SalePrice' in df.columns else None
        
        # Handle missing values
        X = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        
        # Encode categorical variables
        for col in available_categorical:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        print(f"✅ Preprocessed features: {X.shape[1]} features, {X.shape[0]} samples")
        return X, y
    
    def create_synthetic_dataset(self, n_samples=1000):
        """Create a more realistic synthetic dataset with multiple features."""
        print(f"🎲 Creating synthetic dataset with {n_samples} samples...")
        
        np.random.seed(42)
        
        # Generate features
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
            'distance_to_city': np.random.uniform(1, 50, n_samples)
        }
        
        # Ensure positive values
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
            np.random.normal(0, 20000, n_samples)  # Add noise
        )
        
        # Ensure reasonable price range
        price = np.clip(price, 100000, 2000000)
        
        X = pd.DataFrame(data)
        y = price
        
        print("✅ Synthetic dataset created with realistic features!")
        return X, y
    
    def analyze_dataset(self, X, y, dataset_name="Dataset"):
        """Perform comprehensive dataset analysis."""
        print(f"\n📊 {dataset_name} Analysis")
        print("=" * 50)
        
        # Basic statistics
        print(f"Samples: {X.shape[0]:,}")
        print(f"Features: {X.shape[1]}")
        print(f"Price range: ${y.min():,.0f} - ${y.max():,.0f}")
        print(f"Average price: ${y.mean():,.0f}")
        
        # Feature correlation with price
        correlations = []
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                corr = np.corrcoef(X[col], y)[0, 1]
                correlations.append((col, abs(corr)))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🎯 Top 5 Features Correlated with Price:")
        for i, (feature, corr) in enumerate(correlations[:5]):
            print(f"{i+1}. {feature}: {corr:.3f}")
    
    def create_advanced_visualizations(self, X, y, dataset_name="Dataset"):
        """Create advanced visualizations for the dataset."""
        print(f"\n📈 Creating visualizations for {dataset_name}...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{dataset_name} - Comprehensive Analysis', fontsize=16, fontweight='bold')
        
        # 1. Price distribution
        axes[0, 0].hist(y, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Price Distribution')
        axes[0, 0].set_xlabel('Price ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Feature correlation heatmap (top features)
        numeric_cols = X.select_dtypes(include=[np.number]).columns[:8]
        if len(numeric_cols) > 0:
            corr_data = X[numeric_cols].copy()
            corr_data['Price'] = y
            corr_matrix = corr_data.corr()
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[0, 1], fmt='.2f', cbar_kws={'shrink': 0.8})
            axes[0, 1].set_title('Feature Correlation Matrix')
        
        # 3. Price vs most correlated feature
        if len(numeric_cols) > 0:
            best_feature = numeric_cols[0]
            axes[0, 2].scatter(X[best_feature], y, alpha=0.5, color='green')
            axes[0, 2].set_title(f'Price vs {best_feature}')
            axes[0, 2].set_xlabel(best_feature)
            axes[0, 2].set_ylabel('Price ($)')
        
        # 4. Feature importance (if enough numeric features)
        if len(numeric_cols) >= 3:
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X[numeric_cols], y)
            
            importance = rf.feature_importances_
            indices = np.argsort(importance)[::-1]
            
            axes[1, 0].bar(range(len(importance)), importance[indices])
            axes[1, 0].set_title('Feature Importance')
            axes[1, 0].set_xticks(range(len(importance)))
            axes[1, 0].set_xticklabels([numeric_cols[i] for i in indices], rotation=45)
        
        # 5. Price by categories (if categorical features exist)
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            cat_col = categorical_cols[0]
            unique_vals = X[cat_col].unique()[:10]  # Limit to 10 categories
            subset_data = X[X[cat_col].isin(unique_vals)]
            subset_prices = y[X[cat_col].isin(unique_vals)]
            
            axes[1, 1].boxplot([subset_prices[subset_data[cat_col] == val] for val in unique_vals])
            axes[1, 1].set_title(f'Price Distribution by {cat_col}')
            axes[1, 1].set_xticklabels(unique_vals, rotation=45)
        
        # 6. Residual plot (simple linear regression)
        from sklearn.linear_model import LinearRegression
        if len(numeric_cols) > 0:
            lr = LinearRegression()
            lr.fit(X[numeric_cols], y)
            y_pred = lr.predict(X[numeric_cols])
            residuals = y - y_pred
            
            axes[1, 2].scatter(y_pred, residuals, alpha=0.5)
            axes[1, 2].axhline(y=0, color='red', linestyle='--')
            axes[1, 2].set_title('Residual Plot')
            axes[1, 2].set_xlabel('Predicted Price')
            axes[1, 2].set_ylabel('Residuals')
        
        plt.tight_layout()
        plt.show()

def demo_kaggle_datasets():
    """Demonstrate loading and working with different datasets."""
    loader = KaggleHousingLoader()
    
    print("🚀 Kaggle Housing Dataset Demo")
    print("=" * 50)
    
    # Try to load Ames dataset (if available)
    print("\n1️⃣ Attempting to load Ames Housing Dataset...")
    X_ames, y_ames = loader.load_ames_housing("train.csv")
    
    # Create synthetic dataset as fallback
    print("\n2️⃣ Creating Enhanced Synthetic Dataset...")
    X_synthetic, y_synthetic = loader.create_synthetic_dataset(1500)
    
    # Analyze datasets
    if X_ames is not None:
        loader.analyze_dataset(X_ames, y_ames, "Ames Housing")
        loader.create_advanced_visualizations(X_ames, y_ames, "Ames Housing")
    
    loader.analyze_dataset(X_synthetic, y_synthetic, "Enhanced Synthetic")
    loader.create_advanced_visualizations(X_synthetic, y_synthetic, "Enhanced Synthetic")
    
    return X_synthetic, y_synthetic

if __name__ == "__main__":
    demo_kaggle_datasets()
