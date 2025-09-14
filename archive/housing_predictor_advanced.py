"""
Advanced Housing Price Predictor with Real Datasets

This enhanced version uses real housing data and multiple features to create
more accurate and interesting predictions. Supports multiple datasets and
advanced machine learning techniques.

Author: AI Projects Series
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class HousingPredictor:
    """Advanced housing price predictor with multiple algorithms and datasets."""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_california_housing(self):
        """Load and prepare the California housing dataset."""
        print("🏠 Loading California Housing Dataset...")
        
        # Load the dataset
        housing = fetch_california_housing()
        X = pd.DataFrame(housing.data, columns=housing.feature_names)
        y = housing.target
        
        self.feature_names = housing.feature_names
        
        # Display dataset info
        print(f"📊 Dataset shape: {X.shape}")
        print(f"🎯 Target: Median house value (in hundreds of thousands)")
        print(f"📋 Features: {', '.join(self.feature_names)}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return X, y
    
    def train_models(self):
        """Train multiple models on the housing data."""
        print("\n🤖 Training Models...")
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(self.X_train_scaled, self.y_train)
        self.models['Linear Regression'] = lr_model
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model
        
        print("✅ Models trained successfully!")
    
    def evaluate_models(self):
        """Evaluate all trained models."""
        print("\n📈 Model Performance Comparison:")
        print("=" * 60)
        
        results = {}
        
        for name, model in self.models.items():
            if name == 'Linear Regression':
                y_pred = model.predict(self.X_test_scaled)
            else:
                y_pred = model.predict(self.X_test)
            
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2
            }
            
            print(f"{name}:")
            print(f"  RMSE: ${rmse*100000:,.0f}")
            print(f"  MAE:  ${mae*100000:,.0f}")
            print(f"  R²:   {r2:.4f}")
            print()
        
        return results
    
    def feature_importance_analysis(self):
        """Analyze feature importance using Random Forest."""
        if 'Random Forest' not in self.models:
            return
        
        rf_model = self.models['Random Forest']
        importance = rf_model.feature_importances_
        
        # Create feature importance plot
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importance)[::-1]
        
        plt.title('🔍 Feature Importance Analysis', fontsize=16, fontweight='bold')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [self.feature_names[i] for i in indices], rotation=45)
        plt.ylabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
        print("\n🎯 Top 3 Most Important Features:")
        for i in range(3):
            idx = indices[i]
            print(f"{i+1}. {self.feature_names[idx]}: {importance[idx]:.4f}")
    
    def make_predictions(self, house_features=None):
        """Make predictions for custom house features."""
        if house_features is None:
            # Default example house
            house_features = {
                'MedInc': 5.0,        # Median income
                'HouseAge': 10.0,     # House age
                'AveRooms': 6.0,      # Average rooms
                'AveBedrms': 1.2,     # Average bedrooms
                'Population': 3000.0,  # Population
                'AveOccup': 3.0,      # Average occupancy
                'Latitude': 34.0,     # Latitude
                'Longitude': -118.0   # Longitude (LA area)
            }
        
        print("\n🏡 Making Predictions for Sample House:")
        print("=" * 40)
        for feature, value in house_features.items():
            print(f"{feature}: {value}")
        
        # Prepare input
        input_data = np.array([list(house_features.values())]).reshape(1, -1)
        input_scaled = self.scaler.transform(input_data)
        
        print("\n💰 Predicted Prices:")
        for name, model in self.models.items():
            if name == 'Linear Regression':
                prediction = model.predict(input_scaled)[0]
            else:
                prediction = model.predict(input_data)[0]
            
            price = prediction * 100000  # Convert to dollars
            print(f"{name}: ${price:,.0f}")
    
    def create_visualizations(self, X, y):
        """Create comprehensive data visualizations."""
        print("\n📊 Creating Data Visualizations...")
        
        # 1. Correlation heatmap
        plt.figure(figsize=(12, 8))
        df = pd.DataFrame(X, columns=self.feature_names)
        df['Price'] = y
        
        plt.subplot(2, 2, 1)
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        
        # 2. Price distribution
        plt.subplot(2, 2, 2)
        plt.hist(y, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('House Price Distribution')
        plt.xlabel('Price (hundreds of thousands)')
        plt.ylabel('Frequency')
        
        # 3. Income vs Price scatter
        plt.subplot(2, 2, 3)
        plt.scatter(X['MedInc'], y, alpha=0.5, color='green')
        plt.title('Median Income vs House Price')
        plt.xlabel('Median Income')
        plt.ylabel('Price')
        
        # 4. House Age vs Price
        plt.subplot(2, 2, 4)
        plt.scatter(X['HouseAge'], y, alpha=0.5, color='orange')
        plt.title('House Age vs Price')
        plt.xlabel('House Age')
        plt.ylabel('Price')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the advanced housing predictor."""
    print("🚀 Advanced Housing Price Predictor")
    print("=" * 50)
    
    # Initialize predictor
    predictor = HousingPredictor()
    
    # Load and explore data
    X, y = predictor.load_california_housing()
    
    # Create visualizations
    predictor.create_visualizations(X, y)
    
    # Train models
    predictor.train_models()
    
    # Evaluate models
    results = predictor.evaluate_models()
    
    # Feature importance analysis
    predictor.feature_importance_analysis()
    
    # Make sample predictions
    predictor.make_predictions()
    
    # Interactive prediction example
    print("\n" + "=" * 50)
    print("🎮 Try Your Own Predictions!")
    print("=" * 50)
    print("Modify the house_features dictionary in the code to test different scenarios:")
    print("- High income area: MedInc=8.0")
    print("- Beachfront property: Latitude=33.0, Longitude=-117.0")
    print("- Older house: HouseAge=50.0")
    print("- Large house: AveRooms=8.0")

if __name__ == "__main__":
    main()
