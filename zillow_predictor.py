"""
Zillow Housing Price Predictor

This module loads and analyzes real Zillow housing data to create predictive models
for housing prices and rental rates across different metro areas and time periods.

Author: AI Projects Series
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

class ZillowPredictor:
    """Advanced housing price predictor using real Zillow data."""
    
    def __init__(self, data_dir="/Users/km/Documents/AI/AI_PROJECTS/1000 AI Projects/0001 Housing price predictor/Zillow Data"):
        self.data_dir = data_dir
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.datasets = {}
        
    def load_zillow_datasets(self):
        """Load all available Zillow datasets."""
        print("🏠 Loading Zillow Datasets...")
        print("=" * 50)
        
        dataset_info = {
            'zhvi': 'Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',  # Home Values
            'zori': 'Metro_zori_uc_sfrcondomfr_sm_month.csv',  # Rental Index
            'inventory': 'Metro_invt_fs_uc_sfrcondo_sm_month.csv',  # Inventory
            'sales_count': 'Metro_sales_count_now_uc_sfrcondo_month.csv',  # Sales Count
            'market_temp': 'Metro_market_temp_index_uc_sfrcondo_month.csv',  # Market Temperature
            'income_needed': 'Metro_new_homeowner_income_needed_downpayment_0.20_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv'
        }
        
        for key, filename in dataset_info.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    self.datasets[key] = df
                    print(f"✅ {key.upper()}: {df.shape[0]} metros, {df.shape[1]-5} time periods")
                except Exception as e:
                    print(f"❌ Error loading {key}: {e}")
            else:
                print(f"⚠️  {key} dataset not found: {filename}")
        
        return self.datasets
    
    def prepare_time_series_data(self, dataset_key='zhvi', target_metros=None, min_data_points=24):
        """Prepare time series data for machine learning."""
        print(f"\n🔧 Preparing {dataset_key.upper()} Time Series Data...")
        
        if dataset_key not in self.datasets:
            print(f"❌ Dataset {dataset_key} not loaded!")
            return None, None
        
        df = self.datasets[dataset_key].copy()
        
        # Get date columns (skip metadata columns)
        date_cols = [col for col in df.columns if col not in ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']]
        
        # Filter metros if specified
        if target_metros:
            df = df[df['RegionName'].isin(target_metros)]
        
        # Prepare features and targets
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
        
        X = np.array(features)
        y = np.array(targets)
        
        print(f"✅ Prepared {len(X)} training samples from {len(df)} metros")
        print(f"📊 Feature dimensions: {X.shape[1]} (12 months history + metro features)")
        
        return X, y, metro_info
    
    def create_cross_metro_features(self):
        """Create features combining multiple datasets for comprehensive analysis."""
        print("\n🔄 Creating Cross-Metro Features...")
        
        if 'zhvi' not in self.datasets or 'zori' not in self.datasets:
            print("❌ Need both ZHVI and ZORI datasets for cross-metro analysis")
            return None, None
        
        zhvi_df = self.datasets['zhvi']
        zori_df = self.datasets['zori']
        
        # Merge datasets on metro information
        merged = zhvi_df.merge(zori_df, on=['RegionID', 'RegionName', 'StateName'], 
                              suffixes=('_home_value', '_rent'), how='inner')
        
        print(f"✅ Merged data for {len(merged)} metros")
        
        # Get common date columns
        home_value_cols = [col for col in merged.columns if col.endswith('_home_value') and '-' in col]
        rent_cols = [col for col in merged.columns if col.endswith('_rent') and '-' in col]
        
        # Find overlapping dates
        home_dates = set([col.replace('_home_value', '') for col in home_value_cols])
        rent_dates = set([col.replace('_rent', '') for col in rent_cols])
        common_dates = sorted(list(home_dates.intersection(rent_dates)))
        
        if len(common_dates) < 12:
            print(f"❌ Not enough overlapping dates: {len(common_dates)}")
            return None, None
        
        features = []
        targets = []
        metro_info = []
        
        for idx, row in merged.iterrows():
            metro_name = row['RegionName']
            state_name = row['StateName']
            
            # Get time series for both home values and rent
            home_values = []
            rent_values = []
            
            for date in common_dates:
                home_val = row.get(f"{date}_home_value")
                rent_val = row.get(f"{date}_rent")
                
                if pd.notna(home_val) and pd.notna(rent_val):
                    home_values.append(home_val)
                    rent_values.append(rent_val)
            
            if len(home_values) < 12:
                continue
            
            # Create features using sliding window
            window_size = 6
            for i in range(window_size, len(home_values)):
                # Features: past home values, rent values, ratios, trends
                past_home = home_values[i-window_size:i]
                past_rent = rent_values[i-window_size:i]
                
                # Calculate additional features
                home_trend = (past_home[-1] - past_home[0]) / past_home[0] if past_home[0] > 0 else 0
                rent_trend = (past_rent[-1] - past_rent[0]) / past_rent[0] if past_rent[0] > 0 else 0
                price_to_rent_ratio = past_home[-1] / past_rent[-1] if past_rent[-1] > 0 else 0
                
                feature_vector = (past_home + past_rent + 
                                [home_trend, rent_trend, price_to_rent_ratio, 
                                 row['SizeRank_home_value']])
                
                target_value = home_values[i]
                
                features.append(feature_vector)
                targets.append(target_value)
                metro_info.append({
                    'metro': metro_name,
                    'state': state_name,
                    'date': common_dates[i] if i < len(common_dates) else 'unknown'
                })
        
        X = np.array(features)
        y = np.array(targets)
        
        print(f"✅ Created {len(X)} cross-metro samples")
        print(f"📊 Feature dimensions: {X.shape[1]} (home values + rent + ratios + trends)")
        
        return X, y, metro_info
    
    def train_models(self, X, y):
        """Train multiple models on the prepared data."""
        print("\n🤖 Training Models...")
        print("=" * 30)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store test data for evaluation
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        # Train models
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        for name, model in models_to_train.items():
            print(f"Training {name}...")
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train, y_train)
            
            self.models[name] = model
            print(f"✅ {name} trained")
        
        print(f"\n📊 Training completed with {len(X_train)} samples")
        return X_train, X_test, y_train, y_test
    
    def evaluate_models(self):
        """Evaluate all trained models."""
        print("\n📈 Model Performance Evaluation")
        print("=" * 50)
        
        results = {}
        
        for name, model in self.models.items():
            if name == 'Linear Regression':
                y_pred = model.predict(self.X_test)
            else:
                # Use original unscaled data for tree-based models
                X_test_orig = self.scaler.inverse_transform(self.X_test)
                y_pred = model.predict(X_test_orig)
            
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'predictions': y_pred
            }
            
            print(f"{name}:")
            print(f"  RMSE: ${rmse:,.0f}")
            print(f"  MAE:  ${mae:,.0f}")
            print(f"  R²:   {r2:.4f}")
            print()
        
        return results
    
    def create_zillow_visualizations(self, results):
        """Create comprehensive visualizations for Zillow data analysis."""
        print("\n📊 Creating Zillow Data Visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Zillow Housing Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Model Performance Comparison
        model_names = list(results.keys())
        r2_scores = [results[name]['R²'] for name in model_names]
        rmse_scores = [results[name]['RMSE'] for name in model_names]
        
        axes[0, 0].bar(model_names, r2_scores, color=['skyblue', 'lightgreen', 'coral'])
        axes[0, 0].set_title('Model R² Scores')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. RMSE Comparison
        axes[0, 1].bar(model_names, rmse_scores, color=['skyblue', 'lightgreen', 'coral'])
        axes[0, 1].set_title('Model RMSE (Lower is Better)')
        axes[0, 1].set_ylabel('RMSE ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Actual vs Predicted (Best Model)
        best_model = max(results.keys(), key=lambda x: results[x]['R²'])
        y_pred_best = results[best_model]['predictions']
        
        axes[0, 2].scatter(self.y_test, y_pred_best, alpha=0.5, color='green')
        axes[0, 2].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 2].set_title(f'Actual vs Predicted ({best_model})')
        axes[0, 2].set_xlabel('Actual Values')
        axes[0, 2].set_ylabel('Predicted Values')
        
        # 4. Price Distribution
        axes[1, 0].hist(self.y_test, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Housing Price Distribution')
        axes[1, 0].set_xlabel('Price ($)')
        axes[1, 0].set_ylabel('Frequency')
        
        # 5. Residuals Plot
        residuals = self.y_test - y_pred_best
        axes[1, 1].scatter(y_pred_best, residuals, alpha=0.5)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_title('Residuals Plot')
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('Residuals')
        
        # 6. Metro Analysis (if we have metro info)
        if hasattr(self, 'metro_info') and self.metro_info:
            # Show top metros by average price
            metro_prices = {}
            for i, info in enumerate(self.metro_info[-len(self.y_test):]):
                metro = info['metro']
                if metro not in metro_prices:
                    metro_prices[metro] = []
                metro_prices[metro].append(self.y_test[i])
            
            avg_prices = {metro: np.mean(prices) for metro, prices in metro_prices.items()}
            top_metros = sorted(avg_prices.items(), key=lambda x: x[1], reverse=True)[:10]
            
            metros, prices = zip(*top_metros)
            axes[1, 2].barh(range(len(metros)), prices, color='orange')
            axes[1, 2].set_yticks(range(len(metros)))
            axes[1, 2].set_yticklabels(metros)
            axes[1, 2].set_title('Top 10 Most Expensive Metros')
            axes[1, 2].set_xlabel('Average Price ($)')
        
        plt.tight_layout()
        plt.show()
    
    def predict_metro_prices(self, metro_name, months_ahead=6):
        """Predict future prices for a specific metro area."""
        print(f"\n🔮 Predicting {metro_name} Prices ({months_ahead} months ahead)")
        print("=" * 60)
        
        if 'zhvi' not in self.datasets:
            print("❌ ZHVI dataset required for predictions")
            return
        
        df = self.datasets['zhvi']
        metro_data = df[df['RegionName'] == metro_name]
        
        if metro_data.empty:
            print(f"❌ Metro '{metro_name}' not found in dataset")
            available_metros = df['RegionName'].head(10).tolist()
            print(f"Available metros (sample): {available_metros}")
            return
        
        # Get recent data
        date_cols = [col for col in df.columns if col not in ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']]
        recent_values = []
        
        for col in date_cols[-12:]:  # Last 12 months
            val = metro_data.iloc[0][col]
            if pd.notna(val):
                recent_values.append(val)
        
        if len(recent_values) < 12:
            print(f"❌ Insufficient recent data for {metro_name}")
            return
        
        # Make predictions using the best model
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x].score(self.X_test, self.y_test) if hasattr(self.models[x], 'score') else 0)
        best_model = self.models[best_model_name]
        
        predictions = []
        current_window = recent_values[-12:]
        
        for _ in range(months_ahead):
            # Prepare feature vector
            feature_vector = current_window + [metro_data.iloc[0]['SizeRank'], len(recent_values)]
            
            if best_model_name == 'Linear Regression':
                feature_vector = self.scaler.transform([feature_vector])
                pred = best_model.predict(feature_vector)[0]
            else:
                pred = best_model.predict([feature_vector])[0]
            
            predictions.append(pred)
            current_window = current_window[1:] + [pred]  # Slide window
        
        # Display results
        current_price = recent_values[-1]
        print(f"Current Price: ${current_price:,.0f}")
        print(f"Predictions using {best_model_name}:")
        
        for i, pred in enumerate(predictions, 1):
            change = ((pred - current_price) / current_price) * 100
            print(f"  Month {i}: ${pred:,.0f} ({change:+.1f}%)")
        
        return predictions

def main():
    """Main function to run Zillow housing analysis."""
    print("🏠 Zillow Housing Price Predictor")
    print("=" * 50)
    
    # Initialize predictor
    predictor = ZillowPredictor()
    
    # Load datasets
    datasets = predictor.load_zillow_datasets()
    
    if not datasets:
        print("❌ No datasets loaded. Please check the data directory.")
        return
    
    # Prepare time series data
    X, y, metro_info = predictor.prepare_time_series_data('zhvi', min_data_points=36)
    
    if X is None:
        print("❌ Failed to prepare data")
        return
    
    predictor.metro_info = metro_info
    
    # Train models
    X_train, X_test, y_train, y_test = predictor.train_models(X, y)
    
    # Evaluate models
    results = predictor.evaluate_models()
    
    # Create visualizations
    predictor.create_zillow_visualizations(results)
    
    # Example predictions for major metros
    major_metros = ["New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX"]
    
    print("\n🎯 Sample Metro Predictions")
    print("=" * 50)
    
    for metro in major_metros:
        try:
            predictor.predict_metro_prices(metro, months_ahead=3)
            print()
        except Exception as e:
            print(f"❌ Error predicting {metro}: {e}")

if __name__ == "__main__":
    main()
