<<<<<<< HEAD
# Usage Examples

This document provides detailed examples of how to use the Housing Price Predictor in different scenarios.

## Quick Start Examples
=======
# 📚 Usage Examples

This document provides detailed examples of how to use the Housing Price Predictor in different scenarios.

## 🚀 Quick Start Examples
>>>>>>> 3ecc615 (Add .gitignore (+ .gitattributes/.gitkeep))

### 1. Basic Linear Regression
```bash
python 0001.py
```

**Expected Output:**
```
==================================================
HOUSING PRICE PREDICTION MODEL RESULTS
==================================================
Model Coefficient (Slope): 0.5000
Model Intercept: 0.0000
Mean Squared Error: 0.0000
R² Score: 1.0000

Predicted price for 1600 sq.ft house: $800,000.00
```

### 2. Advanced Multi-Feature Analysis
```bash
python housing_predictor_advanced.py
```

**Features:**
- 8 different housing features (income, age, rooms, etc.)
- Model comparison (Linear Regression vs Random Forest)
- Feature importance analysis
- Interactive visualizations

### 3. Real Zillow Data Analysis
```bash
python zillow_predictor.py
```

**Capabilities:**
- 895 metropolitan areas
- 25+ years of historical data
- Time series forecasting
- Multi-model comparison

## 🔧 Custom Usage Examples

### Loading Your Own Dataset

```python
from kaggle_dataset_loader import KaggleHousingLoader

# Initialize loader
loader = KaggleHousingLoader()

# Load Ames Housing Dataset
X, y = loader.load_ames_housing("data/ames_housing.csv")

# Create synthetic dataset
X_synthetic, y_synthetic = loader.create_synthetic_dataset(n_samples=2000)

# Analyze the dataset
loader.analyze_dataset(X, y, "My Custom Dataset")
```

### Predicting Specific Metro Areas

```python
from zillow_predictor import ZillowPredictor

# Initialize predictor
predictor = ZillowPredictor()

# Load datasets
predictor.load_zillow_datasets()

# Prepare data
X, y, metro_info = predictor.prepare_time_series_data('zhvi')

# Train models
predictor.train_models(X, y)

# Predict specific metros
predictor.predict_metro_prices("San Francisco, CA", months_ahead=6)
predictor.predict_metro_prices("Austin, TX", months_ahead=12)
```

### Custom Model Training

```python
from housing_predictor_advanced import HousingPredictor
from sklearn.ensemble import XGBRegressor

# Initialize predictor
predictor = HousingPredictor()

# Load California housing data
X, y = predictor.load_california_housing()

# Add custom model
from sklearn.ensemble import GradientBoostingRegressor
custom_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1)
predictor.models['Custom XGBoost'] = custom_model

# Train all models
predictor.train_models()

# Evaluate performance
results = predictor.evaluate_models()
```

<<<<<<< HEAD
## Visualization Examples
=======
## 📊 Visualization Examples
>>>>>>> 3ecc615 (Add .gitignore (+ .gitattributes/.gitkeep))

### Creating Custom Plots

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
predictor = ZillowPredictor()
datasets = predictor.load_zillow_datasets()

# Create custom visualization
plt.figure(figsize=(12, 8))

# Plot price trends for top 5 metros
top_metros = ["New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ"]

for metro in top_metros:
    # Extract time series data for each metro
    metro_data = datasets['zhvi'][datasets['zhvi']['RegionName'] == metro]
    # Plot the data...

plt.title('Housing Price Trends - Top 5 Metro Areas')
plt.show()
```

### Feature Importance Analysis

```python
# After training Random Forest model
rf_model = predictor.models['Random Forest']
feature_importance = rf_model.feature_importances_

# Create importance plot
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_importance)), feature_importance)
plt.yticks(range(len(feature_importance)), feature_names)
plt.title('Feature Importance Analysis')
plt.show()
```

<<<<<<< HEAD
## Specific Use Cases
=======
## 🎯 Specific Use Cases
>>>>>>> 3ecc615 (Add .gitignore (+ .gitattributes/.gitkeep))

### 1. Real Estate Investment Analysis
```python
# Analyze multiple metros for investment potential
investment_metros = ["Austin, TX", "Nashville, TN", "Denver, CO", "Seattle, WA"]

for metro in investment_metros:
    predictions = predictor.predict_metro_prices(metro, months_ahead=12)
    # Calculate ROI potential
    current_price = predictions[0]
    future_price = predictions[-1]
    roi = ((future_price - current_price) / current_price) * 100
    print(f"{metro}: {roi:.1f}% projected growth")
```

### 2. Market Timing Analysis
```python
# Identify best time to buy/sell
def analyze_market_timing(metro_name):
    predictor = ZillowPredictor()
    
    # Get historical data
    historical_data = predictor.get_historical_prices(metro_name)
    
    # Calculate moving averages
    short_ma = historical_data.rolling(window=6).mean()
    long_ma = historical_data.rolling(window=12).mean()
    
    # Generate buy/sell signals
    signals = []
    for i in range(len(short_ma)):
        if short_ma[i] > long_ma[i]:
            signals.append("BUY")
        else:
            signals.append("SELL")
    
    return signals[-1]  # Current signal
```

### 3. Comparative Market Analysis
```python
# Compare similar metros
def compare_metros(metro_list, feature='price_growth'):
    results = {}
    
    for metro in metro_list:
        predictions = predictor.predict_metro_prices(metro, months_ahead=6)
        current = predictions[0]
        future = predictions[-1]
        growth = ((future - current) / current) * 100
        results[metro] = growth
    
    # Sort by growth potential
    sorted_metros = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print("Metro Rankings by Growth Potential:")
    for i, (metro, growth) in enumerate(sorted_metros, 1):
        print(f"{i}. {metro}: {growth:.1f}%")
    
    return sorted_metros
```

<<<<<<< HEAD
## Troubleshooting Examples
=======
## 🔍 Troubleshooting Examples
>>>>>>> 3ecc615 (Add .gitignore (+ .gitattributes/.gitkeep))

### Common Issues and Solutions

#### 1. Missing Data Files
```python
import os

# Check if Zillow data exists
data_dir = "./Zillow Data"
if not os.path.exists(data_dir):
    print("❌ Zillow Data directory not found")
    print("📥 Download from: https://www.zillow.com/research/data/")
else:
    files = os.listdir(data_dir)
    print(f"✅ Found {len(files)} data files")
```

#### 2. Memory Issues with Large Datasets
```python
# Process data in chunks for large datasets
def process_large_dataset(file_path, chunk_size=10000):
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Process each chunk
        processed_chunk = preprocess_chunk(chunk)
        chunks.append(processed_chunk)
    
    return pd.concat(chunks, ignore_index=True)
```

#### 3. Model Performance Issues
```python
# Improve model performance
from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestRegressor(),
    param_grid,
    cv=5,
    scoring='r2'
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

## 📈 Advanced Analytics Examples

### Time Series Decomposition
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose time series into trend, seasonal, and residual components
def analyze_seasonality(metro_name):
    # Get time series data
    ts_data = get_metro_timeseries(metro_name)
    
    # Perform decomposition
    decomposition = seasonal_decompose(ts_data, model='additive', period=12)
    
    # Plot components
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    decomposition.observed.plot(ax=axes[0], title='Original')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    plt.tight_layout()
    plt.show()
```

### Correlation Analysis
```python
# Analyze correlations between different markets
def market_correlation_analysis():
    major_metros = ["New York, NY", "Los Angeles, CA", "Chicago, IL"]
    
    # Get price data for all metros
    price_data = {}
    for metro in major_metros:
        price_data[metro] = get_metro_prices(metro)
    
    # Create correlation matrix
    df = pd.DataFrame(price_data)
    correlation_matrix = df.corr()
    
    # Visualize
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Metro Price Correlation Matrix')
    plt.show()
    
    return correlation_matrix
```

These examples should help you get started with the Housing Price Predictor and customize it for your specific needs!
