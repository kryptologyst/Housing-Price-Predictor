"""
Project 1: Linear Regression Model for Predicting House Prices

Description:
A Linear Regression model predicts continuous values—like house
prices—based on input features such as size, number of bedrooms, 
location, etc. This basic model uses a synthetic dataset to 
demonstrate how to predict house prices based on the house's area 
in square feet.

Author: AI Projects Series
Date: 2025
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    """Main function to run the housing price prediction model."""
    
    # Sample data: Area in square feet (X) and corresponding house price in $1000s (y)
    # For simplicity, let's simulate a linear relationship: price ≈ 0.5 * area + noise
    X = np.array([[650], [800], [950], [1100], [1250], [1400], [1550], [1700], [1850], [2000]])
    y = np.array([325, 400, 475, 550, 625, 700, 775, 850, 925, 1000])  # Prices in $1000s
    
    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions on the training data to visualize the line of best fit
    y_pred = model.predict(X)
    
    # Print model parameters and performance metrics
    print("=" * 50)
    print("HOUSING PRICE PREDICTION MODEL RESULTS")
    print("=" * 50)
    print(f"Model Coefficient (Slope): {model.coef_[0]:.4f}")
    print(f"Model Intercept: {model.intercept_:.4f}")
    print(f"Mean Squared Error: {mean_squared_error(y, y_pred):.4f}")
    print(f"R² Score: {r2_score(y, y_pred):.4f}")
    
    # Predict price for a new house (e.g., 1600 sq. ft)
    new_area = np.array([[1600]])
    predicted_price = model.predict(new_area)[0]
    print(f"\nPredicted price for 1600 sq.ft house: ${predicted_price * 1000:,.2f}")
    
    # Create visualization
    create_plot(X, y, y_pred, new_area, predicted_price)
    
    # Additional predictions for different house sizes
    print("\n" + "=" * 50)
    print("ADDITIONAL PREDICTIONS")
    print("=" * 50)
    test_areas = [1200, 1500, 1800, 2200]
    for area in test_areas:
        price = model.predict([[area]])[0]
        print(f"House size: {area:4d} sq.ft → Predicted price: ${price * 1000:8,.2f}")

def create_plot(X, y, y_pred, new_area, predicted_price):
    """Create and display the visualization plot."""
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual Data', s=60, alpha=0.7)
    plt.plot(X, y_pred, color='red', label='Regression Line', linewidth=2)
    plt.scatter(new_area, predicted_price, color='green', label='Prediction (1600 sq.ft)', 
                s=100, marker='*', edgecolor='black')
    
    plt.xlabel("Area (sq.ft)", fontsize=12)
    plt.ylabel("Price ($1000s)", fontsize=12)
    plt.title("House Price Prediction using Linear Regression", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

# Note: This basic linear regression model learns a relationship between 
# house size and price. For real-world applications, consider:
# - Using datasets like Boston Housing Dataset or Kaggle housing data
# - Incorporating multiple features (location, bedrooms, age, condition)
# - Feature engineering and data preprocessing
# - Cross-validation and train/test splits
# - More advanced algorithms (Random Forest, XGBoost, etc.)