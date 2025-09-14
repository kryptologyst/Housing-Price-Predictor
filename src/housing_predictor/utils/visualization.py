"""
Visualization utilities for the Housing Price Predictor.

This module provides comprehensive visualization capabilities for
housing data analysis and model evaluation.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Install with: pip install plotly")

from ..config.settings import config


class VisualizationUtils:
    """Utility class for creating visualizations."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualization utilities.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self.setup_style()
    
    def setup_style(self) -> None:
        """Setup matplotlib style."""
        try:
            plt.style.use(self.style)
        except OSError:
            logger.warning(f"Style {self.style} not found, using default")
            plt.style.use('default')
        
        # Set default figure size
        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['figure.dpi'] = config.get('visualization.dpi', 100)
    
    def plot_feature_importance(self, feature_importance: List[Tuple[str, float]], 
                               model_name: str = "Model", top_n: int = 10) -> None:
        """
        Plot feature importance.
        
        Args:
            feature_importance: List of (feature_name, importance) tuples
            model_name: Name of the model
            top_n: Number of top features to show
        """
        if not feature_importance:
            logger.warning("No feature importance data provided")
            return
        
        # Take top N features
        top_features = feature_importance[:top_n]
        features, importances = zip(*top_features)
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(features)), importances, color='skyblue', alpha=0.7)
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(f'Feature Importance - {model_name}')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, evaluation_results: Dict[str, Dict[str, float]]) -> None:
        """
        Plot comparison of model performance.
        
        Args:
            evaluation_results: Dictionary mapping model names to metrics
        """
        if not evaluation_results:
            logger.warning("No evaluation results provided")
            return
        
        models = list(evaluation_results.keys())
        metrics = ['RMSE', 'MAE', 'R2']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(metrics):
            values = [evaluation_results[model].get(metric, 0) for model in models]
            
            bars = axes[i].bar(models, values, alpha=0.7, 
                             color=['skyblue', 'lightgreen', 'coral', 'gold', 'lightcoral'])
            
            axes[i].set_title(f'Model Comparison - {metric}')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2, height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        model_name: str = "Model") -> None:
        """
        Plot actual vs predicted values.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
        """
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, color='blue', s=50)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted - {model_name}\nR² = {r2:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      model_name: str = "Model") -> None:
        """
        Plot residuals (errors) vs predicted values.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
        """
        residuals = y_true - y_pred
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.6, color='red', s=50)
        plt.axhline(y=0, color='black', linestyle='--', lw=2)
        
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'Residuals Plot - {model_name}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_data_distribution(self, X: np.ndarray, y: np.ndarray, 
                              feature_names: Optional[List[str]] = None,
                              dataset_name: str = "Dataset") -> None:
        """
        Plot data distribution and correlations.
        
        Args:
            X: Feature data
            y: Target data
            feature_names: Names of features
            dataset_name: Name of the dataset
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['Price'] = y
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{dataset_name} - Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Price distribution
        axes[0, 0].hist(y, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Price Distribution')
        axes[0, 0].set_xlabel('Price')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Correlation heatmap (top features)
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:8]
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[0, 1], fmt='.2f', cbar_kws={'shrink': 0.8})
            axes[0, 1].set_title('Feature Correlation Matrix')
        
        # 3. Price vs most correlated feature
        if len(numeric_cols) > 1:
            correlations = df[numeric_cols].corr()['Price'].abs().sort_values(ascending=False)
            best_feature = correlations.index[1]  # Skip 'Price' itself
            
            axes[0, 2].scatter(df[best_feature], df['Price'], alpha=0.5, color='green')
            axes[0, 2].set_title(f'Price vs {best_feature}')
            axes[0, 2].set_xlabel(best_feature)
            axes[0, 2].set_ylabel('Price')
        
        # 4. Feature distributions (box plots for top 4 features)
        if len(numeric_cols) > 1:
            top_features = numeric_cols[:4]
            df[top_features].boxplot(ax=axes[1, 0])
            axes[1, 0].set_title('Feature Distributions')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_time_series(self, data: pd.DataFrame, metro_name: str, 
                        value_column: str = 'value') -> None:
        """
        Plot time series data for a specific metro area.
        
        Args:
            data: DataFrame with time series data
            metro_name: Name of the metro area
            value_column: Name of the value column
        """
        metro_data = data[data['metro'] == metro_name]
        
        if metro_data.empty:
            logger.warning(f"No data found for metro: {metro_name}")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(metro_data['date'], metro_data[value_column], 
                marker='o', linewidth=2, markersize=4)
        
        plt.title(f'Housing Prices Over Time - {metro_name}')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_plot(self, data: pd.DataFrame, plot_type: str = 'scatter',
                               x_col: str = 'x', y_col: str = 'y', 
                               title: str = "Interactive Plot") -> Optional[Any]:
        """
        Create interactive plot using Plotly.
        
        Args:
            data: DataFrame with data
            plot_type: Type of plot ('scatter', 'line', 'bar')
            x_col: Name of x column
            y_col: Name of y column
            title: Plot title
            
        Returns:
            Plotly figure or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for interactive plots")
            return None
        
        if plot_type == 'scatter':
            fig = px.scatter(data, x=x_col, y=y_col, title=title)
        elif plot_type == 'line':
            fig = px.line(data, x=x_col, y=y_col, title=title)
        elif plot_type == 'bar':
            fig = px.bar(data, x=x_col, y=y_col, title=title)
        else:
            logger.warning(f"Unknown plot type: {plot_type}")
            return None
        
        fig.show()
        return fig
    
    def save_plot(self, filename: str, dpi: int = 300) -> None:
        """
        Save the current plot.
        
        Args:
            filename: Name of the file to save
            dpi: Resolution for saved plot
        """
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        logger.info(f"Plot saved to {filename}")
    
    def plot_model_performance_over_time(self, performance_history: Dict[str, List[float]],
                                       model_names: List[str]) -> None:
        """
        Plot model performance over time (for iterative training).
        
        Args:
            performance_history: Dictionary mapping metric names to lists of values
            model_names: List of model names
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Over Time', fontsize=16, fontweight='bold')
        
        metrics = ['RMSE', 'MAE', 'R2', 'MSE']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            if metric in performance_history:
                values = performance_history[metric]
                epochs = range(1, len(values) + 1)
                
                ax.plot(epochs, values, marker='o', linewidth=2, markersize=4)
                ax.set_title(f'{metric} Over Time')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
