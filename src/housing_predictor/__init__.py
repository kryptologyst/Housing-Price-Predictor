"""
Housing Price Predictor - A modern ML system for housing price prediction.

This package provides comprehensive tools for housing price prediction using
state-of-the-art machine learning techniques and real-world datasets.
"""

__version__ = "2.0.0"
__author__ = "AI Projects Series"
__email__ = "your.email@example.com"

from .core.predictor import HousingPredictor
from .core.data_loader import DataLoader
from .core.models import ModelFactory
from .utils.visualization import VisualizationUtils
from .utils.preprocessing import PreprocessingUtils

__all__ = [
    "HousingPredictor",
    "DataLoader", 
    "ModelFactory",
    "VisualizationUtils",
    "PreprocessingUtils",
]
