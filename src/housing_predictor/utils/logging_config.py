"""
Logging configuration for the Housing Price Predictor.

This module sets up comprehensive logging for the application.
"""

import logging
import sys
from pathlib import Path
from loguru import logger
from housing_predictor.config.settings import config


def setup_logging():
    """Setup logging configuration."""
    # Remove default loguru handler
    logger.remove()
    
    # Get logging configuration
    log_config = config.logging_config
    log_level = log_config.get('level', 'INFO')
    log_file = log_config.get('file', 'logs/housing_predictor.log')
    log_format = log_config.get('format', 
        "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}")
    
    # Create logs directory
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add console handler
    logger.add(
        sys.stdout,
        level=log_level,
        format=log_format,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Add file handler
    logger.add(
        log_file,
        level=log_level,
        format=log_format,
        rotation=log_config.get('rotation', '10 MB'),
        retention=log_config.get('retention', '30 days'),
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    # Add error file handler
    error_log_file = log_path.parent / "errors.log"
    logger.add(
        error_log_file,
        level="ERROR",
        format=log_format,
        rotation="1 day",
        retention="7 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    logger.info("Logging configured successfully")
    return logger


# Setup logging when module is imported
logger = setup_logging()
