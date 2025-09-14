"""
Configuration management for the Housing Price Predictor.

This module handles loading and managing configuration settings from YAML files
and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from loguru import logger


class Config:
    """Configuration manager for the Housing Price Predictor."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration YAML file. If None, uses default.
        """
        self.config_path = config_path or Path(__file__).parent.parent.parent.parent / "config" / "config.yaml"
        self._config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file) or {}
            logger.info(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            self._config = self._get_default_config()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "data": {
                "zillow_path": "data/zillow_data",
                "cache_dir": "data/cache",
                "synthetic_dir": "data/synthetic"
            },
            "models": {
                "random_forest": {
                    "n_estimators": 100,
                    "max_depth": 20,
                    "random_state": 42
                }
            },
            "training": {
                "test_size": 0.2,
                "random_state": 42
            },
            "logging": {
                "level": "INFO",
                "file": "logs/housing_predictor.log"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.zillow_path')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'data.zillow_path')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        env_mappings = {
            'HOUSING_DATA_PATH': 'data.zillow_path',
            'HOUSING_CACHE_DIR': 'data.cache_dir',
            'HOUSING_LOG_LEVEL': 'logging.level',
            'HOUSING_LOG_FILE': 'logging.file'
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                self.set(config_key, env_value)
                logger.info(f"Updated {config_key} from environment: {env_value}")
    
    def save_config(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            path: Path to save config. If None, uses current config path.
        """
        save_path = path or self.config_path
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as file:
            yaml.dump(self._config, file, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {save_path}")
    
    @property
    def data_paths(self) -> Dict[str, str]:
        """Get data-related paths."""
        return self.get('data', {})
    
    @property
    def model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get model configurations."""
        return self.get('models', {})
    
    @property
    def training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get('training', {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get('logging', {})


# Global configuration instance
config = Config()
