"""Common utilities for all agents."""

import logging
import yaml
from typing import Dict, Any
from pathlib import Path

def setup_logging(name: str, config: Dict[str, Any] = None) -> logging.Logger:
    """Set up logging with consistent configuration.
    
    Args:
        name: Logger name
        config: Optional logging configuration
        
    Returns:
        Configured logger
    """
    default_config = {
        'level': logging.INFO,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
    
    if config:
        default_config.update(config)
    
    logging.basicConfig(**default_config)
    return logging.getLogger(name)

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found, using defaults")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
        raise

def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to ensure
    """
    Path(path).mkdir(parents=True, exist_ok=True) 