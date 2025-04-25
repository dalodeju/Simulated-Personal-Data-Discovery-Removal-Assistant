"""Test configuration loading functionality"""

import pytest
import yaml
import os

def test_config_file_exists():
    """Test that the config file exists"""
    assert os.path.exists('config.yaml'), "Config file not found"

def test_config_file_readable():
    """Test that the config file can be read and parsed"""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        # Check main sections exist
        assert 'web_scraper' in config
        assert 'data_analyzer' in config
        assert 'risk_evaluator' in config
        assert 'logging' in config
        
        # Check some key settings
        assert isinstance(config['web_scraper']['max_retries'], int)
        assert isinstance(config['web_scraper']['confidence_threshold'], float)
        assert isinstance(config['risk_evaluator']['weights']['sensitivity'], float)
        
    except Exception as e:
        pytest.fail(f"Failed to read config: {str(e)}")

def test_config_values_valid():
    """Test that config values are within expected ranges"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Check thresholds are between 0 and 1
    assert 0 <= config['web_scraper']['confidence_threshold'] <= 1
    
    # Check weights sum to 1
    weights = config['risk_evaluator']['weights']
    assert abs(sum(weights.values()) - 1.0) < 0.01
    
    # Check risk thresholds are in ascending order
    thresholds = config['risk_evaluator']['risk_thresholds']
    assert thresholds['low'] < thresholds['medium'] < thresholds['high'] 