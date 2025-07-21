#\!/usr/bin/env python3
"""
Configuration loader for HA Intent Predictor system.
Based on the deployment configuration logic from deploy.py.
"""

import json
import yaml
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConfigLoader:
    """Configuration loader that supports JSON and YAML files"""
    
    def __init__(self, config_path: str = "config/system.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        
        # Fall back to ha_config.json if it exists
        ha_config_path = Path("config/ha_config.json")
        if ha_config_path.exists():
            with open(ha_config_path, "r") as f:
                ha_config = json.load(f)
            return self._expand_from_ha_config(ha_config)
        
        # Create default config based on deploy.py logic
        return self._create_default_config()
    
    def _expand_from_ha_config(self, ha_config: Dict[str, Any]) -> Dict[str, Any]:
        """Expand HA config to full system config using deploy.py structure"""
        return {
            "home_assistant": {
                "url": ha_config.get("ha_url", "http://localhost:8123"),
                "token": ha_config.get("access_token", "")
            },
            "database": {
                "timescale": ha_config.get("database", {
                    "host": "localhost",
                    "port": 5432,
                    "database": "ha_predictor",
                    "user": "ha_predictor",
                    "password": "hapredictor_db_pass"
                })
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0
            },
            "kafka": {
                "bootstrap_servers": ha_config.get("kafka", {}).get("bootstrap_servers", ["localhost:9092"]),
                "topics": {
                    "sensor_events": ha_config.get("kafka", {}).get("topic", "sensor_events"),
                    "predictions": "predictions",
                    "anomalies": "anomalies"
                }
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "log_level": "info"
            },
            "monitoring": {
                "prometheus_enabled": True,
                "grafana_enabled": True
            },
            "model_storage": {
                "backend": "filesystem",
                "path": "models/",
                "retention_days": 30
            }
        }
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration using deploy.py structure"""
        return {
            "home_assistant": {
                "url": "http://homeassistant.local:8123",
                "token": "YOUR_LONG_LIVED_ACCESS_TOKEN"
            },
            "database": {
                "timescale": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "ha_predictor",
                    "user": "ha_predictor",
                    "password": "hapredictor_db_pass"
                }
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0
            },
            "kafka": {
                "bootstrap_servers": ["localhost:9092"],
                "topics": {
                    "sensor_events": "sensor_events"
                }
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (deploy.py style)"""
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_sensors(self) -> List[str]:
        """Get all sensors for ML processing from sensors.yaml configuration"""
        sensors_config = self._load_sensors_config()
        
        # Flatten all sensor categories into a single list
        all_sensors = []
        for category, sensor_list in sensors_config.get('sensors', {}).items():
            if isinstance(sensor_list, list):
                all_sensors.extend(sensor_list)
        
        return all_sensors
    
    def _load_sensors_config(self) -> Dict[str, Any]:
        """Load sensors configuration from sensors.yaml"""
        sensors_path = Path("config/sensors.yaml")
        
        if not sensors_path.exists():
            # Fallback to minimal sensor set if config doesn't exist
            return {
                'sensors': {
                    'presence': [
                        'binary_sensor.presence_livingroom_full',
                        'binary_sensor.kitchen_pressence_full_kitchen',
                        'binary_sensor.bedroom_presence_sensor_full_bedroom',
                        'binary_sensor.office_presence_full_office'
                    ]
                }
            }
        
        with open(sensors_path, 'r') as f:
            return yaml.safe_load(f)
