"""
Main Application - Home Assistant Intent Prediction System
"""

import os
import sys
import yaml
import logging
import schedule
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from flask import Flask, jsonify

# Local imports
from ha_client import HomeAssistantClient
from feature_engine import FeatureEngine
from ml_models import ModelPipeline
from target_generator import analyze_actual_room_usage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/intent-predictor/ha-intent-predictor/logs/intent_predictor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class IntentPredictionApp:
    """Main application class"""
    
    def __init__(self, config_path: str = '/home/intent-predictor/ha-intent-predictor/config/config.yaml'):
        """Initialize the application"""
        
        logger.info("Starting Intent Prediction System")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.ha_client = HomeAssistantClient(
            url=self.config['home_assistant']['url'],
            token=self.config['home_assistant']['token'],
            timeout=self.config['home_assistant']['timeout']
        )
        
        self.feature_engine = FeatureEngine(self.config)
        self.model_pipeline = ModelPipeline(self.config)
        
        # Paths
        self.model_path = os.path.join(self.config['system']['model_save_path'], 'pipeline.joblib')
        self.feature_engine_path = os.path.join(self.config['system']['model_save_path'], 'feature_engine.joblib')
        
        # State
        self.is_trained = False
        self.last_prediction_time = None
        self.prediction_history = []
        
        # Create directories
        os.makedirs(self.config['system']['model_save_path'], exist_ok=True)
        os.makedirs(self.config['system']['data_cache_path'], exist_ok=True)
        os.makedirs('/home/intent-predictor/ha-intent-predictor/logs', exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def initialize(self) -> bool:
        """Initialize the application and test connections"""
        
        logger.info("Initializing application")
        
        # Test HA connection
        if not self.ha_client.test_connection():
            logger.error("Failed to connect to Home Assistant")
            return False
        
        # Create output sensors in HA
        self._create_output_sensors()
        
        # Try to load existing models
        if self._load_existing_models():
            logger.info("Loaded existing models")
            self.is_trained = True
        else:
            logger.info("No existing models found, will train from scratch")
            
            # Perform initial training
            if self._initial_training():
                logger.info("Initial training completed successfully")
                self.is_trained = True
            else:
                logger.error("Initial training failed")
                return False
        
        # Schedule periodic tasks
        self._schedule_tasks()
        
        logger.info("Application initialized successfully")
        return True
    
    def _create_output_sensors(self):
        """Create output sensors in Home Assistant"""
        
        logger.info("Creating output sensors in Home Assistant")
        
        sensors = self.config['output_sensors']
        
        sensor_configs = {
            sensors['current_activity']: {'state': 'UNKNOWN', 'unit': None},
            sensors['activity_confidence']: {'state': 0, 'unit': '%'},
            sensors['activity_duration']: {'state': 0, 'unit': 'min'},
            sensors['office_2h']: {'state': 0, 'unit': '%'},
            sensors['bedroom_2h']: {'state': 0, 'unit': '%'},
            sensors['kitchen_livingroom_2h']: {'state': 0, 'unit': '%'},
            sensors['next_room']: {'state': 'unknown', 'unit': None},
            sensors['next_room_confidence']: {'state': 0, 'unit': '%'},
            sensors['model_accuracy']: {'state': 0, 'unit': '%'},
            sensors['last_retrain']: {'state': 'never', 'unit': None}
        }
        
        for sensor_id, config in sensor_configs.items():
            attributes = {
                'friendly_name': sensor_id.replace('sensor.', '').replace('_', ' ').title(),
                'device_class': None,
                'unit_of_measurement': config['unit']
            }
            
            self.ha_client.create_sensor_if_not_exists(sensor_id, config['state'])
            self.ha_client.update_sensor(sensor_id, config['state'], attributes)
    
    def _load_existing_models(self) -> bool:
        """Try to load existing trained models"""
        
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.feature_engine_path):
                self.model_pipeline = ModelPipeline.load(self.model_path)
                self.feature_engine = FeatureEngine.load(self.feature_engine_path)
                logger.info("Successfully loaded existing models")
                return True
        except Exception as e:
            logger.warning(f"Failed to load existing models: {e}")
        
        return False
    
    def _initial_training(self) -> bool:
        """Perform initial model training with historical data"""
        
        logger.info("Starting initial training with historical data")
        
        try:
            # Fetch historical data
            logger.info("Fetching historical sensor data")
            historical_data = self.ha_client.get_historical_data(
                entity_ids=self.config['data']['presence_entities'],
                days=self.config['data']['historical_days']
            )
            
            if len(historical_data) == 0:
                logger.error("No historical data available for training")
                return False
            
            logger.info(f"Retrieved {len(historical_data)} historical records")
            
            # Fit feature engine
            logger.info("Fitting feature engineering pipeline")
            self.feature_engine.fit(historical_data)
            
            # Extract features for training
            logger.info("Extracting features for training")
            features = self._extract_training_features(historical_data)
            
            if len(features) == 0:
                logger.error("No features extracted for training")
                return False
            
            # Create training targets
            logger.info("Creating training targets")
            targets = analyze_actual_room_usage(
                historical_data, self.config['rooms'], len(features)
            )
            
            # Train models
            logger.info("Training ML models")
            self.model_pipeline.fit(features, [], targets)
            
            # Save trained models
            self._save_models()
            
            # Update last retrain sensor
            self._update_retrain_sensor()
            
            logger.info("Initial training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initial training failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _extract_training_features(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from historical data for training"""
        
        # Group data by time windows for feature extraction
        time_windows = []
        
        # Create hourly windows
        start_time = historical_data['last_changed'].min()
        end_time = historical_data['last_changed'].max()
        
        current_time = start_time
        while current_time < end_time:
            window_end = current_time + timedelta(hours=1)
            
            # Get data for this window
            window_data = historical_data[
                (historical_data['last_changed'] >= current_time) &
                (historical_data['last_changed'] < window_end)
            ]
            
            if len(window_data) > 0:
                # Extract features for this time point
                features = self.feature_engine.extract_features(window_data, current_time)
                time_windows.append(features)
            
            current_time = window_end
        
        if len(time_windows) == 0:
            return pd.DataFrame()
        
        # Combine all feature windows
        combined_features = pd.concat(time_windows, ignore_index=True)
        
        logger.info(f"Extracted {len(combined_features)} feature samples")
        return combined_features
    
    def _schedule_tasks(self):
        """Schedule periodic tasks"""
        
        # Schedule predictions every 5 minutes
        schedule.every(self.config['data']['update_interval']).seconds.do(self._update_predictions)
        
        # Schedule daily retraining
        schedule.every().day.at(self.config['ml']['retrain_schedule']).do(self._daily_retrain)
        
        logger.info("Scheduled periodic tasks")
    
    def _update_predictions(self):
        """Update predictions and send to Home Assistant"""
        
        try:
            logger.debug("Updating predictions")
            
            if not self.is_trained:
                logger.warning("Models not trained, skipping prediction update")
                return
            
            # Get current sensor states
            current_states = self.ha_client.get_current_states(
                self.config['data']['presence_entities']
            )
            
            # Convert to DataFrame format
            current_data = []
            now = datetime.now()
            
            for entity_id, state in current_states.items():
                if state == 'on':
                    current_data.append({
                        'entity_id': entity_id,
                        'state': state,
                        'last_changed': now
                    })
            
            current_df = pd.DataFrame(current_data)
            
            # Extract features
            features = self.feature_engine.extract_features(current_df, now)
            
            # Generate predictions
            predictions = self.model_pipeline.predict(features)
            
            # Update Home Assistant sensors
            self._send_predictions_to_ha(predictions)
            
            # Store prediction for validation
            self.prediction_history.append({
                'timestamp': now,
                'predictions': predictions,
                'actual_states': current_states
            })
            
            # Keep only last 24 hours of predictions
            cutoff = now - timedelta(hours=24)
            self.prediction_history = [
                p for p in self.prediction_history 
                if p['timestamp'] > cutoff
            ]
            
            self.last_prediction_time = now
            
        except Exception as e:
            logger.error(f"Prediction update failed: {e}")
            logger.error(traceback.format_exc())
    
    def _send_predictions_to_ha(self, predictions: Dict[str, Any]):
        """Send predictions to Home Assistant sensors"""
        
        sensors = self.config['output_sensors']
        
        sensor_updates = {
            sensors['current_activity']: {
                'state': predictions.get('current_activity', 'UNKNOWN'),
                'attributes': {'last_updated': datetime.now().isoformat()}
            },
            sensors['activity_confidence']: {
                'state': round(predictions.get('activity_confidence', 0) * 100, 1),
                'attributes': {'unit_of_measurement': '%'}
            },
            sensors['activity_duration']: {
                'state': round(predictions.get('activity_duration_minutes', 0), 0),
                'attributes': {'unit_of_measurement': 'min'}
            },
            sensors['office_2h']: {
                'state': round(predictions.get('office_occupancy_2h', 0) * 100, 1),
                'attributes': {'unit_of_measurement': '%'}
            },
            sensors['bedroom_2h']: {
                'state': round(predictions.get('bedroom_occupancy_2h', 0) * 100, 1),
                'attributes': {'unit_of_measurement': '%'}
            },
            sensors['kitchen_livingroom_2h']: {
                'state': round(predictions.get('kitchen_livingroom_occupancy_2h', 0) * 100, 1),
                'attributes': {'unit_of_measurement': '%'}
            },
            sensors['next_room']: {
                'state': predictions.get('next_room', 'unknown'),
                'attributes': {}
            },
            sensors['next_room_confidence']: {
                'state': round(predictions.get('next_room_confidence', 0) * 100, 1),
                'attributes': {'unit_of_measurement': '%'}
            }
        }
        
        self.ha_client.update_multiple_sensors(sensor_updates)
        
        logger.debug("Predictions sent to Home Assistant")
    
    def _daily_retrain(self):
        """Perform daily model retraining"""
        
        logger.info("Starting daily model retraining")
        
        try:
            # Fetch recent data (last 30 days for efficiency)
            recent_data = self.ha_client.get_historical_data(
                entity_ids=self.config['data']['presence_entities'],
                days=30
            )
            
            if len(recent_data) > 1000:  # Only retrain if we have enough new data
                
                # Extract features
                features = self._extract_training_features(recent_data)
                
                if len(features) > 100:
                    # Create targets
                    targets = analyze_actual_room_usage(
                        features, recent_data, self.config['rooms']
                    )
                    
                    # Retrain models
                    self.model_pipeline.fit(features, [], targets)
                    
                    # Save updated models
                    self._save_models()
                    
                    # Update retrain sensor
                    self._update_retrain_sensor()
                    
                    logger.info("Daily retraining completed successfully")
                else:
                    logger.info("Insufficient features for retraining")
            else:
                logger.info("Insufficient new data for retraining")
                
        except Exception as e:
            logger.error(f"Daily retraining failed: {e}")
            logger.error(traceback.format_exc())
    
    def _save_models(self):
        """Save trained models to disk"""
        
        self.model_pipeline.save(self.model_path)
        self.feature_engine.save(self.feature_engine_path)
        
        logger.info("Models saved to disk")
    
    def _update_retrain_sensor(self):
        """Update the last retrain timestamp sensor"""
        
        sensor_id = self.config['output_sensors']['last_retrain']
        timestamp = datetime.now().isoformat()
        
        self.ha_client.update_sensor(sensor_id, timestamp, {
            'friendly_name': 'Last Model Retrain',
            'device_class': 'timestamp'
        })
    
    def run(self):
        """Main application loop"""
        
        logger.info("Starting main application loop")
        
        # Initialize the application
        if not self.initialize():
            logger.error("Application initialization failed")
            return
        
        # Start the health check server
        health_app = self._create_health_app()
        
        # Run the main loop
        try:
            import threading
            
            # Start health check server in background
            health_thread = threading.Thread(
                target=lambda: health_app.run(
                    host='0.0.0.0', 
                    port=self.config['system']['health_check_port'],
                    debug=False
                ),
                daemon=True
            )
            health_thread.start()
            
            # Main scheduling loop
            while True:
                schedule.run_pending()
                time.sleep(10)  # Check every 10 seconds
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Application error: {e}")
            logger.error(traceback.format_exc())
        finally:
            logger.info("Application shutting down")
    
    def _create_health_app(self) -> Flask:
        """Create health check Flask app"""
        
        app = Flask(__name__)
        
        @app.route('/health')
        def health_check():
            return jsonify({
                'status': 'healthy',
                'is_trained': self.is_trained,
                'last_prediction': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
                'ha_connected': self.ha_client.test_connection()
            })
        
        return app


def main():
    """Main entry point"""
    
    # Create and run the application
    app = IntentPredictionApp()
    app.run()


if __name__ == '__main__':
    main()

