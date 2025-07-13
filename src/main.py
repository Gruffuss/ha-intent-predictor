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
from advanced_feature_engine import AdvancedFeatureEngine
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

class AccuracyTracker:
    """Track prediction accuracy over time"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.room_predictions = []  # Store room occupancy predictions and outcomes
        self.activity_predictions = []  # Store activity predictions and outcomes
        
    def add_prediction(self, prediction: Dict[str, Any], timestamp: datetime):
        """Store a prediction for later validation"""
        
        # Store room occupancy predictions
        room_pred = {
            'timestamp': timestamp,
            'predictions': {
                'office_2h': prediction.get('office_occupancy_2h', 0),
                'bedroom_2h': prediction.get('bedroom_occupancy_2h', 0),
                'kitchen_livingroom_2h': prediction.get('kitchen_livingroom_occupancy_2h', 0)
            },
            'activity': prediction.get('current_activity', 'UNKNOWN'),
            'next_room': prediction.get('next_room', 'unknown'),
            'validated': False
        }
        
        self.room_predictions.append(room_pred)
        
        # Keep only recent predictions
        if len(self.room_predictions) > self.window_size:
            self.room_predictions.pop(0)
    
    def validate_predictions(self, ha_client, rooms_config: Dict) -> float:
        """Validate old predictions against actual outcomes"""
        
        if len(self.room_predictions) == 0:
            return 0.0
        
        from datetime import timezone, timedelta as td
        # Use same timezone as prediction timestamps
        local_tz = timezone(td(hours=3))
        now = datetime.now(local_tz)
        validated_count = 0
        correct_predictions = 0
        
        # Check predictions that are old enough to validate (2+ hours)
        for pred in self.room_predictions:
            if pred['validated']:
                continue
                
            time_elapsed = (now - pred['timestamp']).total_seconds() / 3600  # hours
            
            if time_elapsed >= 2.0:  # 2 hours passed, can validate 2h prediction
                # Get actual occupancy data for the 2-hour period after prediction
                validation_start = pred['timestamp']
                validation_end = pred['timestamp'] + timedelta(hours=2)
                
                try:
                    # Get recent historical data for validation
                    actual_data = ha_client.get_historical_data(
                        entity_ids=[entity for entities in rooms_config.values() for entity in entities],
                        days=1  # Get recent data
                    )
                    
                    # Filter to validation period
                    if len(actual_data) > 0:
                        actual_data = actual_data[
                            (actual_data['last_changed'] >= validation_start) &
                            (actual_data['last_changed'] <= validation_end)
                        ]
                    
                    if len(actual_data) > 0:
                        # Calculate actual room occupancy percentages
                        actual_occupancy = self._calculate_actual_occupancy(actual_data, rooms_config)
                        
                        # Compare predictions vs actual
                        room_accuracy = self._calculate_room_accuracy(
                            pred['predictions'], actual_occupancy
                        )
                        
                        if room_accuracy >= 0.7:  # 70% threshold for "correct"
                            correct_predictions += 1
                        
                        validated_count += 1
                        pred['validated'] = True
                        
                except Exception as e:
                    logger.warning(f"Failed to validate prediction: {e}")
                    pred['validated'] = True  # Mark as validated to avoid retrying
        
        # Calculate accuracy percentage
        if validated_count > 0:
            accuracy = (correct_predictions / validated_count) * 100
            return min(accuracy, 100.0)
        
        return 0.0
    
    def _calculate_actual_occupancy(self, data: pd.DataFrame, rooms_config: Dict) -> Dict[str, float]:
        """Calculate actual room occupancy percentages from historical data"""
        
        room_occupancy = {}
        total_time = len(data) if len(data) > 0 else 1
        
        for room, entities in rooms_config.items():
            room_data = data[data['entity_id'].isin(entities)]
            
            # Count 'on' states as occupancy
            occupancy_time = len(room_data[room_data['state'] == 'on'])
            occupancy_percentage = occupancy_time / total_time
            
            room_occupancy[f'{room}_2h'] = occupancy_percentage
        
        return room_occupancy
    
    def _calculate_room_accuracy(self, predicted: Dict[str, float], actual: Dict[str, float]) -> float:
        """Calculate accuracy between predicted and actual room occupancy"""
        
        total_error = 0
        room_count = 0
        
        for room_key in predicted.keys():
            if room_key in actual:
                pred_val = predicted[room_key]
                actual_val = actual[room_key]
                
                # Calculate absolute percentage error
                error = abs(pred_val - actual_val)
                total_error += error
                room_count += 1
        
        if room_count > 0:
            average_error = total_error / room_count
            accuracy = max(0, 1 - average_error)  # Convert error to accuracy
            return accuracy
        
        return 0.0

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
        
        self.feature_engine = AdvancedFeatureEngine(self.config)
        self.model_pipeline = ModelPipeline(self.config)
        self.accuracy_tracker = AccuracyTracker()
        
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
            sensors['office_15min']: {'state': 0, 'unit': '%'},
            sensors['bedroom_15min']: {'state': 0, 'unit': '%'},
            sensors['kitchen_livingroom_15min']: {'state': 0, 'unit': '%'},
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
                self.feature_engine = AdvancedFeatureEngine.load(self.feature_engine_path)
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
        
        # Sample time points throughout the historical data for feature extraction
        start_time = historical_data['last_changed'].min()
        end_time = historical_data['last_changed'].max()
        total_duration = (end_time - start_time).total_seconds()
        
        # Use actual sensor state changes as training points (much more data!)
        # Get all unique timestamps where sensors changed state
        sensor_change_times = historical_data['last_changed'].drop_duplicates().sort_values()
        
        # Filter to have sufficient history (4+ hours) and future data (1+ hour)
        valid_times = []
        for timestamp in sensor_change_times:
            if (timestamp >= start_time + timedelta(hours=4) and 
                timestamp <= end_time - timedelta(hours=1)):
                valid_times.append(timestamp)
        
        # Use ALL meaningful sensor changes for maximum training data, but limit for memory safety
        sample_times = valid_times
        
        # Memory safety: limit training samples for very large datasets
        max_training_samples = 100000  # Conservative limit for 8-hour stable training
        if len(sample_times) > max_training_samples:
            logger.warning(f"Limiting training samples to {max_training_samples} for memory safety (from {len(sample_times)})")
            # Sample randomly to preserve temporal distribution
            import random
            random.seed(42)  # Reproducible sampling
            sample_times = sorted(random.sample(sample_times, max_training_samples))
        
        logger.info(f"Using {len(sample_times)} sensor change events as training points (from {len(sensor_change_times)} total changes)")
        
        # Extract features for each sample time
        feature_samples = []
        
        total_samples = len(sample_times)
        import time
        self._training_start_time = time.time()  # Track training start time
        for idx, sample_time in enumerate(sample_times):
            # More frequent progress reporting for large datasets (every 1% or 1000 samples)
            progress_interval = max(1000, total_samples // 100)  # Report every 1% or 1000 samples
            if idx % progress_interval == 0:
                progress = (idx / total_samples) * 100
                logger.info(f"Feature extraction progress: {progress:.1f}% ({idx}/{total_samples} samples)")
                
                # Memory monitoring during training
                self._log_memory_usage_if_available(f"Feature extraction at {progress:.1f}%")
                
                # Estimate remaining time for long training
                if idx > 0:
                    import time
                    elapsed = time.time() - getattr(self, '_training_start_time', time.time())
                    rate = idx / elapsed if elapsed > 0 else 0
                    remaining_samples = total_samples - idx
                    eta_seconds = remaining_samples / rate if rate > 0 else 0
                    eta_hours = eta_seconds / 3600
                    logger.info(f"Estimated time remaining: {eta_hours:.1f} hours (rate: {rate:.1f} samples/sec)")
            
            # Get historical data up to this point (last 4 hours)
            cutoff_time = sample_time - timedelta(hours=4)
            window_data = historical_data[
                (historical_data['last_changed'] >= cutoff_time) &
                (historical_data['last_changed'] <= sample_time)
            ]
            
            if len(window_data) > 10:  # Need sufficient data for feature extraction
                try:
                    features = self.feature_engine.extract_features(window_data, sample_time)
                    if len(features) > 0:
                        feature_samples.append(features)
                except Exception as e:
                    logger.warning(f"Failed to extract features for {sample_time}: {e}")
                    continue
        
        if len(feature_samples) == 0:
            logger.error("No feature samples could be extracted from historical data")
            return pd.DataFrame()
        
        # Combine all feature samples with proper handling of duplicate columns
        if len(feature_samples) == 0:
            logger.error("No feature samples could be extracted from historical data")
            return pd.DataFrame()
        
        # Reset index for each DataFrame to avoid conflicts
        for i, df in enumerate(feature_samples):
            feature_samples[i] = df.reset_index(drop=True)
        
        # Combine with error handling
        try:
            combined_features = pd.concat(feature_samples, ignore_index=True)
        except Exception as e:
            logger.error(f"Failed to combine feature samples: {e}")
            # Fallback: use the first sample as template and ensure consistent columns
            template_columns = feature_samples[0].columns.tolist()
            cleaned_samples = []
            for df in feature_samples:
                # Ensure all DataFrames have the same columns in the same order
                df_cleaned = df.reindex(columns=template_columns, fill_value=0)
                cleaned_samples.append(df_cleaned)
            combined_features = pd.concat(cleaned_samples, ignore_index=True)
        
        logger.info(f"Extracted {len(combined_features)} feature samples")
        return combined_features
    
    def _log_memory_usage_if_available(self, stage: str):
        """Log memory usage if psutil is available"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            virtual_memory = psutil.virtual_memory()
            
            logger.info(f"Memory at {stage}: Process: {memory_mb:.1f}MB, System: {virtual_memory.percent:.1f}%")
            
            # Warn if memory usage is getting high
            if virtual_memory.percent > 85:
                logger.warning(f"High system memory usage: {virtual_memory.percent:.1f}%")
            if memory_mb > 2000:  # Process using more than 2GB
                logger.warning(f"High process memory usage: {memory_mb:.1f}MB")
        except ImportError:
            # psutil not available, skip memory monitoring
            pass
        except Exception as e:
            logger.debug(f"Failed to get memory usage at {stage}: {e}")
    
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
            
            # Get recent historical data for feature extraction (same as training)
            from datetime import timezone, timedelta as td
            # Use GMT+3 timezone to match user's location
            local_tz = timezone(td(hours=3))
            now = datetime.now(local_tz)
            recent_data = self.ha_client.get_historical_data(
                entity_ids=self.config['data']['presence_entities'],
                days=1  # Get last 24 hours for feature extraction
            )
            
            if len(recent_data) == 0:
                logger.warning("No recent data available for prediction")
                return
            
            # Filter to recent data only
            cutoff_time = now - timedelta(hours=4)  # Use last 4 hours for features
            recent_data = recent_data[recent_data['last_changed'] >= cutoff_time]
            
            if len(recent_data) == 0:
                logger.warning("No recent sensor activity for prediction")
                return
            
            # Extract features using the same method as training
            features = self.feature_engine.extract_features(recent_data, now)
            
            if len(features) == 0:
                logger.warning("No features extracted for prediction")
                return
            
            # Note: Removed current sensor state features to match training data
            # The AdvancedFeatureEngine already extracts sufficient current state information
            
            # Generate predictions
            predictions = self.model_pipeline.predict(features)
            
            # Store prediction for accuracy tracking
            self.accuracy_tracker.add_prediction(predictions, now)
            
            # Validate old predictions and calculate accuracy
            accuracy = self.accuracy_tracker.validate_predictions(self.ha_client, self.config['rooms'])
            predictions['model_accuracy'] = accuracy
            
            # Update Home Assistant sensors
            self._send_predictions_to_ha(predictions)
            
            # Store prediction for validation
            self.prediction_history.append({
                'timestamp': now,
                'predictions': predictions,
                'actual_states': recent_data.to_dict('records') if len(recent_data) > 0 else []
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
            sensors['office_15min']: {
                'state': round(predictions.get('office_occupancy_15min', 0) * 100, 1),
                'attributes': {'unit_of_measurement': '%'}
            },
            sensors['bedroom_15min']: {
                'state': round(predictions.get('bedroom_occupancy_15min', 0) * 100, 1),
                'attributes': {'unit_of_measurement': '%'}
            },
            sensors['kitchen_livingroom_15min']: {
                'state': round(predictions.get('kitchen_livingroom_occupancy_15min', 0) * 100, 1),
                'attributes': {'unit_of_measurement': '%'}
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
            },
            sensors['model_accuracy']: {
                'state': round(predictions.get('model_accuracy', 0), 1),
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
                        recent_data, self.config['rooms'], len(features)
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

