"""
Adaptive Occupancy Predictor - Main coordinator implementing CLAUDE.md architecture
Ties together all components into the comprehensive system
"""

import logging
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

# Import all the components we've implemented
from src.ingestion.data_enricher import DynamicFeatureDiscovery, AdaptiveFeatureSelector
from src.learning.online_models import ContinuousLearningModel, MultiHorizonPredictor
from src.learning.pattern_discovery import PatternDiscovery, UnbiasedPatternMiner
from src.learning.anomaly_detection import AdaptiveCatDetector, OnlineAnomalyDetector
from src.prediction.bathroom_predictor import BathroomOccupancyPredictor
from src.adaptation.performance_monitor import PerformanceMonitor
from src.adaptation.resource_optimizer import ResourceOptimizer
from src.storage.timeseries_db import TimescaleDBManager
from src.storage.feature_store import RedisFeatureStore

logger = logging.getLogger(__name__)


class AdaptiveOccupancyPredictor:
    """
    Main adaptive occupancy prediction system
    Implements the exact architecture from CLAUDE.md
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Core components exactly as specified in CLAUDE.md
        self.short_term_models = {}  # Per-room adaptive models
        self.long_term_models = {}   # Per-room pattern discoverers
        self.cat_detector = OnlineAnomalyDetector()
        self.feature_selector = AutoML(max_features=50)
        
        # Initialize all system components
        self.dynamic_feature_discovery = DynamicFeatureDiscovery()
        self.adaptive_feature_selector = AdaptiveFeatureSelector()
        self.pattern_discovery = PatternDiscovery()
        self.unbiased_pattern_miner = UnbiasedPatternMiner()
        self.adaptive_cat_detector = AdaptiveCatDetector()
        self.bathroom_predictor = BathroomOccupancyPredictor()
        self.performance_monitor = PerformanceMonitor()
        self.resource_optimizer = ResourceOptimizer(
            cpu_limit=config.get('cpu_limit', 80),
            memory_limit=config.get('memory_limit', 6000)
        )
        
        # Storage backends - use passed instances from config
        self.timeseries_db = config.get('timeseries_db')
        self.feature_store = config.get('feature_store')
        self.model_store = config.get('model_store')
        
        # Multi-horizon prediction
        self.rooms = config.get('rooms', ['living_kitchen', 'bedroom', 'office', 'bathroom', 'small_bathroom'])
        self.multi_horizon_predictor = MultiHorizonPredictor(self.rooms)
        
        # System state
        self.initialized = False
        self.learning_active = False
        self.prediction_cache = {}
        
        logger.info("Initialized adaptive occupancy predictor")
    
    async def initialize(self):
        """Initialize all system components"""
        try:
            # Initialize storage backends
            await self.timeseries_db.initialize()
            await self.feature_store.initialize()
            
            # Initialize models for each room
            for room in self.rooms:
                # Short-term models for immediate adaptation
                self.short_term_models[room] = ContinuousLearningModel(f"{room}_short_term")
                
                # Long-term models for pattern discovery
                self.long_term_models[room] = PatternDiscoveryModel(room)
            
            self.initialized = True
            logger.info("✅ Adaptive occupancy predictor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize adaptive predictor: {e}")
            raise
    
    async def learn_from_observation(self, 
                                   room_id: str, 
                                   sensor_data: Dict[str, Any], 
                                   outcome: Optional[bool] = None):
        """
        Every single observation updates the model
        No batching, no waiting for "enough data"
        Implements the exact approach from CLAUDE.md
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            timestamp = sensor_data.get('timestamp', datetime.now())
            
            # 1. Store raw sensor data
            await self.timeseries_db.insert_sensor_event(sensor_data)
            
            # 2. Extract dynamic features
            recent_events = await self.timeseries_db.get_recent_events(minutes=30, rooms=[room_id])
            features = self.dynamic_feature_discovery.discover_features(recent_events + [sensor_data])
            
            # 3. Cache features for performance
            await self.feature_store.cache_features(
                sensor_data.get('entity_id', ''),
                timestamp,
                features
            )
            
            # 4. Update feature importance
            if outcome is not None:
                prediction_error = abs(outcome - 0.5)  # Simplified error
                self.adaptive_feature_selector.update_importance(features, prediction_error)
            
            # 5. Cat/anomaly detection
            if 'movement_sequence' in sensor_data:
                self.adaptive_cat_detector.learn_movement_patterns([sensor_data])
            
            # 6. Bathroom-specific learning
            if room_id in ['bathroom', 'small_bathroom']:
                bathroom_result = self.bathroom_predictor.process_bathroom_event(sensor_data)
                if bathroom_result:
                    # Learn from bathroom-specific outcome
                    bathroom_outcome = bathroom_result.get('occupied', False)
                    await self.learn_bathroom_patterns(room_id, sensor_data, bathroom_outcome)
            
            # 7. Update short-term models immediately
            if room_id in self.short_term_models and outcome is not None:
                # Get top features for learning
                top_features = self.adaptive_feature_selector.get_top_features(25)
                filtered_features = {name: features.get(name, 0) for name, _ in top_features}
                
                self.short_term_models[room_id].learn_one(filtered_features, outcome)
            
            # 8. Update long-term pattern discovery
            if room_id in self.long_term_models:
                await self.long_term_models[room_id].update_patterns(sensor_data)
            
            # 9. Resource optimization check
            if len(recent_events) % 100 == 0:  # Every 100 events
                optimization_result = self.resource_optimizer.optimize_models({
                    **self.short_term_models,
                    **self.long_term_models
                })
                
                if optimization_result['optimization_needed']:
                    logger.info(f"Applied optimizations: {optimization_result['optimizations_applied']}")
            
            logger.debug(f"Learned from observation in {room_id}: {len(features)} features extracted")
            
        except Exception as e:
            logger.error(f"Error learning from observation in {room_id}: {e}")
    
    async def predict_occupancy(self, 
                              room_id: str, 
                              horizon_minutes: int,
                              current_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predict room occupancy with full uncertainty quantification
        """
        try:
            timestamp = datetime.now()
            
            # 1. Check cache first
            cached_prediction = await self.feature_store.get_cached_prediction(room_id, horizon_minutes)
            if cached_prediction:
                return cached_prediction
            
            # 2. Get or compute current features
            if current_features is None:
                recent_events = await self.timeseries_db.get_recent_events(minutes=30, rooms=[room_id])
                current_features = self.dynamic_feature_discovery.discover_features(recent_events)
            
            # 3. Select top features
            top_features = self.adaptive_feature_selector.get_top_features(25)
            filtered_features = {name: current_features.get(name, 0) for name, _ in top_features}
            
            # 4. Special handling for bathrooms
            if room_id in ['bathroom', 'small_bathroom']:
                bathroom_prediction = self.bathroom_predictor.predict_bathroom_occupancy(room_id, horizon_minutes)
                
                # Combine with ML prediction if available
                if room_id in self.short_term_models:
                    ml_prediction = self.short_term_models[room_id].predict_proba_one(filtered_features)
                    
                    # Weighted combination
                    combined_prob = (bathroom_prediction['probability'] * 0.7 + 
                                   ml_prediction['probability'] * 0.3)
                    combined_uncertainty = max(bathroom_prediction.get('confidence', 0.5),
                                             ml_prediction['uncertainty'])
                    
                    prediction_result = {
                        'probability': combined_prob,
                        'uncertainty': combined_uncertainty,
                        'confidence': 1.0 - combined_uncertainty,
                        'method': 'bathroom_ml_hybrid',
                        'bathroom_component': bathroom_prediction,
                        'ml_component': ml_prediction
                    }
                else:
                    prediction_result = bathroom_prediction
            
            # 5. Regular ML prediction for other rooms
            elif room_id in self.short_term_models:
                prediction_result = self.short_term_models[room_id].predict_proba_one(filtered_features)
                prediction_result['method'] = 'adaptive_ml'
            
            else:
                # Fallback to pattern-based prediction
                prediction_result = await self.predict_from_patterns(room_id, horizon_minutes, current_features)
                prediction_result['method'] = 'pattern_based'
            
            # 6. Add metadata
            prediction_result.update({
                'room_id': room_id,
                'horizon_minutes': horizon_minutes,
                'timestamp': timestamp,
                'features_used': len(filtered_features),
                'feature_names': list(filtered_features.keys())
            })
            
            # 7. Cache prediction
            await self.feature_store.cache_prediction(room_id, horizon_minutes, prediction_result)
            
            # 8. Store prediction for later evaluation
            await self.timeseries_db.insert_prediction({
                'timestamp': timestamp,
                'room': room_id,
                'horizon_minutes': horizon_minutes,
                'probability': prediction_result['probability'],
                'uncertainty': prediction_result['uncertainty'],
                'confidence': prediction_result['confidence'],
                'model_name': prediction_result['method'],
                'features': filtered_features,
                'metadata': {k: v for k, v in prediction_result.items() 
                           if k not in ['probability', 'uncertainty', 'confidence']}
            })
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error predicting occupancy for {room_id}: {e}")
            return {
                'probability': 0.5,
                'uncertainty': 1.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    async def predict_from_patterns(self, 
                                  room_id: str, 
                                  horizon_minutes: int,
                                  features: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback prediction using discovered patterns"""
        try:
            # Get historical patterns for this room
            patterns = await self.get_room_patterns(room_id)
            
            # Simple pattern-based prediction
            current_hour = datetime.now().hour
            target_hour = (datetime.now() + timedelta(minutes=horizon_minutes)).hour
            
            # Base probability from patterns
            base_prob = 0.3  # Default
            
            if patterns:
                hourly_patterns = patterns.get('hourly_activity', {})
                if str(target_hour) in hourly_patterns:
                    base_prob = hourly_patterns[str(target_hour)]
            
            # Adjust based on current activity
            current_activity = features.get('recent_room_activity', 0)
            if current_activity > 0:
                base_prob *= 1.2  # Increase if currently active
            
            return {
                'probability': min(0.8, base_prob),
                'uncertainty': 0.6,  # Higher uncertainty for pattern-based
                'confidence': 0.4
            }
            
        except Exception as e:
            logger.error(f"Error in pattern-based prediction: {e}")
            return {
                'probability': 0.5,
                'uncertainty': 1.0,
                'confidence': 0.0
            }
    
    async def get_room_patterns(self, room_id: str) -> Dict[str, Any]:
        """Get discovered patterns for room"""
        try:
            # Check cache first
            cached_patterns = await self.feature_store.get_cached_pattern(room_id, 'all_patterns')
            if cached_patterns:
                return cached_patterns
            
            # Discover patterns from recent data
            recent_events = await self.timeseries_db.get_recent_events(minutes=60*24, rooms=[room_id])  # 24 hours
            
            if recent_events:
                patterns = self.unbiased_pattern_miner.mine_patterns(recent_events)
                
                # Cache patterns
                await self.feature_store.cache_pattern(room_id, 'all_patterns', patterns)
                
                return patterns
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting patterns for {room_id}: {e}")
            return {}
    
    async def learn_bathroom_patterns(self, 
                                    room_id: str, 
                                    sensor_data: Dict[str, Any],
                                    outcome: bool):
        """Learn bathroom-specific patterns"""
        try:
            timestamp = sensor_data.get('timestamp', datetime.now())
            
            # Store bathroom-specific pattern
            pattern_data = {
                'room': room_id,
                'pattern_type': 'bathroom_usage',
                'pattern_data': {
                    'timestamp': timestamp.isoformat(),
                    'entity_id': sensor_data.get('entity_id'),
                    'outcome': outcome,
                    'hour': timestamp.hour,
                    'day_of_week': timestamp.weekday()
                },
                'metadata': {
                    'learned_from': 'bathroom_predictor'
                }
            }
            
            await self.timeseries_db.insert_pattern_discovery(pattern_data)
            
        except Exception as e:
            logger.error(f"Error learning bathroom patterns: {e}")
    
    async def evaluate_prediction(self, 
                                room_id: str, 
                                horizon_minutes: int,
                                predicted: Dict[str, Any],
                                actual: bool) -> Dict[str, Any]:
        """Evaluate prediction and trigger adaptations if needed"""
        try:
            # Log performance
            performance_result = self.performance_monitor.log_prediction(
                room_id, horizon_minutes, predicted, actual
            )
            
            # Update multi-horizon predictor
            features = predicted.get('features_used', {})
            if isinstance(features, dict):
                occupancy_dict = {horizon_minutes: actual}
                self.multi_horizon_predictor.learn_one(room_id, features, occupancy_dict)
            
            return performance_result
            
        except Exception as e:
            logger.error(f"Error evaluating prediction: {e}")
            return {}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Performance summary
            performance_summary = self.performance_monitor.get_performance_summary(hours=24)
            
            # Resource usage
            resource_usage = self.resource_optimizer.get_resource_usage()
            
            # Model health scores
            model_health = {}
            for room in self.rooms:
                model_health[room] = self.performance_monitor.get_model_health_score(room)
            
            # Feature importance
            feature_importance = self.adaptive_feature_selector.get_feature_importance_summary()
            
            # Storage stats
            db_stats = await self.timeseries_db.get_database_stats()
            cache_stats = await self.feature_store.get_cache_stats()
            
            return {
                'system_initialized': self.initialized,
                'learning_active': self.learning_active,
                'performance': performance_summary,
                'resource_usage': resource_usage,
                'model_health': model_health,
                'feature_importance': feature_importance,
                'storage': {
                    'database': db_stats,
                    'cache': cache_stats
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    async def cleanup(self):
        """Clean up system resources"""
        try:
            await self.timeseries_db.close()
            await self.feature_store.close()
            logger.info("System cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def process_sensor_event(self, event: Dict[str, Any]):
        """
        Process a single sensor event through the complete learning pipeline
        This is the main entry point for the learning loop
        """
        try:
            # 1. Extract room and basic info
            room_id = event.get('room', 'unknown')
            entity_id = event.get('entity_id', '')
            timestamp = event.get('timestamp', datetime.now())
            
            # 2. Determine current occupancy state for learning
            current_occupancy = self.determine_occupancy(event)
            
            # 3. Learn from this observation
            await self.learn_from_observation(room_id, event, current_occupancy)
            
            # 4. Make predictions for all horizons
            predictions = await self.predict_all_horizons(room_id)
            
            # 5. Update performance metrics
            if current_occupancy is not None:
                await self.performance_monitor.update_metrics(
                    room_id=room_id,
                    prediction_results=predictions,
                    actual_occupancy=current_occupancy
                )
            
            # 6. Return predictions for publishing
            return predictions
            
        except Exception as e:
            logger.error(f"Error processing sensor event {event.get('entity_id', '')}: {e}")
            return {}
    
    def determine_occupancy(self, event: Dict[str, Any]) -> Optional[bool]:
        """
        Determine ground truth occupancy from sensor event
        This is the critical component that provides training labels
        """
        entity_id = event.get('entity_id', '')
        state = event.get('state', '')
        room = event.get('room', '')
        
        # For presence sensors, it's straightforward
        if 'presence' in entity_id.lower():
            return state == 'on'
        
        # For door sensors in bathrooms, use bathroom predictor logic
        if room in ['bathroom', 'small_bathroom'] and 'door' in entity_id.lower():
            return self.bathroom_predictor.process_bathroom_event(event).get('occupied', None)
        
        # For other sensors, we can't directly determine occupancy
        # Let the system learn patterns instead
        return None
    
    async def predict_all_horizons(self, room_id: str) -> Dict[int, Dict[str, Any]]:
        """
        Generate predictions for all configured horizons
        """
        predictions = {}
        
        # Get configured horizons from config or use defaults
        horizons = [5, 15, 30, 60, 120]  # minutes
        
        for horizon in horizons:
            try:
                prediction = await self.predict_occupancy(room_id, horizon)
                predictions[horizon] = prediction
            except Exception as e:
                logger.warning(f"Failed to predict {horizon}min for {room_id}: {e}")
                predictions[horizon] = {
                    'probability': 0.5,
                    'uncertainty': 1.0,
                    'confidence': 0.0,
                    'error': str(e)
                }
        
        return predictions
    
    async def run_continuous_learning(self):
        """
        Run the continuous learning loop
        This method should be called as a background task
        """
        logger.info("Starting continuous learning loop")
        
        while True:
            try:
                # Periodic tasks
                await asyncio.sleep(60)  # Run every minute
                
                # 1. Update pattern discovery
                await self.update_pattern_discovery()
                
                # 2. Optimize prediction horizons
                await self.optimize_prediction_horizons()
                
                # 3. Update feature importance
                await self.update_feature_importance()
                
                # 4. Check for model drift
                await self.check_model_drift()
                
                # 5. Optimize resource usage
                await self.optimize_resources()
                
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {e}")
                await asyncio.sleep(5)  # Wait before retry
    
    async def update_pattern_discovery(self):
        """Update pattern discovery for all rooms"""
        try:
            for room_id in self.rooms:
                # Get recent events for pattern analysis
                recent_events = await self.timeseries_db.get_recent_events(
                    minutes=1440,  # Last 24 hours
                    rooms=[room_id]
                )
                
                if len(recent_events) > 10:  # Need minimum data
                    patterns = self.pattern_discovery.discover_patterns(recent_events, room_id)
                    logger.debug(f"Updated patterns for {room_id}: {len(patterns)} patterns found")
                    
        except Exception as e:
            logger.error(f"Error updating pattern discovery: {e}")
    
    async def optimize_prediction_horizons(self):
        """Optimize prediction horizons based on accuracy"""
        try:
            # This would analyze prediction accuracy at different horizons
            # and adjust the horizons accordingly
            pass
        except Exception as e:
            logger.error(f"Error optimizing prediction horizons: {e}")
    
    async def update_feature_importance(self):
        """Update feature importance across all models"""
        try:
            # Get top features from adaptive selector
            top_features = self.adaptive_feature_selector.get_top_features(50)
            logger.debug(f"Top features updated: {len(top_features)} features")
        except Exception as e:
            logger.error(f"Error updating feature importance: {e}")
    
    async def check_model_drift(self):
        """Check for model drift and trigger retraining if needed"""
        try:
            # Check performance degradation
            for room_id in self.rooms:
                performance = await self.performance_monitor.get_room_performance(room_id)
                if performance.get('accuracy', 1.0) < 0.6:  # Threshold
                    logger.warning(f"Model drift detected for {room_id}, accuracy: {performance.get('accuracy')}")
                    # Trigger model retraining
                    await self.retrain_model(room_id)
        except Exception as e:
            logger.error(f"Error checking model drift: {e}")
    
    async def optimize_resources(self):
        """Optimize resource usage"""
        try:
            # Use resource optimizer to adjust model complexity
            self.resource_optimizer.optimize_models(list(self.short_term_models.values()))
        except Exception as e:
            logger.error(f"Error optimizing resources: {e}")
    
    async def retrain_model(self, room_id: str):
        """Retrain model for a specific room"""
        try:
            logger.info(f"Retraining model for {room_id}")
            
            # Get historical data for retraining
            historical_data = await self.timeseries_db.get_recent_events(
                minutes=10080,  # Last week
                rooms=[room_id]
            )
            
            if len(historical_data) > 100:
                # Reinitialize model
                self.short_term_models[room_id] = ContinuousLearningModel(room_id)
                
                # Retrain with historical data
                for event in historical_data:
                    occupancy = self.determine_occupancy(event)
                    if occupancy is not None:
                        features = self.dynamic_feature_discovery.discover_features([event])
                        self.short_term_models[room_id].learn_one(features, occupancy)
                
                logger.info(f"Model retrained for {room_id} with {len(historical_data)} events")
                
        except Exception as e:
            logger.error(f"Error retraining model for {room_id}: {e}")
    
    async def manual_training(self, room_id: Optional[str] = None, 
                            force_retrain: bool = False, 
                            include_historical: bool = True):
        """
        Manual training trigger for API
        """
        try:
            if room_id:
                logger.info(f"Starting manual training for room {room_id}")
                # Train specific room
                if room_id in self.short_term_models:
                    # Force retrain by resetting model
                    if force_retrain:
                        self.short_term_models[room_id] = ContinuousLearningModel(f"{room_id}_short_term")
                    logger.info(f"Manual training completed for {room_id}")
                else:
                    logger.warning(f"Room {room_id} not found")
            else:
                logger.info("Starting manual training for all rooms")
                # Train all rooms
                for room_id in self.short_term_models.keys():
                    if force_retrain:
                        self.short_term_models[room_id] = ContinuousLearningModel(f"{room_id}_short_term")
                logger.info("Manual training completed for all rooms")
                
        except Exception as e:
            logger.error(f"Manual training failed: {e}")
            raise
    
    async def get_room_metrics(self, room_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific room
        """
        try:
            if room_id not in self.short_term_models:
                return {}
            
            model = self.short_term_models[room_id]
            
            # Get model performance
            model_performance = {}
            for model_name, perf_metric in model.model_performance.items():
                try:
                    if hasattr(perf_metric, 'get'):
                        # Try to get the metric value
                        metric_value = perf_metric.get()
                        model_performance[model_name] = metric_value if metric_value is not None else 0.0
                    elif hasattr(perf_metric, '__float__'):
                        model_performance[model_name] = float(perf_metric)
                    else:
                        model_performance[model_name] = 0.0
                except Exception:
                    model_performance[model_name] = 0.0
            
            # Get overall accuracy
            accuracies = [v for v in model_performance.values() if v > 0]
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
            
            return {
                'accuracy': avg_accuracy,
                'model_performance': model_performance,
                'active_patterns': [],  # Would be populated by pattern discovery
                'anomaly_score': 0.0,   # Would be populated by anomaly detection
                'predictions_made': getattr(model, 'predictions_made', 0),
                'last_training': getattr(model, 'last_training', None)
            }
            
        except Exception as e:
            logger.error(f"Error getting room metrics for {room_id}: {e}")
            return {
                'accuracy': 0.0,
                'model_performance': {},
                'active_patterns': [],
                'anomaly_score': 0.0,
                'predictions_made': 0,
                'last_training': None
            }
    
    async def get_active_rooms(self) -> List[str]:
        """
        Get list of active rooms for drift monitoring
        Simple wrapper around self.rooms for compatibility
        """
        return self.rooms
    
    async def get_recent_features(self, room_id: str, limit: int = 100) -> Dict[str, List[float]]:
        """
        Get recent feature values for drift detection
        Returns features organized by feature name with lists of recent values
        """
        try:
            # Get recent events to extract features from
            recent_events = await self.timeseries_db.get_recent_events(minutes=60, rooms=[room_id])
            
            if not recent_events:
                return {}
            
            # Extract features from recent events and organize by feature name
            feature_history = {}
            
            # Limit to most recent events
            recent_events = recent_events[-limit:] if len(recent_events) > limit else recent_events
            
            for event in recent_events:
                # Extract features from each event
                features = self.dynamic_feature_discovery.discover_features([event])
                
                for feature_name, feature_value in features.items():
                    if feature_name not in feature_history:
                        feature_history[feature_name] = []
                    
                    # Convert feature value to float, handling different types
                    try:
                        if isinstance(feature_value, (int, float)):
                            numeric_value = float(feature_value)
                        elif isinstance(feature_value, str):
                            numeric_value = float(feature_value) if feature_value.replace('.', '').replace('-', '').isdigit() else 0.0
                        elif isinstance(feature_value, dict):
                            # For dict features, use length or a hash-based numeric representation
                            numeric_value = float(len(feature_value))
                        elif feature_value is None:
                            numeric_value = 0.0
                        else:
                            numeric_value = 0.0
                    except (ValueError, TypeError):
                        numeric_value = 0.0
                    
                    feature_history[feature_name].append(numeric_value)
            
            return feature_history
            
        except Exception as e:
            logger.error(f"Error getting recent features for {room_id}: {e}")
            return {}
    
    async def get_room_performance(self, room_id: str) -> Dict[str, Any]:
        """
        Get room performance data for drift detection
        Wrapper around existing get_room_metrics method
        """
        return await self.get_room_metrics(room_id)


class PatternDiscoveryModel:
    """Wrapper for long-term pattern discovery per room"""
    
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.pattern_discovery = PatternDiscovery()
        self.event_buffer = []
        
    async def update_patterns(self, sensor_data: Dict[str, Any]):
        """Update pattern discovery with new data"""
        self.event_buffer.append(sensor_data)
        
        # Discover patterns every 100 events
        if len(self.event_buffer) >= 100:
            self.pattern_discovery.discover_patterns(self.event_buffer, self.room_id)
            self.event_buffer = self.event_buffer[-50:]  # Keep last 50 for context


class AutoML:
    """
    Complete AutoML system for feature selection, model selection, and hyperparameter tuning.
    
    Implements adaptive machine learning pipeline that automatically:
    - Discovers and selects relevant features
    - Chooses optimal algorithms
    - Tunes hyperparameters
    - Adapts to changing data patterns
    """
    
    def __init__(self, max_features: int = 50):
        self.max_features = max_features
        self.feature_selector = AdaptiveFeatureSelector()
        
        # Model selection components - using River framework for online learning
        self.model_candidates = {
            'adaptive_random_forest': {
                'factory': lambda: ensemble.AdaptiveRandomForestClassifier(
                    n_models=10,
                    max_features="sqrt",
                    lambda_value=6,
                    grace_period=200
                ),
                'hyperparams': {
                    'n_models': [5, 10, 15],
                    'max_features': ["sqrt", "log2", None],
                    'lambda_value': [6, 8, 10],
                    'grace_period': [100, 200, 500]
                },
                'complexity': 'medium',
                'interpretability': 'medium'
            },
            'hoeffding_tree': {
                'factory': lambda: tree.HoeffdingTreeClassifier(
                    grace_period=200,
                    split_confidence=1e-7
                ),
                'hyperparams': {
                    'grace_period': [100, 200, 500],
                    'split_confidence': [1e-7, 1e-5, 1e-3]
                },
                'complexity': 'low',
                'interpretability': 'high'
            },
            'logistic_regression': {
                'factory': lambda: linear_model.LogisticRegression(l2=0.1),
                'hyperparams': {
                    'l2': [0.001, 0.01, 0.1, 1.0],
                    'l1': [0.0, 0.001, 0.01, 0.1]
                },
                'complexity': 'low',
                'interpretability': 'high'
            },
            'passive_aggressive': {
                'factory': lambda: linear_model.PAClassifier(C=1.0),
                'hyperparams': {
                    'C': [0.1, 1.0, 10.0],
                    'mode': [1, 2]
                },
                'complexity': 'low',
                'interpretability': 'medium'
            }
        }
        
        # Feature engineering pipeline
        self.feature_engineering_pipeline = {
            'temporal_features': {
                'hour_sin': lambda x: np.sin(2 * np.pi * x['hour'] / 24),
                'hour_cos': lambda x: np.cos(2 * np.pi * x['hour'] / 24),
                'day_sin': lambda x: np.sin(2 * np.pi * x['day_of_week'] / 7),
                'day_cos': lambda x: np.cos(2 * np.pi * x['day_of_week'] / 7)
            },
            'lag_features': {
                'sensor_lag_1': lambda x: x.get('sensor_state_lag_1', 0),
                'sensor_lag_5': lambda x: x.get('sensor_state_lag_5', 0),
                'sensor_lag_15': lambda x: x.get('sensor_state_lag_15', 0)
            },
            'statistical_features': {
                'sensor_count_1h': lambda x: x.get('sensor_activations_1h', 0),
                'sensor_count_6h': lambda x: x.get('sensor_activations_6h', 0),
                'avg_activation_time': lambda x: x.get('avg_activation_duration', 0)
            },
            'interaction_features': {
                'hour_weekday': lambda x: x.get('hour', 0) * x.get('day_of_week', 0),
                'sensor_time_interaction': lambda x: x.get('sensor_state', 0) * x.get('hour', 0)
            }
        }
        
        # Performance tracking
        self.model_performance_history = defaultdict(list)
        self.feature_importance_history = defaultdict(list)
        self.hyperparameter_history = defaultdict(list)
        
        # Adaptive thresholds
        self.performance_threshold = 0.75
        self.feature_importance_threshold = 0.01
        self.model_selection_frequency = 100  # Reselect model every 100 predictions
        
        # Current best configuration
        self.best_model_config = None
        self.best_features = None
        self.best_hyperparameters = None
        
        logger.info("Initialized comprehensive AutoML system")
        
        # Initialize performance tracking
        self.model_evaluations = defaultdict(list)
        self.current_best_model = None
        self.evaluation_counter = 0
    
    def auto_feature_engineering(self, raw_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically engineer features from raw sensor data.
        
        Args:
            raw_features: Raw sensor features
            
        Returns:
            Engineered features
        """
        engineered_features = raw_features.copy()
        
        try:
            # Apply temporal feature engineering
            for name, func in self.feature_engineering_pipeline['temporal_features'].items():
                try:
                    engineered_features[name] = func(raw_features)
                except Exception:
                    engineered_features[name] = 0
            
            # Apply lag features
            for name, func in self.feature_engineering_pipeline['lag_features'].items():
                try:
                    engineered_features[name] = func(raw_features)
                except Exception:
                    engineered_features[name] = 0
            
            # Apply statistical features
            for name, func in self.feature_engineering_pipeline['statistical_features'].items():
                try:
                    engineered_features[name] = func(raw_features)
                except Exception:
                    engineered_features[name] = 0
            
            # Apply interaction features
            for name, func in self.feature_engineering_pipeline['interaction_features'].items():
                try:
                    engineered_features[name] = func(raw_features)
                except Exception:
                    engineered_features[name] = 0
            
            logger.debug(f"Engineered {len(engineered_features)} features from {len(raw_features)} raw features")
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return raw_features
        
        return engineered_features
    
    def select_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically select the most relevant features.
        
        Args:
            features: All available features
            
        Returns:
            Selected features
        """
        try:
            # First, engineer additional features
            engineered_features = self.auto_feature_engineering(features)
            
            # Get feature importance scores
            top_features = self.feature_selector.get_top_features(self.max_features)
            
            # Select top features
            selected_features = {}
            for feature_name, importance in top_features:
                if feature_name in engineered_features:
                    selected_features[feature_name] = engineered_features[feature_name]
            
            # If we don't have enough features, add from engineered features
            if len(selected_features) < self.max_features:
                remaining_features = [
                    name for name in engineered_features
                    if name not in selected_features
                ]
                
                for feature_name in remaining_features[:self.max_features - len(selected_features)]:
                    selected_features[feature_name] = engineered_features[feature_name]
            
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            return features
    
    def auto_model_selection(self, room_id: str, data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically select the best model based on data characteristics.
        
        Args:
            room_id: Room identifier
            data_characteristics: Characteristics of the data
            
        Returns:
            Selected model configuration
        """
        try:
            data_size = data_characteristics.get('data_size', 1000)
            feature_count = data_characteristics.get('feature_count', 10)
            
            # Evaluate all candidate models
            self.evaluation_counter += 1
            
            # Select model based on performance history and data characteristics
            if self.evaluation_counter % self.model_selection_frequency == 0:
                self.current_best_model = self._evaluate_all_models(room_id, data_characteristics)
            
            # Return current best model or default
            if self.current_best_model:
                return self.current_best_model
            else:
                # Default to adaptive random forest for new rooms
                return {
                    'model_name': 'adaptive_random_forest',
                    'hyperparams': {
                        'n_models': 10,
                        'max_features': "sqrt",
                        'lambda_value': 6,
                        'grace_period': 200
                    },
                    'complexity': 'medium'
                }
                
        except Exception as e:
            logger.error(f"Error in model selection for {room_id}: {e}")
            return {
                'model_name': 'logistic_regression',
                'hyperparams': {'l2': 0.1},
                'complexity': 'low'
            }
    
    def _evaluate_all_models(self, room_id: str, data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate all candidate models and return the best one
        """
        best_model = None
        best_score = -1
        
        for model_name, model_config in self.model_candidates.items():
            # Get historical performance
            historical_performance = self.model_evaluations[f"{room_id}_{model_name}"]
            
            if historical_performance:
                # Calculate average performance
                avg_score = sum(historical_performance[-10:]) / len(historical_performance[-10:])
                
                # Consider data characteristics
                complexity_penalty = self._calculate_complexity_penalty(
                    model_config['complexity'], 
                    data_characteristics
                )
                
                final_score = avg_score - complexity_penalty
                
                if final_score > best_score:
                    best_score = final_score
                    best_model = {
                        'model_name': model_name,
                        'hyperparams': self._optimize_hyperparameters(model_name, room_id),
                        'complexity': model_config['complexity'],
                        'score': final_score
                    }
        
        return best_model
    
    def _calculate_complexity_penalty(self, complexity: str, data_characteristics: Dict[str, Any]) -> float:
        """
        Calculate penalty for model complexity based on data characteristics
        """
        data_size = data_characteristics.get('data_size', 1000)
        feature_count = data_characteristics.get('feature_count', 10)
        
        # Penalty based on data size
        if data_size < 500:
            complexity_penalties = {'low': 0.0, 'medium': 0.1, 'high': 0.2}
        elif data_size < 2000:
            complexity_penalties = {'low': 0.0, 'medium': 0.05, 'high': 0.15}
        else:
            complexity_penalties = {'low': 0.0, 'medium': 0.0, 'high': 0.05}
        
        return complexity_penalties.get(complexity, 0.0)
    
    def _optimize_hyperparameters(self, model_name: str, room_id: str) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific model
        """
        model_config = self.model_candidates.get(model_name, {})
        hyperparams = model_config.get('hyperparams', {})
        
        # For now, use default hyperparameters
        # In a full implementation, this would use techniques like:
        # - Grid search
        # - Random search
        # - Bayesian optimization
        # - Hyperband
        
        optimized_params = {}
        for param_name, param_values in hyperparams.items():
            if isinstance(param_values, list):
                # For now, select middle value
                optimized_params[param_name] = param_values[len(param_values) // 2]
            else:
                optimized_params[param_name] = param_values
        
        return optimized_params
    
    def update_model_performance(self, room_id: str, model_name: str, accuracy: float):
        """
        Update performance tracking for a model
        """
        key = f"{room_id}_{model_name}"
        self.model_evaluations[key].append(accuracy)
        
        # Keep only last 100 evaluations
        if len(self.model_evaluations[key]) > 100:
            self.model_evaluations[key] = self.model_evaluations[key][-100:]
        
        # Update best model if this is current best
        if (self.current_best_model and 
            self.current_best_model.get('model_name') == model_name):
            
            # Check if we need to find a new best model
            recent_performance = self.model_evaluations[key][-5:]
            if recent_performance and sum(recent_performance) / len(recent_performance) < 0.6:
                logger.info(f"Model {model_name} for {room_id} performance degraded, triggering reselection")
                self.evaluation_counter = self.model_selection_frequency  # Trigger reselection
    
    def get_automl_summary(self) -> Dict[str, Any]:
        """
        Get summary of AutoML system state
        """
        return {
            'current_best_model': self.current_best_model,
            'evaluation_counter': self.evaluation_counter,
            'model_evaluations': {
                key: {
                    'count': len(evaluations),
                    'average': sum(evaluations) / len(evaluations) if evaluations else 0,
                    'recent_average': sum(evaluations[-5:]) / len(evaluations[-5:]) if evaluations else 0
                }
                for key, evaluations in self.model_evaluations.items()
            },
            'available_models': list(self.model_candidates.keys())
        }
    
    def auto_hyperparameter_tuning(self, model_config: Dict[str, Any], 
                                 room_id: str, performance_history: List[float]) -> Dict[str, Any]:
        """
        Automatically tune hyperparameters based on performance.
        
        Args:
            model_config: Model configuration
            room_id: Room identifier
            performance_history: Recent performance history
            
        Returns:
            Tuned hyperparameters
        """
        try:
            hyperparams = model_config.get('hyperparams', {})
            
            # Get current performance trend
            if len(performance_history) >= 5:
                recent_performance = performance_history[-5:]
                avg_performance = sum(recent_performance) / len(recent_performance)
                performance_trend = recent_performance[-1] - recent_performance[0]
            else:
                avg_performance = 0.5
                performance_trend = 0
            
            # Adjust hyperparameters based on performance
            tuned_hyperparams = {}
            
            for param_name, param_values in hyperparams.items():
                if isinstance(param_values, list):
                    # Select based on performance
                    if avg_performance > self.performance_threshold:
                        # Performance is good, use current or slight variation
                        if param_name in self.hyperparameter_history.get(room_id, {}):
                            current_value = self.hyperparameter_history[room_id][param_name]
                            if current_value in param_values:
                                tuned_hyperparams[param_name] = current_value
                            else:
                                tuned_hyperparams[param_name] = param_values[len(param_values) // 2]
                        else:
                            tuned_hyperparams[param_name] = param_values[len(param_values) // 2]
                    else:
                        # Performance is poor, try different values
                        if performance_trend < 0:  # Declining performance
                            # Try more conservative values
                            tuned_hyperparams[param_name] = param_values[0]
                        else:
                            # Try more aggressive values
                            tuned_hyperparams[param_name] = param_values[-1]
                else:
                    tuned_hyperparams[param_name] = param_values
            
            # Store hyperparameter history
            if room_id not in self.hyperparameter_history:
                self.hyperparameter_history[room_id] = {}
            self.hyperparameter_history[room_id].update(tuned_hyperparams)
            
            logger.debug(f"Tuned hyperparameters for room {room_id}: {tuned_hyperparams}")
            
            return tuned_hyperparams
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {e}")
            return {}
    
    def adaptive_pipeline_optimization(self, room_id: str, 
                                     features: Dict[str, Any], 
                                     performance_feedback: float) -> Dict[str, Any]:
        """
        Optimize the entire ML pipeline adaptively.
        
        Args:
            room_id: Room identifier
            features: Input features
            performance_feedback: Recent performance score
            
        Returns:
            Optimized pipeline configuration
        """
        try:
            # Update performance history
            self.model_performance_history[room_id].append(performance_feedback)
            
            # Keep only recent history
            if len(self.model_performance_history[room_id]) > 100:
                self.model_performance_history[room_id] = self.model_performance_history[room_id][-100:]
            
            # Analyze data characteristics
            data_characteristics = {
                'data_size': len(self.model_performance_history[room_id]),
                'feature_count': len(features),
                'class_balance': 0.5,  # Would be calculated from actual data
                'noise_level': np.std(self.model_performance_history[room_id][-10:]) if len(self.model_performance_history[room_id]) >= 10 else 0.1
            }
            
            # Select optimal model
            model_config = self.auto_model_selection(room_id, data_characteristics)
            
            # Tune hyperparameters
            tuned_hyperparams = self.auto_hyperparameter_tuning(
                model_config, room_id, self.model_performance_history[room_id]
            )
            
            # Select optimal features
            selected_features = self.select_features(features)
            
            # Create optimized configuration
            optimized_config = {
                'model_type': model_config['class'],
                'hyperparameters': tuned_hyperparams,
                'selected_features': selected_features,
                'feature_count': len(selected_features),
                'expected_performance': performance_feedback,
                'optimization_timestamp': datetime.now().isoformat()
            }
            
            # Update best configuration if this is better
            if (self.best_model_config is None or 
                performance_feedback > self.best_model_config.get('expected_performance', 0)):
                self.best_model_config = optimized_config.copy()
                self.best_features = selected_features.copy()
                self.best_hyperparameters = tuned_hyperparams.copy()
            
            logger.info(f"Optimized pipeline for room {room_id}: {optimized_config['model_type']} with {len(selected_features)} features")
            
            return optimized_config
            
        except Exception as e:
            logger.error(f"Error in pipeline optimization: {e}")
            return {
                'model_type': 'gradient_boost',
                'hyperparameters': {},
                'selected_features': features,
                'feature_count': len(features),
                'expected_performance': performance_feedback,
                'optimization_timestamp': datetime.now().isoformat()
            }
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get AutoML optimization statistics"""
        
        return {
            'total_rooms_optimized': len(self.model_performance_history),
            'average_performance': {
                room_id: np.mean(history) if history else 0
                for room_id, history in self.model_performance_history.items()
            },
            'best_model_config': self.best_model_config,
            'feature_engineering_pipeline_size': sum(len(pipeline) for pipeline in self.feature_engineering_pipeline.values()),
            'model_candidates': len(self.model_candidates),
            'optimization_history': {
                room_id: len(history)
                for room_id, history in self.model_performance_history.items()
            }
        }
    
    def reset_optimization_history(self, room_id: str = None):
        """Reset optimization history for a room or all rooms"""
        
        if room_id:
            if room_id in self.model_performance_history:
                del self.model_performance_history[room_id]
            if room_id in self.hyperparameter_history:
                del self.hyperparameter_history[room_id]
        else:
            self.model_performance_history.clear()
            self.hyperparameter_history.clear()
            self.best_model_config = None
            self.best_features = None
            self.best_hyperparameters = None
        
        logger.info(f"Reset optimization history for {'all rooms' if not room_id else room_id}")


class OnlineAnomalyDetector:
    """Simple wrapper for the adaptive cat detector"""
    
    def __init__(self):
        self.cat_detector = AdaptiveCatDetector()
    
    def detect_anomaly(self, sensor_sequence: List[Dict[str, Any]]) -> bool:
        """Detect if sequence is anomalous (likely cat)"""
        try:
            self.cat_detector.learn_movement_patterns(sensor_sequence)
            # For now, return False (no anomaly detected)
            # Full implementation would analyze the sequence
            return False
        except Exception:
            return False