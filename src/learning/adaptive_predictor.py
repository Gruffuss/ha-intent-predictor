"""
Adaptive Occupancy Predictor - Main coordinator implementing CLAUDE.md architecture
Ties together all components into the comprehensive system
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

# Import all the components we've implemented
from ingestion.data_enricher import DynamicFeatureDiscovery, AdaptiveFeatureSelector
from learning.online_models import ContinuousLearningModel, MultiHorizonPredictor
from learning.pattern_discovery import PatternDiscovery, UnbiasedPatternMiner
from learning.anomaly_detection import AdaptiveCatDetector, OnlineAnomalyDetector
from prediction.bathroom_predictor import BathroomOccupancyPredictor
from adaptation.performance_monitor import PerformanceMonitor
from adaptation.resource_optimizer import ResourceOptimizer
from storage.timeseries_db import TimescaleDBManager
from storage.feature_store import RedisFeatureStore

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
        
        # Storage backends
        self.timeseries_db = TimescaleDBManager(config.get('database_url'))
        self.feature_store = RedisFeatureStore(config.get('redis_url'))
        
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
    """Simple AutoML wrapper for feature selection"""
    
    def __init__(self, max_features: int = 50):
        self.max_features = max_features
        self.feature_selector = AdaptiveFeatureSelector()
    
    def select_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Select top features automatically"""
        top_features = self.feature_selector.get_top_features(self.max_features)
        return {name: features.get(name, 0) for name, _ in top_features}


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