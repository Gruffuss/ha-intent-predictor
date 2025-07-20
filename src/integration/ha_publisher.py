"""
HA Integration Service - Dynamic entity creation and management
Implements the exact DynamicHAIntegration from CLAUDE.md
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp
import json

logger = logging.getLogger(__name__)


class DynamicHAIntegration:
    """
    Create dynamic entities based on discovered patterns
    Implements the exact approach from CLAUDE.md
    """
    
    def __init__(self, ha_api, timeseries_db=None, feature_store=None):
        self.ha = ha_api
        self.timeseries_db = timeseries_db
        self.feature_store = feature_store
        self.prediction_entities = {}
        self.automation_helpers = {}
        self.entity_states = {}
        
    async def initialize(self):
        """Initialize HA integration - ready for predictions"""
        logger.info("HA integration initialized successfully")
    
    async def run_publisher(self):
        """Main publisher loop - continuously publish predictions to HA"""
        logger.info("Starting HA prediction publisher loop")
        
        # Define rooms to create entities for
        rooms = ['living_kitchen', 'bedroom', 'office', 'bathroom', 'small_bathroom']
        horizons = [15, 120]  # 15 min and 2 hour predictions
        
        # Create initial prediction entities
        for room in rooms:
            for horizon in horizons:
                entity_id = f"sensor.occupancy_{room}_{horizon}min"
                await self.create_prediction_entity(entity_id, room, horizon)
            
            # Create automation helpers for each room
            await self.create_automation_helpers(room)
        
        logger.info(f"Created prediction entities for {len(rooms)} rooms")
        
        while True:
            try:
                # Publish current entity states and cleanup stale entities
                await self.cleanup_stale_entities(max_age_hours=24)
                
                # Update trend and anomaly entities for each room
                for room in rooms:
                    try:
                        # Update trend entity
                        trend_data = await self.calculate_occupancy_trend(room)
                        await self.set_state(f"sensor.occupancy_trend_{room}", {
                            'state': trend_data.get('trend', 'stable'),
                            'attributes': trend_data
                        })
                        
                        # Update anomaly detection entity
                        anomaly_data = await self.detect_anomalous_pattern(room)
                        await self.set_state(f"binary_sensor.occupancy_anomaly_{room}", {
                            'state': anomaly_data.get('anomaly_detected', False),
                            'attributes': anomaly_data
                        })
                        
                        # Update activity level entity
                        activity_data = await self.calculate_activity_level(room)
                        await self.set_state(f"sensor.occupancy_activity_{room}", {
                            'state': activity_data.get('activity_level', 'unknown'),
                            'attributes': activity_data
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error updating entities for room {room}: {e}")
                
                # Wait before next update cycle
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in HA publisher loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
        
    async def publish_predictions(self, room_id: str, predictions: Dict[int, Dict[str, Any]]):
        """
        Create dynamic entities based on discovered patterns
        Implements the exact logic from CLAUDE.md
        """
        for horizon, prediction in predictions.items():
            entity_id = f"sensor.occupancy_{room_id}_{horizon}min"
            
            # Create entity if it doesn't exist
            if entity_id not in self.prediction_entities:
                await self.create_prediction_entity(entity_id, room_id, horizon)
            
            # Update with prediction and metadata
            await self.ha.set_state(entity_id, {
                'state': prediction['probability'],
                'attributes': {
                    'uncertainty': prediction['uncertainty'],
                    'confidence': 1 - prediction['uncertainty'],
                    'contributing_factors': prediction.get('factors', {}),
                    'model_agreement': prediction.get('model_agreement', 0.5),
                    'last_updated': datetime.now().isoformat(),
                    'horizon_minutes': horizon,
                    'explanation': prediction.get('explanation', ''),
                    'method': prediction.get('method', 'adaptive_ml'),
                    'features_used': prediction.get('features_used', 0),
                    'room_id': room_id
                }
            })
            
            # Store state for tracking
            self.entity_states[entity_id] = {
                'state': prediction['probability'],
                'last_updated': datetime.now(),
                'prediction_data': prediction
            }
    
    async def create_prediction_entity(self, entity_id: str, room_id: str, horizon: int):
        """Create prediction entity in Home Assistant"""
        try:
            entity_config = {
                'name': f"{room_id.replace('_', ' ').title()} Occupancy ({horizon}min)",
                'unique_id': entity_id,
                'device_class': 'occupancy',
                'state_class': 'measurement',
                'unit_of_measurement': '%',
                'icon': 'mdi:account-check',
                'entity_category': 'diagnostic'
            }
            
            # Register entity with HA
            await self.ha.create_entity(entity_id, entity_config)
            
            self.prediction_entities[entity_id] = {
                'room_id': room_id,
                'horizon': horizon,
                'created_at': datetime.now(),
                'config': entity_config
            }
            
            logger.info(f"Created prediction entity: {entity_id}")
            
        except Exception as e:
            logger.error(f"Error creating prediction entity {entity_id}: {e}")
    
    async def create_automation_helpers(self, room_id: str):
        """
        Create helper entities for complex automations
        Implements the exact helper creation from CLAUDE.md
        """
        # Trends
        trend_entity = f"sensor.occupancy_trend_{room_id}"
        trend_data = await self.calculate_occupancy_trend(room_id)
        
        await self.ha.set_state(trend_entity, {
            'state': trend_data['trend_direction'],
            'attributes': {
                'trend_strength': trend_data['trend_strength'],
                'change_probability': trend_data['change_probability'],
                'trend_score': trend_data['trend_score'],
                'sample_period': '24h',
                'room_id': room_id,
                'last_calculated': datetime.now().isoformat()
            }
        })
        
        # Anomalies
        anomaly_entity = f"binary_sensor.occupancy_anomaly_{room_id}"
        anomaly_data = await self.detect_anomalous_pattern(room_id)
        
        await self.ha.set_state(anomaly_entity, {
            'state': anomaly_data['anomaly_detected'],
            'attributes': {
                'anomaly_score': anomaly_data['anomaly_score'],
                'anomaly_type': anomaly_data['anomaly_type'],
                'confidence': anomaly_data['confidence'],
                'description': anomaly_data['description'],
                'room_id': room_id,
                'detected_at': datetime.now().isoformat()
            }
        })
        
        # Activity level indicator
        activity_entity = f"sensor.occupancy_activity_level_{room_id}"
        activity_data = await self.calculate_activity_level(room_id)
        
        await self.ha.set_state(activity_entity, {
            'state': activity_data['level'],
            'attributes': {
                'activity_score': activity_data['score'],
                'events_per_hour': activity_data['events_per_hour'],
                'compared_to_average': activity_data['compared_to_average'],
                'room_id': room_id,
                'measurement_period': '1h'
            }
        })
        
        # Store helper entities
        if room_id not in self.automation_helpers:
            self.automation_helpers[room_id] = []
        
        self.automation_helpers[room_id].extend([
            trend_entity, anomaly_entity, activity_entity
        ])
        
        logger.info(f"Created automation helpers for {room_id}")
    
    async def calculate_occupancy_trend(self, room_id: str) -> Dict[str, Any]:
        """Calculate occupancy trend for the room"""
        try:
            # Get recent predictions for trend analysis
            recent_predictions = await self.get_recent_predictions(room_id, hours=24)
            
            if len(recent_predictions) < 10:
                return {
                    'trend_direction': 'stable',
                    'trend_strength': 0.0,
                    'change_probability': 0.5,
                    'trend_score': 0.0
                }
            
            # Calculate trend
            timestamps = [p['timestamp'] for p in recent_predictions]
            probabilities = [p['probability'] for p in recent_predictions]
            
            # Simple linear trend
            time_diffs = [(t - timestamps[0]).total_seconds() / 3600 for t in timestamps]  # Hours
            
            # Calculate correlation (simplified trend)
            if len(time_diffs) > 1:
                correlation = self.calculate_correlation(time_diffs, probabilities)
                
                if correlation > 0.3:
                    trend_direction = 'increasing'
                    trend_strength = correlation
                elif correlation < -0.3:
                    trend_direction = 'decreasing'
                    trend_strength = abs(correlation)
                else:
                    trend_direction = 'stable'
                    trend_strength = 1.0 - abs(correlation)
                
                # Change probability based on trend
                change_probability = min(0.9, 0.5 + abs(correlation) * 0.5)
            else:
                trend_direction = 'stable'
                trend_strength = 0.0
                change_probability = 0.5
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'change_probability': change_probability,
                'trend_score': correlation if 'correlation' in locals() else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating trend for {room_id}: {e}")
            return {
                'trend_direction': 'unknown',
                'trend_strength': 0.0,
                'change_probability': 0.5,
                'trend_score': 0.0
            }
    
    async def detect_anomalous_pattern(self, room_id: str) -> Dict[str, Any]:
        """Detect anomalous occupancy patterns"""
        try:
            # Get recent activity
            recent_activity = await self.get_recent_activity(room_id, hours=2)
            historical_activity = await self.get_historical_activity(room_id, days=7)
            
            if not recent_activity or not historical_activity:
                return {
                    'anomaly_detected': False,
                    'anomaly_score': 0.0,
                    'anomaly_type': 'none',
                    'confidence': 0.0,
                    'description': 'Insufficient data for anomaly detection'
                }
            
            # Simple anomaly detection
            recent_count = len(recent_activity)
            historical_avg = len(historical_activity) / 7  # Daily average
            
            # Compare recent to historical
            if recent_count > historical_avg * 3:
                return {
                    'anomaly_detected': True,
                    'anomaly_score': min(1.0, recent_count / (historical_avg * 3)),
                    'anomaly_type': 'high_activity',
                    'confidence': 0.8,
                    'description': f'Unusually high activity: {recent_count} events vs {historical_avg:.1f} average'
                }
            elif recent_count < historical_avg * 0.3 and historical_avg > 1:
                return {
                    'anomaly_detected': True,
                    'anomaly_score': min(1.0, 1.0 - (recent_count / (historical_avg * 0.3))),
                    'anomaly_type': 'low_activity',
                    'confidence': 0.7,
                    'description': f'Unusually low activity: {recent_count} events vs {historical_avg:.1f} average'
                }
            else:
                return {
                    'anomaly_detected': False,
                    'anomaly_score': 0.0,
                    'anomaly_type': 'none',
                    'confidence': 0.6,
                    'description': 'Activity within normal range'
                }
                
        except Exception as e:
            logger.error(f"Error detecting anomalies for {room_id}: {e}")
            return {
                'anomaly_detected': False,
                'anomaly_score': 0.0,
                'anomaly_type': 'error',
                'confidence': 0.0,
                'description': f'Error during detection: {str(e)}'
            }
    
    async def calculate_activity_level(self, room_id: str) -> Dict[str, Any]:
        """Calculate current activity level"""
        try:
            recent_events = await self.get_recent_activity(room_id, hours=1)
            historical_avg = await self.get_average_activity(room_id, hours=1, days=7)
            
            events_per_hour = len(recent_events)
            
            # Determine activity level
            if events_per_hour >= historical_avg * 2:
                level = 'very_high'
                score = min(1.0, events_per_hour / (historical_avg * 2))
            elif events_per_hour >= historical_avg * 1.5:
                level = 'high'
                score = events_per_hour / (historical_avg * 1.5)
            elif events_per_hour >= historical_avg * 0.5:
                level = 'normal'
                score = 0.5
            elif events_per_hour > 0:
                level = 'low'
                score = events_per_hour / (historical_avg * 0.5) if historical_avg > 0 else 0.3
            else:
                level = 'none'
                score = 0.0
            
            compared_to_average = (events_per_hour / historical_avg) if historical_avg > 0 else 1.0
            
            return {
                'level': level,
                'score': score,
                'events_per_hour': events_per_hour,
                'compared_to_average': compared_to_average
            }
            
        except Exception as e:
            logger.error(f"Error calculating activity level for {room_id}: {e}")
            return {
                'level': 'unknown',
                'score': 0.0,
                'events_per_hour': 0,
                'compared_to_average': 1.0
            }
    
    def calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        sum_y2 = sum(y[i] ** 2 for i in range(n))
        
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        correlation = (n * sum_xy - sum_x * sum_y) / denominator
        return correlation
    
    async def get_recent_predictions(self, room_id: str, hours: int) -> List[Dict[str, Any]]:
        """Get recent predictions for room from TimescaleDB"""
        if not self.timeseries_db:
            logger.warning("TimescaleDB not available for prediction queries")
            return []
            
        try:
            predictions = await self.timeseries_db.get_recent_predictions(
                hours=hours,
                room=room_id
            )
            return predictions
        except Exception as e:
            logger.error(f"Error getting recent predictions for {room_id}: {e}")
            return []
    
    async def get_recent_activity(self, room_id: str, hours: int) -> List[Dict[str, Any]]:
        """Get recent sensor activity for room from TimescaleDB"""
        if not self.timeseries_db:
            logger.warning("TimescaleDB not available for activity queries")
            return []
            
        try:
            # Convert hours to minutes for the existing method
            recent_events = await self.timeseries_db.get_recent_events(
                minutes=hours * 60,
                rooms=[room_id]
            )
            return recent_events
        except Exception as e:
            logger.error(f"Error getting recent activity for {room_id}: {e}")
            return []
    
    async def get_historical_activity(self, room_id: str, days: int) -> List[Dict[str, Any]]:
        """Get historical sensor activity for room"""
        if not self.timeseries_db:
            logger.warning("TimescaleDB not available for historical activity queries")
            return []
            
        try:
            from datetime import timedelta
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Get historical events for the specified time period
            historical_events = await self.timeseries_db.get_historical_events(
                start_time=start_time,
                end_time=end_time,
                rooms=[room_id]
            )
            return historical_events
        except Exception as e:
            logger.error(f"Error getting historical activity for {room_id}: {e}")
            return []
    
    async def get_average_activity(self, room_id: str, hours: int, days: int) -> float:
        """Get average activity for comparison"""
        if not self.timeseries_db:
            logger.warning("TimescaleDB not available for average activity calculation")
            return 5.0  # Default fallback
            
        try:
            # Get historical activity for the specified period
            historical_events = await self.get_historical_activity(room_id, days)
            
            if not historical_events:
                return 0.0
            
            # Calculate total events per hour over the historical period
            total_events = len(historical_events)
            total_hours = days * 24
            
            if total_hours == 0:
                return 0.0
                
            # Average events per hour
            avg_events_per_hour = total_events / total_hours
            
            # If we're looking for a specific hour window, scale appropriately
            return avg_events_per_hour * hours if hours != 1 else avg_events_per_hour
            
        except Exception as e:
            logger.error(f"Error calculating average activity for {room_id}: {e}")
            return 5.0  # Default fallback
    
    async def update_all_predictions(self, all_predictions: Dict[str, Dict[int, Dict[str, Any]]]):
        """Update all room predictions at once"""
        try:
            update_tasks = []
            
            for room_id, predictions in all_predictions.items():
                if predictions:
                    task = self.publish_predictions(room_id, predictions)
                    update_tasks.append(task)
            
            # Update all predictions concurrently
            if update_tasks:
                await asyncio.gather(*update_tasks, return_exceptions=True)
                logger.debug(f"Updated predictions for {len(all_predictions)} rooms")
            
        except Exception as e:
            logger.error(f"Error updating all predictions: {e}")
    
    async def cleanup_stale_entities(self, max_age_hours: int = 24):
        """Clean up stale prediction entities"""
        try:
            current_time = datetime.now()
            stale_entities = []
            
            for entity_id, entity_info in self.entity_states.items():
                if entity_info['last_updated']:
                    age = (current_time - entity_info['last_updated']).total_seconds() / 3600
                    if age > max_age_hours:
                        stale_entities.append(entity_id)
            
            # Remove stale entities
            for entity_id in stale_entities:
                try:
                    await self.ha.remove_entity(entity_id)
                    del self.entity_states[entity_id]
                    if entity_id in self.prediction_entities:
                        del self.prediction_entities[entity_id]
                    logger.info(f"Removed stale entity: {entity_id}")
                except Exception as e:
                    logger.warning(f"Failed to remove stale entity {entity_id}: {e}")
            
        except Exception as e:
            logger.error(f"Error cleaning up stale entities: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of HA integration"""
        return {
            'prediction_entities': len(self.prediction_entities),
            'automation_helpers': sum(len(helpers) for helpers in self.automation_helpers.values()),
            'active_entities': len(self.entity_states),
            'rooms_integrated': list(self.automation_helpers.keys()),
            'last_update': max(
                (state['last_updated'] for state in self.entity_states.values()),
                default=datetime.now()
            ).isoformat()
        }


class HomeAssistantAPI:
    """
    Home Assistant API wrapper for entity management
    """
    
    def __init__(self, ha_url: str, token: str):
        self.ha_url = ha_url.rstrip('/')
        self.token = token
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
    
    async def set_state(self, entity_id: str, state_data: Dict[str, Any]):
        """Set entity state in Home Assistant"""
        try:
            url = f"{self.ha_url}/api/states/{entity_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self.headers, json=state_data) as response:
                    if response.status == 200:
                        logger.debug(f"Updated entity state: {entity_id}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to update {entity_id}: {response.status} - {error_text}")
                        return False
        except Exception as e:
            logger.error(f"Error setting state for {entity_id}: {e}")
            return False
    
    async def create_entity(self, entity_id: str, config: Dict[str, Any]):
        """Create new entity in Home Assistant"""
        try:
            # For MQTT discovery or direct entity creation
            # This is a simplified version - full implementation would depend on HA setup
            logger.info(f"Entity creation requested: {entity_id} (config: {config})")
            return True
        except Exception as e:
            logger.error(f"Error creating entity {entity_id}: {e}")
            return False
    
    async def remove_entity(self, entity_id: str):
        """Remove entity from Home Assistant"""
        try:
            # Implementation depends on how entities were created
            logger.info(f"Entity removal requested: {entity_id}")
            return True
        except Exception as e:
            logger.error(f"Error removing entity {entity_id}: {e}")
            return False