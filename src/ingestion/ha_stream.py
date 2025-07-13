"""
Continuous HA Data Streaming - Aggressive sensor data collection
Implements the detailed streaming system from CLAUDE.md
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, AsyncGenerator
import aiohttp
import websocket
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EnrichedEvent:
    """Enriched sensor event with context"""
    timestamp: datetime
    entity_id: str
    state: str
    attributes: Dict[str, Any]
    room: str
    sensor_type: str
    derived: Dict[str, Any]


class HADataStream:
    """
    Aggressively pull all sensor data continuously
    Implements the exact streaming approach from CLAUDE.md
    """
    
    def __init__(self, ha_url: str, token: str):
        self.ha_url = ha_url.rstrip('/')
        self.token = token
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        # State tracking for enrichment
        self.last_state_changes = {}
        self.concurrent_events = {}
        
        # Event queue for processing
        self.event_queue = asyncio.Queue()
        
        # Zone configuration from CLAUDE.md
        self.zone_config = self._initialize_zone_config()
        
        logger.info("Initialized HA data stream")
    
    def _initialize_zone_config(self) -> Dict[str, Dict]:
        """
        Initialize zone configuration exactly as specified in CLAUDE.md
        """
        return {
            'bedroom': {
                'full_zone': 'binary_sensor.bedroom_presence_sensor_full_bedroom',
                'subzones': [
                    'binary_sensor.bedroom_presence_sensor_anca_bed_side',
                    'binary_sensor.bedroom_vladimir_bed_side', 
                    'binary_sensor.bedroom_floor',
                    'binary_sensor.bedroom_entrance'
                ]
            },
            'office': {
                'full_zone': 'binary_sensor.office_presence_full_office',
                'subzones': [
                    'binary_sensor.office_presence_anca_desk',
                    'binary_sensor.office_presence_vladimir_desk',
                    'binary_sensor.office_entrance'
                ]
            },
            'living_kitchen': {
                'full_zones': [
                    'binary_sensor.presence_livingroom_full',
                    'binary_sensor.kitchen_pressence_full_kitchen'
                ],
                'subzones': {
                    'livingroom': ['binary_sensor.presence_livingroom_couch'],
                    'kitchen': [
                        'binary_sensor.kitchen_pressence_stove',
                        'binary_sensor.kitchen_pressence_sink', 
                        'binary_sensor.kitchen_pressence_dining_table'
                    ]
                }
            }
        }
    
    async def stream_all_sensors(self) -> AsyncGenerator[EnrichedEvent, None]:
        """
        Aggressively pull all sensor data continuously
        Subscribe to ALL state changes as specified in CLAUDE.md
        """
        logger.info("Starting continuous sensor streaming")
        
        # Start WebSocket connection for real-time events
        websocket_task = asyncio.create_task(self._websocket_stream())
        
        # Start periodic polling as backup
        polling_task = asyncio.create_task(self._polling_stream())
        
        try:
            # Process events from queue
            while True:
                try:
                    # Get event from queue (blocks until available)
                    raw_event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                    
                    # Transform to standardized format
                    enriched_event = await self.enrich_event(raw_event)
                    
                    if enriched_event:
                        yield enriched_event
                        
                        # Immediate feature computation
                        await self.compute_streaming_features(enriched_event)
                    
                except asyncio.TimeoutError:
                    # No events in queue, continue
                    continue
                    
                except Exception as e:
                    logger.error(f"Error processing event: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in sensor streaming: {e}")
        finally:
            # Clean up tasks
            websocket_task.cancel()
            polling_task.cancel()
    
    async def _websocket_stream(self):
        """WebSocket connection for real-time events"""
        while True:
            try:
                # Connect to HA WebSocket API
                uri = f"ws://{self.ha_url.replace('http://', '').replace('https://', '')}/api/websocket"
                
                async with websocket.connect(uri) as ws:
                    # Authenticate
                    auth_msg = await ws.recv()
                    auth_data = json.loads(auth_msg)
                    
                    if auth_data['type'] == 'auth_required':
                        await ws.send(json.dumps({
                            'type': 'auth',
                            'access_token': self.token
                        }))
                        
                        auth_result = await ws.recv()
                        auth_result_data = json.loads(auth_result)
                        
                        if auth_result_data['type'] != 'auth_ok':
                            logger.error("WebSocket authentication failed")
                            return
                    
                    # Subscribe to state changes
                    await ws.send(json.dumps({
                        'id': 1,
                        'type': 'subscribe_events',
                        'event_type': 'state_changed'
                    }))
                    
                    logger.info("WebSocket connected and subscribed to state changes")
                    
                    # Process incoming events
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            
                            if data.get('type') == 'event' and data.get('event', {}).get('event_type') == 'state_changed':
                                event_data = data['event']['data']
                                
                                # Queue event for processing
                                await self.event_queue.put({
                                    'timestamp': datetime.now(),
                                    'entity_id': event_data['entity_id'],
                                    'state': event_data['new_state']['state'],
                                    'attributes': event_data['new_state']['attributes'],
                                    'source': 'websocket'
                                })
                                
                        except Exception as e:
                            logger.warning(f"Error processing WebSocket message: {e}")
                            continue
                            
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
                await asyncio.sleep(5)  # Wait before reconnecting
    
    async def _polling_stream(self):
        """Polling backup for WebSocket failures"""
        last_poll_time = datetime.now()
        
        while True:
            try:
                await asyncio.sleep(2)  # Poll every 2 seconds
                
                # Get all current states
                async with aiohttp.ClientSession() as session:
                    async with session.get(f'{self.ha_url}/api/states', headers=self.headers) as response:
                        if response.status == 200:
                            states = await response.json()
                            
                            current_time = datetime.now()
                            
                            # Check for state changes since last poll
                            for state in states:
                                entity_id = state['entity_id']
                                last_changed = datetime.fromisoformat(state['last_changed'].replace('Z', '+00:00'))
                                
                                # Only process if changed since last poll
                                if last_changed > last_poll_time:
                                    await self.event_queue.put({
                                        'timestamp': last_changed,
                                        'entity_id': entity_id,
                                        'state': state['state'],
                                        'attributes': state['attributes'],
                                        'source': 'polling'
                                    })
                            
                            last_poll_time = current_time
                            
            except Exception as e:
                logger.warning(f"Polling error: {e}")
                await asyncio.sleep(5)
    
    async def enrich_event(self, event: Dict[str, Any]) -> Optional[EnrichedEvent]:
        """
        Add context without making assumptions
        Implements the exact enrichment from CLAUDE.md
        """
        try:
            entity_id = event['entity_id']
            
            # Special handling for combined living/kitchen space
            room = self.identify_room(entity_id)
            if room in ['livingroom', 'kitchen']:
                room = 'living_kitchen'  # Unified space
            
            # Calculate derived features
            derived = {
                'time_since_last_change': self.calculate_time_delta(event),
                'state_transition': self.get_transition_type(event),
                'concurrent_events': await self.find_concurrent_events(event),
                'zone_info': self.extract_zone_info(entity_id)
            }
            
            return EnrichedEvent(
                timestamp=event['timestamp'],
                entity_id=entity_id,
                state=event['state'],
                attributes=event['attributes'],
                room=room,
                sensor_type=self.identify_sensor_type(entity_id),
                derived=derived
            )
            
        except Exception as e:
            logger.warning(f"Error enriching event {event}: {e}")
            return None
    
    def extract_zone_info(self, entity_id: str) -> Dict[str, Any]:
        """
        Extract zone information from multi-zone rooms
        Implements the exact zone analysis from CLAUDE.md
        """
        # Check living_kitchen first (special dual-zone handling)
        if any(zone in entity_id for zone in ['livingroom', 'kitchen']):
            if 'livingroom_full' in entity_id or 'presence_livingroom_full' in entity_id:
                return {
                    'room': 'living_kitchen',
                    'zone': 'livingroom_full',
                    'zone_type': 'full',
                    'coverage': 'general_living_area',
                    'related_subzones': ['livingroom_couch']
                }
            elif 'kitchen_full' in entity_id or 'kitchen_pressence_full' in entity_id:
                return {
                    'room': 'living_kitchen',
                    'zone': 'kitchen_full',
                    'zone_type': 'full',
                    'coverage': 'general_kitchen_area',
                    'related_subzones': ['kitchen_stove', 'kitchen_sink', 'kitchen_dining']
                }
            elif 'livingroom_couch' in entity_id:
                return {
                    'room': 'living_kitchen',
                    'zone': 'livingroom_couch',
                    'zone_type': 'subzone',
                    'area': 'livingroom',
                    'specific_location': 'couch'
                }
            elif any(subzone in entity_id for subzone in ['kitchen_stove', 'kitchen_sink', 'kitchen_dining']):
                subzone = next(s for s in ['kitchen_stove', 'kitchen_sink', 'kitchen_dining'] if s in entity_id)
                return {
                    'room': 'living_kitchen',
                    'zone': subzone,
                    'zone_type': 'subzone',
                    'area': 'kitchen',
                    'specific_location': subzone.replace('kitchen_', '')
                }
        
        # Check other rooms
        for room, config in self.zone_config.items():
            if room == 'living_kitchen':
                continue  # Already handled above
                
            # Check full zone
            if config.get('full_zone') and config['full_zone'] in entity_id:
                return {
                    'room': room,
                    'zone': config['full_zone'],
                    'zone_type': 'full',
                    'coverage': 'general_room_area'
                }
            
            # Check subzones
            for subzone in config.get('subzones', []):
                if subzone in entity_id:
                    return {
                        'room': room,
                        'zone': subzone,
                        'zone_type': 'subzone',
                        'specific_location': subzone.split('_')[-1]
                    }
        
        return {
            'room': self.identify_room(entity_id),
            'zone': 'main',
            'zone_type': 'unknown'
        }
    
    def identify_room(self, entity_id: str) -> str:
        """Identify room from entity ID"""
        if 'livingroom' in entity_id:
            return 'livingroom'
        elif 'kitchen' in entity_id:
            return 'kitchen'
        elif 'bedroom' in entity_id:
            return 'bedroom'
        elif 'office' in entity_id:
            return 'office'
        elif 'bathroom' in entity_id:
            return 'bathroom' if 'small' not in entity_id else 'small_bathroom'
        elif 'guest' in entity_id:
            return 'guest_bedroom'
        else:
            return 'unknown'
    
    def identify_sensor_type(self, entity_id: str) -> str:
        """Identify sensor type from entity ID"""
        if 'presence' in entity_id or 'pressence' in entity_id:
            return 'presence'
        elif 'door' in entity_id and 'contact' in entity_id:
            return 'door'
        elif 'entrance' in entity_id:
            return 'entrance'
        elif 'temperature' in entity_id:
            return 'temperature'
        elif 'humidity' in entity_id:
            return 'humidity'
        elif 'light' in entity_id:
            return 'light'
        else:
            return 'unknown'
    
    def calculate_time_delta(self, event: Dict[str, Any]) -> float:
        """Calculate time since last change for this entity"""
        entity_id = event['entity_id']
        current_time = event['timestamp']
        
        if entity_id in self.last_state_changes:
            last_time = self.last_state_changes[entity_id]
            delta = (current_time - last_time).total_seconds()
        else:
            delta = 0.0
        
        self.last_state_changes[entity_id] = current_time
        return delta
    
    def get_transition_type(self, event: Dict[str, Any]) -> str:
        """Determine type of state transition"""
        state = event['state']
        
        if state in ['on', 'open', 'detected']:
            return 'activation'
        elif state in ['off', 'closed', 'clear']:
            return 'deactivation'
        else:
            return 'unknown'
    
    async def find_concurrent_events(self, event: Dict[str, Any]) -> List[str]:
        """Find events that happened concurrently (within 5 seconds)"""
        current_time = event['timestamp']
        entity_id = event['entity_id']
        
        # Look for events within 5 seconds
        window_start = current_time - timedelta(seconds=5)
        window_end = current_time + timedelta(seconds=5)
        
        concurrent = []
        
        # Check recent events (simplified - in full implementation would check event store)
        for other_entity, last_time in self.last_state_changes.items():
            if (other_entity != entity_id and 
                window_start <= last_time <= window_end):
                concurrent.append(other_entity)
        
        return concurrent
    
    async def compute_streaming_features(self, enriched_event: EnrichedEvent):
        """
        Immediate feature computation for real-time learning
        This would feed into the online learning models
        """
        try:
            # Extract features from current event context
            features = {
                'room': enriched_event.room,
                'sensor_type': enriched_event.sensor_type,
                'state': 1 if enriched_event.state in ['on', 'open', 'detected'] else 0,
                'hour': enriched_event.timestamp.hour,
                'day_of_week': enriched_event.timestamp.weekday(),
                'time_since_last_change': enriched_event.derived['time_since_last_change'],
                'concurrent_events_count': len(enriched_event.derived['concurrent_events']),
                'zone_type': enriched_event.derived['zone_info']['zone_type']
            }
            
            # This would be sent to online learning models for immediate updating
            logger.debug(f"Computed streaming features for {enriched_event.entity_id}: {features}")
            
        except Exception as e:
            logger.warning(f"Error computing streaming features: {e}")


class HomeAssistantAPI:
    """
    Wrapper for Home Assistant API interactions
    """
    
    def __init__(self, ha_url: str, token: str):
        self.ha_url = ha_url
        self.token = token
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
    
    async def subscribe_events(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to HA events via WebSocket"""
        # This would implement the full WebSocket event subscription
        # For now, placeholder that yields sample events
        while True:
            await asyncio.sleep(1)
            yield {
                'timestamp': datetime.now(),
                'entity_id': 'binary_sensor.test',
                'state': 'on',
                'attributes': {}
            }