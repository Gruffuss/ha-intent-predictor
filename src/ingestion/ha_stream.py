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
import websockets
from dataclasses import dataclass
from kafka import KafkaProducer
from aiokafka import AIOKafkaProducer
import websocket

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
    Implements the exact streaming approach from CLAUDE.md with Kafka integration
    """
    
    def __init__(self, ha_config: Dict, kafka_config: Dict, timeseries_db=None, feature_store=None):
        self.ha_url = ha_config['url'].rstrip('/')
        self.token = ha_config['token']
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        
        # Kafka configuration
        self.kafka_config = kafka_config
        self.kafka_producer = None
        
        # Database connections
        self.timeseries_db = timeseries_db
        self.feature_store = feature_store
        
        # State tracking for enrichment
        self.last_state_changes = {}
        self.concurrent_events = {}
        
        # Event queue for processing
        self.event_queue = asyncio.Queue()
        
        # Zone configuration from CLAUDE.md
        self.zone_config = self._initialize_zone_config()
        
        # Configured sensors from CLAUDE.md - ONLY these sensors
        self.configured_sensors = self._get_configured_sensors()
        
        logger.info(f"Initialized HA data stream with {len(self.configured_sensors)} configured sensors")
    
    async def initialize(self):
        """Initialize Kafka producer and connections"""
        try:
            # Initialize async Kafka producer
            self.kafka_producer = AIOKafkaProducer(
                bootstrap_servers=self.kafka_config['bootstrap_servers'],
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                **self.kafka_config.get('producer', {})
            )
            
            # Start the producer
            await self.kafka_producer.start()
            
            logger.info("Async Kafka producer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
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
    
    def _get_configured_sensors(self) -> set:
        """
        Get only the sensors configured in CLAUDE.md
        Prevents processing ALL HA entities
        """
        return {
            # Presence sensors
            'binary_sensor.presence_livingroom_full',
            'binary_sensor.presence_livingroom_couch',
            'binary_sensor.kitchen_pressence_full_kitchen',
            'binary_sensor.kitchen_pressence_stove',
            'binary_sensor.kitchen_pressence_sink',
            'binary_sensor.kitchen_pressence_dining_table',
            'binary_sensor.bedroom_presence_sensor_full_bedroom',
            'binary_sensor.bedroom_presence_sensor_anca_bed_side',
            'binary_sensor.bedroom_vladimir_bed_side',
            'binary_sensor.bedroom_floor',
            'binary_sensor.bedroom_entrance',
            'binary_sensor.office_presence_full_office',
            'binary_sensor.office_presence_anca_desk',
            'binary_sensor.office_presence_vladimir_desk',
            'binary_sensor.office_entrance',
            'binary_sensor.bathroom_entrance',
            'binary_sensor.presence_small_bathroom_entrance',
            'binary_sensor.guest_bedroom_entrance',
            'binary_sensor.presence_ground_floor_hallway',
            'binary_sensor.upper_hallway',
            'binary_sensor.upper_hallway_upstairs',
            'binary_sensor.upper_hallway_downstairs',
            'binary_sensor.presence_stairs_up_ground_floor',
            
            # Door sensors
            'binary_sensor.bathroom_door_sensor_contact',
            'binary_sensor.bedroom_door_sensor_contact',
            'binary_sensor.office_door_sensor_contact',
            'binary_sensor.guest_bedroom_door_sensor_contact',
            'binary_sensor.small_bathroom_door_sensor_contact',
            
            # Climate sensors
            'sensor.livingroom_env_sensor_temperature',
            'sensor.livingroom_env_sensor_humidity',
            'sensor.bedroom_env_sensor_temperature',
            'sensor.bedroom_env_sensor_humidity',
            'sensor.office_env_sensor_temperature',
            'sensor.office_env_sensor_humidity',
            'sensor.bathroom_env_sensor_temperature',
            'sensor.bathroom_env_sensor_humidity',
            'sensor.guest_bedroom_env_sensor_temperature',
            'sensor.guest_bedroom_env_sensor_humidity',
            'sensor.upper_hallway_env_sensor_temperature',
            'sensor.upper_hallway_env_sensor_humidity',
            'sensor.attic_env_sensor_temperature',
            'sensor.attic_env_sensor_humidity',
            'sensor.big_bath_env_sensor_temperature',
            'sensor.big_bath_env_sensor_humidity',
            
            # Light level sensors
            'sensor.bedroom_presence_light_level',
            'sensor.kitchen_pressence_light_level',
            'sensor.livingroom_pressence_light_level',
            'sensor.office_presence_light_level',
            'sensor.upper_hallway_pressence_light_level'
        }
    
    def _is_configured_sensor(self, entity_id: str) -> bool:
        """Check if sensor is in our configured list"""
        return entity_id in self.configured_sensors
    
    async def stream_all_sensors(self):
        """
        Aggressively pull all sensor data continuously
        Subscribe to ALL state changes as specified in CLAUDE.md
        """
        logger.info("Starting continuous sensor streaming")
        
        # Start WebSocket connection for real-time events
        websocket_task = asyncio.create_task(self._websocket_stream())
        
        # Start periodic polling as backup
        polling_task = asyncio.create_task(self._polling_stream())
        
        # Start event processing task
        processing_task = asyncio.create_task(self._process_event_queue())
        
        try:
            # Wait for all tasks to complete
            await asyncio.gather(websocket_task, polling_task, processing_task)
                    
        except Exception as e:
            logger.error(f"Error in sensor streaming: {e}")
        finally:
            # Clean up tasks
            websocket_task.cancel()
            polling_task.cancel()
            processing_task.cancel()
    
    async def _process_event_queue(self):
        """Process events from the queue and send to Kafka"""
        while True:
            try:
                # Get event from queue (blocks until available)
                raw_event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Transform to standardized format
                enriched_event = await self.enrich_event(raw_event)
                
                if enriched_event:
                    # Send to Kafka for processing
                    await self.send_to_kafka(enriched_event)
                    
                    # Store in TimescaleDB if available
                    if self.timeseries_db:
                        await self.store_event(enriched_event)
                    
                    # Immediate feature computation
                    await self.compute_streaming_features(enriched_event)
                
            except asyncio.TimeoutError:
                # No events in queue, continue
                continue
                
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                continue
    
    async def send_to_kafka(self, enriched_event: EnrichedEvent):
        """Send enriched event to Kafka for processing"""
        try:
            if not self.kafka_producer:
                logger.warning("Kafka producer not initialized")
                return
            
            # Convert to dict for JSON serialization
            event_dict = {
                'timestamp': enriched_event.timestamp.isoformat(),
                'entity_id': enriched_event.entity_id,
                'state': enriched_event.state,
                'attributes': enriched_event.attributes,
                'room': enriched_event.room,
                'sensor_type': enriched_event.sensor_type,
                'derived': enriched_event.derived
            }
            
            # Send to sensor events topic
            topic = self.kafka_config['topics']['sensor_events']
            
            # Use room as partition key for locality
            partition_key = enriched_event.room.encode('utf-8') if enriched_event.room else None
            
            await self.kafka_producer.send(
                topic=topic,
                key=partition_key,
                value=event_dict
            )
            
            logger.debug(f"Sent event to Kafka: {enriched_event.entity_id}")
            
        except Exception as e:
            logger.error(f"Error sending to Kafka: {e}")
    
    async def store_event(self, enriched_event: EnrichedEvent):
        """Store event in TimescaleDB"""
        try:
            # Extract numeric value if applicable
            numeric_value = None
            if enriched_event.state and enriched_event.state.replace('.', '').replace('-', '').isdigit():
                try:
                    numeric_value = float(enriched_event.state)
                except ValueError:
                    pass
            
            # Store in database
            await self.timeseries_db.execute("""
                INSERT INTO sensor_events (
                    timestamp, entity_id, state, numeric_value, attributes,
                    room, sensor_type, zone_type, zone_info, person, enriched_data
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """, 
                enriched_event.timestamp,
                enriched_event.entity_id,
                enriched_event.state,
                numeric_value,
                json.dumps(enriched_event.attributes),
                enriched_event.room,
                enriched_event.sensor_type,
                enriched_event.derived.get('zone_info', {}).get('zone_type'),
                json.dumps(enriched_event.derived.get('zone_info', {})),
                enriched_event.derived.get('zone_info', {}).get('person'),
                json.dumps(enriched_event.derived)
            )
            
        except Exception as e:
            logger.error(f"Error storing event in database: {e}")
    
    async def shutdown(self):
        """Shutdown Kafka producer and connections"""
        try:
            if self.kafka_producer:
                await self.kafka_producer.stop()
                logger.info("Kafka producer stopped")
        except Exception as e:
            logger.error(f"Error stopping Kafka producer: {e}")
    
    async def _websocket_stream(self):
        """WebSocket connection for real-time events using websockets library"""
        while True:
            try:
                # Connect to HA WebSocket API
                uri = f"ws://{self.ha_url.replace('http://', '').replace('https://', '')}/api/websocket"
                
                async with websockets.connect(uri) as websocket:
                    # Authenticate
                    auth_msg = await websocket.recv()
                    auth_data = json.loads(auth_msg)
                    
                    if auth_data['type'] == 'auth_required':
                        await websocket.send(json.dumps({
                            'type': 'auth',
                            'access_token': self.token
                        }))
                        
                        auth_result = await websocket.recv()
                        auth_result_data = json.loads(auth_result)
                        
                        if auth_result_data['type'] != 'auth_ok':
                            logger.error("WebSocket authentication failed")
                            return
                    
                    # Subscribe to state changes
                    await websocket.send(json.dumps({
                        'id': 1,
                        'type': 'subscribe_events',
                        'event_type': 'state_changed'
                    }))
                    
                    logger.info("WebSocket connected and subscribed to state changes")
                    
                    # Process incoming events
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            
                            if data.get('type') == 'event' and data.get('event', {}).get('event_type') == 'state_changed':
                                event_data = data['event']['data']
                                entity_id = event_data['entity_id']
                                
                                # ONLY process configured sensors
                                if not self._is_configured_sensor(entity_id):
                                    continue
                                
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
        from datetime import timezone
        last_poll_time = datetime.now(timezone.utc)
        
        while True:
            try:
                await asyncio.sleep(2)  # Poll every 2 seconds
                
                # Get all current states
                async with aiohttp.ClientSession() as session:
                    async with session.get(f'{self.ha_url}/api/states', headers=self.headers) as response:
                        if response.status == 200:
                            states = await response.json()
                            
                            current_time = datetime.now(timezone.utc)
                            
                            # Check for state changes since last poll
                            for state in states:
                                entity_id = state['entity_id']
                                
                                # ONLY process configured sensors
                                if not self._is_configured_sensor(entity_id):
                                    continue
                                    
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