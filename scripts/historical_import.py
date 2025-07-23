#!/usr/bin/env python3
"""
Historical Data Import Script for HA Intent Predictor

Imports 180 days of historical sensor data from Home Assistant
following CLAUDE.md specifications for pure learning approach.

This script:
- Connects to Home Assistant database directly
- Imports all 98 sensor data points 
- Processes data for multi-zone room analysis
- Handles person-specific sensor data
- Prepares data for adaptive learning without assumptions
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import asyncpg
import aiohttp
import pandas as pd
from tqdm import tqdm

# Add src and config to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from storage.timeseries_db import TimescaleDBManager
from storage.feature_store import RedisFeatureStore
from config.config_loader import ConfigLoader
from ingestion.data_enricher import DynamicFeatureDiscovery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HistoricalDataImporter:
    """
    Imports historical sensor data from Home Assistant.
    
    Follows CLAUDE.md philosophy:
    - No assumptions about patterns or schedules
    - Pure data-driven learning approach
    - Person-specific data handling
    - Multi-zone room analysis
    """
    
    def __init__(self, config_path: str = "config/system.yaml"):
        self.config = ConfigLoader(config_path)
        # Use ConfigLoader's get_sensors method to read from sensors.yaml
        self.sensors_config = self.config.get_sensors()
        self.rooms_config = self._get_default_rooms()
        
        # Initialize components
        self.timeseries_db = None
        self.feature_store = None
        self.data_enricher = None
        
        # Import statistics
        self.stats = {
            'total_records': 0,
            'processed_records': 0,
            'errors': 0,
            'sensor_counts': {},
            'room_counts': {},
            'person_specific_counts': {'anca': 0, 'vladimir': 0},
            'anomalies_detected': 0
        }
    
    
    def _get_default_rooms(self):
        """Get default room configuration"""
        return [
            'living_kitchen',  # Combined space
            'bedroom',
            'office',
            'bathroom',
            'small_bathroom',
            'guest_bedroom'
        ]
    
    async def initialize(self):
        """Initialize database connections and components"""
        logger.info("Initializing historical data importer...")
        
        # Initialize TimescaleDB
        db_config = self.config.get('database.timescale')
        db_connection_string = f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        self.timeseries_db = TimescaleDBManager(db_connection_string)
        await self.timeseries_db.initialize()
        
        # Initialize Redis for feature caching
        redis_config = self.config.get('redis')
        redis_url = f"redis://{redis_config['host']}:{redis_config['port']}/{redis_config['db']}"
        self.feature_store = RedisFeatureStore(redis_url)
        await self.feature_store.initialize()
        
        # Initialize data enricher
        self.data_enricher = DynamicFeatureDiscovery()
        
        # Skip table creation if tables already exist (bootstrap creates them)
        try:
            # Check if main table exists
            async with self.timeseries_db.engine.begin() as conn:
                from sqlalchemy import text
                result = await conn.execute(text("SELECT COUNT(*) FROM sensor_events LIMIT 1;"))
                print("  ✓ Using existing database tables")
        except:
            # Tables don't exist, create them
            await self.create_tables()
        
        logger.info("Initialization complete")
    
    async def _query_database(self, query: str, params: dict = None) -> List[Dict]:
        """Generic database query method to replace missing fetch method"""
        try:
            async with self.timeseries_db.session_factory() as session:
                from sqlalchemy import text
                result = await session.execute(text(query), params or {})
                rows = result.fetchall()
                
                # Convert rows to dictionaries
                if rows:
                    columns = result.keys()
                    return [dict(zip(columns, row)) for row in rows]
                return []
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return []
    
    async def create_tables(self):
        """Create TimescaleDB tables for historical data"""
        logger.info("Creating database tables...")
        
        async with self.timeseries_db.engine.begin() as conn:
            from sqlalchemy import text
            
            # Main sensor events table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sensor_events (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    entity_id VARCHAR(255) NOT NULL,
                    state VARCHAR(255),
                    numeric_value DOUBLE PRECISION,
                    attributes JSONB,
                    room VARCHAR(100),
                    sensor_type VARCHAR(50),
                    zone_type VARCHAR(50),
                    zone_info JSONB,
                    person VARCHAR(50),
                    enriched_data JSONB,
                    processed_at TIMESTAMPTZ DEFAULT NOW()
                );
            """))
            
            # Create hypertable for time-series optimization
            try:
                await conn.execute(text("""
                    SELECT create_hypertable('sensor_events', 'timestamp', 
                                           if_not_exists => TRUE,
                                           chunk_time_interval => INTERVAL '1 day');
                """))
            except Exception as e:
                logger.warning(f"Hypertable creation failed (may already exist): {e}")
            
            # Create indices for efficient querying
            indices = [
                "CREATE INDEX IF NOT EXISTS idx_sensor_events_entity_id ON sensor_events (entity_id, timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_sensor_events_room ON sensor_events (room, timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_sensor_events_sensor_type ON sensor_events (sensor_type, timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_sensor_events_person ON sensor_events (person, timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_sensor_events_zone_type ON sensor_events (zone_type, timestamp DESC);"
            ]
            
            for index_sql in indices:
                try:
                    await conn.execute(text(index_sql))
                except Exception as e:
                    logger.warning(f"Index creation failed (may already exist): {e}")
            
            # Pattern discovery table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS discovered_patterns (
                    id BIGSERIAL PRIMARY KEY,
                    room VARCHAR(100) NOT NULL,
                    pattern_type VARCHAR(100) NOT NULL,
                    pattern_data JSONB NOT NULL,
                    significance_score DOUBLE PRECISION,
                    confidence DOUBLE PRECISION,
                    support_count INTEGER,
                    discovered_at TIMESTAMPTZ DEFAULT NOW(),
                    last_validated TIMESTAMPTZ,
                    is_active BOOLEAN DEFAULT TRUE
                );
            """))
            
            # Room occupancy inference table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS room_occupancy (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    room VARCHAR(100) NOT NULL,
                    occupied BOOLEAN NOT NULL,
                    confidence DOUBLE PRECISION,
                    inference_method VARCHAR(100),
                    supporting_evidence JSONB,
                    person VARCHAR(50),
                    duration_minutes INTEGER,
                    processed_at TIMESTAMPTZ DEFAULT NOW()
                );
            """))
            
            # Create hypertable for occupancy data
            try:
                await conn.execute(text("""
                    SELECT create_hypertable('room_occupancy', 'timestamp',
                                           if_not_exists => TRUE,
                                           chunk_time_interval => INTERVAL '1 day');
                """))
            except Exception as e:
                logger.warning(f"Room occupancy hypertable creation failed (may already exist): {e}")
        
        logger.info("Database tables created successfully")
    
    async def import_from_ha(self, days: int = 180):
        """
        Import historical data from Home Assistant.
        
        Args:
            days: Number of days to import (default 180 as per CLAUDE.md)
        """
        logger.info(f"Starting import of {days} days of historical data...")
        
        # Calculate date range - stop at yesterday to avoid infinite current-day import
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)  # Start of today
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Import range: {start_date} to {end_date}")
        
        # Get all sensor entities
        all_sensors = self.get_all_sensor_entities()
        logger.info(f"Found {len(all_sensors)} sensors to import")
        
        # Import data in chunks to avoid memory issues
        chunk_size = 7  # Process 7 days at a time
        current_date = start_date
        
        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=chunk_size), end_date)
            
            logger.info(f"Processing chunk: {current_date} to {chunk_end}")
            
            await self.import_date_range(
                sensors=all_sensors,
                start_date=current_date,
                end_date=chunk_end
            )
            
            current_date = chunk_end
        
        # Post-processing
        await self.post_process_data()
        
        # Generate summary report
        await self.generate_import_report()
        
        logger.info("Historical data import completed successfully")
    
    def get_all_sensor_entities(self) -> List[str]:
        """Get all sensor entities from configuration"""
        # The configuration now returns a flat list of sensors
        return list(set(self.sensors_config))  # Remove duplicates
    
    async def import_date_range(self, sensors: List[str], start_date: datetime, end_date: datetime):
        """Import sensor data for a specific date range"""
        
        # Connect to Home Assistant database
        ha_db_config = self.config.get('home_assistant.database', {})
        
        if not ha_db_config:
            # Fall back to Home Assistant API
            await self.import_via_api(sensors, start_date, end_date)
            return
        
        # Direct database connection (faster)
        try:
            ha_conn = await asyncpg.connect(
                host=ha_db_config.get('host', 'localhost'),
                port=ha_db_config.get('port', 5432),
                database=ha_db_config.get('database', 'homeassistant'),
                user=ha_db_config.get('user', 'homeassistant'),
                password=ha_db_config.get('password')
            )
            
            await self.import_via_database(ha_conn, sensors, start_date, end_date)
            await ha_conn.close()
            
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
            logger.info("Falling back to API import...")
            await self.import_via_api(sensors, start_date, end_date)
    
    async def import_via_database(self, ha_conn, sensors: List[str], start_date: datetime, end_date: datetime):
        """Import data directly from Home Assistant database"""
        
        # Query historical data
        query = """
            SELECT 
                s.entity_id,
                s.state,
                s.attributes,
                s.last_changed,
                s.last_updated,
                smd.device_class,
                smd.unit_of_measurement
            FROM states s
            LEFT JOIN states_meta sm ON s.metadata_id = sm.metadata_id
            LEFT JOIN state_meta_data smd ON sm.entity_id = smd.entity_id
            WHERE s.entity_id = ANY($1)
            AND s.last_changed >= $2
            AND s.last_changed < $3
            ORDER BY s.last_changed ASC;
        """
        
        # Process in batches
        batch_size = 1000
        records_processed = 0
        
        async with ha_conn.transaction():
            async for record in ha_conn.cursor(query, sensors, start_date, end_date):
                # Process each record
                await self.process_sensor_record(record)
                
                records_processed += 1
                
                # Commit in batches
                if records_processed % batch_size == 0:
                    logger.info(f"Processed {records_processed} records...")
    
    async def import_via_api(self, sensors: List[str], start_date: datetime, end_date: datetime):
        """Import data via Home Assistant API (slower but more reliable)"""
        
        ha_config = self.config.get('home_assistant')
        headers = {
            'Authorization': f"Bearer {ha_config['token']}",
            'Content-Type': 'application/json'
        }
        
        async with aiohttp.ClientSession(headers=headers) as session:
            # Import each sensor's history
            for sensor in tqdm(sensors, desc="Importing sensors"):
                await self.import_sensor_history(session, sensor, start_date, end_date)
    
    async def import_sensor_history(self, session: aiohttp.ClientSession, entity_id: str, start_date: datetime, end_date: datetime):
        """Import history for a single sensor"""
        
        ha_config = self.config.get('home_assistant')
        url = f"{ha_config['url']}/api/history/period/{start_date.isoformat()}"
        
        params = {
            'filter_entity_id': entity_id,
            'end_time': end_date.isoformat(),
            'minimal_response': 'true'
        }
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Process sensor data
                    if data and len(data) > 0:
                        sensor_data = data[0]  # First (and only) entity
                        
                        for state_record in sensor_data:
                            await self.process_api_record(entity_id, state_record)
                            
                        self.stats['sensor_counts'][entity_id] = len(sensor_data)
                else:
                    logger.warning(f"Failed to fetch history for {entity_id}: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error importing {entity_id}: {e}")
            self.stats['errors'] += 1
    
    async def process_api_record(self, entity_id: str, record: Dict):
        """Process a single API record"""
        
        # Extract basic information
        timestamp = datetime.fromisoformat(record['last_changed'].replace('Z', '+00:00'))
        state = record.get('state')
        attributes = record.get('attributes', {})
        
        # Enrich event data with room identification
        room = self._identify_room(entity_id)
        # Apply living/kitchen merge immediately during import
        if room in ['livingroom', 'kitchen']:
            room = 'living_kitchen'
            
        sensor_type = self._identify_sensor_type(entity_id)
        
        event_data = {
            'entity_id': entity_id,
            'timestamp': timestamp,
            'state': state,
            'attributes': attributes,
            'room': room,
            'sensor_type': sensor_type
        }
        await self.store_sensor_event(event_data)
        
        # Update statistics
        self.stats['processed_records'] += 1
        
        # Track person-specific data
        if event_data.get('person'):
            self.stats['person_specific_counts'][event_data['person']] += 1
    
    async def process_sensor_record(self, record):
        """Process a single database record"""
        
        # Convert to standard format
        timestamp = record['last_changed']
        entity_id = record['entity_id']
        state = record['state']
        attributes = json.loads(record['attributes']) if record['attributes'] else {}
        
        # Enrich event data with room identification
        room = self._identify_room(entity_id)
        # Apply living/kitchen merge immediately during import
        if room in ['livingroom', 'kitchen']:
            room = 'living_kitchen'
            
        sensor_type = self._identify_sensor_type(entity_id)
        
        event_data = {
            'entity_id': entity_id,
            'timestamp': timestamp,
            'state': state,
            'attributes': attributes,
            'room': room,
            'sensor_type': sensor_type
        }
        await self.store_sensor_event(event_data)
        
        # Update statistics
        self.stats['processed_records'] += 1
    
    async def store_sensor_event(self, event_data: Dict):
        """Store enriched sensor event in TimescaleDB"""
        
        # Extract numeric value if applicable
        numeric_value = None
        if event_data.get('state') and event_data['state'].replace('.', '').replace('-', '').isdigit():
            try:
                numeric_value = float(event_data['state'])
            except ValueError:
                pass
        
        # Insert into database using proper method
        event_data['numeric_value'] = numeric_value
        await self.timeseries_db.insert_sensor_event(event_data)
    
    async def post_process_data(self):
        """Post-process imported data for learning"""
        logger.info("Starting post-processing...")
        
        # Merge living room and kitchen data as per CLAUDE.md
        await self.merge_room_data('livingroom', 'kitchen', 'living_kitchen')
        
        # Generate room occupancy inferences
        await self.generate_occupancy_inferences()
        
        # Detect and flag anomalies (potential cat activity)
        await self.detect_anomalies()
        
        # Create initial feature cache
        await self.create_initial_feature_cache()
        
        logger.info("Post-processing completed")
    
    async def merge_room_data(self, room1: str, room2: str, new_room: str):
        """Merge room data for combined spaces"""
        logger.info(f"Merging {room1} and {room2} into {new_room}")
        
        # Update room field for combined space
        async with self.timeseries_db.engine.begin() as conn:
            from sqlalchemy import text
            await conn.execute(text("""
                UPDATE sensor_events 
                SET room = :new_room 
                WHERE room IN (:room1, :room2)
            """), {'new_room': new_room, 'room1': room1, 'room2': room2})
        
        # Update statistics
        self.stats['room_counts'][new_room] = (
            self.stats['room_counts'].get(room1, 0) + 
            self.stats['room_counts'].get(room2, 0)
        )
    
    async def generate_occupancy_inferences(self):
        """Generate room occupancy inferences from sensor data"""
        logger.info("Generating occupancy inferences...")
        
        # Get all rooms with sensors
        rooms = await self._query_database("""
            SELECT DISTINCT room FROM sensor_events 
            WHERE room IS NOT NULL
        """)
        
        for room_record in rooms:
            room = room_record['room']
            
            if room in ['bathroom', 'small_bathroom']:
                # Special handling for bathrooms
                await self.infer_bathroom_occupancy(room)
            else:
                # Standard presence-based inference
                await self.infer_room_occupancy(room)
    
    async def infer_bathroom_occupancy(self, room: str):
        """Infer bathroom occupancy from entrance and door sensors"""
        
        # Get entrance and door events
        events = await self._query_database("""
            SELECT timestamp, entity_id, state, zone_type
            FROM sensor_events
            WHERE room = $1
            AND (zone_type = 'entrance_zone' OR entity_id LIKE '%door%')
            ORDER BY timestamp
        """, room)
        
        # Apply bathroom occupancy inference logic
        occupancy_periods = []
        current_occupancy = None
        
        for event in events:
            # Implement bathroom occupancy logic from CLAUDE.md
            if 'entrance' in event['entity_id'] and event['state'] == 'on':
                if not current_occupancy:
                    current_occupancy = {
                        'start_time': event['timestamp'],
                        'room': room,
                        'method': 'entrance_door_correlation'
                    }
            
            elif 'door' in event['entity_id']:
                if current_occupancy and event['state'] == 'off':  # Door opened
                    # Check if enough time passed
                    duration = (event['timestamp'] - current_occupancy['start_time']).total_seconds() / 60
                    
                    if duration > 0.5:  # Minimum 30 seconds
                        occupancy_periods.append({
                            'start_time': current_occupancy['start_time'],
                            'end_time': event['timestamp'],
                            'room': room,
                            'duration': duration,
                            'confidence': 0.8
                        })
                    
                    current_occupancy = None
        
        # Store occupancy periods
        for period in occupancy_periods:
            await self.store_occupancy_period(period)
    
    async def infer_room_occupancy(self, room: str):
        """Infer occupancy for rooms with presence sensors"""
        
        # Get presence events
        events = await self._query_database("""
            SELECT timestamp, entity_id, state, zone_type
            FROM sensor_events
            WHERE room = $1
            AND sensor_type = 'presence'
            ORDER BY timestamp
        """, room)
        
        # Simple occupancy inference based on presence sensors
        occupancy_periods = []
        current_occupancy = None
        
        for event in events:
            if event['state'] == 'on' and not current_occupancy:
                current_occupancy = {
                    'start_time': event['timestamp'],
                    'room': room,
                    'method': 'presence_sensor'
                }
            
            elif event['state'] == 'off' and current_occupancy:
                duration = (event['timestamp'] - current_occupancy['start_time']).total_seconds() / 60
                
                if duration > 0.1:  # Minimum 6 seconds
                    occupancy_periods.append({
                        'start_time': current_occupancy['start_time'],
                        'end_time': event['timestamp'],
                        'room': room,
                        'duration': duration,
                        'confidence': 0.9
                    })
                
                current_occupancy = None
        
        # Store occupancy periods
        for period in occupancy_periods:
            await self.store_occupancy_period(period)
    
    async def store_occupancy_period(self, period: Dict):
        """Store an occupancy period in the database"""
        
        async with self.timeseries_db.engine.begin() as conn:
            from sqlalchemy import text
            await conn.execute(text("""
                INSERT INTO room_occupancy (
                    timestamp, room, occupied, confidence, 
                    inference_method, duration_minutes
                ) VALUES (:start_time, :room, :occupied, :confidence, :method, :duration)
            """), {
                'start_time': period['start_time'],
                'room': period['room'],
                'occupied': True,
                'confidence': period['confidence'],
                'method': period.get('method', 'unknown'),
                'duration': period['duration']
            })
    
    async def detect_anomalies(self):
        """Detect anomalies in sensor data (potential cat activity)"""
        logger.info("Detecting anomalies...")
        
        # Look for impossible movements
        anomalies = await self._query_database("""
            WITH rapid_movements AS (
                SELECT 
                    e1.timestamp as t1,
                    e2.timestamp as t2,
                    e1.entity_id as sensor1,
                    e2.entity_id as sensor2,
                    e1.room as room1,
                    e2.room as room2,
                    EXTRACT(EPOCH FROM (e2.timestamp - e1.timestamp)) as time_diff
                FROM sensor_events e1
                JOIN sensor_events e2 ON e2.timestamp > e1.timestamp
                WHERE e1.sensor_type = 'presence'
                AND e2.sensor_type = 'presence'
                AND e1.state = 'on'
                AND e2.state = 'on'
                AND e1.room != e2.room
                AND EXTRACT(EPOCH FROM (e2.timestamp - e1.timestamp)) < 5
            )
            SELECT * FROM rapid_movements
            WHERE time_diff < 3  -- Less than 3 seconds between rooms
        """)
        
        self.stats['anomalies_detected'] = len(anomalies)
        
        # Store anomalies for later analysis
        if anomalies:
            async with self.timeseries_db.engine.begin() as conn:
                from sqlalchemy import text
                for anomaly in anomalies:
                    # Convert anomaly data to JSON-serializable format
                    # SQLAlchemy returns Row objects from queries, access via _mapping
                    anomaly_dict = dict(anomaly._mapping)
                    
                    # Convert non-JSON-serializable objects to strings
                    for key, value in anomaly_dict.items():
                        if hasattr(value, 'isoformat'):  # datetime objects
                            anomaly_dict[key] = value.isoformat()
                        elif hasattr(value, '__float__'):  # Decimal objects
                            anomaly_dict[key] = float(value)
                        elif not isinstance(value, (str, int, float, bool, type(None))):
                            # Convert any other non-serializable objects to string
                            anomaly_dict[key] = str(value)
                    
                    await conn.execute(text("""
                        INSERT INTO discovered_patterns (
                            room, pattern_type, pattern_data, significance_score
                        ) VALUES (:room, :pattern_type, :pattern_data, :significance_score)
                    """), {
                        'room': 'multiple',
                        'pattern_type': 'rapid_movement_anomaly',
                        'pattern_data': json.dumps(anomaly_dict),
                        'significance_score': 0.8
                    })
    
    async def create_initial_feature_cache(self):
        """Create initial feature cache in Redis"""
        logger.info("Creating initial feature cache...")
        
        # Cache basic room statistics
        rooms = await self._query_database("""
            SELECT 
                room,
                COUNT(*) as event_count,
                COUNT(DISTINCT entity_id) as sensor_count,
                MIN(timestamp) as first_event,
                MAX(timestamp) as last_event
            FROM sensor_events
            GROUP BY room
        """)
        
        for room_stat in rooms:
            await self.feature_store.set(
                f"room_stats:{room_stat['room']}", 
                json.dumps(dict(room_stat), default=str),
                ttl=3600
            )
        
        # Cache person-specific statistics
        person_stats = await self._query_database("""
            SELECT 
                person,
                COUNT(*) as event_count,
                COUNT(DISTINCT room) as room_count
            FROM sensor_events
            WHERE person IS NOT NULL
            GROUP BY person
        """)
        
        for person_stat in person_stats:
            await self.feature_store.set(
                f"person_stats:{person_stat['person']}", 
                json.dumps(dict(person_stat), default=str),
                ttl=3600
            )
    
    async def generate_import_report(self):
        """Generate comprehensive import report"""
        
        report = {
            'import_timestamp': datetime.now().isoformat(),
            'statistics': self.stats,
            'data_quality': await self.assess_data_quality(),
            'room_summary': await self.get_room_summary(),
            'sensor_coverage': await self.get_sensor_coverage(),
            'anomaly_summary': await self.get_anomaly_summary()
        }
        
        # Save report to file
        report_path = Path("logs/import_report.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Import report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("HISTORICAL DATA IMPORT SUMMARY")
        print("="*50)
        print(f"Total records processed: {self.stats['processed_records']:,}")
        print(f"Total sensors: {len(self.stats['sensor_counts'])}")
        print(f"Errors encountered: {self.stats['errors']}")
        print(f"Anomalies detected: {self.stats['anomalies_detected']}")
        print(f"Anca-specific events: {self.stats['person_specific_counts']['anca']:,}")
        print(f"Vladimir-specific events: {self.stats['person_specific_counts']['vladimir']:,}")
        print("\nData is ready for adaptive learning!")
        print("="*50)
    
    async def assess_data_quality(self) -> Dict:
        """Assess the quality of imported data"""
        
        # Check for data gaps
        gaps = await self._query_database("""
            SELECT 
                room,
                COUNT(*) as event_count,
                MAX(timestamp) - MIN(timestamp) as time_span,
                COUNT(DISTINCT DATE(timestamp)) as days_with_data
            FROM sensor_events
            GROUP BY room
        """)
        
        return {
            'room_gaps': [dict(gap) for gap in gaps],
            'total_sensors_active': len(self.stats['sensor_counts']),
            'data_completeness': min(100, (self.stats['processed_records'] / 1000000) * 100)
        }
    
    async def get_room_summary(self) -> Dict:
        """Get summary statistics by room"""
        
        room_stats = await self._query_database("""
            SELECT 
                room,
                COUNT(*) as events,
                COUNT(DISTINCT entity_id) as sensors,
                COUNT(DISTINCT person) as people
            FROM sensor_events
            WHERE room IS NOT NULL
            GROUP BY room
            ORDER BY events DESC
        """)
        
        return {room['room']: dict(room) for room in room_stats}
    
    async def get_sensor_coverage(self) -> Dict:
        """Get sensor coverage statistics"""
        
        coverage = await self._query_database("""
            SELECT 
                sensor_type,
                COUNT(DISTINCT entity_id) as sensor_count,
                COUNT(*) as total_events
            FROM sensor_events
            GROUP BY sensor_type
            ORDER BY total_events DESC
        """)
        
        return {sensor['sensor_type']: dict(sensor) for sensor in coverage}
    
    def _identify_room(self, entity_id: str) -> str:
        """Identify room from entity ID - copied from ha_stream.py"""
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
    
    def _identify_sensor_type(self, entity_id: str) -> str:
        """Identify sensor type from entity ID - copied from ha_stream.py"""
        if 'presence' in entity_id or 'pressence' in entity_id:
            return 'presence'
        elif 'temperature' in entity_id:
            return 'temperature'
        elif 'humidity' in entity_id:
            return 'humidity'
        elif 'light_level' in entity_id:
            return 'light'
        elif 'door' in entity_id:
            return 'door'
        elif 'contact' in entity_id:
            return 'contact'
        elif 'motion' in entity_id:
            return 'motion'
        else:
            return 'unknown'
    
    async def get_anomaly_summary(self) -> Dict:
        """Get summary of detected anomalies"""
        
        anomalies = await self._query_database("""
            SELECT 
                pattern_type,
                COUNT(*) as count,
                AVG(significance_score) as avg_significance
            FROM discovered_patterns
            GROUP BY pattern_type
        """)
        
        return {anomaly['pattern_type']: dict(anomaly) for anomaly in anomalies}
    
    async def shutdown(self):
        """Cleanup resources"""
        if self.timeseries_db:
            await self.timeseries_db.close()
        if self.feature_store:
            await self.feature_store.close()


async def main():
    """Main entry point for historical data import"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Import historical data from Home Assistant')
    parser.add_argument('--days', type=int, default=180, help='Number of days to import')
    parser.add_argument('--config', type=str, default='config/system.yaml', help='Configuration file path')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Initialize importer
    importer = HistoricalDataImporter(args.config)
    
    try:
        await importer.initialize()
        await importer.import_from_ha(days=args.days)
        
    except Exception as e:
        logger.error(f"Import failed: {e}")
        sys.exit(1)
    
    finally:
        await importer.shutdown()


if __name__ == "__main__":
    asyncio.run(main())