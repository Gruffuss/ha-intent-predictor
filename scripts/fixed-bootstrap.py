#!/usr/bin/env python3
"""
Fixed Bootstrap System for HA Intent Predictor
Corrects the database schema mismatches and uses proper class names
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

# Add src and config to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent))

from storage.timeseries_db import TimescaleDBManager
from storage.feature_store import RedisFeatureStore
from storage.model_store import ModelStore
from learning.pattern_discovery import PatternDiscovery
from learning.adaptive_predictor import AdaptiveOccupancyPredictor
from integration.ha_publisher import DynamicHAIntegration
from config.config_loader import ConfigLoader
from learning.anomaly_detection import AdaptiveCatDetector

logger = logging.getLogger(__name__)


class FixedSystemBootstrap:
    """
    Fixed bootstrap that creates the correct database schema
    """
    
    def __init__(self, config_path: str = "config/system.yaml"):
        self.config = ConfigLoader(config_path)
        self.components = {}
        
        # Sensor groups from CLAUDE.md specification
        self.sensor_groups = {
            'presence_zones': [
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
                'binary_sensor.presence_stairs_up_ground_floor'
            ],
            'doors': [
                'binary_sensor.bathroom_door_sensor_contact',
                'binary_sensor.bedroom_door_sensor_contact',
                'binary_sensor.office_door_sensor_contact',
                'binary_sensor.guest_bedroom_door_sensor_contact',
                'binary_sensor.small_bathroom_door_sensor_contact'
            ],
            'climate': [
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
                'sensor.big_bath_env_sensor_humidity'
            ],
            'light_levels': [
                'sensor.bedroom_presence_light_level',
                'sensor.kitchen_pressence_light_level',
                'sensor.livingroom_pressence_light_level',
                'sensor.office_presence_light_level',
                'sensor.upper_hallway_pressence_light_level'
            ]
        }
        
        logger.info("Fixed system bootstrap initialized")
    
    async def bootstrap_system(self):
        """
        Complete bootstrap process with correct schema
        """
        
        print("="*60)
        print("🚀 HA INTENT PREDICTOR SYSTEM BOOTSTRAP (FIXED)")
        print("="*60)
        
        try:
            # Step 1: Initialize storage infrastructure
            print("\n1. 🏗️  Initializing storage infrastructure...")
            await self._initialize_storage()
            print("   ✓ Storage infrastructure completed")
            
            # Step 2: Create correct database schema
            print("\n2. 📊 Creating correct database schema...")
            await self._create_correct_database_schema()
            print("   ✓ Database schema created")
            
            # Step 3: Initialize learning components
            print("\n3. 🧠 Initializing adaptive learning components...")
            await self._initialize_learning()
            print("   ✓ Learning components completed")
            
            # Step 4: Import historical data
            print("\n4. 📊 Importing historical data...")
            success = await self._check_or_import_historical_data()
            if success:
                print("   ✓ Historical data ready")
            else:
                print("   ⚠️  Historical data import needs to be run separately")
            
            # Step 5: Configure combined living/kitchen space
            print("\n5. 🏠 Configuring unified living/kitchen space...")
            await self._configure_combined_spaces()
            print("   ✓ Combined spaces configured")
            
            # Step 6: Initialize person-specific learning
            print("\n6. 👥 Initializing person-specific learning...")
            await self._initialize_person_learning()
            print("   ✓ Person-specific learning initialized")
            
            # Step 7: Set up Home Assistant integration
            print("\n7. 🔌 Setting up Home Assistant integration...")
            await self._setup_ha_integration()
            print("   ✓ Home Assistant integration configured")
            
            # Step 8: Validate system
            print("\n8. ✅ Validating system readiness...")
            await self._validate_system()
            
            print("\n" + "="*60)
            print("🎉 BOOTSTRAP COMPLETED SUCCESSFULLY!")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Bootstrap failed: {e}")
            print(f"\n❌ BOOTSTRAP FAILED: {e}")
            raise
    
    async def _initialize_storage(self):
        """Initialize storage connections"""
        
        # Initialize TimescaleDB
        db_config = self.config.get('database.timescale')
        db_connection_string = f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        self.components['timeseries_db'] = TimescaleDBManager(db_connection_string)
        await self.components['timeseries_db'].initialize()
        
        # Initialize Redis
        redis_config = self.config.get('redis')
        redis_url = f"redis://{redis_config['host']}:{redis_config['port']}/{redis_config['db']}"
        self.components['feature_store'] = RedisFeatureStore(redis_url)
        await self.components['feature_store'].initialize()
        
        # Initialize model storage
        self.components['model_store'] = ModelStore(self.config.get('model_storage'))
        await self.components['model_store'].initialize()
    
    async def _create_correct_database_schema(self):
        """Create the correct database schema that matches the implementation"""
        
        async with self.components['timeseries_db'].engine.begin() as conn:
            from sqlalchemy import text
            
            # Execute each SQL statement individually to avoid multi-statement errors
            sql_statements = [
                # Enable TimescaleDB extension
                "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;",
                
                # Main sensor events table
                """CREATE TABLE IF NOT EXISTS sensor_events (
                    id BIGSERIAL,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    entity_id VARCHAR(255) NOT NULL,
                    state VARCHAR(50),
                    numeric_value DOUBLE PRECISION,
                    attributes JSONB,
                    room VARCHAR(100),
                    sensor_type VARCHAR(50),
                    zone_type VARCHAR(50),
                    zone_info JSONB,
                    person VARCHAR(50),
                    derived_features JSONB,
                    processed_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (timestamp, entity_id)
                );""",
                
                # Predictions table
                """CREATE TABLE IF NOT EXISTS predictions (
                    timestamp TIMESTAMPTZ NOT NULL,
                    room TEXT NOT NULL,
                    horizon_minutes INTEGER NOT NULL,
                    probability FLOAT NOT NULL,
                    uncertainty FLOAT NOT NULL,
                    confidence FLOAT NOT NULL,
                    model_name TEXT,
                    features JSONB,
                    metadata JSONB,
                    PRIMARY KEY (timestamp, room, horizon_minutes)
                );""",
                
                # Pattern discoveries table
                """CREATE TABLE IF NOT EXISTS pattern_discoveries (
                    timestamp TIMESTAMPTZ NOT NULL,
                    room TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    pattern_data JSONB NOT NULL,
                    significance_score FLOAT NOT NULL,
                    frequency INTEGER NOT NULL,
                    metadata JSONB,
                    PRIMARY KEY (timestamp, room, pattern_type)
                );""",
                
                # Model performance table
                """CREATE TABLE IF NOT EXISTS model_performance (
                    timestamp TIMESTAMPTZ NOT NULL,
                    model_name TEXT NOT NULL,
                    room TEXT NOT NULL,
                    horizon_minutes INTEGER NOT NULL,
                    accuracy FLOAT,
                    auc_score FLOAT,
                    precision_score FLOAT,
                    recall_score FLOAT,
                    sample_count INTEGER,
                    metadata JSONB,
                    PRIMARY KEY (timestamp, model_name, room, horizon_minutes)
                );""",
                
                # Room occupancy table
                """CREATE TABLE IF NOT EXISTS room_occupancy (
                    id BIGSERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    room VARCHAR(100) NOT NULL,
                    occupied BOOLEAN NOT NULL,
                    confidence DOUBLE PRECISION,
                    inference_method VARCHAR(100),
                    supporting_evidence JSONB,
                    person VARCHAR(50),
                    duration_minutes INTEGER,
                    processed_at TIMESTAMPTZ DEFAULT NOW()
                );""",
                
                # Discovered patterns table (for initial bootstrap patterns)
                """CREATE TABLE IF NOT EXISTS discovered_patterns (
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
                );"""
            ]
            
            # Execute each statement individually
            for sql_statement in sql_statements:
                try:
                    await conn.execute(text(sql_statement))
                    logger.info(f"Successfully executed SQL statement")
                except Exception as e:
                    logger.warning(f"SQL statement execution warning (may already exist): {e}")
        
        # Create hypertables in separate transaction
        async with self.components['timeseries_db'].engine.begin() as conn:
            hypertables = [
                ("sensor_events", "timestamp"),
                ("predictions", "timestamp"),
                ("pattern_discoveries", "timestamp"),
                ("model_performance", "timestamp"),
                ("room_occupancy", "timestamp")
            ]
            
            for table, time_column in hypertables:
                try:
                    await conn.execute(text(f"""
                        SELECT create_hypertable('{table}', '{time_column}',
                                               if_not_exists => TRUE,
                                               chunk_time_interval => INTERVAL '1 day');
                    """))
                    logger.info(f"{table} hypertable created successfully")
                except Exception as e:
                    logger.warning(f"{table} hypertable creation failed (may already exist): {e}")
        
        # Create indexes
        async with self.components['timeseries_db'].engine.begin() as conn:
            indices = [
                "CREATE INDEX IF NOT EXISTS idx_sensor_events_timestamp ON sensor_events (timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_sensor_events_entity_id ON sensor_events (entity_id, timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_sensor_events_room ON sensor_events (room, timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_predictions_room_time ON predictions (room, timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_pattern_discoveries_room ON pattern_discoveries (room, timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_room_occupancy_room ON room_occupancy (room, timestamp DESC);"
            ]
            
            for index_sql in indices:
                try:
                    await conn.execute(text(index_sql))
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")
    
    async def _initialize_learning(self):
        """Initialize learning components"""
        
        # Initialize pattern discovery
        self.components['pattern_discovery'] = PatternDiscovery()
        
        # Initialize main prediction engine
        predictor_config = self.config.config.copy()
        predictor_config['timeseries_db'] = self.components['timeseries_db']
        predictor_config['feature_store'] = self.components['feature_store']
        self.components['predictor'] = AdaptiveOccupancyPredictor(config=predictor_config)
        await self.components['predictor'].initialize()
        
        # Initialize cat detection
        self.components['cat_detector'] = AdaptiveCatDetector()
    
    async def _check_or_import_historical_data(self):
        """Check if historical data exists, suggest import if not"""
        
        async with self.components['timeseries_db'].engine.begin() as conn:
            from sqlalchemy import text
            
            result = await conn.execute(text("SELECT COUNT(*) FROM sensor_events"))
            count = result.fetchone()[0]
            
            if count > 100000:
                print(f"    - Found {count:,} existing events")
                return True
            else:
                print(f"    - Only {count:,} events found")
                print("    - Run: python scripts/historical_import.py --days 180")
                return False
    
    async def _configure_combined_spaces(self):
        """Configure unified living/kitchen space"""
        
        async with self.components['timeseries_db'].engine.begin() as conn:
            from sqlalchemy import text
            
            # Update sensor events
            await conn.execute(text("""
                UPDATE sensor_events 
                SET room = 'living_kitchen' 
                WHERE room IN ('livingroom', 'kitchen')
            """))
    
    async def _initialize_person_learning(self):
        """Initialize person-specific learning"""
        
        persons = ['anca', 'vladimir']
        
        for person in persons:
            person_data = {
                'initialized_at': datetime.now().isoformat(),
                'person': person,
                'learning_enabled': True
            }
            
            await self.components['feature_store'].store_model_state(
                f"person_learning_{person}_initialized",
                person_data
            )
    
    async def _setup_ha_integration(self):
        """Set up Home Assistant integration"""
        
        ha_api = self.components.get('ha_api')
        self.components['ha_integration'] = DynamicHAIntegration(
            ha_api=ha_api,
            timeseries_db=self.components['timeseries_db'],
            feature_store=self.components['feature_store']
        )
        await self.components['ha_integration'].initialize()
    
    async def _validate_system(self):
        """Validate system readiness"""
        
        # Test database
        stats = await self.components['timeseries_db'].get_database_stats()
        if stats:
            print("    - Database connection: OK")
        
        # Test Redis
        test_data = {"test": "data"}
        await self.components['feature_store'].store_model_state("test_key", test_data)
        retrieved = await self.components['feature_store'].get_model_state("test_key")
        if retrieved:
            print("    - Redis connection: OK")
        
        # Check for data
        if 'sensor_events' in stats:
            event_count = stats['sensor_events'].get('row_count', 0)
            print(f"    - Sensor events: {event_count:,}")
    
    async def cleanup(self):
        """Clean up resources"""
        
        for name, component in self.components.items():
            if hasattr(component, 'close'):
                await component.close()


async def main():
    """Main entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed HA Intent Predictor bootstrap')
    parser.add_argument('--config', type=str, default='config/system.yaml', 
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    bootstrap = FixedSystemBootstrap(args.config)
    
    try:
        await bootstrap.bootstrap_system()
    except Exception as e:
        logger.error(f"Bootstrap failed: {e}")
        sys.exit(1)
    finally:
        await bootstrap.cleanup()


if __name__ == "__main__":
    asyncio.run(main())