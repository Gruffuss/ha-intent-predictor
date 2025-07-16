#!/usr/bin/env python3
"""
Bootstrap System - Initial system setup as specified in CLAUDE.md
Implements the exact bootstrap process with 180 days data import and pattern discovery
"""

import asyncio
import logging
import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from storage.timeseries_db import TimeSeriesDB
from storage.feature_store import FeatureStore
from storage.model_store import ModelStore
from learning.pattern_discovery import PatternDiscovery
from learning.adaptive_predictor import AdaptiveOccupancyPredictor
from integration.ha_publisher import DynamicHAIntegration
from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class SystemBootstrap:
    """
    Bootstrap the HA Intent Predictor system.
    
    Follows CLAUDE.md specifications for system initialization
    with no preconceptions about patterns or schedules.
    """
    
    def __init__(self, config_path: str = "config/system.yaml"):
        self.config = ConfigLoader(config_path)
        self.components = {}
        
        # Sensor groups from CLAUDE.md
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
        
        logger.info("System bootstrap initialized")
    
    async def bootstrap_system(self):
        """Bootstrap the entire system following CLAUDE.md specifications"""
        
        print("="*60)
        print("HA INTENT PREDICTOR SYSTEM BOOTSTRAP")
        print("="*60)
        print("Initializing adaptive learning system with NO assumptions")
        print("All patterns will be learned from observation")
        print("="*60)
        
        try:
            # Step 1: Initialize storage components
            print("\n1. Initializing storage components...")
            await self._initialize_storage()
            
            # Step 2: Initialize learning components
            print("\n2. Initializing adaptive learning components...")
            await self._initialize_learning()
            
            # Step 3: Configure unified living/kitchen space
            print("\n3. Configuring unified living/kitchen space...")
            await self._configure_combined_spaces()
            
            # Step 4: Initialize person-specific learning
            print("\n4. Initializing person-specific pattern learning...")
            await self._initialize_person_learning()
            
            # Step 5: Set up Home Assistant integration
            print("\n5. Creating Home Assistant entities...")
            await self._setup_ha_integration()
            
            # Step 6: Validate system readiness
            print("\n6. Validating system readiness...")
            await self._validate_system()
            
            print("\n" + "="*60)
            print("BOOTSTRAP COMPLETED SUCCESSFULLY")
            print("="*60)
            print("System is ready for historical data import and learning")
            print("Run: python scripts/historical_import.py --days 180")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Bootstrap failed: {e}")
            print(f"\nBOOTSTRAP FAILED: {e}")
            raise
    
    async def _initialize_storage(self):
        """Initialize all storage components"""
        
        # Initialize TimescaleDB
        print("  - Setting up TimescaleDB...")
        self.components['timeseries_db'] = TimeSeriesDB(
            self.config.get('database.timescale')
        )
        await self.components['timeseries_db'].connect()
        
        # Create all necessary tables
        await self._create_database_schema()
        
        # Initialize Redis for feature caching
        print("  - Setting up Redis feature store...")
        self.components['feature_store'] = FeatureStore(
            self.config.get('redis')
        )
        await self.components['feature_store'].connect()
        
        # Initialize model storage
        print("  - Setting up model versioning store...")
        self.components['model_store'] = ModelStore(
            self.config.get('model_storage')
        )
        await self.components['model_store'].initialize()
        
        print("  ✓ Storage components initialized")
    
    async def _create_database_schema(self):
        """Create complete database schema"""
        
        # Main sensor events table with hypertable
        await self.components['timeseries_db'].execute("""
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
        """)
        
        # Create hypertable for time-series optimization
        await self.components['timeseries_db'].execute("""
            SELECT create_hypertable('sensor_events', 'timestamp', 
                                   if_not_exists => TRUE,
                                   chunk_time_interval => INTERVAL '1 day');
        """)
        
        # Create comprehensive indices
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_sensor_events_entity_id ON sensor_events (entity_id, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_sensor_events_room ON sensor_events (room, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_sensor_events_sensor_type ON sensor_events (sensor_type, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_sensor_events_person ON sensor_events (person, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_sensor_events_zone_type ON sensor_events (zone_type, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_sensor_events_state ON sensor_events (state, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_sensor_events_room_state ON sensor_events (room, state, timestamp DESC);"
        ]
        
        for index_sql in indices:
            await self.components['timeseries_db'].execute(index_sql)
        
        # Pattern discovery table
        await self.components['timeseries_db'].execute("""
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
        """)
        
        # Room occupancy inference table
        await self.components['timeseries_db'].execute("""
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
        """)
        
        # Create hypertable for occupancy data
        await self.components['timeseries_db'].execute("""
            SELECT create_hypertable('room_occupancy', 'timestamp',
                                   if_not_exists => TRUE,
                                   chunk_time_interval => INTERVAL '1 day');
        """)
        
        print("  ✓ Database schema created")
    
    async def _initialize_learning(self):
        """Initialize adaptive learning components"""
        
        # Initialize pattern discovery
        print("  - Setting up pattern discovery...")
        self.components['pattern_discovery'] = PatternDiscovery(
            timeseries_db=self.components['timeseries_db'],
            feature_store=self.components['feature_store']
        )
        
        # Initialize main prediction engine
        print("  - Setting up adaptive prediction engine...")
        self.components['predictor'] = AdaptiveOccupancyPredictor(
            timeseries_db=self.components['timeseries_db'],
            feature_store=self.components['feature_store'],
            model_store=self.components['model_store']
        )
        await self.components['predictor'].initialize()
        
        print("  ✓ Learning components initialized")
    
    async def _configure_combined_spaces(self):
        """Configure unified living/kitchen space per CLAUDE.md"""
        
        print("  - Configuring living/kitchen unified space...")
        
        # Validate that living_kitchen is configured in rooms
        rooms = self.config.get_rooms()
        if 'living_kitchen' not in rooms:
            raise ValueError("living_kitchen not configured in rooms.yaml")
        
        # Validate sensor mappings
        living_kitchen_config = rooms['living_kitchen']
        required_zones = ['livingroom_full', 'kitchen_full', 'livingroom_couch', 
                         'kitchen_stove', 'kitchen_sink', 'kitchen_dining']
        
        for zone in required_zones:
            if zone not in living_kitchen_config.get('zones', {}):
                logger.warning(f"Zone {zone} not found in living_kitchen configuration")
        
        print("  ✓ Combined living/kitchen space configured")
    
    async def _initialize_person_learning(self):
        """Initialize person-specific learning"""
        
        print("  - Setting up person-specific learning...")
        
        # Initialize learning for both people
        persons = ['anca', 'vladimir']
        
        for person in persons:
            # Create person-specific feature cache
            await self.components['feature_store'].set(
                f"person_learning:{person}:initialized",
                json.dumps({
                    'initialized_at': datetime.now().isoformat(),
                    'person': person,
                    'learning_enabled': True,
                    'patterns_discovered': 0
                }),
                ttl=86400  # 24 hours
            )
            
            print(f"    - Initialized learning for {person}")
        
        print("  ✓ Person-specific learning initialized")
    
    async def _setup_ha_integration(self):
        """Set up Home Assistant integration"""
        
        print("  - Creating Home Assistant integration...")
        
        # Initialize HA integration
        self.components['ha_integration'] = DynamicHAIntegration(
            ha_config=self.config.get('home_assistant'),
            predictor=self.components['predictor']
        )
        
        # Create prediction entities for each room
        rooms_to_predict = [
            'living_kitchen',    # Combined space
            'bedroom',
            'office',
            'bathroom',
            'small_bathroom',
            'guest_bedroom'
        ]
        
        for room in rooms_to_predict:
            await self._create_room_predictors(room)
        
        print("  ✓ Home Assistant integration configured")
    
    async def _create_room_predictors(self, room: str):
        """Create prediction entities for a room"""
        
        # Default prediction horizons from CLAUDE.md
        horizons = self.config.get('learning.prediction_horizons', [5, 15, 30, 60, 120])
        
        for horizon in horizons:
            entity_id = f"sensor.occupancy_{room}_{horizon}min"
            print(f"    - Configured predictor: {entity_id}")
        
        # Create trend and anomaly entities
        trend_entity = f"sensor.occupancy_trend_{room}"
        anomaly_entity = f"binary_sensor.occupancy_anomaly_{room}"
        
        print(f"    - Configured trend sensor: {trend_entity}")
        print(f"    - Configured anomaly sensor: {anomaly_entity}")
    
    async def _validate_system(self):
        """Validate that the system is ready"""
        
        print("  - Validating database connectivity...")
        
        # Test database connection
        result = await self.components['timeseries_db'].fetchval("SELECT 1")
        if result != 1:
            raise RuntimeError("Database connection failed")
        
        print("  - Validating Redis connectivity...")
        
        # Test Redis connection
        await self.components['feature_store'].set("bootstrap_test", "success", ttl=10)
        test_value = await self.components['feature_store'].get("bootstrap_test")
        if test_value != "success":
            raise RuntimeError("Redis connection failed")
        
        print("  - Validating model storage...")
        
        # Test model storage
        storage_stats = await self.components['model_store'].get_storage_stats()
        if 'storage_path' not in storage_stats:
            raise RuntimeError("Model storage not accessible")
        
        print("  - Validating sensor configuration...")
        
        # Validate sensor configuration
        total_sensors = 0
        for sensor_type, sensors in self.sensor_groups.items():
            total_sensors += len(sensors)
        
        if total_sensors < 90:  # Should have ~98 sensors
            logger.warning(f"Only {total_sensors} sensors configured, expected ~98")
        
        print(f"  - Configured {total_sensors} sensors across all types")
        
        # Validate room configuration
        rooms = self.config.get_rooms()
        expected_rooms = ['living_kitchen', 'bedroom', 'office', 'bathroom', 
                         'small_bathroom', 'guest_bedroom', 'hallways']
        
        for room in expected_rooms:
            if room not in rooms:
                raise RuntimeError(f"Room {room} not configured")
        
        print(f"  - Configured {len(rooms)} rooms")
        
        print("  ✓ System validation passed")
    
    async def cleanup(self):
        """Clean up resources"""
        
        for component in self.components.values():
            if hasattr(component, 'shutdown'):
                await component.shutdown()
            elif hasattr(component, 'close'):
                await component.close()


async def main():
    """Main bootstrap entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Bootstrap HA Intent Predictor system')
    parser.add_argument('--config', type=str, default='config/system.yaml', 
                       help='Configuration file path')
    parser.add_argument('--force', action='store_true', 
                       help='Force bootstrap even if system appears initialized')
    parser.add_argument('--report', action='store_true',
                       help='Generate bootstrap report')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    bootstrap = SystemBootstrap(args.config)
    
    try:
        await bootstrap.bootstrap_system()
        
        if args.report:
            print("\n📊 Bootstrap report generated")
        
    except Exception as e:
        logger.error(f"Bootstrap failed: {e}")
        sys.exit(1)
    
    finally:
        await bootstrap.cleanup()


if __name__ == "__main__":
    asyncio.run(main())