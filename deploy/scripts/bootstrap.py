#!/usr/bin/env python3
"""
HA Intent Predictor Bootstrap Script
Initializes the system with historical data and starts learning
"""

import asyncio
import logging
import sys
import yaml
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import click
import json

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.timeseries_db import TimescaleDBConnection
from src.storage.feature_store import FeatureStore
from src.ingestion.ha_stream import HADataImporter
from src.learning.pattern_discovery import PatternDiscovery
from src.learning.online_models import ContinuousLearningSystem
from src.integration.ha_publisher import HAIntegration

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemBootstrap:
    """Bootstrap the HA Intent Predictor system"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.db = None
        self.feature_store = None
        self.ha_importer = None
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    async def initialize_database(self):
        """Initialize database connections"""
        logger.info("Initializing database connections...")
        
        db_config = self.config['database']
        self.db = TimescaleDBConnection(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['name'],
            user=db_config['user'],
            password=db_config['password']
        )
        
        await self.db.connect()
        logger.info("Database connection established")
        
        # Initialize feature store
        redis_config = self.config['redis']
        self.feature_store = FeatureStore(
            host=redis_config['host'],
            port=redis_config['port'],
            db=redis_config['db'],
            password=redis_config.get('password')
        )
        
        await self.feature_store.connect()
        logger.info("Feature store connection established")
    
    async def setup_home_assistant_importer(self):
        """Setup Home Assistant data importer"""
        logger.info("Setting up Home Assistant data importer...")
        
        ha_config = self.config['home_assistant']
        self.ha_importer = HADataImporter(
            ha_url=ha_config['url'],
            ha_token=ha_config['token'],
            sensor_config=ha_config['sensors'],
            db_connection=self.db
        )
        
        # Test connection
        is_connected = await self.ha_importer.test_connection()
        if not is_connected:
            raise Exception("Failed to connect to Home Assistant")
        
        logger.info("Home Assistant connection verified")
    
    async def import_historical_data(self, days: int = 180, batch_size: int = 1000):
        """Import historical data from Home Assistant"""
        logger.info(f"Starting historical data import for {days} days...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Get all configured sensors
        all_sensors = []
        for sensor_type, sensors in self.config['home_assistant']['sensors'].items():
            for sensor in sensors:
                all_sensors.append(sensor['entity_id'])
        
        logger.info(f"Importing data for {len(all_sensors)} sensors...")
        
        # Import in batches to avoid overwhelming the system
        total_events = 0
        for i in range(0, len(all_sensors), batch_size):
            sensor_batch = all_sensors[i:i + batch_size]
            logger.info(f"Processing sensor batch {i//batch_size + 1}/{(len(all_sensors) + batch_size - 1)//batch_size}")
            
            try:
                events = await self.ha_importer.import_sensor_history(
                    entity_ids=sensor_batch,
                    start_time=start_time,
                    end_time=end_time
                )
                total_events += events
                logger.info(f"Imported {events} events from batch")
                
                # Small delay to avoid overwhelming HA
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to import batch: {e}")
                continue
        
        logger.info(f"Historical data import completed. Total events: {total_events}")
        return total_events
    
    async def configure_room_mappings(self):
        """Configure room mappings and unified spaces"""
        logger.info("Configuring room mappings...")
        
        # Handle combined living/kitchen space
        await self.db.execute("""
            -- Create or update room mapping for unified living/kitchen space
            CREATE OR REPLACE VIEW unified_living_kitchen AS
            SELECT 
                timestamp,
                'living_kitchen' as room,
                entity_id,
                state,
                attributes,
                sensor_type,
                zone_info
            FROM raw_data.sensor_events 
            WHERE room IN ('livingroom', 'kitchen', 'living_kitchen');
        """)
        
        # Create room state tracking
        await self.db.execute("""
            CREATE OR REPLACE FUNCTION update_room_occupancy() RETURNS TRIGGER AS $$
            BEGIN
                -- Logic to determine room occupancy from sensor events
                -- This will be enhanced by the pattern discovery system
                INSERT INTO raw_data.room_occupancy (timestamp, room, occupied, detected_by)
                VALUES (NEW.timestamp, NEW.room, 
                       CASE WHEN NEW.state = 'on' THEN true ELSE false END,
                       NEW.entity_id)
                ON CONFLICT (room, timestamp) DO UPDATE SET
                    occupied = EXCLUDED.occupied,
                    detected_by = EXCLUDED.detected_by;
                
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            
            DROP TRIGGER IF EXISTS trigger_update_room_occupancy ON raw_data.sensor_events;
            CREATE TRIGGER trigger_update_room_occupancy
                AFTER INSERT ON raw_data.sensor_events
                FOR EACH ROW EXECUTE FUNCTION update_room_occupancy();
        """)
        
        logger.info("Room mappings configured")
    
    async def initialize_pattern_discovery(self):
        """Initialize pattern discovery system"""
        logger.info("Initializing pattern discovery system...")
        
        pattern_discovery = PatternDiscovery(
            db_connection=self.db,
            feature_store=self.feature_store
        )
        
        # Configure rooms for pattern discovery
        rooms = list(self.config['rooms'].keys())
        
        logger.info(f"Starting pattern discovery for rooms: {rooms}")
        
        # Discover patterns for each room
        for room in rooms:
            logger.info(f"Discovering patterns for {room}...")
            try:
                patterns = await pattern_discovery.discover_room_patterns(room)
                logger.info(f"Discovered {len(patterns)} patterns for {room}")
            except Exception as e:
                logger.error(f"Pattern discovery failed for {room}: {e}")
        
        logger.info("Pattern discovery initialization completed")
    
    async def initialize_learning_system(self):
        """Initialize the continuous learning system"""
        logger.info("Initializing continuous learning system...")
        
        learning_system = ContinuousLearningSystem(
            db_connection=self.db,
            feature_store=self.feature_store,
            config=self.config['ml']
        )
        
        # Initialize models for each room and horizon
        rooms = list(self.config['rooms'].keys())
        horizons = self.config['ml']['prediction_horizons']
        
        for room in rooms:
            for horizon in horizons:
                logger.info(f"Initializing model for {room} at {horizon}min horizon...")
                try:
                    await learning_system.initialize_room_model(room, horizon)
                except Exception as e:
                    logger.error(f"Model initialization failed for {room}/{horizon}: {e}")
        
        logger.info("Learning system initialization completed")
    
    async def setup_home_assistant_integration(self):
        """Setup Home Assistant integration entities"""
        logger.info("Setting up Home Assistant integration...")
        
        ha_integration = HAIntegration(
            ha_url=self.config['home_assistant']['url'],
            ha_token=self.config['home_assistant']['token']
        )
        
        # Create prediction entities for each room and horizon
        rooms = list(self.config['rooms'].keys())
        horizons = self.config['ml']['prediction_horizons']
        
        for room in rooms:
            for horizon in horizons:
                entity_id = f"sensor.occupancy_prediction_{room}_{horizon}min"
                
                try:
                    await ha_integration.create_prediction_entity(
                        entity_id=entity_id,
                        room=room,
                        horizon=horizon,
                        friendly_name=f"{room.title()} Occupancy Prediction ({horizon}min)"
                    )
                    logger.info(f"Created HA entity: {entity_id}")
                except Exception as e:
                    logger.error(f"Failed to create HA entity {entity_id}: {e}")
        
        # Create helper entities
        for room in rooms:
            try:
                # Occupancy trend entity
                trend_entity = f"sensor.occupancy_trend_{room}"
                await ha_integration.create_trend_entity(trend_entity, room)
                
                # Anomaly detection entity
                anomaly_entity = f"binary_sensor.occupancy_anomaly_{room}"
                await ha_integration.create_anomaly_entity(anomaly_entity, room)
                
                logger.info(f"Created helper entities for {room}")
            except Exception as e:
                logger.error(f"Failed to create helper entities for {room}: {e}")
        
        logger.info("Home Assistant integration setup completed")
    
    async def verify_system_health(self):
        """Verify system health after bootstrap"""
        logger.info("Verifying system health...")
        
        health_checks = {
            'database': False,
            'feature_store': False,
            'home_assistant': False,
            'data_availability': False
        }
        
        # Check database
        try:
            result = await self.db.fetch_one("SELECT COUNT(*) as count FROM raw_data.sensor_events")
            if result and result['count'] > 0:
                health_checks['database'] = True
                logger.info(f"Database healthy: {result['count']} sensor events")
            else:
                logger.warning("Database empty or inaccessible")
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
        
        # Check feature store
        try:
            await self.feature_store.set("health_check", "ok", ttl=60)
            value = await self.feature_store.get("health_check")
            if value == "ok":
                health_checks['feature_store'] = True
                logger.info("Feature store healthy")
            else:
                logger.warning("Feature store not responding correctly")
        except Exception as e:
            logger.error(f"Feature store health check failed: {e}")
        
        # Check Home Assistant
        try:
            is_connected = await self.ha_importer.test_connection()
            if is_connected:
                health_checks['home_assistant'] = True
                logger.info("Home Assistant connection healthy")
            else:
                logger.warning("Home Assistant not accessible")
        except Exception as e:
            logger.error(f"Home Assistant health check failed: {e}")
        
        # Check data availability
        try:
            rooms_with_data = await self.db.fetch_all(
                "SELECT DISTINCT room, COUNT(*) as events FROM raw_data.sensor_events GROUP BY room"
            )
            if rooms_with_data and len(rooms_with_data) > 0:
                health_checks['data_availability'] = True
                logger.info(f"Data available for {len(rooms_with_data)} rooms")
                for row in rooms_with_data:
                    logger.info(f"  {row['room']}: {row['events']} events")
            else:
                logger.warning("No data available in any rooms")
        except Exception as e:
            logger.error(f"Data availability check failed: {e}")
        
        # Summary
        healthy_components = sum(health_checks.values())
        total_components = len(health_checks)
        
        logger.info(f"System health: {healthy_components}/{total_components} components healthy")
        
        if healthy_components == total_components:
            logger.info("🎉 System bootstrap completed successfully!")
        elif healthy_components >= total_components * 0.75:
            logger.warning("⚠️ System bootstrap completed with warnings")
        else:
            logger.error("❌ System bootstrap completed with errors")
        
        return health_checks
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.db:
            await self.db.disconnect()
        if self.feature_store:
            await self.feature_store.disconnect()


@click.command()
@click.option('--config', '-c', default='/opt/ha-intent-predictor/config/app.yaml',
              help='Path to configuration file')
@click.option('--days', '-d', default=180, help='Days of historical data to import')
@click.option('--batch-size', '-b', default=1000, help='Batch size for data import')
@click.option('--skip-import', is_flag=True, help='Skip historical data import')
@click.option('--verify-only', is_flag=True, help='Only run system verification')
async def main(config: str, days: int, batch_size: int, skip_import: bool, verify_only: bool):
    """Bootstrap the HA Intent Predictor system"""
    
    bootstrap = SystemBootstrap(config)
    
    try:
        if verify_only:
            logger.info("Running system verification only...")
            await bootstrap.initialize_database()
            await bootstrap.setup_home_assistant_importer()
            health = await bootstrap.verify_system_health()
            return
        
        logger.info("Starting HA Intent Predictor bootstrap...")
        logger.info(f"Configuration: {config}")
        logger.info(f"Historical data: {days} days")
        logger.info(f"Batch size: {batch_size}")
        
        # Initialize core components
        await bootstrap.initialize_database()
        await bootstrap.setup_home_assistant_importer()
        await bootstrap.configure_room_mappings()
        
        # Import historical data
        if not skip_import:
            total_events = await bootstrap.import_historical_data(days, batch_size)
            if total_events == 0:
                logger.warning("No historical data imported - continuing with empty system")
        else:
            logger.info("Skipping historical data import")
        
        # Initialize ML components
        await bootstrap.initialize_pattern_discovery()
        await bootstrap.initialize_learning_system()
        
        # Setup HA integration
        await bootstrap.setup_home_assistant_integration()
        
        # Verify system health
        health = await bootstrap.verify_system_health()
        
        logger.info("Bootstrap completed!")
        
    except Exception as e:
        logger.error(f"Bootstrap failed: {e}")
        raise
    finally:
        await bootstrap.cleanup()


if __name__ == "__main__":
    asyncio.run(main())