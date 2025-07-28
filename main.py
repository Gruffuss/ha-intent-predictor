#!/usr/bin/env python3
"""
Main application entry point for the HA Intent Predictor system.

This orchestrates all components according to CLAUDE.md specifications:
- Continuous data ingestion from Home Assistant
- Real-time ML prediction engine
- Adaptive learning without assumptions
- Home Assistant integration
"""

import asyncio
import logging
import signal
import sys
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import uvicorn
from fastapi import FastAPI
from aiokafka import AIOKafkaConsumer

from src.ingestion.ha_stream import HADataStream
from src.learning.adaptive_predictor import AdaptiveOccupancyPredictor
from src.integration.api import create_api_app
from src.integration.ha_publisher import DynamicHAIntegration
from src.integration.monitoring import PerformanceMonitor
from src.storage.timeseries_db import TimescaleDBManager
from src.storage.feature_store import RedisFeatureStore
from src.storage.model_store import ModelStore
from src.adaptation.drift_detection import DriftDetector
from config.config_loader import ConfigLoader

# Configure logging with fallback for Docker permissions
import os
handlers = [logging.StreamHandler()]
try:
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    handlers.append(logging.FileHandler('logs/ha_predictor.log'))
except (PermissionError, OSError):
    # Fall back to console only in Docker environments
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)

class HAIntentPredictorSystem:
    """
    Main system orchestrator following CLAUDE.md architecture.
    
    Coordinates all components for continuous learning and prediction
    without making assumptions about occupancy patterns.
    """
    
    def __init__(self, config_path: str = "config/system.yaml"):
        self.config = ConfigLoader(config_path)
        self.components = {}
        self.running = False
        self.tasks = []
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.shutdown())
    
    def _build_db_connection_string(self):
        """Build PostgreSQL connection string from config"""
        db_config = self.config.get('database.timescale')
        return f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    def _build_redis_url(self):
        """Build Redis connection URL from config"""
        redis_config = self.config.get('redis')
        if redis_config.get('password'):
            return f"redis://:{redis_config['password']}@{redis_config['host']}:{redis_config['port']}"
        else:
            return f"redis://{redis_config['host']}:{redis_config['port']}"
    
    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing HA Intent Predictor system...")
        
        try:
            # Initialize storage components
            await self._init_storage()
            
            # Initialize learning components
            await self._init_learning()
            
            # Initialize ingestion components
            await self._init_ingestion()
            
            # Initialize integration components
            await self._init_integration()
            
            # Initialize monitoring
            await self._init_monitoring()
            
            logger.info("System initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise
    
    async def _init_storage(self):
        """Initialize storage components"""
        logger.info("Initializing storage components...")
        
        # TimescaleDB for time series data
        self.components['timeseries_db'] = TimescaleDBManager(
            self._build_db_connection_string()
        )
        await self.components['timeseries_db'].initialize()
        
        # Redis for feature caching
        self.components['feature_store'] = RedisFeatureStore(
            self._build_redis_url()
        )
        await self.components['feature_store'].initialize()
        
        # Model versioning store
        self.components['model_store'] = ModelStore(
            self.config.get('model_storage')
        )
        await self.components['model_store'].initialize()
    
    async def _init_learning(self):
        """Initialize adaptive learning components"""
        logger.info("Initializing adaptive learning components...")
        
        # Main prediction engine - learns without assumptions
        predictor_config = {
            'timeseries_db': self.components['timeseries_db'],
            'feature_store': self.components['feature_store'],
            'model_store': self.components['model_store'],
            'cpu_limit': 80,
            'memory_limit': 6000
        }
        self.components['predictor'] = AdaptiveOccupancyPredictor(predictor_config)
        await self.components['predictor'].initialize()
        
        # TEMPORARILY DISABLED: Drift detection for continuous adaptation
        # self.components['drift_detector'] = DriftDetector()
        # await self.components['drift_detector'].initialize()
    
    async def _init_ingestion(self):
        """Initialize data ingestion components"""
        logger.info("Initializing data ingestion...")
        
        # Home Assistant data stream
        self.components['ha_stream'] = HADataStream(
            ha_config=self.config.get('home_assistant'),
            kafka_config=self.config.get('kafka'),
            timeseries_db=self.components['timeseries_db'],
            feature_store=self.components['feature_store']
        )
        await self.components['ha_stream'].initialize()
    
    async def _init_integration(self):
        """Initialize Home Assistant integration"""
        logger.info("Initializing HA integration...")
        
        # Dynamic entity creation and updates
        # Import the correct HA API class for entity management
        from src.integration.ha_publisher import HomeAssistantAPI
        ha_api = HomeAssistantAPI(
            ha_url=self.config.get('home_assistant.url'),
            token=self.config.get('home_assistant.token')
        )
        
        self.components['ha_publisher'] = DynamicHAIntegration(
            ha_api,
            predictor=self.components['predictor'],
            timeseries_db=self.components['timeseries_db'],
            feature_store=self.components['feature_store']
        )
        await self.components['ha_publisher'].initialize()
    
    async def _init_monitoring(self):
        """Initialize monitoring and metrics"""
        logger.info("Initializing monitoring...")
        
        self.components['monitor'] = PerformanceMonitor(
            metrics_config=self.config.get('monitoring')
        )
        await self.components['monitor'].initialize()
    
    async def start(self):
        """Start all system components"""
        logger.info("Starting HA Intent Predictor system...")
        
        self.running = True
        
        # Start background tasks
        self.tasks = [
            asyncio.create_task(self._run_data_ingestion()),
            asyncio.create_task(self._run_prediction_engine()),
            asyncio.create_task(self._run_drift_detection()),
            asyncio.create_task(self._run_ha_publisher()),
            asyncio.create_task(self._run_monitoring())
        ]
        
        logger.info("All components started successfully")
        
        # Wait for all tasks
        await asyncio.gather(*self.tasks, return_exceptions=True)
    
    async def _run_data_ingestion(self):
        """Run continuous data ingestion from Home Assistant"""
        logger.info("Starting data ingestion...")
        
        try:
            # Stream all sensor data continuously
            await self.components['ha_stream'].stream_all_sensors()
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            if self.running:
                await self.shutdown()
    
    async def _run_prediction_engine(self):
        """Run the adaptive prediction engine"""
        logger.info("Starting prediction engine...")
        
        try:
            # Create two tasks: learning loop AND event consumer
            learning_task = asyncio.create_task(
                self.components['predictor'].run_continuous_learning()
            )
            
            consumer_task = asyncio.create_task(
                self._run_kafka_consumer()
            )
            
            # Run both concurrently
            await asyncio.gather(learning_task, consumer_task)
            
        except Exception as e:
            logger.error(f"Prediction engine failed: {e}")
            if self.running:
                await self.shutdown()
    
    async def _run_kafka_consumer(self):
        """Consume events from Kafka for real-time learning"""
        kafka_config = self.config.get('kafka')
        
        consumer = AIOKafkaConsumer(
            'sensor_events',
            bootstrap_servers=kafka_config.get('bootstrap_servers', 'localhost:9092'),
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='ha_predictor_learning'
        )
        
        await consumer.start()
        logger.info("Kafka consumer started for real-time learning")
        
        try:
            async for msg in consumer:
                if not self.running:
                    break
                
                try:
                    event = msg.value
                    
                    # Process event through the learning pipeline
                    predictions = await self.components['predictor'].process_sensor_event(event)
                    
                    # Store predictions for later evaluation
                    if predictions and event.get('room'):
                        # Cache predictions for evaluation when ground truth arrives
                        await self._cache_predictions_for_evaluation(
                            event['room'], 
                            event.get('timestamp'), 
                            predictions
                        )
                    
                except Exception as e:
                    logger.error(f"Error processing event: {e}")
                    continue
                    
        finally:
            await consumer.stop()
            logger.info("Kafka consumer stopped")
    
    async def _cache_predictions_for_evaluation(self, room_id: str, timestamp: datetime, predictions: Dict[int, Dict]):
        """Cache predictions for later evaluation when ground truth arrives"""
        try:
            # Store predictions with TTL for later evaluation
            for horizon, prediction in predictions.items():
                cache_key = f"prediction_{room_id}_{horizon}_{timestamp.timestamp()}"
                await self.components['feature_store'].cache_prediction(
                    room_id,
                    horizon,
                    prediction,
                    ttl=horizon * 60 + 300  # TTL = horizon + 5 minutes buffer
                )
        except Exception as e:
            logger.error(f"Error caching predictions: {e}")
    
    async def _run_drift_detection(self):
        """Run drift detection for model adaptation"""
        logger.info("Drift detection DISABLED - skipping...")
        
        # TEMPORARILY DISABLED: Drift detection
        # try:
        #     await self.components['drift_detector'].monitor_drift(
        #         predictor=self.components['predictor']
        #     )
        # except Exception as e:
        #     logger.error(f"Drift detection failed: {e}")
    
    async def _run_ha_publisher(self):
        """Run Home Assistant entity publisher"""
        logger.info("Starting HA publisher...")
        
        try:
            await self.components['ha_publisher'].run_publisher()
        except Exception as e:
            logger.error(f"HA publisher failed: {e}")
    
    async def _run_monitoring(self):
        """Run system monitoring"""
        logger.info("Starting monitoring...")
        
        try:
            await self.components['monitor'].run_monitoring()
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown all components"""
        logger.info("Shutting down HA Intent Predictor system...")
        
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Shutdown components in reverse order
        shutdown_order = [
            'monitor',
            'ha_publisher', 
            # 'drift_detector',  # DISABLED
            'predictor',
            'ha_stream',
            'model_store',
            'feature_store',
            'timeseries_db'
        ]
        
        for component_name in shutdown_order:
            if component_name in self.components:
                try:
                    await self.components[component_name].shutdown()
                    logger.info(f"Shutdown {component_name}")
                except Exception as e:
                    logger.error(f"Error shutting down {component_name}: {e}")
        
        logger.info("System shutdown complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager"""
    # Startup
    system = HAIntentPredictorSystem()
    await system.initialize()
    
    # Store system instance for API access
    app.state.system = system
    
    # Start background tasks
    system_task = asyncio.create_task(system.start())
    
    yield
    
    # Shutdown
    await system.shutdown()
    system_task.cancel()


async def main():
    """Main entry point"""
    logger.info("Starting HA Intent Predictor System")
    
    try:
        # Create system instance
        system = HAIntentPredictorSystem()
        
        # Initialize all components
        await system.initialize()
        
        # Create and configure FastAPI app
        api_app = create_api_app(system)
        
        # Start API server and system concurrently
        config = system.config.get('api', {})
        server_config = uvicorn.Config(
            app=api_app,
            host=config.get('host', '0.0.0.0'),
            port=config.get('port', 8001),
            log_level=config.get('log_level', 'info'),
            lifespan='on'
        )
        
        server = uvicorn.Server(server_config)
        
        # Run system and API server concurrently
        await asyncio.gather(
            system.start(),
            server.serve()
        )
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure log directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Run the main application
    asyncio.run(main())