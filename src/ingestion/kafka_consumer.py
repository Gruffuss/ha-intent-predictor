"""
Kafka Consumer for processing sensor events.

Consumes events from Kafka topics and feeds them into the ML pipeline
for continuous learning and prediction.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from kafka import KafkaConsumer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)


class SensorEventConsumer:
    """
    Kafka consumer for processing sensor events.
    
    Processes events from the sensor_events topic and feeds them
    into the adaptive learning pipeline.
    """
    
    def __init__(self, kafka_config: Dict, event_processor: Callable = None):
        self.kafka_config = kafka_config
        self.event_processor = event_processor
        self.consumer = None
        self.running = False
        
        # Event statistics
        self.stats = {
            'events_processed': 0,
            'events_failed': 0,
            'last_event_time': None,
            'processing_errors': []
        }
        
        logger.info("Initialized Kafka sensor event consumer")
    
    async def initialize(self):
        """Initialize Kafka consumer"""
        try:
            # Create Kafka consumer
            self.consumer = KafkaConsumer(
                self.kafka_config['topics']['sensor_events'],
                bootstrap_servers=self.kafka_config['bootstrap_servers'],
                group_id=self.kafka_config['consumer']['group_id'],
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                **self.kafka_config['consumer']
            )
            
            logger.info("Kafka consumer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            raise
    
    async def start_consuming(self):
        """Start consuming events from Kafka"""
        if not self.consumer:
            raise RuntimeError("Consumer not initialized")
        
        self.running = True
        logger.info("Starting Kafka event consumption...")
        
        try:
            # Process events in a separate thread to avoid blocking
            import threading
            
            def consume_events():
                for message in self.consumer:
                    if not self.running:
                        break
                    
                    try:
                        # Process the event
                        event_data = message.value
                        
                        # Convert timestamp back to datetime
                        event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
                        
                        # Process the event
                        if self.event_processor:
                            asyncio.run_coroutine_threadsafe(
                                self.event_processor(event_data),
                                asyncio.get_event_loop()
                            )
                        
                        # Update statistics
                        self.stats['events_processed'] += 1
                        self.stats['last_event_time'] = datetime.now()
                        
                        logger.debug(f"Processed event: {event_data['entity_id']}")
                        
                    except Exception as e:
                        logger.error(f"Error processing event: {e}")
                        self.stats['events_failed'] += 1
                        self.stats['processing_errors'].append({
                            'error': str(e),
                            'timestamp': datetime.now(),
                            'event_data': event_data if 'event_data' in locals() else None
                        })
                        
                        # Keep only last 100 errors
                        if len(self.stats['processing_errors']) > 100:
                            self.stats['processing_errors'] = self.stats['processing_errors'][-100:]
            
            # Start consuming in background thread
            consume_thread = threading.Thread(target=consume_events)
            consume_thread.daemon = True
            consume_thread.start()
            
            # Wait for the thread to complete
            while self.running and consume_thread.is_alive():
                await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error in Kafka consumption: {e}")
            raise
    
    async def stop_consuming(self):
        """Stop consuming events"""
        self.running = False
        
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer stopped")
    
    def get_stats(self) -> Dict:
        """Get consumer statistics"""
        return self.stats.copy()


class PredictionConsumer:
    """
    Kafka consumer for processing prediction results.
    
    Consumes prediction results and sends them to Home Assistant
    or other downstream systems.
    """
    
    def __init__(self, kafka_config: Dict, prediction_handler: Callable = None):
        self.kafka_config = kafka_config
        self.prediction_handler = prediction_handler
        self.consumer = None
        self.running = False
        
        logger.info("Initialized Kafka prediction consumer")
    
    async def initialize(self):
        """Initialize Kafka consumer for predictions"""
        try:
            self.consumer = KafkaConsumer(
                self.kafka_config['topics']['predictions'],
                bootstrap_servers=self.kafka_config['bootstrap_servers'],
                group_id=f"{self.kafka_config['consumer']['group_id']}_predictions",
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                **self.kafka_config['consumer']
            )
            
            logger.info("Kafka prediction consumer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize prediction consumer: {e}")
            raise
    
    async def start_consuming(self):
        """Start consuming prediction results"""
        if not self.consumer:
            raise RuntimeError("Prediction consumer not initialized")
        
        self.running = True
        logger.info("Starting prediction consumption...")
        
        try:
            import threading
            
            def consume_predictions():
                for message in self.consumer:
                    if not self.running:
                        break
                    
                    try:
                        prediction_data = message.value
                        
                        # Process the prediction
                        if self.prediction_handler:
                            asyncio.run_coroutine_threadsafe(
                                self.prediction_handler(prediction_data),
                                asyncio.get_event_loop()
                            )
                        
                        logger.debug(f"Processed prediction: {prediction_data.get('room')}")
                        
                    except Exception as e:
                        logger.error(f"Error processing prediction: {e}")
            
            # Start consuming in background thread
            consume_thread = threading.Thread(target=consume_predictions)
            consume_thread.daemon = True
            consume_thread.start()
            
            # Wait for the thread to complete
            while self.running and consume_thread.is_alive():
                await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error in prediction consumption: {e}")
            raise
    
    async def stop_consuming(self):
        """Stop consuming predictions"""
        self.running = False
        
        if self.consumer:
            self.consumer.close()
            logger.info("Prediction consumer stopped")


class AnomalyConsumer:
    """
    Kafka consumer for processing anomaly detection results.
    
    Consumes anomaly alerts and handles them appropriately.
    """
    
    def __init__(self, kafka_config: Dict, anomaly_handler: Callable = None):
        self.kafka_config = kafka_config
        self.anomaly_handler = anomaly_handler
        self.consumer = None
        self.running = False
        
        logger.info("Initialized Kafka anomaly consumer")
    
    async def initialize(self):
        """Initialize Kafka consumer for anomalies"""
        try:
            self.consumer = KafkaConsumer(
                self.kafka_config['topics']['anomalies'],
                bootstrap_servers=self.kafka_config['bootstrap_servers'],
                group_id=f"{self.kafka_config['consumer']['group_id']}_anomalies",
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                **self.kafka_config['consumer']
            )
            
            logger.info("Kafka anomaly consumer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize anomaly consumer: {e}")
            raise
    
    async def start_consuming(self):
        """Start consuming anomaly alerts"""
        if not self.consumer:
            raise RuntimeError("Anomaly consumer not initialized")
        
        self.running = True
        logger.info("Starting anomaly consumption...")
        
        try:
            import threading
            
            def consume_anomalies():
                for message in self.consumer:
                    if not self.running:
                        break
                    
                    try:
                        anomaly_data = message.value
                        
                        # Process the anomaly
                        if self.anomaly_handler:
                            asyncio.run_coroutine_threadsafe(
                                self.anomaly_handler(anomaly_data),
                                asyncio.get_event_loop()
                            )
                        
                        logger.info(f"Processed anomaly: {anomaly_data.get('type')}")
                        
                    except Exception as e:
                        logger.error(f"Error processing anomaly: {e}")
            
            # Start consuming in background thread
            consume_thread = threading.Thread(target=consume_anomalies)
            consume_thread.daemon = True
            consume_thread.start()
            
            # Wait for the thread to complete
            while self.running and consume_thread.is_alive():
                await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error in anomaly consumption: {e}")
            raise
    
    async def stop_consuming(self):
        """Stop consuming anomalies"""
        self.running = False
        
        if self.consumer:
            self.consumer.close()
            logger.info("Anomaly consumer stopped")


class KafkaEventProcessor:
    """
    Orchestrates all Kafka consumers for the system.
    
    Manages the lifecycle of all consumers and coordinates
    event processing across the entire pipeline.
    """
    
    def __init__(self, kafka_config: Dict):
        self.kafka_config = kafka_config
        self.consumers = {}
        self.running = False
        
        logger.info("Initialized Kafka event processor")
    
    async def initialize(self, event_processor=None, prediction_handler=None, anomaly_handler=None):
        """Initialize all consumers"""
        try:
            # Create consumers
            self.consumers['sensor_events'] = SensorEventConsumer(
                self.kafka_config, 
                event_processor
            )
            
            self.consumers['predictions'] = PredictionConsumer(
                self.kafka_config,
                prediction_handler
            )
            
            self.consumers['anomalies'] = AnomalyConsumer(
                self.kafka_config,
                anomaly_handler
            )
            
            # Initialize all consumers
            for consumer in self.consumers.values():
                await consumer.initialize()
            
            logger.info("All Kafka consumers initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumers: {e}")
            raise
    
    async def start_all(self):
        """Start all consumers"""
        if not self.consumers:
            raise RuntimeError("Consumers not initialized")
        
        self.running = True
        logger.info("Starting all Kafka consumers...")
        
        try:
            # Start all consumers concurrently
            consume_tasks = [
                asyncio.create_task(consumer.start_consuming())
                for consumer in self.consumers.values()
            ]
            
            # Wait for all tasks to complete
            await asyncio.gather(*consume_tasks)
            
        except Exception as e:
            logger.error(f"Error starting Kafka consumers: {e}")
            raise
    
    async def stop_all(self):
        """Stop all consumers"""
        self.running = False
        
        # Stop all consumers
        for consumer in self.consumers.values():
            await consumer.stop_consuming()
        
        logger.info("All Kafka consumers stopped")
    
    def get_stats(self) -> Dict:
        """Get statistics from all consumers"""
        stats = {}
        
        for consumer_name, consumer in self.consumers.items():
            if hasattr(consumer, 'get_stats'):
                stats[consumer_name] = consumer.get_stats()
        
        return stats
    
    async def health_check(self) -> Dict:
        """Health check for all consumers"""
        health = {
            'status': 'healthy',
            'consumers': {}
        }
        
        for consumer_name, consumer in self.consumers.items():
            try:
                consumer_health = {
                    'status': 'running' if consumer.running else 'stopped',
                    'initialized': consumer.consumer is not None
                }
                
                if hasattr(consumer, 'get_stats'):
                    stats = consumer.get_stats()
                    consumer_health['stats'] = stats
                
                health['consumers'][consumer_name] = consumer_health
                
            except Exception as e:
                health['consumers'][consumer_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                health['status'] = 'degraded'
        
        return health