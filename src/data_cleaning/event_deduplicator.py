"""
Event Deduplicator for Home Assistant Sensor Data

High-performance deduplication system that:
1. Removes duplicate consecutive state events (same entity_id with same state)
2. Keeps only actual state changes (on→off, off→on transitions)
3. Processes data in chunks to handle 1.1M+ events efficiently
4. Maintains data integrity and proper logging
5. Provides metrics for duplicate removal rates

Built for TimescaleDB with async/await performance optimization.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, and_, or_
from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationMetrics:
    """Metrics for deduplication process"""
    total_events_processed: int = 0
    duplicate_events_removed: int = 0
    state_transitions_kept: int = 0
    processing_time_seconds: float = 0.0
    chunks_processed: int = 0
    entities_processed: Set[str] = None
    
    def __post_init__(self):
        if self.entities_processed is None:
            self.entities_processed = set()
    
    @property
    def duplicate_rate(self) -> float:
        """Calculate duplicate removal rate"""
        if self.total_events_processed == 0:
            return 0.0
        return (self.duplicate_events_removed / self.total_events_processed) * 100.0
    
    @property
    def events_per_second(self) -> float:
        """Calculate processing rate"""
        if self.processing_time_seconds == 0:
            return 0.0
        return self.total_events_processed / self.processing_time_seconds


@dataclass
class EntityState:
    """Track current state for each entity"""
    entity_id: str
    last_state: str
    last_timestamp: datetime
    consecutive_count: int = 1


class EventDeduplicator:
    """
    High-performance event deduplicator for Home Assistant sensor data.
    
    Features:
    - Chunk-based processing for memory efficiency
    - State transition detection
    - Duplicate removal with configurable thresholds
    - Comprehensive metrics and logging
    - Async database operations for performance
    """
    
    def __init__(self, 
                 connection_string: str = None,
                 chunk_size: int = 10000,
                 max_time_window_seconds: int = 5,
                 min_state_duration_seconds: int = 1):
        """
        Initialize the deduplicator.
        
        Args:
            connection_string: Database connection string (defaults to config)
            chunk_size: Number of events to process per chunk
            max_time_window_seconds: Maximum time window for considering duplicates
            min_state_duration_seconds: Minimum duration a state must be held
        """
        self.chunk_size = chunk_size
        self.max_time_window_seconds = max_time_window_seconds
        self.min_state_duration_seconds = min_state_duration_seconds
        
        # Initialize database connection
        if connection_string is None:
            config = ConfigLoader()
            db_config = config.get("database.timescale")
            self.connection_string = (
                f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
        else:
            self.connection_string = connection_string
            
        self.engine = None
        self.session_factory = None
        self.initialized = False
        
        # State tracking for deduplication
        self.entity_states: Dict[str, EntityState] = {}
        self.metrics = DeduplicationMetrics()
        self.has_previous_state = False  # Will be set during schema verification
        
    async def initialize(self):
        """Initialize database connection"""
        try:
            self.engine = create_async_engine(
                self.connection_string,
                echo=False,
                pool_size=20,
                max_overflow=40,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            self.session_factory = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Verify connection and schema
            await self._verify_schema()
            
            self.initialized = True
            logger.info("EventDeduplicator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize EventDeduplicator: {e}")
            raise
    
    async def _verify_schema(self):
        """Verify that required tables and columns exist"""
        async with self.session_factory() as session:
            try:
                # Check if sensor_events table exists and has required columns
                result = await session.execute(text("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'sensor_events' 
                    AND table_schema = 'public'
                    ORDER BY ordinal_position
                """))
                
                columns = {row.column_name: row.data_type for row in result.fetchall()}
                
                required_columns = ['timestamp', 'entity_id', 'state']
                missing_columns = [col for col in required_columns if col not in columns]
                
                if missing_columns:
                    logger.error(f"Critical columns missing in sensor_events: {missing_columns}")
                    raise ValueError(f"Required columns missing: {missing_columns}")
                
                # Check for previous_state column (optional, but helpful for optimization)
                self.has_previous_state = 'previous_state' in columns
                if self.has_previous_state:
                    logger.info("Schema verification successful (with previous_state column)")
                else:
                    logger.warning("Schema verification successful (missing previous_state column - will calculate on-the-fly)")
                    
            except Exception as e:
                logger.error(f"Schema verification failed: {e}")
                raise
    
    async def deduplicate_historical_data(self, 
                                        start_date: datetime = None,
                                        end_date: datetime = None,
                                        entity_ids: List[str] = None,
                                        dry_run: bool = False) -> DeduplicationMetrics:
        """
        Deduplicate historical sensor data.
        
        Args:
            start_date: Start date for processing (defaults to 180 days ago)
            end_date: End date for processing (defaults to now)
            entity_ids: Specific entity IDs to process (defaults to all)
            dry_run: If True, only analyze without making changes
            
        Returns:
            DeduplicationMetrics with processing results
        """
        if not self.initialized:
            await self.initialize()
        
        # Set default date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=180)
        
        logger.info(f"Starting deduplication from {start_date} to {end_date}")
        logger.info(f"Chunk size: {self.chunk_size}, Dry run: {dry_run}")
        
        start_time = datetime.now()
        self.metrics = DeduplicationMetrics()
        
        try:
            # Process data in time-based chunks
            await self._process_time_chunks(start_date, end_date, entity_ids, dry_run)
            
            # Calculate final metrics
            self.metrics.processing_time_seconds = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Deduplication completed:")
            logger.info(f"  Total events processed: {self.metrics.total_events_processed:,}")
            logger.info(f"  Duplicate events removed: {self.metrics.duplicate_events_removed:,}")
            logger.info(f"  State transitions kept: {self.metrics.state_transitions_kept:,}")
            logger.info(f"  Duplicate rate: {self.metrics.duplicate_rate:.2f}%")
            logger.info(f"  Processing rate: {self.metrics.events_per_second:.1f} events/sec")
            logger.info(f"  Entities processed: {len(self.metrics.entities_processed)}")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Deduplication failed: {e}")
            raise
    
    async def _process_time_chunks(self, 
                                 start_date: datetime, 
                                 end_date: datetime,
                                 entity_ids: List[str] = None,
                                 dry_run: bool = False):
        """Process data in time-based chunks to manage memory usage"""
        
        # Calculate optimal chunk duration based on expected event volume
        total_duration = end_date - start_date
        chunk_duration_hours = max(1, min(24, self.chunk_size // 1000))  # Adaptive chunk size
        chunk_duration = timedelta(hours=chunk_duration_hours)
        
        current_start = start_date
        chunk_number = 0
        
        while current_start < end_date:
            chunk_number += 1
            current_end = min(current_start + chunk_duration, end_date)
            
            logger.info(f"Processing chunk {chunk_number}: {current_start} to {current_end}")
            
            try:
                await self._process_chunk(current_start, current_end, entity_ids, dry_run)
                self.metrics.chunks_processed += 1
                
            except Exception as e:
                logger.error(f"Failed to process chunk {chunk_number}: {e}")
                # Continue with next chunk rather than failing entirely
                
            current_start = current_end
            
            # Small delay to prevent overwhelming the database
            await asyncio.sleep(0.1)
    
    async def _process_chunk(self, 
                           start_time: datetime, 
                           end_time: datetime,
                           entity_ids: List[str] = None,
                           dry_run: bool = False):
        """Process a single time chunk of data"""
        
        async with self.session_factory() as session:
            # Build query for this chunk - conditionally include previous_state
            if self.has_previous_state:
                query = """
                    SELECT timestamp, entity_id, state, previous_state, room, sensor_type
                    FROM sensor_events
                    WHERE timestamp >= :start_time AND timestamp < :end_time
                """
            else:
                query = """
                    SELECT timestamp, entity_id, state, room, sensor_type
                    FROM sensor_events
                    WHERE timestamp >= :start_time AND timestamp < :end_time
                """
            params = {'start_time': start_time, 'end_time': end_time}
            
            if entity_ids:
                query += " AND entity_id = ANY(:entity_ids)"
                params['entity_ids'] = entity_ids
            
            query += " ORDER BY entity_id, timestamp"
            
            # Fetch chunk data
            result = await session.execute(text(query), params)
            events = result.fetchall()
            
            if not events:
                logger.debug(f"No events found in chunk {start_time} to {end_time}")
                return
            
            logger.debug(f"Processing {len(events)} events in chunk")
            
            # Group events by entity for sequential processing
            entity_events = defaultdict(list)
            for event in events:
                entity_events[event.entity_id].append(event)
            
            # Process each entity's events sequentially
            events_to_delete = []
            events_to_update = []
            
            for entity_id, entity_event_list in entity_events.items():
                self.metrics.entities_processed.add(entity_id)
                duplicates, updates = await self._process_entity_events(entity_event_list)
                events_to_delete.extend(duplicates)
                if self.has_previous_state:
                    events_to_update.extend(updates)
            
            self.metrics.total_events_processed += len(events)
            self.metrics.duplicate_events_removed += len(events_to_delete) 
            self.metrics.state_transitions_kept += len(events) - len(events_to_delete)
            
            if not dry_run and (events_to_delete or events_to_update):
                await self._apply_changes(session, events_to_delete, events_to_update)
    
    async def _process_entity_events(self, events: List[Any]) -> Tuple[List[Tuple], List[Dict]]:
        """
        Process events for a single entity to identify duplicates and transitions.
        
        Args:
            events: List of events for a single entity, ordered by timestamp
            
        Returns:
            Tuple of (events_to_delete, events_to_update)
            events_to_delete: List of (timestamp, entity_id) tuples to identify events
            events_to_update: List of update dictionaries (only if has_previous_state)
        """
        if not events:
            return [], []
        
        events_to_delete = []
        events_to_update = []
        
        # Track state for this entity
        last_state = None
        last_timestamp = None
        
        for i, event in enumerate(events):
            current_state = event.state
            current_timestamp = event.timestamp
            entity_id = event.entity_id
            
            # Always keep the first event
            if last_state is None:
                last_state = current_state
                last_timestamp = current_timestamp
                continue
            
            # Check if this is a duplicate (same state as previous)
            if current_state == last_state:
                time_diff = (current_timestamp - last_timestamp).total_seconds()
                
                # Remove duplicates within the time window
                if time_diff <= self.max_time_window_seconds:
                    # Use (timestamp, entity_id) tuple as identifier since we may not have id column
                    events_to_delete.append((current_timestamp, entity_id))
                    continue
                    
                # If it's been longer than the time window, it might be a valid
                # re-assertion of state (e.g., sensor heartbeat), so keep it
                
            # This is a state transition - update previous event's previous_state if column exists
            if self.has_previous_state and i > 0:
                previous_event = events[i-1]
                # Check if previous event is not being deleted
                if (previous_event.timestamp, previous_event.entity_id) not in events_to_delete:
                    events_to_update.append({
                        'timestamp': previous_event.timestamp,
                        'entity_id': previous_event.entity_id,
                        'previous_state': last_state
                    })
            
            # Update tracking
            last_state = current_state
            last_timestamp = current_timestamp
        
        return events_to_delete, events_to_update
    
    async def _apply_changes(self, 
                           session: AsyncSession, 
                           events_to_delete: List[Tuple], 
                           events_to_update: List[Dict]):
        """Apply the identified changes to the database"""
        
        try:
            # Delete duplicate events in batches using timestamp and entity_id
            if events_to_delete:
                logger.debug(f"Deleting {len(events_to_delete)} duplicate events")
                
                # Process deletes in batches to avoid parameter limits
                batch_size = 1000
                for i in range(0, len(events_to_delete), batch_size):
                    batch = events_to_delete[i:i + batch_size]
                    
                    # Build parameterized query for batch deletion
                    conditions = []
                    params = {}
                    
                    for j, (timestamp, entity_id) in enumerate(batch):
                        condition = f"(timestamp = :ts_{j} AND entity_id = :eid_{j})"
                        conditions.append(condition)
                        params[f'ts_{j}'] = timestamp
                        params[f'eid_{j}'] = entity_id
                    
                    delete_query = f"DELETE FROM sensor_events WHERE {' OR '.join(conditions)}"
                    await session.execute(text(delete_query), params)
            
            # Update events with corrected previous_state (only if column exists)
            if events_to_update and self.has_previous_state:
                logger.debug(f"Updating {len(events_to_update)} events")
                
                for update in events_to_update:
                    await session.execute(
                        text("""
                            UPDATE sensor_events 
                            SET previous_state = :previous_state 
                            WHERE timestamp = :timestamp AND entity_id = :entity_id
                        """),
                        update
                    )
            
            await session.commit()
            logger.debug("Changes committed successfully")
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Failed to apply changes: {e}")
            raise
    
    async def get_duplicate_analysis(self, 
                                   start_date: datetime = None,
                                   end_date: datetime = None,
                                   limit: int = 1000) -> Dict[str, Any]:
        """
        Analyze potential duplicates without removing them.
        
        Returns detailed analysis of duplicate patterns.
        """
        if not self.initialized:
            await self.initialize()
        
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=7)
        
        async with self.session_factory() as session:
            # Find potential duplicates
            query = """
                WITH duplicate_candidates AS (
                    SELECT 
                        entity_id,
                        state,
                        timestamp,
                        LAG(state) OVER (PARTITION BY entity_id ORDER BY timestamp) as prev_state,
                        LAG(timestamp) OVER (PARTITION BY entity_id ORDER BY timestamp) as prev_timestamp,
                        EXTRACT(EPOCH FROM timestamp - LAG(timestamp) OVER (PARTITION BY entity_id ORDER BY timestamp)) as time_diff_seconds
                    FROM sensor_events
                    WHERE timestamp >= :start_date AND timestamp <= :end_date
                )
                SELECT 
                    entity_id,
                    state,
                    COUNT(*) as duplicate_count,
                    AVG(time_diff_seconds) as avg_time_diff,
                    MIN(time_diff_seconds) as min_time_diff,
                    MAX(time_diff_seconds) as max_time_diff
                FROM duplicate_candidates
                WHERE state = prev_state 
                AND time_diff_seconds <= :max_window
                GROUP BY entity_id, state
                ORDER BY duplicate_count DESC
                LIMIT :limit
            """
            
            result = await session.execute(text(query), {
                'start_date': start_date,
                'end_date': end_date,
                'max_window': self.max_time_window_seconds,
                'limit': limit
            })
            
            duplicates = result.fetchall()
            
            # Aggregate statistics
            total_duplicates = sum(row.duplicate_count for row in duplicates)
            entities_with_duplicates = len(set(row.entity_id for row in duplicates))
            
            # Get total event count for comparison
            count_result = await session.execute(text("""
                SELECT COUNT(*) as total_events
                FROM sensor_events
                WHERE timestamp >= :start_date AND timestamp <= :end_date
            """), {'start_date': start_date, 'end_date': end_date})
            
            total_events = count_result.fetchone().total_events
            
            return {
                'analysis_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                'summary': {
                    'total_events': total_events,
                    'total_duplicates': total_duplicates,
                    'duplicate_rate': (total_duplicates / total_events * 100) if total_events > 0 else 0,
                    'entities_with_duplicates': entities_with_duplicates
                },
                'top_duplicate_patterns': [
                    {
                        'entity_id': row.entity_id,
                        'state': row.state,
                        'duplicate_count': row.duplicate_count,
                        'avg_time_diff_seconds': float(row.avg_time_diff or 0),
                        'min_time_diff_seconds': float(row.min_time_diff or 0),
                        'max_time_diff_seconds': float(row.max_time_diff or 0)
                    }
                    for row in duplicates
                ]
            }
    
    async def clean_ongoing_duplicates(self, 
                                     monitor_duration_minutes: int = 60,
                                     check_interval_seconds: int = 30) -> None:
        """
        Monitor and clean duplicates in real-time for ongoing data.
        
        Args:
            monitor_duration_minutes: How long to monitor (0 = indefinite)
            check_interval_seconds: How often to check for new duplicates
        """
        if not self.initialized:
            await self.initialize()
        
        logger.info(f"Starting ongoing duplicate cleaning (check every {check_interval_seconds}s)")
        
        start_time = datetime.now()
        last_check = datetime.now() - timedelta(minutes=5)  # Look back 5 minutes initially
        
        try:
            while True:
                current_time = datetime.now()
                
                # Check if we should stop monitoring
                if monitor_duration_minutes > 0:
                    if (current_time - start_time).total_seconds() > monitor_duration_minutes * 60:
                        logger.info("Monitoring duration completed")
                        break
                
                # Process recent events
                await self._process_chunk(last_check, current_time, dry_run=False)
                
                last_check = current_time
                await asyncio.sleep(check_interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error during ongoing cleaning: {e}")
            raise
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current deduplication metrics"""
        return {
            'total_events_processed': self.metrics.total_events_processed,
            'duplicate_events_removed': self.metrics.duplicate_events_removed,
            'state_transitions_kept': self.metrics.state_transitions_kept,
            'duplicate_rate': self.metrics.duplicate_rate,
            'processing_time_seconds': self.metrics.processing_time_seconds,
            'events_per_second': self.metrics.events_per_second,
            'chunks_processed': self.metrics.chunks_processed,
            'entities_processed': len(self.metrics.entities_processed),
            'configuration': {
                'chunk_size': self.chunk_size,
                'max_time_window_seconds': self.max_time_window_seconds,
                'min_state_duration_seconds': self.min_state_duration_seconds
            }
        }
    
    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            logger.info("EventDeduplicator connections closed")


# Utility functions for common deduplication tasks

async def analyze_duplicates(connection_string: str = None, 
                           days_back: int = 7) -> Dict[str, Any]:
    """
    Quick analysis of duplicate events in the database.
    
    Args:
        connection_string: Database connection (optional)
        days_back: Number of days to analyze
        
    Returns:
        Analysis report
    """
    deduplicator = EventDeduplicator(connection_string)
    try:
        start_date = datetime.now() - timedelta(days=days_back)
        return await deduplicator.get_duplicate_analysis(start_date=start_date)
    finally:
        await deduplicator.close()


async def clean_historical_data(connection_string: str = None,
                              days_back: int = 180,
                              dry_run: bool = True) -> DeduplicationMetrics:
    """
    Clean historical data of duplicates.
    
    Args:
        connection_string: Database connection (optional)
        days_back: Number of days to process
        dry_run: If True, only analyze without making changes
        
    Returns:
        Processing metrics
    """
    deduplicator = EventDeduplicator(connection_string)
    try:
        start_date = datetime.now() - timedelta(days=days_back)
        return await deduplicator.deduplicate_historical_data(
            start_date=start_date,
            dry_run=dry_run
        )
    finally:
        await deduplicator.close()