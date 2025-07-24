"""
TimescaleDB Operations - Time-series optimized storage backend
Implements the exact storage approach from CLAUDE.md
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

logger = logging.getLogger(__name__)


class TimescaleDBManager:
    """
    PostgreSQL with TimescaleDB extension for time-series optimized storage
    Implements the exact database operations from CLAUDE.md
    """
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = None
        self.session_factory = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize database connection and create tables"""
        try:
            self.engine = create_async_engine(
                self.connection_string,
                echo=False,  # Set to True for SQL debugging
                pool_size=10,
                max_overflow=20
            )
            
            self.session_factory = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            await self.create_tables()
            self.initialized = True
            logger.info("TimescaleDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TimescaleDB: {e}")
            raise
    
    async def create_tables(self):
        """Create tables and hypertables for time-series data"""
        async with self.engine.begin() as conn:
            # Enable TimescaleDB extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
            
            # Sensor events table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS sensor_events (
                    timestamp TIMESTAMPTZ NOT NULL,
                    entity_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    room TEXT,
                    sensor_type TEXT,
                    attributes JSONB,
                    derived_features JSONB,
                    PRIMARY KEY (timestamp, entity_id)
                );
            """))
            
            # Convert to hypertable (time-series optimized)
            try:
                await conn.execute(text("""
                    SELECT create_hypertable('sensor_events', 'timestamp', 
                                           if_not_exists => TRUE,
                                           chunk_time_interval => INTERVAL '1 day');
                """))
            except Exception as e:
                logger.warning(f"Hypertable creation failed (may already exist): {e}")
            
            # Predictions table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS predictions (
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
                );
            """))
            
            # Convert predictions to hypertable
            try:
                await conn.execute(text("""
                    SELECT create_hypertable('predictions', 'timestamp',
                                           if_not_exists => TRUE,
                                           chunk_time_interval => INTERVAL '1 day');
                """))
            except Exception as e:
                logger.warning(f"Predictions hypertable creation failed: {e}")
            
            # Model performance table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS model_performance (
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
                );
            """))
            
            # Pattern discoveries table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS pattern_discoveries (
                    timestamp TIMESTAMPTZ NOT NULL,
                    room TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    pattern_data JSONB NOT NULL,
                    significance_score FLOAT,
                    frequency INTEGER,
                    metadata JSONB,
                    PRIMARY KEY (timestamp, room, pattern_type)
                );
            """))
            
            # Create indexes for performance
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_sensor_events_room_time 
                ON sensor_events (room, timestamp DESC);
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_sensor_events_entity_time 
                ON sensor_events (entity_id, timestamp DESC);
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_predictions_room_time 
                ON predictions (room, timestamp DESC);
            """))
            
            logger.info("Database tables and indexes created successfully")
    
    async def insert_sensor_event(self, event: Dict[str, Any]) -> bool:
        """Insert sensor event into time-series database"""
        if not self.initialized:
            await self.initialize()
        
        try:
            async with self.session_factory() as session:
                await session.execute(
                    text("""
                        INSERT INTO sensor_events 
                        (timestamp, entity_id, state, room, sensor_type, attributes, derived_features)
                        VALUES (:timestamp, :entity_id, :state, :room, :sensor_type, :attributes, :derived_features)
                        ON CONFLICT (timestamp, entity_id) DO UPDATE SET
                        state = EXCLUDED.state,
                        room = EXCLUDED.room,
                        sensor_type = EXCLUDED.sensor_type,
                        attributes = EXCLUDED.attributes,
                        derived_features = EXCLUDED.derived_features
                    """),
                    {
                        'timestamp': event['timestamp'],
                        'entity_id': event['entity_id'],
                        'state': event['state'],
                        'room': event.get('room'),
                        'sensor_type': event.get('sensor_type'),
                        'attributes': json.dumps(event.get('attributes', {})),
                        'derived_features': json.dumps(event.get('derived', {}))
                    }
                )
                await session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error inserting sensor event: {e}")
            return False
    
    async def insert_prediction(self, prediction: Dict[str, Any]) -> bool:
        """Insert prediction into database"""
        try:
            async with self.session_factory() as session:
                await session.execute(
                    text("""
                        INSERT INTO predictions 
                        (timestamp, room, horizon_minutes, probability, uncertainty, confidence, model_name, features, metadata)
                        VALUES (:timestamp, :room, :horizon_minutes, :probability, :uncertainty, :confidence, :model_name, :features, :metadata)
                        ON CONFLICT (timestamp, room, horizon_minutes) DO UPDATE SET
                        probability = EXCLUDED.probability,
                        uncertainty = EXCLUDED.uncertainty,
                        confidence = EXCLUDED.confidence,
                        model_name = EXCLUDED.model_name,
                        features = EXCLUDED.features,
                        metadata = EXCLUDED.metadata
                    """),
                    {
                        'timestamp': prediction['timestamp'],
                        'room': prediction['room'],
                        'horizon_minutes': prediction['horizon_minutes'],
                        'probability': prediction['probability'],
                        'uncertainty': prediction['uncertainty'],
                        'confidence': prediction['confidence'],
                        'model_name': prediction.get('model_name'),
                        'features': json.dumps(prediction.get('features', {})),
                        'metadata': json.dumps(prediction.get('metadata', {}))
                    }
                )
                await session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error inserting prediction: {e}")
            return False
    
    async def get_historical_events(self, 
                                  start_time: datetime, 
                                  end_time: datetime,
                                  rooms: Optional[List[str]] = None,
                                  entity_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get historical sensor events from time-series database"""
        try:
            query = """
                SELECT timestamp, entity_id, state, room, sensor_type, attributes, derived_features
                FROM sensor_events 
                WHERE timestamp >= :start_time AND timestamp <= :end_time
            """
            params = {'start_time': start_time, 'end_time': end_time}
            
            if rooms:
                query += " AND room = ANY(:rooms)"
                params['rooms'] = rooms
            
            if entity_ids:
                query += " AND entity_id = ANY(:entity_ids)"
                params['entity_ids'] = entity_ids
            
            query += " ORDER BY timestamp ASC"
            
            async with self.session_factory() as session:
                result = await session.execute(text(query), params)
                rows = result.fetchall()
                
                events = []
                for row in rows:
                    events.append({
                        'timestamp': row.timestamp,
                        'entity_id': row.entity_id,
                        'state': row.state,
                        'room': row.room,
                        'sensor_type': row.sensor_type,
                        'attributes': row.attributes or {},
                        'derived': row.derived_features or {}
                    })
                
                return events
                
        except Exception as e:
            logger.error(f"Error getting historical events: {e}")
            return []
    
    async def get_recent_events(self, 
                               minutes: int = 30,
                               rooms: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get recent sensor events"""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes)
        
        return await self.get_historical_events(start_time, end_time, rooms)
    
    async def insert_pattern_discovery(self, pattern: Dict[str, Any]) -> bool:
        """Insert discovered pattern into database"""
        try:
            async with self.session_factory() as session:
                await session.execute(
                    text("""
                        INSERT INTO pattern_discoveries 
                        (timestamp, room, pattern_type, pattern_data, significance_score, frequency, metadata)
                        VALUES (:timestamp, :room, :pattern_type, :pattern_data, :significance_score, :frequency, :metadata)
                        ON CONFLICT (timestamp, room, pattern_type) DO UPDATE SET
                        pattern_data = EXCLUDED.pattern_data,
                        significance_score = EXCLUDED.significance_score,
                        frequency = EXCLUDED.frequency,
                        metadata = EXCLUDED.metadata
                    """),
                    {
                        'timestamp': pattern.get('timestamp', datetime.now()),
                        'room': pattern['room'],
                        'pattern_type': pattern['pattern_type'],
                        'pattern_data': pattern['pattern_data'],
                        'significance_score': pattern.get('significance_score', 0.0),
                        'frequency': pattern.get('frequency', 1),
                        'metadata': pattern.get('metadata', {})
                    }
                )
                await session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error inserting pattern discovery: {e}")
            return False
    
    async def get_room_occupancy_history(self, 
                                       room: str, 
                                       hours: int = 24) -> pd.DataFrame:
        """Get room occupancy history as DataFrame for analysis"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            query = """
                SELECT timestamp, entity_id, state, sensor_type, derived_features
                FROM sensor_events 
                WHERE room = :room 
                AND timestamp >= :start_time 
                AND timestamp <= :end_time
                ORDER BY timestamp ASC
            """
            
            async with self.session_factory() as session:
                result = await session.execute(
                    text(query), 
                    {'room': room, 'start_time': start_time, 'end_time': end_time}
                )
                rows = result.fetchall()
                
                if not rows:
                    return pd.DataFrame()
                
                # Convert to DataFrame
                data = []
                for row in rows:
                    data.append({
                        'timestamp': row.timestamp,
                        'entity_id': row.entity_id,
                        'state': row.state,
                        'sensor_type': row.sensor_type,
                        'derived_features': row.derived_features or {}
                    })
                
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
                
        except Exception as e:
            logger.error(f"Error getting room occupancy history: {e}")
            return pd.DataFrame()
    
    async def cleanup_old_data(self, retention_days: int = 90):
        """Clean up old data beyond retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            async with self.session_factory() as session:
                # Clean sensor events
                result = await session.execute(
                    text("DELETE FROM sensor_events WHERE timestamp < :cutoff_date"),
                    {'cutoff_date': cutoff_date}
                )
                deleted_events = result.rowcount
                
                # Clean predictions
                result = await session.execute(
                    text("DELETE FROM predictions WHERE timestamp < :cutoff_date"),
                    {'cutoff_date': cutoff_date}
                )
                deleted_predictions = result.rowcount
                
                # Clean old pattern discoveries
                result = await session.execute(
                    text("DELETE FROM pattern_discoveries WHERE timestamp < :cutoff_date"),
                    {'cutoff_date': cutoff_date}
                )
                deleted_patterns = result.rowcount
                
                await session.commit()
                
                logger.info(f"Cleanup completed: {deleted_events} events, {deleted_predictions} predictions, {deleted_patterns} patterns deleted")
                
        except Exception as e:
            logger.error(f"Error during data cleanup: {e}")
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            async with self.session_factory() as session:
                # Get table sizes
                stats_query = """
                    SELECT 
                        schemaname,
                        tablename,
                        attname,
                        n_distinct,
                        correlation
                    FROM pg_stats 
                    WHERE schemaname = 'public' 
                    AND tablename IN ('sensor_events', 'predictions', 'pattern_discoveries');
                """
                
                # Get row counts
                counts_query = """
                    SELECT 
                        'sensor_events' as table_name,
                        COUNT(*) as row_count,
                        MIN(timestamp) as oldest_record,
                        MAX(timestamp) as newest_record
                    FROM sensor_events
                    UNION ALL
                    SELECT 
                        'predictions' as table_name,
                        COUNT(*) as row_count,
                        MIN(timestamp) as oldest_record,
                        MAX(timestamp) as newest_record
                    FROM predictions;
                """
                
                counts_result = await session.execute(text(counts_query))
                counts_rows = counts_result.fetchall()
                
                stats = {}
                for row in counts_rows:
                    stats[row.table_name] = {
                        'row_count': row.row_count,
                        'oldest_record': row.oldest_record,
                        'newest_record': row.newest_record
                    }
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    async def get_recent_predictions(self, 
                                   hours: int = 24,
                                   room: Optional[str] = None,
                                   horizon: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent predictions from database"""
        try:
            async with self.session_factory() as session:
                query = f"""
                    SELECT timestamp, room, horizon_minutes, probability, uncertainty, 
                           confidence, model_name, features, metadata
                    FROM predictions 
                    WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
                """
                params = {}
                
                if room:
                    query += " AND room = :room"
                    params['room'] = room
                    
                if horizon:
                    query += " AND horizon_minutes = :horizon"
                    params['horizon'] = horizon
                    
                query += " ORDER BY timestamp DESC"
                
                result = await session.execute(text(query), params)
                rows = result.fetchall()
                
                predictions = []
                for row in rows:
                    predictions.append({
                        'timestamp': row.timestamp,
                        'room': row.room,
                        'horizon_minutes': row.horizon_minutes,
                        'probability': row.probability,
                        'uncertainty': row.uncertainty,
                        'confidence': row.confidence,
                        'model_name': row.model_name,
                        'features': row.features or {},
                        'metadata': row.metadata or {}
                    })
                
                return predictions
                
        except Exception as e:
            logger.error(f"Error getting recent predictions: {e}")
            return []
    
    async def get_room_activity_summary(self, 
                                      room: str,
                                      hours: int = 24) -> Dict[str, Any]:
        """Get activity summary for a specific room"""
        try:
            async with self.session_factory() as session:
                query = """
                    SELECT 
                        COUNT(*) as total_events,
                        COUNT(CASE WHEN state = '1' OR state = 'on' THEN 1 END) as active_events,
                        AVG(CASE WHEN numeric_value IS NOT NULL THEN numeric_value END) as avg_numeric_value,
                        array_agg(DISTINCT sensor_type) as sensor_types,
                        MIN(timestamp) as first_event,
                        MAX(timestamp) as last_event
                    FROM sensor_events 
                    WHERE room = :room 
                    AND timestamp >= NOW() - INTERVAL ':hours hours'
                """
                
                result = await session.execute(text(query), {'room': room, 'hours': hours})
                row = result.fetchone()
                
                if row:
                    return {
                        'room': room,
                        'total_events': row.total_events or 0,
                        'active_events': row.active_events or 0,
                        'avg_numeric_value': float(row.avg_numeric_value) if row.avg_numeric_value else 0.0,
                        'sensor_types': row.sensor_types or [],
                        'first_event': row.first_event,
                        'last_event': row.last_event,
                        'activity_rate': (row.total_events or 0) / hours if hours > 0 else 0
                    }
                else:
                    return {
                        'room': room,
                        'total_events': 0,
                        'active_events': 0,
                        'avg_numeric_value': 0.0,
                        'sensor_types': [],
                        'first_event': None,
                        'last_event': None,
                        'activity_rate': 0.0
                    }
                
        except Exception as e:
            logger.error(f"Error getting room activity summary for {room}: {e}")
            return {'room': room, 'total_events': 0, 'active_events': 0, 'activity_rate': 0.0}

    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")