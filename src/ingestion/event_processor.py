"""
Real-time Event Processing - Stream processing infrastructure from CLAUDE.md
"""

import asyncio
import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from cachetools import TTLCache
import numpy as np

logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    Efficient stream processing with sliding windows and feature caching
    Implements the exact approach from CLAUDE.md
    """
    
    def __init__(self):
        self.window_manager = SlidingWindowManager()
        self.feature_cache = TTLCache(maxsize=10000, ttl=300)  # 5-minute TTL
        self.event_buffer = []
        self.processing_stats = {
            'events_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    async def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute features incrementally
        Update all relevant models asynchronously
        """
        self.processing_stats['events_processed'] += 1
        
        # Add to sliding window
        self.window_manager.add_event(event)
        
        # Compute features incrementally
        features = await self.incremental_feature_computation(event)
        
        # Update all relevant models asynchronously
        await asyncio.gather(
            self.update_short_term_models(features),
            self.update_long_term_models(features),
            self.update_pattern_miners(features),
            return_exceptions=True
        )
        
        return features
    
    async def incremental_feature_computation(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Compute features incrementally with caching"""
        # Create cache key
        cache_key = self.create_cache_key(event)
        
        # Check cache first
        if cache_key in self.feature_cache:
            self.processing_stats['cache_hits'] += 1
            return self.feature_cache[cache_key]
        
        self.processing_stats['cache_misses'] += 1
        
        # Compute features
        features = await self.compute_features(event)
        
        # Cache results
        self.feature_cache[cache_key] = features
        
        return features
    
    def create_cache_key(self, event: Dict[str, Any]) -> str:
        """Create cache key for event"""
        key_parts = [
            event.get('entity_id', ''),
            event.get('room', ''),
            str(event.get('state', '')),
            str(int(event['timestamp'].timestamp() // 60))  # Round to minute
        ]
        return "|".join(key_parts)
    
    async def compute_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Compute comprehensive features for event"""
        features = {}
        
        # Basic event features
        features.update(self.extract_basic_features(event))
        
        # Temporal features from sliding window
        features.update(self.extract_temporal_features(event))
        
        # Contextual features
        features.update(await self.extract_contextual_features(event))
        
        # Zone-based features
        features.update(self.extract_zone_features(event))
        
        return features
    
    def extract_basic_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic features from event"""
        timestamp = event['timestamp']
        
        return {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'minute_of_day': timestamp.hour * 60 + timestamp.minute,
            'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
            'is_morning': 1 if 6 <= timestamp.hour <= 11 else 0,
            'is_afternoon': 1 if 12 <= timestamp.hour <= 17 else 0,
            'is_evening': 1 if 18 <= timestamp.hour <= 23 else 0,
            'is_night': 1 if timestamp.hour <= 5 or timestamp.hour >= 24 else 0,
            'state': 1 if event.get('state') in ['on', 'open', 'detected'] else 0,
            'room': event.get('room', 'unknown'),
            'sensor_type': event.get('sensor_type', 'unknown')
        }
    
    def extract_temporal_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from sliding window"""
        window_events = self.window_manager.get_recent_events(
            event['timestamp'], 
            timedelta(minutes=30)
        )
        
        temporal_features = {
            'recent_activity_count': len(window_events),
            'recent_room_activity': 0,
            'recent_sensor_type_activity': 0,
            'time_since_last_event': 0.0
        }
        
        # Room-specific activity
        room = event.get('room')
        room_events = [e for e in window_events if e.get('room') == room]
        temporal_features['recent_room_activity'] = len(room_events)
        
        # Sensor type activity
        sensor_type = event.get('sensor_type')
        type_events = [e for e in window_events if e.get('sensor_type') == sensor_type]
        temporal_features['recent_sensor_type_activity'] = len(type_events)
        
        # Time since last event
        if window_events:
            last_event = max(window_events, key=lambda x: x['timestamp'])
            time_diff = (event['timestamp'] - last_event['timestamp']).total_seconds()
            temporal_features['time_since_last_event'] = time_diff
        
        return temporal_features
    
    async def extract_contextual_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract contextual features based on other sensors"""
        contextual = {
            'concurrent_activations': 0,
            'cross_room_activity': 0,
            'person_specific_indicators': {}
        }
        
        # Find concurrent events (within 5 seconds)
        current_time = event['timestamp']
        concurrent_window = timedelta(seconds=5)
        
        concurrent_events = self.window_manager.get_events_in_window(
            current_time - concurrent_window,
            current_time + concurrent_window
        )
        
        concurrent_events = [e for e in concurrent_events if e['entity_id'] != event['entity_id']]
        contextual['concurrent_activations'] = len(concurrent_events)
        
        # Cross-room activity
        current_room = event.get('room')
        other_room_events = [e for e in concurrent_events if e.get('room') != current_room]
        contextual['cross_room_activity'] = len(other_room_events)
        
        # Person-specific indicators
        entity_id = event.get('entity_id', '')
        if 'anca' in entity_id.lower():
            contextual['person_specific_indicators']['anca_activity'] = 1
        elif 'vladimir' in entity_id.lower():
            contextual['person_specific_indicators']['vladimir_activity'] = 1
        
        return contextual
    
    def extract_zone_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract zone-based features"""
        zone_info = event.get('derived', {}).get('zone_info', {})
        
        zone_features = {
            'zone_type': zone_info.get('zone_type', 'unknown'),
            'is_full_zone': 1 if zone_info.get('zone_type') == 'full' else 0,
            'is_subzone': 1 if zone_info.get('zone_type') == 'subzone' else 0,
            'zone_coverage_score': 0.0
        }
        
        # Calculate zone coverage score
        if zone_info.get('coverage'):
            coverage = zone_info['coverage']
            if coverage == 'general_room_area':
                zone_features['zone_coverage_score'] = 0.5
            elif coverage in ['general_living_area', 'general_kitchen_area']:
                zone_features['zone_coverage_score'] = 0.7
            elif 'specific_location' in zone_info:
                zone_features['zone_coverage_score'] = 1.0
        
        return zone_features
    
    async def update_short_term_models(self, features: Dict[str, Any]):
        """Update short-term models with new features"""
        try:
            # This would update the online learning models
            logger.debug(f"Updating short-term models with {len(features)} features")
        except Exception as e:
            logger.error(f"Error updating short-term models: {e}")
    
    async def update_long_term_models(self, features: Dict[str, Any]):
        """Update long-term models with new features"""
        try:
            # This would update pattern discovery models
            logger.debug(f"Updating long-term models with {len(features)} features")
        except Exception as e:
            logger.error(f"Error updating long-term models: {e}")
    
    async def update_pattern_miners(self, features: Dict[str, Any]):
        """Update pattern mining models"""
        try:
            # This would update temporal pattern miners
            logger.debug(f"Updating pattern miners with {len(features)} features")
        except Exception as e:
            logger.error(f"Error updating pattern miners: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        cache_hit_rate = 0.0
        total_requests = self.processing_stats['cache_hits'] + self.processing_stats['cache_misses']
        if total_requests > 0:
            cache_hit_rate = self.processing_stats['cache_hits'] / total_requests
        
        return {
            **self.processing_stats,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.feature_cache),
            'window_size': len(self.window_manager.events)
        }


class SlidingWindowManager:
    """
    Manages sliding windows of events for temporal feature extraction
    """
    
    def __init__(self, max_window_size: timedelta = timedelta(hours=24)):
        self.events = []
        self.max_window_size = max_window_size
        self.room_indexes = defaultdict(list)  # Index events by room for faster lookup
        self.entity_indexes = defaultdict(list)  # Index events by entity
        
    def add_event(self, event: Dict[str, Any]):
        """Add event to sliding window"""
        self.events.append(event)
        
        # Update indexes
        room = event.get('room')
        if room:
            self.room_indexes[room].append(len(self.events) - 1)
        
        entity_id = event.get('entity_id')
        if entity_id:
            self.entity_indexes[entity_id].append(len(self.events) - 1)
        
        # Clean old events
        self.cleanup_old_events()
    
    def cleanup_old_events(self):
        """Remove events older than max window size"""
        if not self.events:
            return
        
        cutoff_time = datetime.now() - self.max_window_size
        
        # Find first event to keep
        keep_from_index = 0
        for i, event in enumerate(self.events):
            if event['timestamp'] >= cutoff_time:
                keep_from_index = i
                break
        
        # Remove old events
        if keep_from_index > 0:
            self.events = self.events[keep_from_index:]
            
            # Update indexes
            self.rebuild_indexes()
    
    def rebuild_indexes(self):
        """Rebuild indexes after cleanup"""
        self.room_indexes.clear()
        self.entity_indexes.clear()
        
        for i, event in enumerate(self.events):
            room = event.get('room')
            if room:
                self.room_indexes[room].append(i)
            
            entity_id = event.get('entity_id')
            if entity_id:
                self.entity_indexes[entity_id].append(i)
    
    def get_recent_events(self, reference_time: datetime, window_size: timedelta) -> List[Dict[str, Any]]:
        """Get events within window before reference time"""
        start_time = reference_time - window_size
        
        recent_events = []
        for event in self.events:
            if start_time <= event['timestamp'] <= reference_time:
                recent_events.append(event)
        
        return recent_events
    
    def get_events_in_window(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get events within specific time window"""
        window_events = []
        for event in self.events:
            if start_time <= event['timestamp'] <= end_time:
                window_events.append(event)
        
        return window_events
    
    def get_room_events(self, room: str, window_size: timedelta) -> List[Dict[str, Any]]:
        """Get recent events for specific room"""
        if room not in self.room_indexes:
            return []
        
        cutoff_time = datetime.now() - window_size
        room_events = []
        
        for event_index in self.room_indexes[room]:
            if event_index < len(self.events):
                event = self.events[event_index]
                if event['timestamp'] >= cutoff_time:
                    room_events.append(event)
        
        return room_events
    
    def get_entity_events(self, entity_id: str, window_size: timedelta) -> List[Dict[str, Any]]:
        """Get recent events for specific entity"""
        if entity_id not in self.entity_indexes:
            return []
        
        cutoff_time = datetime.now() - window_size
        entity_events = []
        
        for event_index in self.entity_indexes[entity_id]:
            if event_index < len(self.events):
                event = self.events[event_index]
                if event['timestamp'] >= cutoff_time:
                    entity_events.append(event)
        
        return entity_events
    
    def get_window_statistics(self) -> Dict[str, Any]:
        """Get statistics about the sliding window"""
        if not self.events:
            return {
                'total_events': 0,
                'time_span': 0,
                'rooms_active': 0,
                'entities_active': 0
            }
        
        oldest_event = min(self.events, key=lambda x: x['timestamp'])
        newest_event = max(self.events, key=lambda x: x['timestamp'])
        
        return {
            'total_events': len(self.events),
            'time_span': (newest_event['timestamp'] - oldest_event['timestamp']).total_seconds(),
            'rooms_active': len(self.room_indexes),
            'entities_active': len(self.entity_indexes),
            'oldest_event': oldest_event['timestamp'],
            'newest_event': newest_event['timestamp']
        }


class EventBuffer:
    """
    Buffers events for batch processing and backpressure handling
    """
    
    def __init__(self, max_size: int = 1000, flush_interval: int = 10):
        self.buffer = []
        self.max_size = max_size
        self.flush_interval = flush_interval
        self.last_flush = datetime.now()
        self.buffer_stats = {
            'events_buffered': 0,
            'flushes_performed': 0,
            'buffer_overflows': 0
        }
    
    async def add_event(self, event: Dict[str, Any]) -> bool:
        """Add event to buffer, returns True if buffer needs flushing"""
        self.buffer.append(event)
        self.buffer_stats['events_buffered'] += 1
        
        # Check if buffer is full
        if len(self.buffer) >= self.max_size:
            self.buffer_stats['buffer_overflows'] += 1
            return True
        
        # Check if flush interval reached
        if (datetime.now() - self.last_flush).total_seconds() >= self.flush_interval:
            return True
        
        return False
    
    async def flush_buffer(self) -> List[Dict[str, Any]]:
        """Flush buffer and return events"""
        events = self.buffer.copy()
        self.buffer.clear()
        self.last_flush = datetime.now()
        self.buffer_stats['flushes_performed'] += 1
        
        return events
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            **self.buffer_stats,
            'current_buffer_size': len(self.buffer),
            'time_since_last_flush': (datetime.now() - self.last_flush).total_seconds()
        }