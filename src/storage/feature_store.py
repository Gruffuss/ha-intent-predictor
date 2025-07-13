"""
Redis Feature Cache - Real-time feature computation and caching
Implements the exact Redis caching approach from CLAUDE.md
"""

import logging
import json
import pickle
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import redis.asyncio as redis
import numpy as np

logger = logging.getLogger(__name__)


class RedisFeatureStore:
    """
    Redis for real-time feature computation and caching
    Implements the exact caching strategy from CLAUDE.md
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.connected = False
        
        # Cache configuration
        self.cache_ttl = {
            'features': 300,      # 5 minutes for features
            'predictions': 120,   # 2 minutes for predictions
            'patterns': 3600,     # 1 hour for patterns
            'stats': 600         # 10 minutes for statistics
        }
        
        # Key prefixes for organization
        self.key_prefixes = {
            'features': 'feat:',
            'predictions': 'pred:',
            'patterns': 'patt:',
            'stats': 'stat:',
            'models': 'model:',
            'room_state': 'room:',
            'entity_state': 'entity:'
        }
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=False,  # We'll handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            self.connected = True
            logger.info("Redis feature store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    async def cache_features(self, 
                           entity_id: str, 
                           timestamp: datetime, 
                           features: Dict[str, Any],
                           ttl: Optional[int] = None) -> bool:
        """Cache extracted features for entity at timestamp"""
        if not self.connected:
            await self.initialize()
        
        try:
            key = f"{self.key_prefixes['features']}{entity_id}:{int(timestamp.timestamp())}"
            
            # Serialize features
            features_data = {
                'timestamp': timestamp.isoformat(),
                'entity_id': entity_id,
                'features': features,
                'cached_at': datetime.now().isoformat()
            }
            
            serialized = pickle.dumps(features_data)
            
            # Cache with TTL
            cache_ttl = ttl or self.cache_ttl['features']
            await self.redis_client.setex(key, cache_ttl, serialized)
            
            return True
            
        except Exception as e:
            logger.error(f"Error caching features for {entity_id}: {e}")
            return False
    
    async def get_cached_features(self, 
                                entity_id: str, 
                                timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Get cached features for entity at timestamp"""
        try:
            key = f"{self.key_prefixes['features']}{entity_id}:{int(timestamp.timestamp())}"
            
            cached_data = await self.redis_client.get(key)
            if cached_data:
                features_data = pickle.loads(cached_data)
                return features_data['features']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached features for {entity_id}: {e}")
            return None
    
    async def cache_prediction(self, 
                             room: str, 
                             horizon_minutes: int,
                             prediction: Dict[str, Any],
                             timestamp: Optional[datetime] = None) -> bool:
        """Cache room occupancy prediction"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            key = f"{self.key_prefixes['predictions']}{room}:{horizon_minutes}"
            
            prediction_data = {
                'room': room,
                'horizon_minutes': horizon_minutes,
                'timestamp': timestamp.isoformat(),
                'prediction': prediction,
                'cached_at': datetime.now().isoformat()
            }
            
            serialized = pickle.dumps(prediction_data)
            await self.redis_client.setex(key, self.cache_ttl['predictions'], serialized)
            
            return True
            
        except Exception as e:
            logger.error(f"Error caching prediction for {room}: {e}")
            return False
    
    async def get_cached_prediction(self, 
                                  room: str, 
                                  horizon_minutes: int) -> Optional[Dict[str, Any]]:
        """Get cached prediction for room and horizon"""
        try:
            key = f"{self.key_prefixes['predictions']}{room}:{horizon_minutes}"
            
            cached_data = await self.redis_client.get(key)
            if cached_data:
                prediction_data = pickle.loads(cached_data)
                return prediction_data['prediction']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached prediction for {room}: {e}")
            return None
    
    async def update_room_state(self, 
                              room: str, 
                              state_data: Dict[str, Any]) -> bool:
        """Update current room state"""
        try:
            key = f"{self.key_prefixes['room_state']}{room}"
            
            room_state = {
                'room': room,
                'state': state_data,
                'last_updated': datetime.now().isoformat()
            }
            
            serialized = pickle.dumps(room_state)
            # Room state has longer TTL as it's frequently accessed
            await self.redis_client.setex(key, 1800, serialized)  # 30 minutes
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating room state for {room}: {e}")
            return False
    
    async def get_room_state(self, room: str) -> Optional[Dict[str, Any]]:
        """Get current room state"""
        try:
            key = f"{self.key_prefixes['room_state']}{room}"
            
            cached_data = await self.redis_client.get(key)
            if cached_data:
                room_state = pickle.loads(cached_data)
                return room_state['state']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting room state for {room}: {e}")
            return None
    
    async def cache_pattern(self, 
                          room: str, 
                          pattern_type: str,
                          pattern_data: Dict[str, Any]) -> bool:
        """Cache discovered patterns"""
        try:
            key = f"{self.key_prefixes['patterns']}{room}:{pattern_type}"
            
            pattern_cache = {
                'room': room,
                'pattern_type': pattern_type,
                'pattern_data': pattern_data,
                'discovered_at': datetime.now().isoformat()
            }
            
            serialized = pickle.dumps(pattern_cache)
            await self.redis_client.setex(key, self.cache_ttl['patterns'], serialized)
            
            return True
            
        except Exception as e:
            logger.error(f"Error caching pattern for {room}: {e}")
            return False
    
    async def get_cached_pattern(self, 
                               room: str, 
                               pattern_type: str) -> Optional[Dict[str, Any]]:
        """Get cached pattern"""
        try:
            key = f"{self.key_prefixes['patterns']}{room}:{pattern_type}"
            
            cached_data = await self.redis_client.get(key)
            if cached_data:
                pattern_cache = pickle.loads(cached_data)
                return pattern_cache['pattern_data']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached pattern for {room}: {e}")
            return None
    
    async def store_model_state(self, 
                              model_name: str, 
                              model_data: Dict[str, Any]) -> bool:
        """Store model state for persistence"""
        try:
            key = f"{self.key_prefixes['models']}{model_name}"
            
            model_state = {
                'model_name': model_name,
                'model_data': model_data,
                'saved_at': datetime.now().isoformat()
            }
            
            serialized = pickle.dumps(model_state)
            # Models have longer TTL
            await self.redis_client.setex(key, 7200, serialized)  # 2 hours
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing model state for {model_name}: {e}")
            return False
    
    async def get_model_state(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get stored model state"""
        try:
            key = f"{self.key_prefixes['models']}{model_name}"
            
            cached_data = await self.redis_client.get(key)
            if cached_data:
                model_state = pickle.loads(cached_data)
                return model_state['model_data']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting model state for {model_name}: {e}")
            return None
    
    async def increment_counter(self, counter_name: str, increment: int = 1) -> int:
        """Increment a counter (for statistics)"""
        try:
            key = f"{self.key_prefixes['stats']}counter:{counter_name}"
            result = await self.redis_client.incrby(key, increment)
            
            # Set expiration if this is a new counter
            await self.redis_client.expire(key, 86400)  # 24 hours
            
            return result
            
        except Exception as e:
            logger.error(f"Error incrementing counter {counter_name}: {e}")
            return 0
    
    async def get_counter(self, counter_name: str) -> int:
        """Get counter value"""
        try:
            key = f"{self.key_prefixes['stats']}counter:{counter_name}"
            result = await self.redis_client.get(key)
            
            if result:
                return int(result)
            return 0
            
        except Exception as e:
            logger.error(f"Error getting counter {counter_name}: {e}")
            return 0
    
    async def cache_aggregated_features(self, 
                                      room: str,
                                      time_window: str,  # '1h', '24h', etc.
                                      aggregated_features: Dict[str, Any]) -> bool:
        """Cache aggregated features for different time windows"""
        try:
            key = f"{self.key_prefixes['features']}agg:{room}:{time_window}"
            
            agg_data = {
                'room': room,
                'time_window': time_window,
                'features': aggregated_features,
                'computed_at': datetime.now().isoformat()
            }
            
            serialized = pickle.dumps(agg_data)
            
            # Longer TTL for aggregated features
            ttl = self.cache_ttl['features'] * 2
            await self.redis_client.setex(key, ttl, serialized)
            
            return True
            
        except Exception as e:
            logger.error(f"Error caching aggregated features for {room}: {e}")
            return False
    
    async def get_aggregated_features(self, 
                                    room: str,
                                    time_window: str) -> Optional[Dict[str, Any]]:
        """Get cached aggregated features"""
        try:
            key = f"{self.key_prefixes['features']}agg:{room}:{time_window}"
            
            cached_data = await self.redis_client.get(key)
            if cached_data:
                agg_data = pickle.loads(cached_data)
                return agg_data['features']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting aggregated features for {room}: {e}")
            return None
    
    async def invalidate_cache(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        try:
            # Find keys matching pattern
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                deleted = await self.redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries matching {pattern}")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Error invalidating cache with pattern {pattern}: {e}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        try:
            info = await self.redis_client.info()
            
            # Count keys by prefix
            key_counts = {}
            for prefix_name, prefix in self.key_prefixes.items():
                count = 0
                async for _ in self.redis_client.scan_iter(match=f"{prefix}*"):
                    count += 1
                key_counts[prefix_name] = count
            
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'key_counts_by_type': key_counts
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    async def cleanup_expired_keys(self):
        """Clean up expired keys (Redis handles this automatically, but we can force it)"""
        try:
            # Force expire check
            await self.redis_client.expire("dummy_key_for_cleanup", 1)
            await self.redis_client.delete("dummy_key_for_cleanup")
            
            logger.debug("Cache cleanup triggered")
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
    
    async def batch_cache_features(self, 
                                 feature_batch: List[Dict[str, Any]]) -> int:
        """Cache multiple features in batch for better performance"""
        try:
            pipe = self.redis_client.pipeline()
            successful_operations = 0
            
            for feature_data in feature_batch:
                entity_id = feature_data['entity_id']
                timestamp = feature_data['timestamp']
                features = feature_data['features']
                
                key = f"{self.key_prefixes['features']}{entity_id}:{int(timestamp.timestamp())}"
                
                features_data = {
                    'timestamp': timestamp.isoformat(),
                    'entity_id': entity_id,
                    'features': features,
                    'cached_at': datetime.now().isoformat()
                }
                
                serialized = pickle.dumps(features_data)
                pipe.setex(key, self.cache_ttl['features'], serialized)
                successful_operations += 1
            
            # Execute batch
            await pipe.execute()
            
            logger.debug(f"Batch cached {successful_operations} feature sets")
            return successful_operations
            
        except Exception as e:
            logger.error(f"Error in batch cache operation: {e}")
            return 0
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            self.connected = False
            logger.info("Redis connection closed")