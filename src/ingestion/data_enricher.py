"""
Dynamic Feature Computation - Implements the sophisticated feature discovery from CLAUDE.md
"""

import logging
from collections import defaultdict
from typing import Dict, List, Any, Set, Tuple
from datetime import datetime, timedelta
import numpy as np
from river import stats
from .event_processor import StreamProcessor

logger = logging.getLogger(__name__)


class DynamicFeatureDiscovery:
    """
    Discovers what features matter without predefined assumptions
    Implements the exact approach from CLAUDE.md
    """
    
    def __init__(self):
        self.feature_importance_tracker = {}
        self.interaction_detector = InteractionDiscovery()
        self.temporal_pattern_miner = TemporalMiner()
        # CRITICAL FIX: Use StreamProcessor for proper temporal features
        self.stream_processor = StreamProcessor()
        # Store reference to feature store for pattern access
        self.feature_store = None
        self.pattern_cache = {}
        self.pattern_cache_ttl = {}
        
    def set_feature_store(self, feature_store):
        """Set the feature store reference for pattern access"""
        self.feature_store = feature_store
        logger.info("Feature store connected to DynamicFeatureDiscovery")
        
    async def discover_features(self, sensor_stream: List[Dict[str, Any]], room_id: str = None) -> Dict[str, Any]:
        """
        Extract different types of features from zone combinations
        INCLUDING discovered patterns from Redis
        """
        # CRITICAL FIX: Extract basic temporal features using StreamProcessor
        basic_temporal_features = self.extract_basic_temporal_features_from_stream(sensor_stream)
        
        # Extract zone combination features
        zone_features = self.extract_zone_combination_features(sensor_stream)
        
        # Generate interaction features
        interaction_features = self.interaction_detector.detect_interactions(sensor_stream)
        
        # Mine advanced temporal patterns
        advanced_temporal_features = self.temporal_pattern_miner.mine_patterns(sensor_stream)
        
        # CRITICAL FIX: Get discovered patterns from Redis and convert to features
        pattern_features = {}
        if room_id and self.feature_store:
            pattern_features = await self.extract_pattern_based_features(room_id, basic_temporal_features)
        
        # Combine all feature types
        all_features = {
            **basic_temporal_features,
            **zone_features,
            **interaction_features,
            **advanced_temporal_features,
            **pattern_features  # Now includes the discovered patterns!
        }
        
        # Select only statistically significant features
        significant_features = self.select_significant_features(all_features)
        
        return significant_features
    
    async def extract_pattern_based_features(self, room_id: str, current_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features based on discovered patterns from Redis
        This is the CRITICAL missing piece!
        """
        try:
            # Check cache first
            cache_key = f"{room_id}_patterns"
            if cache_key in self.pattern_cache:
                cache_time = self.pattern_cache_ttl.get(cache_key, 0)
                if (datetime.now().timestamp() - cache_time) < 300:  # 5 minute cache
                    patterns = self.pattern_cache[cache_key]
                else:
                    patterns = None
            else:
                patterns = None
            
            # Fetch from Redis if not cached
            if patterns is None:
                pattern_data = await self.feature_store.get_cached_pattern(room_id, 'discovered_patterns')
                if pattern_data:
                    patterns = pattern_data.get('patterns', {})
                    self.pattern_cache[cache_key] = patterns
                    self.pattern_cache_ttl[cache_key] = datetime.now().timestamp()
                else:
                    logger.debug(f"No patterns found in Redis for {room_id}")
                    return {}
            
            # Convert patterns to ML features
            pattern_features = {}
            
            # 1. Recurring pattern features
            if 'recurring' in patterns:
                pattern_features.update(self._extract_recurring_pattern_features(patterns['recurring'], current_features))
            
            # 2. Seasonal pattern features
            if 'seasonal' in patterns:
                pattern_features.update(self._extract_seasonal_pattern_features(patterns['seasonal'], current_features))
            
            # 3. Duration pattern features
            if 'durations' in patterns:
                pattern_features.update(self._extract_duration_pattern_features(patterns['durations']))
            
            # 4. Sequential pattern features
            if 'sequential' in patterns:
                pattern_features.update(self._extract_sequential_pattern_features(patterns['sequential']))
            
            # 5. Anomaly features
            if 'anomalies' in patterns:
                pattern_features.update(self._extract_anomaly_features(patterns['anomalies'], current_features))
            
            # 6. Transition pattern features
            if 'transitions' in patterns:
                pattern_features.update(self._extract_transition_features(patterns['transitions']))
            
            logger.debug(f"Extracted {len(pattern_features)} pattern-based features for {room_id}")
            return pattern_features
            
        except Exception as e:
            logger.error(f"Error extracting pattern features for {room_id}: {e}")
            return {}
    
    def _extract_recurring_pattern_features(self, recurring_patterns: Dict, current_features: Dict) -> Dict[str, float]:
        """Convert recurring patterns into ML features"""
        features = {}
        
        current_hour = current_features.get('hour', 0)
        current_minute = current_features.get('minute_of_day', 0)
        
        # Check each time window for pattern matches
        for window_label, pattern_info in recurring_patterns.items():
            if pattern_info.get('found', False):
                # Pattern strength feature
                features[f'pattern_strength_{window_label}'] = pattern_info.get('pattern_strength', 0.0)
                
                # Number of recurring patterns at this scale
                features[f'pattern_count_{window_label}'] = float(pattern_info.get('pattern_count', 0))
                
                # Check if current time matches any top patterns
                if 'top_patterns' in pattern_info:
                    match_score = 0.0
                    for pattern in pattern_info['top_patterns']:
                        # Simple time-based matching
                        pattern_start = pattern.get('start_time', '')
                        if pattern_start:
                            try:
                                pattern_time = datetime.fromisoformat(pattern_start)
                                if pattern_time.hour == current_hour:
                                    match_score += 1.0 - pattern.get('distance', 1.0)
                            except:
                                pass
                    
                    features[f'pattern_match_{window_label}'] = match_score
        
        return features
    
    def _extract_seasonal_pattern_features(self, seasonal_patterns: Dict, current_features: Dict) -> Dict[str, float]:
        """Convert seasonal patterns into ML features"""
        features = {}
        
        current_hour = current_features.get('hour', 0)
        is_weekend = current_features.get('is_weekend', 0)
        
        # Daily pattern strength
        features['has_daily_pattern'] = float(seasonal_patterns.get('has_daily_pattern', False))
        features['daily_pattern_strength'] = seasonal_patterns.get('pattern_strength', 0.0)
        
        # Peak hour features
        peak_hours = seasonal_patterns.get('peak_hours', [])
        features['is_peak_hour'] = float(current_hour in peak_hours)
        features['hours_to_nearest_peak'] = min([abs(current_hour - ph) for ph in peak_hours]) if peak_hours else 12.0
        
        # Quiet hour features
        quiet_hours = seasonal_patterns.get('quiet_hours', [])
        features['is_quiet_hour'] = float(current_hour in quiet_hours)
        
        # Weekend pattern features
        features['weekend_different'] = float(seasonal_patterns.get('weekend_different', False))
        if is_weekend:
            features['expected_occupancy'] = seasonal_patterns.get('weekend_occupancy', 0.5)
        else:
            features['expected_occupancy'] = seasonal_patterns.get('weekday_occupancy', 0.5)
        
        # Hourly pattern lookup
        hourly_pattern = seasonal_patterns.get('hourly_pattern', {})
        features['hourly_expected_occupancy'] = hourly_pattern.get(str(current_hour), 0.5)
        
        # Trend feature
        trend = seasonal_patterns.get('trend', 'stable')
        features['trend_increasing'] = float(trend == 'increasing')
        features['trend_decreasing'] = float(trend == 'decreasing')
        
        return features
    
    def _extract_duration_pattern_features(self, duration_patterns: Dict) -> Dict[str, float]:
        """Convert duration patterns into ML features"""
        features = {}
        
        # Basic duration statistics
        features['mean_duration_minutes'] = duration_patterns.get('mean_duration', 60.0)
        features['median_duration_minutes'] = duration_patterns.get('median_duration', 60.0)
        features['duration_variability'] = duration_patterns.get('std_duration', 30.0)
        
        # Weekend vs weekday patterns
        features['weekday_mean_duration'] = duration_patterns.get('weekday_mean', 60.0)
        features['weekend_mean_duration'] = duration_patterns.get('weekend_mean', 60.0)
        features['weekend_duration_ratio'] = (
            features['weekend_mean_duration'] / features['weekday_mean_duration']
            if features['weekday_mean_duration'] > 0 else 1.0
        )
        
        # Duration clusters
        if 'duration_clusters' in duration_patterns:
            clusters = duration_patterns['duration_clusters']
            features['num_duration_clusters'] = float(len(clusters))
            
            # Features for each cluster
            for i, cluster in enumerate(clusters[:3]):  # Top 3 clusters
                features[f'cluster_{i}_duration'] = cluster.get('mean_duration', 60.0)
                features[f'cluster_{i}_frequency'] = cluster.get('percentage', 0.0) / 100.0
        
        return features
    
    def _extract_sequential_pattern_features(self, sequential_patterns: Dict) -> Dict[str, float]:
        """Convert sequential patterns into ML features"""
        features = {}
        
        # Basic sequential pattern stats
        features['has_sequential_patterns'] = float(not sequential_patterns.get('no_patterns', True))
        features['session_count'] = float(sequential_patterns.get('session_count', 0))
        
        # Frequent itemset features
        frequent_sets = sequential_patterns.get('frequent_sets', [])
        features['num_frequent_patterns'] = float(len(frequent_sets))
        
        # Top pattern support scores
        for i, pattern in enumerate(frequent_sets[:5]):  # Top 5 patterns
            features[f'top_pattern_{i}_support'] = pattern.get('support', 0.0)
            features[f'top_pattern_{i}_frequency'] = float(pattern.get('frequency', 0))
        
        # Association rule features
        rules = sequential_patterns.get('rules', [])
        features['num_association_rules'] = float(len(rules))
        
        # Top rule confidence scores
        for i, rule in enumerate(rules[:3]):  # Top 3 rules
            features[f'rule_{i}_confidence'] = rule.get('confidence', 0.0)
            features[f'rule_{i}_lift'] = rule.get('lift', 1.0)
        
        return features
    
    def _extract_anomaly_features(self, anomaly_patterns: Dict, current_features: Dict) -> Dict[str, float]:
        """Convert anomaly patterns into ML features"""
        features = {}
        
        # Anomaly statistics
        features['anomaly_percentage'] = anomaly_patterns.get('anomaly_percentage', 0.0)
        features['total_anomaly_count'] = float(anomaly_patterns.get('anomaly_count', 0))
        
        # Check if current time matches anomaly patterns
        current_hour = current_features.get('hour', 0)
        current_dow = current_features.get('day_of_week', 0)
        
        anomaly_match_score = 0.0
        top_anomalies = anomaly_patterns.get('top_anomalies', [])
        
        for anomaly in top_anomalies:
            if (anomaly.get('hour') == current_hour and 
                anomaly.get('day_of_week') == current_dow):
                anomaly_match_score += 1.0
        
        features['anomaly_time_match'] = anomaly_match_score / max(len(top_anomalies), 1)
        
        return features
    
    def _extract_transition_features(self, transition_patterns: Dict) -> Dict[str, float]:
        """Convert transition patterns into ML features"""
        features = {}
        
        # Basic transition stats
        features['total_transitions'] = float(transition_patterns.get('total_transitions', 0))
        features['mean_transition_time'] = transition_patterns.get('mean_transition_time', 300.0)
        features['median_transition_time'] = transition_patterns.get('median_transition_time', 300.0)
        
        # Transition matrix probabilities
        transition_matrix = transition_patterns.get('transition_matrix', {})
        
        # Extract key transition probabilities
        if 'empty' in transition_matrix:
            features['prob_empty_to_occupied'] = transition_matrix['empty'].get('occupied', 0.5)
        if 'occupied' in transition_matrix:
            features['prob_occupied_to_empty'] = transition_matrix['occupied'].get('empty', 0.5)
        
        # Top entity transition features
        top_transitions = transition_patterns.get('top_entity_transitions', [])
        features['num_common_transitions'] = float(len(top_transitions))
        
        return features
    
    def extract_basic_temporal_features_from_stream(self, sensor_stream: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract basic temporal features using StreamProcessor
        CRITICAL FIX: Use existing StreamProcessor temporal feature extraction
        """
        if not sensor_stream:
            return {}
        
        # Get the most recent event for temporal context
        latest_event = max(sensor_stream, key=lambda x: x.get('timestamp', 0))
        
        try:
            # Use StreamProcessor's extract_basic_features method
            basic_features = self.stream_processor.extract_basic_features(latest_event)
            
            # Use StreamProcessor's extract_temporal_features method  
            temporal_features = self.stream_processor.extract_temporal_features(latest_event)
            
            # Combine both feature sets
            combined_features = {**basic_features, **temporal_features}
            
            return combined_features
            
        except Exception as e:
            # Fallback to manual extraction if StreamProcessor fails
            if 'timestamp' in latest_event:
                timestamp = latest_event['timestamp']
                return {
                    'hour': timestamp.hour,
                    'day_of_week': timestamp.weekday(),
                    'minute_of_day': timestamp.hour * 60 + timestamp.minute,
                    'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
                    'is_morning': 1 if 6 <= timestamp.hour <= 11 else 0,
                    'is_afternoon': 1 if 12 <= timestamp.hour <= 17 else 0,
                    'is_evening': 1 if 18 <= timestamp.hour <= 23 else 0,
                    'is_night': 1 if timestamp.hour <= 5 or timestamp.hour >= 24 else 0
                }
            return {}
    
    def extract_zone_combination_features(self, sensor_stream: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract features from full zone + subzone combinations
        Implements the exact logic from CLAUDE.md
        """
        features = {
            'full_without_sub': self.detect_general_area_presence(sensor_stream),
            'specific_locations': self.extract_subzone_patterns(sensor_stream),
            'zone_coverage': self.calculate_zone_coverage(sensor_stream),
            'movement_precision': self.analyze_movement_granularity(sensor_stream)
        }
        
        return features
    
    def detect_general_area_presence(self, stream: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect when someone is in general area but not in any subzone
        This indicates transitional movement or areas without subzone coverage
        """
        patterns = []
        
        for event in stream:
            if event.get('derived', {}).get('zone_info', {}).get('zone_type') == 'full':
                # Check if any related subzones are active
                room_state = self.get_room_state(event['room'], event['timestamp'], stream)
                subzones_active = any(
                    sz.get('state') == 1 
                    for sz in room_state.get('subzones', [])
                )
                
                if not subzones_active:
                    # Person in general area, not in specific location
                    patterns.append({
                        'type': 'general_presence',
                        'room': event['room'],
                        'timestamp': event['timestamp'],
                        'indicates': 'transitional_movement'
                    })
        
        return patterns
    
    def get_room_state(self, room: str, timestamp: datetime, stream: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get state of all sensors in room at given timestamp"""
        # Find events around this timestamp for the room
        time_window = timedelta(seconds=30)
        
        relevant_events = [
            event for event in stream
            if (event.get('room') == room and 
                abs((event['timestamp'] - timestamp).total_seconds()) <= time_window.total_seconds())
        ]
        
        # Group by zone type
        state = {
            'subzones': [],
            'full_zones': []
        }
        
        for event in relevant_events:
            zone_info = event.get('derived', {}).get('zone_info', {})
            zone_type = zone_info.get('zone_type')
            
            if zone_type == 'subzone':
                state['subzones'].append({
                    'zone': zone_info.get('zone'),
                    'state': event.get('state', 0)
                })
            elif zone_type == 'full':
                state['full_zones'].append({
                    'zone': zone_info.get('zone'),
                    'state': event.get('state', 0)
                })
        
        return state
    
    def extract_subzone_patterns(self, stream: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract patterns from subzone activities"""
        subzone_patterns = {
            'activity_sequences': [],
            'location_preferences': {},
            'transition_patterns': []
        }
        
        # Track subzone-to-subzone movements
        subzone_events = [
            event for event in stream
            if event.get('derived', {}).get('zone_info', {}).get('zone_type') == 'subzone'
        ]
        
        for i in range(len(subzone_events) - 1):
            curr = subzone_events[i]
            next_event = subzone_events[i + 1]
            
            if curr.get('state') == 1 and next_event.get('state') == 1:
                transition = {
                    'from': curr.get('derived', {}).get('zone_info', {}).get('zone'),
                    'to': next_event.get('derived', {}).get('zone_info', {}).get('zone'),
                    'duration': (next_event['timestamp'] - curr['timestamp']).total_seconds()
                }
                subzone_patterns['transition_patterns'].append(transition)
        
        return subzone_patterns
    
    def calculate_zone_coverage(self, stream: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate how much of each room is covered by subzones"""
        coverage = {}
        
        # Group events by room
        room_events = defaultdict(list)
        for event in stream:
            room = event.get('room')
            if room:
                room_events[room].append(event)
        
        for room, events in room_events.items():
            full_zone_events = [e for e in events if e.get('derived', {}).get('zone_info', {}).get('zone_type') == 'full']
            subzone_events = [e for e in events if e.get('derived', {}).get('zone_info', {}).get('zone_type') == 'subzone']
            
            if full_zone_events:
                coverage_ratio = len(subzone_events) / len(full_zone_events)
                coverage[room] = min(1.0, coverage_ratio)
            else:
                coverage[room] = 0.0
        
        return coverage
    
    def analyze_movement_granularity(self, stream: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        Analyze movement patterns using both full and subzone data
        Implements the exact analysis from CLAUDE.md
        """
        movement_patterns = {
            'precise_movements': [],  # Subzone to subzone
            'general_movements': [],  # Full zone without subzone detail
            'hybrid_movements': []    # Combinations
        }
        
        for i in range(len(stream) - 1):
            curr = stream[i]
            next_event = stream[i + 1]
            
            curr_zone_info = curr.get('derived', {}).get('zone_info', {})
            next_zone_info = next_event.get('derived', {}).get('zone_info', {})
            
            curr_type = curr_zone_info.get('zone_type')
            next_type = next_zone_info.get('zone_type')
            
            if curr_type == 'subzone' and next_type == 'subzone':
                # Precise movement tracking
                movement_patterns['precise_movements'].append({
                    'from': curr_zone_info.get('specific_location'),
                    'to': next_zone_info.get('specific_location'),
                    'duration': (next_event['timestamp'] - curr['timestamp']).total_seconds()
                })
            elif curr_type == 'full' and next_type == 'full':
                # General movement between rooms
                movement_patterns['general_movements'].append({
                    'from_room': curr.get('room'),
                    'to_room': next_event.get('room'),
                    'precision': 'low'
                })
            else:
                # Hybrid - entering/leaving specific zones
                movement_patterns['hybrid_movements'].append({
                    'transition_type': f"{curr_type}_to_{next_type}",
                    'details': (curr, next_event)
                })
        
        return movement_patterns
    
    def select_significant_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Select only statistically significant features"""
        significant = {}
        
        for feature_name, feature_value in features.items():
            if self.is_feature_significant(feature_name, feature_value):
                significant[feature_name] = feature_value
        
        return significant
    
    def is_feature_significant(self, feature_name: str, feature_value: Any) -> bool:
        """Check if a feature is statistically significant"""
        # Simplified significance test
        if isinstance(feature_value, (list, dict)):
            return len(feature_value) > 0
        elif isinstance(feature_value, (int, float)):
            return abs(feature_value) > 0.01
        else:
            return feature_value is not None


class InteractionDiscovery:
    """Detects non-linear interactions between sensors"""
    
    def __init__(self):
        self.interaction_cache = {}
    
    def detect_interactions(self, sensor_stream: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect non-linear interactions between sensors"""
        interactions = {}
        
        # Group events by timestamp windows
        time_windows = self.group_by_time_windows(sensor_stream, window_size=60)  # 60-second windows
        
        for window_time, events in time_windows.items():
            if len(events) >= 2:
                # Detect concurrent sensor activations
                concurrent_interactions = self.find_concurrent_activations(events)
                if concurrent_interactions:
                    interactions[f"concurrent_{window_time}"] = concurrent_interactions
                
                # Detect sequential patterns
                sequential_interactions = self.find_sequential_patterns(events)
                if sequential_interactions:
                    interactions[f"sequential_{window_time}"] = sequential_interactions
        
        return interactions
    
    def group_by_time_windows(self, events: List[Dict[str, Any]], window_size: int) -> Dict[str, List[Dict]]:
        """Group events into time windows"""
        windows = defaultdict(list)
        
        for event in events:
            timestamp = event['timestamp']
            window_key = int(timestamp.timestamp() // window_size) * window_size
            windows[str(window_key)].append(event)
        
        return windows
    
    def find_concurrent_activations(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find sensors that activate simultaneously"""
        concurrent = []
        
        # Group by exact timestamp
        timestamp_groups = defaultdict(list)
        for event in events:
            timestamp_groups[event['timestamp']].append(event)
        
        for timestamp, simultaneous_events in timestamp_groups.items():
            if len(simultaneous_events) >= 2:
                entity_ids = [e['entity_id'] for e in simultaneous_events]
                concurrent.append({
                    'timestamp': timestamp,
                    'entities': entity_ids,
                    'interaction_type': 'simultaneous_activation'
                })
        
        return concurrent
    
    def find_sequential_patterns(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find sequential activation patterns"""
        sequential = []
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda x: x['timestamp'])
        
        for i in range(len(sorted_events) - 1):
            curr = sorted_events[i]
            next_event = sorted_events[i + 1]
            
            time_diff = (next_event['timestamp'] - curr['timestamp']).total_seconds()
            
            # Look for patterns within 30 seconds
            if 0 < time_diff <= 30:
                sequential.append({
                    'first': curr['entity_id'],
                    'second': next_event['entity_id'],
                    'delay': time_diff,
                    'interaction_type': 'sequential_activation'
                })
        
        return sequential


class TemporalMiner:
    """Mines variable-length temporal patterns"""
    
    def __init__(self):
        self.pattern_cache = {}
    
    def mine_patterns(self, sensor_stream: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find variable-length temporal patterns"""
        patterns = {
            'periodic_patterns': self.find_periodic_patterns(sensor_stream),
            'burst_patterns': self.find_burst_patterns(sensor_stream),
            'quiet_periods': self.find_quiet_periods(sensor_stream)
        }
        
        return patterns
    
    def find_periodic_patterns(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find repeating patterns at different time scales"""
        periodic = []
        
        # Group events by entity
        entity_events = defaultdict(list)
        for event in events:
            entity_events[event['entity_id']].append(event['timestamp'])
        
        for entity_id, timestamps in entity_events.items():
            if len(timestamps) >= 3:
                # Calculate intervals between events
                intervals = []
                for i in range(len(timestamps) - 1):
                    interval = (timestamps[i + 1] - timestamps[i]).total_seconds()
                    intervals.append(interval)
                
                # Look for repeating intervals
                if self.has_periodic_pattern(intervals):
                    periodic.append({
                        'entity_id': entity_id,
                        'pattern_type': 'periodic',
                        'average_interval': np.mean(intervals),
                        'interval_variance': np.var(intervals)
                    })
        
        return periodic
    
    def has_periodic_pattern(self, intervals: List[float]) -> bool:
        """Check if intervals show periodic behavior"""
        if len(intervals) < 3:
            return False
        
        # Simple check: low variance relative to mean
        mean_interval = np.mean(intervals)
        variance = np.var(intervals)
        
        if mean_interval > 0:
            coefficient_of_variation = np.sqrt(variance) / mean_interval
            return coefficient_of_variation < 0.3  # 30% variation threshold
        
        return False
    
    def find_burst_patterns(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find periods of high activity"""
        bursts = []
        
        # Group events by 5-minute windows
        time_windows = defaultdict(int)
        for event in events:
            window_key = int(event['timestamp'].timestamp() // 300) * 300  # 5-minute windows
            time_windows[window_key] += 1
        
        # Find windows with unusually high activity
        activity_counts = list(time_windows.values())
        if activity_counts:
            mean_activity = np.mean(activity_counts)
            std_activity = np.std(activity_counts)
            
            burst_threshold = mean_activity + 2 * std_activity
            
            for window_time, count in time_windows.items():
                if count > burst_threshold:
                    bursts.append({
                        'timestamp': datetime.fromtimestamp(window_time),
                        'activity_count': count,
                        'pattern_type': 'burst'
                    })
        
        return bursts
    
    def find_quiet_periods(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find periods of low activity"""
        quiet_periods = []
        
        if len(events) < 2:
            return quiet_periods
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda x: x['timestamp'])
        
        # Find large gaps between events
        for i in range(len(sorted_events) - 1):
            curr = sorted_events[i]
            next_event = sorted_events[i + 1]
            
            gap = (next_event['timestamp'] - curr['timestamp']).total_seconds()
            
            # Consider gaps > 1 hour as quiet periods
            if gap > 3600:
                quiet_periods.append({
                    'start': curr['timestamp'],
                    'end': next_event['timestamp'],
                    'duration': gap,
                    'pattern_type': 'quiet_period'
                })
        
        return quiet_periods


class AdaptiveFeatureSelector:
    """
    Uses SHAP values in online manner for feature selection
    Implements the exact approach from CLAUDE.md
    """
    
    def __init__(self):
        self.feature_scores = defaultdict(lambda: stats.Mean())
        
    def update_importance(self, features: Dict[str, Any], prediction_error: float):
        """
        Use SHAP values in online manner
        Update feature importance based on prediction errors
        """
        for feature, value in features.items():
            contribution = self.estimate_contribution(feature, value, prediction_error)
            self.feature_scores[feature].update(contribution)
        
        # Prune features that don't contribute
        self.prune_useless_features()
    
    def estimate_contribution(self, feature: str, value: Any, prediction_error: float) -> float:
        """Estimate feature contribution using simplified SHAP-like approach"""
        # Simplified contribution estimation
        if isinstance(value, (int, float)):
            # Higher values with lower errors indicate positive contribution
            contribution = abs(value) * (1.0 - min(1.0, abs(prediction_error)))
        elif isinstance(value, (list, dict)):
            # For complex features, use size as proxy for importance
            size = len(value) if hasattr(value, '__len__') else 1
            contribution = size * (1.0 - min(1.0, abs(prediction_error)))
        else:
            contribution = 1.0 - min(1.0, abs(prediction_error))
        
        return contribution
    
    def prune_useless_features(self):
        """Remove features that consistently show low importance"""
        features_to_remove = []
        
        for feature_name, score_tracker in self.feature_scores.items():
            if score_tracker.n > 10:  # Need minimum samples
                avg_score = score_tracker.get()
                if avg_score < 0.1:  # Low importance threshold
                    features_to_remove.append(feature_name)
        
        for feature in features_to_remove:
            del self.feature_scores[feature]
            logger.info(f"Pruned low-importance feature: {feature}")
    
    def get_top_features(self, n: int = 50) -> List[Tuple[str, float]]:
        """
        Return dynamically selected best features
        Implements the exact method from CLAUDE.md
        """
        feature_importance_pairs = []
        
        for feature_name, score_tracker in self.feature_scores.items():
            if score_tracker.n > 0:
                importance = score_tracker.get()
                feature_importance_pairs.append((feature_name, importance))
        
        # Sort by importance (descending) and return top N
        sorted_features = sorted(feature_importance_pairs, 
                               key=lambda x: x[1], 
                               reverse=True)
        
        return sorted_features[:n]
    
    def get_feature_importance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of feature importance tracking"""
        summary = {}
        
        for feature_name, score_tracker in self.feature_scores.items():
            summary[feature_name] = {
                'importance_score': score_tracker.get() if score_tracker.n > 0 else 0.0,
                'sample_count': score_tracker.n,
                'status': 'active' if score_tracker.get() >= 0.1 else 'low_importance'
            }
        
        return summary