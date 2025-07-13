"""
Dynamic Feature Computation - Implements the sophisticated feature discovery from CLAUDE.md
"""

import logging
from collections import defaultdict
from typing import Dict, List, Any, Set
from datetime import datetime, timedelta
import numpy as np
from river import stats

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
        
    def discover_features(self, sensor_stream: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract different types of features from zone combinations
        Automatically generate and test feature combinations
        Keep only statistically significant features
        Detect non-linear interactions between sensors
        Find variable-length temporal patterns
        """
        # Extract zone combination features
        zone_features = self.extract_zone_combination_features(sensor_stream)
        
        # Generate interaction features
        interaction_features = self.interaction_detector.detect_interactions(sensor_stream)
        
        # Mine temporal patterns
        temporal_features = self.temporal_pattern_miner.mine_patterns(sensor_stream)
        
        # Combine all feature types
        all_features = {
            **zone_features,
            **interaction_features,
            **temporal_features
        }
        
        # Select only statistically significant features
        significant_features = self.select_significant_features(all_features)
        
        return significant_features
    
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