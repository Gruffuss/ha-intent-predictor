"""
Cat and Anomaly Detection - Adaptive detection without assumptions
Implements the sophisticated cat detection system from CLAUDE.md
"""

import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from river import stats
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


class AdaptiveCatDetector:
    """
    Learn what's normal vs abnormal without hardcoding
    Special handling for multi-person households
    Implements the exact approach from CLAUDE.md
    """
    
    def __init__(self):
        self.movement_clusterer = IncrementalDBSCAN()
        self.impossible_sequences = set()
        
        # Person-specific patterns as specified in CLAUDE.md
        self.person_specific_patterns = {
            'anca': {
                'common_zones': set(),
                'movement_speed': stats.Mean()
            },
            'vladimir': {
                'common_zones': set(), 
                'movement_speed': stats.Mean()
            }
        }
        
        # Track door constraints
        self.door_constraints = {
            'bathroom': 'binary_sensor.bathroom_door_sensor_contact',
            'small_bathroom': 'binary_sensor.small_bathroom_door_sensor_contact',
            'bedroom': 'binary_sensor.bedroom_door_sensor_contact',
            'office': 'binary_sensor.office_door_sensor_contact'
        }
        
        # Minimum transition times between zones (from CLAUDE.md)
        self.min_transition_times = {
            ('bedroom_anca_side', 'kitchen_stove'): 5,  # Across house
            ('bedroom_vladimir_side', 'kitchen_sink'): 5,
            ('office_anca_desk', 'bedroom_vladimir_side'): 3,  # Different rooms
            ('office_vladimir_desk', 'bedroom_anca_side'): 3,
            ('livingroom_couch', 'bedroom_floor'): 4,
            # Same room transitions are faster
            ('kitchen_stove', 'kitchen_sink'): 1,
            ('bedroom_anca_side', 'bedroom_vladimir_side'): 2,
            ('office_anca_desk', 'office_vladimir_desk'): 2,
        }
        
        logger.info("Initialized adaptive cat detector")
    
    def learn_movement_patterns(self, sensor_sequence: List[Dict[str, Any]]):
        """
        Learn what's normal vs abnormal without hardcoding
        Implements exact algorithm from CLAUDE.md
        """
        if len(sensor_sequence) < 2:
            return
        
        # Extract movement velocity between sensors
        velocities = self.calculate_velocities(sensor_sequence)
        
        # Check if movement is between person-specific zones
        person = self.identify_person_from_zones(sensor_sequence)
        if person:
            self.person_specific_patterns[person]['movement_speed'].update(velocities)
            
            # Track common zones for this person
            for event in sensor_sequence:
                zone_info = event.get('derived', {}).get('zone_info', {})
                zone = zone_info.get('zone', '')
                if zone:
                    self.person_specific_patterns[person]['common_zones'].add(zone)
        
        # Cluster movements - outliers might be cats
        cluster_label = self.movement_clusterer.partial_fit_predict(velocities)
        
        if cluster_label == -1:  # Outlier
            # This might be cat movement
            self.analyze_outlier(sensor_sequence)
    
    def calculate_velocities(self, sequence: List[Dict[str, Any]]) -> List[float]:
        """Calculate movement velocities between consecutive sensors"""
        velocities = []
        
        for i in range(len(sequence) - 1):
            curr = sequence[i]
            next_event = sequence[i + 1]
            
            curr_time = curr.get('timestamp', datetime.now())
            next_time = next_event.get('timestamp', datetime.now())
            
            if isinstance(curr_time, str):
                curr_time = datetime.fromisoformat(curr_time.replace('Z', '+00:00'))
            if isinstance(next_time, str):
                next_time = datetime.fromisoformat(next_time.replace('Z', '+00:00'))
            
            time_diff = (next_time - curr_time).total_seconds()
            
            if time_diff > 0:
                # Simplified velocity: 1/time (faster movement = higher velocity)
                velocity = 1.0 / time_diff
                velocities.append(velocity)
        
        return velocities
    
    def identify_person_from_zones(self, sequence: List[Dict[str, Any]]) -> Optional[str]:
        """
        Identify if movement is person-specific based on zones
        Implements exact logic from CLAUDE.md
        """
        zones = []
        for event in sequence:
            zone_info = event.get('derived', {}).get('zone_info', {})
            zone = zone_info.get('zone', '')
            if zone:
                zones.append(zone)
        
        # Check for person-specific zones
        if any('anca' in zone for zone in zones):
            return 'anca'
        elif any('vladimir' in zone for zone in zones):
            return 'vladimir'
        
        return None
    
    def analyze_outlier(self, sequence: List[Dict[str, Any]]):
        """
        Determine if outlier is cat vs human edge case
        Implements exact analysis from CLAUDE.md
        """
        logger.debug(f"Analyzing outlier sequence with {len(sequence)} events")
        
        # Look for patterns like:
        # - Simultaneous triggers in distant rooms
        # - Repeated fast transitions
        # - Sensors that only trigger with other cat-like patterns
        # - Movement through multiple zones too quickly
        # - Presence without door opening (for rooms with doors)
        
        features = self.extract_anomaly_features(sequence)
        
        # Special check for impossible human movements
        if self.is_physically_impossible(sequence):
            self.tag_as_cat_activity(sequence)
            return
        
        # Let the system learn what's cat vs unusual human
        if self.is_consistent_with_previous_cats(features):
            self.tag_as_cat_activity(sequence)
    
    def extract_anomaly_features(self, sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features that help distinguish cat vs human movement"""
        features = {
            'sequence_length': len(sequence),
            'total_duration': 0.0,
            'rooms_visited': set(),
            'zone_types': set(),
            'simultaneous_triggers': 0,
            'fast_transitions': 0,
            'door_violations': 0
        }
        
        if not sequence:
            return features
        
        # Calculate total duration
        first_time = sequence[0].get('timestamp', datetime.now())
        last_time = sequence[-1].get('timestamp', datetime.now())
        
        if isinstance(first_time, str):
            first_time = datetime.fromisoformat(first_time.replace('Z', '+00:00'))
        if isinstance(last_time, str):
            last_time = datetime.fromisoformat(last_time.replace('Z', '+00:00'))
        
        features['total_duration'] = (last_time - first_time).total_seconds()
        
        # Analyze each event
        for i, event in enumerate(sequence):
            # Track rooms and zones
            room = event.get('room', '')
            if room:
                features['rooms_visited'].add(room)
            
            zone_info = event.get('derived', {}).get('zone_info', {})
            zone_type = zone_info.get('zone_type', '')
            if zone_type:
                features['zone_types'].add(zone_type)
            
            # Check for simultaneous triggers
            if i > 0:
                prev_event = sequence[i - 1]
                time_diff = self._get_time_diff(prev_event, event)
                
                if time_diff < 0.5:  # Less than 0.5 seconds
                    features['simultaneous_triggers'] += 1
                
                if time_diff < 2.0:  # Less than 2 seconds
                    features['fast_transitions'] += 1
        
        return features
    
    def is_physically_impossible(self, sequence: List[Dict[str, Any]]) -> bool:
        """
        Check for movements that violate physics for humans
        Implements exact logic from CLAUDE.md
        """
        for i in range(len(sequence) - 1):
            curr = sequence[i]
            next_event = sequence[i + 1]
            
            time_diff = self._get_time_diff(curr, next_event)
            
            # Check door constraints for bathrooms
            next_room = next_event.get('room', '')
            if next_room in ['bathroom', 'small_bathroom']:
                door_entity = self.door_constraints.get(next_room)
                if door_entity:
                    curr_time = curr.get('timestamp', datetime.now())
                    next_time = next_event.get('timestamp', datetime.now())
                    
                    if not self.was_door_opened(door_entity, curr_time, next_time):
                        logger.debug(f"Door violation detected for {next_room}")
                        return True
            
            # Analyze movement pattern
            movement = self.analyze_movement(curr, next_event, time_diff)
            
            if movement['impossible']:
                logger.debug(f"Impossible movement detected: {movement['reason']}")
                return True
        
        return False
    
    def analyze_movement(self, curr_event: Dict[str, Any], next_event: Dict[str, Any], time_diff: float) -> Dict[str, Any]:
        """
        Analyze if movement is possible for humans
        Implements exact analysis from CLAUDE.md
        """
        curr_zone = curr_event.get('derived', {}).get('zone_info', {})
        next_zone = next_event.get('derived', {}).get('zone_info', {})
        
        curr_room = curr_zone.get('room', '')
        next_room = next_zone.get('room', '')
        
        # Different room transitions
        if curr_room != next_room and curr_room and next_room:
            # Check if transition time is reasonable
            min_transition_time = self.get_min_transition_time(curr_room, next_room)
            
            if time_diff < min_transition_time:
                # Too fast for human movement
                return {'impossible': True, 'reason': 'inter_room_speed'}
        
        # Same room but different zones
        elif curr_room == next_room and curr_zone.get('zone') != next_zone.get('zone'):
            curr_zone_name = curr_zone.get('zone', '')
            next_zone_name = next_zone.get('zone', '')
            
            # Subzone to subzone movement
            if (curr_zone.get('zone_type') == 'subzone' and 
                next_zone.get('zone_type') == 'subzone'):
                
                # Check specific subzone transitions using CLAUDE.md constraints
                if self.is_subzone_transition_impossible(curr_zone_name, next_zone_name, time_diff):
                    return {'impossible': True, 'reason': 'subzone_speed'}
            
            # Full zone to subzone or vice versa
            elif curr_zone.get('zone_type') != next_zone.get('zone_type'):
                # This is normal - person moving from general area to specific location
                return {'impossible': False, 'reason': 'normal_precision_change'}
        
        return {'impossible': False}
    
    def is_subzone_transition_impossible(self, zone1: str, zone2: str, time_seconds: float) -> bool:
        """
        Check if specific subzone transition is too fast
        Uses exact transition times from CLAUDE.md
        """
        # Check both directions
        min_time = self.min_transition_times.get((zone1, zone2)) or \
                  self.min_transition_times.get((zone2, zone1))
        
        if min_time and time_seconds < min_time:
            return True
        
        # Check for physically impossible same-instant triggers
        if time_seconds < 0.5 and zone1 != zone2:
            # Different zones triggered within 0.5 seconds - likely cat
            return True
        
        return False
    
    def get_min_transition_time(self, room1: str, room2: str) -> float:
        """Get minimum transition time between rooms"""
        # General inter-room transition times
        inter_room_times = {
            ('bedroom', 'kitchen'): 4.0,
            ('bedroom', 'office'): 3.0,
            ('office', 'kitchen'): 3.0,
            ('living_kitchen', 'bedroom'): 4.0,
            ('living_kitchen', 'office'): 3.0,
        }
        
        return (inter_room_times.get((room1, room2)) or 
                inter_room_times.get((room2, room1)) or 
                2.0)  # Default minimum
    
    def was_door_opened(self, door_entity: str, start_time: datetime, end_time: datetime) -> bool:
        """
        Check if door was opened during time window
        In full implementation, would query historical data
        """
        # Placeholder - would check actual door sensor history
        logger.debug(f"Checking door {door_entity} between {start_time} and {end_time}")
        return True  # Assume door was opened for now
    
    def tag_as_cat_activity(self, sequence: List[Dict[str, Any]]):
        """Tag sequence as cat activity for learning"""
        sequence_signature = self.create_sequence_signature(sequence)
        self.impossible_sequences.add(sequence_signature)
        
        logger.info(f"Tagged sequence as cat activity: {sequence_signature}")
    
    def is_consistent_with_previous_cats(self, features: Dict[str, Any]) -> bool:
        """Check if features match previously identified cat patterns"""
        # Simplified check - in full implementation would use ML
        if (features['fast_transitions'] > 2 or 
            features['simultaneous_triggers'] > 1 or
            features['total_duration'] < 1.0):
            return True
        
        return False
    
    def create_sequence_signature(self, sequence: List[Dict[str, Any]]) -> str:
        """Create unique signature for a sequence"""
        parts = []
        for event in sequence:
            room = event.get('room', 'unknown')
            state = event.get('state', 'unknown')
            parts.append(f"{room}:{state}")
        
        return "->".join(parts)
    
    def _get_time_diff(self, event1: Dict[str, Any], event2: Dict[str, Any]) -> float:
        """Get time difference between two events in seconds"""
        time1 = event1.get('timestamp', datetime.now())
        time2 = event2.get('timestamp', datetime.now())
        
        if isinstance(time1, str):
            time1 = datetime.fromisoformat(time1.replace('Z', '+00:00'))
        if isinstance(time2, str):
            time2 = datetime.fromisoformat(time2.replace('Z', '+00:00'))
        
        return abs((time2 - time1).total_seconds())
    
    def get_cat_detection_summary(self) -> Dict[str, Any]:
        """Get summary of cat detection performance"""
        return {
            'impossible_sequences_count': len(self.impossible_sequences),
            'person_patterns': {
                person: {
                    'common_zones_count': len(patterns['common_zones']),
                    'movement_samples': patterns['movement_speed'].n,
                    'avg_movement_speed': patterns['movement_speed'].get() if patterns['movement_speed'].n > 0 else 0.0
                }
                for person, patterns in self.person_specific_patterns.items()
            }
        }


class IncrementalDBSCAN:
    """
    Incremental clustering for movement pattern detection
    Simplified version of incremental DBSCAN
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 3):
        self.eps = eps
        self.min_samples = min_samples
        self.data_points = []
        self.cluster_labels = []
    
    def partial_fit_predict(self, new_points: List[float]) -> int:
        """Add new points and return cluster label (-1 for outlier)"""
        if not new_points:
            return -1
        
        # Add new points
        self.data_points.extend(new_points)
        
        # Perform clustering if we have enough points
        if len(self.data_points) >= self.min_samples:
            # Use regular DBSCAN for simplicity
            try:
                X = np.array(self.data_points).reshape(-1, 1)
                clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
                labels = clustering.fit_predict(X)
                
                # Return label for the last point
                return labels[-1] if len(labels) > 0 else -1
                
            except Exception as e:
                logger.warning(f"Clustering error: {e}")
                return -1
        
        return -1  # Not enough data yet


class OnlineAnomalyDetector:
    """
    Online anomaly detection for sensor patterns
    Learns normal patterns and detects deviations
    """
    
    def __init__(self):
        self.normal_patterns = defaultdict(lambda: stats.Mean())
        self.anomaly_threshold = 2.0  # Standard deviations
    
    def learn_normal_pattern(self, pattern_id: str, value: float):
        """Learn a normal pattern value"""
        self.normal_patterns[pattern_id].update(value)
    
    def is_anomalous(self, pattern_id: str, value: float) -> bool:
        """Check if a value is anomalous for given pattern"""
        if pattern_id not in self.normal_patterns:
            return False  # No baseline yet
        
        stats = self.normal_patterns[pattern_id]
        if stats.n < 10:  # Need minimum samples
            return False
        
        mean = stats.get()
        # Use simple variance estimation for anomaly detection
        std = 1.0  # Simplified for now
        
        # Check if value is beyond threshold
        z_score = abs(value - mean) / std
        return z_score > self.anomaly_threshold