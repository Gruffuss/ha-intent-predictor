"""
Bathroom Occupancy Prediction - Specialized logic for bathroom inference
Implements the exact BathroomOccupancyPredictor from CLAUDE.md
"""

import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from river import stats

logger = logging.getLogger(__name__)


class BathroomOccupancyPredictor:
    """
    Infers bathroom occupancy from entrance zones and door sensors
    Implements the exact approach from CLAUDE.md
    """
    
    def __init__(self):
        self.bathroom_state = {
            'bathroom': {
                'occupied': False, 
                'entry_time': None, 
                'door_closed': False,
                'entrance_triggered': False
            },
            'small_bathroom': {
                'occupied': False, 
                'entry_time': None, 
                'door_closed': False,
                'entrance_triggered': False
            }
        }
        
        self.duration_learner = DurationPatternLearner()
        self.occupancy_patterns = {
            'duration_by_hour': defaultdict(lambda: stats.Mean()),
            'door_usage_probability': defaultdict(float),
            'pre_bathroom_activity': defaultdict(lambda: defaultdict(int)),
            'post_bathroom_activity': defaultdict(lambda: defaultdict(int))
        }
        
        logger.info("Initialized bathroom occupancy predictor")
    
    def process_bathroom_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Infer bathroom occupancy from entrance zones and door sensors
        Implements the exact logic from CLAUDE.md
        """
        room = self.identify_bathroom(event.get('entity_id', ''))
        if not room:
            return None
        
        entity_id = event.get('entity_id', '')
        timestamp = event.get('timestamp', datetime.now())
        state = event.get('state', 'off')
        
        result = None
        
        if 'entrance' in entity_id:
            # Someone triggered entrance zone
            if state in ['on', 'detected']:
                result = self.handle_entrance_trigger(room, timestamp)
            else:
                # Entrance zone cleared - might be leaving
                result = self.handle_entrance_clear(room, timestamp)
                
        elif 'door_sensor_contact' in entity_id:
            # Door state changed
            door_closed = (state == 'off')  # Contact sensor: off = closed
            self.bathroom_state[room]['door_closed'] = door_closed
            result = self.update_occupancy_logic(room, timestamp)
        
        # Learn from this event
        if result:
            self.learn_bathroom_patterns(room, event, result)
        
        return result
    
    def identify_bathroom(self, entity_id: str) -> Optional[str]:
        """Identify which bathroom from entity ID"""
        if 'small_bathroom' in entity_id or 'small_bath' in entity_id:
            return 'small_bathroom'
        elif 'bathroom' in entity_id:
            return 'bathroom'
        return None
    
    def handle_entrance_trigger(self, room: str, timestamp: datetime) -> Dict[str, Any]:
        """
        Someone approached bathroom entrance
        Implements the exact logic from CLAUDE.md
        """
        state = self.bathroom_state[room]
        
        if not state['occupied']:
            # Likely entering
            state['occupied'] = True
            state['entry_time'] = timestamp
            state['entrance_triggered'] = True
            
            # Learn typical duration for this time of day
            predicted_duration = self.duration_learner.predict_duration(
                room, timestamp, self.get_context_features(timestamp)
            )
            
            confidence = self.calculate_entry_confidence(room, timestamp)
            
            logger.info(f"Bathroom entry detected: {room} at {timestamp}")
            
            return {
                'room': room,
                'occupied': True,
                'entry_type': 'entrance_trigger',
                'predicted_duration': predicted_duration,
                'confidence': confidence,
                'timestamp': timestamp
            }
        else:
            # Already occupied, might be false trigger or someone else
            return {
                'room': room,
                'occupied': True,
                'entry_type': 'already_occupied',
                'confidence': 0.3,  # Lower confidence for ambiguous situation
                'timestamp': timestamp
            }
    
    def handle_entrance_clear(self, room: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Handle entrance zone clearing"""
        state = self.bathroom_state[room]
        
        if state['occupied'] and state['entry_time']:
            # Check if enough time passed for bathroom use
            duration = (timestamp - state['entry_time']).total_seconds()
            typical_duration = self.get_typical_duration(room, timestamp)
            
            # If person was inside long enough, they might be leaving
            if duration > typical_duration * 0.5:  # At least 50% of typical duration
                # Check door state for additional confirmation
                if not state['door_closed']:  # Door is open
                    state['occupied'] = False
                    state['entrance_triggered'] = False
                    
                    logger.info(f"Bathroom exit detected: {room} after {duration:.1f}s")
                    
                    return {
                        'room': room,
                        'occupied': False,
                        'exit_type': 'entrance_clear',
                        'duration': duration,
                        'confidence': 0.7,
                        'timestamp': timestamp
                    }
        
        return None
    
    def update_occupancy_logic(self, room: str, timestamp: datetime) -> Dict[str, Any]:
        """
        Complex logic combining entrance and door states
        Implements the exact approach from CLAUDE.md
        """
        state = self.bathroom_state[room]
        
        # If door just closed after entrance trigger, high confidence occupied
        if state['door_closed'] and state['entrance_triggered']:
            state['occupied'] = True
            
            logger.debug(f"High confidence occupancy: {room} (door closed after entrance)")
            
            return {
                'room': room,
                'occupied': True, 
                'confidence': 0.95,
                'reason': 'door_closed_after_entrance',
                'timestamp': timestamp
            }
            
        # If door opened after being closed, might be leaving
        if not state['door_closed'] and state['occupied'] and state['entry_time']:
            # Check if enough time passed for typical bathroom use
            duration = (timestamp - state['entry_time']).total_seconds()
            typical_duration = self.get_typical_duration(room, timestamp)
            
            if duration > typical_duration * 0.8:
                # Likely leaving
                state['occupied'] = False
                state['entrance_triggered'] = False
                
                logger.info(f"Exit inferred from door opening: {room} after {duration:.1f}s")
                
                return {
                    'room': room,
                    'occupied': False, 
                    'confidence': 0.8,
                    'reason': 'door_opened_after_duration',
                    'duration': duration,
                    'timestamp': timestamp
                }
        
        # Default case - maintain current state
        return {
            'room': room,
            'occupied': state['occupied'],
            'confidence': 0.5,
            'reason': 'state_maintained',
            'timestamp': timestamp
        }
    
    def calculate_entry_confidence(self, room: str, timestamp: datetime) -> float:
        """Calculate confidence for bathroom entry prediction"""
        hour = timestamp.hour
        
        # Higher confidence during typical bathroom hours
        if 6 <= hour <= 9 or 21 <= hour <= 23:  # Morning and evening routines
            base_confidence = 0.8
        elif 12 <= hour <= 14:  # Lunch time
            base_confidence = 0.6
        else:
            base_confidence = 0.5
        
        # Adjust based on historical patterns
        if room in self.occupancy_patterns['duration_by_hour']:
            hourly_stats = self.occupancy_patterns['duration_by_hour'][hour]
            if hourly_stats.n > 5:  # Have enough samples
                # Higher confidence if we've seen this pattern before
                base_confidence += 0.1
        
        return min(0.95, base_confidence)
    
    def get_typical_duration(self, room: str, timestamp: datetime) -> float:
        """Get typical bathroom duration for this time"""
        hour = timestamp.hour
        
        # Check learned patterns first
        if hour in self.occupancy_patterns['duration_by_hour']:
            hourly_stats = self.occupancy_patterns['duration_by_hour'][hour]
            if hourly_stats.n > 3:
                return hourly_stats.get()
        
        # Default durations by time of day
        if 6 <= hour <= 9:  # Morning routine
            return 180.0  # 3 minutes
        elif 21 <= hour <= 23:  # Evening routine
            return 300.0  # 5 minutes
        else:
            return 120.0  # 2 minutes default
    
    def get_context_features(self, timestamp: datetime) -> Dict[str, Any]:
        """Get contextual features for duration prediction"""
        return {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_weekend': timestamp.weekday() >= 5,
            'is_morning_routine': 6 <= timestamp.hour <= 9,
            'is_evening_routine': 21 <= timestamp.hour <= 23
        }
    
    def learn_bathroom_patterns(self, 
                              room: str, 
                              event: Dict[str, Any], 
                              result: Dict[str, Any]):
        """
        Learn patterns specific to bathroom usage
        Implements the exact learning from CLAUDE.md
        """
        timestamp = event.get('timestamp', datetime.now())
        hour = timestamp.hour
        
        # Learn duration patterns
        if result.get('occupied') == False and 'duration' in result:
            duration = result['duration']
            self.occupancy_patterns['duration_by_hour'][hour].update(duration)
        
        # Learn door usage patterns
        if 'door_sensor_contact' in event.get('entity_id', ''):
            door_closed = event.get('state') == 'off'
            
            # Update door usage probability
            current_prob = self.occupancy_patterns['door_usage_probability'][room]
            if door_closed:
                # Person closed door
                self.occupancy_patterns['door_usage_probability'][room] = current_prob * 0.9 + 0.1 * 1.0
            else:
                # Person left door open
                self.occupancy_patterns['door_usage_probability'][room] = current_prob * 0.9 + 0.1 * 0.0
    
    def predict_bathroom_occupancy(self, 
                                 room: str, 
                                 horizon_minutes: int) -> Dict[str, Any]:
        """Predict bathroom occupancy for given horizon"""
        current_state = self.bathroom_state[room]
        current_time = datetime.now()
        
        if current_state['occupied'] and current_state['entry_time']:
            # Currently occupied - predict when they'll leave
            time_elapsed = (current_time - current_state['entry_time']).total_seconds()
            typical_duration = self.get_typical_duration(room, current_time)
            
            # Probability they'll still be there
            if time_elapsed >= typical_duration:
                # They've been there longer than typical
                probability = max(0.1, 1.0 - (time_elapsed - typical_duration) / typical_duration)
            else:
                # Still within typical duration
                probability = 1.0 - (time_elapsed / typical_duration) * 0.3
            
            return {
                'room': room,
                'horizon_minutes': horizon_minutes,
                'probability': min(0.95, probability),
                'confidence': 0.8,
                'reasoning': 'currently_occupied_duration_based'
            }
        
        else:
            # Not currently occupied - predict based on patterns
            target_time = current_time + timedelta(minutes=horizon_minutes)
            target_hour = target_time.hour
            
            # Base probability from historical patterns
            base_prob = self.get_historical_probability(room, target_hour)
            
            # Adjust for time since last use
            time_since_last_use = self.get_time_since_last_use(room)
            if time_since_last_use and time_since_last_use > 3600:  # More than 1 hour
                base_prob *= 1.2  # Slightly more likely
            
            return {
                'room': room,
                'horizon_minutes': horizon_minutes,
                'probability': min(0.8, base_prob),  # Cap at 80% for unoccupied
                'confidence': 0.6,
                'reasoning': 'pattern_based_prediction'
            }
    
    def get_historical_probability(self, room: str, hour: int) -> float:
        """Get historical probability of bathroom use at this hour"""
        if hour in self.occupancy_patterns['duration_by_hour']:
            hourly_stats = self.occupancy_patterns['duration_by_hour'][hour]
            if hourly_stats.n > 0:
                # If we have data for this hour, there's some probability
                return min(0.3, hourly_stats.n / 100.0)  # Scale by sample count
        
        # Default probabilities by hour
        if 6 <= hour <= 9:  # Morning
            return 0.4
        elif 21 <= hour <= 23:  # Evening
            return 0.3
        elif 12 <= hour <= 14:  # Lunch
            return 0.2
        else:
            return 0.1
    
    def get_time_since_last_use(self, room: str) -> Optional[float]:
        """Get time since bathroom was last used"""
        state = self.bathroom_state[room]
        if state['entry_time'] and not state['occupied']:
            return (datetime.now() - state['entry_time']).total_seconds()
        return None
    
    def get_bathroom_state_summary(self) -> Dict[str, Any]:
        """Get summary of bathroom states for monitoring"""
        summary = {}
        
        for room, state in self.bathroom_state.items():
            summary[room] = {
                'occupied': state['occupied'],
                'door_closed': state['door_closed'],
                'time_since_entry': None,
                'predicted_exit_time': None
            }
            
            if state['entry_time']:
                time_since_entry = (datetime.now() - state['entry_time']).total_seconds()
                summary[room]['time_since_entry'] = time_since_entry
                
                if state['occupied']:
                    typical_duration = self.get_typical_duration(room, datetime.now())
                    predicted_exit = state['entry_time'] + timedelta(seconds=typical_duration)
                    summary[room]['predicted_exit_time'] = predicted_exit
        
        return summary
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of learned bathroom patterns"""
        return {
            'duration_patterns': {
                hour: {
                    'average_duration': stats_obj.get() if stats_obj.n > 0 else 0,
                    'sample_count': stats_obj.n
                }
                for hour, stats_obj in self.occupancy_patterns['duration_by_hour'].items()
                if stats_obj.n > 0
            },
            'door_usage': dict(self.occupancy_patterns['door_usage_probability'])
        }


class DurationPatternLearner:
    """
    Learns bathroom duration patterns by context
    """
    
    def __init__(self):
        self.duration_patterns = {
            'by_hour': defaultdict(lambda: stats.Mean()),
            'by_day_of_week': defaultdict(lambda: stats.Mean()),
            'by_context': defaultdict(lambda: stats.Mean())
        }
    
    def learn_duration(self, 
                      room: str, 
                      timestamp: datetime, 
                      duration: float,
                      context: Dict[str, Any]):
        """Learn duration pattern"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Update patterns
        self.duration_patterns['by_hour'][hour].update(duration)
        self.duration_patterns['by_day_of_week'][day_of_week].update(duration)
        
        # Context-specific patterns
        if context.get('is_morning_routine'):
            self.duration_patterns['by_context']['morning'].update(duration)
        elif context.get('is_evening_routine'):
            self.duration_patterns['by_context']['evening'].update(duration)
    
    def predict_duration(self, 
                        room: str, 
                        timestamp: datetime,
                        context: Dict[str, Any]) -> float:
        """Predict bathroom duration based on learned patterns"""
        hour = timestamp.hour
        
        # Try hour-specific pattern first
        if hour in self.duration_patterns['by_hour']:
            hourly_stats = self.duration_patterns['by_hour'][hour]
            if hourly_stats.n > 3:
                return hourly_stats.get()
        
        # Try context-specific patterns
        if context.get('is_morning_routine') and 'morning' in self.duration_patterns['by_context']:
            morning_stats = self.duration_patterns['by_context']['morning']
            if morning_stats.n > 3:
                return morning_stats.get()
        
        if context.get('is_evening_routine') and 'evening' in self.duration_patterns['by_context']:
            evening_stats = self.duration_patterns['by_context']['evening']
            if evening_stats.n > 3:
                return evening_stats.get()
        
        # Default durations
        if 6 <= hour <= 9:  # Morning
            return 180.0  # 3 minutes
        elif 21 <= hour <= 23:  # Evening
            return 300.0  # 5 minutes
        else:
            return 120.0  # 2 minutes