"""
Real target generation from historical Home Assistant data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

def analyze_actual_room_usage(events: pd.DataFrame, rooms: Dict[str, List[str]], 
                             num_features: int) -> Dict[str, np.ndarray]:
    """
    Analyze real historical data to create proper training targets
    """
    
    logger.info(f"Analyzing actual room usage patterns from {len(events)} events")
    
    # Convert events to room-level occupancy
    room_events = convert_to_room_events(events, rooms)
    
    # Create targets matching the number of feature samples
    targets = create_aligned_targets(room_events, rooms, num_features)
    
    logger.info(f"Generated {len(targets['longterm_occupancy'])} real training targets")
    logger.info(f"Room order: {list(rooms.keys())}")
    logger.info(f"Average room usage probabilities: {np.mean(targets['longterm_occupancy'], axis=0)}")
    
    return targets

def convert_to_room_events(events: pd.DataFrame, rooms: Dict[str, List[str]]) -> pd.DataFrame:
    """Convert zone-level events to room-level occupancy sessions"""
    
    # Create zone to room mapping
    zone_to_room = {}
    for room_name, zones in rooms.items():
        for zone in zones:
            zone_to_room[zone] = room_name
    
    # Filter to 'on' events and add room info
    room_events = events[events['state'] == 'on'].copy()
    room_events['room'] = room_events['entity_id'].map(zone_to_room)
    room_events = room_events.dropna(subset=['room'])
    
    # Sort by time
    room_events = room_events.sort_values('last_changed').reset_index(drop=True)
    
    return room_events

def create_aligned_targets(room_events: pd.DataFrame, rooms: Dict[str, List[str]], 
                          num_samples: int) -> Dict[str, np.ndarray]:
    """Create exactly num_samples targets to match features"""
    
    if len(room_events) == 0:
        # Return empty targets with correct shape
        return {'longterm_occupancy': np.zeros((num_samples, len(rooms)))}
    
    start_time = room_events['last_changed'].min()
    end_time = room_events['last_changed'].max()
    
    # Create evenly spaced time points
    time_points = pd.date_range(start=start_time, end=end_time, periods=num_samples)
    
    room_names = list(rooms.keys())
    targets = []
    
    for current_time in time_points:
        # Look ahead 2 hours
        future_cutoff = current_time + timedelta(hours=2)
        future_events = room_events[
            (room_events['last_changed'] > current_time) &
            (room_events['last_changed'] <= future_cutoff)
        ]
        
        # Create target for this time point
        target = np.zeros(len(room_names))
        
        for i, room in enumerate(room_names):
            room_future_events = future_events[future_events['room'] == room]
            if len(room_future_events) > 0:
                # Calculate probability based on activation count
                probability = min(1.0, len(room_future_events) * 0.1)
                target[i] = probability
        
        targets.append(target)
    
    return {'longterm_occupancy': np.array(targets)}