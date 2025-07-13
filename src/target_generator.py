"""
Real target generation from historical Home Assistant data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
from collections import Counter

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
    
    logger.info(f"Generated {len(targets['longterm_room_labels'])} real training targets")
    logger.info(f"Room order: {list(rooms.keys())}")
    
    # Count distribution of room labels for analysis
    shortterm_counts = Counter(targets['shortterm_room_labels'])
    longterm_counts = Counter(targets['longterm_room_labels'])
    logger.info(f"Short-term room distribution: {dict(shortterm_counts)}")
    logger.info(f"Long-term room distribution: {dict(longterm_counts)}")
    
    return targets

def convert_to_room_events(events: pd.DataFrame, rooms: Dict[str, List[str]]) -> pd.DataFrame:
    """Convert zone-level events to room-level occupancy sessions with proper deduplication"""
    
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
    
    # Deduplicate overlapping room events (merge events in same room within 30 seconds)
    if len(room_events) > 0:
        deduplicated_events = []
        
        for room_name in rooms.keys():
            room_data = room_events[room_events['room'] == room_name].copy()
            if len(room_data) == 0:
                continue
                
            # Group events within 30 seconds of each other as single occupancy sessions
            room_data = room_data.sort_values('last_changed')
            session_events = []
            
            current_session_start = None
            for _, event in room_data.iterrows():
                if current_session_start is None:
                    current_session_start = event['last_changed']
                    session_events.append(event)
                else:
                    # If more than 30 seconds since last event, start new session
                    time_diff = (event['last_changed'] - current_session_start).total_seconds()
                    if time_diff > 30:
                        current_session_start = event['last_changed']
                        session_events.append(event)
                    # Otherwise, skip this event (it's part of the same session)
            
            deduplicated_events.extend(session_events)
        
        if deduplicated_events:
            room_events = pd.DataFrame(deduplicated_events).sort_values('last_changed').reset_index(drop=True)
        else:
            room_events = pd.DataFrame()
    
    return room_events

def create_aligned_targets(room_events: pd.DataFrame, rooms: Dict[str, List[str]], 
                          num_samples: int) -> Dict[str, np.ndarray]:
    """Create room occupancy classification targets"""
    
    if len(room_events) == 0:
        # Return 'none' labels for all samples
        return {
            'shortterm_room_labels': np.array(['none'] * num_samples),
            'longterm_room_labels': np.array(['none'] * num_samples)
        }
    
    start_time = room_events['last_changed'].min()
    end_time = room_events['last_changed'].max()
    
    # Create evenly spaced time points
    time_points = pd.date_range(start=start_time, end=end_time, periods=num_samples)
    
    room_names = list(rooms.keys())
    shortterm_labels = []
    longterm_labels = []
    
    for current_time in time_points:
        # Look ahead 15 minutes for short-term prediction
        shortterm_cutoff = current_time + timedelta(minutes=15)
        shortterm_events = room_events[
            (room_events['last_changed'] > current_time) &
            (room_events['last_changed'] <= shortterm_cutoff)
        ]
        
        # Look ahead 2 hours for long-term prediction
        longterm_cutoff = current_time + timedelta(hours=2)
        longterm_events = room_events[
            (room_events['last_changed'] > current_time) &
            (room_events['last_changed'] <= longterm_cutoff)
        ]
        
        # Determine most likely room for 15-minute window
        shortterm_room_scores = {}
        for room in room_names:
            room_events_15min = shortterm_events[shortterm_events['room'] == room]
            # Score based on number of occupancy sessions and recency
            if len(room_events_15min) > 0:
                # Weight recent events more heavily
                time_weights = []
                for _, event in room_events_15min.iterrows():
                    minutes_from_now = (event['last_changed'] - current_time).total_seconds() / 60
                    weight = max(0, 1 - (minutes_from_now / 15))  # Linear decay over 15 minutes
                    time_weights.append(weight)
                score = sum(time_weights)
            else:
                score = 0
            shortterm_room_scores[room] = score
        
        # Select room with highest score for short-term
        if max(shortterm_room_scores.values()) > 0.1:  # Minimum threshold
            shortterm_room = max(shortterm_room_scores, key=shortterm_room_scores.get)
        else:
            shortterm_room = 'none'
        
        # Determine most likely room for 2-hour window
        longterm_room_scores = {}
        for room in room_names:
            room_events_2h = longterm_events[longterm_events['room'] == room]
            if len(room_events_2h) > 0:
                # For longer predictions, weight by total activity
                score = len(room_events_2h)
            else:
                score = 0
            longterm_room_scores[room] = score
        
        # Select room with highest score for long-term
        if max(longterm_room_scores.values()) > 0:
            longterm_room = max(longterm_room_scores, key=longterm_room_scores.get)
        else:
            longterm_room = 'none'
        
        shortterm_labels.append(shortterm_room)
        longterm_labels.append(longterm_room)
    
    return {
        'shortterm_room_labels': np.array(shortterm_labels),
        'longterm_room_labels': np.array(longterm_labels)
    }