"""
Feature Engineering Pipeline for Home Assistant sensor data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

logger = logging.getLogger(__name__)

class FeatureEngine:
    """Extract ML features from Home Assistant sensor data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rooms = config['rooms']
        self.feature_windows = config['ml']['feature_windows']
        self.sequence_length = config['ml']['sequence_length']
        
        # Encoders for categorical features
        self.zone_encoder = LabelEncoder()
        self.room_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Zone to room mapping
        self.zone_to_room = self._create_zone_room_mapping()
        
        # Fitted flag
        self.is_fitted = False
        
    def _create_zone_room_mapping(self) -> Dict[str, str]:
        """Create mapping from zone entity to room name"""
        mapping = {}
        for room_name, zones in self.rooms.items():
            for zone in zones:
                mapping[zone] = room_name
        return mapping
    
    def fit(self, historical_data: pd.DataFrame) -> 'FeatureEngine':
        """Fit encoders and scalers on historical data"""
        
        logger.info("Fitting feature engineering pipeline")
        
        # Process historical data
        processed_data = self._process_sensor_events(historical_data)
        
        if len(processed_data) == 0:
            raise ValueError("No processed data available for fitting")
        
        # Fit zone encoder
        all_zones = historical_data['entity_id'].unique()
        self.zone_encoder.fit(all_zones)
        
        # Fit room encoder  
        all_rooms = list(self.rooms.keys()) + ['hallway', 'bathroom', 'unknown']
        self.room_encoder.fit(all_rooms)
        
        self.is_fitted = True
        logger.info("Feature engineering pipeline fitted successfully")
        
        return self
    
    def _process_sensor_events(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Process raw sensor events into structured format"""
        
        if len(raw_data) == 0:
            return pd.DataFrame()
        
        # Filter to only 'on' events and valid entities
        events = raw_data[
            (raw_data['state'] == 'on') & 
            (raw_data['entity_id'].isin(self.zone_to_room.keys()))
        ].copy()
        
        if len(events) == 0:
            return pd.DataFrame()
        
        # Add room information
        events['room'] = events['entity_id'].map(self.zone_to_room)
        
        # Sort by timestamp
        events = events.sort_values('last_changed').reset_index(drop=True)
        
        return events
    
    def extract_features(self, current_data: pd.DataFrame, timestamp: datetime = None) -> pd.DataFrame:
        """Extract ML features from current sensor data"""
        
        if not self.is_fitted:
            raise ValueError("FeatureEngine must be fitted before extracting features")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Process current events
        processed_events = self._process_sensor_events(current_data)
        
        if len(processed_events) == 0:
            return self._create_empty_features(timestamp)
        
        # Extract different types of features
        basic_features = self._extract_basic_features(processed_events, timestamp)
        temporal_features = self._extract_temporal_features(timestamp)
        activity_features = self._extract_activity_features(processed_events)
        
        # Combine all features
        features = pd.concat([
            basic_features,
            temporal_features,
            activity_features
        ], axis=1)
        
        return features
    
    def _extract_basic_features(self, events: pd.DataFrame, timestamp: datetime = None) -> pd.DataFrame:
        """Extract basic statistical features"""
        
        if len(events) == 0:
            return pd.DataFrame({
                'total_activations_1h': [0],
                'unique_zones_1h': [0],
                'unique_rooms_1h': [0]
            })
        
        if timestamp is None:
            timestamp = events['last_changed'].max()
        
        # Filter to last hour
        cutoff = timestamp - timedelta(hours=1)
        recent_events = events[events['last_changed'] >= cutoff]
        
        features = {
            'total_activations_1h': len(recent_events),
            'unique_zones_1h': recent_events['entity_id'].nunique(),
            'unique_rooms_1h': recent_events['room'].nunique()
        }
        
        return pd.DataFrame([features])
    
    def _extract_temporal_features(self, timestamp: datetime) -> pd.DataFrame:
        """Extract time-based features"""
        
        features = {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_weekend': int(timestamp.weekday() >= 5),
            'minute_of_day': timestamp.hour * 60 + timestamp.minute
        }
        
        return pd.DataFrame([features])
    
    def _extract_activity_features(self, events: pd.DataFrame) -> pd.DataFrame:
        """Extract activity-related features"""
        
        if len(events) == 0:
            return pd.DataFrame([{
                'room_diversity': 0,
                'activity_5min': 0,
                'activity_15min': 0,
                'activity_30min': 0,
                'activity_60min': 0
            }])
        
        now = events['last_changed'].max()
        
        # Activity in different time windows
        activity_scores = {}
        for window in [5, 15, 30, 60]:
            cutoff = now - timedelta(minutes=window)
            window_events = events[events['last_changed'] >= cutoff]
            activity_scores[f'activity_{window}min'] = len(window_events)
        
        # Room diversity
        recent_30min = events[events['last_changed'] >= now - timedelta(minutes=30)]
        room_diversity = recent_30min['room'].nunique()
        
        features = {
            'room_diversity': room_diversity,
            **activity_scores
        }
        
        return pd.DataFrame([features])
    
    def _create_empty_features(self, timestamp: datetime) -> pd.DataFrame:
        """Create empty feature set with proper structure"""
        
        features = {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_weekend': int(timestamp.weekday() >= 5),
            'minute_of_day': timestamp.hour * 60 + timestamp.minute,
            'total_activations_1h': 0,
            'unique_zones_1h': 0,
            'unique_rooms_1h': 0,
            'room_diversity': 0,
            'activity_5min': 0,
            'activity_15min': 0,
            'activity_30min': 0,
            'activity_60min': 0
        }
        
        return pd.DataFrame([features])
    
    def save(self, filepath: str):
        """Save the fitted feature engine"""
        
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted FeatureEngine")
        
        save_data = {
            'config': self.config,
            'zone_encoder': self.zone_encoder,
            'room_encoder': self.room_encoder,
            'scaler': self.scaler,
            'zone_to_room': self.zone_to_room,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"FeatureEngine saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureEngine':
        """Load a saved feature engine"""
        
        save_data = joblib.load(filepath)
        
        engine = cls(save_data['config'])
        engine.zone_encoder = save_data['zone_encoder']
        engine.room_encoder = save_data['room_encoder'] 
        engine.scaler = save_data['scaler']
        engine.zone_to_room = save_data['zone_to_room']
        engine.is_fitted = save_data['is_fitted']
        
        logger.info(f"FeatureEngine loaded from {filepath}")
        return engine
