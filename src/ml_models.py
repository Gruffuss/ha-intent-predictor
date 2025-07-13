"""
Machine Learning Models for Intent Prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import logging
import joblib

# ML imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ActivityClassifier:
    """Rule-based activity classification from sensor patterns"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rooms = list(config['rooms'].keys())
        
        # Activity definitions based ONLY on room usage patterns (no hardcoded times!)
        self.activity_patterns = {
            'OFFICE_WORK': {
                'primary_rooms': ['office'],
                'activity_threshold': 0.6,
                'description': 'Active in office space'
            },
            'BEDROOM_REST': {
                'primary_rooms': ['bedroom'],
                'activity_threshold': 0.8,
                'description': 'Resting or sleeping in bedroom'
            },
            'KITCHEN_ACTIVITY': {
                'primary_rooms': ['kitchen_livingroom'],
                'activity_threshold': 0.7,
                'description': 'Cooking, eating, or kitchen activities'
            },
            'LIVINGROOM_RELAX': {
                'primary_rooms': ['kitchen_livingroom'],
                'activity_threshold': 0.5,
                'description': 'Relaxing in living area'
            },
            'TRANSITIONING': {
                'primary_rooms': ['hallway'],
                'activity_threshold': 0.3,
                'description': 'Moving between rooms'
            }
        }
        
    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Predict current activity from features"""
        
        if len(features) == 0:
            return {
                'current_activity': 'UNKNOWN',
                'activity_confidence': 0.0,
                'activity_duration_minutes': 0.0
            }
        
        # Get the latest feature row
        latest_features = features.iloc[-1]
        
        # Extract relevant features
        current_hour = latest_features.get('hour', 12)
        
        # Calculate room activity scores
        room_scores = {}
        for room in self.rooms:
            # Check recent activity in each room
            activity_30min = latest_features.get(f'{room}_activity_30min', 0)
            activity_15min = latest_features.get(f'{room}_activity_15min', 0)
            last_seen = latest_features.get(f'{room}_last_seen_minutes', 1440)
            
            # Score based on recent activity and recency
            activity_score = (activity_30min * 0.4 + activity_15min * 0.6)
            recency_score = max(0, 1 - (last_seen / 60))  # Decay over 1 hour
            
            room_scores[room] = activity_score * 0.7 + recency_score * 0.3
        
        # Find the most active room
        primary_room = max(room_scores, key=room_scores.get) if room_scores else 'unknown'
        max_room_score = room_scores.get(primary_room, 0)
        
        # Determine activity based on room and time
        best_activity = 'UNKNOWN'
        best_confidence = 0.0
        
        for activity, pattern in self.activity_patterns.items():
            confidence = 0.0
            
            # Check room match (primary factor)
            if primary_room in pattern['primary_rooms']:
                confidence += 0.7  # Increased weight for room match
            
            # Check activity level (based on actual sensor activity)
            if max_room_score >= pattern['activity_threshold']:
                confidence += 0.3  # Activity intensity bonus
            
            # Update best activity
            if confidence > best_confidence:
                best_activity = activity
                best_confidence = confidence
        
        # Calculate estimated duration based on activity type and current session
        duration_estimate = self._estimate_activity_duration(
            best_activity, latest_features, primary_room
        )
        
        return {
            'current_activity': best_activity,
            'activity_confidence': min(best_confidence, 1.0),
            'activity_duration_minutes': duration_estimate
        }
    
    def _estimate_activity_duration(self, activity: str, features: pd.Series, primary_room: str) -> float:
        """Estimate how long the current activity will continue"""
        
        # Get activity intensity
        activity_15min = features.get(f'{primary_room}_activity_15min', 0)
        activity_30min = features.get(f'{primary_room}_activity_30min', 0)
        
        # Base duration estimates by activity type (data-driven, not time-based)
        base_durations = {
            'OFFICE_WORK': 90,        # 1.5 hours typical work session
            'BEDROOM_REST': 120,      # 2 hours typical rest period
            'KITCHEN_ACTIVITY': 30,   # 30 minutes cooking/eating
            'LIVINGROOM_RELAX': 60,   # 1 hour relaxing
            'TRANSITIONING': 5,       # 5 minutes moving
            'UNKNOWN': 15             # 15 minutes default
        }
        
        base_duration = base_durations.get(activity, 30)
        
        # Adjust based on recent activity intensity
        if activity_15min > activity_30min * 0.8:  # Increasing activity
            duration_multiplier = 1.2
        elif activity_15min < activity_30min * 0.3:  # Decreasing activity
            duration_multiplier = 0.7
        else:
            duration_multiplier = 1.0
        
        return base_duration * duration_multiplier


class IntentPredictor:
    """Multi-horizon room occupancy and duration prediction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rooms = list(config['rooms'].keys())
        
        # Create room labels for classification (including 'none' for no occupancy)
        self.room_labels = list(self.rooms) + ['none']
        
        # Short-term: 15-minute room occupancy classification
        self.shortterm_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            n_jobs=4,
            class_weight='balanced'  # Handle class imbalance
        )
        
        # Long-term: 2-hour room occupancy classification  
        self.longterm_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=4,
            class_weight='balanced'
        )
        
        self.is_fitted = False
        
    def fit(self, features: pd.DataFrame, targets: Dict[str, np.ndarray]) -> 'IntentPredictor':
        """Fit the intent prediction models"""
        
        logger.info("Fitting intent prediction models")
        
        if len(features) == 0:
            raise ValueError("No features provided for fitting")
        
        # Fit short-term occupancy model (15 minutes)
        if 'shortterm_room_labels' in targets and len(targets['shortterm_room_labels']) > 0:
            # Validate room labels consistency
            expected_labels = set(self.room_labels)
            actual_labels = set(targets['shortterm_room_labels'])
            missing_labels = expected_labels - actual_labels
            if missing_labels:
                logger.warning(f"Missing room labels in short-term training data: {missing_labels}")
            
            self.shortterm_model.fit(features, targets['shortterm_room_labels'])
            logger.info(f"Short-term occupancy classifier fitted with {len(actual_labels)} unique labels")
            logger.info(f"Label distribution: {dict(pd.Series(targets['shortterm_room_labels']).value_counts())}")
        
        # Fit long-term occupancy model (2 hours)
        if 'longterm_room_labels' in targets and len(targets['longterm_room_labels']) > 0:
            # Validate room labels consistency
            expected_labels = set(self.room_labels)
            actual_labels = set(targets['longterm_room_labels'])
            missing_labels = expected_labels - actual_labels
            if missing_labels:
                logger.warning(f"Missing room labels in long-term training data: {missing_labels}")
            
            self.longterm_model.fit(features, targets['longterm_room_labels'])
            logger.info(f"Long-term occupancy classifier fitted with {len(actual_labels)} unique labels")
            logger.info(f"Label distribution: {dict(pd.Series(targets['longterm_room_labels']).value_counts())}")
        
        self.is_fitted = True
        logger.info("Intent prediction models fitted successfully")
        
        return self
    
    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Predict room occupancy intent using classification"""
        
        if not self.is_fitted:
            raise ValueError("IntentPredictor must be fitted before prediction")
        
        if len(features) == 0:
            return self._empty_predictions()
        
        predictions = {}
        
        try:
            # Short-term room prediction (15 minutes)
            shortterm_room = self.shortterm_model.predict(features)[0]
            shortterm_probs = self.shortterm_model.predict_proba(features)[0]
            
            # Convert classification to room occupancy probabilities
            for i, room in enumerate(self.rooms):
                if room == shortterm_room:
                    # Get confidence for predicted room
                    room_idx = self.shortterm_model.classes_.tolist().index(room)
                    prob = shortterm_probs[room_idx]
                else:
                    prob = 0.0
                predictions[f'{room}_occupancy_15min'] = prob
                
        except Exception as e:
            logger.warning(f"Short-term prediction failed: {e}")
            for room in self.rooms:
                predictions[f'{room}_occupancy_15min'] = 0.0
        
        try:
            # Long-term room prediction (2 hours)
            longterm_room = self.longterm_model.predict(features)[0]
            longterm_probs = self.longterm_model.predict_proba(features)[0]
            
            # Convert classification to room occupancy probabilities
            for i, room in enumerate(self.rooms):
                if room == longterm_room:
                    # Get confidence for predicted room
                    room_idx = self.longterm_model.classes_.tolist().index(room)
                    prob = longterm_probs[room_idx]
                else:
                    prob = 0.0
                predictions[f'{room}_occupancy_2h'] = prob
                
        except Exception as e:
            logger.warning(f"Long-term prediction failed: {e}")
            for room in self.rooms:
                predictions[f'{room}_occupancy_2h'] = 0.0
        
        return predictions
    
    def _empty_predictions(self) -> Dict[str, Any]:
        """Return empty predictions when no data available"""
        
        predictions = {}
        for room in self.rooms:
            predictions[f'{room}_occupancy_15min'] = 0.0
            predictions[f'{room}_occupancy_2h'] = 0.0
        
        return predictions


class ModelPipeline:
    """Complete ML pipeline for room occupancy prediction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.intent_predictor = IntentPredictor(config)
        self.activity_classifier = ActivityClassifier(config)
        self.is_fitted = False
        
    def fit(self, features: pd.DataFrame, activity_sessions: List[Dict], 
            prediction_targets: Dict[str, np.ndarray]) -> 'ModelPipeline':
        """Fit the complete ML pipeline"""
        
        logger.info("Fitting complete ML pipeline")
        
        # Fit intent predictor
        self.intent_predictor.fit(features, prediction_targets)
        
        self.is_fitted = True
        logger.info("ML pipeline fitted successfully")
        
        return self
    
    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Generate complete predictions"""
        
        if not self.is_fitted:
            raise ValueError("ModelPipeline must be fitted before prediction")
        
        # Get intent predictions (room occupancy probabilities)
        intent_predictions = self.intent_predictor.predict(features)
        
        # Get activity classification
        activity_predictions = self.activity_classifier.predict(features)
        
        # Get next room prediction
        next_room_predictions = self._predict_next_room(features)
        
        # Combine all predictions
        combined_predictions = {
            **intent_predictions,
            **activity_predictions,
            **next_room_predictions
        }
        
        return combined_predictions
    
    def _predict_next_room(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Predict next room based on transition patterns"""
        
        if len(features) == 0:
            return {
                'next_room': 'unknown',
                'next_room_confidence': 0.0
            }
        
        latest_features = features.iloc[-1]
        
        # Find current most active room
        room_scores = {}
        for room in self.intent_predictor.rooms:
            activity_15min = latest_features.get(f'{room}_activity_15min', 0)
            last_seen = latest_features.get(f'{room}_last_seen_minutes', 1440)
            
            # Score based on recent activity
            if last_seen < 5:  # Currently active
                room_scores[room] = activity_15min
            else:
                room_scores[room] = 0
        
        current_room = max(room_scores, key=room_scores.get) if room_scores else 'unknown'
        
        # Predict next room based on transition patterns and time
        hour = latest_features.get('hour', 12)
        
        # Simple transition logic based on time and current room
        next_room_probs = {}
        
        if current_room == 'office':
            if 11 <= hour <= 13:  # Lunch time
                next_room_probs = {'kitchen_livingroom': 0.7, 'office': 0.2, 'bedroom': 0.1}
            elif hour >= 18:  # Evening
                next_room_probs = {'kitchen_livingroom': 0.6, 'bedroom': 0.3, 'office': 0.1}
            else:
                next_room_probs = {'office': 0.6, 'kitchen_livingroom': 0.3, 'bedroom': 0.1}
                
        elif current_room == 'kitchen_livingroom':
            if 22 <= hour or hour <= 6:  # Night
                next_room_probs = {'bedroom': 0.8, 'kitchen_livingroom': 0.2}
            elif 8 <= hour <= 17:  # Work hours
                next_room_probs = {'office': 0.6, 'kitchen_livingroom': 0.4}
            else:
                next_room_probs = {'kitchen_livingroom': 0.5, 'bedroom': 0.3, 'office': 0.2}
                
        elif current_room == 'bedroom':
            if 6 <= hour <= 8:  # Morning
                next_room_probs = {'kitchen_livingroom': 0.5, 'office': 0.3, 'bedroom': 0.2}
            elif hour >= 22:  # Night
                next_room_probs = {'bedroom': 0.9, 'kitchen_livingroom': 0.1}
            else:
                next_room_probs = {'office': 0.4, 'kitchen_livingroom': 0.4, 'bedroom': 0.2}
        else:
            # Default probabilities
            next_room_probs = {'kitchen_livingroom': 0.4, 'office': 0.3, 'bedroom': 0.3}
        
        # Find most likely next room
        next_room = max(next_room_probs, key=next_room_probs.get)
        confidence = next_room_probs[next_room]
        
        return {
            'next_room': next_room,
            'next_room_confidence': confidence
        }
    
    def save(self, filepath: str):
        """Save the complete pipeline"""
        
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted ModelPipeline")
        
        save_data = {
            'config': self.config,
            'intent_predictor': self.intent_predictor,
            'activity_classifier': self.activity_classifier,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"ModelPipeline saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelPipeline':
        """Load a saved pipeline"""
        
        save_data = joblib.load(filepath)
        
        pipeline = cls(save_data['config'])
        pipeline.intent_predictor = save_data['intent_predictor']
        pipeline.activity_classifier = save_data.get('activity_classifier', ActivityClassifier(save_data['config']))
        pipeline.is_fitted = save_data['is_fitted']
        
        logger.info(f"ModelPipeline loaded from {filepath}")
        return pipeline


