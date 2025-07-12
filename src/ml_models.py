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

class IntentPredictor:
    """Multi-horizon room occupancy and duration prediction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rooms = list(config['rooms'].keys())
        
        # Long-term: 2-hour room occupancy probabilities
        self.longterm_model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=50,
                max_depth=6,
                random_state=42,
                n_jobs=4
            )
        )
        
        self.is_fitted = False
        
    def fit(self, features: pd.DataFrame, targets: Dict[str, np.ndarray]) -> 'IntentPredictor':
        """Fit the intent prediction models"""
        
        logger.info("Fitting intent prediction models")
        
        if len(features) == 0:
            raise ValueError("No features provided for fitting")
        
        # Fit long-term occupancy model
        if 'longterm_occupancy' in targets and len(targets['longterm_occupancy']) > 0:
            self.longterm_model.fit(features, targets['longterm_occupancy'])
            logger.info("Long-term occupancy model fitted")
        
        self.is_fitted = True
        logger.info("Intent prediction models fitted successfully")
        
        return self
    
    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Predict room occupancy intent"""
        
        if not self.is_fitted:
            raise ValueError("IntentPredictor must be fitted before prediction")
        
        if len(features) == 0:
            return self._empty_predictions()
        
        predictions = {}
        
        try:
            # Long-term room occupancy (2 hours)
            longterm_probs = self.longterm_model.predict(features)[0]
            
            for i, room in enumerate(self.rooms):
                prob = max(0, min(1, longterm_probs[i])) if i < len(longterm_probs) else 0
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
            predictions[f'{room}_occupancy_2h'] = 0.0
        
        return predictions


class ModelPipeline:
    """Complete ML pipeline for room occupancy prediction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.intent_predictor = IntentPredictor(config)
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
        
        # Get intent predictions
        intent_predictions = self.intent_predictor.predict(features)
        
        # Add default activity info
        combined_predictions = {
            'current_activity': 'UNKNOWN',
            'activity_confidence': 0.5,
            'activity_duration_minutes': 30.0,
            'next_room': 'unknown',
            'next_room_confidence': 0.0,
            **intent_predictions
        }
        
        return combined_predictions
    
    def save(self, filepath: str):
        """Save the complete pipeline"""
        
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted ModelPipeline")
        
        save_data = {
            'config': self.config,
            'intent_predictor': self.intent_predictor,
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
        pipeline.is_fitted = save_data['is_fitted']
        
        logger.info(f"ModelPipeline loaded from {filepath}")
        return pipeline


def create_targets_from_historical_data(features: pd.DataFrame, events: pd.DataFrame, 
                                       rooms: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """Create training targets from historical sensor data"""
    
    logger.info("Creating training targets from historical data")
    
    targets = {
        'longterm_occupancy': []
    }
    
    # Create simple targets for now
    num_rooms = len(rooms)
    for i in range(len(features)):
        # Random occupancy probabilities for now (placeholder)
        room_probs = np.random.random(num_rooms) * 0.5  # 0-50% chance
        targets['longterm_occupancy'].append(room_probs)
    
    # Convert to numpy arrays
    targets['longterm_occupancy'] = np.array(targets['longterm_occupancy'])
    
    logger.info(f"Created targets: {len(targets['longterm_occupancy'])} samples")
    
    return targets
