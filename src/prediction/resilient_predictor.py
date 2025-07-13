"""
Error Handling and Recovery - Resilient prediction with fallbacks
Implements the exact ResilientPredictor from CLAUDE.md
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ResilientPredictor:
    """
    Prediction with fallback mechanisms
    Implements the exact approach from CLAUDE.md
    """
    
    def __init__(self, primary_predictor, fallback_predictor=None):
        self.primary_predictor = primary_predictor
        self.fallback_predictor = fallback_predictor or SimpleFallbackPredictor()
        self.failure_count = 0
        self.last_failure_time = None
        
    def predict_with_fallback(self, room_id: str, horizon: int) -> Dict[str, Any]:
        """
        Predict with fallback exactly as specified in CLAUDE.md
        """
        try:
            # Try primary prediction
            result = self.primary_predictor.predict(room_id, horizon)
            
            # Reset failure count on success
            self.failure_count = 0
            
            return result
            
        except Exception as e:
            logger.error(f"Primary prediction failed: {e}")
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            # Fallback to simpler model
            try:
                fallback_result = self.fallback_predictor.predict(room_id, horizon)
                fallback_result['fallback_used'] = True
                fallback_result['primary_error'] = str(e)
                
                return fallback_result
                
            except Exception as fallback_error:
                logger.error(f"Fallback prediction also failed: {fallback_error}")
                
                # Last resort - return uncertain prediction
                return {
                    'probability': 0.5,
                    'uncertainty': 1.0,
                    'confidence': 0.0,
                    'error': 'Prediction system temporarily unavailable',
                    'primary_error': str(e),
                    'fallback_error': str(fallback_error),
                    'timestamp': datetime.now().isoformat()
                }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of prediction system"""
        return {
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'primary_available': True,  # Would check actual status
            'fallback_available': True
        }


class SimpleFallbackPredictor:
    """
    Simple fallback predictor using basic heuristics
    """
    
    def __init__(self):
        # Simple time-based probabilities
        self.hourly_probabilities = {
            # Morning routine
            6: 0.3, 7: 0.5, 8: 0.6, 9: 0.4,
            # Day time
            10: 0.2, 11: 0.2, 12: 0.4, 13: 0.3, 14: 0.2, 15: 0.2, 16: 0.2, 17: 0.3,
            # Evening routine
            18: 0.4, 19: 0.6, 20: 0.7, 21: 0.6, 22: 0.5, 23: 0.3,
            # Night
            0: 0.1, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.1
        }
    
    def predict(self, room_id: str, horizon: int) -> Dict[str, Any]:
        """Simple time-based prediction fallback"""
        current_time = datetime.now()
        target_time = current_time.hour
        
        # Adjust for horizon
        target_hour = (target_time + (horizon // 60)) % 24
        
        # Base probability from time patterns
        base_prob = self.hourly_probabilities.get(target_hour, 0.2)
        
        # Room-specific adjustments
        if room_id == 'living_kitchen':
            base_prob *= 1.2  # More active space
        elif room_id in ['bathroom', 'small_bathroom']:
            base_prob *= 0.5  # Less frequent use
        
        return {
            'probability': min(0.8, base_prob),
            'uncertainty': 0.7,  # High uncertainty for fallback
            'confidence': 0.3,
            'method': 'time_based_fallback',
            'room_id': room_id,
            'horizon_minutes': horizon
        }