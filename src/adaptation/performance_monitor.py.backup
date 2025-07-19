"""
Continuous Performance Monitoring - Tracks system performance and triggers adaptations
Implements the exact PerformanceMonitor from CLAUDE.md
"""

import logging
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from river import stats, drift
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Tracks prediction accuracy, detects drift, and triggers adaptations
    Implements the exact approach from CLAUDE.md
    """
    
    def __init__(self):
        self.metrics = {
            'prediction_accuracy': RunningWindowMetric(window=1000),
            'false_positive_rate': RunningWindowMetric(window=1000),
            'model_drift': DriftDetector(),
            'feature_relevance': FeatureRelevanceTracker()
        }
        
        # Per-room and per-horizon tracking
        self.room_metrics = defaultdict(lambda: defaultdict(lambda: RunningWindowMetric(window=500)))
        self.horizon_metrics = defaultdict(lambda: RunningWindowMetric(window=500))
        
        # Performance history
        self.performance_history = deque(maxlen=10000)
        self.adaptation_triggers = []
        
        # Thresholds
        self.accuracy_threshold = 0.7
        self.drift_threshold = 0.05
        self.relevance_threshold = 0.1
        
        logger.info("Initialized performance monitor")
    
    def log_prediction(self, 
                      room_id: str, 
                      horizon_minutes: int,
                      prediction: Dict[str, Any], 
                      actual: bool,
                      timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Track prediction performance
        Implements the exact tracking from CLAUDE.md
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Extract prediction details
        predicted_prob = prediction.get('probability', 0.5)
        predicted_binary = predicted_prob > 0.5
        confidence = prediction.get('confidence', 0.0)
        uncertainty = prediction.get('uncertainty', 1.0)
        
        # Calculate metrics
        accuracy = 1.0 if predicted_binary == actual else 0.0
        false_positive = 1.0 if predicted_binary and not actual else 0.0
        false_negative = 1.0 if not predicted_binary and actual else 0.0
        
        # Update global metrics
        self.metrics['prediction_accuracy'].update(accuracy)
        self.metrics['false_positive_rate'].update(false_positive)
        
        # Update room-specific metrics
        self.room_metrics[room_id]['accuracy'].update(accuracy)
        self.room_metrics[room_id]['false_positive_rate'].update(false_positive)
        self.room_metrics[room_id]['false_negative_rate'].update(false_negative)
        self.room_metrics[room_id]['confidence'].update(confidence)
        
        # Update horizon-specific metrics
        self.horizon_metrics[horizon_minutes].update(accuracy)
        
        # Check for drift
        drift_detected = self.metrics['model_drift'].detect_drift(predicted_prob, actual)
        
        # Log performance record
        performance_record = {
            'timestamp': timestamp,
            'room_id': room_id,
            'horizon_minutes': horizon_minutes,
            'predicted_prob': predicted_prob,
            'predicted_binary': predicted_binary,
            'actual': actual,
            'accuracy': accuracy,
            'false_positive': false_positive,
            'false_negative': false_negative,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'drift_detected': drift_detected
        }
        self.performance_history.append(performance_record)
        
        # Check if adaptation is needed
        adaptation_needed = self.check_adaptation_triggers(room_id, horizon_minutes, performance_record)
        
        if adaptation_needed:
            self.trigger_model_adaptation(room_id, adaptation_needed)
        
        return {
            'accuracy': accuracy,
            'drift_detected': drift_detected,
            'adaptation_triggered': adaptation_needed is not None,
            'global_accuracy': self.metrics['prediction_accuracy'].get_current_value(),
            'room_accuracy': self.room_metrics[room_id]['accuracy'].get_current_value()
        }
    
    def check_adaptation_triggers(self, 
                                room_id: str, 
                                horizon_minutes: int,
                                performance_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if model adaptation should be triggered"""
        triggers = []
        
        # 1. Low accuracy trigger
        room_accuracy = self.room_metrics[room_id]['accuracy'].get_current_value()
        if room_accuracy < self.accuracy_threshold:
            sample_count = self.room_metrics[room_id]['accuracy'].get_sample_count()
            if sample_count >= 50:  # Need minimum samples
                triggers.append({
                    'type': 'low_accuracy',
                    'room_id': room_id,
                    'current_accuracy': room_accuracy,
                    'threshold': self.accuracy_threshold,
                    'sample_count': sample_count
                })
        
        # 2. Drift detection trigger
        if performance_record['drift_detected']:
            triggers.append({
                'type': 'concept_drift',
                'room_id': room_id,
                'horizon_minutes': horizon_minutes,
                'drift_score': self.metrics['model_drift'].get_drift_score()
            })
        
        # 3. High false positive rate
        fp_rate = self.room_metrics[room_id]['false_positive_rate'].get_current_value()
        if fp_rate > 0.3:  # More than 30% false positives
            triggers.append({
                'type': 'high_false_positives',
                'room_id': room_id,
                'false_positive_rate': fp_rate
            })
        
        # 4. Inconsistent confidence
        if self.room_metrics[room_id]['confidence'].get_sample_count() >= 20:
            avg_confidence = self.room_metrics[room_id]['confidence'].get_current_value()
            if avg_confidence < 0.5:  # Low confidence predictions
                triggers.append({
                    'type': 'low_confidence',
                    'room_id': room_id,
                    'average_confidence': avg_confidence
                })
        
        # 5. Feature relevance degradation
        feature_issues = self.metrics['feature_relevance'].check_degradation()
        if feature_issues:
            triggers.extend(feature_issues)
        
        if triggers:
            return {
                'triggers': triggers,
                'timestamp': datetime.now(),
                'room_id': room_id,
                'horizon_minutes': horizon_minutes
            }
        
        return None
    
    def trigger_model_adaptation(self, room_id: str, adaptation_info: Dict[str, Any]):
        """
        Trigger model adaptation based on performance issues
        """
        adaptation_event = {
            'timestamp': datetime.now(),
            'room_id': room_id,
            'adaptation_info': adaptation_info,
            'triggered_by': [trigger['type'] for trigger in adaptation_info['triggers']]
        }
        
        self.adaptation_triggers.append(adaptation_event)
        
        logger.warning(f"Model adaptation triggered for {room_id}: {adaptation_event['triggered_by']}")
        
        # In full implementation, this would trigger actual model retraining
        # For now, log the adaptation need
        
        return adaptation_event
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_records = [
            record for record in self.performance_history
            if record['timestamp'] >= cutoff_time
        ]
        
        if not recent_records:
            return {'message': 'No recent performance data'}
        
        # Calculate overall metrics
        total_predictions = len(recent_records)
        total_accurate = sum(1 for r in recent_records if r['accuracy'] == 1.0)
        overall_accuracy = total_accurate / total_predictions if total_predictions > 0 else 0.0
        
        # Per-room breakdown
        room_breakdown = defaultdict(lambda: {'predictions': 0, 'accurate': 0})
        for record in recent_records:
            room = record['room_id']
            room_breakdown[room]['predictions'] += 1
            if record['accuracy'] == 1.0:
                room_breakdown[room]['accurate'] += 1
        
        # Calculate room accuracies
        for room, data in room_breakdown.items():
            data['accuracy'] = data['accurate'] / data['predictions'] if data['predictions'] > 0 else 0.0
        
        # Drift analysis
        drift_events = sum(1 for r in recent_records if r['drift_detected'])
        
        # Recent adaptations
        recent_adaptations = [
            trigger for trigger in self.adaptation_triggers
            if trigger['timestamp'] >= cutoff_time
        ]
        
        return {
            'time_window_hours': hours,
            'total_predictions': total_predictions,
            'overall_accuracy': overall_accuracy,
            'drift_events': drift_events,
            'adaptations_triggered': len(recent_adaptations),
            'room_breakdown': dict(room_breakdown),
            'recent_adaptations': recent_adaptations
        }
    
    def get_model_health_score(self, room_id: str) -> Dict[str, Any]:
        """Calculate overall health score for a room's model"""
        if room_id not in self.room_metrics:
            return {'health_score': 0.0, 'status': 'no_data'}
        
        metrics = self.room_metrics[room_id]
        
        # Component scores (0-1)
        accuracy_score = min(1.0, metrics['accuracy'].get_current_value() / 0.9)  # Target 90%
        
        fp_rate = metrics['false_positive_rate'].get_current_value()
        fp_score = max(0.0, 1.0 - fp_rate / 0.2)  # Penalize if >20% FP
        
        fn_rate = metrics['false_negative_rate'].get_current_value()
        fn_score = max(0.0, 1.0 - fn_rate / 0.2)  # Penalize if >20% FN
        
        confidence_score = metrics['confidence'].get_current_value()
        
        # Sample count factor (need enough data)
        sample_count = metrics['accuracy'].get_sample_count()
        sample_factor = min(1.0, sample_count / 100)  # Full score at 100+ samples
        
        # Weighted health score
        health_score = (
            accuracy_score * 0.4 +
            fp_score * 0.2 +
            fn_score * 0.2 +
            confidence_score * 0.2
        ) * sample_factor
        
        # Determine status
        if health_score >= 0.8:
            status = 'excellent'
        elif health_score >= 0.6:
            status = 'good'
        elif health_score >= 0.4:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'health_score': health_score,
            'status': status,
            'components': {
                'accuracy': accuracy_score,
                'false_positive_control': fp_score,
                'false_negative_control': fn_score,
                'confidence': confidence_score,
                'sample_factor': sample_factor
            },
            'sample_count': sample_count
        }
    
    def get_feature_importance_trends(self) -> Dict[str, Any]:
        """Get trends in feature importance over time"""
        return self.metrics['feature_relevance'].get_trends()
    
    def reset_metrics(self, room_id: Optional[str] = None):
        """Reset metrics for debugging or after major changes"""
        if room_id:
            if room_id in self.room_metrics:
                for metric in self.room_metrics[room_id].values():
                    metric.reset()
                logger.info(f"Reset metrics for room {room_id}")
        else:
            # Reset all metrics
            for metric in self.metrics.values():
                if hasattr(metric, 'reset'):
                    metric.reset()
            
            self.room_metrics.clear()
            self.horizon_metrics.clear()
            self.performance_history.clear()
            self.adaptation_triggers.clear()
            
            logger.info("Reset all performance metrics")


class RunningWindowMetric:
    """
    Running window metric tracker
    Implements sliding window statistics
    """
    
    def __init__(self, window: int = 1000):
        self.window_size = window
        self.values = deque(maxlen=window)
        self.sum_values = 0.0
        
    def update(self, value: float):
        """Update metric with new value"""
        if len(self.values) == self.window_size:
            # Remove oldest value from sum
            self.sum_values -= self.values[0]
        
        self.values.append(value)
        self.sum_values += value
    
    def get_current_value(self) -> float:
        """Get current metric value (mean)"""
        if not self.values:
            return 0.0
        return self.sum_values / len(self.values)
    
    def get_sample_count(self) -> int:
        """Get number of samples"""
        return len(self.values)
    
    def reset(self):
        """Reset the metric"""
        self.values.clear()
        self.sum_values = 0.0


class DriftDetector:
    """
    Concept drift detection using statistical tests
    """
    
    def __init__(self):
        self.drift_detector = drift.ADWIN(delta=0.002)
        self.recent_errors = deque(maxlen=100)
        self.drift_events = []
        
    def detect_drift(self, prediction: float, actual: bool) -> bool:
        """Detect if concept drift occurred"""
        try:
            # Calculate prediction error
            predicted_binary = prediction > 0.5
            error = 0.0 if predicted_binary == actual else 1.0
            
            self.recent_errors.append(error)
            
            # Update drift detector
            self.drift_detector.update(error)
            
            # Check for drift
            if self.drift_detector.drift_detected:
                drift_event = {
                    'timestamp': datetime.now(),
                    'error_rate': sum(self.recent_errors) / len(self.recent_errors),
                    'prediction': prediction,
                    'actual': actual
                }
                self.drift_events.append(drift_event)
                
                logger.warning(f"Concept drift detected: {drift_event}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            return False
    
    def get_drift_score(self) -> float:
        """Get current drift score"""
        if self.recent_errors:
            return sum(self.recent_errors) / len(self.recent_errors)
        return 0.0
    
    def reset(self):
        """Reset drift detector"""
        self.drift_detector = drift.ADWIN(delta=0.002)
        self.recent_errors.clear()
        self.drift_events.clear()


class FeatureRelevanceTracker:
    """
    Tracks feature relevance over time
    """
    
    def __init__(self):
        self.feature_scores = defaultdict(lambda: deque(maxlen=1000))
        self.feature_usage = defaultdict(int)
        self.degradation_threshold = 0.1
        
    def update_feature_importance(self, feature_importance: Dict[str, float]):
        """Update feature importance scores"""
        timestamp = datetime.now()
        
        for feature, importance in feature_importance.items():
            self.feature_scores[feature].append({
                'timestamp': timestamp,
                'importance': importance
            })
            self.feature_usage[feature] += 1
    
    def check_degradation(self) -> List[Dict[str, Any]]:
        """Check for feature relevance degradation"""
        issues = []
        
        for feature, scores in self.feature_scores.items():
            if len(scores) >= 10:  # Need minimum history
                recent_scores = [s['importance'] for s in list(scores)[-10:]]
                older_scores = [s['importance'] for s in list(scores)[-20:-10]] if len(scores) >= 20 else []
                
                if older_scores:
                    recent_avg = np.mean(recent_scores)
                    older_avg = np.mean(older_scores)
                    
                    # Check for significant drop
                    if older_avg > 0 and (older_avg - recent_avg) / older_avg > self.degradation_threshold:
                        issues.append({
                            'type': 'feature_degradation',
                            'feature': feature,
                            'older_importance': older_avg,
                            'recent_importance': recent_avg,
                            'degradation_percent': (older_avg - recent_avg) / older_avg
                        })
        
        return issues
    
    def get_trends(self) -> Dict[str, Any]:
        """Get feature importance trends"""
        trends = {}
        
        for feature, scores in self.feature_scores.items():
            if len(scores) >= 5:
                importances = [s['importance'] for s in scores]
                timestamps = [s['timestamp'] for s in scores]
                
                # Simple trend calculation
                recent_trend = np.mean(importances[-5:]) - np.mean(importances[:5]) if len(importances) >= 10 else 0
                
                trends[feature] = {
                    'current_importance': importances[-1] if importances else 0,
                    'trend': recent_trend,
                    'usage_count': self.feature_usage[feature],
                    'stability': np.std(importances) if len(importances) > 1 else 0
                }
        
        return trends
    
    def reset(self):
        """Reset feature tracking"""
        self.feature_scores.clear()
        self.feature_usage.clear()