"""
Drift Detection for HA Intent Predictor.

Detects changes in behavior patterns and triggers model adaptation
as specified in CLAUDE.md.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of concept drift"""
    SUDDEN = "sudden"
    GRADUAL = "gradual"
    INCREMENTAL = "incremental"
    RECURRING = "recurring"


@dataclass
class DriftAlert:
    """Drift detection alert"""
    room_id: str
    drift_type: DriftType
    severity: float
    detected_at: datetime
    confidence: float
    affected_features: List[str]
    description: str
    recommended_action: str


class PageHinkleyTest:
    """Page-Hinkley test for drift detection"""
    
    def __init__(self, delta: float = 0.05, lambda_: float = 50.0):
        self.delta = delta
        self.lambda_ = lambda_
        self.sum_positive = 0.0
        self.sum_negative = 0.0
        self.min_positive = 0.0
        self.max_negative = 0.0
        self.values = deque(maxlen=1000)
        
    def update(self, value: float) -> bool:
        """Update with new value and return True if drift detected"""
        self.values.append(value)
        
        if len(self.values) < 2:
            return False
        
        # Calculate cumulative sum
        self.sum_positive += value - self.delta
        self.sum_negative += value + self.delta
        
        # Update min/max
        if self.sum_positive < self.min_positive:
            self.min_positive = self.sum_positive
        
        if self.sum_negative > self.max_negative:
            self.max_negative = self.sum_negative
        
        # Check for drift
        drift_positive = self.sum_positive - self.min_positive > self.lambda_
        drift_negative = self.max_negative - self.sum_negative > self.lambda_
        
        if drift_positive or drift_negative:
            self.reset()
            return True
        
        return False
    
    def reset(self):
        """Reset the test"""
        self.sum_positive = 0.0
        self.sum_negative = 0.0
        self.min_positive = 0.0
        self.max_negative = 0.0


class ADWINDetector:
    """Adaptive Windowing (ADWIN) drift detector"""
    
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window = deque()
        self.variance = 0.0
        self.width = 0
        
    def update(self, value: float) -> bool:
        """Update with new value and return True if drift detected"""
        self.window.append(value)
        self.width += 1
        
        if self.width < 2:
            return False
        
        # Simple ADWIN implementation
        # Check if we should compress window
        if self.width > 1000:
            # Remove oldest half
            for _ in range(self.width // 2):
                self.window.popleft()
            self.width = len(self.window)
        
        # Calculate variance
        mean = sum(self.window) / len(self.window)
        self.variance = sum((x - mean) ** 2 for x in self.window) / len(self.window)
        
        # Simple change detection
        if len(self.window) >= 50:
            recent_mean = sum(list(self.window)[-25:]) / 25
            older_mean = sum(list(self.window)[-50:-25]) / 25
            
            if abs(recent_mean - older_mean) > 3 * np.sqrt(self.variance):
                return True
        
        return False


class KSTest:
    """Kolmogorov-Smirnov test for distribution drift"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reference_window = deque(maxlen=window_size)
        self.current_window = deque(maxlen=window_size)
        
    def update(self, value: float) -> Tuple[bool, float]:
        """Update and return (drift_detected, p_value)"""
        if len(self.reference_window) < self.window_size:
            self.reference_window.append(value)
            return False, 1.0
        
        self.current_window.append(value)
        
        if len(self.current_window) < self.window_size:
            return False, 1.0
        
        # Perform KS test
        ref_sorted = sorted(self.reference_window)
        curr_sorted = sorted(self.current_window)
        
        # Simple KS statistic calculation
        max_diff = 0.0
        for i in range(len(ref_sorted)):
            cdf_ref = (i + 1) / len(ref_sorted)
            
            # Find position in current window
            pos = sum(1 for x in curr_sorted if x <= ref_sorted[i])
            cdf_curr = pos / len(curr_sorted)
            
            max_diff = max(max_diff, abs(cdf_ref - cdf_curr))
        
        # Critical value for α = 0.05
        critical_value = 1.36 * np.sqrt(2 / self.window_size)
        
        drift_detected = max_diff > critical_value
        p_value = 1.0 - max_diff  # Simplified p-value calculation
        
        if drift_detected:
            # Update reference window
            self.reference_window = deque(self.current_window, maxlen=self.window_size)
            self.current_window.clear()
        
        return drift_detected, p_value


class FeatureDriftDetector:
    """Detects drift in individual features"""
    
    def __init__(self, feature_name: str):
        self.feature_name = feature_name
        self.ph_test = PageHinkleyTest()
        self.adwin = ADWINDetector()
        self.ks_test = KSTest()
        
        # Statistics tracking
        self.values = deque(maxlen=1000)
        self.drift_count = 0
        self.last_drift = None
        
    def update(self, value: float) -> Optional[Dict[str, Any]]:
        """Update feature with new value and return drift info if detected"""
        self.values.append(value)
        
        # Run multiple drift detectors
        ph_drift = self.ph_test.update(value)
        adwin_drift = self.adwin.update(value)
        ks_drift, ks_p_value = self.ks_test.update(value)
        
        # Aggregate results
        drift_detected = ph_drift or adwin_drift or ks_drift
        
        if drift_detected:
            self.drift_count += 1
            self.last_drift = datetime.now()
            
            return {
                'feature': self.feature_name,
                'drift_detected': True,
                'ph_drift': ph_drift,
                'adwin_drift': adwin_drift,
                'ks_drift': ks_drift,
                'ks_p_value': ks_p_value,
                'drift_count': self.drift_count,
                'detected_at': self.last_drift
            }
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get feature statistics"""
        if not self.values:
            return {}
        
        values_array = np.array(self.values)
        
        return {
            'feature': self.feature_name,
            'count': len(self.values),
            'mean': np.mean(values_array),
            'std': np.std(values_array),
            'min': np.min(values_array),
            'max': np.max(values_array),
            'drift_count': self.drift_count,
            'last_drift': self.last_drift.isoformat() if self.last_drift else None
        }


class DriftDetector:
    """
    Main drift detection system.
    
    Monitors for concept drift in occupancy patterns and triggers
    adaptive model updates as specified in CLAUDE.md.
    """
    
    def __init__(self):
        self.room_detectors: Dict[str, Dict[str, FeatureDriftDetector]] = defaultdict(dict)
        self.drift_alerts = deque(maxlen=1000)
        self.monitoring_active = False
        
        # Configuration
        self.detection_sensitivity = 0.05
        self.min_samples_for_detection = 100
        self.alert_cooldown = timedelta(minutes=30)
        
        # Statistics
        self.total_drifts_detected = 0
        self.room_drift_counts = defaultdict(int)
        self.last_alerts = {}
        
        logger.info("Drift detector initialized")
    
    async def initialize(self):
        """Initialize drift detection system"""
        self.monitoring_active = True
        logger.info("Drift detection system initialized")
    
    async def monitor_drift(self, predictor):
        """Main drift monitoring loop"""
        logger.info("Starting drift monitoring...")
        
        while self.monitoring_active:
            try:
                # Check for drift in all rooms
                await self._check_room_drifts(predictor)
                
                # Check for system-wide drift patterns
                await self._check_system_drift()
                
                # Process any alerts
                await self._process_alerts(predictor)
                
                # Sleep before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in drift monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _check_room_drifts(self, predictor):
        """Check for drift in individual rooms"""
        
        # Get recent predictions and features for each room
        rooms = await predictor.get_active_rooms()
        
        for room_id in rooms:
            try:
                # Get recent model performance
                performance_data = await predictor.get_room_performance(room_id)
                
                if not performance_data:
                    continue
                
                # Check accuracy drift
                await self._check_accuracy_drift(room_id, performance_data)
                
                # Check feature drift
                await self._check_feature_drift(room_id, predictor)
                
            except Exception as e:
                logger.error(f"Error checking drift for room {room_id}: {e}")
    
    async def _check_accuracy_drift(self, room_id: str, performance_data: Dict):
        """Check for drift in prediction accuracy"""
        
        if 'accuracy_history' not in performance_data:
            return
        
        # Initialize detector if needed
        if 'accuracy' not in self.room_detectors[room_id]:
            self.room_detectors[room_id]['accuracy'] = FeatureDriftDetector(f"{room_id}_accuracy")
        
        detector = self.room_detectors[room_id]['accuracy']
        
        # Update with recent accuracy values
        for accuracy_value in performance_data['accuracy_history'][-10:]:  # Last 10 values
            drift_info = detector.update(accuracy_value)
            
            if drift_info:
                await self._trigger_drift_alert(
                    room_id, 
                    DriftType.GRADUAL,
                    0.8,  # High severity for accuracy drift
                    ['accuracy'],
                    f"Accuracy drift detected in room {room_id}",
                    "retrain_model"
                )
    
    async def _check_feature_drift(self, room_id: str, predictor):
        """Check for drift in input features"""
        
        try:
            # Get recent feature values
            recent_features = await predictor.get_recent_features(room_id, limit=100)
            
            if not recent_features:
                return
            
            # Check each feature for drift
            for feature_name, feature_values in recent_features.items():
                if feature_name not in self.room_detectors[room_id]:
                    self.room_detectors[room_id][feature_name] = FeatureDriftDetector(
                        f"{room_id}_{feature_name}"
                    )
                
                detector = self.room_detectors[room_id][feature_name]
                
                # Update with recent values
                # Ensure feature_values is a list before slicing
                if isinstance(feature_values, list) and len(feature_values) > 0:
                    recent_values = feature_values[-10:]  # Last 10 values
                elif hasattr(feature_values, '__iter__') and not isinstance(feature_values, (str, dict)):
                    # Handle other iterable types by converting to list first
                    try:
                        feature_list = list(feature_values)
                        recent_values = feature_list[-10:] if len(feature_list) > 0 else []
                    except (TypeError, ValueError):
                        recent_values = []
                else:
                    recent_values = []
                
                for value in recent_values:
                    drift_info = detector.update(value)
                    
                    if drift_info:
                        await self._trigger_drift_alert(
                            room_id,
                            DriftType.SUDDEN,
                            0.6,  # Medium severity for feature drift
                            [feature_name],
                            f"Feature drift detected in {feature_name} for room {room_id}",
                            "adapt_features"
                        )
        
        except Exception as e:
            logger.error(f"Error checking feature drift for room {room_id}: {e}")
    
    async def _check_system_drift(self):
        """Check for system-wide drift patterns"""
        
        # Check if multiple rooms are experiencing drift simultaneously
        recent_alerts = [
            alert for alert in self.drift_alerts 
            if alert.detected_at > datetime.now() - timedelta(hours=1)
        ]
        
        if len(recent_alerts) > 3:  # Multiple rooms with drift
            # This might indicate a system-wide change
            affected_rooms = list(set(alert.room_id for alert in recent_alerts))
            
            await self._trigger_system_drift_alert(
                affected_rooms,
                "Multiple rooms experiencing drift simultaneously",
                "system_wide_adaptation"
            )
    
    async def _trigger_drift_alert(self, room_id: str, drift_type: DriftType, 
                                 severity: float, affected_features: List[str],
                                 description: str, recommended_action: str):
        """Trigger a drift alert"""
        
        # Check cooldown period
        if room_id in self.last_alerts:
            if datetime.now() - self.last_alerts[room_id] < self.alert_cooldown:
                return
        
        alert = DriftAlert(
            room_id=room_id,
            drift_type=drift_type,
            severity=severity,
            detected_at=datetime.now(),
            confidence=0.8,  # Fixed confidence for now
            affected_features=affected_features,
            description=description,
            recommended_action=recommended_action
        )
        
        self.drift_alerts.append(alert)
        self.total_drifts_detected += 1
        self.room_drift_counts[room_id] += 1
        self.last_alerts[room_id] = datetime.now()
        
        logger.warning(f"Drift alert: {description}")
        
        # Send alert to monitoring system
        await self._send_alert_to_monitoring(alert)
    
    async def _trigger_system_drift_alert(self, affected_rooms: List[str], 
                                        description: str, recommended_action: str):
        """Trigger a system-wide drift alert"""
        
        alert = DriftAlert(
            room_id="system",
            drift_type=DriftType.SUDDEN,
            severity=0.9,
            detected_at=datetime.now(),
            confidence=0.9,
            affected_features=["system_wide"],
            description=description,
            recommended_action=recommended_action
        )
        
        self.drift_alerts.append(alert)
        
        logger.critical(f"System-wide drift alert: {description}")
        
        await self._send_alert_to_monitoring(alert)
    
    async def _send_alert_to_monitoring(self, alert: DriftAlert):
        """Send alert to monitoring system"""
        
        # This would integrate with the monitoring system
        # For now, just log the alert
        alert_data = {
            'room_id': alert.room_id,
            'drift_type': alert.drift_type.value,
            'severity': alert.severity,
            'detected_at': alert.detected_at.isoformat(),
            'confidence': alert.confidence,
            'affected_features': alert.affected_features,
            'description': alert.description,
            'recommended_action': alert.recommended_action
        }
        
        logger.info(f"Drift alert sent to monitoring: {json.dumps(alert_data)}")
    
    async def _process_alerts(self, predictor):
        """Process drift alerts and trigger appropriate actions"""
        
        # Get unprocessed alerts
        recent_alerts = [
            alert for alert in self.drift_alerts[-10:]  # Last 10 alerts
            if alert.detected_at > datetime.now() - timedelta(minutes=5)
        ]
        
        for alert in recent_alerts:
            try:
                await self._handle_drift_alert(alert, predictor)
            except Exception as e:
                logger.error(f"Error handling drift alert: {e}")
    
    async def _handle_drift_alert(self, alert: DriftAlert, predictor):
        """Handle a specific drift alert"""
        
        if alert.recommended_action == "retrain_model":
            # Trigger model retraining
            logger.info(f"Triggering model retraining for room {alert.room_id}")
            await predictor.retrain_model(alert.room_id)
            
        elif alert.recommended_action == "adapt_features":
            # Trigger feature adaptation
            logger.info(f"Triggering feature adaptation for room {alert.room_id}")
            await predictor.adapt_features(alert.room_id, alert.affected_features)
            
        elif alert.recommended_action == "system_wide_adaptation":
            # Trigger system-wide adaptation
            logger.info("Triggering system-wide adaptation")
            await predictor.system_wide_adaptation()
    
    def get_drift_statistics(self) -> Dict[str, Any]:
        """Get drift detection statistics"""
        
        recent_alerts = [
            alert for alert in self.drift_alerts
            if alert.detected_at > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            'total_drifts_detected': self.total_drifts_detected,
            'room_drift_counts': dict(self.room_drift_counts),
            'recent_alerts_24h': len(recent_alerts),
            'active_detectors': sum(len(detectors) for detectors in self.room_detectors.values()),
            'monitoring_active': self.monitoring_active,
            'alert_types': {
                drift_type.value: sum(1 for alert in recent_alerts if alert.drift_type == drift_type)
                for drift_type in DriftType
            }
        }
    
    def get_room_drift_info(self, room_id: str) -> Dict[str, Any]:
        """Get drift information for a specific room"""
        
        if room_id not in self.room_detectors:
            return {'room_id': room_id, 'detectors': 0}
        
        detectors_info = {}
        for feature_name, detector in self.room_detectors[room_id].items():
            detectors_info[feature_name] = detector.get_statistics()
        
        room_alerts = [
            alert for alert in self.drift_alerts
            if alert.room_id == room_id and alert.detected_at > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            'room_id': room_id,
            'detectors': len(self.room_detectors[room_id]),
            'detectors_info': detectors_info,
            'drift_count': self.room_drift_counts[room_id],
            'recent_alerts_24h': len(room_alerts),
            'last_alert': room_alerts[-1].detected_at.isoformat() if room_alerts else None
        }
    
    async def health_check(self) -> str:
        """Health check for drift detection system"""
        
        try:
            if not self.monitoring_active:
                return "inactive"
            
            # Check if detectors are working
            active_detectors = sum(len(detectors) for detectors in self.room_detectors.values())
            
            if active_detectors == 0:
                return "no_detectors"
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"Drift detector health check failed: {e}")
            return "error"
    
    async def shutdown(self):
        """Shutdown drift detection system"""
        
        self.monitoring_active = False
        logger.info("Drift detection system shut down")