"""
Performance Monitoring Service for HA Intent Predictor.

Tracks system health, performance metrics, and alerting
as specified in CLAUDE.md.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import psutil
import threading

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_connections: int
    load_average: List[float]
    timestamp: datetime


@dataclass
class PredictionMetrics:
    """Prediction performance metrics"""
    total_predictions: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    response_time_ms: float
    error_rate: float
    timestamp: datetime


@dataclass
class RoomMetrics:
    """Per-room performance metrics"""
    room: str
    predictions_made: int
    accuracy: float
    last_prediction: datetime
    model_performance: Dict[str, float]
    anomaly_score: float
    active_patterns: List[str]


class RunningStats:
    """Calculate running statistics without storing all values"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.sum = 0.0
        self.sum_squares = 0.0
    
    def update(self, value: float):
        """Update with a new value"""
        if len(self.values) == self.window_size:
            # Remove oldest value from sums
            old_value = self.values[0]
            self.sum -= old_value
            self.sum_squares -= old_value * old_value
        
        self.values.append(value)
        self.sum += value
        self.sum_squares += value * value
    
    def mean(self) -> float:
        """Calculate mean"""
        if not self.values:
            return 0.0
        return self.sum / len(self.values)
    
    def variance(self) -> float:
        """Calculate variance"""
        if len(self.values) < 2:
            return 0.0
        mean_val = self.mean()
        return (self.sum_squares / len(self.values)) - (mean_val * mean_val)
    
    def std_dev(self) -> float:
        """Calculate standard deviation"""
        return self.variance() ** 0.5
    
    def count(self) -> int:
        """Get count of values"""
        return len(self.values)


class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(self, alert_config: Dict):
        self.alert_config = alert_config
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        
        # Alert thresholds
        self.thresholds = alert_config.get('alerts', {})
        
        logger.info("Alert manager initialized")
    
    async def check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and trigger alerts"""
        
        current_time = datetime.now()
        new_alerts = []
        
        # Check system metrics
        if 'system' in metrics:
            system_metrics = metrics['system']
            
            # CPU usage alert
            if system_metrics.cpu_usage > self.thresholds.get('cpu_usage_threshold', 0.8):
                await self.trigger_alert('high_cpu_usage', {
                    'current': system_metrics.cpu_usage,
                    'threshold': self.thresholds.get('cpu_usage_threshold', 0.8)
                })
            
            # Memory usage alert
            if system_metrics.memory_usage > self.thresholds.get('memory_usage_threshold', 0.8):
                await self.trigger_alert('high_memory_usage', {
                    'current': system_metrics.memory_usage,
                    'threshold': self.thresholds.get('memory_usage_threshold', 0.8)
                })
        
        # Check prediction metrics
        if 'predictions' in metrics:
            pred_metrics = metrics['predictions']
            
            # Accuracy alert
            if pred_metrics.accuracy < self.thresholds.get('prediction_accuracy_threshold', 0.7):
                await self.trigger_alert('low_prediction_accuracy', {
                    'current': pred_metrics.accuracy,
                    'threshold': self.thresholds.get('prediction_accuracy_threshold', 0.7)
                })
            
            # Error rate alert
            if pred_metrics.error_rate > self.thresholds.get('error_rate_threshold', 0.05):
                await self.trigger_alert('high_error_rate', {
                    'current': pred_metrics.error_rate,
                    'threshold': self.thresholds.get('error_rate_threshold', 0.05)
                })
            
            # Response time alert
            if pred_metrics.response_time_ms > self.thresholds.get('response_time_threshold', 500):
                await self.trigger_alert('high_response_time', {
                    'current': pred_metrics.response_time_ms,
                    'threshold': self.thresholds.get('response_time_threshold', 500)
                })
    
    async def trigger_alert(self, alert_type: str, details: Dict):
        """Trigger an alert"""
        
        alert_id = f"{alert_type}_{int(time.time())}"
        
        alert = {
            'id': alert_id,
            'type': alert_type,
            'details': details,
            'timestamp': datetime.now(),
            'resolved': False
        }
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"Alert triggered: {alert_type} - {details}")
        
        # Here you would integrate with notification systems
        # (email, Slack, PagerDuty, etc.)
    
    def get_active_alerts(self) -> List[Dict]:
        """Get all active alerts"""
        return [alert for alert in self.active_alerts.values() if not alert['resolved']]
    
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id]['resolved'] = True
            self.active_alerts[alert_id]['resolved_at'] = datetime.now()
            logger.info(f"Alert resolved: {alert_id}")


class PrometheusExporter:
    """Export metrics in Prometheus format"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.metrics_cache = {}
        self.last_update = datetime.now()
        
        if enabled:
            logger.info("Prometheus exporter enabled")
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update metrics cache for Prometheus export"""
        if not self.enabled:
            return
        
        self.metrics_cache = metrics
        self.last_update = datetime.now()
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        if not self.enabled:
            return ""
        
        lines = []
        
        # System metrics
        if 'system' in self.metrics_cache:
            system = self.metrics_cache['system']
            lines.append(f'# HELP ha_predictor_cpu_usage CPU usage percentage')
            lines.append(f'# TYPE ha_predictor_cpu_usage gauge')
            lines.append(f'ha_predictor_cpu_usage {system.cpu_usage}')
            
            lines.append(f'# HELP ha_predictor_memory_usage Memory usage percentage')
            lines.append(f'# TYPE ha_predictor_memory_usage gauge')
            lines.append(f'ha_predictor_memory_usage {system.memory_usage}')
            
            lines.append(f'# HELP ha_predictor_disk_usage Disk usage percentage')
            lines.append(f'# TYPE ha_predictor_disk_usage gauge')
            lines.append(f'ha_predictor_disk_usage {system.disk_usage}')
        
        # Prediction metrics
        if 'predictions' in self.metrics_cache:
            pred = self.metrics_cache['predictions']
            lines.append(f'# HELP ha_predictor_total_predictions Total predictions made')
            lines.append(f'# TYPE ha_predictor_total_predictions counter')
            lines.append(f'ha_predictor_total_predictions {pred.total_predictions}')
            
            lines.append(f'# HELP ha_predictor_accuracy Prediction accuracy')
            lines.append(f'# TYPE ha_predictor_accuracy gauge')
            lines.append(f'ha_predictor_accuracy {pred.accuracy}')
            
            lines.append(f'# HELP ha_predictor_response_time Response time in milliseconds')
            lines.append(f'# TYPE ha_predictor_response_time gauge')
            lines.append(f'ha_predictor_response_time {pred.response_time_ms}')
            
            lines.append(f'# HELP ha_predictor_error_rate Error rate')
            lines.append(f'# TYPE ha_predictor_error_rate gauge')
            lines.append(f'ha_predictor_error_rate {pred.error_rate}')
        
        # Room metrics
        if 'rooms' in self.metrics_cache:
            for room, room_metrics in self.metrics_cache['rooms'].items():
                lines.append(f'# HELP ha_predictor_room_accuracy Per-room prediction accuracy')
                lines.append(f'# TYPE ha_predictor_room_accuracy gauge')
                lines.append(f'ha_predictor_room_accuracy{{room="{room}"}} {room_metrics.accuracy}')
                
                lines.append(f'# HELP ha_predictor_room_predictions Per-room prediction count')
                lines.append(f'# TYPE ha_predictor_room_predictions counter')
                lines.append(f'ha_predictor_room_predictions{{room="{room}"}} {room_metrics.predictions_made}')
        
        return '\n'.join(lines)


class PerformanceMonitor:
    """
    Main performance monitoring service.
    
    Collects, analyzes, and reports on system performance
    following CLAUDE.md specifications.
    """
    
    def __init__(self, metrics_config: Dict):
        self.metrics_config = metrics_config
        self.enabled = metrics_config.get('enabled', True)
        self.metrics_interval = metrics_config.get('metrics_interval', 60)
        self.performance_window = metrics_config.get('performance_window', 1000)
        
        # Metrics storage
        self.system_metrics = RunningStats(self.performance_window)
        self.prediction_metrics = RunningStats(self.performance_window)
        self.room_metrics = defaultdict(lambda: RunningStats(self.performance_window))
        
        # Current metrics
        self.current_metrics = {
            'system': None,
            'predictions': None,
            'rooms': {},
            'alerts': []
        }
        
        # Components
        self.alert_manager = AlertManager(metrics_config)
        self.prometheus_exporter = PrometheusExporter(
            metrics_config.get('prometheus', {}).get('enabled', True)
        )
        
        # System info
        self.start_time = datetime.now()
        self.last_metrics_update = datetime.now()
        
        # Prediction tracking
        self.prediction_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'response_times': deque(maxlen=1000),
            'accuracy_scores': deque(maxlen=1000)
        }
        
        # Room tracking
        self.room_stats = defaultdict(lambda: {
            'predictions_made': 0,
            'accuracy_scores': deque(maxlen=100),
            'last_prediction': None,
            'model_performance': {},
            'anomaly_scores': deque(maxlen=100)
        })
        
        logger.info("Performance monitor initialized")
    
    async def initialize(self):
        """Initialize monitoring components"""
        if not self.enabled:
            logger.info("Performance monitoring disabled")
            return
        
        logger.info("Starting performance monitoring...")
        
        # Start metrics collection
        self.metrics_task = asyncio.create_task(self._collect_metrics_loop())
        
        # Start alert checking
        self.alert_task = asyncio.create_task(self._check_alerts_loop())
        
        logger.info("Performance monitoring started")
    
    async def _collect_metrics_loop(self):
        """Main metrics collection loop"""
        while True:
            try:
                await self._collect_system_metrics()
                await self._collect_prediction_metrics()
                await self._collect_room_metrics()
                
                # Update Prometheus metrics
                self.prometheus_exporter.update_metrics(self.current_metrics)
                
                self.last_metrics_update = datetime.now()
                
                await asyncio.sleep(self.metrics_interval)
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(self.metrics_interval)
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_stats = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            
            # Active connections
            connections = len(psutil.net_connections())
            
            # Load average
            load_avg = list(psutil.getloadavg())
            
            # Create system metrics
            system_metrics = SystemMetrics(
                cpu_usage=cpu_percent / 100.0,
                memory_usage=memory_percent / 100.0,
                disk_usage=disk_percent / 100.0,
                network_io=network_stats,
                active_connections=connections,
                load_average=load_avg,
                timestamp=datetime.now()
            )
            
            self.current_metrics['system'] = system_metrics
            
            # Update running stats
            self.system_metrics.update(cpu_percent)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_prediction_metrics(self):
        """Collect prediction performance metrics"""
        try:
            # Calculate metrics from stored stats
            total_predictions = self.prediction_stats['total_predictions']
            successful_predictions = self.prediction_stats['successful_predictions']
            failed_predictions = self.prediction_stats['failed_predictions']
            
            accuracy = (successful_predictions / total_predictions) if total_predictions > 0 else 0.0
            error_rate = (failed_predictions / total_predictions) if total_predictions > 0 else 0.0
            
            # Average response time
            response_times = list(self.prediction_stats['response_times'])
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
            
            # Calculate precision, recall, F1 (simplified)
            precision = accuracy  # Simplified
            recall = accuracy     # Simplified
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            prediction_metrics = PredictionMetrics(
                total_predictions=total_predictions,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                response_time_ms=avg_response_time,
                error_rate=error_rate,
                timestamp=datetime.now()
            )
            
            self.current_metrics['predictions'] = prediction_metrics
            
        except Exception as e:
            logger.error(f"Error collecting prediction metrics: {e}")
    
    async def _collect_room_metrics(self):
        """Collect per-room performance metrics"""
        try:
            room_metrics = {}
            
            for room, stats in self.room_stats.items():
                accuracy_scores = list(stats['accuracy_scores'])
                avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
                
                anomaly_scores = list(stats['anomaly_scores'])
                avg_anomaly = sum(anomaly_scores) / len(anomaly_scores) if anomaly_scores else 0.0
                
                room_metric = RoomMetrics(
                    room=room,
                    predictions_made=stats['predictions_made'],
                    accuracy=avg_accuracy,
                    last_prediction=stats['last_prediction'] or datetime.now(),
                    model_performance=stats['model_performance'].copy(),
                    anomaly_score=avg_anomaly,
                    active_patterns=[]  # Would be populated from pattern discovery
                )
                
                room_metrics[room] = room_metric
            
            self.current_metrics['rooms'] = room_metrics
            
        except Exception as e:
            logger.error(f"Error collecting room metrics: {e}")
    
    async def _check_alerts_loop(self):
        """Check for alerts periodically"""
        while True:
            try:
                await self.alert_manager.check_alerts(self.current_metrics)
                
                # Update current alerts
                self.current_metrics['alerts'] = self.alert_manager.get_active_alerts()
                
                await asyncio.sleep(self.metrics_interval)
                
            except Exception as e:
                logger.error(f"Error checking alerts: {e}")
                await asyncio.sleep(self.metrics_interval)
    
    async def record_prediction(self, room: str, prediction_time_ms: float, 
                              accuracy: float = None, success: bool = True):
        """Record a prediction event"""
        
        # Update global stats
        self.prediction_stats['total_predictions'] += 1
        if success:
            self.prediction_stats['successful_predictions'] += 1
        else:
            self.prediction_stats['failed_predictions'] += 1
        
        self.prediction_stats['response_times'].append(prediction_time_ms)
        
        if accuracy is not None:
            self.prediction_stats['accuracy_scores'].append(accuracy)
        
        # Update room stats
        self.room_stats[room]['predictions_made'] += 1
        self.room_stats[room]['last_prediction'] = datetime.now()
        
        if accuracy is not None:
            self.room_stats[room]['accuracy_scores'].append(accuracy)
    
    async def record_anomaly(self, room: str, anomaly_score: float):
        """Record an anomaly detection event"""
        self.room_stats[room]['anomaly_scores'].append(anomaly_score)
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        return {
            'system': asdict(self.current_metrics['system']) if self.current_metrics['system'] else None,
            'predictions': asdict(self.current_metrics['predictions']) if self.current_metrics['predictions'] else None,
            'rooms': {k: asdict(v) for k, v in self.current_metrics['rooms'].items()},
            'alerts': self.current_metrics['alerts'],
            'uptime': (datetime.now() - self.start_time).total_seconds(),
            'last_update': self.last_metrics_update.isoformat(),
            'active_models': len(self.room_stats)
        }
    
    async def get_metrics_range(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get metrics for a time range"""
        # This would query historical metrics from storage
        # For now, return current metrics
        return await self.get_current_metrics()
    
    async def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return self.prometheus_exporter.export_metrics()
    
    async def run_monitoring(self):
        """Main monitoring loop"""
        if not self.enabled:
            return
        
        try:
            # Wait for tasks to complete
            await asyncio.gather(
                self.metrics_task,
                self.alert_task
            )
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            raise
    
    async def health_check(self) -> str:
        """Health check for monitoring service"""
        if not self.enabled:
            return "disabled"
        
        try:
            # Check if metrics are being updated
            time_since_update = (datetime.now() - self.last_metrics_update).total_seconds()
            
            if time_since_update > self.metrics_interval * 2:
                return "stale"
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return "error"
    
    async def shutdown(self):
        """Shutdown monitoring service"""
        logger.info("Shutting down performance monitoring...")
        
        # Cancel tasks
        if hasattr(self, 'metrics_task'):
            self.metrics_task.cancel()
        
        if hasattr(self, 'alert_task'):
            self.alert_task.cancel()
        
        logger.info("Performance monitoring shut down")