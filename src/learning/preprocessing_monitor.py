"""
Preprocessing Monitor for Smart Data Preprocessor
Provides logging, metrics, and monitoring for preprocessing decisions
"""

import logging
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import asdict
from collections import defaultdict, Counter
import numpy as np

from .smart_data_preprocessor import DataCharacteristics, PreprocessingDecision, OccupancyPattern

logger = logging.getLogger(__name__)


class PreprocessingMetrics:
    """Metrics collection for preprocessing operations"""
    
    def __init__(self):
        self.reset_metrics()
        
    def reset_metrics(self):
        """Reset all metrics"""
        self.total_requests = 0
        self.preprocessing_applied = 0
        self.stumpy_ready_rate = 0.0
        self.method_usage = Counter()
        self.pattern_type_distribution = Counter()
        self.room_statistics = defaultdict(dict)
        self.error_count = 0
        self.processing_time_stats = []
        
    def record_preprocessing(self, room_name: str, decision: PreprocessingDecision, 
                           processing_time: float, validation_results: Dict[str, Any]):
        """Record a preprocessing operation"""
        self.total_requests += 1
        
        if decision.should_preprocess:
            self.preprocessing_applied += 1
            
        self.method_usage[decision.method] += 1
        self.pattern_type_distribution[decision.original_characteristics.pattern_type.value] += 1
        
        # Update room statistics
        if room_name not in self.room_statistics:
            self.room_statistics[room_name] = {
                'total_operations': 0,
                'preprocessing_rate': 0.0,
                'success_rate': 0.0,
                'common_methods': Counter(),
                'avg_occupancy_rate': 0.0,
                'occupancy_rates': []
            }
            
        room_stats = self.room_statistics[room_name]
        room_stats['total_operations'] += 1
        room_stats['common_methods'][decision.method] += 1
        room_stats['occupancy_rates'].append(decision.original_characteristics.occupancy_rate)
        room_stats['avg_occupancy_rate'] = np.mean(room_stats['occupancy_rates'])
        
        # Update rates
        room_stats['preprocessing_rate'] = room_stats['common_methods'].total() / room_stats['total_operations']
        room_stats['success_rate'] = sum(1 for op in room_stats.get('operations', []) 
                                        if op.get('stumpy_ready', False)) / room_stats['total_operations']
        
        self.processing_time_stats.append(processing_time)
        
        # Update global success rate
        if validation_results.get('stumpy_ready', False):
            self.stumpy_ready_rate = ((self.stumpy_ready_rate * (self.total_requests - 1)) + 1) / self.total_requests
        else:
            self.stumpy_ready_rate = (self.stumpy_ready_rate * (self.total_requests - 1)) / self.total_requests
            
    def record_error(self, error_type: str, error_message: str):
        """Record a preprocessing error"""
        self.error_count += 1
        logger.error(f"Preprocessing error ({error_type}): {error_message}")
        
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        if self.total_requests == 0:
            return {'no_data': True}
            
        return {
            'total_requests': self.total_requests,
            'preprocessing_rate': self.preprocessing_applied / self.total_requests,
            'stumpy_ready_rate': self.stumpy_ready_rate,
            'error_rate': self.error_count / self.total_requests,
            'method_usage': dict(self.method_usage),
            'pattern_distribution': dict(self.pattern_type_distribution),
            'avg_processing_time': np.mean(self.processing_time_stats) if self.processing_time_stats else 0,
            'room_count': len(self.room_statistics),
            'top_rooms_by_operations': sorted(
                [(room, stats['total_operations']) for room, stats in self.room_statistics.items()],
                key=lambda x: x[1], reverse=True
            )[:10]
        }


class PreprocessingLogger:
    """Advanced logging for preprocessing operations"""
    
    def __init__(self, log_level: int = logging.INFO):
        self.logger = logging.getLogger(f"{__name__}.PreprocessingLogger")
        self.logger.setLevel(log_level)
        
        # Create specialized formatter for preprocessing logs
        formatter = logging.Formatter(
            '%(asctime)s - PREPROCESSING - %(levelname)s - %(message)s'
        )
        
        # Add handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
    def log_analysis_start(self, room_name: str, data_length: int, window_size: int):
        """Log start of data analysis"""
        self.logger.info(f"🔍 Starting preprocessing analysis: {room_name}")
        self.logger.info(f"   Data points: {data_length:,}, Window size: {window_size}")
        
    def log_characteristics(self, characteristics: DataCharacteristics):
        """Log detailed data characteristics"""
        self.logger.info(f"📊 Data characteristics analysis:")
        self.logger.info(f"   Pattern type: {characteristics.pattern_type.value}")
        self.logger.info(f"   Occupancy rate: {characteristics.occupancy_rate:.3f}")
        self.logger.info(f"   Transition rate: {characteristics.transition_rate:.4f}")
        self.logger.info(f"   Unique values: {characteristics.unique_values}")
        self.logger.info(f"   Std deviation: {characteristics.std_deviation:.8f}")
        self.logger.info(f"   Constant windows: {characteristics.constant_window_percentage:.1f}%")
        
        if characteristics.constant_window_percentage > 80:
            self.logger.warning(f"⚠️ High constant window percentage detected: {characteristics.constant_window_percentage:.1f}%")
            
    def log_decision(self, decision: PreprocessingDecision):
        """Log preprocessing decision"""
        if decision.should_preprocess:
            self.logger.info(f"🔧 Preprocessing REQUIRED: {decision.method}")
            self.logger.info(f"   Reason: {decision.reason}")
            self.logger.info(f"   Threshold: {decision.constant_threshold}%")
        else:
            self.logger.info(f"✅ No preprocessing needed")
            self.logger.info(f"   Reason: {decision.reason}")
            
    def log_preprocessing_start(self, method: str, parameters: Dict[str, Any]):
        """Log start of preprocessing"""
        self.logger.info(f"🚀 Applying preprocessing method: {method}")
        self.logger.debug(f"   Parameters: {json.dumps(parameters, indent=2, default=str)}")
        
    def log_preprocessing_complete(self, original_std: float, processed_std: float, 
                                 sequences_processed: int = None):
        """Log completion of preprocessing"""
        improvement = (processed_std / original_std) if original_std > 0 else float('inf')
        self.logger.info(f"✅ Preprocessing complete:")
        self.logger.info(f"   Std deviation: {original_std:.8f} → {processed_std:.8f} ({improvement:.2f}x)")
        
        if sequences_processed is not None:
            self.logger.info(f"   Sequences enhanced: {sequences_processed}")
            
    def log_validation_results(self, validation: Dict[str, Any]):
        """Log validation results"""
        self.logger.info(f"🔍 Validation results:")
        self.logger.info(f"   Binary semantics preserved: {validation.get('binary_semantics_preserved', 'N/A')}")
        self.logger.info(f"   STUMPY ready: {validation.get('stumpy_ready', 'N/A')}")
        
        if 'constant_window_percentage' in validation:
            self.logger.info(f"   Final constant windows: {validation['constant_window_percentage']:.1f}%")
            
        if not validation.get('stumpy_ready', True):
            self.logger.warning(f"⚠️ Data may still have STUMPY compatibility issues")
            
    def log_error(self, error_type: str, error_message: str, room_name: str = None):
        """Log preprocessing errors"""
        context = f" for {room_name}" if room_name else ""
        self.logger.error(f"❌ Preprocessing error{context} ({error_type}): {error_message}")
        
    def log_performance_warning(self, processing_time: float, threshold: float = 1.0):
        """Log performance warnings"""
        if processing_time > threshold:
            self.logger.warning(f"⏱️ Slow preprocessing: {processing_time:.2f}s > {threshold:.2f}s threshold")
            
    def log_room_pattern_insights(self, room_name: str, pattern_type: OccupancyPattern, 
                                 occupancy_rate: float, recommendation: str):
        """Log insights about room patterns"""
        self.logger.info(f"🏠 Room pattern insights for {room_name}:")
        self.logger.info(f"   Type: {pattern_type.value} ({occupancy_rate:.1%} occupancy)")
        self.logger.info(f"   Recommendation: {recommendation}")


class PreprocessingMonitor:
    """Comprehensive monitoring system for preprocessing operations"""
    
    def __init__(self, enable_detailed_logging: bool = True):
        self.metrics = PreprocessingMetrics()
        self.logger = PreprocessingLogger()
        self.enable_detailed_logging = enable_detailed_logging
        self.session_start_time = datetime.now(timezone.utc)
        
    async def monitor_preprocessing(self, room_name: str, data: np.ndarray, 
                                  window_size: int, preprocessor) -> tuple:
        """
        Monitor a complete preprocessing operation
        
        Returns:
            Tuple of (processed_data, processing_info, monitoring_data)
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            if self.enable_detailed_logging:
                self.logger.log_analysis_start(room_name, len(data), window_size)
            
            # Run preprocessing with monitoring
            processed_data, processing_info = preprocessor.preprocess_for_stumpy(
                data, window_size, room_name
            )
            
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Extract decision and characteristics from preprocessing history
            if room_name in preprocessor.preprocessing_history:
                latest_entry = preprocessor.preprocessing_history[room_name][-1]
                decision = latest_entry['decision']
                characteristics = latest_entry['characteristics']
                validation = latest_entry['validation']
                
                # Record metrics
                self.metrics.record_preprocessing(
                    room_name, decision, processing_time, validation
                )
                
                # Detailed logging
                if self.enable_detailed_logging:
                    self.logger.log_characteristics(characteristics)
                    self.logger.log_decision(decision)
                    
                    if decision.should_preprocess:
                        self.logger.log_preprocessing_complete(
                            characteristics.std_deviation,
                            processing_info['processed_std']
                        )
                    
                    self.logger.log_validation_results(validation)
                    self.logger.log_performance_warning(processing_time)
                    
                    # Generate room insights
                    recommendation = self._generate_room_recommendation(
                        characteristics.pattern_type, 
                        characteristics.occupancy_rate,
                        decision.should_preprocess
                    )
                    
                    self.logger.log_room_pattern_insights(
                        room_name, characteristics.pattern_type,
                        characteristics.occupancy_rate, recommendation
                    )
            
            # Create monitoring data
            monitoring_data = {
                'processing_time': processing_time,
                'session_uptime': (datetime.now(timezone.utc) - self.session_start_time).total_seconds(),
                'room_name': room_name,
                'data_length': len(data),
                'window_size': window_size,
                'success': True
            }
            
            return processed_data, processing_info, monitoring_data
            
        except Exception as e:
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            self.metrics.record_error(type(e).__name__, str(e))
            self.logger.log_error(type(e).__name__, str(e), room_name)
            
            monitoring_data = {
                'processing_time': processing_time,
                'session_uptime': (datetime.now(timezone.utc) - self.session_start_time).total_seconds(),
                'room_name': room_name,
                'data_length': len(data),
                'window_size': window_size,
                'success': False,
                'error': str(e)
            }
            
            # Re-raise the exception
            raise e
            
    def _generate_room_recommendation(self, pattern_type: OccupancyPattern, 
                                    occupancy_rate: float, preprocessing_applied: bool) -> str:
        """Generate recommendations based on room patterns"""
        if pattern_type == OccupancyPattern.VERY_LOW:
            if occupancy_rate < 0.01:
                return "Room rarely occupied - consider checking sensor placement or using different analysis approach"
            else:
                return "Low occupancy room - preprocessing likely needed for STUMPY analysis"
                
        elif pattern_type == OccupancyPattern.BALANCED:
            return "Well-balanced occupancy - ideal for pattern discovery without preprocessing"
            
        elif pattern_type == OccupancyPattern.VERY_HIGH:
            return "Almost always occupied - verify sensor functionality and consider different thresholds"
            
        else:
            if preprocessing_applied:
                return f"Pattern type {pattern_type.value} required preprocessing - monitor STUMPY performance"
            else:
                return f"Pattern type {pattern_type.value} suitable for direct analysis"
                
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        base_summary = self.metrics.get_summary()
        
        if base_summary.get('no_data'):
            return base_summary
            
        # Add session information
        base_summary['session_info'] = {
            'start_time': self.session_start_time.isoformat(),
            'uptime_seconds': (datetime.now(timezone.utc) - self.session_start_time).total_seconds(),
            'detailed_logging_enabled': self.enable_detailed_logging
        }
        
        # Add insights
        base_summary['insights'] = self._generate_insights(base_summary)
        
        return base_summary
        
    def _generate_insights(self, summary: Dict[str, Any]) -> List[str]:
        """Generate insights from metrics"""
        insights = []
        
        if summary['preprocessing_rate'] > 0.8:
            insights.append("High preprocessing rate indicates many rooms have problematic data patterns")
            
        if summary['stumpy_ready_rate'] < 0.9:
            insights.append("Low STUMPY ready rate - some rooms may still have division by zero risks")
            
        if summary['error_rate'] > 0.1:
            insights.append("High error rate - investigate preprocessing failures")
            
        # Pattern-based insights
        pattern_dist = summary.get('pattern_distribution', {})
        if pattern_dist.get('very_low', 0) > pattern_dist.get('balanced', 0):
            insights.append("Many low-occupancy rooms detected - typical for residential properties")
            
        # Method insights
        method_usage = summary.get('method_usage', {})
        if method_usage.get('microscopic_noise_with_trends', 0) > method_usage.get('none', 0):
            insights.append("Frequent use of advanced preprocessing indicates challenging data patterns")
            
        return insights
        
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        summary = self.get_metrics_summary()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        logger.info(f"📊 Preprocessing metrics exported to {filepath}")
        
    def log_session_summary(self):
        """Log a summary of the current session"""
        summary = self.get_metrics_summary()
        
        if summary.get('no_data'):
            logger.info("📊 No preprocessing operations recorded in this session")
            return
            
        logger.info("📊 Preprocessing Session Summary:")
        logger.info(f"   Total requests: {summary['total_requests']}")
        logger.info(f"   Preprocessing rate: {summary['preprocessing_rate']:.1%}")
        logger.info(f"   STUMPY ready rate: {summary['stumpy_ready_rate']:.1%}")
        logger.info(f"   Error rate: {summary['error_rate']:.1%}")
        logger.info(f"   Average processing time: {summary['avg_processing_time']:.3f}s")
        logger.info(f"   Rooms processed: {summary['room_count']}")
        
        # Log insights
        insights = summary.get('insights', [])
        if insights:
            logger.info("💡 Key insights:")
            for insight in insights:
                logger.info(f"   • {insight}")


# Global monitor instance for easy access
global_preprocessing_monitor = PreprocessingMonitor()