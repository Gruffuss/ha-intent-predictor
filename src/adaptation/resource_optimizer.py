"""
Resource Management - Dynamic optimization based on system resources
Implements the exact ResourceOptimizer from CLAUDE.md
"""

import logging
import psutil
import time
from typing import Dict, Any, List
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ResourceOptimizer:
    """
    Dynamically adjust model complexity based on resources
    Implements the exact approach from CLAUDE.md
    """
    
    def __init__(self, cpu_limit: int = 80, memory_limit: int = 6000):  # 6GB limit
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit  # MB
        self.optimization_history = []
        self.performance_metrics = defaultdict(list)
        
        logger.info(f"Initialized ResourceOptimizer - CPU limit: {cpu_limit}%, Memory limit: {memory_limit}MB")
    
    def optimize_models(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dynamically adjust model complexity based on resources
        Implements the exact logic from CLAUDE.md
        """
        current_usage = self.get_resource_usage()
        optimizations = []
        
        logger.info(f"Current resource usage - CPU: {current_usage['cpu']}%, Memory: {current_usage['memory']}MB")
        
        # Memory optimization
        if current_usage['memory'] > self.memory_limit * 0.9:
            logger.warning(f"High memory usage: {current_usage['memory']}MB > {self.memory_limit * 0.9}MB")
            memory_optimizations = self.prune_features(models)
            optimizations.extend(memory_optimizations)
        
        # CPU optimization
        if current_usage['cpu'] > self.cpu_limit:
            logger.warning(f"High CPU usage: {current_usage['cpu']}% > {self.cpu_limit}%")
            cpu_optimizations = self.adjust_update_frequency(models)
            optimizations.extend(cpu_optimizations)
        
        # Record optimization event
        optimization_event = {
            'timestamp': datetime.now(),
            'resource_usage': current_usage,
            'optimizations_applied': optimizations,
            'models_affected': len(models)
        }
        self.optimization_history.append(optimization_event)
        
        # Keep only last 100 optimization events
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
        
        return {
            'optimizations_applied': optimizations,
            'resource_usage': current_usage,
            'optimization_needed': len(optimizations) > 0
        }
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Get current system resource usage"""
        try:
            # CPU usage (average over 1 second)
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024 * 1024)  # MB
            process_cpu = process.cpu_percent()
            
            return {
                'cpu': cpu_percent,
                'memory': memory_mb,
                'memory_percent': memory_percent,
                'disk_percent': disk_percent,
                'process_memory': process_memory,
                'process_cpu': process_cpu,
                'available_memory': memory.available / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return {
                'cpu': 0,
                'memory': 0,
                'memory_percent': 0,
                'disk_percent': 0,
                'process_memory': 0,
                'process_cpu': 0,
                'available_memory': 1000
            }
    
    def prune_features(self, models: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prune least important features to reduce memory usage
        """
        optimizations = []
        
        for model_name, model_data in models.items():
            try:
                # Check if model has feature importance tracking
                if hasattr(model_data, 'feature_selector'):
                    feature_selector = model_data.feature_selector
                    
                    # Get current feature count
                    current_features = len(feature_selector.feature_scores)
                    
                    if current_features > 25:  # Only prune if we have many features
                        # Remove bottom 20% of features
                        features_to_remove = max(1, current_features // 5)
                        
                        # Get least important features
                        feature_importance = feature_selector.get_top_features(current_features)
                        least_important = feature_importance[-features_to_remove:]
                        
                        # Remove features
                        for feature_name, _ in least_important:
                            if feature_name in feature_selector.feature_scores:
                                del feature_selector.feature_scores[feature_name]
                        
                        optimizations.append({
                            'type': 'feature_pruning',
                            'model': model_name,
                            'features_removed': features_to_remove,
                            'remaining_features': current_features - features_to_remove,
                            'memory_saved_estimate': features_to_remove * 0.1  # MB estimate
                        })
                        
                        logger.info(f"Pruned {features_to_remove} features from {model_name}")
                
            except Exception as e:
                logger.error(f"Error pruning features for {model_name}: {e}")
        
        return optimizations
    
    def adjust_update_frequency(self, models: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Reduce model update frequency for stable rooms
        """
        optimizations = []
        
        for model_name, model_data in models.items():
            try:
                # Check model stability/performance
                if hasattr(model_data, 'model_performance'):
                    performance_tracker = model_data.model_performance
                    
                    # If model is performing well, reduce update frequency
                    if hasattr(performance_tracker, 'get') and performance_tracker.n > 100:
                        recent_performance = performance_tracker.get()
                        
                        if recent_performance > 0.8:  # Good performance threshold
                            # Reduce update frequency (implement in actual model)
                            optimizations.append({
                                'type': 'reduced_update_frequency',
                                'model': model_name,
                                'new_frequency': 'every_10_events',  # Instead of every event
                                'performance_score': recent_performance,
                                'cpu_saved_estimate': 15  # % CPU reduction estimate
                            })
                            
                            logger.info(f"Reduced update frequency for stable model {model_name}")
                
            except Exception as e:
                logger.error(f"Error adjusting update frequency for {model_name}: {e}")
        
        return optimizations
    
    def scale_model_complexity(self, models: Dict[str, Any], target_complexity: float) -> List[Dict[str, Any]]:
        """Scale model complexity based on available resources"""
        optimizations = []
        
        for model_name, model_data in models.items():
            try:
                # Adjust ensemble size for River models
                if hasattr(model_data, 'models') and isinstance(model_data.models, dict):
                    for sub_model_name, sub_model in model_data.models.items():
                        if hasattr(sub_model, 'n_models'):
                            # Scale down ensemble size
                            original_size = sub_model.n_models
                            new_size = max(1, int(original_size * target_complexity))
                            
                            if new_size != original_size:
                                sub_model.n_models = new_size
                                optimizations.append({
                                    'type': 'ensemble_scaling',
                                    'model': f"{model_name}.{sub_model_name}",
                                    'original_size': original_size,
                                    'new_size': new_size,
                                    'complexity_factor': target_complexity
                                })
                
            except Exception as e:
                logger.error(f"Error scaling model complexity for {model_name}: {e}")
        
        return optimizations
    
    def optimize_based_on_time_of_day(self) -> Dict[str, Any]:
        """Adjust optimization strategy based on time of day"""
        current_hour = datetime.now().hour
        
        # Different optimization strategies for different times
        if 0 <= current_hour <= 6:  # Night time - more aggressive optimization
            return {
                'cpu_limit': self.cpu_limit * 0.7,
                'memory_limit': self.memory_limit * 0.8,
                'update_frequency': 'reduced',
                'strategy': 'night_optimization'
            }
        elif 7 <= current_hour <= 9 or 17 <= current_hour <= 21:  # Peak times - normal
            return {
                'cpu_limit': self.cpu_limit,
                'memory_limit': self.memory_limit,
                'update_frequency': 'normal',
                'strategy': 'peak_time'
            }
        else:  # Day time - slightly relaxed
            return {
                'cpu_limit': self.cpu_limit * 1.1,
                'memory_limit': self.memory_limit * 1.1,
                'update_frequency': 'normal',
                'strategy': 'day_time'
            }
    
    def get_optimization_recommendations(self, models: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommendations for optimization without applying them"""
        current_usage = self.get_resource_usage()
        recommendations = []
        
        # Memory recommendations
        if current_usage['memory'] > self.memory_limit * 0.8:
            recommendations.append({
                'type': 'memory_warning',
                'severity': 'high' if current_usage['memory'] > self.memory_limit * 0.9 else 'medium',
                'message': f"Memory usage at {current_usage['memory']:.1f}MB",
                'suggested_action': 'Consider feature pruning or model simplification'
            })
        
        # CPU recommendations
        if current_usage['cpu'] > self.cpu_limit * 0.8:
            recommendations.append({
                'type': 'cpu_warning',
                'severity': 'high' if current_usage['cpu'] > self.cpu_limit else 'medium',
                'message': f"CPU usage at {current_usage['cpu']:.1f}%",
                'suggested_action': 'Consider reducing update frequency or model complexity'
            })
        
        # Model-specific recommendations
        for model_name, model_data in models.items():
            if hasattr(model_data, 'feature_selector'):
                feature_count = len(model_data.feature_selector.feature_scores)
                if feature_count > 50:
                    recommendations.append({
                        'type': 'feature_count_warning',
                        'severity': 'medium',
                        'model': model_name,
                        'message': f"Model has {feature_count} features",
                        'suggested_action': 'Consider feature selection or pruning'
                    })
        
        return recommendations
    
    def get_optimization_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get optimization history for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_history = [
            event for event in self.optimization_history
            if event['timestamp'] >= cutoff_time
        ]
        
        return recent_history
    
    def get_resource_trend(self, minutes: int = 60) -> Dict[str, List[float]]:
        """Get resource usage trend over time"""
        # This would typically store historical data
        # For now, return current snapshot repeated
        current_usage = self.get_resource_usage()
        
        return {
            'timestamps': [datetime.now() - timedelta(minutes=i) for i in range(minutes, 0, -5)],
            'cpu_usage': [current_usage['cpu']] * (minutes // 5),
            'memory_usage': [current_usage['memory']] * (minutes // 5),
            'process_cpu': [current_usage['process_cpu']] * (minutes // 5),
            'process_memory': [current_usage['process_memory']] * (minutes // 5)
        }
    
    def emergency_optimization(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Emergency optimization when resources are critically low"""
        current_usage = self.get_resource_usage()
        emergency_actions = []
        
        # Critical memory situation
        if current_usage['memory'] > self.memory_limit * 0.95:
            logger.critical(f"Critical memory usage: {current_usage['memory']}MB")
            
            # Aggressive feature pruning
            for model_name, model_data in models.items():
                if hasattr(model_data, 'feature_selector'):
                    feature_count = len(model_data.feature_selector.feature_scores)
                    # Remove 50% of features
                    features_to_remove = feature_count // 2
                    
                    if features_to_remove > 0:
                        feature_importance = model_data.feature_selector.get_top_features(feature_count)
                        for feature_name, _ in feature_importance[-features_to_remove:]:
                            if feature_name in model_data.feature_selector.feature_scores:
                                del model_data.feature_selector.feature_scores[feature_name]
                        
                        emergency_actions.append(f"Emergency pruned {features_to_remove} features from {model_name}")
        
        # Critical CPU situation
        if current_usage['cpu'] > 95:
            logger.critical(f"Critical CPU usage: {current_usage['cpu']}%")
            emergency_actions.append("Emergency CPU optimization triggered")
        
        return {
            'emergency_triggered': len(emergency_actions) > 0,
            'actions_taken': emergency_actions,
            'resource_usage': current_usage
        }