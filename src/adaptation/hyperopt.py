"""
Hyperparameter Optimization for HA Intent Predictor.

Continuous hyperparameter tuning using Optuna and adaptive methods
as specified in CLAUDE.md.
"""

import asyncio
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterTrial:
    """A hyperparameter optimization trial"""
    trial_id: str
    room_id: str
    model_type: str
    hyperparameters: Dict[str, Any]
    objective_value: float
    timestamp: datetime
    duration_seconds: float
    pruned: bool = False
    success: bool = True
    metadata: Dict[str, Any] = None


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization"""
    room_id: str
    model_type: str
    best_hyperparameters: Dict[str, Any]
    best_objective_value: float
    total_trials: int
    successful_trials: int
    optimization_time: float
    improvement: float
    timestamp: datetime


class ModelObjective:
    """Objective function for hyperparameter optimization"""
    
    def __init__(self, room_id: str, model_type: str, predictor, validation_data):
        self.room_id = room_id
        self.model_type = model_type
        self.predictor = predictor
        self.validation_data = validation_data
        self.trial_count = 0
        
    def __call__(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna"""
        
        self.trial_count += 1
        
        try:
            # Suggest hyperparameters based on model type
            hyperparams = self._suggest_hyperparameters(trial)
            
            # Train model with suggested hyperparameters
            model_performance = self._train_and_evaluate(hyperparams)
            
            # Return objective value (we want to maximize accuracy)
            return model_performance.get('accuracy', 0.0)
            
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            # Return poor performance to indicate failure
            return 0.0
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters based on model type"""
        
        if self.model_type == 'gradient_boost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            }
        
        elif self.model_type == 'neural_net':
            n_layers = trial.suggest_int('n_layers', 1, 4)
            hidden_dims = []
            for i in range(n_layers):
                hidden_dims.append(trial.suggest_int(f'hidden_dim_{i}', 10, 100))
            
            return {
                'hidden_dims': hidden_dims,
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
                'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
            }
        
        elif self.model_type == 'hoeffding_tree':
            return {
                'grace_period': trial.suggest_int('grace_period', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'split_confidence': trial.suggest_float('split_confidence', 1e-7, 1e-3, log=True),
                'tie_threshold': trial.suggest_float('tie_threshold', 0.01, 0.1),
                'leaf_prediction': trial.suggest_categorical('leaf_prediction', ['mc', 'nb', 'nba'])
            }
        
        elif self.model_type == 'ensemble':
            return {
                'voting': trial.suggest_categorical('voting', ['hard', 'soft']),
                'gb_weight': trial.suggest_float('gb_weight', 0.1, 0.8),
                'nn_weight': trial.suggest_float('nn_weight', 0.1, 0.8),
                'ht_weight': trial.suggest_float('ht_weight', 0.1, 0.8),
                'meta_learner': trial.suggest_categorical('meta_learner', ['logistic', 'decision_tree', 'random_forest'])
            }
        
        else:
            return {}
    
    def _train_and_evaluate(self, hyperparams: Dict[str, Any]) -> Dict[str, float]:
        """Train model with hyperparameters and evaluate performance"""
        
        # This would integrate with the actual model training
        # For now, simulate training and evaluation
        
        # Simulate training time based on complexity
        complexity_factor = self._calculate_complexity_factor(hyperparams)
        training_time = np.random.uniform(0.5, 2.0) * complexity_factor
        
        # Simulate performance based on hyperparameters
        # Better hyperparameters should generally lead to better performance
        base_performance = np.random.uniform(0.6, 0.9)
        
        # Adjust performance based on hyperparameters
        if self.model_type == 'gradient_boost':
            if 50 <= hyperparams.get('n_estimators', 100) <= 150:
                base_performance += 0.05
            if 0.05 <= hyperparams.get('learning_rate', 0.1) <= 0.15:
                base_performance += 0.03
        
        elif self.model_type == 'neural_net':
            if 0.001 <= hyperparams.get('learning_rate', 0.01) <= 0.01:
                base_performance += 0.04
            if 0.1 <= hyperparams.get('dropout', 0.2) <= 0.3:
                base_performance += 0.02
        
        # Add some noise
        performance = base_performance + np.random.normal(0, 0.02)
        performance = np.clip(performance, 0.0, 1.0)
        
        return {
            'accuracy': performance,
            'training_time': training_time,
            'complexity_factor': complexity_factor
        }
    
    def _calculate_complexity_factor(self, hyperparams: Dict[str, Any]) -> float:
        """Calculate complexity factor from hyperparameters"""
        
        factor = 1.0
        
        if 'n_estimators' in hyperparams:
            factor *= hyperparams['n_estimators'] / 100
        
        if 'max_depth' in hyperparams:
            factor *= hyperparams['max_depth'] / 6
        
        if 'hidden_dims' in hyperparams:
            factor *= len(hyperparams['hidden_dims']) / 2
            factor *= sum(hyperparams['hidden_dims']) / 60
        
        return factor


class AdaptiveHyperparameterOptimizer:
    """
    Adaptive hyperparameter optimization system.
    
    Continuously optimizes hyperparameters based on model performance
    and changing data patterns as specified in CLAUDE.md.
    """
    
    def __init__(self, timeseries_db, model_store, performance_monitor):
        self.timeseries_db = timeseries_db
        self.model_store = model_store
        self.performance_monitor = performance_monitor
        
        # Optimization tracking
        self.optimization_history = deque(maxlen=1000)
        self.room_optimizations = defaultdict(lambda: {
            'total_trials': 0,
            'successful_trials': 0,
            'best_objective': 0.0,
            'last_optimization': None,
            'current_study': None
        })
        
        # Configuration
        self.optimization_enabled = True
        self.optimization_interval = timedelta(hours=12)  # Optimize every 12 hours
        self.max_trials_per_optimization = 50
        self.min_improvement_threshold = 0.01
        
        # Optuna configuration
        self.sampler = TPESampler(seed=42)
        self.pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        logger.info("Adaptive hyperparameter optimizer initialized")
    
    async def initialize(self):
        """Initialize hyperparameter optimization system"""
        
        # Start optimization loop
        asyncio.create_task(self._optimization_loop())
        
        logger.info("Hyperparameter optimization system started")
    
    async def _optimization_loop(self):
        """Main optimization loop"""
        
        while self.optimization_enabled:
            try:
                # Get rooms that need optimization
                rooms_to_optimize = await self._get_rooms_needing_optimization()
                
                for room_id in rooms_to_optimize:
                    await self._optimize_room_hyperparameters(room_id)
                
                # Sleep until next optimization cycle
                await asyncio.sleep(self.optimization_interval.total_seconds())
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _get_rooms_needing_optimization(self) -> List[str]:
        """Get rooms that need hyperparameter optimization"""
        
        try:
            # Get active rooms
            active_rooms = await self.timeseries_db.fetch("""
                SELECT DISTINCT room 
                FROM sensor_events 
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                AND room IS NOT NULL
            """)
            
            rooms_needing_optimization = []
            
            for room_record in active_rooms:
                room_id = room_record['room']
                
                # Check if optimization is needed
                if await self._should_optimize_room(room_id):
                    rooms_needing_optimization.append(room_id)
            
            return rooms_needing_optimization
            
        except Exception as e:
            logger.error(f"Error getting rooms needing optimization: {e}")
            return []
    
    async def _should_optimize_room(self, room_id: str) -> bool:
        """Check if room needs hyperparameter optimization"""
        
        # Check if enough time has passed since last optimization
        last_optimization = self.room_optimizations[room_id]['last_optimization']
        if last_optimization and datetime.now() - last_optimization < self.optimization_interval:
            return False
        
        # Check if model performance is below threshold
        recent_performance = await self._get_recent_performance(room_id)
        if recent_performance and recent_performance < 0.75:
            return True
        
        # Check if performance has degraded
        performance_trend = await self._get_performance_trend(room_id)
        if performance_trend and len(performance_trend) > 10:
            recent_avg = np.mean(performance_trend[-5:])
            older_avg = np.mean(performance_trend[-10:-5])
            if recent_avg < older_avg - 0.02:  # 2% degradation
                return True
        
        # Periodic optimization for all rooms
        if not last_optimization:
            return True
        
        return False
    
    async def _get_recent_performance(self, room_id: str) -> Optional[float]:
        """Get recent performance for room"""
        
        try:
            result = await self.timeseries_db.fetchval("""
                SELECT AVG(accuracy_score) 
                FROM prediction_results 
                WHERE room = $1 
                AND timestamp > NOW() - INTERVAL '24 hours'
                AND accuracy_score IS NOT NULL
            """, room_id)
            
            return float(result) if result else None
            
        except Exception as e:
            logger.error(f"Error getting recent performance for room {room_id}: {e}")
            return None
    
    async def _get_performance_trend(self, room_id: str) -> Optional[List[float]]:
        """Get performance trend for room"""
        
        try:
            results = await self.timeseries_db.fetch("""
                SELECT accuracy_score 
                FROM prediction_results 
                WHERE room = $1 
                AND timestamp > NOW() - INTERVAL '7 days'
                AND accuracy_score IS NOT NULL
                ORDER BY timestamp ASC
            """, room_id)
            
            if results:
                return [row['accuracy_score'] for row in results]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting performance trend for room {room_id}: {e}")
            return None
    
    async def _optimize_room_hyperparameters(self, room_id: str):
        """Optimize hyperparameters for a specific room"""
        
        try:
            logger.info(f"Starting hyperparameter optimization for room {room_id}")
            
            # Get current model configuration
            current_config = await self._get_current_model_config(room_id)
            if not current_config:
                logger.warning(f"No model configuration found for room {room_id}")
                return
            
            model_type = current_config.get('model_type', 'gradient_boost')
            
            # Get validation data
            validation_data = await self._get_validation_data(room_id)
            if not validation_data:
                logger.warning(f"No validation data for room {room_id}")
                return
            
            # Create optimization study
            study_name = f"room_{room_id}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            study = optuna.create_study(
                direction='maximize',
                sampler=self.sampler,
                pruner=self.pruner,
                study_name=study_name
            )
            
            # Store study reference
            self.room_optimizations[room_id]['current_study'] = study
            
            # Create objective function
            objective = ModelObjective(room_id, model_type, None, validation_data)
            
            # Run optimization
            optimization_start = datetime.now()
            
            study.optimize(
                objective,
                n_trials=self.max_trials_per_optimization,
                timeout=3600,  # 1 hour timeout
                callbacks=[self._trial_callback]
            )
            
            optimization_end = datetime.now()
            optimization_time = (optimization_end - optimization_start).total_seconds()
            
            # Get best hyperparameters
            best_params = study.best_params
            best_value = study.best_value
            
            # Calculate improvement
            current_performance = await self._get_recent_performance(room_id)
            improvement = best_value - (current_performance or 0.0)
            
            # Create optimization result
            result = OptimizationResult(
                room_id=room_id,
                model_type=model_type,
                best_hyperparameters=best_params,
                best_objective_value=best_value,
                total_trials=len(study.trials),
                successful_trials=len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                optimization_time=optimization_time,
                improvement=improvement,
                timestamp=datetime.now()
            )
            
            # Store result
            self.optimization_history.append(result)
            
            # Update room optimization tracking
            self.room_optimizations[room_id]['total_trials'] += result.total_trials
            self.room_optimizations[room_id]['successful_trials'] += result.successful_trials
            self.room_optimizations[room_id]['best_objective'] = max(
                self.room_optimizations[room_id]['best_objective'],
                best_value
            )
            self.room_optimizations[room_id]['last_optimization'] = datetime.now()
            
            # Apply best hyperparameters if improvement is significant
            if improvement > self.min_improvement_threshold:
                await self._apply_best_hyperparameters(room_id, model_type, best_params)
                logger.info(f"Applied optimized hyperparameters for room {room_id}: {improvement:.3f} improvement")
            else:
                logger.info(f"No significant improvement for room {room_id}: {improvement:.3f}")
            
            logger.info(f"Hyperparameter optimization completed for room {room_id}")
            
        except Exception as e:
            logger.error(f"Error optimizing hyperparameters for room {room_id}: {e}")
    
    async def _get_current_model_config(self, room_id: str) -> Optional[Dict[str, Any]]:
        """Get current model configuration"""
        
        try:
            # Get latest model from model store
            latest_model = await self.model_store.get_latest_model(room_id)
            
            if latest_model:
                return {
                    'model_type': latest_model.metadata.model_type,
                    'training_config': latest_model.training_config,
                    'feature_schema': latest_model.feature_schema,
                    'performance_metrics': latest_model.metadata.performance_metrics
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting current model config for room {room_id}: {e}")
            return None
    
    async def _get_validation_data(self, room_id: str) -> Optional[Dict[str, Any]]:
        """Get validation data for optimization"""
        
        try:
            # Get recent sensor events for validation
            events = await self.timeseries_db.fetch("""
                SELECT timestamp, entity_id, state, attributes
                FROM sensor_events
                WHERE room = $1
                AND timestamp > NOW() - INTERVAL '7 days'
                ORDER BY timestamp DESC
                LIMIT 1000
            """, room_id)
            
            if not events:
                return None
            
            # Convert to validation format
            validation_data = []
            for event in events:
                validation_data.append({
                    'timestamp': event['timestamp'],
                    'entity_id': event['entity_id'],
                    'state': event['state'],
                    'attributes': event['attributes']
                })
            
            return {
                'room_id': room_id,
                'events': validation_data,
                'count': len(validation_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting validation data for room {room_id}: {e}")
            return None
    
    def _trial_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Callback function for optimization trials"""
        
        # Log trial completion
        if trial.state == optuna.trial.TrialState.COMPLETE:
            logger.debug(f"Trial {trial.number} completed: {trial.value}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            logger.debug(f"Trial {trial.number} pruned")
        elif trial.state == optuna.trial.TrialState.FAIL:
            logger.warning(f"Trial {trial.number} failed")
    
    async def _apply_best_hyperparameters(self, room_id: str, model_type: str, best_params: Dict[str, Any]):
        """Apply the best hyperparameters to the model"""
        
        try:
            # This would integrate with the actual model training system
            # For now, we'll just log the application
            
            logger.info(f"Applying best hyperparameters for room {room_id}:")
            logger.info(f"Model type: {model_type}")
            logger.info(f"Parameters: {json.dumps(best_params, indent=2)}")
            
            # Here you would:
            # 1. Update model configuration
            # 2. Retrain model with new hyperparameters
            # 3. Validate performance
            # 4. Deploy new model if successful
            
        except Exception as e:
            logger.error(f"Error applying best hyperparameters for room {room_id}: {e}")
    
    async def manual_optimization(self, room_id: str, model_type: str = None, 
                                n_trials: int = None) -> Optional[OptimizationResult]:
        """Manually trigger hyperparameter optimization"""
        
        try:
            logger.info(f"Manual optimization triggered for room {room_id}")
            
            # Override configuration if provided
            if n_trials is None:
                n_trials = self.max_trials_per_optimization
            
            # Force optimization regardless of normal conditions
            await self._optimize_room_hyperparameters(room_id)
            
            # Return latest optimization result
            if self.optimization_history:
                latest_result = self.optimization_history[-1]
                if latest_result.room_id == room_id:
                    return latest_result
            
            return None
            
        except Exception as e:
            logger.error(f"Error in manual optimization for room {room_id}: {e}")
            return None
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        
        total_optimizations = len(self.optimization_history)
        successful_optimizations = sum(1 for result in self.optimization_history if result.improvement > 0)
        
        recent_optimizations = [
            result for result in self.optimization_history
            if result.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            'total_optimizations': total_optimizations,
            'successful_optimizations': successful_optimizations,
            'success_rate': successful_optimizations / total_optimizations if total_optimizations > 0 else 0,
            'recent_optimizations_24h': len(recent_optimizations),
            'average_improvement': np.mean([result.improvement for result in self.optimization_history]) if self.optimization_history else 0,
            'room_statistics': {
                room_id: {
                    'total_trials': stats['total_trials'],
                    'successful_trials': stats['successful_trials'],
                    'success_rate': stats['successful_trials'] / stats['total_trials'] if stats['total_trials'] > 0 else 0,
                    'best_objective': stats['best_objective'],
                    'last_optimization': stats['last_optimization'].isoformat() if stats['last_optimization'] else None
                }
                for room_id, stats in self.room_optimizations.items()
            }
        }
    
    def get_room_optimization_info(self, room_id: str) -> Dict[str, Any]:
        """Get optimization information for a specific room"""
        
        room_stats = self.room_optimizations[room_id]
        
        room_results = [
            result for result in self.optimization_history
            if result.room_id == room_id
        ]
        
        return {
            'room_id': room_id,
            'total_trials': room_stats['total_trials'],
            'successful_trials': room_stats['successful_trials'],
            'success_rate': room_stats['successful_trials'] / room_stats['total_trials'] if room_stats['total_trials'] > 0 else 0,
            'best_objective': room_stats['best_objective'],
            'last_optimization': room_stats['last_optimization'].isoformat() if room_stats['last_optimization'] else None,
            'optimization_count': len(room_results),
            'recent_results': [asdict(result) for result in room_results[-5:]]  # Last 5 results
        }
    
    async def health_check(self) -> str:
        """Health check for hyperparameter optimization system"""
        
        try:
            if not self.optimization_enabled:
                return "disabled"
            
            # Check if optimization is working
            recent_optimizations = [
                result for result in self.optimization_history
                if result.timestamp > datetime.now() - timedelta(hours=24)
            ]
            
            if not recent_optimizations and len(self.room_optimizations) > 0:
                return "stale"
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization health check failed: {e}")
            return "error"
    
    async def shutdown(self):
        """Shutdown hyperparameter optimization system"""
        
        self.optimization_enabled = False
        
        # Cancel any running studies
        for room_id, room_data in self.room_optimizations.items():
            if room_data['current_study']:
                # Optuna studies don't need explicit cancellation
                room_data['current_study'] = None
        
        logger.info("Hyperparameter optimization system shut down")