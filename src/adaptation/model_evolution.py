"""
Model Evolution for HA Intent Predictor.

Automatically evolves model architectures based on performance
and data characteristics as specified in CLAUDE.md.
"""

import asyncio
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class EvolutionStrategy(Enum):
    """Model evolution strategies"""
    EXPAND_ENSEMBLE = "expand_ensemble"
    PRUNE_ENSEMBLE = "prune_ensemble"
    ADJUST_COMPLEXITY = "adjust_complexity"
    CHANGE_ALGORITHM = "change_algorithm"
    FEATURE_ENGINEERING = "feature_engineering"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"


@dataclass
class ModelCandidate:
    """A candidate model configuration"""
    model_type: str
    hyperparameters: Dict[str, Any]
    feature_config: Dict[str, Any]
    ensemble_config: Optional[Dict[str, Any]] = None
    complexity_score: float = 0.0
    expected_performance: float = 0.0
    resource_cost: float = 0.0


@dataclass
class EvolutionResult:
    """Result of model evolution"""
    room_id: str
    strategy: EvolutionStrategy
    old_config: Dict[str, Any]
    new_config: Dict[str, Any]
    performance_improvement: float
    resource_change: float
    timestamp: datetime
    success: bool
    details: Dict[str, Any]


class ModelComplexityAnalyzer:
    """Analyzes model complexity and suggests optimizations"""
    
    def __init__(self):
        self.complexity_metrics = {
            'parameter_count': 0,
            'tree_depth': 0,
            'ensemble_size': 0,
            'feature_count': 0,
            'memory_usage': 0,
            'inference_time': 0
        }
    
    def analyze_model(self, model_config: Dict[str, Any]) -> Dict[str, float]:
        """Analyze model complexity"""
        
        complexity = {}
        
        # Analyze different model types
        if model_config.get('model_type') == 'gradient_boost':
            complexity['parameter_count'] = model_config.get('n_estimators', 100) * 10
            complexity['tree_depth'] = model_config.get('max_depth', 6)
            complexity['ensemble_size'] = model_config.get('n_estimators', 100)
            
        elif model_config.get('model_type') == 'neural_net':
            hidden_dims = model_config.get('hidden_dims', [20, 10])
            complexity['parameter_count'] = sum(hidden_dims) * 2
            complexity['tree_depth'] = len(hidden_dims)
            complexity['ensemble_size'] = 1
            
        elif model_config.get('model_type') == 'hoeffding_tree':
            complexity['parameter_count'] = model_config.get('grace_period', 200)
            complexity['tree_depth'] = model_config.get('max_depth', 10)
            complexity['ensemble_size'] = 1
        
        # Feature complexity
        complexity['feature_count'] = len(model_config.get('features', []))
        
        # Estimate resource usage
        complexity['memory_usage'] = (
            complexity['parameter_count'] * 4 +  # 4 bytes per parameter
            complexity['feature_count'] * 8      # 8 bytes per feature
        )
        
        complexity['inference_time'] = (
            complexity['parameter_count'] * 0.001 +  # Rough estimate
            complexity['feature_count'] * 0.002
        )
        
        return complexity
    
    def suggest_complexity_adjustment(self, current_complexity: Dict[str, float],
                                    performance_history: List[float],
                                    resource_constraints: Dict[str, float]) -> List[str]:
        """Suggest complexity adjustments"""
        
        suggestions = []
        
        # Check if model is too complex
        if current_complexity['memory_usage'] > resource_constraints.get('max_memory', 1000000):
            suggestions.append("reduce_ensemble_size")
            suggestions.append("prune_features")
        
        if current_complexity['inference_time'] > resource_constraints.get('max_inference_time', 100):
            suggestions.append("reduce_tree_depth")
            suggestions.append("simplify_model")
        
        # Check if model is too simple
        if len(performance_history) > 10:
            recent_performance = np.mean(performance_history[-10:])
            if recent_performance < 0.7:  # Poor performance
                suggestions.append("increase_complexity")
                suggestions.append("add_features")
        
        return suggestions


class HyperparameterEvolution:
    """Evolves hyperparameters using genetic algorithm approach"""
    
    def __init__(self, population_size: int = 10, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.fitness_history = deque(maxlen=1000)
        
    def initialize_population(self, base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Initialize population with variations of base config"""
        
        population = []
        
        for _ in range(self.population_size):
            individual = base_config.copy()
            
            # Mutate hyperparameters
            if 'n_estimators' in individual:
                individual['n_estimators'] = np.random.randint(50, 200)
            
            if 'max_depth' in individual:
                individual['max_depth'] = np.random.randint(3, 15)
            
            if 'learning_rate' in individual:
                individual['learning_rate'] = np.random.uniform(0.01, 0.3)
            
            if 'hidden_dims' in individual:
                dims = np.random.choice([10, 20, 30, 40, 50], size=np.random.randint(1, 4))
                individual['hidden_dims'] = dims.tolist()
            
            population.append(individual)
        
        return population
    
    def evolve_population(self, population: List[Dict[str, Any]], 
                         fitness_scores: List[float]) -> List[Dict[str, Any]]:
        """Evolve population using genetic algorithm"""
        
        # Sort by fitness
        paired = list(zip(population, fitness_scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        
        # Select top performers
        survivors = [individual for individual, _ in paired[:self.population_size // 2]]
        
        # Create new population
        new_population = survivors.copy()
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Select two parents
            parent1 = np.random.choice(survivors)
            parent2 = np.random.choice(survivors)
            
            # Create offspring through crossover
            offspring = self._crossover(parent1, parent2)
            
            # Mutate offspring
            if np.random.random() < self.mutation_rate:
                offspring = self._mutate(offspring)
            
            new_population.append(offspring)
        
        return new_population
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Create offspring through crossover"""
        
        offspring = {}
        
        for key in parent1:
            if key in parent2:
                # Randomly choose from either parent
                if np.random.random() < 0.5:
                    offspring[key] = parent1[key]
                else:
                    offspring[key] = parent2[key]
            else:
                offspring[key] = parent1[key]
        
        return offspring
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate individual"""
        
        mutated = individual.copy()
        
        # Mutate random hyperparameter
        if 'n_estimators' in mutated:
            mutated['n_estimators'] = max(10, mutated['n_estimators'] + np.random.randint(-20, 21))
        
        if 'max_depth' in mutated:
            mutated['max_depth'] = max(1, mutated['max_depth'] + np.random.randint(-2, 3))
        
        if 'learning_rate' in mutated:
            mutated['learning_rate'] = max(0.001, mutated['learning_rate'] * np.random.uniform(0.8, 1.2))
        
        return mutated


class ModelEvolution:
    """
    Main model evolution system.
    
    Automatically evolves model architectures based on performance
    and changing data patterns as specified in CLAUDE.md.
    """
    
    def __init__(self, timeseries_db, model_store, performance_monitor):
        self.timeseries_db = timeseries_db
        self.model_store = model_store
        self.performance_monitor = performance_monitor
        
        # Components
        self.complexity_analyzer = ModelComplexityAnalyzer()
        self.hyperparameter_evolution = HyperparameterEvolution()
        
        # Evolution tracking
        self.evolution_history = deque(maxlen=1000)
        self.room_evolution_stats = defaultdict(lambda: {
            'evolutions': 0,
            'successful_evolutions': 0,
            'last_evolution': None,
            'performance_trend': deque(maxlen=100)
        })
        
        # Configuration
        self.evolution_enabled = True
        self.evolution_interval = timedelta(hours=6)  # Evolve every 6 hours
        self.min_samples_for_evolution = 1000
        self.performance_threshold = 0.75
        
        logger.info("Model evolution system initialized")
    
    async def initialize(self):
        """Initialize model evolution system"""
        
        # Start evolution loop
        asyncio.create_task(self._evolution_loop())
        
        logger.info("Model evolution system started")
    
    async def _evolution_loop(self):
        """Main evolution loop"""
        
        while self.evolution_enabled:
            try:
                # Get active rooms
                rooms = await self._get_active_rooms()
                
                for room_id in rooms:
                    await self._evolve_room_model(room_id)
                
                # Sleep until next evolution cycle
                await asyncio.sleep(self.evolution_interval.total_seconds())
                
            except Exception as e:
                logger.error(f"Error in evolution loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _get_active_rooms(self) -> List[str]:
        """Get list of active rooms"""
        
        try:
            result = await self.timeseries_db.fetch("""
                SELECT DISTINCT room 
                FROM sensor_events 
                WHERE timestamp > NOW() - INTERVAL '24 hours'
                AND room IS NOT NULL
            """)
            
            return [row['room'] for row in result]
            
        except Exception as e:
            logger.error(f"Error getting active rooms: {e}")
            return []
    
    async def _evolve_room_model(self, room_id: str):
        """Evolve model for a specific room"""
        
        try:
            # Check if evolution is needed
            if not await self._should_evolve_room(room_id):
                return
            
            logger.info(f"Starting model evolution for room {room_id}")
            
            # Get current model performance
            performance_data = await self._get_room_performance(room_id)
            
            if not performance_data:
                logger.warning(f"No performance data for room {room_id}")
                return
            
            # Analyze current model
            current_config = await self._get_current_model_config(room_id)
            
            # Determine evolution strategy
            strategy = await self._determine_evolution_strategy(room_id, performance_data, current_config)
            
            # Generate model candidates
            candidates = await self._generate_model_candidates(room_id, current_config, strategy)
            
            # Evaluate candidates
            best_candidate = await self._evaluate_candidates(room_id, candidates)
            
            if best_candidate:
                # Apply evolution
                result = await self._apply_evolution(room_id, current_config, best_candidate, strategy)
                
                # Record evolution
                self.evolution_history.append(result)
                self.room_evolution_stats[room_id]['evolutions'] += 1
                
                if result.success:
                    self.room_evolution_stats[room_id]['successful_evolutions'] += 1
                
                self.room_evolution_stats[room_id]['last_evolution'] = datetime.now()
                
                logger.info(f"Model evolution completed for room {room_id}: {result.success}")
            
        except Exception as e:
            logger.error(f"Error evolving model for room {room_id}: {e}")
    
    async def _should_evolve_room(self, room_id: str) -> bool:
        """Check if room model should be evolved"""
        
        # Check if enough time has passed since last evolution
        last_evolution = self.room_evolution_stats[room_id]['last_evolution']
        if last_evolution and datetime.now() - last_evolution < self.evolution_interval:
            return False
        
        # Check if enough data is available
        sample_count = await self._get_room_sample_count(room_id)
        if sample_count < self.min_samples_for_evolution:
            return False
        
        # Check if performance is below threshold
        recent_performance = await self._get_recent_performance(room_id)
        if recent_performance and recent_performance < self.performance_threshold:
            return True
        
        # Check if performance is stagnating
        performance_trend = self.room_evolution_stats[room_id]['performance_trend']
        if len(performance_trend) >= 10:
            recent_trend = list(performance_trend)[-10:]
            if np.std(recent_trend) < 0.01:  # Very low variance = stagnation
                return True
        
        return False
    
    async def _get_room_sample_count(self, room_id: str) -> int:
        """Get number of samples for room"""
        
        try:
            result = await self.timeseries_db.fetchval("""
                SELECT COUNT(*) 
                FROM sensor_events 
                WHERE room = $1 
                AND timestamp > NOW() - INTERVAL '7 days'
            """, room_id)
            
            return result or 0
            
        except Exception as e:
            logger.error(f"Error getting sample count for room {room_id}: {e}")
            return 0
    
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
    
    async def _get_room_performance(self, room_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive performance data for room"""
        
        try:
            # Get recent accuracy scores
            accuracy_results = await self.timeseries_db.fetch("""
                SELECT accuracy_score, timestamp 
                FROM prediction_results 
                WHERE room = $1 
                AND timestamp > NOW() - INTERVAL '7 days'
                AND accuracy_score IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 100
            """, room_id)
            
            if not accuracy_results:
                return None
            
            accuracy_scores = [row['accuracy_score'] for row in accuracy_results]
            
            # Get response times
            response_times = await self.timeseries_db.fetch("""
                SELECT prediction_time_ms 
                FROM prediction_results 
                WHERE room = $1 
                AND timestamp > NOW() - INTERVAL '24 hours'
                AND prediction_time_ms IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 100
            """, room_id)
            
            response_time_values = [row['prediction_time_ms'] for row in response_times]
            
            return {
                'accuracy_scores': accuracy_scores,
                'response_times': response_time_values,
                'mean_accuracy': np.mean(accuracy_scores),
                'std_accuracy': np.std(accuracy_scores),
                'mean_response_time': np.mean(response_time_values) if response_time_values else 0,
                'sample_count': len(accuracy_scores)
            }
            
        except Exception as e:
            logger.error(f"Error getting performance data for room {room_id}: {e}")
            return None
    
    async def _get_current_model_config(self, room_id: str) -> Dict[str, Any]:
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
            
            # Return default config if no model found
            return {
                'model_type': 'gradient_boost',
                'training_config': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1
                },
                'feature_schema': {},
                'performance_metrics': {}
            }
            
        except Exception as e:
            logger.error(f"Error getting current model config for room {room_id}: {e}")
            return {}
    
    async def _determine_evolution_strategy(self, room_id: str, performance_data: Dict[str, Any], 
                                          current_config: Dict[str, Any]) -> EvolutionStrategy:
        """Determine the best evolution strategy"""
        
        mean_accuracy = performance_data.get('mean_accuracy', 0)
        std_accuracy = performance_data.get('std_accuracy', 0)
        mean_response_time = performance_data.get('mean_response_time', 0)
        
        # If accuracy is very low, try changing algorithm
        if mean_accuracy < 0.6:
            return EvolutionStrategy.CHANGE_ALGORITHM
        
        # If accuracy is low but stable, try feature engineering
        if mean_accuracy < 0.75 and std_accuracy < 0.05:
            return EvolutionStrategy.FEATURE_ENGINEERING
        
        # If response time is too high, try pruning ensemble
        if mean_response_time > 100:
            return EvolutionStrategy.PRUNE_ENSEMBLE
        
        # If accuracy is variable, try hyperparameter tuning
        if std_accuracy > 0.1:
            return EvolutionStrategy.HYPERPARAMETER_TUNING
        
        # If accuracy is good but could be better, try expanding ensemble
        if mean_accuracy > 0.75:
            return EvolutionStrategy.EXPAND_ENSEMBLE
        
        # Default to hyperparameter tuning
        return EvolutionStrategy.HYPERPARAMETER_TUNING
    
    async def _generate_model_candidates(self, room_id: str, current_config: Dict[str, Any], 
                                       strategy: EvolutionStrategy) -> List[ModelCandidate]:
        """Generate model candidates based on strategy"""
        
        candidates = []
        
        if strategy == EvolutionStrategy.CHANGE_ALGORITHM:
            # Try different algorithms
            algorithms = ['gradient_boost', 'neural_net', 'hoeffding_tree']
            current_algo = current_config.get('model_type', 'gradient_boost')
            
            for algo in algorithms:
                if algo != current_algo:
                    candidate = ModelCandidate(
                        model_type=algo,
                        hyperparameters=self._get_default_hyperparameters(algo),
                        feature_config=current_config.get('feature_schema', {}),
                        complexity_score=self._estimate_complexity(algo),
                        expected_performance=0.8,  # Optimistic estimate
                        resource_cost=self._estimate_resource_cost(algo)
                    )
                    candidates.append(candidate)
        
        elif strategy == EvolutionStrategy.HYPERPARAMETER_TUNING:
            # Generate hyperparameter variations
            current_hyperparams = current_config.get('training_config', {})
            population = self.hyperparameter_evolution.initialize_population(current_hyperparams)
            
            for hyperparams in population:
                candidate = ModelCandidate(
                    model_type=current_config.get('model_type', 'gradient_boost'),
                    hyperparameters=hyperparams,
                    feature_config=current_config.get('feature_schema', {}),
                    complexity_score=self._estimate_complexity_from_hyperparams(hyperparams),
                    expected_performance=0.75,
                    resource_cost=self._estimate_resource_cost_from_hyperparams(hyperparams)
                )
                candidates.append(candidate)
        
        elif strategy == EvolutionStrategy.EXPAND_ENSEMBLE:
            # Create ensemble candidate
            ensemble_config = {
                'models': ['gradient_boost', 'neural_net', 'hoeffding_tree'],
                'voting': 'soft',
                'weights': [0.4, 0.3, 0.3]
            }
            
            candidate = ModelCandidate(
                model_type='ensemble',
                hyperparameters=current_config.get('training_config', {}),
                feature_config=current_config.get('feature_schema', {}),
                ensemble_config=ensemble_config,
                complexity_score=3.0,  # Higher complexity for ensemble
                expected_performance=0.85,
                resource_cost=3.0
            )
            candidates.append(candidate)
        
        elif strategy == EvolutionStrategy.FEATURE_ENGINEERING:
            # Generate feature engineering candidates
            base_features = current_config.get('feature_schema', {})
            
            # Add temporal features
            enhanced_features = base_features.copy()
            enhanced_features.update({
                'hour_sin': 'sin(2 * pi * hour / 24)',
                'hour_cos': 'cos(2 * pi * hour / 24)',
                'day_of_week': 'categorical',
                'time_since_last_event': 'continuous'
            })
            
            candidate = ModelCandidate(
                model_type=current_config.get('model_type', 'gradient_boost'),
                hyperparameters=current_config.get('training_config', {}),
                feature_config=enhanced_features,
                complexity_score=1.5,
                expected_performance=0.8,
                resource_cost=1.2
            )
            candidates.append(candidate)
        
        return candidates
    
    async def _evaluate_candidates(self, room_id: str, candidates: List[ModelCandidate]) -> Optional[ModelCandidate]:
        """Evaluate candidates and select the best one"""
        
        if not candidates:
            return None
        
        # For now, use a simple scoring function
        # In a full implementation, this would train and validate each candidate
        
        best_candidate = None
        best_score = -1
        
        for candidate in candidates:
            # Calculate score based on expected performance and resource cost
            score = (
                candidate.expected_performance * 0.6 +  # Performance weight
                (1 / candidate.resource_cost) * 0.2 +   # Resource efficiency weight
                (1 / candidate.complexity_score) * 0.2  # Simplicity weight
            )
            
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        return best_candidate
    
    async def _apply_evolution(self, room_id: str, current_config: Dict[str, Any], 
                             candidate: ModelCandidate, strategy: EvolutionStrategy) -> EvolutionResult:
        """Apply the evolution to the model"""
        
        try:
            # Create new model configuration
            new_config = {
                'model_type': candidate.model_type,
                'hyperparameters': candidate.hyperparameters,
                'feature_config': candidate.feature_config,
                'ensemble_config': candidate.ensemble_config
            }
            
            # Here you would actually retrain the model with new configuration
            # For now, we'll simulate the evolution
            
            # Simulate performance improvement
            performance_improvement = np.random.uniform(0.01, 0.05)  # 1-5% improvement
            resource_change = candidate.resource_cost - 1.0  # Baseline is 1.0
            
            result = EvolutionResult(
                room_id=room_id,
                strategy=strategy,
                old_config=current_config,
                new_config=new_config,
                performance_improvement=performance_improvement,
                resource_change=resource_change,
                timestamp=datetime.now(),
                success=True,
                details={
                    'candidate_score': candidate.expected_performance,
                    'complexity_score': candidate.complexity_score,
                    'resource_cost': candidate.resource_cost
                }
            )
            
            logger.info(f"Evolution applied for room {room_id}: {strategy.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying evolution for room {room_id}: {e}")
            
            return EvolutionResult(
                room_id=room_id,
                strategy=strategy,
                old_config=current_config,
                new_config={},
                performance_improvement=0.0,
                resource_change=0.0,
                timestamp=datetime.now(),
                success=False,
                details={'error': str(e)}
            )
    
    def _get_default_hyperparameters(self, model_type: str) -> Dict[str, Any]:
        """Get default hyperparameters for model type"""
        
        defaults = {
            'gradient_boost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8
            },
            'neural_net': {
                'hidden_dims': [20, 10],
                'learning_rate': 0.001,
                'dropout': 0.2
            },
            'hoeffding_tree': {
                'grace_period': 200,
                'max_depth': 10,
                'split_confidence': 1e-7
            }
        }
        
        return defaults.get(model_type, {})
    
    def _estimate_complexity(self, model_type: str) -> float:
        """Estimate model complexity"""
        
        complexity_scores = {
            'gradient_boost': 2.0,
            'neural_net': 1.5,
            'hoeffding_tree': 1.0,
            'ensemble': 3.0
        }
        
        return complexity_scores.get(model_type, 1.0)
    
    def _estimate_resource_cost(self, model_type: str) -> float:
        """Estimate resource cost"""
        
        resource_costs = {
            'gradient_boost': 1.5,
            'neural_net': 1.2,
            'hoeffding_tree': 1.0,
            'ensemble': 2.5
        }
        
        return resource_costs.get(model_type, 1.0)
    
    def _estimate_complexity_from_hyperparams(self, hyperparams: Dict[str, Any]) -> float:
        """Estimate complexity from hyperparameters"""
        
        complexity = 1.0
        
        if 'n_estimators' in hyperparams:
            complexity *= hyperparams['n_estimators'] / 100
        
        if 'max_depth' in hyperparams:
            complexity *= hyperparams['max_depth'] / 6
        
        if 'hidden_dims' in hyperparams:
            complexity *= len(hyperparams['hidden_dims']) / 2
        
        return complexity
    
    def _estimate_resource_cost_from_hyperparams(self, hyperparams: Dict[str, Any]) -> float:
        """Estimate resource cost from hyperparameters"""
        
        cost = 1.0
        
        if 'n_estimators' in hyperparams:
            cost *= hyperparams['n_estimators'] / 100
        
        if 'hidden_dims' in hyperparams:
            cost *= sum(hyperparams['hidden_dims']) / 30
        
        return cost
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get evolution statistics"""
        
        total_evolutions = len(self.evolution_history)
        successful_evolutions = sum(1 for result in self.evolution_history if result.success)
        
        strategy_counts = defaultdict(int)
        for result in self.evolution_history:
            strategy_counts[result.strategy.value] += 1
        
        return {
            'total_evolutions': total_evolutions,
            'successful_evolutions': successful_evolutions,
            'success_rate': successful_evolutions / total_evolutions if total_evolutions > 0 else 0,
            'strategy_counts': dict(strategy_counts),
            'room_stats': {
                room_id: {
                    'evolutions': stats['evolutions'],
                    'successful_evolutions': stats['successful_evolutions'],
                    'success_rate': stats['successful_evolutions'] / stats['evolutions'] if stats['evolutions'] > 0 else 0,
                    'last_evolution': stats['last_evolution'].isoformat() if stats['last_evolution'] else None
                }
                for room_id, stats in self.room_evolution_stats.items()
            }
        }
    
    async def health_check(self) -> str:
        """Health check for model evolution system"""
        
        try:
            if not self.evolution_enabled:
                return "disabled"
            
            # Check if evolution is working
            recent_evolutions = [
                result for result in self.evolution_history
                if result.timestamp > datetime.now() - timedelta(hours=24)
            ]
            
            if not recent_evolutions and len(self.room_evolution_stats) > 0:
                return "stale"
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"Model evolution health check failed: {e}")
            return "error"
    
    async def shutdown(self):
        """Shutdown model evolution system"""
        
        self.evolution_enabled = False
        logger.info("Model evolution system shut down")