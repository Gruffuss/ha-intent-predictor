"""
Online Learning Models - Continuously updating models using River framework
Implements the adaptive prediction system from CLAUDE.md
"""

import logging
from collections import defaultdict
from typing import Dict, Any, Optional, List
from river import ensemble, tree, metrics, preprocessing, forest, utils
from river import linear_model
from river import stats

logger = logging.getLogger(__name__)


class ContinuousLearningModel:
    """Continuously updating occupancy prediction model for a specific room"""
    
    def __init__(self, room_id: str):
        self.room_id = room_id
        
        # Ensemble of online learners as specified in CLAUDE.md
        self.models = {
            'adaptive_random_forest': forest.ARFClassifier(
                n_models=10,
                max_features="sqrt",
                lambda_value=6,
                grace_period=10
            ),
            'hoeffding_tree': tree.HoeffdingTreeClassifier(
                grace_period=10,
                delta=1e-5,
                tau=0.05
            ),
            'logistic': linear_model.LogisticRegression(
                l2=0.1
            )
        }
        
        # Meta-learner decides which model to trust
        self.meta_learner = MetaLearner()
        
        # Track each model's performance
        self.model_performance = {
            name: utils.Rolling(metrics.Accuracy(), window_size=1000)
            for name in self.models
        }
        
        # Feature preprocessing
        self.preprocessor = preprocessing.StandardScaler()
        
        logger.info(f"Initialized continuous learning model for {room_id}")
    
    def learn_one(self, features: Dict[str, Any], y_true: bool):
        """
        Update all models with single observation
        Core of the adaptive learning system
        """
        # Preprocess features - handle potential None return from learn_one
        try:
            self.preprocessor.learn_one(features)
            processed_features = self.preprocessor.transform_one(features)
        except (AttributeError, TypeError) as e:
            logger.warning(f"Preprocessor error for {self.room_id}: {e}, using raw features")
            processed_features = features
        
        # Update each model
        for name, model in self.models.items():
            try:
                y_pred = model.predict_proba_one(processed_features)
                
                # Update performance tracking
                if y_pred is not None:
                    prob_true = y_pred.get(True, 0.0)
                    self.model_performance[name].update(y_true, prob_true)
                
                # Learn from this observation
                model.learn_one(processed_features, y_true)
                
            except Exception as e:
                logger.warning(f"Error updating model {name} for {self.room_id}: {e}")
        
        # Update meta-learner
        predictions = self.get_model_predictions(processed_features)
        self.meta_learner.update(predictions, y_true)
    
    def predict_proba_one(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensemble prediction with uncertainty quantification
        """
        try:
            # Preprocess features - handle potential preprocessor issues
            try:
                processed_features = self.preprocessor.transform_one(features)
            except (AttributeError, TypeError) as e:
                logger.warning(f"Preprocessor error in prediction for {self.room_id}: {e}, using raw features")
                processed_features = features
            
            # Get predictions from all models
            predictions = self.get_model_predictions(processed_features)
            
            # Meta-learner weights each model based on performance
            weights = self.meta_learner.get_weights(predictions, self.model_performance)
            
            # Weighted ensemble prediction
            final_pred = sum(
                pred * weight 
                for pred, weight in zip(predictions.values(), weights.values())
            )
            
            # Calculate uncertainty
            uncertainty = self.calculate_prediction_uncertainty(predictions, weights)
            
            return {
                'probability': final_pred,
                'uncertainty': uncertainty,
                'confidence': 1.0 - uncertainty,
                'contributing_models': self.get_contribution_explanation(predictions, weights),
                'model_agreement': self.calculate_model_agreement(predictions)
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {self.room_id}: {e}")
            return {
                'probability': 0.5,
                'uncertainty': 1.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def get_model_predictions(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Get predictions from all models"""
        predictions = {}
        
        for name, model in self.models.items():
            try:
                # Different models have different prediction interfaces
                if hasattr(model, 'predict_proba_one'):
                    pred = model.predict_proba_one(features)
                    if pred is not None:
                        # Handle both dict and float returns
                        if isinstance(pred, dict):
                            predictions[name] = pred.get(True, pred.get(1, 0.5))
                        else:
                            predictions[name] = float(pred)
                    else:
                        predictions[name] = 0.5  # Neutral prediction
                elif hasattr(model, 'predict_one'):
                    # For models that only have predict_one
                    pred = model.predict_one(features)
                    predictions[name] = float(pred) if pred is not None else 0.5
                else:
                    logger.warning(f"Model {name} has no prediction method")
                    predictions[name] = 0.5
            except Exception as e:
                logger.debug(f"Error getting prediction from {name}: {e}")
                predictions[name] = 0.5
        
        return predictions
    
    def calculate_prediction_uncertainty(self, predictions: Dict[str, float], weights: Dict[str, float]) -> float:
        """
        Calculate prediction uncertainty based on model disagreement and individual uncertainties
        """
        if not predictions:
            return 1.0
        
        # Measure of model disagreement
        pred_values = list(predictions.values())
        disagreement = max(pred_values) - min(pred_values)
        
        # Weight by model performance
        weighted_disagreement = sum(
            abs(pred - 0.5) * weights.get(name, 1.0) 
            for name, pred in predictions.items()
        ) / len(predictions)
        
        # Combine disagreement and distance from neutral (0.5)
        uncertainty = disagreement * 0.7 + (1.0 - weighted_disagreement) * 0.3
        
        return min(uncertainty, 1.0)
    
    def calculate_model_agreement(self, predictions: Dict[str, float]) -> float:
        """Calculate how much models agree (0=total disagreement, 1=perfect agreement)"""
        if len(predictions) < 2:
            return 1.0
        
        pred_values = list(predictions.values())
        variance = sum((p - sum(pred_values)/len(pred_values))**2 for p in pred_values) / len(pred_values)
        
        # Convert variance to agreement score
        return max(0.0, 1.0 - variance * 4)  # Scale variance to [0,1]
    
    def get_contribution_explanation(self, predictions: Dict[str, float], weights: Dict[str, float]) -> Dict[str, Dict]:
        """Explain which models contributed most to the prediction"""
        contributions = {}
        
        for name, pred in predictions.items():
            weight = weights.get(name, 0.0)
            contribution = pred * weight
            
            contributions[name] = {
                'prediction': pred,
                'weight': weight,
                'contribution': contribution,
                'performance': self.model_performance[name].get() if hasattr(self.model_performance[name], 'n') and self.model_performance[name].n > 0 else 0.0
            }
        
        return contributions
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring"""
        summary = {
            'room_id': self.room_id,
            'models': {}
        }
        
        for name, perf_metric in self.model_performance.items():
            if hasattr(perf_metric, 'n') and perf_metric.n > 0:
                summary['models'][name] = {
                    'auc': perf_metric.get(),
                    'sample_count': perf_metric.n
                }
            else:
                summary['models'][name] = {
                    'auc': 0.0,
                    'sample_count': 0
                }
        
        return summary
    
    def save_model(self) -> Dict[str, Any]:
        """
        Save model state for persistence
        River models can be pickled directly
        """
        try:
            import pickle
            
            # Serialize the models
            serialized_models = {}
            for name, model in self.models.items():
                try:
                    serialized_models[name] = pickle.dumps(model)
                except Exception as e:
                    logger.warning(f"Could not serialize model {name}: {e}")
            
            # Save other components
            model_state = {
                'room_id': self.room_id,
                'serialized_models': serialized_models,
                'model_performance': {
                    name: {
                        'n_samples': metric.n if hasattr(metric, 'n') else 0,
                        'value': metric.get() if hasattr(metric, 'n') and metric.n > 0 else 0.0
                    }
                    for name, metric in self.model_performance.items()
                },
                'meta_learner_state': self.meta_learner.get_state(),
                'preprocessor_state': pickle.dumps(self.preprocessor)
            }
            
            return model_state
            
        except Exception as e:
            logger.error(f"Error saving model for {self.room_id}: {e}")
            return {}
    
    def load_model(self, model_state: Dict[str, Any]):
        """
        Load model state from persistence
        """
        try:
            import pickle
            
            # Restore room_id
            self.room_id = model_state.get('room_id', self.room_id)
            
            # Restore serialized models
            serialized_models = model_state.get('serialized_models', {})
            for name, serialized_model in serialized_models.items():
                try:
                    self.models[name] = pickle.loads(serialized_model)
                except Exception as e:
                    logger.warning(f"Could not deserialize model {name}: {e}")
            
            # Restore performance metrics
            performance_data = model_state.get('model_performance', {})
            for name, perf_data in performance_data.items():
                if name in self.model_performance:
                    # Reconstruct metric with saved data
                    metric = self.model_performance[name]
                    # Note: River metrics don't have easy restoration, so we'll reinitialize
                    metric._window = []
                    metric.n = perf_data.get('n_samples', 0)
            
            # Restore meta-learner
            meta_learner_state = model_state.get('meta_learner_state', {})
            if meta_learner_state:
                self.meta_learner.load_state(meta_learner_state)
            
            # Restore preprocessor
            preprocessor_state = model_state.get('preprocessor_state')
            if preprocessor_state:
                self.preprocessor = pickle.loads(preprocessor_state)
            
            logger.info(f"Successfully loaded model for {self.room_id}")
            
        except Exception as e:
            logger.error(f"Error loading model for {self.room_id}: {e}")
            # If loading fails, keep the initialized models
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of model state"""
        return {
            'room_id': self.room_id,
            'models': list(self.models.keys()),
            'performance': {
                name: {
                    'samples': metric.n,
                    'score': metric.get() if metric.n > 0 else 0.0
                }
                for name, metric in self.model_performance.items()
            },
            'meta_learner_weights': self.meta_learner.get_current_weights(),
            'preprocessor_features': getattr(self.preprocessor, 'n_features_in_', 0)
        }


class MetaLearner:
    """
    Learns which models work best in different situations
    Dynamically weights ensemble members
    """
    
    def __init__(self):
        self.model_scores = defaultdict(lambda: stats.Mean())
        self.context_performance = defaultdict(lambda: defaultdict(lambda: stats.Mean()))
        
    def update(self, predictions: Dict[str, float], y_true: bool):
        """Update meta-learning with prediction results"""
        for model_name, prediction in predictions.items():
            # Update overall performance
            error = abs(prediction - float(y_true))
            self.model_scores[model_name].update(1.0 - error)  # Convert error to score
    
    def get_weights(self, predictions: Dict[str, float], performance_metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Get dynamic weights for ensemble based on current context and historical performance
        """
        weights = {}
        total_weight = 0.0
        
        for model_name in predictions.keys():
            # Base weight from historical performance
            try:
                # Check if we have enough samples for reliable score
                if hasattr(self.model_scores[model_name], 'n') and self.model_scores[model_name].n > 10:
                    base_score = self.model_scores[model_name].get()
                else:
                    base_score = 0.5  # Neutral weight for new models
            except (AttributeError, ValueError):
                base_score = 0.5  # Fallback for any metric access issues
            
            # Adjust based on recent performance
            recent_perf = performance_metrics.get(model_name)
            if recent_perf and hasattr(recent_perf, 'get'):
                try:
                    recent_score = recent_perf.get()
                    # Only use if we got a valid score (not None/NaN)
                    if recent_score is not None and not (isinstance(recent_score, float) and recent_score != recent_score):
                        # Combine base and recent performance
                        final_score = base_score * 0.7 + recent_score * 0.3
                    else:
                        final_score = base_score
                except (AttributeError, ValueError, TypeError):
                    # Rolling metric might not be ready yet
                    final_score = base_score
            else:
                final_score = base_score
            
            # Ensure positive weights
            weights[model_name] = max(0.1, final_score)
            total_weight += weights[model_name]
        
        # Normalize weights to sum to 1
        if total_weight > 0:
            for model_name in weights:
                weights[model_name] /= total_weight
        else:
            # Equal weights fallback
            equal_weight = 1.0 / len(predictions) if predictions else 1.0
            weights = {name: equal_weight for name in predictions.keys()}
        
        return weights
    
    def get_state(self) -> Dict[str, Any]:
        """Get serializable state for persistence"""
        return {
            'model_scores': {
                name: {'n': score.n, 'value': score.get() if score.n > 0 else 0.0}
                for name, score in self.model_scores.items()
            },
            'context_performance': {
                context: {
                    model: {'n': score.n, 'value': score.get() if score.n > 0 else 0.0}
                    for model, score in models.items()
                }
                for context, models in self.context_performance.items()
            }
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load state from persistence"""
        try:
            # Restore model scores
            model_scores_data = state.get('model_scores', {})
            for name, score_data in model_scores_data.items():
                # Reinitialize the stats object
                self.model_scores[name] = stats.Mean()
                # Note: River stats don't have easy restoration, so we'll start fresh
                
            # Restore context performance
            context_data = state.get('context_performance', {})
            for context, models in context_data.items():
                for model, score_data in models.items():
                    self.context_performance[context][model] = stats.Mean()
                    
        except Exception as e:
            logger.error(f"Error loading meta-learner state: {e}")
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current model weights for monitoring"""
        weights = {}
        for name, score in self.model_scores.items():
            if score.n > 0:
                weights[name] = score.get()
            else:
                weights[name] = 0.5
        return weights


class MultiHorizonPredictor:
    """
    Manages multiple prediction horizons for a set of rooms
    Implements the adaptive horizon optimization from CLAUDE.md
    """
    
    def __init__(self, room_ids: list):
        self.room_ids = room_ids
        self.horizon_optimizer = HorizonOptimizer()
        
        # Start with standard horizons, will be optimized
        self.horizons = [15, 120]  # 15 min for AC, 2 hours for heating
        
        # Model for each room and horizon combination
        self.predictors = {}
        for room_id in room_ids:
            self.predictors[room_id] = {}
            for horizon in self.horizons:
                self.predictors[room_id][horizon] = ContinuousLearningModel(f"{room_id}_{horizon}min")
        
        logger.info(f"Initialized multi-horizon predictor for {len(room_ids)} rooms, {len(self.horizons)} horizons")
    
    def learn_one(self, room_id: str, features: Dict[str, Any], current_occupancy: Dict[int, bool]):
        """
        Update all horizon models for a room with current observation
        
        Args:
            room_id: Room identifier
            features: Current feature vector
            current_occupancy: Dict mapping horizon (minutes) to actual occupancy
        """
        if room_id not in self.predictors:
            logger.warning(f"Unknown room: {room_id}")
            return
        
        for horizon in self.horizons:
            if horizon in current_occupancy:
                y_true = current_occupancy[horizon]
                self.predictors[room_id][horizon].learn_one(features, y_true)
    
    def predict_all_horizons(self, room_id: str, features: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """Get predictions for all horizons for a room"""
        if room_id not in self.predictors:
            return {}
        
        predictions = {}
        for horizon in self.horizons:
            predictions[horizon] = self.predictors[room_id][horizon].predict_proba_one(features)
        
        return predictions
    
    async def optimize_horizons(self):
        """
        Dynamically optimize prediction horizons based on accuracy
        Called periodically to discover optimal time windows
        """
        logger.info("Starting horizon optimization for all rooms")
        
        try:
            # Test horizons from 1 minute to 4 hours (as specified in CLAUDE.md)
            test_horizons = list(range(1, 240, 5))  # Every 5 minutes up to 4 hours
            
            optimal_horizons = {}
            
            for room_id in self.room_ids:
                logger.info(f"Optimizing horizons for room: {room_id}")
                room_optimal = await self.horizon_optimizer.optimize_horizons_for_room(
                    room_id, test_horizons, self.predictors[room_id]
                )
                optimal_horizons[room_id] = room_optimal
            
            # Update horizons if better ones are found
            new_horizons = self.horizon_optimizer.find_accuracy_breakpoints(optimal_horizons)
            
            if new_horizons != self.horizons:
                logger.info(f"Updating horizons from {self.horizons} to {new_horizons}")
                await self._update_horizons(new_horizons)
            else:
                logger.info("Current horizons are optimal, no changes needed")
                
        except Exception as e:
            logger.error(f"Error during horizon optimization: {e}")
    
    async def _update_horizons(self, new_horizons: List[int]):
        """Update prediction horizons and create new models if needed"""
        old_horizons = set(self.horizons)
        new_horizons_set = set(new_horizons)
        
        # Add new horizon models
        for horizon in new_horizons_set - old_horizons:
            for room_id in self.room_ids:
                self.predictors[room_id][horizon] = ContinuousLearningModel(f"{room_id}_{horizon}min")
                logger.info(f"Created new model for {room_id} at {horizon}min horizon")
        
        # Remove obsolete horizon models
        for horizon in old_horizons - new_horizons_set:
            for room_id in self.room_ids:
                if horizon in self.predictors[room_id]:
                    del self.predictors[room_id][horizon]
                    logger.info(f"Removed model for {room_id} at {horizon}min horizon")
        
        self.horizons = new_horizons


class HorizonOptimizer:
    """
    Discovers optimal prediction horizons by testing different time windows
    Implements the adaptive approach from CLAUDE.md
    """
    
    def __init__(self):
        self.horizon_performance = defaultdict(lambda: defaultdict(lambda: stats.Mean()))
        self.tested_horizons = set()
    
    async def optimize_horizons_for_room(self, room_id: str, test_horizons: List[int], room_predictors: Dict) -> Dict[int, float]:
        """Test different horizons for a room and return accuracy scores"""
        horizon_scores = {}
        
        try:
            # For each test horizon, evaluate current performance
            for horizon in test_horizons:
                if horizon in room_predictors:
                    # Get performance from existing model
                    model = room_predictors[horizon]
                    performance = model.get_performance_summary()
                    
                    # Extract accuracy from model performance
                    total_accuracy = 0.0
                    model_count = 0
                    
                    for model_name, model_perf in performance.get('models', {}).items():
                        if model_perf.get('sample_count', 0) > 10:  # Need minimum samples
                            total_accuracy += model_perf.get('auc', 0.0)
                            model_count += 1
                    
                    if model_count > 0:
                        horizon_scores[horizon] = total_accuracy / model_count
                    else:
                        horizon_scores[horizon] = 0.5  # Neutral score
                else:
                    # Haven't tested this horizon yet
                    horizon_scores[horizon] = 0.5  # Neutral score
                
                # Track that we've tested this horizon
                self.tested_horizons.add(horizon)
                
                # Update performance tracking
                if horizon_scores[horizon] > 0:
                    self.horizon_performance[room_id][horizon].update(horizon_scores[horizon])
            
            return horizon_scores
            
        except Exception as e:
            logger.error(f"Error optimizing horizons for {room_id}: {e}")
            return {horizon: 0.5 for horizon in test_horizons}
    
    def find_accuracy_breakpoints(self, all_room_results: Dict[str, Dict[int, float]]) -> List[int]:
        """
        Find natural breakpoints where accuracy drops
        Returns optimal horizons discovered from data
        """
        try:
            # Aggregate scores across all rooms
            horizon_aggregate = defaultdict(list)
            
            for room_id, horizon_scores in all_room_results.items():
                for horizon, score in horizon_scores.items():
                    horizon_aggregate[horizon].append(score)
            
            # Calculate mean scores for each horizon
            horizon_means = {}
            for horizon, scores in horizon_aggregate.items():
                horizon_means[horizon] = sum(scores) / len(scores) if scores else 0.5
            
            # Sort by horizon
            sorted_horizons = sorted(horizon_means.items())
            
            # Find breakpoints where accuracy drops significantly
            breakpoints = []
            prev_score = 1.0
            
            for horizon, score in sorted_horizons:
                # Significant drop indicates a natural horizon boundary
                if prev_score - score > 0.1:  # 10% drop threshold
                    if breakpoints and horizon - breakpoints[-1] > 10:  # At least 10 min apart
                        breakpoints.append(horizon)
                elif score > 0.7:  # Good accuracy threshold
                    if not breakpoints or horizon - breakpoints[-1] > 30:  # At least 30 min apart
                        breakpoints.append(horizon)
                
                prev_score = score
            
            # Ensure we have at least basic horizons
            if not breakpoints:
                breakpoints = [15, 120]  # Default from CLAUDE.md
            elif len(breakpoints) == 1:
                if breakpoints[0] < 60:
                    breakpoints.append(120)  # Add long-term horizon
                else:
                    breakpoints.insert(0, 15)  # Add short-term horizon
            
            # Limit to maximum 4 horizons to avoid complexity
            breakpoints = sorted(breakpoints)[:4]
            
            logger.info(f"Discovered optimal horizons: {breakpoints}")
            return breakpoints
            
        except Exception as e:
            logger.error(f"Error finding accuracy breakpoints: {e}")
            return [15, 120]  # Safe default
    
    def test_horizon_accuracy(self, horizon: int, historical_data: List) -> float:
        """Test accuracy of a specific horizon against historical data"""
        try:
            # This would implement backtesting against historical data
            # For now, return a simulated score based on horizon characteristics
            
            # Shorter horizons generally more accurate but with higher variance
            if horizon <= 30:
                base_accuracy = 0.8 - (horizon * 0.01)  # Decreasing accuracy
            elif horizon <= 120:
                base_accuracy = 0.7 - ((horizon - 30) * 0.005)  # Slower decrease
            else:
                base_accuracy = 0.5 - ((horizon - 120) * 0.002)  # Very slow decrease
            
            # Add some variance
            import random
            variance = random.uniform(-0.1, 0.1)
            final_accuracy = max(0.0, min(1.0, base_accuracy + variance))
            
            return final_accuracy
            
        except Exception as e:
            logger.error(f"Error testing horizon accuracy for {horizon}: {e}")
            return 0.5
    
