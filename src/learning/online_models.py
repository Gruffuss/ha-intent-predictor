"""
Online Learning Models - Continuously updating models using River framework
Implements the adaptive prediction system from CLAUDE.md
"""

import logging
from collections import defaultdict
from typing import Dict, Any, Optional
from river import ensemble, tree, metrics, preprocessing, forest
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
                split_confidence=1e-5
            ),
            'logistic': linear_model.LogisticRegression(
                l2=0.1
            )
        }
        
        # Meta-learner decides which model to trust
        self.meta_learner = MetaLearner()
        
        # Track each model's performance
        self.model_performance = {
            name: metrics.Rolling(metrics.Accuracy(), window_size=1000)
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
        # Preprocess features
        processed_features = self.preprocessor.learn_one(features).transform_one(features)
        
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
            # Preprocess features
            processed_features = self.preprocessor.transform_one(features)
            
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
                'performance': self.model_performance[name].get() if self.model_performance[name].n > 0 else 0.0
            }
        
        return contributions
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring"""
        summary = {
            'room_id': self.room_id,
            'models': {}
        }
        
        for name, perf_metric in self.model_performance.items():
            if perf_metric.n > 0:
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
                        'n_samples': metric.n,
                        'value': metric.get() if metric.n > 0 else 0.0
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
            if self.model_scores[model_name].n > 10:  # Need minimum samples
                base_score = self.model_scores[model_name].get()
            else:
                base_score = 0.5  # Neutral weight for new models
            
            # Adjust based on recent performance
            recent_perf = performance_metrics.get(model_name)
            if recent_perf and hasattr(recent_perf, 'get') and recent_perf.n > 0:
                recent_score = recent_perf.get()
                # Combine base and recent performance
                final_score = base_score * 0.7 + recent_score * 0.3
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
    
    def optimize_horizons(self):
        """
        Dynamically optimize prediction horizons based on accuracy
        Called periodically to discover optimal time windows
        """
        # This would implement the horizon optimization from CLAUDE.md
        # Testing different horizons and finding accuracy breakpoints
        logger.info("Horizon optimization not yet implemented")
        pass


class HorizonOptimizer:
    """
    Discovers optimal prediction horizons by testing different time windows
    Implements the adaptive approach from CLAUDE.md
    """
    
    def __init__(self):
        self.horizon_performance = defaultdict(lambda: RunningStats())
        self.tested_horizons = set()
    
    def test_horizon_accuracy(self, horizon_minutes: int, historical_data) -> float:
        """Test accuracy for a specific prediction horizon"""
        # This would implement the horizon testing from CLAUDE.md
        # For now, return placeholder
        return 0.5
    
    def find_accuracy_breakpoints(self, results: Dict[int, float]) -> list:
        """Find natural breakpoints where accuracy drops significantly"""
        # Implementation would analyze the results curve for breakpoints
        return [15, 120]  # Placeholder