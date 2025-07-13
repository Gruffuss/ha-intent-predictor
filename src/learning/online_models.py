"""
Online Learning Models - Continuously updating models using River framework
Implements the adaptive prediction system from CLAUDE.md
"""

import logging
from collections import defaultdict
from typing import Dict, Any, Optional
from river import ensemble, tree, neural_net, metrics, preprocessing
from river import stats

logger = logging.getLogger(__name__)


class ContinuousLearningModel:
    """Continuously updating occupancy prediction model for a specific room"""
    
    def __init__(self, room_id: str):
        self.room_id = room_id
        
        # Ensemble of online learners as specified in CLAUDE.md
        self.models = {
            'gradient_boost': ensemble.AdaptiveRandomForestClassifier(
                n_models=10,
                max_features="sqrt",
                lambda_value=6,
                grace_period=10
            ),
            'hoeffding_tree': tree.ExtremelyFastDecisionTreeClassifier(
                grace_period=10,
                split_confidence=1e-5,
                nominal_attributes=['sensor_type', 'room']
            ),
            'neural': neural_net.MLPClassifier(
                hidden_dims=(20, 10),
                activations=('relu', 'relu', 'identity'),
                learning_rate=0.001
            )
        }
        
        # Meta-learner decides which model to trust
        self.meta_learner = MetaLearner()
        
        # Track each model's performance
        self.model_performance = {
            name: metrics.Rolling(metrics.ROCAUC(), window_size=1000)
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
                pred = model.predict_proba_one(features)
                if pred is not None:
                    predictions[name] = pred.get(True, 0.0)
                else:
                    predictions[name] = 0.5  # Neutral prediction
            except:
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