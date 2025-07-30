"""
Smart Data Preprocessor for STUMPY Division by Zero Prevention
Adaptive preprocessing strategies based on occupancy patterns and data characteristics
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class OccupancyPattern(Enum):
    """Classification of occupancy patterns for adaptive preprocessing"""
    VERY_LOW = "very_low"      # < 5% ones (bedroom, low-use rooms)
    LOW = "low"                # 5-15% ones 
    BALANCED = "balanced"      # 15-85% ones (normal occupancy)
    HIGH = "high"              # 85-95% ones
    VERY_HIGH = "very_high"    # > 95% ones (rarely empty)


@dataclass
class DataCharacteristics:
    """Data characteristics for preprocessing decisions"""
    total_points: int
    unique_values: int
    occupancy_rate: float
    transition_rate: float
    std_deviation: float
    pattern_type: OccupancyPattern
    constant_window_percentage: float
    longest_constant_sequence: int


@dataclass 
class PreprocessingDecision:
    """Decision made by the preprocessor"""
    should_preprocess: bool
    method: str
    reason: str
    parameters: Dict[str, Any]
    constant_threshold: float
    original_characteristics: DataCharacteristics


class SmartDataPreprocessor:
    """
    Intelligent data preprocessor that adaptively handles different occupancy patterns
    to prevent STUMPY division by zero errors while preserving binary semantics
    """
    
    def __init__(self):
        """Initialize the smart preprocessor with adaptive thresholds"""
        self.preprocessing_history = {}
        
        # Adaptive thresholds based on investigation findings
        self.PATTERN_THRESHOLDS = {
            OccupancyPattern.VERY_LOW: {
                'constant_window_threshold': 95.0,  # Allow up to 95% constant windows
                'min_transition_rate': 0.005,       # 0.5% minimum transitions
                'preprocessing_method': 'microscopic_noise_with_trends'
            },
            OccupancyPattern.LOW: {
                'constant_window_threshold': 85.0,
                'min_transition_rate': 0.01,
                'preprocessing_method': 'minimal_linear_trends'
            },
            OccupancyPattern.BALANCED: {
                'constant_window_threshold': 70.0,
                'min_transition_rate': 0.02,
                'preprocessing_method': 'none'  # Usually doesn't need preprocessing
            },
            OccupancyPattern.HIGH: {
                'constant_window_threshold': 85.0,
                'min_transition_rate': 0.01,
                'preprocessing_method': 'minimal_linear_trends'
            },
            OccupancyPattern.VERY_HIGH: {
                'constant_window_threshold': 95.0,
                'min_transition_rate': 0.005,
                'preprocessing_method': 'microscopic_noise_with_trends'
            }
        }
        
    def analyze_data_characteristics(self, data: np.ndarray, window_size: int) -> DataCharacteristics:
        """
        Analyze data characteristics to determine preprocessing needs
        
        Args:
            data: Input time series data
            window_size: STUMPY window size for validation
            
        Returns:
            DataCharacteristics object with analysis results
        """
        logger.info(f"🔍 Analyzing data characteristics: {len(data)} points, window size {window_size}")
        
        # Basic statistics
        unique_values = len(np.unique(data))
        occupancy_rate = np.mean(data)
        std_deviation = np.std(data)
        
        # Calculate transition rate
        transitions = np.sum(np.abs(np.diff(data))) if len(data) > 1 else 0
        transition_rate = transitions / len(data) if len(data) > 0 else 0
        
        # Classify occupancy pattern
        if occupancy_rate < 0.05:
            pattern_type = OccupancyPattern.VERY_LOW
        elif occupancy_rate < 0.15:
            pattern_type = OccupancyPattern.LOW
        elif occupancy_rate < 0.85:
            pattern_type = OccupancyPattern.BALANCED
        elif occupancy_rate < 0.95:
            pattern_type = OccupancyPattern.HIGH
        else:
            pattern_type = OccupancyPattern.VERY_HIGH
        
        # Analyze sliding windows for constant sequences
        constant_window_percentage, longest_constant = self._analyze_sliding_windows(data, window_size)
        
        characteristics = DataCharacteristics(
            total_points=len(data),
            unique_values=unique_values,
            occupancy_rate=occupancy_rate,
            transition_rate=transition_rate,
            std_deviation=std_deviation,
            pattern_type=pattern_type,
            constant_window_percentage=constant_window_percentage,
            longest_constant_sequence=longest_constant
        )
        
        logger.info(f"📊 Data analysis complete:")
        logger.info(f"   Pattern type: {pattern_type.value}")
        logger.info(f"   Occupancy rate: {occupancy_rate:.3f}")
        logger.info(f"   Transition rate: {transition_rate:.3f}")
        logger.info(f"   Constant windows: {constant_window_percentage:.1f}%")
        logger.info(f"   Std deviation: {std_deviation:.6f}")
        
        return characteristics
        
    def _analyze_sliding_windows(self, data: np.ndarray, window_size: int) -> Tuple[float, int]:
        """
        Efficiently analyze sliding windows for constant sequences
        
        Returns:
            Tuple of (constant_window_percentage, longest_constant_sequence)
        """
        if len(data) < window_size:
            return 0.0, 0
            
        total_windows = len(data) - window_size + 1
        
        # Efficient sampling for large datasets
        if total_windows > 10000:
            # Sample every Nth window for performance
            sample_step = max(1, total_windows // 1000)
            sample_indices = range(0, total_windows, sample_step)
        else:
            sample_indices = range(total_windows)
        
        constant_windows = 0
        longest_constant = 0
        current_constant = 0
        
        for i in sample_indices:
            window_data = data[i:i+window_size]
            window_std = np.std(window_data)
            
            if window_std < 1e-15:  # Constant window
                constant_windows += 1
                current_constant += 1
                longest_constant = max(longest_constant, current_constant)
            else:
                current_constant = 0
        
        # Calculate percentage from sampled windows
        sampled_windows = len(sample_indices)
        constant_percentage = (constant_windows / sampled_windows * 100) if sampled_windows > 0 else 0
        
        return constant_percentage, longest_constant
        
    def should_preprocess(self, characteristics: DataCharacteristics) -> PreprocessingDecision:
        """
        Determine if preprocessing is needed based on data characteristics
        
        Args:
            characteristics: Analyzed data characteristics
            
        Returns:
            PreprocessingDecision with recommendation
        """
        pattern_config = self.PATTERN_THRESHOLDS[characteristics.pattern_type]
        constant_threshold = pattern_config['constant_window_threshold']
        min_transition_rate = pattern_config['min_transition_rate']
        
        # Decision logic
        needs_preprocessing = False
        reason_parts = []
        
        # Check for insufficient variation
        if characteristics.std_deviation < 1e-10:
            needs_preprocessing = True
            reason_parts.append(f"zero std deviation ({characteristics.std_deviation:.2e})")
            
        # Check for insufficient transitions
        if characteristics.transition_rate < min_transition_rate:
            needs_preprocessing = True
            reason_parts.append(f"low transition rate ({characteristics.transition_rate:.3f} < {min_transition_rate})")
            
        # Check for excessive constant windows
        if characteristics.constant_window_percentage > constant_threshold:
            needs_preprocessing = True
            reason_parts.append(f"high constant windows ({characteristics.constant_window_percentage:.1f}% > {constant_threshold}%)")
            
        # Check for extreme occupancy rates (nearly all 0s or 1s)
        if characteristics.occupancy_rate < 0.001 or characteristics.occupancy_rate > 0.999:
            needs_preprocessing = True
            reason_parts.append(f"extreme occupancy rate ({characteristics.occupancy_rate:.3f})")
        
        # Build decision
        if needs_preprocessing:
            method = pattern_config['preprocessing_method']
            reason = f"Data requires preprocessing: {'; '.join(reason_parts)}"
        else:
            method = "none"
            reason = f"Data suitable for STUMPY without preprocessing (pattern: {characteristics.pattern_type.value})"
        
        decision = PreprocessingDecision(
            should_preprocess=needs_preprocessing,
            method=method,
            reason=reason,
            parameters=self._get_method_parameters(characteristics, method),
            constant_threshold=constant_threshold,
            original_characteristics=characteristics
        )
        
        logger.info(f"🎯 Preprocessing decision: {decision.method}")
        logger.info(f"   Reason: {decision.reason}")
        
        return decision
        
    def _get_method_parameters(self, characteristics: DataCharacteristics, method: str) -> Dict[str, Any]:
        """Get parameters for the chosen preprocessing method"""
        base_epsilon = np.finfo(float).eps * 1000  # Base epsilon for numerical stability
        
        parameters = {
            'base_epsilon': base_epsilon,
            'occupancy_rate': characteristics.occupancy_rate,
            'pattern_type': characteristics.pattern_type.value
        }
        
        if method == 'microscopic_noise_with_trends':
            # For very low occupancy rooms (bedroom case)
            parameters.update({
                'noise_scale': base_epsilon * 10,  # Slightly larger noise
                'trend_threshold': 50,  # Apply trends to sequences > 50 points (250 min)
                'trend_scale': base_epsilon * 5
            })
        elif method == 'minimal_linear_trends':
            # For low occupancy rooms  
            parameters.update({
                'noise_scale': base_epsilon,
                'trend_threshold': 30,  # Apply trends to sequences > 30 points (150 min)
                'trend_scale': base_epsilon * 2
            })
        
        return parameters
        
    def preprocess_data(self, data: np.ndarray, decision: PreprocessingDecision) -> np.ndarray:
        """
        Apply preprocessing based on the decision
        
        Args:
            data: Original time series data
            decision: Preprocessing decision from should_preprocess()
            
        Returns:
            Preprocessed data ready for STUMPY
        """
        if not decision.should_preprocess:
            logger.info("✅ No preprocessing needed - using original data")
            return data.copy()
            
        logger.info(f"🔧 Applying preprocessing method: {decision.method}")
        
        # Convert to float for processing
        processed_data = data.astype(float)
        
        if decision.method == 'none':
            return processed_data
            
        elif decision.method == 'microscopic_noise_with_trends':
            return self._apply_microscopic_noise_with_trends(processed_data, decision.parameters)
            
        elif decision.method == 'minimal_linear_trends':
            return self._apply_minimal_linear_trends(processed_data, decision.parameters)
            
        else:
            logger.warning(f"Unknown preprocessing method: {decision.method}")
            return processed_data
            
    def _apply_microscopic_noise_with_trends(self, data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Apply microscopic noise and linear trends for very low occupancy data
        Designed for bedroom-type rooms with 97.5% zeros
        """
        logger.info("🔬 Applying microscopic noise with trends for very low occupancy data")
        
        processed_data = data.copy()
        noise_scale = params['noise_scale']
        trend_threshold = params['trend_threshold']
        trend_scale = params['trend_scale']
        
        # Step 1: Add deterministic index-based noise
        index_noise = np.arange(len(processed_data)) * noise_scale
        processed_data += index_noise
        
        # Step 2: Add subtle linear trends to long constant sequences
        sequences_processed = 0
        
        # Find constant sequences (transitions from 0 to 1 and 1 to 0)
        diff = np.diff(np.concatenate(([0], processed_data, [0])))
        
        # For zero sequences
        zero_starts = np.where((diff != 0) & (processed_data[:-1] == 0))[0]
        zero_ends = np.where((diff != 0) & (processed_data[1:] == 0))[0] + 1
        
        # For one sequences  
        one_starts = np.where((diff != 0) & (processed_data[:-1] > 0.5))[0]
        one_ends = np.where((diff != 0) & (processed_data[1:] > 0.5))[0] + 1
        
        # Process zero sequences
        for start, end in zip(zero_starts, zero_ends):
            if end - start > trend_threshold:
                sequence_length = end - start
                # Add very subtle increasing trend to break up constant windows
                linear_trend = np.linspace(0, trend_scale * sequence_length, sequence_length)
                processed_data[start:end] += linear_trend
                sequences_processed += 1
        
        # Process one sequences (occupied periods)
        for start, end in zip(one_starts, one_ends):
            if end - start > trend_threshold:
                sequence_length = end - start
                # Add subtle decreasing trend (realistic: occupancy confidence decreases over time)
                linear_trend = np.linspace(trend_scale * sequence_length, 0, sequence_length)
                processed_data[start:end] += linear_trend
                sequences_processed += 1
        
        logger.info(f"✅ Microscopic preprocessing complete:")
        logger.info(f"   Added index noise: {noise_scale:.2e}")
        logger.info(f"   Enhanced {sequences_processed} long sequences (>{trend_threshold} points)")
        logger.info(f"   Final std: {np.std(processed_data):.8f}")
        
        return processed_data
        
    def _apply_minimal_linear_trends(self, data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Apply minimal linear trends for low occupancy data
        """
        logger.info("🔧 Applying minimal linear trends for low occupancy data")
        
        processed_data = data.copy()
        noise_scale = params['noise_scale']
        trend_threshold = params['trend_threshold']
        trend_scale = params['trend_scale']
        
        # Step 1: Add minimal index-based noise
        index_noise = np.arange(len(processed_data)) * noise_scale
        processed_data += index_noise
        
        # Step 2: Add subtle trends to medium-length constant sequences
        sequences_processed = 0
        
        # Identify constant sequences
        for value in [0, 1]:
            # Find sequences of constant value
            is_value = (data == value)
            transitions = np.diff(np.concatenate(([False], is_value, [False])).astype(int))
            starts = np.where(transitions == 1)[0]
            ends = np.where(transitions == -1)[0]
            
            for start, end in zip(starts, ends):
                if end - start > trend_threshold:
                    sequence_length = end - start
                    # Add minimal trend based on value
                    if value == 0:
                        # Slight increasing trend for zero sequences
                        trend = np.linspace(0, trend_scale * np.sqrt(sequence_length), sequence_length)
                    else:
                        # Slight decreasing trend for one sequences
                        trend = np.linspace(trend_scale * np.sqrt(sequence_length), 0, sequence_length)
                    
                    processed_data[start:end] += trend
                    sequences_processed += 1
        
        logger.info(f"✅ Minimal preprocessing complete:")
        logger.info(f"   Added index noise: {noise_scale:.2e}")
        logger.info(f"   Enhanced {sequences_processed} sequences (>{trend_threshold} points)")
        logger.info(f"   Final std: {np.std(processed_data):.8f}")
        
        return processed_data
        
    def validate_preprocessed_data(self, original_data: np.ndarray, processed_data: np.ndarray, 
                                 window_size: int) -> Dict[str, Any]:
        """
        Validate that preprocessing achieved its goals
        
        Returns:
            Validation results dictionary
        """
        logger.info("🔍 Validating preprocessed data...")
        
        # Basic validation
        validation = {
            'length_preserved': len(original_data) == len(processed_data),
            'binary_semantics_preserved': True,
            'std_improved': np.std(processed_data) > np.std(original_data),
            'original_std': np.std(original_data),
            'processed_std': np.std(processed_data)
        }
        
        # Check if binary semantics are preserved (values should still be close to 0 or 1)
        for i, (orig, proc) in enumerate(zip(original_data, processed_data)):
            if orig == 0 and abs(proc) > 0.1:  # Zero values shouldn't change much
                validation['binary_semantics_preserved'] = False
                break
            elif orig == 1 and abs(proc - 1) > 0.1:  # One values shouldn't change much
                validation['binary_semantics_preserved'] = False
                break
        
        # Check sliding windows
        if len(processed_data) >= window_size:
            constant_windows = 0
            total_windows = len(processed_data) - window_size + 1
            
            # Sample check for performance
            sample_step = max(1, total_windows // 100)
            sampled_windows = 0
            
            for i in range(0, total_windows, sample_step):
                window_data = processed_data[i:i+window_size]
                if np.std(window_data) < 1e-15:
                    constant_windows += 1
                sampled_windows += 1
            
            validation['constant_window_percentage'] = (constant_windows / sampled_windows * 100) if sampled_windows > 0 else 0
            validation['stumpy_ready'] = validation['constant_window_percentage'] < 95  # Very conservative
            
        logger.info(f"✅ Validation complete:")
        logger.info(f"   Binary semantics preserved: {validation['binary_semantics_preserved']}")
        logger.info(f"   Std improved: {validation['original_std']:.8f} → {validation['processed_std']:.8f}")
        if 'constant_window_percentage' in validation:
            logger.info(f"   Constant windows: {validation['constant_window_percentage']:.1f}%")
            logger.info(f"   STUMPY ready: {validation['stumpy_ready']}")
        
        return validation
        
    def get_preprocessing_summary(self, room_name: str) -> Dict[str, Any]:
        """Get preprocessing history summary for a room"""
        if room_name not in self.preprocessing_history:
            return {'no_history': True}
            
        history = self.preprocessing_history[room_name]
        return {
            'total_preprocessings': len(history),
            'methods_used': list(set(h['method'] for h in history)),
            'success_rate': sum(1 for h in history if h.get('validation', {}).get('stumpy_ready', False)) / len(history),
            'recent_decisions': history[-5:]  # Last 5 decisions
        }

    def preprocess_for_stumpy(self, data: np.ndarray, window_size: int, room_name: str = "unknown") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Main entry point: analyze data and apply preprocessing if needed
        
        Args:
            data: Time series data (binary occupancy)
            window_size: STUMPY window size
            room_name: Room identifier for logging and history
            
        Returns:
            Tuple of (preprocessed_data, processing_info)
        """
        logger.info(f"🚀 Smart preprocessing for {room_name}, window size {window_size}")
        
        # Step 1: Analyze data characteristics
        characteristics = self.analyze_data_characteristics(data, window_size)
        
        # Step 2: Decide if preprocessing is needed
        decision = self.should_preprocess(characteristics)
        
        # Step 3: Apply preprocessing if needed
        processed_data = self.preprocess_data(data, decision)
        
        # Step 4: Validate results
        validation = self.validate_preprocessed_data(data, processed_data, window_size)
        
        # Store in history
        if room_name not in self.preprocessing_history:
            self.preprocessing_history[room_name] = []
            
        history_entry = {
            'timestamp': datetime.now(timezone.utc),
            'window_size': window_size,
            'characteristics': characteristics,
            'decision': decision,
            'validation': validation,
            'method': decision.method
        }
        
        self.preprocessing_history[room_name].append(history_entry)
        
        # Create summary info
        processing_info = {
            'preprocessing_applied': decision.should_preprocess,
            'method': decision.method,
            'reason': decision.reason,
            'original_occupancy_rate': characteristics.occupancy_rate,
            'original_std': characteristics.std_deviation,
            'processed_std': np.std(processed_data),
            'pattern_type': characteristics.pattern_type.value,
            'validation': validation,
            'ready_for_stumpy': validation.get('stumpy_ready', True)
        }
        
        if decision.should_preprocess:
            logger.info(f"✅ Preprocessing complete for {room_name}")
            logger.info(f"   Method: {decision.method}")
            logger.info(f"   Ready for STUMPY: {processing_info['ready_for_stumpy']}")
        else:
            logger.info(f"✅ No preprocessing needed for {room_name} - data ready for STUMPY")
        
        return processed_data, processing_info