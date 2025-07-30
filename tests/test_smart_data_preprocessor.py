"""
Test suite for SmartDataPreprocessor
Validates preprocessing strategies for different occupancy patterns
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from learning.smart_data_preprocessor import (
    SmartDataPreprocessor, 
    OccupancyPattern, 
    DataCharacteristics, 
    PreprocessingDecision
)


class TestSmartDataPreprocessor:
    """Test the smart data preprocessor with various occupancy patterns"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.preprocessor = SmartDataPreprocessor()
        
    def create_synthetic_data(self, pattern_type: str, length: int = 1000) -> np.ndarray:
        """Create synthetic data for testing different patterns"""
        np.random.seed(42)  # Reproducible tests
        
        if pattern_type == "bedroom":
            # 97.5% zeros, 2.5% ones - high risk for STUMPY division by zero
            data = np.zeros(length)
            # Add few occupied periods
            occupied_indices = np.random.choice(length, size=int(length * 0.025), replace=False)
            data[occupied_indices] = 1
            return data
            
        elif pattern_type == "office_balanced":
            # 49.3% zeros, 50.7% ones - should be safe for STUMPY
            data = np.random.choice([0, 1], size=length, p=[0.493, 0.507])
            return data
            
        elif pattern_type == "living_kitchen":
            # Balanced data with realistic patterns - 0% zero stddev windows
            data = np.zeros(length)
            # Create realistic occupancy patterns with transitions
            current_pos = 0
            while current_pos < length:
                # Occupied period
                occupied_duration = np.random.randint(10, 60)  # 10-60 time units
                end_occupied = min(current_pos + occupied_duration, length)
                data[current_pos:end_occupied] = 1
                current_pos = end_occupied
                
                # Unoccupied period
                unoccupied_duration = np.random.randint(5, 30)  # 5-30 time units
                current_pos += unoccupied_duration
            
            return data
            
        elif pattern_type == "constant_zeros":
            # Extreme case: all zeros
            return np.zeros(length)
            
        elif pattern_type == "constant_ones":
            # Extreme case: all ones
            return np.ones(length)
            
        elif pattern_type == "minimal_variation":
            # Very low variation
            data = np.zeros(length)
            data[length//2] = 1  # Single occupied point
            return data
            
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    def test_analyze_data_characteristics_bedroom(self):
        """Test analysis of bedroom-type data (very low occupancy)"""
        data = self.create_synthetic_data("bedroom", 1000)
        window_size = 48  # 4 hours
        
        characteristics = self.preprocessor.analyze_data_characteristics(data, window_size)
        
        assert characteristics.pattern_type == OccupancyPattern.VERY_LOW
        assert characteristics.occupancy_rate < 0.05
        assert characteristics.unique_values == 2
        assert characteristics.transition_rate > 0  # Should have some transitions
        
    def test_analyze_data_characteristics_balanced(self):
        """Test analysis of balanced occupancy data"""
        data = self.create_synthetic_data("office_balanced", 1000)
        window_size = 48
        
        characteristics = self.preprocessor.analyze_data_characteristics(data, window_size)
        
        assert characteristics.pattern_type == OccupancyPattern.BALANCED
        assert 0.15 <= characteristics.occupancy_rate <= 0.85
        assert characteristics.unique_values == 2
        
    def test_analyze_data_characteristics_constant(self):
        """Test analysis of constant data"""
        data = self.create_synthetic_data("constant_zeros", 1000)
        window_size = 48
        
        characteristics = self.preprocessor.analyze_data_characteristics(data, window_size)
        
        assert characteristics.pattern_type == OccupancyPattern.VERY_LOW
        assert characteristics.occupancy_rate == 0.0
        assert characteristics.unique_values == 1
        assert characteristics.transition_rate == 0.0
        assert characteristics.std_deviation == 0.0
        
    def test_should_preprocess_bedroom_case(self):
        """Test preprocessing decision for bedroom-type data"""
        data = self.create_synthetic_data("bedroom", 1000)
        window_size = 48
        
        characteristics = self.preprocessor.analyze_data_characteristics(data, window_size)
        decision = self.preprocessor.should_preprocess(characteristics)
        
        # Bedroom data might need preprocessing depending on constant window percentage
        if characteristics.constant_window_percentage > 95:
            assert decision.should_preprocess == True
            assert decision.method == 'microscopic_noise_with_trends'
        
        assert decision.constant_threshold == 95.0  # Very low occupancy threshold
        
    def test_should_preprocess_balanced_case(self):
        """Test preprocessing decision for balanced data"""
        data = self.create_synthetic_data("living_kitchen", 1000)
        window_size = 48
        
        characteristics = self.preprocessor.analyze_data_characteristics(data, window_size)
        decision = self.preprocessor.should_preprocess(characteristics)
        
        # Balanced data should rarely need preprocessing
        assert decision.constant_threshold == 70.0  # Balanced occupancy threshold
        
    def test_should_preprocess_constant_data(self):
        """Test preprocessing decision for constant data"""
        data = self.create_synthetic_data("constant_zeros", 1000)
        window_size = 48
        
        characteristics = self.preprocessor.analyze_data_characteristics(data, window_size)
        decision = self.preprocessor.should_preprocess(characteristics)
        
        # Constant data should definitely need preprocessing
        assert decision.should_preprocess == True
        assert decision.method == 'microscopic_noise_with_trends'
        assert 'zero std deviation' in decision.reason or 'extreme occupancy rate' in decision.reason
        
    def test_preprocess_data_microscopic_noise(self):
        """Test microscopic noise preprocessing"""
        data = self.create_synthetic_data("bedroom", 1000)
        window_size = 48
        
        characteristics = self.preprocessor.analyze_data_characteristics(data, window_size)
        decision = PreprocessingDecision(
            should_preprocess=True,
            method='microscopic_noise_with_trends',
            reason='Test case',
            parameters=self.preprocessor._get_method_parameters(characteristics, 'microscopic_noise_with_trends'),
            constant_threshold=95.0,
            original_characteristics=characteristics
        )
        
        processed_data = self.preprocessor.preprocess_data(data, decision)
        
        # Verify preprocessing effects
        assert len(processed_data) == len(data)
        assert np.std(processed_data) > np.std(data)  # Should increase variation
        
        # Binary semantics should be preserved (values close to 0 or 1)
        for orig, proc in zip(data, processed_data):
            if orig == 0:
                assert abs(proc) < 0.1  # Zeros should stay close to 0
            else:
                assert abs(proc - 1) < 0.1  # Ones should stay close to 1
                
    def test_preprocess_data_minimal_trends(self):
        """Test minimal linear trends preprocessing"""
        # Create data with some long constant sequences
        data = np.zeros(1000)
        data[100:200] = 1  # Long occupied sequence
        data[500:600] = 1  # Another long sequence
        
        window_size = 48
        characteristics = self.preprocessor.analyze_data_characteristics(data, window_size)
        decision = PreprocessingDecision(
            should_preprocess=True,
            method='minimal_linear_trends',
            reason='Test case',
            parameters=self.preprocessor._get_method_parameters(characteristics, 'minimal_linear_trends'),
            constant_threshold=85.0,
            original_characteristics=characteristics
        )
        
        processed_data = self.preprocessor.preprocess_data(data, decision)
        
        # Verify preprocessing effects
        assert len(processed_data) == len(data)
        assert np.std(processed_data) > np.std(data)
        
        # Check that long sequences got trends
        occupied_sequence = processed_data[100:200]
        assert len(np.unique(occupied_sequence.round(10))) > 1  # Should have variation
        
    def test_validate_preprocessed_data(self):
        """Test validation of preprocessed data"""
        original_data = self.create_synthetic_data("bedroom", 1000)
        
        # Apply preprocessing
        processed_data, _ = self.preprocessor.preprocess_for_stumpy(
            original_data, window_size=48, room_name="test_bedroom"
        )
        
        validation = self.preprocessor.validate_preprocessed_data(
            original_data, processed_data, window_size=48
        )
        
        assert validation['length_preserved'] == True
        assert validation['binary_semantics_preserved'] == True
        assert validation['std_improved'] == True
        assert 'constant_window_percentage' in validation
        assert 'stumpy_ready' in validation
        
    def test_preprocess_for_stumpy_bedroom(self):
        """Test complete preprocessing pipeline for bedroom data"""
        data = self.create_synthetic_data("bedroom", 1000)
        window_size = 48
        
        processed_data, info = self.preprocessor.preprocess_for_stumpy(
            data, window_size, "test_bedroom"
        )
        
        # Check results
        assert len(processed_data) == len(data)
        assert info['pattern_type'] == 'very_low'
        assert 'preprocessing_applied' in info
        assert 'ready_for_stumpy' in info
        
        # If preprocessing was applied, should improve std
        if info['preprocessing_applied']:
            assert info['processed_std'] > info['original_std']
            
    def test_preprocess_for_stumpy_balanced(self):
        """Test complete preprocessing pipeline for balanced data"""
        data = self.create_synthetic_data("office_balanced", 1000)
        window_size = 48
        
        processed_data, info = self.preprocessor.preprocess_for_stumpy(
            data, window_size, "test_office"
        )
        
        # Balanced data might not need preprocessing
        assert len(processed_data) == len(data)
        assert info['pattern_type'] == 'balanced'
        assert info['ready_for_stumpy'] == True
        
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        # Empty data
        empty_data = np.array([])
        characteristics = self.preprocessor.analyze_data_characteristics(empty_data, 10)
        assert characteristics.total_points == 0
        
        # Single point
        single_data = np.array([1])
        characteristics = self.preprocessor.analyze_data_characteristics(single_data, 10)
        assert characteristics.total_points == 1
        assert characteristics.transition_rate == 0
        
        # Window size larger than data
        small_data = np.array([0, 1, 0])
        window_size = 10
        processed_data, info = self.preprocessor.preprocess_for_stumpy(
            small_data, window_size, "test_small"
        )
        # Should handle gracefully
        assert len(processed_data) == len(small_data)
        
    def test_preprocessing_history(self):
        """Test preprocessing history tracking"""
        data = self.create_synthetic_data("bedroom", 500)
        
        # Process same room multiple times
        for i in range(3):
            self.preprocessor.preprocess_for_stumpy(
                data, window_size=24, room_name="test_history"
            )
        
        summary = self.preprocessor.get_preprocessing_summary("test_history")
        
        assert summary['total_preprocessings'] == 3
        assert 'methods_used' in summary
        assert 'success_rate' in summary
        assert len(summary['recent_decisions']) <= 5
        
    def test_different_window_sizes(self):
        """Test preprocessing with different window sizes"""
        data = self.create_synthetic_data("bedroom", 1000)
        
        window_sizes = [12, 24, 48, 96, 144]  # 1h, 2h, 4h, 8h, 12h
        
        for window_size in window_sizes:
            processed_data, info = self.preprocessor.preprocess_for_stumpy(
                data, window_size, f"test_window_{window_size}"
            )
            
            assert len(processed_data) == len(data)
            assert info['ready_for_stumpy'] in [True, False]  # Should have a decision
            
    def test_real_world_patterns(self):
        """Test with patterns mimicking real investigation data"""
        
        # Bedroom pattern: 97.5% zeros, 82% zero stddev windows
        bedroom_data = np.zeros(2000)
        # Add sparse occupancy
        occupied_indices = np.random.choice(2000, size=50, replace=False)
        bedroom_data[occupied_indices] = 1
        
        processed_data, info = self.preprocessor.preprocess_for_stumpy(
            bedroom_data, window_size=48, room_name="real_bedroom"
        )
        
        assert info['pattern_type'] == 'very_low'
        assert info['original_occupancy_rate'] < 0.05
        
        # Office pattern: 49.3% zeros, 0% zero stddev windows  
        office_data = np.random.choice([0, 1], size=2000, p=[0.493, 0.507])
        
        processed_data, info = self.preprocessor.preprocess_for_stumpy(
            office_data, window_size=48, room_name="real_office"
        )
        
        assert info['pattern_type'] in ['balanced', 'low']
        assert 0.4 < info['original_occupancy_rate'] < 0.6
        
    def test_stumpy_compatibility(self):
        """Test that processed data is compatible with STUMPY-style operations"""
        data = self.create_synthetic_data("bedroom", 1000)
        window_size = 48
        
        processed_data, info = self.preprocessor.preprocess_for_stumpy(
            data, window_size, "stumpy_test"
        )
        
        if info['ready_for_stumpy']:
            # Test operations that STUMPY performs internally
            
            # 1. Overall standard deviation should be non-zero
            overall_std = np.std(processed_data)
            assert overall_std > 1e-15
            
            # 2. Sample sliding windows to ensure they're not all constant
            total_windows = len(processed_data) - window_size + 1
            if total_windows > 0:
                sample_indices = np.random.choice(
                    total_windows, 
                    size=min(100, total_windows), 
                    replace=False
                )
                
                constant_windows = 0
                for i in sample_indices:
                    window = processed_data[i:i+window_size]
                    if np.std(window) < 1e-15:
                        constant_windows += 1
                
                constant_percentage = constant_windows / len(sample_indices) * 100
                
                # Should be below the adaptive threshold for the pattern type
                pattern_config = self.preprocessor.PATTERN_THRESHOLDS[
                    OccupancyPattern.VERY_LOW if info['original_occupancy_rate'] < 0.05
                    else OccupancyPattern.BALANCED
                ]
                expected_threshold = pattern_config['constant_window_threshold']
                
                # Allow some tolerance for randomness in testing
                assert constant_percentage <= expected_threshold + 5
                
    @pytest.mark.parametrize("pattern_type", [
        "bedroom", "office_balanced", "living_kitchen", "minimal_variation"
    ])
    def test_all_pattern_types(self, pattern_type):
        """Test preprocessing for all pattern types"""
        data = self.create_synthetic_data(pattern_type, 1000)
        window_size = 48
        
        processed_data, info = self.preprocessor.preprocess_for_stumpy(
            data, window_size, f"test_{pattern_type}"
        )
        
        # Basic assertions that should hold for all pattern types
        assert len(processed_data) == len(data)
        assert info['pattern_type'] in [p.value for p in OccupancyPattern]
        assert isinstance(info['ready_for_stumpy'], bool)
        assert info['original_occupancy_rate'] >= 0.0
        assert info['processed_std'] >= 0.0
        
        # If preprocessing was applied, std should improve
        if info['preprocessing_applied']:
            assert info['processed_std'] > info['original_std']


class TestIntegrationWithPatternDiscovery:
    """Integration tests with the pattern discovery system"""
    
    def setup_method(self):
        """Set up integration test fixtures"""
        self.preprocessor = SmartDataPreprocessor()
        
    @patch('src.learning.pattern_discovery.TimescaleDBManager')
    @patch('src.learning.pattern_discovery.ConfigLoader')
    def test_pattern_discovery_integration(self, mock_config, mock_db):
        """Test integration with pattern discovery system"""
        # Mock configuration and database
        mock_config.return_value.get.return_value = {
            'host': 'localhost', 'port': 5432, 
            'user': 'test', 'password': 'test', 'database': 'test'
        }
        
        # Create mock database manager
        mock_db_instance = MagicMock()
        mock_db.return_value = mock_db_instance
        
        # This test would require more extensive mocking to fully test
        # the integration, but validates the basic structure
        assert self.preprocessor is not None
        
    def test_preprocessing_with_real_stumpy_call(self):
        """Test preprocessing with actual STUMPY call (if available)"""
        try:
            import stumpy
            
            # Create problematic data
            data = np.zeros(500)
            data[50:60] = 1  # Small occupied period
            
            window_size = 20
            
            # Preprocess
            processed_data, info = self.preprocessor.preprocess_for_stumpy(
                data, window_size, "stumpy_integration_test"
            )
            
            if info['ready_for_stumpy']:
                # Try actual STUMPY call
                try:
                    mp = stumpy.stump(processed_data, m=window_size)
                    assert mp is not None
                    assert mp.shape[0] == len(processed_data) - window_size + 1
                    
                    # Check for NaN or infinite values that indicate division by zero
                    assert not np.any(np.isnan(mp))
                    assert not np.any(np.isinf(mp))
                    
                except Exception as e:
                    pytest.fail(f"STUMPY failed even after preprocessing: {e}")
                    
        except ImportError:
            pytest.skip("STUMPY not available for integration test")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])