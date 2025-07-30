#!/usr/bin/env python3
"""
Fix for STUMPY division by zero errors in pattern_discovery.py

Based on comprehensive debugging, this script implements the correct fix:
1. STUMPY can handle some zero stddev windows (they become inf, which STUMPY handles)
2. The issue is when ALL or nearly ALL windows have zero stddev
3. We need smarter preprocessing that allows STUMPY to work while preventing real hangs
4. Focus on adding minimal deterministic noise to preserve binary semantics
"""

import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

def add_minimal_deterministic_noise(data_values: np.ndarray, occupancy_rate: float = None) -> np.ndarray:
    """
    Add minimal deterministic noise to binary occupancy data to prevent STUMPY issues
    while preserving the binary semantic meaning (0=unoccupied, 1=occupied)
    
    Args:
        data_values: Binary occupancy data (0s and 1s)
        occupancy_rate: Optional occupancy rate for optimization
        
    Returns:
        Preprocessed data with minimal noise to prevent zero stddev windows
    """
    if occupancy_rate is None:
        occupancy_rate = np.mean(data_values)
    
    logger.info(f"Adding deterministic noise to {len(data_values)} points (occupancy: {occupancy_rate:.1%})")
    
    # Convert to float for processing
    processed_data = data_values.astype(float)
    
    # Method 1: Deterministic epsilon based on index position
    # This preserves exact binary values while adding microscopic variation
    epsilon = np.finfo(float).eps * 1000  # Small but not too small
    index_noise = np.arange(len(processed_data)) * epsilon
    
    # Apply noise only to long constant sequences to minimize impact
    # This is more surgical than adding noise everywhere
    diff = np.diff(np.concatenate(([0], processed_data, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    sequences_processed = 0
    for start, end in zip(starts, ends):
        sequence_length = end - start
        if sequence_length > 20:  # Only process long constant sequences (>100 minutes)
            # Add microscopic linear trend within the sequence
            linear_trend = np.linspace(0, epsilon * sequence_length, sequence_length)
            processed_data[start:end] += linear_trend
            sequences_processed += 1
    
    # For any remaining problematic areas, add the index-based epsilon
    processed_data += index_noise
    
    # Verify the result still looks binary
    min_val, max_val = processed_data.min(), processed_data.max()
    std_noise = np.std(processed_data - data_values)
    
    logger.info(f"Preprocessing complete:")
    logger.info(f"  - Processed {sequences_processed} long constant sequences")
    logger.info(f"  - Value range: {min_val:.10f} to {max_val:.10f}")
    logger.info(f"  - Noise std: {std_noise:.2e}")
    logger.info(f"  - Preserves binary: {np.allclose(processed_data, np.round(processed_data), atol=1e-8)}")
    
    return processed_data

def validate_stumpy_readiness(data_values: np.ndarray, window_size: int, 
                            max_constant_percentage: float = 80.0) -> Dict[str, Any]:
    """
    Validate if data is ready for STUMPY processing
    
    Args:
        data_values: Time series data
        window_size: STUMPY window size
        max_constant_percentage: Maximum allowed percentage of constant windows
        
    Returns:
        Validation results with recommendations
    """
    total_windows = len(data_values) - window_size + 1
    
    if total_windows < 1:
        return {
            'ready': False,
            'reason': 'Insufficient data for any windows',
            'recommendation': 'need_more_data'
        }
    
    # Sample check for performance (check every 50th window)
    sample_step = max(1, total_windows // 100)
    constant_windows = 0
    sampled_windows = 0
    
    for i in range(0, total_windows, sample_step):
        window_data = data_values[i:i+window_size]
        window_std = np.std(window_data)
        if window_std < 1e-15:  # Effectively zero
            constant_windows += 1
        sampled_windows += 1
    
    # Estimate percentage from sample
    constant_percentage = (constant_windows / sampled_windows * 100) if sampled_windows > 0 else 0
    
    # For binary occupancy data, be more lenient with low occupancy rooms
    occupancy_rate = np.mean(data_values)
    if occupancy_rate < 0.1:  # Very low occupancy (< 10%)
        adjusted_threshold = min(95.0, max_constant_percentage + 15)
    elif occupancy_rate < 0.2:  # Low occupancy (< 20%) 
        adjusted_threshold = min(90.0, max_constant_percentage + 10)
    else:
        adjusted_threshold = max_constant_percentage
    
    ready = constant_percentage <= adjusted_threshold
    
    result = {
        'ready': ready,
        'constant_percentage': constant_percentage,
        'threshold_used': adjusted_threshold,
        'occupancy_rate': occupancy_rate,
        'sampled_windows': sampled_windows,
        'total_windows': total_windows
    }
    
    if ready:
        result['recommendation'] = 'proceed'
    elif constant_percentage > 95:
        result['reason'] = f'{constant_percentage:.1f}% constant windows - too extreme for STUMPY'
        result['recommendation'] = 'add_noise'
    else:
        result['reason'] = f'{constant_percentage:.1f}% constant windows exceeds {adjusted_threshold:.1f}% threshold'  
        result['recommendation'] = 'add_noise'
    
    return result

def preprocess_for_stumpy(data_values: np.ndarray, window_size: int) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Preprocess time series data for STUMPY to prevent division by zero
    
    Args:
        data_values: Raw time series data
        window_size: STUMPY window size
        
    Returns:
        (processed_data, processing_info)
    """
    # Initial validation
    validation = validate_stumpy_readiness(data_values, window_size)
    
    processing_info = {
        'original_validation': validation,
        'preprocessing_applied': False,
        'method_used': None
    }
    
    if validation['ready']:
        logger.info(f"Data ready for STUMPY: {validation['constant_percentage']:.1f}% constant windows")
        return data_values, processing_info
    
    # Apply preprocessing
    logger.info(f"Applying preprocessing: {validation['reason']}")
    
    occupancy_rate = validation['occupancy_rate']
    processed_data = add_minimal_deterministic_noise(data_values, occupancy_rate)
    
    # Validate processed data
    post_validation = validate_stumpy_readiness(processed_data, window_size)
    
    processing_info.update({
        'preprocessing_applied': True,
        'method_used': 'deterministic_noise',
        'post_validation': post_validation,
        'improvement': validation['constant_percentage'] - post_validation['constant_percentage']
    })
    
    if post_validation['ready']:
        logger.info(f"✅ Preprocessing successful: {post_validation['constant_percentage']:.1f}% constant windows")
    else:
        logger.warning(f"⚠️ Preprocessing insufficient: {post_validation['constant_percentage']:.1f}% still constant")
    
    return processed_data, processing_info

# Integration code for pattern_discovery.py
STUMPY_PREPROCESSING_CODE = '''
def preprocess_for_stumpy(data_values: np.ndarray, window_size: int) -> tuple[np.ndarray, Dict[str, Any]]:
    """Preprocess time series data for STUMPY to prevent division by zero"""
    
    def add_minimal_deterministic_noise(data_values: np.ndarray) -> np.ndarray:
        """Add minimal deterministic noise preserving binary semantics"""
        processed_data = data_values.astype(float)
        epsilon = np.finfo(float).eps * 1000
        
        # Add microscopic index-based variation
        index_noise = np.arange(len(processed_data)) * epsilon
        processed_data += index_noise
        
        # Add linear trends to long constant sequences
        diff = np.diff(np.concatenate(([0], processed_data, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start, end in zip(starts, ends):
            if end - start > 20:  # Long sequences only
                sequence_length = end - start
                linear_trend = np.linspace(0, epsilon * sequence_length, sequence_length)
                processed_data[start:end] += linear_trend
        
        return processed_data
    
    # Quick validation: check sample of windows for constant data
    total_windows = len(data_values) - window_size + 1
    sample_step = max(1, total_windows // 100)
    constant_count = 0
    
    for i in range(0, total_windows, sample_step):
        if np.std(data_values[i:i+window_size]) < 1e-15:
            constant_count += 1
    
    constant_percentage = (constant_count / (total_windows // sample_step)) * 100
    occupancy_rate = np.mean(data_values)
    
    # Adaptive threshold based on occupancy rate
    threshold = 95 if occupancy_rate < 0.1 else 85 if occupancy_rate < 0.2 else 75
    
    if constant_percentage <= threshold:
        return data_values, {'preprocessing_applied': False, 'constant_percentage': constant_percentage}
    
    # Apply preprocessing
    processed_data = add_minimal_deterministic_noise(data_values)
    
    return processed_data, {
        'preprocessing_applied': True, 
        'original_constant_percentage': constant_percentage,
        'method': 'deterministic_noise'
    }

# Modified STUMPY worker function
def stumpy_worker():
    """Run STUMPY with preprocessing and proper error handling"""
    try:
        logger.info(f"🚀 STUMPY thread started for window {window_size}")
        
        # CRITICAL: Preprocess data before STUMPY
        data_for_stumpy, preprocessing_info = preprocess_for_stumpy(ts.values, window_size)
        
        if preprocessing_info['preprocessing_applied']:
            logger.info(f"📝 Applied preprocessing: {preprocessing_info.get('method', 'unknown')}")
        
        # Final validation
        overall_std = np.std(data_for_stumpy)
        if overall_std < 1e-15:
            raise ValueError(f"Data has zero variance after preprocessing: std={overall_std:.2e}")
        
        logger.info(f"🎯 STUMPY input: {len(data_for_stumpy)} points, std={overall_std:.8f}")
        
        # Run STUMPY with preprocessed data and warnings control
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='divide by zero encountered.*', category=RuntimeWarning)
            warnings.filterwarnings('ignore', message='invalid value encountered.*', category=RuntimeWarning)
            
            mp = stumpy.stump(data_for_stumpy, m=window_size)
        
        logger.info(f"✅ STUMPY completed for {label}: matrix profile shape {mp.shape}")
        return mp
        
    except Exception as e:
        logger.error(f"💥 STUMPY failed for {label}: {e}")
        return None
'''

if __name__ == "__main__":
    print("This is a utility module for fixing STUMPY preprocessing.")
    print("Use the functions above in your pattern_discovery.py file.")
    print("\\nKey functions:")
    print("- preprocess_for_stumpy(): Main preprocessing function")
    print("- add_minimal_deterministic_noise(): Add minimal noise to binary data")  
    print("- validate_stumpy_readiness(): Check if data is ready for STUMPY")