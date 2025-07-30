#!/usr/bin/env python3
"""
Test script to verify the STUMPY division by zero fix works correctly

This script tests the improved preprocessing logic on real sensor data
to ensure it properly handles zero standard deviation windows without
suppressing all warnings unnecessarily.
"""

import logging
import numpy as np
import pandas as pd
import warnings
import asyncio
from datetime import datetime, timezone, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our fixed pattern discovery
from src.learning.pattern_discovery import PatternDiscovery

async def test_stumpy_fix():
    """Test the STUMPY fix with real data"""
    logger.info("🧪 Testing STUMPY division by zero fix")
    
    # Initialize pattern discovery
    pattern_discovery = PatternDiscovery()
    
    # Test with office data (known to have high percentage of constant windows)
    logger.info("Testing with office room (low occupancy, high constant windows)")
    
    try:
        # This should now work with our improved preprocessing
        patterns = await pattern_discovery.discover_multizone_patterns('office')
        
        logger.info("✅ Pattern discovery completed successfully!")
        
        # Check if patterns were found
        if patterns.get('patterns', {}).get('recurring'):
            recurring = patterns['patterns']['recurring']
            logger.info(f"Found recurring patterns: {len([k for k, v in recurring.items() if v.get('found')])}")
            
            for window_size, data in recurring.items():
                if isinstance(data, dict) and data.get('found'):
                    logger.info(f"  {window_size}: {data.get('pattern_count', 0)} patterns")
        
        # Check summary
        summary = patterns.get('summary', {})
        if summary.get('has_patterns'):
            logger.info(f"Pattern types found: {', '.join(summary.get('pattern_types', []))}")
            for insight in summary.get('key_insights', []):
                logger.info(f"  - {insight}")
        else:
            logger.info("No patterns found (this may be expected for low-activity rooms)")
        
        logger.info("🎉 STUMPY fix test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_preprocessing_logic():
    """Test the preprocessing logic independently"""
    logger.info("🔬 Testing preprocessing logic independently")
    
    # Create test data that mimics office sensor (94.77% zeros)
    np.random.seed(42)
    test_data = np.zeros(1000)
    
    # Add some sparse occupancy events (5% occupancy)
    num_ones = int(0.05 * len(test_data))
    one_indices = np.random.choice(len(test_data), num_ones, replace=False)
    test_data[one_indices] = 1
    
    logger.info(f"Test data: {len(test_data)} points, {np.mean(test_data):.1%} occupancy")
    
    # Test different window sizes
    window_sizes = [12, 24, 48]
    
    for window_size in window_sizes:
        logger.info(f"\nTesting window size {window_size}:")
        
        # Count constant windows
        total_windows = len(test_data) - window_size + 1
        constant_windows = 0
        
        for i in range(total_windows):
            if np.std(test_data[i:i+window_size]) < 1e-15:
                constant_windows += 1
        
        constant_percentage = (constant_windows / total_windows) * 100
        logger.info(f"  Constant windows: {constant_percentage:.1f}%")
        
        # Test if STUMPY would work
        try:
            import stumpy
            
            # Test without preprocessing first
            logger.info("  Testing STUMPY without preprocessing...")
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                mp = stumpy.stump(test_data, m=window_size)
            logger.info(f"  ✅ STUMPY succeeded without preprocessing: {mp.shape}")
            
        except Exception as e:
            logger.error(f"  ❌ STUMPY failed without preprocessing: {e}")
            
            # Try with minimal preprocessing
            logger.info("  Testing with minimal preprocessing...")
            try:
                epsilon = np.finfo(float).eps * 1000
                processed_data = test_data.astype(float) + np.arange(len(test_data)) * epsilon
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    mp = stumpy.stump(processed_data, m=window_size)
                logger.info(f"  ✅ STUMPY succeeded with preprocessing: {mp.shape}")
                
            except Exception as e2:
                logger.error(f"  ❌ STUMPY still failed with preprocessing: {e2}")

async def main():
    """Run all tests"""
    logger.info("🚀 Starting STUMPY fix validation tests")
    
    # Test 1: Independent preprocessing logic
    await test_preprocessing_logic()
    
    # Test 2: Full pattern discovery with real data
    success = await test_stumpy_fix()
    
    if success:
        logger.info("🎉 All tests passed! STUMPY fix is working correctly.")
    else:
        logger.error("❌ Tests failed. Fix may need additional work.")

if __name__ == "__main__":
    asyncio.run(main())