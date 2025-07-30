#!/usr/bin/env python3
"""
STUMPY Division by Zero Investigation Script

This script examines the actual data patterns that cause zero standard deviation
in sliding windows, leading to division by zero errors in STUMPY's correlation calculations.

It investigates:
1. What data characteristics cause stddev to be zero
2. Why sliding window validation is failing  
3. Specific data patterns that trigger the issue
4. How to prevent zero stddev before STUMPY
"""

import numpy as np
import pandas as pd
import asyncio
from datetime import datetime, timedelta, timezone
import logging
from typing import List, Tuple, Dict, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config_loader import ConfigLoader
from src.storage.timeseries_db import TimescaleDBManager
from sqlalchemy import text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_sliding_window_stddev(data: np.ndarray, window_size: int) -> Dict[str, Any]:
    """
    Analyze standard deviation characteristics of sliding windows in data.
    
    This is the CORE function that reveals why stddev becomes zero.
    """
    if len(data) < window_size:
        return {"error": "Data too short for window size"}
    
    # Calculate sliding window standard deviations
    stddevs = []
    zero_stddev_windows = []
    constant_patterns = []
    
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        window_std = np.std(window)
        stddevs.append(window_std)
        
        # Track zero stddev windows and their patterns
        if window_std == 0.0:
            zero_stddev_windows.append(i)
            unique_vals = np.unique(window)
            constant_patterns.append({
                'start_idx': i,
                'value': unique_vals[0] if len(unique_vals) == 1 else 'mixed',
                'window': window.tolist()
            })
    
    stddevs = np.array(stddevs)
    
    # Comprehensive analysis
    analysis = {
        'total_windows': len(stddevs),
        'zero_stddev_count': np.sum(stddevs == 0.0),
        'zero_stddev_percentage': (np.sum(stddevs == 0.0) / len(stddevs)) * 100,
        'min_stddev': np.min(stddevs),
        'max_stddev': np.max(stddevs),
        'mean_stddev': np.mean(stddevs),
        'stddev_distribution': {
            'zero': np.sum(stddevs == 0.0),
            'very_small': np.sum((stddevs > 0) & (stddevs < 0.01)),
            'small': np.sum((stddevs >= 0.01) & (stddevs < 0.1)),
            'normal': np.sum(stddevs >= 0.1)
        },
        'zero_stddev_indices': zero_stddev_windows[:10],  # First 10 for examination
        'constant_patterns': constant_patterns[:5],  # First 5 patterns
        'data_characteristics': {
            'total_points': len(data),
            'unique_values': len(np.unique(data)),
            'value_distribution': {str(val): int(count) for val, count in zip(*np.unique(data, return_counts=True))},
            'zero_percentage': (np.sum(data == 0) / len(data)) * 100,
            'one_percentage': (np.sum(data == 1) / len(data)) * 100,
        }
    }
    
    return analysis

def simulate_stumpy_correlation_issue(data: np.ndarray, window_size: int) -> Dict[str, Any]:
    """
    Simulate the exact conditions that cause STUMPY's correlation calculation to fail.
    
    This reproduces the division by zero at:
    c /= stddev[:, None]  
    c /= stddev[None, :]
    """
    if len(data) < window_size:
        return {"error": "Data too short"}
    
    # Extract sliding windows (like STUMPY does)
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i + window_size])
    
    windows = np.array(windows)
    
    # Calculate standard deviation for each window (like STUMPY does)
    stddevs = np.std(windows, axis=1)
    
    # Identify problematic cases
    zero_stddev_mask = stddevs == 0.0
    problematic_windows = windows[zero_stddev_mask]
    
    # Simulate the division operation that fails
    division_errors = []
    if np.any(zero_stddev_mask):
        try:
            # This is what STUMPY tries to do
            dummy_c = np.ones((len(stddevs), len(stddevs)))
            with np.errstate(divide='raise', invalid='raise'):
                dummy_c /= stddevs[:, None]  # This line fails
        except (FloatingPointError, RuntimeWarning) as e:
            division_errors.append(f"Division by stddev[:, None] failed: {e}")
        
        try:
            dummy_c = np.ones((len(stddevs), len(stddevs)))
            with np.errstate(divide='raise', invalid='raise'):
                dummy_c /= stddevs[None, :]  # This line fails  
        except (FloatingPointError, RuntimeWarning) as e:
            division_errors.append(f"Division by stddev[None, :] failed: {e}")
    
    return {
        'total_windows': len(windows),
        'zero_stddev_count': np.sum(zero_stddev_mask),
        'zero_stddev_percentage': (np.sum(zero_stddev_mask) / len(windows)) * 100,
        'stddev_values': stddevs.tolist(),
        'problematic_windows': problematic_windows.tolist() if len(problematic_windows) > 0 else [],
        'division_errors': division_errors,
        'will_cause_stumpy_error': np.any(zero_stddev_mask)
    }

def examine_current_validation_logic() -> str:
    """
    Examine why our current sliding window validation is failing to catch the issue.
    """
    validation_analysis = """
    CURRENT VALIDATION LOGIC ANALYSIS:
    
    Our validation checks for:
    1. Constant windows (all values same)
    2. Percentage of constant windows > threshold
    
    POTENTIAL ISSUES:
    1. Sampling rate too low - we might miss problematic windows
    2. Threshold too permissive - allowing too many constant windows
    3. Edge cases - near-constant windows that still have zero stddev due to floating point precision
    4. Timing - validation on subset vs full data STUMPY processes
    
    INVESTIGATION NEEDED:
    - Are we validating the same data STUMPY processes?
    - Is our sampling missing clustered constant windows?
    - Are there floating point precision issues creating false negatives?
    """
    return validation_analysis

async def get_real_sensor_data(room: str = 'office', hours: int = 48) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Get real sensor data from the container to analyze actual problematic patterns.
    """
    try:
        config = ConfigLoader()
        db_config = config.get('database.timescale')
        
        db = TimescaleDBManager(
            f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        await db.initialize()
        
        # Get recent sensor data for the specified room
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        query = text("""
            SELECT timestamp, state, entity_id
            FROM sensor_events 
            WHERE room = :room
            AND entity_id LIKE '%presence%'
            AND timestamp >= :start_time
            AND timestamp <= :end_time
            ORDER BY timestamp ASC
        """)
        
        async with db.engine.begin() as conn:
            result = await conn.execute(query, {
                'room': room,
                'start_time': start_time,
                'end_time': end_time
            })
            
            rows = result.fetchall()
            
        await db.close()
        
        if not rows:
            logger.warning(f"No presence data found for room {room}")
            return np.array([]), {}
        
        # Convert to time series
        timestamps = [row[0] for row in rows]
        states = [row[1] for row in rows]  # 'on'/'off' values
        entity_ids = [row[2] for row in rows]
        
        # Convert on/off to 1/0 binary values
        binary_values = [1.0 if state == 'on' else 0.0 for state in states]
        data_array = np.array(binary_values)
        
        metadata = {
            'room': room,
            'entity_ids': list(set(entity_ids)),
            'start_time': start_time,
            'end_time': end_time,
            'total_points': len(data_array),
            'time_range_hours': hours
        }
        
        return data_array, metadata
        
    except Exception as e:
        logger.error(f"Failed to get sensor data: {e}")
        return np.array([]), {'error': str(e)}

def generate_test_patterns() -> Dict[str, np.ndarray]:
    """
    Generate specific test patterns that are known to cause zero standard deviation.
    """
    patterns = {
        'all_zeros': np.zeros(1000),
        'all_ones': np.ones(1000),
        'mostly_zeros_with_few_ones': np.concatenate([np.zeros(950), np.ones(50)]),
        'alternating_blocks': np.tile(np.concatenate([np.zeros(100), np.ones(100)]), 5),
        'long_constant_sequences': np.concatenate([
            np.zeros(300), np.ones(200), np.zeros(400), np.ones(100)
        ]),
        'typical_office_pattern': np.concatenate([
            np.zeros(800),  # Mostly unoccupied (like office)
            np.ones(50),    # Short occupied period
            np.zeros(100),  # Back to unoccupied
            np.ones(30),    # Another short period
            np.zeros(20)    # End unoccupied
        ])
    }
    
    return patterns

async def main():
    """
    Main investigation function - runs comprehensive analysis of the division by zero issue.
    """
    print("🔍 STUMPY Division by Zero Investigation")
    print("=" * 60)
    
    # 1. Examine current validation logic
    print("\n1. VALIDATION LOGIC ANALYSIS:")
    print(examine_current_validation_logic())
    
    # 2. Test with generated patterns
    print("\n2. TESTING WITH KNOWN PROBLEMATIC PATTERNS:")
    patterns = generate_test_patterns()
    
    for pattern_name, pattern_data in patterns.items():
        print(f"\n--- Pattern: {pattern_name} ---")
        
        # Test with different window sizes
        for window_size in [30, 60, 120]:  # 30min, 1hr, 2hr at 1-min resolution
            analysis = analyze_sliding_window_stddev(pattern_data, window_size)
            stumpy_sim = simulate_stumpy_correlation_issue(pattern_data, window_size)
            
            print(f"Window Size {window_size}min:")
            print(f"  Zero stddev windows: {analysis['zero_stddev_count']} ({analysis['zero_stddev_percentage']:.1f}%)")
            print(f"  Will cause STUMPY error: {stumpy_sim['will_cause_stumpy_error']}")
            
            if stumpy_sim['division_errors']:
                print(f"  Division errors: {stumpy_sim['division_errors']}")
    
    # 3. Test with real sensor data
    print("\n3. TESTING WITH REAL SENSOR DATA:")
    
    for room in ['office', 'bedroom', 'living_kitchen']:
        print(f"\n--- Room: {room} ---")
        
        try:
            data, metadata = await get_real_sensor_data(room, hours=48)
            
            if len(data) == 0:
                print(f"  No data available for {room}")
                continue
                
            print(f"  Data points: {metadata['total_points']}")
            print(f"  Entity IDs: {metadata.get('entity_ids', 'unknown')}")
            
            # Analyze with different window sizes
            for window_size in [30, 60, 120]:
                if len(data) >= window_size:
                    analysis = analyze_sliding_window_stddev(data, window_size)
                    stumpy_sim = simulate_stumpy_correlation_issue(data, window_size)
                    
                    print(f"  Window Size {window_size}min:")
                    print(f"    Zero stddev windows: {analysis['zero_stddev_count']} ({analysis['zero_stddev_percentage']:.1f}%)")
                    print(f"    Data characteristics: {analysis['data_characteristics']['zero_percentage']:.1f}% zeros, {analysis['data_characteristics']['one_percentage']:.1f}% ones")
                    print(f"    Will cause STUMPY error: {stumpy_sim['will_cause_stumpy_error']}")
                    
                    if analysis['zero_stddev_percentage'] > 50:
                        print(f"    ⚠️  HIGH RISK: {analysis['zero_stddev_percentage']:.1f}% zero stddev windows")
                        
                        # Show sample constant patterns
                        if analysis['constant_patterns']:
                            print(f"    Sample constant patterns:")
                            for i, pattern in enumerate(analysis['constant_patterns'][:3]):
                                print(f"      Pattern {i+1}: value={pattern['value']}, start={pattern['start_idx']}")
                    
        except Exception as e:
            print(f"  Error analyzing {room}: {e}")
    
    # 4. Recommendations
    print("\n4. ROOT CAUSE ANALYSIS & RECOMMENDATIONS:")
    print("=" * 50)
    
    recommendations = """
    ROOT CAUSE FINDINGS:
    
    1. ZERO STDDEV OCCURS WHEN:
       - Sliding windows contain only identical values (all 0s or all 1s)
       - This is common in binary occupancy data, especially for low-occupancy rooms
       - Office/bedroom sensors can have 80%+ constant windows during night/weekend periods
    
    2. CURRENT VALIDATION FAILURES:
       - Sampling rate may miss clustered constant windows
       - Thresholds may be too permissive for binary data
       - Validation timing doesn't match STUMPY's exact processing
    
    3. DATA-LEVEL PREPROCESSING SOLUTIONS:
       - Add minimal noise to break constant sequences (preserve binary semantics)
       - Use time-based features (trend, position) to add variation
       - Implement smart window selection to skip highly constant periods
       - Use adaptive thresholds based on data characteristics
    
    4. STUMPY-SPECIFIC SOLUTIONS:
       - Pre-filter data to ensure minimum variation in all windows
       - Use robust correlation methods that handle zero variance
       - Implement fallback algorithms for constant data regions
    
    RECOMMENDED FIXES:
    1. Implement adaptive noise injection based on constant window percentage
    2. Use time-index based epsilon addition for deterministic variation
    3. Skip pattern discovery for data with >90% constant windows
    4. Add comprehensive logging of stddev distributions before STUMPY
    """
    
    print(recommendations)

if __name__ == "__main__":
    asyncio.run(main())