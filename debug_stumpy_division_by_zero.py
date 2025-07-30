#!/usr/bin/env python3
"""
Debug script to investigate STUMPY division by zero errors

This script examines the exact data patterns that cause stddev to be zero
in STUMPY's correlation calculations, leading to division by zero errors.

The error occurs at these lines in STUMPY:
    c /= stddev[:, None]
    c /= stddev[None, :]

This means `stddev` contains zero values despite our validation.
"""

import logging
import numpy as np
import pandas as pd
import warnings
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Tuple, Optional

# Configure logging to see all details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import STUMPY with warnings enabled to see exactly where the error occurs
import stumpy

async def load_real_sensor_data(room_name: str = 'office') -> pd.DataFrame:
    """Load real sensor data from the container to reproduce the issue"""
    from src.storage.timeseries_db import TimescaleDBManager
    from config.config_loader import ConfigLoader
    
    config = ConfigLoader()
    db_config = config.get("database.timescale")
    db = TimescaleDBManager(
        f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@"
        f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    
    await db.initialize()
    
    try:
        async with db.engine.begin() as conn:
            from sqlalchemy import text
            
            # Get presence sensor data for office (most likely to have issues)
            query = """
                SELECT timestamp, entity_id, state, attributes
                FROM sensor_events 
                WHERE entity_id = 'binary_sensor.office_presence_full_office'
                AND timestamp >= NOW() - INTERVAL '30 days'
                ORDER BY timestamp DESC
                LIMIT 10000
            """
            
            result = await conn.execute(text(query))
            events = [dict(row._mapping) for row in result.fetchall()]
            
            if not events:
                logger.warning(f"No events found for office presence sensor")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(events)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['occupied'] = df['state'].isin(['on', 'occupied', '1', 1]).astype(int)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Loaded {len(df)} events from {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
    finally:
        await db.close()

def analyze_data_characteristics(df: pd.DataFrame) -> Dict:
    """Analyze the characteristics of the data that might cause issues"""
    logger.info("=== ANALYZING DATA CHARACTERISTICS ===")
    
    if df.empty:
        return {'error': 'No data to analyze'}
    
    # Create time series
    ts = df.set_index('timestamp')['occupied'].resample('5min').max().fillna(0)
    data_values = ts.values
    
    analysis = {
        'total_points': len(data_values),
        'unique_values': len(np.unique(data_values)),
        'value_range': (float(data_values.min()), float(data_values.max())),
        'mean': float(np.mean(data_values)),
        'std': float(np.std(data_values)),
        'zero_count': int(np.sum(data_values == 0)),
        'one_count': int(np.sum(data_values == 1)),
        'zero_percentage': float(np.sum(data_values == 0) / len(data_values) * 100),
        'transitions': int(np.sum(np.abs(np.diff(data_values)))),
        'transition_rate': float(np.sum(np.abs(np.diff(data_values))) / len(data_values))
    }
    
    logger.info(f"Data characteristics:")
    logger.info(f"  Total points: {analysis['total_points']}")
    logger.info(f"  Unique values: {analysis['unique_values']}")
    logger.info(f"  Value range: {analysis['value_range']}")
    logger.info(f"  Mean: {analysis['mean']:.6f}")
    logger.info(f"  Std: {analysis['std']:.6f}")
    logger.info(f"  Zero percentage: {analysis['zero_percentage']:.2f}%")
    logger.info(f"  Transition rate: {analysis['transition_rate']:.4f}")
    
    return analysis

def find_constant_sliding_windows(data_values: np.ndarray, window_size: int) -> Dict:
    """Find exactly which sliding windows have zero standard deviation"""
    logger.info(f"=== ANALYZING SLIDING WINDOWS (size={window_size}) ===")
    
    total_windows = len(data_values) - window_size + 1
    constant_windows = []
    zero_std_windows = []
    
    # Check every window (not just a sample)
    for i in range(total_windows):
        window_data = data_values[i:i+window_size]
        window_std = np.std(window_data)
        window_var = np.var(window_data)
        
        if window_std == 0.0:
            constant_windows.append({
                'start_idx': i,
                'end_idx': i + window_size - 1,
                'values': window_data.tolist(),
                'std': window_std,
                'var': window_var
            })
        elif window_std < 1e-12:  # Near-zero std
            zero_std_windows.append({
                'start_idx': i,
                'end_idx': i + window_size - 1,
                'values': window_data.tolist(),
                'std': window_std,
                'var': window_var
            })
    
    constant_percentage = len(constant_windows) / total_windows * 100
    near_zero_percentage = len(zero_std_windows) / total_windows * 100
    
    logger.info(f"Window analysis results:")
    logger.info(f"  Total windows: {total_windows}")
    logger.info(f"  Constant windows (std=0): {len(constant_windows)} ({constant_percentage:.2f}%)")
    logger.info(f"  Near-zero std windows: {len(zero_std_windows)} ({near_zero_percentage:.2f}%)")
    
    # Show examples of problematic windows
    if constant_windows:
        logger.info(f"Examples of constant windows:")
        for i, window_info in enumerate(constant_windows[:5]):
            logger.info(f"    Window {i+1}: indices {window_info['start_idx']}-{window_info['end_idx']}, "
                       f"values: {window_info['values'][:10]}...")
    
    if zero_std_windows:
        logger.info(f"Examples of near-zero std windows:")
        for i, window_info in enumerate(zero_std_windows[:5]):
            logger.info(f"    Window {i+1}: indices {window_info['start_idx']}-{window_info['end_idx']}, "
                       f"std: {window_info['std']:.2e}")
    
    return {
        'total_windows': total_windows,
        'constant_windows': len(constant_windows),
        'near_zero_windows': len(zero_std_windows),
        'constant_percentage': constant_percentage,
        'near_zero_percentage': near_zero_percentage,
        'constant_examples': constant_windows[:5],
        'near_zero_examples': zero_std_windows[:5]
    }

def test_stumpy_with_problematic_data(data_values: np.ndarray, window_size: int) -> Dict:
    """Test STUMPY with the actual data that causes division by zero"""
    logger.info(f"=== TESTING STUMPY WITH PROBLEMATIC DATA ===")
    
    # Enable all numpy warnings to see exactly where the error occurs
    np.seterr(all='warn')
    warnings.filterwarnings('error', category=RuntimeWarning, message='.*divide by zero.*')
    warnings.filterwarnings('error', category=RuntimeWarning, message='.*invalid value.*')
    
    try:
        logger.info(f"Running STUMPY with {len(data_values)} points, window size {window_size}")
        logger.info(f"Data std: {np.std(data_values):.10f}")
        logger.info(f"Data range: {data_values.min()} to {data_values.max()}")
        
        # This should trigger the division by zero warning/error
        mp = stumpy.stump(data_values, m=window_size)
        
        logger.info(f"STUMPY completed successfully! Matrix profile shape: {mp.shape}")
        return {'success': True, 'matrix_profile_shape': mp.shape}
        
    except Warning as w:
        logger.error(f"STUMPY warning caught: {w}")
        return {'success': False, 'error_type': 'warning', 'error': str(w)}
    except Exception as e:
        logger.error(f"STUMPY error caught: {e}")
        return {'success': False, 'error_type': 'exception', 'error': str(e)}
    finally:
        # Reset numpy error handling
        np.seterr(all='ignore')
        warnings.resetwarnings()

def analyze_stumpy_correlation_calculation(data_values: np.ndarray, window_size: int) -> Dict:
    """
    Manually replicate STUMPY's correlation calculation to see where division by zero occurs
    This mimics the internal STUMPY logic that causes the error
    """
    logger.info(f"=== MANUAL CORRELATION ANALYSIS ===")
    
    n = len(data_values)
    if n < window_size * 2:
        return {'error': 'Insufficient data for analysis'}
    
    # Extract sliding windows (similar to what STUMPY does internally)
    windows = []
    for i in range(n - window_size + 1):
        windows.append(data_values[i:i+window_size])
    
    windows = np.array(windows)
    logger.info(f"Created {len(windows)} sliding windows")
    
    # Calculate standard deviation for each window (this is where STUMPY fails)
    stddevs = np.std(windows, axis=1, ddof=0)  # STUMPY uses ddof=0
    
    # Find problematic standard deviations
    zero_stddevs = np.sum(stddevs == 0.0)
    near_zero_stddevs = np.sum(stddevs < 1e-12)
    min_stddev = np.min(stddevs)
    max_stddev = np.max(stddevs)
    
    logger.info(f"Standard deviation analysis:")
    logger.info(f"  Zero stddevs: {zero_stddevs}")
    logger.info(f"  Near-zero stddevs (<1e-12): {near_zero_stddevs}")
    logger.info(f"  Min stddev: {min_stddev:.2e}")
    logger.info(f"  Max stddev: {max_stddev:.2e}")
    
    # Show examples of windows with zero standard deviation
    zero_indices = np.where(stddevs == 0.0)[0]
    if len(zero_indices) > 0:
        logger.info(f"Windows with zero standard deviation:")
        for i, idx in enumerate(zero_indices[:10]):  # Show first 10
            window_data = windows[idx]
            logger.info(f"  Window {idx}: {window_data.tolist()}")
    
    # Test the division that causes the error
    problematic_divisions = []
    if zero_stddevs > 0:
        logger.info("Testing division operations that cause STUMPY to fail:")
        
        # Simulate STUMPY's correlation normalization
        for i, stddev in enumerate(stddevs):
            if stddev == 0.0:
                try:
                    # This is the operation that fails in STUMPY
                    result = 1.0 / stddev
                    logger.info(f"  Division 1/stddev[{i}] = 1/{stddev} = {result}")
                except ZeroDivisionError:
                    logger.error(f"  Division by zero at window {i}: 1/{stddev}")
                    problematic_divisions.append(i)
    
    return {
        'total_windows': len(windows),
        'zero_stddevs': int(zero_stddevs),
        'near_zero_stddevs': int(near_zero_stddevs),
        'min_stddev': float(min_stddev),
        'max_stddev': float(max_stddev),
        'zero_indices': zero_indices.tolist()[:20],  # First 20 problematic windows
        'problematic_divisions': problematic_divisions
    }

def propose_data_preprocessing_fixes(data_values: np.ndarray, analysis_results: Dict) -> Dict:
    """Propose specific data preprocessing fixes to prevent zero stddev"""
    logger.info(f"=== PROPOSING PREPROCESSING FIXES ===")
    
    fixes = {}
    
    # Fix 1: Add minimal noise to break constant sequences
    epsilon = np.finfo(float).eps * 100  # Small deterministic offset
    fix1_data = data_values.astype(float) + np.arange(len(data_values)) * epsilon
    fix1_std = np.std(fix1_data)
    fixes['deterministic_noise'] = {
        'description': 'Add deterministic minimal noise (index * epsilon)',
        'epsilon': float(epsilon),
        'resulting_std': float(fix1_std),
        'preserves_binary': np.allclose(fix1_data, np.round(fix1_data), atol=1e-10)
    }
    
    # Fix 2: Add tiny random noise
    noise_level = max(1e-15, np.std(data_values) * 1e-8)
    np.random.seed(42)  # Reproducible
    fix2_data = data_values.astype(float) + np.random.normal(0, noise_level, size=len(data_values))
    fix2_std = np.std(fix2_data)
    fixes['random_noise'] = {
        'description': 'Add tiny Gaussian noise',
        'noise_level': float(noise_level),
        'resulting_std': float(fix2_std),
        'preserves_binary': np.allclose(fix2_data, np.round(fix2_data), atol=1e-6)
    }
    
    # Fix 3: Interpolation-based smoothing for constant sequences
    fix3_data = data_values.copy().astype(float)
    # Find constant sequences and add minimal variation
    diff = np.diff(np.concatenate(([0], data_values, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    constant_sequences_fixed = 0
    for start, end in zip(starts, ends):
        if end - start > 10:  # Only fix long constant sequences
            # Add minimal linear trend within the sequence
            sequence_length = end - start
            linear_trend = np.linspace(0, epsilon * sequence_length, sequence_length)
            fix3_data[start:end] += linear_trend
            constant_sequences_fixed += 1
    
    fix3_std = np.std(fix3_data)
    fixes['linear_interpolation'] = {
        'description': 'Add linear trends to long constant sequences',
        'sequences_fixed': constant_sequences_fixed,
        'resulting_std': float(fix3_std),
        'preserves_binary': np.allclose(fix3_data, np.round(fix3_data), atol=1e-10)
    }
    
    # Test each fix with a small window
    test_window_size = min(12, len(data_values) // 4)
    for fix_name, fix_info in fixes.items():
        if fix_name == 'deterministic_noise':
            test_data = fix1_data
        elif fix_name == 'random_noise':
            test_data = fix2_data
        else:
            test_data = fix3_data
        
        # Check if this fix eliminates zero stddev windows
        window_analysis = find_constant_sliding_windows(test_data, test_window_size)
        fix_info['eliminates_zero_stddev'] = window_analysis['constant_windows'] == 0
        fix_info['test_window_size'] = test_window_size
        fix_info['remaining_constant_windows'] = window_analysis['constant_windows']
    
    # Recommend the best fix
    best_fix = None
    for fix_name, fix_info in fixes.items():
        if fix_info['eliminates_zero_stddev'] and fix_info['preserves_binary']:
            best_fix = fix_name
            break
    
    fixes['recommendation'] = {
        'best_fix': best_fix,
        'reason': 'Eliminates zero stddev while preserving binary semantics' if best_fix else 'No ideal fix found'
    }
    
    logger.info(f"Preprocessing fix analysis:")
    for fix_name, fix_info in fixes.items():
        if fix_name != 'recommendation':
            logger.info(f"  {fix_name}: {fix_info['description']}")
            logger.info(f"    Eliminates zero stddev: {fix_info.get('eliminates_zero_stddev', 'Unknown')}")
            logger.info(f"    Preserves binary: {fix_info.get('preserves_binary', 'Unknown')}")
    
    if best_fix:
        logger.info(f"RECOMMENDATION: Use {best_fix}")
    else:
        logger.warning("No ideal preprocessing fix found")
    
    return fixes

async def main():
    """Main debug function to investigate STUMPY division by zero"""
    logger.info("🔍 Starting STUMPY division by zero investigation")
    
    # Step 1: Load real sensor data
    logger.info("Step 1: Loading real sensor data...")
    df = await load_real_sensor_data('office')
    
    if df.empty:
        logger.error("No data loaded - cannot proceed with analysis")
        return
    
    # Step 2: Analyze data characteristics
    logger.info("Step 2: Analyzing data characteristics...")
    data_analysis = analyze_data_characteristics(df)
    
    # Convert to time series
    ts = df.set_index('timestamp')['occupied'].resample('5min').max().fillna(0)
    data_values = ts.values
    
    # Step 3: Test different window sizes that are likely to cause issues
    problematic_windows = [12, 24, 48, 144]  # 1h, 2h, 4h, 12h in 5-min intervals
    
    all_results = {
        'data_analysis': data_analysis,
        'window_tests': {}
    }
    
    for window_size in problematic_windows:
        if len(data_values) < window_size * 2:
            logger.info(f"Skipping window size {window_size} - insufficient data")
            continue
        
        logger.info(f"Step 3.{window_size}: Testing window size {window_size}")
        
        # Analyze sliding windows
        window_analysis = find_constant_sliding_windows(data_values, window_size)
        
        # Test STUMPY with this data
        stumpy_results = test_stumpy_with_problematic_data(data_values, window_size)
        
        # Analyze correlation calculation manually
        correlation_analysis = analyze_stumpy_correlation_calculation(data_values, window_size)
        
        all_results['window_tests'][window_size] = {
            'window_analysis': window_analysis,
            'stumpy_results': stumpy_results,
            'correlation_analysis': correlation_analysis
        }
        
        # If we found the problematic case, analyze preprocessing fixes
        if window_analysis['constant_windows'] > 0:
            logger.info(f"Found problematic data with window size {window_size}!")
            preprocessing_fixes = propose_data_preprocessing_fixes(data_values, all_results)
            all_results['preprocessing_fixes'] = preprocessing_fixes
            break
    
    # Step 4: Generate summary report
    logger.info("🎯 SUMMARY REPORT")
    logger.info("=" * 60)
    
    logger.info(f"Data characteristics:")
    logger.info(f"  - {data_analysis['total_points']} data points")
    logger.info(f"  - {data_analysis['zero_percentage']:.1f}% zeros (unoccupied)")
    logger.info(f"  - {data_analysis['transition_rate']:.4f} transition rate")
    logger.info(f"  - Overall std: {data_analysis['std']:.6f}")
    
    for window_size, results in all_results['window_tests'].items():
        window_info = results['window_analysis']
        stumpy_info = results['stumpy_results']
        
        logger.info(f"Window size {window_size}:")
        logger.info(f"  - {window_info['constant_percentage']:.2f}% constant windows")
        logger.info(f"  - STUMPY success: {stumpy_info['success']}")
        if not stumpy_info['success']:
            logger.info(f"  - Error: {stumpy_info['error']}")
    
    if 'preprocessing_fixes' in all_results:
        best_fix = all_results['preprocessing_fixes']['recommendation']['best_fix']
        if best_fix:
            logger.info(f"RECOMMENDED FIX: {best_fix}")
        else:
            logger.info("NO IDEAL FIX FOUND")
    
    logger.info("=" * 60)
    logger.info("🔍 Investigation complete!")

if __name__ == "__main__":
    asyncio.run(main())