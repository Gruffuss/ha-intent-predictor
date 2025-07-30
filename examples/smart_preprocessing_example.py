"""
Example usage of Smart Data Preprocessor for STUMPY Division by Zero Prevention
Demonstrates how to use the adaptive preprocessing system with different occupancy patterns
"""

import sys
import os
import numpy as np
import asyncio

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from learning.smart_data_preprocessor import SmartDataPreprocessor
from learning.preprocessing_monitor import PreprocessingMonitor
from learning.pattern_discovery import PatternDiscovery


def create_bedroom_data(length: int = 2000) -> np.ndarray:
    """Create bedroom-type data: 97.5% zeros, causing STUMPY division by zero"""
    np.random.seed(42)
    data = np.zeros(length)
    
    # Add sparse occupancy periods
    num_occupied = int(length * 0.025)  # 2.5% occupancy
    occupied_indices = np.random.choice(length, size=num_occupied, replace=False)
    data[occupied_indices] = 1
    
    # Add a few longer occupied periods to be more realistic
    for _ in range(3):
        start = np.random.randint(0, length - 50)
        duration = np.random.randint(10, 30)
        data[start:start+duration] = 1
    
    return data


def create_office_data(length: int = 2000) -> np.ndarray:
    """Create office-type data: 49.3% zeros, should be safer for STUMPY"""
    np.random.seed(123)
    
    # Create more balanced but realistic office occupancy
    data = np.zeros(length)
    current_pos = 0
    
    while current_pos < length:
        # Work session (occupied)
        work_duration = np.random.randint(60, 180)  # 1-3 hours
        end_work = min(current_pos + work_duration, length)
        data[current_pos:end_work] = 1
        current_pos = end_work
        
        # Break (unoccupied)
        break_duration = np.random.randint(10, 60)  # 10min-1hour break
        current_pos += break_duration
    
    return data


def create_living_room_data(length: int = 2000) -> np.ndarray:
    """Create living room data: balanced with good transitions"""
    np.random.seed(456)
    data = np.zeros(length)
    
    # Create realistic living room patterns with natural transitions
    current_pos = 0
    
    while current_pos < length:
        # Occupied period
        occupied_duration = np.random.randint(20, 120)  # 20min-2hours
        end_occupied = min(current_pos + occupied_duration, length)
        data[current_pos:end_occupied] = 1
        current_pos = end_occupied
        
        # Unoccupied period
        unoccupied_duration = np.random.randint(5, 60)  # 5min-1hour
        current_pos += unoccupied_duration
    
    return data


async def demonstrate_smart_preprocessing():
    """Demonstrate the smart preprocessing system"""
    print("🚀 Smart Data Preprocessor Demonstration")
    print("="*60)
    
    # Initialize preprocessor and monitor
    preprocessor = SmartDataPreprocessor()
    monitor = PreprocessingMonitor(enable_detailed_logging=True)
    
    # Test different room types
    test_cases = [
        ("bedroom", create_bedroom_data(), "High risk: 97.5% zeros"),
        ("office", create_office_data(), "Medium risk: 49.3% zeros"),
        ("living_room", create_living_room_data(), "Low risk: balanced occupancy")
    ]
    
    window_size = 48  # 4-hour windows (5-minute intervals)
    
    for room_name, data, description in test_cases:
        print(f"\n📍 Testing {room_name.upper()}: {description}")
        print("-" * 50)
        
        # Basic statistics
        occupancy_rate = np.mean(data)
        transitions = np.sum(np.abs(np.diff(data)))
        transition_rate = transitions / len(data)
        
        print(f"Original data statistics:")
        print(f"  Length: {len(data):,} points")
        print(f"  Occupancy rate: {occupancy_rate:.3f} ({occupancy_rate:.1%})")
        print(f"  Transition rate: {transition_rate:.4f}")
        print(f"  Std deviation: {np.std(data):.8f}")
        
        # Apply smart preprocessing with monitoring
        try:
            processed_data, info, monitoring_data = await monitor.monitor_preprocessing(
                room_name, data, window_size, preprocessor
            )
            
            print(f"\nSmart preprocessing results:")
            print(f"  Method applied: {info['method']}")
            print(f"  Pattern type: {info['pattern_type']}")
            print(f"  Preprocessing needed: {info['preprocessing_applied']}")
            print(f"  STUMPY ready: {info['ready_for_stumpy']}")
            print(f"  Processing time: {monitoring_data['processing_time']:.3f}s")
            
            if info['preprocessing_applied']:
                print(f"  Std improvement: {info['original_std']:.8f} → {info['processed_std']:.8f}")
                
                # Verify binary semantics preservation
                max_deviation = 0
                for orig, proc in zip(data, processed_data):
                    if orig == 0:
                        max_deviation = max(max_deviation, abs(proc))
                    else:
                        max_deviation = max(max_deviation, abs(proc - 1))
                        
                print(f"  Max binary deviation: {max_deviation:.8f}")
                print(f"  Binary semantics preserved: {max_deviation < 0.1}")
            
            # Test STUMPY compatibility (if available)
            try:
                import stumpy
                
                print(f"\n🧪 Testing STUMPY compatibility...")
                
                # Try to run STUMPY on processed data
                mp = stumpy.stump(processed_data, m=window_size)
                
                # Check for division by zero indicators
                has_nan = np.any(np.isnan(mp))
                has_inf = np.any(np.isinf(mp))
                
                print(f"  STUMPY execution: ✅ SUCCESS")
                print(f"  Matrix profile shape: {mp.shape}")
                print(f"  Contains NaN: {has_nan}")
                print(f"  Contains Inf: {has_inf}")
                
                if has_nan or has_inf:
                    print(f"  ⚠️ WARNING: Matrix profile contains invalid values")
                else:
                    print(f"  ✅ Matrix profile looks healthy")
                    
                    # Find top patterns
                    min_distance_idx = np.argmin(mp[:, 0])
                    min_distance = mp[min_distance_idx, 0]
                    print(f"  Best pattern distance: {min_distance:.6f} at index {min_distance_idx}")
                
            except ImportError:
                print(f"  ℹ️ STUMPY not available for compatibility test")
            except Exception as e:
                print(f"  ❌ STUMPY failed: {e}")
                
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
    
    # Show overall session metrics
    print(f"\n📊 Session Summary")
    print("="*60)
    
    metrics = monitor.get_metrics_summary()
    if not metrics.get('no_data'):
        print(f"Total preprocessing operations: {metrics['total_requests']}")
        print(f"Preprocessing rate: {metrics['preprocessing_rate']:.1%}")
        print(f"STUMPY ready rate: {metrics['stumpy_ready_rate']:.1%}")
        print(f"Error rate: {metrics['error_rate']:.1%}")
        print(f"Average processing time: {metrics['avg_processing_time']:.3f}s")
        
        print(f"\nMethod usage:")
        for method, count in metrics['method_usage'].items():
            print(f"  {method}: {count}")
            
        print(f"\nPattern distribution:")
        for pattern, count in metrics['pattern_distribution'].items():
            print(f"  {pattern}: {count}")
            
        print(f"\nKey insights:")
        for insight in metrics.get('insights', []):
            print(f"  • {insight}")
    
    print(f"\n✅ Smart preprocessing demonstration complete!")


async def demonstrate_pattern_discovery_integration():
    """Demonstrate integration with pattern discovery"""
    print(f"\n🔍 Pattern Discovery Integration")
    print("="*60)
    
    # Create a pattern discovery instance
    pattern_discovery = PatternDiscovery()
    
    # This would normally load from database, but we'll simulate
    bedroom_data = create_bedroom_data(1000)
    
    # Create mock events
    events = []
    for i, value in enumerate(bedroom_data):
        events.append({
            'timestamp': f"2024-01-01 00:{i//12:02d}:{(i%12)*5:02d}",  # 5-minute intervals
            'entity_id': 'binary_sensor.bedroom_presence_sensor_full_bedroom',
            'state': 'on' if value == 1 else 'off',
            'room': 'bedroom',
            'sensor_type': 'occupancy'
        })
    
    print(f"Simulated {len(events)} events for bedroom pattern discovery...")
    
    # Run pattern discovery (this will use smart preprocessing internally)
    result = await pattern_discovery.discover_multizone_patterns('bedroom', events)
    
    print(f"\nPattern discovery results:")
    print(f"  Room: {result['room']}")
    print(f"  Total events: {result['total_events']:,}")
    print(f"  Patterns discovered: {result['summary']['has_patterns']}")
    
    if 'preprocessing_metrics' in result:
        metrics = result['preprocessing_metrics']
        if not metrics.get('no_data'):
            print(f"  Preprocessing operations: {metrics['total_requests']}")
            print(f"  STUMPY ready rate: {metrics['stumpy_ready_rate']:.1%}")
    
    # Show pattern details
    patterns = result['patterns']
    for window_type, pattern_data in patterns.items():
        if isinstance(pattern_data, dict) and 'preprocessing_info' in pattern_data:
            preprocessing_info = pattern_data['preprocessing_info']
            print(f"\n  {window_type} window:")
            print(f"    Preprocessing: {preprocessing_info.get('method', 'N/A')}")
            print(f"    Pattern found: {pattern_data.get('found', False)}")
            if pattern_data.get('found'):
                print(f"    Pattern count: {pattern_data.get('pattern_count', 0)}")


if __name__ == "__main__":
    # Run the demonstrations
    asyncio.run(demonstrate_smart_preprocessing())
    asyncio.run(demonstrate_pattern_discovery_integration())