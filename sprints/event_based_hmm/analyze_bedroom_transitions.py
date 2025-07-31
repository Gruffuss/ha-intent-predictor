#!/usr/bin/env python3
"""
Bedroom Presence Sensor Transition Analysis for Event-Based HMM Implementation

This script analyzes state transitions of the bedroom presence sensor to understand
patterns suitable for HMM training, focusing on TRUE state changes rather than time-series data.

Target sensor: binary_sensor.bedroom_presence_sensor_full_bedroom
Analysis focus: occupied→unoccupied and unoccupied→occupied patterns
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# Add src directory to path
sys.path.append('/opt/ha-intent-predictor/src')
sys.path.append('/opt/ha-intent-predictor')

from src.storage.timeseries_db import TimescaleDBManager
from config.config_loader import ConfigLoader

async def analyze_bedroom_transitions():
    """Analyze bedroom presence sensor state transitions for HMM training"""
    
    print("🔍 BEDROOM PRESENCE SENSOR TRANSITION ANALYSIS")
    print("=" * 60)
    
    # Initialize database connection
    config = ConfigLoader()
    db_config = config.get("database.timescale")
    db = TimescaleDBManager(f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
    await db.initialize()
    
    try:
        from sqlalchemy import text
        
        # Step 1: Find all bedroom presence sensors
        print("1. DISCOVERING BEDROOM PRESENCE SENSORS")
        print("-" * 40)
        
        async with db.engine.begin() as conn:
            result = await conn.execute(text("""
                SELECT DISTINCT entity_id, COUNT(*) as event_count
                FROM sensor_events 
                WHERE entity_id ILIKE '%bedroom%presence%'
                   OR entity_id ILIKE '%bedroom%occupancy%'
                   OR (entity_id ILIKE '%bedroom%' AND entity_id ILIKE '%sensor%')
                GROUP BY entity_id
                ORDER BY event_count DESC
            """))
            
            sensors = result.fetchall()
            
            if not sensors:
                print("❌ No bedroom presence sensors found!")
                return
            
            print(f"Found {len(sensors)} bedroom-related sensors:")
            for sensor in sensors:
                print(f"  • {sensor[0]}: {sensor[1]:,} events")
        
        # Step 2: Focus on the main bedroom presence sensor
        target_sensor = "binary_sensor.bedroom_presence_sensor_full_bedroom"
        
        # Check if our target sensor exists, otherwise use the most active one
        sensor_ids = [s[0] for s in sensors]
        if target_sensor not in sensor_ids:
            target_sensor = sensors[0][0]  # Use most active sensor
            print(f"\n⚠️  Target sensor not found, using: {target_sensor}")
        else:
            print(f"\n✅ Found target sensor: {target_sensor}")
            
        # If the primary sensor has no transitions, try alternatives
        async with db.engine.begin() as conn:
            result = await conn.execute(text("""
                SELECT COUNT(*) as transition_count
                FROM sensor_events 
                WHERE entity_id = :sensor_id
                  AND state != previous_state
                  AND state IS NOT NULL
                  AND previous_state IS NOT NULL
            """), {"sensor_id": target_sensor})
            
            transition_count = result.fetchone()[0]
            
            if transition_count == 0:
                print(f"⚠️  {target_sensor} has no state transitions, trying alternatives...")
                
                # Try other bedroom sensors in order of preference
                alternatives = [
                    "binary_sensor.bedroom_vladimir_bed_side",
                    "binary_sensor.bedroom_floor", 
                    "binary_sensor.bedroom_presence_sensor_anca_bed_side"
                ]
                
                for alt_sensor in alternatives:
                    if alt_sensor in sensor_ids:
                        result = await conn.execute(text("""
                            SELECT COUNT(*) as transition_count
                            FROM sensor_events 
                            WHERE entity_id = :sensor_id
                              AND state != previous_state
                              AND state IS NOT NULL
                              AND previous_state IS NOT NULL
                        """), {"sensor_id": alt_sensor})
                        
                        alt_transitions = result.fetchone()[0]
                        if alt_transitions > 0:
                            target_sensor = alt_sensor
                            print(f"✅ Using {target_sensor} with {alt_transitions:,} transitions")
                            break
                
                if transition_count == 0:
                    print("❌ No bedroom sensors with state transitions found!")
                    return
        
        # Step 3: Analyze state transitions
        print(f"\n2. ANALYZING STATE TRANSITIONS FOR: {target_sensor}")
        print("-" * 60)
        
        async with db.engine.begin() as conn:
            # Get all state changes (where state != previous_state)
            result = await conn.execute(text("""
                SELECT timestamp, state, previous_state
                FROM sensor_events 
                WHERE entity_id = :sensor_id
                  AND state != previous_state
                  AND state IS NOT NULL
                  AND previous_state IS NOT NULL
                ORDER BY timestamp ASC
            """), {"sensor_id": target_sensor})
            
            transitions = result.fetchall()
            
            if not transitions:
                print("❌ No state transitions found!")
                return
            
            print(f"Total state transitions: {len(transitions):,}")
            
            # Step 4: Analyze transition patterns
            print(f"\n3. TRANSITION PATTERN ANALYSIS")
            print("-" * 40)
            
            # Count transition types
            transition_counts = Counter()
            daily_transitions = defaultdict(int)
            hourly_distribution = defaultdict(int)
            
            valid_transitions = []
            
            for i, (timestamp, current_state, previous_state) in enumerate(transitions):
                # Normalize states to consistent format
                prev_normalized = str(previous_state).lower() in ['true', 'on', '1', 'occupied']
                curr_normalized = str(current_state).lower() in ['true', 'on', '1', 'occupied']
                
                # Create transition key
                transition_type = f"{'occupied' if prev_normalized else 'unoccupied'} → {'occupied' if curr_normalized else 'unoccupied'}"
                transition_counts[transition_type] += 1
                
                # Track daily and hourly patterns
                day_key = timestamp.date()
                daily_transitions[day_key] += 1
                hourly_distribution[timestamp.hour] += 1
                
                valid_transitions.append({
                    'timestamp': timestamp,
                    'previous_state': prev_normalized,
                    'current_state': curr_normalized,
                    'transition_type': transition_type
                })
            
            # Display transition type counts
            print("Transition Type Distribution:")
            for transition_type, count in transition_counts.most_common():
                percentage = (count / len(transitions)) * 100
                print(f"  • {transition_type}: {count:,} ({percentage:.1f}%)")
            
            # Step 5: Time-based analysis
            print(f"\n4. TEMPORAL DISTRIBUTION ANALYSIS")
            print("-" * 40)
            
            # Daily transition frequency
            daily_avg = sum(daily_transitions.values()) / len(daily_transitions)
            print(f"Average daily transitions: {daily_avg:.1f}")
            print(f"Days with data: {len(daily_transitions)}")
            
            # Hourly distribution
            print("\nHourly Distribution (top 8 hours):")
            for hour, count in Counter(hourly_distribution).most_common(8):
                percentage = (count / len(transitions)) * 100
                print(f"  • {hour:02d}:00: {count:,} transitions ({percentage:.1f}%)")
            
            # Step 6: Sample transition sequences
            print(f"\n5. SAMPLE TRANSITION SEQUENCES")
            print("-" * 40)
            
            # Show first 10 transitions
            print("First 10 transitions (chronological):")
            for i, t in enumerate(valid_transitions[:10]):
                state_change = "🟢 OCCUPIED" if t['current_state'] else "🔴 UNOCCUPIED"
                print(f"  {i+1:2d}. {t['timestamp']}: {t['transition_type']} → {state_change}")
            
            print("\nMost recent 10 transitions:")
            for i, t in enumerate(valid_transitions[-10:]):
                state_change = "🟢 OCCUPIED" if t['current_state'] else "🔴 UNOCCUPIED"
                print(f"  {i+1:2d}. {t['timestamp']}: {t['transition_type']} → {state_change}")
            
            # Step 7: HMM Training Data Assessment
            print(f"\n6. HMM TRAINING DATA ASSESSMENT")
            print("-" * 40)
            
            # Calculate sequence lengths between transitions
            if len(valid_transitions) > 1:
                sequence_lengths = []
                for i in range(1, len(valid_transitions)):
                    time_diff = valid_transitions[i]['timestamp'] - valid_transitions[i-1]['timestamp']
                    sequence_lengths.append(time_diff.total_seconds() / 60)  # Convert to minutes
                
                if sequence_lengths:
                    avg_duration = sum(sequence_lengths) / len(sequence_lengths)
                    min_duration = min(sequence_lengths)
                    max_duration = max(sequence_lengths)
                    
                    print(f"State duration statistics:")
                    print(f"  • Average duration between transitions: {avg_duration:.1f} minutes")
                    print(f"  • Minimum duration: {min_duration:.1f} minutes")
                    print(f"  • Maximum duration: {max_duration:.1f} minutes")
            
            # Step 8: Data quality assessment
            print(f"\n7. DATA QUALITY FOR HMM TRAINING")
            print("-" * 40)
            
            # Check for sufficient data
            min_transitions_needed = 100  # Minimum for HMM training
            if len(transitions) >= min_transitions_needed:
                print(f"✅ Sufficient transitions for HMM training ({len(transitions):,} ≥ {min_transitions_needed})")
            else:
                print(f"❌ Insufficient transitions for HMM training ({len(transitions):,} < {min_transitions_needed})")
            
            # Check for balanced transitions
            occupied_to_unoccupied = transition_counts.get('occupied → unoccupied', 0)
            unoccupied_to_occupied = transition_counts.get('unoccupied → occupied', 0)
            
            if occupied_to_unoccupied > 0 and unoccupied_to_occupied > 0:
                balance_ratio = min(occupied_to_unoccupied, unoccupied_to_occupied) / max(occupied_to_unoccupied, unoccupied_to_occupied)
                print(f"✅ Balanced transitions (ratio: {balance_ratio:.2f})")
            else:
                print(f"❌ Unbalanced transitions - missing one direction")
            
            # Check time span
            if transitions:
                time_span = transitions[-1][0] - transitions[0][0]
                print(f"✅ Data spans {time_span.days} days")
            
            print(f"\n8. RECOMMENDATIONS FOR HMM IMPLEMENTATION")
            print("-" * 50)
            
            if len(transitions) >= min_transitions_needed:
                print("🎯 READY FOR HMM TRAINING:")
                print(f"  • Use {len(transitions):,} state transitions as training data")
                print(f"  • Implement 2-state HMM (occupied/unoccupied)")
                print(f"  • Consider time-of-day patterns in emission probabilities")
                print(f"  • Account for {daily_avg:.1f} average daily transitions")
                
                # Suggest observation features
                print(f"\n🔧 SUGGESTED HMM FEATURES:")
                print(f"  • Primary observation: binary state (occupied/unoccupied)")
                print(f"  • Additional features: hour of day, day of week")
                print(f"  • Sequence modeling: transition timing patterns")
            else:
                print("⚠️  INSUFFICIENT DATA:")
                print(f"  • Need more historical data for reliable HMM training")
                print(f"  • Consider collecting more data or using alternative sensors")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(analyze_bedroom_transitions())