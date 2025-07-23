#!/usr/bin/env python3
"""
Update anomaly detection with more realistic thresholds and patterns
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src and config to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from storage.timeseries_db import TimescaleDBManager
from config.config_loader import ConfigLoader

async def update_anomaly_detection():
    """Update anomaly detection with improved patterns"""
    
    config = ConfigLoader()
    db_config = config.get('database.timescale')
    db_connection_string = f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    db = TimescaleDBManager(db_connection_string)
    await db.initialize()
    
    print("🔄 Updating anomaly detection with improved patterns...")
    
    async with db.engine.begin() as conn:
        from sqlalchemy import text
        
        # Clear existing rapid movement anomalies
        await conn.execute(text("""
            DELETE FROM discovered_patterns 
            WHERE pattern_type = 'rapid_movement_anomaly'
        """))
        
        print("   ✓ Cleared old anomaly records")
        
        # Detect multiple types of anomalies
        anomaly_types = []
        
        # 1. Impossible rapid movements (< 2 seconds between distant rooms)
        print("   🔍 Finding impossible rapid movements...")
        result = await conn.execute(text("""
            WITH rapid_movements AS (
                SELECT 
                    e1.timestamp as t1,
                    e2.timestamp as t2,
                    e1.entity_id as sensor1,
                    e2.entity_id as sensor2,
                    e1.room as room1,
                    e2.room as room2,
                    EXTRACT(EPOCH FROM (e2.timestamp - e1.timestamp)) as time_diff
                FROM sensor_events e1
                JOIN sensor_events e2 ON e2.timestamp > e1.timestamp
                WHERE e1.sensor_type = 'presence'
                AND e2.sensor_type = 'presence'
                AND e1.state = 'on'
                AND e2.state = 'on'
                AND e1.room != e2.room
                AND e1.room NOT IN ('unknown', 'ground_floor_hallway', 'upper_hallway')
                AND e2.room NOT IN ('unknown', 'ground_floor_hallway', 'upper_hallway')
                AND e2.timestamp - e1.timestamp < INTERVAL '2 seconds'
            )
            SELECT * FROM rapid_movements
        """))
        
        rapid_movements = result.fetchall()
        anomaly_types.append(('impossible_rapid_movement', len(rapid_movements)))
        
        # Store rapid movement anomalies
        for anomaly in rapid_movements:
            anomaly_dict = {
                't1': anomaly[0].isoformat(),
                't2': anomaly[1].isoformat(),
                'sensor1': anomaly[2],
                'sensor2': anomaly[3],
                'room1': anomaly[4],
                'room2': anomaly[5],
                'time_diff': float(anomaly[6])
            }
            
            await conn.execute(text("""
                INSERT INTO discovered_patterns (
                    room, pattern_type, pattern_data, significance_score
                ) VALUES (:room, :pattern_type, :pattern_data, :significance_score)
            """), {
                'room': 'multiple',
                'pattern_type': 'impossible_rapid_movement',
                'pattern_data': json.dumps(anomaly_dict),
                'significance_score': 0.9
            })
        
        # 2. Simultaneous presence in multiple rooms
        print("   🔍 Finding simultaneous presence anomalies...")
        result = await conn.execute(text("""
            WITH simultaneous_presence AS (
                SELECT 
                    e1.timestamp,
                    e1.room as room1,
                    e2.room as room2,
                    e1.entity_id as sensor1,
                    e2.entity_id as sensor2
                FROM sensor_events e1
                JOIN sensor_events e2 ON ABS(EXTRACT(EPOCH FROM (e2.timestamp - e1.timestamp))) < 0.5
                WHERE e1.sensor_type = 'presence'
                AND e2.sensor_type = 'presence'
                AND e1.state = 'on'
                AND e2.state = 'on'
                AND e1.room != e2.room
                AND e1.entity_id != e2.entity_id
                AND e1.room NOT IN ('unknown', 'ground_floor_hallway', 'upper_hallway')
                AND e2.room NOT IN ('unknown', 'ground_floor_hallway', 'upper_hallway')
            )
            SELECT * FROM simultaneous_presence
        """))
        
        simultaneous = result.fetchall()
        anomaly_types.append(('simultaneous_presence', len(simultaneous)))
        
        # Store simultaneous presence anomalies
        for anomaly in simultaneous:
            anomaly_dict = {
                'timestamp': anomaly[0].isoformat(),
                'room1': anomaly[1],
                'room2': anomaly[2],
                'sensor1': anomaly[3],
                'sensor2': anomaly[4]
            }
            
            await conn.execute(text("""
                INSERT INTO discovered_patterns (
                    room, pattern_type, pattern_data, significance_score
                ) VALUES (:room, :pattern_type, :pattern_data, :significance_score)
            """), {
                'room': 'multiple',
                'pattern_type': 'simultaneous_presence',
                'pattern_data': json.dumps(anomaly_dict),
                'significance_score': 0.85
            })
        
        # 3. High-frequency sensor activity (potential cat activity)
        print("   🐱 Finding high-frequency activity patterns...")
        result = await conn.execute(text("""
            WITH high_frequency AS (
                SELECT 
                    DATE_TRUNC('minute', e1.timestamp) as time_window,
                    e1.room,
                    COUNT(*) as events_per_minute
                FROM sensor_events e1
                WHERE e1.sensor_type = 'presence'
                AND e1.state = 'on'
                AND e1.room NOT IN ('unknown', 'ground_floor_hallway', 'upper_hallway')
                GROUP BY DATE_TRUNC('minute', e1.timestamp), e1.room
                HAVING COUNT(*) >= 8  -- 8+ events per minute indicates unusual activity
            )
            SELECT * FROM high_frequency
            ORDER BY events_per_minute DESC
            LIMIT 50
        """))
        
        high_freq = result.fetchall()
        anomaly_types.append(('high_frequency_activity', len(high_freq)))
        
        # Store high-frequency anomalies
        for anomaly in high_freq:
            anomaly_dict = {
                'time_window': anomaly[0].isoformat(),
                'room': anomaly[1],
                'events_per_minute': anomaly[2]
            }
            
            await conn.execute(text("""
                INSERT INTO discovered_patterns (
                    room, pattern_type, pattern_data, significance_score
                ) VALUES (:room, :pattern_type, :pattern_data, :significance_score)
            """), {
                'room': anomaly[1],
                'pattern_type': 'high_frequency_activity',
                'pattern_data': json.dumps(anomaly_dict),
                'significance_score': min(0.9, 0.5 + (anomaly[2] * 0.05))  # Higher score for more events
            })
        
        # Summary
        print("\n📊 Anomaly Detection Results:")
        total_anomalies = sum(count for _, count in anomaly_types)
        for anomaly_type, count in anomaly_types:
            print(f"   {anomaly_type}: {count}")
        print(f"   Total anomalies detected: {total_anomalies}")
        
        # Verify storage
        result = await conn.execute(text("SELECT COUNT(*) FROM discovered_patterns"))
        stored_count = result.fetchone()[0]
        print(f"   Anomalies stored in database: {stored_count}")
    
    await db.close()
    print("\n✅ Anomaly detection updated successfully")

if __name__ == "__main__":
    asyncio.run(update_anomaly_detection())