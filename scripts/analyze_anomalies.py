#!/usr/bin/env python3
"""
Analyze anomaly detection patterns in the sensor data
"""

import asyncio
import sys
from pathlib import Path

# Add src and config to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from storage.timeseries_db import TimescaleDBManager
from config.config_loader import ConfigLoader

async def analyze_anomalies():
    """Analyze presence sensor data for anomaly detection patterns"""
    
    config = ConfigLoader()
    db_config = config.get('database.timescale')
    db_connection_string = f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    db = TimescaleDBManager(db_connection_string)
    await db.initialize()
    
    print("🔍 Analyzing presence sensor data for anomaly detection...")
    
    async with db.engine.begin() as conn:
        from sqlalchemy import text
        
        # First, let's see how many presence events we have and their distribution
        result = await conn.execute(text("""
            SELECT 
                COUNT(*) as total_events,
                COUNT(CASE WHEN state = 'on' THEN 1 END) as on_events,
                COUNT(CASE WHEN state = 'off' THEN 1 END) as off_events,
                COUNT(DISTINCT room) as rooms_with_presence,
                COUNT(DISTINCT entity_id) as presence_sensors
            FROM sensor_events 
            WHERE sensor_type = 'presence'
        """))
        
        stats = result.fetchone()
        print(f"📊 Presence sensor statistics:")
        print(f"   Total events: {stats[0]:,}")
        print(f"   ON events: {stats[1]:,}")
        print(f"   OFF events: {stats[2]:,}")
        print(f"   Rooms with presence: {stats[3]}")
        print(f"   Presence sensors: {stats[4]}")
        
        # Look for rapid movements with more relaxed criteria
        print(f"\n🔍 Testing broader anomaly detection (within 10 seconds)...")
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
                AND e2.timestamp - e1.timestamp < INTERVAL '10 seconds'
                LIMIT 20
            )
            SELECT COUNT(*) as rapid_count FROM rapid_movements
            WHERE time_diff < 10
        """))
        
        rapid_10s = result.fetchone()[0]
        print(f"   Rapid movements (< 10 seconds): {rapid_10s}")
        
        # Check for any same-person movements between rooms
        print(f"\n🔍 Looking for impossible simultaneous presence...")
        result = await conn.execute(text("""
            WITH simultaneous_presence AS (
                SELECT 
                    e1.timestamp,
                    e1.room as room1,
                    e2.room as room2,
                    e1.entity_id as sensor1,
                    e2.entity_id as sensor2
                FROM sensor_events e1
                JOIN sensor_events e2 ON ABS(EXTRACT(EPOCH FROM (e2.timestamp - e1.timestamp))) < 1
                WHERE e1.sensor_type = 'presence'
                AND e2.sensor_type = 'presence'
                AND e1.state = 'on'
                AND e2.state = 'on'
                AND e1.room != e2.room
                AND e1.entity_id != e2.entity_id
                LIMIT 10
            )
            SELECT COUNT(*) FROM simultaneous_presence
        """))
        
        simultaneous = result.fetchone()[0]
        print(f"   Simultaneous presence in different rooms: {simultaneous}")
        
        # Look for sensor patterns that might indicate cat activity
        print(f"\n🐱 Looking for potential cat activity patterns...")
        result = await conn.execute(text("""
            WITH quick_sequences AS (
                SELECT 
                    e1.timestamp,
                    e1.room,
                    e1.entity_id,
                    COUNT(*) OVER (
                        PARTITION BY e1.room 
                        ORDER BY e1.timestamp 
                        RANGE BETWEEN CURRENT ROW AND INTERVAL '30 seconds' FOLLOWING
                    ) as events_in_30s
                FROM sensor_events e1
                WHERE e1.sensor_type = 'presence'
                AND e1.state = 'on'
            )
            SELECT 
                room,
                COUNT(*) as high_frequency_periods,
                MAX(events_in_30s) as max_events_30s
            FROM quick_sequences
            WHERE events_in_30s >= 5
            GROUP BY room
            ORDER BY high_frequency_periods DESC
        """))
        
        cat_patterns = result.fetchall()
        if cat_patterns:
            print(f"   Rooms with high-frequency sensor activity:")
            for room, periods, max_events in cat_patterns:
                print(f"     {room}: {periods} periods with {max_events} max events/30s")
        else:
            print(f"   No high-frequency patterns detected")
        
        # Sample some actual rapid movements to understand what we're seeing
        print(f"\n📋 Sample rapid movements (first 5):")
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
                AND e2.timestamp - e1.timestamp < INTERVAL '5 seconds'
                LIMIT 5
            )
            SELECT * FROM rapid_movements
            WHERE time_diff < 5
        """))
        
        samples = result.fetchall()
        for i, sample in enumerate(samples):
            print(f"   {i+1}. {sample[4]} → {sample[5]} in {sample[6]:.1f}s")
            print(f"      {sample[0]} → {sample[1]}")
    
    await db.close()
    print("\n✅ Analysis complete")

if __name__ == "__main__":
    asyncio.run(analyze_anomalies())