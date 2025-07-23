#!/usr/bin/env python3
"""
Test the improved anomaly detection with realistic thresholds
"""

import asyncio
import sys
from pathlib import Path

# Add src and config to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from storage.timeseries_db import TimescaleDBManager
from config.config_loader import ConfigLoader

async def test_improved_anomalies():
    """Test the improved anomaly detection logic"""
    
    config = ConfigLoader()
    db_config = config.get('database.timescale')
    db_connection_string = f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    db = TimescaleDBManager(db_connection_string)
    await db.initialize()
    
    print("🧪 Testing improved anomaly detection logic...")
    
    async with db.engine.begin() as conn:
        from sqlalchemy import text
        
        # Test the new improved logic
        result = await conn.execute(text("""
            WITH impossible_movements AS (
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
                -- Exclude hallways and unknown rooms from anomaly detection
                AND e1.room NOT IN ('unknown', 'ground_floor_hallway', 'upper_hallway')
                AND e2.room NOT IN ('unknown', 'ground_floor_hallway', 'upper_hallway')
                -- Only flag movements between distant rooms (not adjacent)
                AND NOT (
                    (e1.room = 'living_kitchen' AND e2.room IN ('bedroom', 'office')) OR
                    (e2.room = 'living_kitchen' AND e1.room IN ('bedroom', 'office')) OR
                    (e1.room = 'bedroom' AND e2.room = 'office') OR
                    (e2.room = 'bedroom' AND e1.room = 'office')
                )
                AND EXTRACT(EPOCH FROM (e2.timestamp - e1.timestamp)) < 2
                LIMIT 10
            )
            SELECT * FROM impossible_movements
            WHERE time_diff < 1  -- Less than 1 second between distant rooms
        """))
        
        impossible_movements = result.fetchall()
        print(f"📊 Impossible movements found: {len(impossible_movements)}")
        
        if impossible_movements:
            print("\n🔍 Sample impossible movements:")
            for i, movement in enumerate(impossible_movements[:5]):
                print(f"   {i+1}. {movement[4]} → {movement[5]} in {movement[6]:.2f}s")
                print(f"      {movement[0]} → {movement[1]}")
        
        # Test normal movements that should NOT be flagged
        print(f"\n✅ Testing normal movements (should NOT be anomalies):")
        result = await conn.execute(text("""
            WITH normal_movements AS (
                SELECT 
                    e1.timestamp as t1,
                    e2.timestamp as t2,
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
                -- Adjacent room movements (should be normal)
                AND (
                    (e1.room = 'living_kitchen' AND e2.room IN ('bedroom', 'office')) OR
                    (e2.room = 'living_kitchen' AND e1.room IN ('bedroom', 'office')) OR
                    (e1.room = 'bedroom' AND e2.room = 'office') OR
                    (e2.room = 'bedroom' AND e1.room = 'office')
                )
                AND EXTRACT(EPOCH FROM (e2.timestamp - e1.timestamp)) BETWEEN 2 AND 8
                LIMIT 5
            )
            SELECT * FROM normal_movements
        """))
        
        normal_movements = result.fetchall()
        print(f"   Normal adjacent movements (2-8 seconds): {len(normal_movements)}")
        for i, movement in enumerate(normal_movements):
            print(f"     {i+1}. {movement[2]} → {movement[3]} in {movement[4]:.1f}s ✓")
        
        # Check existing anomaly patterns in database
        result = await conn.execute(text("""
            SELECT pattern_type, COUNT(*) 
            FROM discovered_patterns 
            GROUP BY pattern_type
            ORDER BY COUNT(*) DESC
        """))
        
        existing_patterns = result.fetchall()
        print(f"\n📋 Existing anomaly patterns in database:")
        for pattern_type, count in existing_patterns:
            print(f"   {pattern_type}: {count}")
    
    await db.close()
    print("\n✅ Improved anomaly detection test complete")

if __name__ == "__main__":
    asyncio.run(test_improved_anomalies())