#!/usr/bin/env python3
"""
Test the cat-specific anomaly detection with 300ms threshold
"""

import asyncio
import sys
from pathlib import Path

# Add src and config to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from storage.timeseries_db import TimescaleDBManager
from config.config_loader import ConfigLoader

async def test_cat_anomalies():
    """Test the cat-specific anomaly detection"""
    
    config = ConfigLoader()
    db_config = config.get('database.timescale')
    db_connection_string = f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    db = TimescaleDBManager(db_connection_string)
    await db.initialize()
    
    print("🐱 Testing cat-specific anomaly detection (300ms threshold)...")
    
    async with db.engine.begin() as conn:
        from sqlalchemy import text
        
        # Test the new cat-movement logic
        result = await conn.execute(text("""
            WITH rapid_sequences AS (
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
                -- Exclude hallways/unknown/stairway from anomaly detection
                AND e1.room NOT IN ('unknown', 'ground_floor_hallway', 'upper_hallway', 'stairway', 'attic')
                AND e2.room NOT IN ('unknown', 'ground_floor_hallway', 'upper_hallway', 'stairway', 'attic')
                -- Very rapid transitions that suggest non-human movement
                AND EXTRACT(EPOCH FROM (e2.timestamp - e1.timestamp)) < 0.5
                LIMIT 20
            )
            SELECT * FROM rapid_sequences
            WHERE time_diff < 0.3  -- Less than 300ms between rooms (cat-like speed)
        """))
        
        cat_movements = result.fetchall()
        print(f"📊 Cat-like movements found: {len(cat_movements)}")
        
        if cat_movements:
            print("\n🔍 Detected cat movements:")
            for i, movement in enumerate(cat_movements):
                print(f"   {i+1}. {movement[4]} → {movement[5]} in {movement[6]*1000:.0f}ms")
                print(f"      {movement[0]} → {movement[1]}")
        else:
            print("   No cat-like rapid movements detected")
        
        # Show some normal human movements for comparison
        print(f"\n👥 Normal human movements (0.5-3 seconds):")
        result = await conn.execute(text("""
            SELECT 
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
            AND e1.room NOT IN ('unknown', 'ground_floor_hallway', 'upper_hallway', 'stairway', 'attic')
            AND e2.room NOT IN ('unknown', 'ground_floor_hallway', 'upper_hallway', 'stairway', 'attic')
            AND EXTRACT(EPOCH FROM (e2.timestamp - e1.timestamp)) BETWEEN 0.5 AND 3
            ORDER BY time_diff ASC
            LIMIT 10
        """))
        
        normal_movements = result.fetchall()
        for i, movement in enumerate(normal_movements):
            print(f"   {i+1}. {movement[0]} → {movement[1]} in {movement[2]:.1f}s ✓")
        
        # Check distribution of movement times
        print(f"\n📈 Movement time distribution:")
        result = await conn.execute(text("""
            WITH movement_times AS (
                SELECT 
                    CASE 
                        WHEN EXTRACT(EPOCH FROM (e2.timestamp - e1.timestamp)) < 0.3 THEN 'Ultra-rapid (<300ms)'
                        WHEN EXTRACT(EPOCH FROM (e2.timestamp - e1.timestamp)) < 1 THEN 'Very rapid (300ms-1s)'  
                        WHEN EXTRACT(EPOCH FROM (e2.timestamp - e1.timestamp)) < 3 THEN 'Rapid (1-3s)'
                        WHEN EXTRACT(EPOCH FROM (e2.timestamp - e1.timestamp)) < 10 THEN 'Normal (3-10s)'
                        ELSE 'Slow (>10s)'
                    END as speed_category,
                    COUNT(*) as count
                FROM sensor_events e1
                JOIN sensor_events e2 ON e2.timestamp > e1.timestamp
                WHERE e1.sensor_type = 'presence'
                AND e2.sensor_type = 'presence'
                AND e1.state = 'on'
                AND e2.state = 'on'
                AND e1.room != e2.room
                AND e1.room NOT IN ('unknown', 'ground_floor_hallway', 'upper_hallway', 'stairway', 'attic')
                AND e2.room NOT IN ('unknown', 'ground_floor_hallway', 'upper_hallway', 'stairway', 'attic')
                AND EXTRACT(EPOCH FROM (e2.timestamp - e1.timestamp)) < 30
                GROUP BY speed_category
            )
            SELECT * FROM movement_times
            ORDER BY 
                CASE speed_category
                    WHEN 'Ultra-rapid (<300ms)' THEN 1
                    WHEN 'Very rapid (300ms-1s)' THEN 2
                    WHEN 'Rapid (1-3s)' THEN 3
                    WHEN 'Normal (3-10s)' THEN 4
                    WHEN 'Slow (>10s)' THEN 5
                END
        """))
        
        distribution = result.fetchall()
        for category, count in distribution:
            print(f"   {category}: {count:,}")
    
    await db.close()
    print("\n✅ Cat anomaly detection test complete")

if __name__ == "__main__":
    asyncio.run(test_cat_anomalies())