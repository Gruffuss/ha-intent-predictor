#!/usr/bin/env python3
"""
Check training data availability and quality
"""

import asyncio
from src.storage.timeseries_db import TimescaleDBManager
from config.config_loader import ConfigLoader
from sqlalchemy import text

async def check_training_data():
    """Check what training data is available for the ML models"""
    try:
        config = ConfigLoader()
        db_config = config.get("database.timescale")
        
        db = TimescaleDBManager(
            f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        await db.initialize()
        
        async with db.engine.begin() as conn:
            print("=== TRAINING DATA ANALYSIS ===\n")
            
            # Total events
            result = await conn.execute(text("SELECT COUNT(*) FROM sensor_events"))
            total_events = result.fetchone()[0]
            print(f"Total sensor events: {total_events:,}")
            
            # Events with room data
            result = await conn.execute(text("SELECT COUNT(*) FROM sensor_events WHERE room IS NOT NULL"))
            room_events = result.fetchone()[0]
            print(f"Events with room data: {room_events:,}")
            
            # Events by room
            result = await conn.execute(text("""
                SELECT room, COUNT(*) as count 
                FROM sensor_events 
                WHERE room IS NOT NULL 
                GROUP BY room 
                ORDER BY count DESC
            """))
            room_breakdown = result.fetchall()
            print("\nEvents by room:")
            for room, count in room_breakdown:
                print(f"  {room}: {count:,}")
            
            # Presence sensor events (direct training labels)
            result = await conn.execute(text("SELECT COUNT(*) FROM sensor_events WHERE entity_id LIKE '%presence%'"))
            presence_events = result.fetchone()[0]
            print(f"\nPresence sensor events (direct labels): {presence_events:,}")
            
            # Door sensor events (potential training labels)
            result = await conn.execute(text("SELECT COUNT(*) FROM sensor_events WHERE entity_id LIKE '%door%'"))
            door_events = result.fetchone()[0]
            print(f"Door sensor events (potential labels): {door_events:,}")
            
            # Bathroom door events specifically
            result = await conn.execute(text("""
                SELECT COUNT(*) 
                FROM sensor_events 
                WHERE entity_id LIKE '%door%' 
                AND (room = 'bathroom' OR room = 'small_bathroom')
            """))
            bathroom_door_events = result.fetchone()[0]            
            print(f"Bathroom door events (bathroom labels): {bathroom_door_events:,}")
            
            # Data time range
            result = await conn.execute(text("SELECT MIN(timestamp), MAX(timestamp) FROM sensor_events"))
            min_time, max_time = result.fetchone()
            print(f"\nData time range: {min_time} to {max_time}")
            
            # Calculate days of data
            days_of_data = (max_time - min_time).days if min_time and max_time else 0
            print(f"Days of data: {days_of_data}")
            
            # Analyze training labels availability
            print("\n=== TRAINING LABELS ANALYSIS ===")
            total_potential_labels = presence_events + bathroom_door_events
            print(f"Total potential training labels: {total_potential_labels:,}")
            
            if total_potential_labels == 0:
                print("\n❌ CRITICAL ISSUE: NO TRAINING LABELS AVAILABLE!")
                print("   The models cannot learn without ground truth occupancy data.")
                print("   This explains why predictions are 0.5 (no learning).")
                print("\n   Solutions:")
                print("   1. Ensure presence sensors are properly configured in HA")
                print("   2. Check bathroom door sensors are working")
                print("   3. Verify sensor data is being imported correctly")
            elif total_potential_labels < 100:
                print(f"\n⚠️  CRITICAL: Very few training labels ({total_potential_labels:,})")
                print("   Models need more labeled data to learn effectively.")
                print("   This likely explains poor prediction performance.")
            elif total_potential_labels < 1000:
                print(f"\n⚠️  WARNING: Few training labels ({total_potential_labels:,})")
                print("   Models may struggle with limited training data.")
            else:
                print(f"\n✓ Training data looks adequate: {total_potential_labels:,} labels")
            
            # Check specific sensor types that provide labels
            print("\n=== LABEL-PROVIDING SENSORS ===")
            result = await conn.execute(text("""
                SELECT entity_id, COUNT(*) as count
                FROM sensor_events 
                WHERE entity_id LIKE '%presence%' OR 
                      (entity_id LIKE '%door%' AND (room = 'bathroom' OR room = 'small_bathroom'))
                GROUP BY entity_id
                ORDER BY count DESC
                LIMIT 10
            """))
            label_sensors = result.fetchall()
            
            if label_sensors:
                print("Top label-providing sensors:")
                for entity_id, count in label_sensors:
                    print(f"  {entity_id}: {count:,} events")
            else:
                print("No label-providing sensors found!")
            
            # Check recent activity to see if sensors are active
            result = await conn.execute(text("""
                SELECT COUNT(*) 
                FROM sensor_events 
                WHERE timestamp > NOW() - INTERVAL '24 hours'
            """))
            recent_events = result.fetchone()[0]
            print(f"\nEvents in last 24 hours: {recent_events:,}")
            
            if recent_events == 0:
                print("⚠️  No recent sensor activity - check if data ingestion is working")
        
        await db.close()
        
    except Exception as e:
        print(f"Error checking training data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_training_data())