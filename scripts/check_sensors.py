#!/usr/bin/env python3
"""
Check available sensors and door sensors in the database
"""

import asyncio
import sys
from pathlib import Path

# Add src and config to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from storage.timeseries_db import TimescaleDBManager
from config.config_loader import ConfigLoader

async def check_sensors():
    """Check available sensors and door sensors"""
    
    config = ConfigLoader()
    db_config = config.get('database.timescale')
    db_connection_string = f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    db = TimescaleDBManager(db_connection_string)
    await db.initialize()
    
    print("🚪 Checking door sensors and room layout...")
    
    async with db.engine.begin() as conn:
        from sqlalchemy import text
        
        # Check door sensors
        result = await conn.execute(text("""
            SELECT DISTINCT entity_id, sensor_type, room
            FROM sensor_events
            WHERE entity_id LIKE '%door%' OR sensor_type = 'door' OR sensor_type = 'contact'
            ORDER BY room, entity_id
        """))
        
        doors = result.fetchall()
        if doors:
            print(f"   Found {len(doors)} door/contact sensors:")
            for entity_id, sensor_type, room in doors:
                print(f"     {entity_id} ({sensor_type}) in {room}")
        else:
            print("   No door sensors found")
        
        # Check all sensor types
        print(f"\n📊 All sensor types in database:")
        result = await conn.execute(text("""
            SELECT sensor_type, COUNT(DISTINCT entity_id) as sensor_count, COUNT(*) as event_count
            FROM sensor_events
            GROUP BY sensor_type
            ORDER BY event_count DESC
        """))
        
        sensor_types = result.fetchall()
        for sensor_type, sensor_count, event_count in sensor_types:
            print(f"   {sensor_type}: {sensor_count} sensors, {event_count:,} events")
        
        # Check room distribution
        print(f"\n🏠 Room distribution:")
        result = await conn.execute(text("""
            SELECT room, COUNT(DISTINCT entity_id) as sensors, COUNT(*) as events
            FROM sensor_events
            GROUP BY room
            ORDER BY events DESC
        """))
        
        room_stats = result.fetchall()
        for room, sensors, events in room_stats:
            print(f"   {room}: {sensors} sensors, {events:,} events")
            
        # Check for contact sensors specifically
        print(f"\n🔗 Contact sensors:")
        result = await conn.execute(text("""
            SELECT entity_id, room, COUNT(*) as events
            FROM sensor_events
            WHERE sensor_type = 'contact'
            GROUP BY entity_id, room
            ORDER BY events DESC
        """))
        
        contacts = result.fetchall()
        for entity_id, room, events in contacts:
            print(f"   {entity_id} in {room}: {events:,} events")
    
    await db.close()
    print("\n✅ Sensor check complete")

if __name__ == "__main__":
    asyncio.run(check_sensors())