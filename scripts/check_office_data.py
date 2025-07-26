#!/usr/bin/env python3
import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/opt/ha-intent-predictor')

from src.storage.timeseries_db import TimescaleDBManager
from config.config_loader import ConfigLoader
from datetime import datetime, timezone, timedelta

async def check_office_data():
    config = ConfigLoader()
    db_config = config.get("database.timescale")
    connection_string = f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    db = TimescaleDBManager(connection_string)
    await db.initialize()
    
    async with db.engine.begin() as conn:
        from sqlalchemy import text
        
        # Check total office events
        result = await conn.execute(text("SELECT COUNT(*) FROM sensor_events WHERE room = 'office'"))
        total_office = result.fetchone()[0]
        print(f"Total office events: {total_office:,}")
        
        # Check office events by weekday around 11:30 AM (10:30-12:30 range)
        query = """
        SELECT 
            EXTRACT(dow FROM timestamp) as day_of_week,
            DATE(timestamp) as date,
            COUNT(*) as events,
            SUM(CASE WHEN state = 'on' OR state = 'true' OR state::text = 'True' THEN 1 ELSE 0 END) as occupied_events
        FROM sensor_events 
        WHERE room = 'office' 
        AND EXTRACT(hour FROM timestamp) BETWEEN 10 AND 12
        AND EXTRACT(dow FROM timestamp) BETWEEN 1 AND 5
        GROUP BY EXTRACT(dow FROM timestamp), DATE(timestamp)
        ORDER BY date DESC
        LIMIT 30
        """
        
        result = await conn.execute(text(query))
        rows = result.fetchall()
        
        print("\n=== Office Events on Weekdays 10:30-12:30 (Last 30 days) ===")
        print("Day | Date       | Events | Occupied | Occupancy %")
        print("-" * 50)
        
        occupied_days = 0
        total_days = 0
        for row in rows:
            dow_name = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][int(row[0])]
            occupancy_pct = (row[3] / row[2] * 100) if row[2] > 0 else 0
            print(f"{dow_name} | {row[1]} | {row[2]:6} | {row[3]:8} | {occupancy_pct:7.1f}%")
            
            if occupancy_pct > 50:  # Consider day as "occupied" if >50% of events show occupancy
                occupied_days += 1
            total_days += 1
        
        if total_days > 0:
            actual_occupancy_rate = (occupied_days / total_days) * 100
            print(f"\nActual occupancy rate: {occupied_days}/{total_days} days = {actual_occupancy_rate:.1f}%")
        
        # Also check what sensors are in the office
        print("\n=== Office Sensors ===")
        sensor_query = """
        SELECT DISTINCT entity_id, COUNT(*) as event_count
        FROM sensor_events 
        WHERE room = 'office'
        GROUP BY entity_id
        ORDER BY event_count DESC
        """
        
        result = await conn.execute(text(sensor_query))
        sensor_rows = result.fetchall()
        
        for sensor_row in sensor_rows:
            print(f"Sensor: {sensor_row[0]} - Events: {sensor_row[1]:,}")
    
    await db.close()

if __name__ == "__main__":
    asyncio.run(check_office_data())