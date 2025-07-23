#!/usr/bin/env python3
"""
Verify and fix database schema for HA Intent Predictor
"""

import asyncio
import sys
from pathlib import Path
from sqlalchemy import text
from tabulate import tabulate

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from storage.timeseries_db import TimescaleDBManager
from config.config_loader import ConfigLoader


async def verify_schema():
    """Verify current database schema and show what needs fixing"""
    
    config = ConfigLoader("config/system.yaml")
    db_config = config.get('database.timescale')
    db_connection_string = f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    db = TimescaleDBManager(db_connection_string)
    await db.initialize()
    
    print("🔍 DATABASE SCHEMA VERIFICATION")
    print("="*60)
    
    async with db.engine.begin() as conn:
        # Check what tables exist
        result = await conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """))
        
        existing_tables = [row[0] for row in result]
        
        print("\n📋 Existing Tables:")
        for table in existing_tables:
            print(f"  ✓ {table}")
        
        # Check required tables
        required_tables = [
            'sensor_events',
            'predictions', 
            'pattern_discoveries',
            'model_performance',
            'room_occupancy',
            'discovered_patterns'
        ]
        
        missing_tables = [t for t in required_tables if t not in existing_tables]
        if missing_tables:
            print(f"\n❌ Missing Tables: {', '.join(missing_tables)}")
        
        # Check sensor_events schema
        print("\n📊 sensor_events table schema:")
        result = await conn.execute(text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'sensor_events'
            ORDER BY ordinal_position;
        """))
        
        columns = []
        for row in result:
            columns.append([row[0], row[1], row[2]])
        
        if columns:
            print(tabulate(columns, headers=['Column', 'Type', 'Nullable'], tablefmt='grid'))
        else:
            print("  ❌ Table does not exist")
        
        # Check what the code expects vs what exists
        expected_columns = [
            'timestamp', 'entity_id', 'state', 'numeric_value', 
            'attributes', 'room', 'sensor_type', 'zone_type', 
            'zone_info', 'person', 'enriched_data'
        ]
        
        if columns:
            existing_cols = [col[0] for col in columns]
            missing_cols = [col for col in expected_columns if col not in existing_cols]
            
            if missing_cols:
                print(f"\n❌ Missing columns in sensor_events: {', '.join(missing_cols)}")
            
            # Check for extra columns
            extra_cols = [col for col in existing_cols if col not in expected_columns and col not in ['id', 'processed_at']]
            if extra_cols:
                print(f"\n⚠️  Extra columns in sensor_events: {', '.join(extra_cols)}")
        
        # Check hypertables
        print("\n🕐 Hypertables:")
        result = await conn.execute(text("""
            SELECT hypertable_name 
            FROM timescaledb_information.hypertables
            WHERE hypertable_schema = 'public';
        """))
        
        hypertables = [row[0] for row in result]
        for table in hypertables:
            print(f"  ✓ {table}")
        
        # Check row counts
        print("\n📈 Table Statistics:")
        for table in existing_tables:
            try:
                result = await conn.execute(text(f"SELECT COUNT(*) FROM {table};"))
                count = result.fetchone()[0]
                print(f"  {table}: {count:,} rows")
            except:
                print(f"  {table}: Error counting rows")
    
    await db.close()
    
    print("\n" + "="*60)
    print("✅ Verification complete")
    print("\nTO FIX SCHEMA ISSUES:")
    print("1. Backup your data if needed")
    print("2. Run: python scripts/fixed_bootstrap.py")
    print("3. Then import historical data: python scripts/historical_import.py --days 180")


if __name__ == "__main__":
    asyncio.run(verify_schema())