#!/usr/bin/env python3
"""
Drop existing database and run complete bootstrap with proper schema
"""

import asyncio
import logging
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent))

from storage.timeseries_db import TimescaleDBManager
from config.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def drop_all_tables():
    """Drop all existing tables to start fresh"""
    print("🗑️  Dropping existing database tables...")
    
    config = ConfigLoader()
    db_config = config.get('database.timescale')
    db_conn = f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    db = TimescaleDBManager(db_conn)
    await db.initialize()
    
    # List of all tables that might exist
    tables = [
        'sensor_events', 
        'room_occupancy', 
        'discovered_patterns', 
        'predictions', 
        'pattern_discoveries', 
        'model_performance'
    ]
    
    async with db.engine.begin() as conn:
        from sqlalchemy import text
        
        # Drop in correct order (avoid foreign key issues)
        for table in tables:
            try:
                await conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
                print(f"  ✓ Dropped table: {table}")
            except Exception as e:
                print(f"  - Warning dropping {table}: {e}")
    
    await db.close()
    print("✅ Database cleaned successfully")

async def run_complete_bootstrap():
    """Run the complete bootstrap process"""
    print("🚀 Starting complete bootstrap process...")
    
    # Run bootstrap_complete.py
    script_path = Path(__file__).parent / "bootstrap_complete.py"
    config_path = Path(__file__).parent.parent / "config" / "system.yaml"
    
    try:
        # Execute bootstrap with force flag
        cmd = [
            sys.executable, str(script_path),
            "--config", str(config_path),
            "--force"
        ]
        
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, 
                              cwd=Path(__file__).parent.parent,
                              text=True,
                              capture_output=False)  # Show output in real-time
        
        if result.returncode == 0:
            print("✅ Bootstrap completed successfully!")
            return True
        else:
            print(f"❌ Bootstrap failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Bootstrap execution failed: {e}")
        return False

async def main():
    """Main execution"""
    print("=" * 60)
    print("🔄 COMPLETE DATABASE RESET & BOOTSTRAP")
    print("=" * 60)
    print("This will:")
    print("1. Drop all existing database tables")
    print("2. Run complete bootstrap with proper schema")
    print("3. Import 180 days of historical data")
    print("4. Initialize ML system")
    print("=" * 60)
    
    try:
        # Step 1: Drop existing tables
        await drop_all_tables()
        
        # Step 2: Run complete bootstrap
        success = await run_complete_bootstrap()
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 COMPLETE RESET & BOOTSTRAP SUCCESSFUL!")
            print("=" * 60)
            print("✅ Database schema properly created")
            print("✅ Historical data import completed")
            print("✅ ML system initialized")
            print("✅ System ready for operation")
            print("=" * 60)
        else:
            print("\n❌ Bootstrap process failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Complete bootstrap failed: {e}")
        print(f"\n❌ FATAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())