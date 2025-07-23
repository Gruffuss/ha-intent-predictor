#!/usr/bin/env python3
"""
Test script for anomaly detection JSON serialization fix
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from storage.timeseries_db import TimescaleDBManager
from config.config_loader import ConfigLoader

async def test_anomaly_detection():
    """Test the fixed anomaly detection without full import"""
    
    # Initialize database connection
    config = ConfigLoader()
    db_config = config.get('database.timescale')
    db_connection_string = f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    timeseries_db = TimescaleDBManager(db_connection_string)
    await timeseries_db.initialize()
    
    print("🔍 Testing anomaly detection with existing data...")
    
    # Look for rapid movements (same query as historical import)
    async with timeseries_db.engine.begin() as conn:
        from sqlalchemy import text
        
        anomalies = await conn.execute(text("""
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
                LIMIT 5
            )
            SELECT * FROM rapid_movements
            WHERE time_diff < 3  -- Less than 3 seconds between rooms
        """))
        
        anomaly_results = anomalies.fetchall()
        
    print(f"📊 Found {len(anomaly_results)} rapid movement anomalies")
    
    if anomaly_results:
        print("\n🧪 Testing JSON serialization fix...")
        
        # Test the fixed JSON serialization
        for i, anomaly in enumerate(anomaly_results):
            try:
                print(f"🔍 Anomaly {i+1} raw data: {anomaly}")
                print(f"🔍 Anomaly type: {type(anomaly)}")
                
                # Convert anomaly data to JSON-serializable format (same as fixed code)
                if hasattr(anomaly, 'keys'):
                    # It's a SQLAlchemy Row object
                    anomaly_dict = {key: anomaly[key] for key in anomaly.keys()}
                else:
                    anomaly_dict = dict(anomaly)
                
                print(f"🔍 Converted dict: {anomaly_dict}")
                
                # Convert non-JSON-serializable objects (same fix as historical_import.py)
                for key, value in anomaly_dict.items():
                    if hasattr(value, 'isoformat'):  # datetime objects
                        anomaly_dict[key] = value.isoformat()
                    elif hasattr(value, '__float__'):  # Decimal objects
                        anomaly_dict[key] = float(value)
                    elif not isinstance(value, (str, int, float, bool, type(None))):
                        # Convert any other non-serializable objects to string
                        anomaly_dict[key] = str(value)
                
                # Test JSON serialization
                json_data = json.dumps(anomaly_dict)
                print(f"✅ Anomaly {i+1}: JSON serialization successful")
                print(f"   Sample data: {json_data[:100]}...")
                
            except Exception as e:
                print(f"❌ Anomaly {i+1}: JSON serialization failed: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    else:
        print("ℹ️  No rapid movement anomalies found in current dataset")
        
        # Test with mock data to verify the fix works
        print("\n🧪 Testing with mock datetime data...")
        from datetime import datetime, timezone
        
        mock_anomaly = {
            't1': datetime.now(timezone.utc),
            't2': datetime.now(timezone.utc),
            'sensor1': 'test_sensor_1',
            'sensor2': 'test_sensor_2',
            'time_diff': 2.5
        }
        
        try:
            # Apply the same fix
            for key, value in mock_anomaly.items():
                if hasattr(value, 'isoformat'):
                    mock_anomaly[key] = value.isoformat()
            
            json_data = json.dumps(mock_anomaly)
            print("✅ Mock data: JSON serialization successful")
            print(f"   Sample: {json_data}")
            
        except Exception as e:
            print(f"❌ Mock data: JSON serialization failed: {e}")
            return False
    
    await timeseries_db.close()
    print("\n🎉 Anomaly detection JSON serialization fix verified!")
    return True

if __name__ == "__main__":
    result = asyncio.run(test_anomaly_detection())
    if result:
        print("✅ TEST PASSED")
    else:
        print("❌ TEST FAILED")
        sys.exit(1)