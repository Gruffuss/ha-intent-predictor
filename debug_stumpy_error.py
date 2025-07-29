#!/usr/bin/env python3
"""
Debug script to reproduce and analyze the STUMPY division by zero error
that's causing the bootstrap process to fail.
"""

import pandas as pd
import numpy as np
import warnings
import sys
import os

# Add src directory to path for imports
sys.path.append('src')

from storage.timeseries_db import TimescaleDBManager
from config.config_loader import ConfigLoader
import asyncio

async def debug_stumpy_error():
    """Reproduce the exact conditions causing STUMPY to fail"""
    
    config = ConfigLoader()
    db_config = config.get("database.timescale")
    db = TimescaleDBManager(f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
    await db.initialize()
    
    async with db.engine.begin() as conn:
        from sqlalchemy import text
        
        # Get ALL office data to replicate the exact pattern discovery process
        query = text("SELECT timestamp, state FROM sensor_events WHERE entity_id = :entity_id ORDER BY timestamp ASC")
        result = await conn.execute(query, {"entity_id": "binary_sensor.office_presence_full_office"})
        rows = result.fetchall()
        
        print(f"Loading {len(rows)} events for analysis...")
        
        # Replicate the exact pattern discovery data processing
        data = []
        for row in rows:
            data.append({
                "timestamp": row[0],
                "state": row[1]
            })
        
        df = pd.DataFrame(data)
        
        # Create binary occupancy column (exact same logic as pattern discovery)
        df['occupied'] = df['state'].isin(['on', 'occupied', '1', 1]).astype(int)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Raw data:")
        print(f"  - Total events: {len(df)}")
        print(f"  - Variance: {df['occupied'].var()}")
        print(f"  - Std dev: {df['occupied'].std()}")
        print(f"  - Mean occupancy: {df['occupied'].mean():.3f}")
        
        # Do the EXACT resampling that's in pattern discovery
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            ts = df.set_index('timestamp')['occupied'].resample('5min').max().fillna(0)
        
        print(f"\nAfter 5-minute resampling:")
        print(f"  - Data points: {len(ts)}")
        print(f"  - Variance: {ts.var()}")
        print(f"  - Std dev: {ts.std()}")
        print(f"  - Mean: {ts.mean():.3f}")
        print(f"  - Unique values: {sorted(ts.unique())}")
        print(f"  - All zeros: {(ts == 0).all()}")
        print(f"  - All ones: {(ts == 1).all()}")
        
        # Check sliding window characteristics
        window_size = 12  # Same as pattern discovery
        if len(ts) >= window_size * 2:
            print(f"\nSliding window analysis (window_size={window_size}):")
            
            constant_windows = 0
            total_windows = len(ts) - window_size + 1
            
            # Check every window for constant values
            for i in range(total_windows):
                window_data = ts.iloc[i:i+window_size].values
                if np.std(window_data) == 0:
                    constant_windows += 1
            
            constant_percentage = (constant_windows / total_windows) * 100
            print(f"  - Total sliding windows: {total_windows}")
            print(f"  - Constant windows: {constant_windows}")
            print(f"  - Constant percentage: {constant_percentage:.1f}%")
            
            if constant_percentage > 50:
                print("  ⚠️  HIGH CONSTANT WINDOW PERCENTAGE - This causes STUMPY division errors!")
        
        # Now test STUMPY with this exact data
        print(f"\nTesting STUMPY...")
        
        if len(ts) >= 24:
            try:
                import stumpy
                
                # Test with various window sizes
                for window_size in [6, 12, 24]:
                    if len(ts) >= window_size * 2:
                        print(f"  Testing window size {window_size}...")
                        
                        # This is the exact call that fails
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore')
                            mp = stumpy.stump(ts.values, m=window_size)
                            
                        print(f"    ✅ Success! Matrix profile shape: {mp.shape}")
                        
            except Exception as e:
                print(f"    ❌ STUMPY failed: {e}")
                print(f"    This is the exact error causing the bootstrap failure!")
                
                # Show the problematic data characteristics
                print(f"\nProblematic data characteristics:")
                print(f"  - Data type: {ts.values.dtype}")
                print(f"  - Contains NaN: {np.isnan(ts.values).any()}")
                print(f"  - Contains Inf: {np.isinf(ts.values).any()}")
                print(f"  - Min value: {ts.values.min()}")
                print(f"  - Max value: {ts.values.max()}")
                
        else:
            print(f"  Not enough data points ({len(ts)}) for STUMPY analysis")
        
    await db.close()

if __name__ == "__main__":
    asyncio.run(debug_stumpy_error())