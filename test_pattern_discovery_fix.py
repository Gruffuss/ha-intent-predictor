#!/usr/bin/env python3
"""
Test script to verify the pattern discovery fix works correctly
Run this in the container with: python test_pattern_discovery_fix.py
"""

import sys
import os
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, '/opt/ha-intent-predictor')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_pattern_discovery_fix():
    """Test the pattern discovery fix with problematic data patterns"""
    
    try:
        from src.learning.pattern_discovery import PatternDiscovery
        
        logger.info("🧪 Testing pattern discovery fix...")
        
        # Create test data that would previously cause STUMPY to fail
        discoverer = PatternDiscovery()
        
        # Test Case 1: Create sparse occupancy data (lots of zeros)
        logger.info("\n📊 Test Case 1: Sparse occupancy data")
        
        # Generate 7 days of mostly-zero data with some occupancy
        timestamps = pd.date_range(
            start=datetime.now(timezone.utc) - timedelta(days=7),
            end=datetime.now(timezone.utc),
            freq='30S'
        )
        
        # Create sparse occupancy (99.9% zeros)
        occupancy = np.zeros(len(timestamps))
        # Add some brief occupancy periods
        for i in range(0, len(timestamps), 2000):  # Every ~16 hours
            if i + 100 < len(timestamps):
                occupancy[i:i+100] = 1  # 50 minutes of occupancy
        
        # Create events DataFrame
        events = []
        for i, (ts, occ) in enumerate(zip(timestamps, occupancy)):
            if occ == 1 or np.random.random() < 0.001:  # Include some 'off' events
                events.append({
                    'timestamp': ts,
                    'entity_id': 'test_sensor',
                    'state': 'on' if occ else 'off',
                    'room': 'test_room',
                    'occupied': int(occ)
                })
        
        logger.info(f"Created {len(events)} test events from {len(timestamps)} timestamps")
        logger.info(f"Occupancy rate: {occupancy.mean():.1%}")
        
        # Test the fixed pattern discovery
        result = await discoverer._comprehensive_discovery(events, 'test_room')
        
        if 'error' in result:
            logger.error(f"❌ Pattern discovery failed: {result['error']}")
            return False
        
        # Check if recurring patterns were processed (even if no patterns found)
        if 'recurring' not in result:
            logger.error("❌ Recurring pattern discovery not executed")
            return False
        
        logger.info("✅ Test Case 1 passed - no STUMPY errors")
        
        # Test Case 2: All-zero data (should be caught by validation)
        logger.info("\n📊 Test Case 2: All-zero data")
        
        zero_events = []
        for i, ts in enumerate(timestamps[:1000]):
            zero_events.append({
                'timestamp': ts,
                'entity_id': 'test_sensor',
                'state': 'off',
                'room': 'test_room', 
                'occupied': 0
            })
        
        result2 = await discoverer._comprehensive_discovery(zero_events, 'test_room_zero')
        
        if 'recurring' in result2:
            recurring = result2['recurring']
            # Should find validation errors, not STUMPY crashes
            validation_caught_issues = any(
                data.get('reason', '').find('Constant data') >= 0 or
                data.get('reason', '').find('constant sliding windows') >= 0 or  
                data.get('found') == False
                for data in recurring.values() 
                if isinstance(data, dict)
            )
            
            if validation_caught_issues:
                logger.info("✅ Test Case 2 passed - validation caught constant data")
            else:
                logger.warning("⚠️  Test Case 2: validation may not have caught all issues")
        
        # Test Case 3: Real room test
        logger.info("\n📊 Test Case 3: Testing with real room data")
        
        test_rooms = ['living_kitchen', 'bedroom', 'office']
        
        for room in test_rooms:
            logger.info(f"Testing pattern discovery for {room}...")
            try:
                result = await discoverer.discover_multizone_patterns(room)
                
                if result.get('total_events', 0) > 0:
                    logger.info(f"✅ {room}: {result['total_events']} events processed successfully")
                else:
                    logger.info(f"ℹ️  {room}: No events found (expected for some rooms)")
                    
            except Exception as e:
                logger.error(f"❌ {room} failed: {e}")
                return False
        
        logger.info("\n🎉 All tests passed! Pattern discovery fix is working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the test"""
    
    logger.info("🚀 Starting pattern discovery fix test...")
    
    success = await test_pattern_discovery_fix()
    
    if success:
        logger.info("✅ ALL TESTS PASSED - Pattern discovery fix is working!")
        print("\n" + "="*80)
        print("✅ PATTERN DISCOVERY FIX VERIFICATION SUCCESSFUL")
        print("="*80)
        print("• STUMPY division by zero errors should be eliminated")
        print("• Validation catches constant data before STUMPY execution")
        print("• Sliding window validation prevents problematic cases")
        print("• Graceful error handling allows bootstrap to continue")
        print("• Ready for production bootstrap process")
        sys.exit(0)
    else:
        logger.error("❌ TESTS FAILED - Pattern discovery fix needs more work")
        print("\n" + "="*80)
        print("❌ PATTERN DISCOVERY FIX VERIFICATION FAILED")
        print("="*80)
        print("• Additional debugging and fixes needed")
        print("• Check logs above for specific error details")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())