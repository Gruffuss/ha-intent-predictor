#!/usr/bin/env python3
"""
Test Script for Event Deduplication

This script tests the event deduplication system on a small sample
of data to verify it works correctly before running on the full dataset.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_cleaning.event_deduplicator import EventDeduplicator, analyze_duplicates
from config.config_loader import ConfigLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_small_sample():
    """Test deduplication on a small sample of recent data"""
    logger.info("🧪 Testing event deduplication on a small sample...")
    
    config = ConfigLoader()
    db_config = config.get("database.timescale")
    connection_string = (
        f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    
    deduplicator = EventDeduplicator(
        connection_string=connection_string,
        chunk_size=1000,  # Small chunk for testing
        max_time_window_seconds=5
    )
    
    try:
        await deduplicator.initialize()
        
        # Test on last 24 hours of data
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=24)
        
        logger.info(f"Testing period: {start_date} to {end_date}")
        
        # Get sample data overview first
        async with deduplicator.session_factory() as session:
            from sqlalchemy import text
            
            # Count total events in test period
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as total_events,
                    COUNT(DISTINCT entity_id) as unique_entities,
                    MIN(timestamp) as earliest,
                    MAX(timestamp) as latest
                FROM sensor_events
                WHERE timestamp >= :start_date AND timestamp <= :end_date
            """), {'start_date': start_date, 'end_date': end_date})
            
            stats = result.fetchone()
            logger.info(f"Sample data: {stats.total_events:,} events from {stats.unique_entities} entities")
            
            if stats.total_events == 0:
                logger.warning("No events found in test period. Expanding to last 7 days...")
                start_date = end_date - timedelta(days=7)
                
                result = await session.execute(text("""
                    SELECT 
                        COUNT(*) as total_events,
                        COUNT(DISTINCT entity_id) as unique_entities
                    FROM sensor_events
                    WHERE timestamp >= :start_date AND timestamp <= :end_date
                """), {'start_date': start_date, 'end_date': end_date})
                
                stats = result.fetchone()
                logger.info(f"Expanded sample: {stats.total_events:,} events from {stats.unique_entities} entities")
            
            # Get a sample of entities with most events
            result = await session.execute(text("""
                SELECT entity_id, COUNT(*) as event_count
                FROM sensor_events
                WHERE timestamp >= :start_date AND timestamp <= :end_date
                GROUP BY entity_id
                ORDER BY event_count DESC
                LIMIT 5
            """), {'start_date': start_date, 'end_date': end_date})
            
            logger.info("Top entities by event count:")
            test_entities = []
            for row in result.fetchall():
                logger.info(f"  {row.entity_id}: {row.event_count:,} events")
                test_entities.append(row.entity_id)
        
        # Run analysis first (dry run)
        logger.info("\n📊 Running duplicate analysis...")
        analysis = await deduplicator.get_duplicate_analysis(
            start_date=start_date,
            end_date=end_date,
            limit=20
        )
        
        logger.info(f"Analysis results:")
        logger.info(f"  Total events: {analysis['summary']['total_events']:,}")
        logger.info(f"  Total duplicates: {analysis['summary']['total_duplicates']:,}")
        logger.info(f"  Duplicate rate: {analysis['summary']['duplicate_rate']:.2f}%")
        logger.info(f"  Entities with duplicates: {analysis['summary']['entities_with_duplicates']}")
        
        if analysis['top_duplicate_patterns']:
            logger.info("\nTop duplicate patterns:")
            for i, pattern in enumerate(analysis['top_duplicate_patterns'][:5], 1):
                logger.info(f"  {i}. {pattern['entity_id']} (state: {pattern['state']})")
                logger.info(f"     Duplicates: {pattern['duplicate_count']}, Avg gap: {pattern['avg_time_diff_seconds']:.2f}s")
        
        # Run deduplication in dry-run mode
        logger.info("\n🔍 Running deduplication (DRY RUN)...")
        metrics = await deduplicator.deduplicate_historical_data(
            start_date=start_date,
            end_date=end_date,
            entity_ids=test_entities[:2] if test_entities else None,  # Test on top 2 entities
            dry_run=True
        )
        
        logger.info("Deduplication test results:")
        logger.info(f"  Events analyzed: {metrics.total_events_processed:,}")
        logger.info(f"  Duplicates found: {metrics.duplicate_events_removed:,}")
        logger.info(f"  Transitions kept: {metrics.state_transitions_kept:,}")
        logger.info(f"  Duplicate rate: {metrics.duplicate_rate:.2f}%")
        logger.info(f"  Processing rate: {metrics.events_per_second:.1f} events/sec")
        
        # Show some sample duplicate patterns
        if metrics.duplicate_events_removed > 0:
            logger.info("\n✅ Test completed successfully! Duplicates detected and would be removed.")
            return True
        else:
            logger.info("\n⚠️  No duplicates found in test sample. This is normal for recent data.")
            return True
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
    finally:
        await deduplicator.close()


async def test_database_connection():
    """Test basic database connectivity"""
    logger.info("🔌 Testing database connection...")
    
    config = ConfigLoader()
    db_config = config.get("database.timescale")
    connection_string = (
        f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    
    deduplicator = EventDeduplicator(connection_string)
    
    try:
        await deduplicator.initialize()
        
        async with deduplicator.session_factory() as session:
            from sqlalchemy import text
            
            # Test basic query
            result = await session.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            logger.info(f"Database version: {version}")
            
            # Check sensor_events table
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as total_events,
                    COUNT(DISTINCT entity_id) as unique_entities,
                    MIN(timestamp) as earliest,
                    MAX(timestamp) as latest
                FROM sensor_events
            """))
            
            stats = result.fetchone()
            logger.info(f"Total events: {stats.total_events:,}")
            logger.info(f"Unique entities: {stats.unique_entities}")
            logger.info(f"Data range: {stats.earliest} to {stats.latest}")
            
            # Check if previous_state column exists
            result = await session.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'sensor_events' 
                AND column_name = 'previous_state'
            """))
            
            has_previous_state = result.fetchone() is not None
            logger.info(f"Has previous_state column: {has_previous_state}")
            
        logger.info("✅ Database connection test successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        return False
    finally:
        await deduplicator.close()


async def main():
    """Main test function"""
    logger.info("🚀 Starting Event Deduplication Tests")
    
    # Test 1: Database Connection
    logger.info("\n" + "="*50)
    logger.info("TEST 1: DATABASE CONNECTION")
    logger.info("="*50)
    
    if not await test_database_connection():
        logger.error("Database connection test failed. Aborting.")
        return
    
    # Test 2: Small Sample Deduplication
    logger.info("\n" + "="*50)
    logger.info("TEST 2: SMALL SAMPLE DEDUPLICATION")
    logger.info("="*50)
    
    if not await test_small_sample():
        logger.error("Sample deduplication test failed.")
        return
    
    logger.info("\n✅ All tests completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Run migration: python scripts/clean_historical_data.py --migrate")
    logger.info("2. Populate previous_state: python scripts/clean_historical_data.py --populate")
    logger.info("3. Analyze data: python scripts/clean_historical_data.py --analyze")
    logger.info("4. Clean data: python scripts/clean_historical_data.py --clean --real-run")


if __name__ == "__main__":
    asyncio.run(main())