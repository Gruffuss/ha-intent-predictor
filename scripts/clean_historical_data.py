#!/usr/bin/env python3
"""
Historical Data Cleaning Script for Home Assistant Occupancy Sensor Events

This script enhances the sensor_events table by:
1. Adding a previous_state column if missing
2. Populating previous_state for existing data
3. Removing duplicate consecutive state events
4. Keeping only actual state transitions for HMM training

Designed to handle 1.18M+ events efficiently with chunked processing.
"""

import asyncio
import logging
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_cleaning.event_deduplicator import EventDeduplicator, analyze_duplicates, clean_historical_data
from config.config_loader import ConfigLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_cleaning.log')
    ]
)
logger = logging.getLogger(__name__)


async def run_migration():
    """Run the database migration to add previous_state column"""
    logger.info("Running database migration to add previous_state column...")
    
    config = ConfigLoader()
    db_config = config.get("database.timescale")
    connection_string = (
        f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    
    deduplicator = EventDeduplicator(connection_string)
    
    try:
        await deduplicator.initialize()
        
        # Read and execute migration SQL
        migration_path = project_root / "src" / "data_cleaning" / "migrations" / "add_previous_state.sql"
        
        if not migration_path.exists():
            raise FileNotFoundError(f"Migration file not found: {migration_path}")
        
        with open(migration_path, 'r') as f:
            migration_sql = f.read()
        
        async with deduplicator.session_factory() as session:
            # Execute migration in parts (split by ;)
            statements = [stmt.strip() for stmt in migration_sql.split(';') if stmt.strip()]
            
            for statement in statements:
                if statement.strip():
                    try:
                        from sqlalchemy import text
                        await session.execute(text(statement))
                        logger.debug(f"Executed: {statement[:100]}...")
                    except Exception as e:
                        logger.warning(f"Statement may have already been executed: {e}")
            
            await session.commit()
        
        logger.info("✅ Database migration completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        await deduplicator.close()


async def populate_previous_state(entity_filter=None, batch_size=10000):
    """Populate previous_state for existing data"""
    logger.info(f"Populating previous_state column (batch_size: {batch_size})...")
    
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
            
            # Get entities to process
            if entity_filter:
                entities_query = "SELECT DISTINCT entity_id FROM sensor_events WHERE entity_id = :entity_filter ORDER BY entity_id"
                params = {'entity_filter': entity_filter}
            else:
                entities_query = "SELECT DISTINCT entity_id FROM sensor_events ORDER BY entity_id"
                params = {}
            
            result = await session.execute(text(entities_query), params)
            entities = [row.entity_id for row in result.fetchall()]
            
            logger.info(f"Processing {len(entities)} entities...")
            
            total_processed = 0
            total_updated = 0
            
            for i, entity_id in enumerate(entities, 1):
                logger.info(f"Processing entity {i}/{len(entities)}: {entity_id}")
                
                # Process entity in batches
                entity_processed = 0
                entity_updated = 0
                
                while True:
                    # Update previous_state for this entity's events
                    update_query = """
                        WITH entity_events AS (
                            SELECT 
                                timestamp,
                                entity_id,
                                state,
                                LAG(state) OVER (ORDER BY timestamp) as prev_state,
                                ROW_NUMBER() OVER (ORDER BY timestamp) as rn
                            FROM sensor_events 
                            WHERE entity_id = :entity_id
                                AND previous_state IS NULL
                            ORDER BY timestamp
                            LIMIT :batch_size
                        ),
                        updated_rows AS (
                            UPDATE sensor_events 
                            SET previous_state = entity_events.prev_state
                            FROM entity_events
                            WHERE sensor_events.entity_id = entity_events.entity_id
                                AND sensor_events.timestamp = entity_events.timestamp
                                AND entity_events.prev_state IS NOT NULL
                            RETURNING 1
                        )
                        SELECT COUNT(*) as updated_count FROM updated_rows
                    """
                    
                    result = await session.execute(text(update_query), {
                        'entity_id': entity_id,
                        'batch_size': batch_size
                    })
                    
                    batch_updated = result.fetchone().updated_count
                    entity_updated += batch_updated
                    entity_processed += batch_size
                    
                    await session.commit()
                    
                    logger.debug(f"  Batch: {batch_updated} updated")
                    
                    # Break if no more updates
                    if batch_updated == 0:
                        break
                
                total_processed += entity_processed
                total_updated += entity_updated
                
                logger.info(f"  Entity {entity_id}: {entity_updated} records updated")
            
            logger.info(f"✅ Previous state population completed:")
            logger.info(f"  Total processed: {total_processed:,}")
            logger.info(f"  Total updated: {total_updated:,}")
            
    except Exception as e:
        logger.error(f"❌ Previous state population failed: {e}")
        raise
    finally:
        await deduplicator.close()


async def analyze_data_quality(days_back=7):
    """Analyze current data quality and duplicate patterns"""
    logger.info(f"Analyzing data quality for last {days_back} days...")
    
    try:
        analysis = await analyze_duplicates(days_back=days_back)
        
        logger.info("📊 Data Quality Analysis Results:")
        logger.info(f"Analysis period: {analysis['analysis_period']['start_date']} to {analysis['analysis_period']['end_date']}")
        logger.info(f"Total events: {analysis['summary']['total_events']:,}")
        logger.info(f"Total duplicates: {analysis['summary']['total_duplicates']:,}")
        logger.info(f"Duplicate rate: {analysis['summary']['duplicate_rate']:.2f}%")
        logger.info(f"Entities with duplicates: {analysis['summary']['entities_with_duplicates']}")
        
        logger.info("\n🔍 Top 10 Duplicate Patterns:")
        for i, pattern in enumerate(analysis['top_duplicate_patterns'][:10], 1):
            logger.info(f"  {i}. {pattern['entity_id']} (state: {pattern['state']})")
            logger.info(f"     Duplicates: {pattern['duplicate_count']:,}, Avg time diff: {pattern['avg_time_diff_seconds']:.2f}s")
        
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Data quality analysis failed: {e}")
        raise


async def run_deduplication(dry_run=True, days_back=180, chunk_size=10000):
    """Run the main deduplication process"""
    logger.info(f"Starting deduplication process (dry_run: {dry_run}, days_back: {days_back})...")
    
    try:
        metrics = await clean_historical_data(
            days_back=days_back,
            dry_run=dry_run
        )
        
        logger.info("🎯 Deduplication Results:")
        logger.info(f"Total events processed: {metrics.total_events_processed:,}")
        logger.info(f"Duplicate events removed: {metrics.duplicate_events_removed:,}")
        logger.info(f"State transitions kept: {metrics.state_transitions_kept:,}")
        logger.info(f"Duplicate rate: {metrics.duplicate_rate:.2f}%")
        logger.info(f"Processing rate: {metrics.events_per_second:.1f} events/sec")
        logger.info(f"Processing time: {metrics.processing_time_seconds:.1f} seconds")
        logger.info(f"Entities processed: {len(metrics.entities_processed)}")
        logger.info(f"Chunks processed: {metrics.chunks_processed}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Deduplication failed: {e}")
        raise


async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Clean Home Assistant historical sensor data")
    parser.add_argument("--migrate", action="store_true", help="Run database migration only")
    parser.add_argument("--populate", action="store_true", help="Populate previous_state column only")
    parser.add_argument("--analyze", action="store_true", help="Analyze data quality only")
    parser.add_argument("--clean", action="store_true", help="Run full deduplication")
    parser.add_argument("--all", action="store_true", help="Run all steps (migrate, populate, analyze, clean)")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Dry run mode (default: True)")
    parser.add_argument("--real-run", action="store_true", help="Actually modify data (overrides dry-run)")
    parser.add_argument("--days-back", type=int, default=180, help="Days of historical data to process")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for processing")
    parser.add_argument("--entity-filter", type=str, help="Process only specific entity_id")
    
    args = parser.parse_args()
    
    # Determine dry_run mode
    dry_run = args.dry_run and not args.real_run
    
    logger.info("🚀 Starting Home Assistant Data Cleaning Process")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'REAL RUN - WILL MODIFY DATA'}")
    logger.info(f"Days back: {args.days_back}")
    logger.info(f"Batch size: {args.batch_size}")
    if args.entity_filter:
        logger.info(f"Entity filter: {args.entity_filter}")
    
    start_time = datetime.now()
    
    try:
        # Step 1: Database Migration
        if args.migrate or args.all:
            logger.info("\n" + "="*50)
            logger.info("STEP 1: DATABASE MIGRATION")
            logger.info("="*50)
            await run_migration()
        
        # Step 2: Populate Previous State
        if args.populate or args.all:
            logger.info("\n" + "="*50)
            logger.info("STEP 2: POPULATE PREVIOUS STATE")
            logger.info("="*50)
            await populate_previous_state(
                entity_filter=args.entity_filter,
                batch_size=args.batch_size
            )
        
        # Step 3: Analyze Data Quality
        if args.analyze or args.all:
            logger.info("\n" + "="*50)
            logger.info("STEP 3: DATA QUALITY ANALYSIS")
            logger.info("="*50)
            await analyze_data_quality(days_back=min(args.days_back, 30))  # Limit analysis to 30 days
        
        # Step 4: Run Deduplication
        if args.clean or args.all:
            logger.info("\n" + "="*50)
            logger.info("STEP 4: DATA DEDUPLICATION")
            logger.info("="*50)
            await run_deduplication(
                dry_run=dry_run,
                days_back=args.days_back,
                chunk_size=args.batch_size
            )
        
        # Show usage if no specific action
        if not any([args.migrate, args.populate, args.analyze, args.clean, args.all]):
            logger.info("No action specified. Use --help for available options.")
            logger.info("Common usage:")
            logger.info("  python scripts/clean_historical_data.py --all --dry-run")
            logger.info("  python scripts/clean_historical_data.py --analyze")
            logger.info("  python scripts/clean_historical_data.py --clean --real-run")
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n✅ Process completed successfully in {total_time:.1f} seconds")
        
    except Exception as e:
        logger.error(f"\n❌ Process failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())