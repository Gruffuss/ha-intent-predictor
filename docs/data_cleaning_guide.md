# Data Cleaning Guide: Event Deduplication for HMM Training

## Overview

The Home Assistant occupancy prediction system needs clean, transition-only data for effective Hidden Markov Model (HMM) training. Raw sensor data contains many duplicate consecutive states that can skew model performance.

This guide covers the enhanced event deduplication system that:
- Removes duplicate consecutive state events (same entity_id with same state within seconds)
- Keeps only actual state changes (on→off, off→on transitions)
- Handles large datasets efficiently (1.18M+ events)
- Maintains data integrity with proper indexing and error handling

## Problem Statement

### Raw Data Issues
Home Assistant sensors often report duplicate states due to:
- **Sensor heartbeats**: Regular status updates with no actual change
- **Network retransmissions**: Duplicate messages due to connectivity issues
- **Component redundancy**: Multiple sensors reporting the same state
- **Timing precision**: Microsecond-level duplicates from rapid state changes

### Impact on HMM Training
Duplicate events create several problems:
- **State probability skew**: Overweighting certain states in training
- **Transition matrix distortion**: False transition patterns
- **Memory bloat**: Unnecessary computational overhead
- **Pattern noise**: Obscuring genuine occupancy patterns

## Solution Architecture

### Core Components

1. **EventDeduplicator Class** (`src/data_cleaning/event_deduplicator.py`)
   - High-performance async deduplication engine
   - Chunk-based processing for memory efficiency
   - Configurable time windows and thresholds
   - Comprehensive metrics and logging

2. **Database Migration** (`src/data_cleaning/migrations/add_previous_state.sql`)
   - Adds `previous_state` column to `sensor_events` table
   - Creates optimized indices for deduplication queries
   - Includes utility functions for analysis

3. **Cleaning Script** (`scripts/clean_historical_data.py`)
   - End-to-end orchestration of cleaning process
   - Handles migration, population, analysis, and deduplication
   - Configurable batch processing for large datasets

4. **Test Framework** (`scripts/test_deduplication.py`)
   - Validates deduplication logic on small samples
   - Verifies database connectivity and schema
   - Safe testing before full dataset processing

### Deduplication Algorithm

```
For each entity's events (ordered by timestamp):
  1. Initialize: previous_state = null, previous_timestamp = null
  2. For each event:
     a. If first event: Keep and update tracking
     b. If same state as previous:
        - Calculate time difference
        - If ≤ max_time_window_seconds: Mark as duplicate
        - If > max_time_window_seconds: Keep (valid re-assertion)
     c. If different state: Keep (genuine transition)
     d. Update tracking variables
  3. Delete all marked duplicates in batch
```

### Key Features

- **Adaptive chunking**: Automatically adjusts batch size based on data volume
- **Entity-wise processing**: Maintains state consistency per sensor
- **Time-window filtering**: Configurable duplicate detection threshold (default: 5 seconds)
- **Safe deletion**: Uses (timestamp, entity_id) composite keys for precise targeting
- **Progress tracking**: Comprehensive metrics and logging
- **Error resilience**: Graceful handling of schema variations and connection issues

## Current Database Status

### Schema Information
```sql
-- Current sensor_events table structure
CREATE TABLE sensor_events (
    timestamp TIMESTAMPTZ NOT NULL,
    entity_id TEXT NOT NULL,
    state TEXT NOT NULL,
    room TEXT,
    sensor_type TEXT,
    attributes JSONB,
    derived_features JSONB,
    previous_state TEXT,  -- Added by migration
    PRIMARY KEY (timestamp, entity_id)
);
```

### Data Volume
- **Total events**: 1,177,379
- **Unique entities**: 49 sensors
- **Date range**: ~180 days of historical data
- **Storage**: TimescaleDB hypertable with time-based partitioning

### Index Strategy
```sql
-- Core indices for deduplication performance
CREATE INDEX idx_sensor_events_state_transitions 
ON sensor_events (entity_id, timestamp DESC, state, previous_state);

CREATE INDEX idx_sensor_events_consecutive_states
ON sensor_events (entity_id, state, timestamp DESC);
```

## Usage Guide

### 1. Initial Setup

#### Run Database Migration
```bash
# SSH into the container
ssh ha-predictor

# Navigate to project directory
cd /opt/ha-intent-predictor && source venv/bin/activate

# Run migration to add previous_state column
python scripts/clean_historical_data.py --migrate
```

#### Test System
```bash
# Run comprehensive tests
python scripts/test_deduplication.py
```

### 2. Data Analysis

#### Analyze Duplicate Patterns
```bash
# Analyze last 7 days for duplicate patterns
python scripts/clean_historical_data.py --analyze

# Sample output:
# 📊 Data Quality Analysis Results:
# Total events: 15,432
# Total duplicates: 3,421 (22.1%)
# Entities with duplicates: 23
```

#### Get Detailed Analysis
```python
from src.data_cleaning.event_deduplicator import analyze_duplicates

# Analyze specific time period
analysis = await analyze_duplicates(days_back=30)
print(f"Duplicate rate: {analysis['summary']['duplicate_rate']:.2f}%")
```

### 3. Data Cleaning Process

#### Full Cleaning Workflow
```bash
# Complete process: migrate → populate → analyze → clean
python scripts/clean_historical_data.py --all --dry-run

# When satisfied with dry-run results, run for real
python scripts/clean_historical_data.py --all --real-run
```

#### Step-by-Step Process
```bash
# Step 1: Populate previous_state for existing data
python scripts/clean_historical_data.py --populate --batch-size 10000

# Step 2: Analyze data quality
python scripts/clean_historical_data.py --analyze

# Step 3: Run deduplication (dry run first)
python scripts/clean_historical_data.py --clean --dry-run

# Step 4: Run actual deduplication
python scripts/clean_historical_data.py --clean --real-run
```

#### Specific Entity Processing
```bash
# Process only a specific sensor
python scripts/clean_historical_data.py --clean --entity-filter "binary_sensor.living_room_motion" --real-run
```

### 4. Configuration Options

#### EventDeduplicator Parameters
```python
deduplicator = EventDeduplicator(
    chunk_size=10000,                    # Events per processing batch
    max_time_window_seconds=5,           # Duplicate detection window
    min_state_duration_seconds=1         # Minimum state hold time
)
```

#### Script Arguments
```bash
# Common options
--dry-run / --real-run      # Safety mode vs actual changes
--days-back 180             # Historical data range
--batch-size 10000          # Processing batch size
--entity-filter ENTITY_ID   # Process specific entity only
```

## Performance Considerations

### Processing Efficiency
- **Chunked processing**: Handles 1M+ events without memory issues
- **Batch operations**: Minimizes database round trips
- **Async architecture**: Concurrent processing where possible
- **Index optimization**: Query performance for large time ranges

### Expected Performance
- **Processing rate**: ~10,000-50,000 events/second (depending on duplicate ratio)
- **Memory usage**: ~100-200MB for typical batch sizes
- **Duplicate detection**: 15-30% duplicate rate typical for HA sensors
- **Storage reduction**: 20-40% reduction in event count

### Monitoring
```python
# Get processing metrics
metrics = await deduplicator.get_metrics()
print(f"Processed: {metrics['total_events_processed']:,} events")
print(f"Removed: {metrics['duplicate_events_removed']:,} duplicates")
print(f"Rate: {metrics['events_per_second']:.1f} events/sec")
```

## Data Quality Impact

### Before Deduplication
```
Entity: binary_sensor.living_room_motion
Time        State   Notes
10:00:00    on      Motion detected
10:00:01    on      Duplicate (heartbeat)
10:00:02    on      Duplicate (heartbeat)
10:00:15    off     Genuine state change
10:00:16    off     Duplicate (heartbeat)
```

### After Deduplication
```
Entity: binary_sensor.living_room_motion
Time        State   Previous   Notes
10:00:00    on      null       Initial state
10:00:15    off     on         Clean transition only
```

### HMM Training Benefits
- **Cleaner transitions**: Only genuine state changes
- **Accurate probabilities**: No duplicate state bias
- **Better patterns**: Clearer occupancy sequences
- **Faster training**: Reduced dataset size
- **Improved accuracy**: Less noise in training data

## Troubleshooting

### Common Issues

#### Schema Errors
```
Error: column "previous_state" does not exist
Solution: Run migration first: --migrate
```

#### Memory Issues
```
Error: Out of memory processing large chunks
Solution: Reduce --batch-size to 5000 or lower
```

#### Performance Problems
```
Issue: Slow processing on large datasets
Solutions:
- Reduce batch size
- Process specific entities: --entity-filter
- Limit time range: --days-back 30
```

#### Connection Timeouts
```
Error: Database connection timeout
Solution: Check Docker services:
docker compose ps
docker compose restart timescaledb
```

### Validation Queries

#### Check Duplicate Removal
```sql
-- Compare before/after counts
SELECT 
    entity_id,
    COUNT(*) as total_events,
    COUNT(DISTINCT timestamp) as unique_timestamps
FROM sensor_events
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY entity_id
ORDER BY total_events DESC;
```

#### Verify State Transitions
```sql
-- Check state transition patterns
SELECT 
    entity_id,
    state,
    previous_state,
    COUNT(*) as transition_count
FROM sensor_events
WHERE previous_state IS NOT NULL
    AND timestamp >= NOW() - INTERVAL '7 days'
GROUP BY entity_id, state, previous_state
ORDER BY transition_count DESC;
```

## Best Practices

### Production Deployment
1. **Always test first**: Use `--dry-run` and `test_deduplication.py`
2. **Backup data**: Create database backup before major cleaning
3. **Monitor progress**: Watch logs and metrics during processing
4. **Validate results**: Run analysis queries after cleaning
5. **Staged approach**: Process in smaller time chunks for large datasets

### Maintenance
- **Regular cleaning**: Schedule weekly deduplication for new data
- **Monitor duplicates**: Track duplicate rates to identify sensor issues
- **Index maintenance**: Reindex tables after major cleanings
- **Performance tuning**: Adjust batch sizes based on system performance

### Integration with ML Pipeline
```python
# Use cleaned data for HMM training
from src.learning.hmm_models import OccupancyHMM

# Load cleaned transition data
hmm = OccupancyHMM()
await hmm.train_from_clean_transitions(
    start_date=start_date,
    end_date=end_date,
    require_clean_data=True  # Only use deduplicated events
)
```

## Next Steps

After successful deduplication:
1. **Verify data quality**: Run analysis queries to confirm cleaning
2. **Retrain models**: Update HMM models with cleaned data
3. **Monitor performance**: Track prediction accuracy improvements
4. **Schedule maintenance**: Set up regular cleaning for new data
5. **Document results**: Record duplicate rates and performance gains

The cleaned, transition-only data will provide a much better foundation for HMM training and occupancy prediction accuracy.