-- Migration: Add previous_state column to sensor_events table
-- Purpose: Enable efficient event deduplication and state transition tracking
-- Date: 2025-07-30

-- Add the previous_state column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'sensor_events' 
        AND column_name = 'previous_state'
        AND table_schema = 'public'
    ) THEN
        ALTER TABLE sensor_events ADD COLUMN previous_state TEXT;
        
        -- Create index for efficient deduplication queries
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sensor_events_state_transitions 
        ON sensor_events (entity_id, timestamp DESC, state, previous_state);
        
        -- Create index for finding consecutive same states
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sensor_events_consecutive_states
        ON sensor_events (entity_id, state, timestamp DESC);
        
        RAISE NOTICE 'Added previous_state column and indices to sensor_events table';
    ELSE
        RAISE NOTICE 'previous_state column already exists in sensor_events table';
    END IF;
END $$;

-- Create a function to populate previous_state for existing data
CREATE OR REPLACE FUNCTION populate_previous_state(
    batch_size INTEGER DEFAULT 10000,
    entity_filter TEXT DEFAULT NULL
) 
RETURNS TABLE(
    processed_count BIGINT,
    updated_count BIGINT,
    entity_id TEXT
) 
LANGUAGE plpgsql AS $$
DECLARE
    _entity_id TEXT;
    _processed BIGINT := 0;
    _updated BIGINT := 0;
    _batch_processed BIGINT := 0;
    _batch_updated BIGINT := 0;
BEGIN
    -- Get all entities or filter to specific one
    FOR _entity_id IN 
        SELECT DISTINCT se.entity_id 
        FROM sensor_events se 
        WHERE (entity_filter IS NULL OR se.entity_id = entity_filter)
        ORDER BY se.entity_id
    LOOP
        -- Process each entity in batches
        WITH entity_events AS (
            SELECT 
                timestamp,
                entity_id,
                state,
                LAG(state) OVER (ORDER BY timestamp) as prev_state
            FROM sensor_events 
            WHERE sensor_events.entity_id = _entity_id
                AND previous_state IS NULL  -- Only update rows that haven't been processed
            ORDER BY timestamp
            LIMIT batch_size
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
        SELECT COUNT(*) INTO _batch_updated FROM updated_rows;
        
        SELECT COUNT(*) INTO _batch_processed 
        FROM sensor_events 
        WHERE sensor_events.entity_id = _entity_id
            AND previous_state IS NULL
        LIMIT batch_size;
        
        _processed := _processed + _batch_processed;
        _updated := _updated + _batch_updated;
        
        -- Return progress for this entity
        RETURN QUERY SELECT _processed, _updated, _entity_id;
        
        -- Exit if no more rows to process for this entity
        EXIT WHEN _batch_processed = 0;
        
        -- Commit batch (important for large datasets)
        COMMIT;
    END LOOP;
END $$;

-- Create analysis function to identify duplicates
CREATE OR REPLACE FUNCTION analyze_duplicates(
    days_back INTEGER DEFAULT 7,
    entity_filter TEXT DEFAULT NULL
)
RETURNS TABLE(
    entity_id TEXT,
    state TEXT,
    duplicate_count BIGINT,
    avg_time_diff_seconds NUMERIC,
    min_time_diff_seconds NUMERIC,
    max_time_diff_seconds NUMERIC
)
LANGUAGE sql AS $$
    WITH duplicate_candidates AS (
        SELECT 
            se.entity_id,
            se.state,
            se.timestamp,
            LAG(se.state) OVER (PARTITION BY se.entity_id ORDER BY se.timestamp) as prev_state,
            LAG(se.timestamp) OVER (PARTITION BY se.entity_id ORDER BY se.timestamp) as prev_timestamp
        FROM sensor_events se
        WHERE se.timestamp >= NOW() - INTERVAL '%s days'
            AND (entity_filter IS NULL OR se.entity_id = entity_filter)
    ),
    duplicate_analysis AS (
        SELECT 
            dc.entity_id,
            dc.state,
            EXTRACT(EPOCH FROM dc.timestamp - dc.prev_timestamp) as time_diff_seconds
        FROM duplicate_candidates dc
        WHERE dc.state = dc.prev_state 
            AND EXTRACT(EPOCH FROM dc.timestamp - dc.prev_timestamp) <= 5  -- 5 second window
    )
    SELECT 
        da.entity_id,
        da.state,
        COUNT(*) as duplicate_count,
        AVG(da.time_diff_seconds) as avg_time_diff_seconds,
        MIN(da.time_diff_seconds) as min_time_diff_seconds,
        MAX(da.time_diff_seconds) as max_time_diff_seconds
    FROM duplicate_analysis da
    GROUP BY da.entity_id, da.state
    ORDER BY duplicate_count DESC;
$$;

COMMENT ON FUNCTION populate_previous_state IS 'Populates the previous_state column for existing sensor_events data in batches to avoid memory issues';
COMMENT ON FUNCTION analyze_duplicates IS 'Analyzes duplicate events within a time window to identify deduplication opportunities';