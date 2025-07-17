-- TimescaleDB Schema for HA Intent Prediction System
-- Implements the exact data storage architecture from CLAUDE.md

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create database tables for comprehensive sensor data storage
-- Following CLAUDE.md requirements for aggressive data collection

-- 1. Raw sensor events table - stores all sensor state changes
CREATE TABLE IF NOT EXISTS sensor_events (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    entity_id VARCHAR(255) NOT NULL,
    state VARCHAR(50) NOT NULL,
    previous_state VARCHAR(50),
    attributes JSONB,
    room VARCHAR(100),
    sensor_type VARCHAR(50),
    zone_type VARCHAR(50), -- 'full_zone', 'subzone', 'entrance_zone', 'hallway_zone'
    zone_name VARCHAR(100),
    person_specific VARCHAR(50), -- 'anca', 'vladimir', or NULL
    activity_type VARCHAR(50), -- 'sleeping', 'work', 'cooking', 'transit', etc.
    
    -- Derived fields for enrichment
    time_since_last_change INTERVAL,
    state_transition VARCHAR(100), -- 'off_to_on', 'on_to_off', etc.
    concurrent_events INTEGER DEFAULT 0,
    
    -- Anomaly detection fields
    anomaly_score FLOAT DEFAULT 0.0,
    is_anomaly BOOLEAN DEFAULT FALSE,
    anomaly_type VARCHAR(50), -- 'cat_activity', 'impossible_movement', etc.
    
    -- Indexing for fast queries
    INDEX idx_sensor_events_timestamp ON sensor_events (timestamp DESC),
    INDEX idx_sensor_events_entity ON sensor_events (entity_id, timestamp DESC),
    INDEX idx_sensor_events_room ON sensor_events (room, timestamp DESC),
    INDEX idx_sensor_events_person ON sensor_events (person_specific, timestamp DESC) WHERE person_specific IS NOT NULL,
    INDEX idx_sensor_events_anomaly ON sensor_events (is_anomaly, timestamp DESC) WHERE is_anomaly = TRUE
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('sensor_events', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- 2. Computed features table - stores dynamically discovered features
CREATE TABLE IF NOT EXISTS computed_features (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    room VARCHAR(100) NOT NULL,
    
    -- Dynamic feature storage - flexible schema
    features JSONB NOT NULL,
    
    -- Feature metadata
    feature_count INTEGER,
    feature_names TEXT[],
    feature_importance JSONB,
    
    -- Context information
    context_window_minutes INTEGER,
    data_points_used INTEGER,
    
    -- Indexing
    INDEX idx_computed_features_timestamp ON computed_features (timestamp DESC),
    INDEX idx_computed_features_room ON computed_features (room, timestamp DESC),
    INDEX idx_computed_features_gin ON computed_features USING gin (features)
);

-- Convert to hypertable
SELECT create_hypertable('computed_features', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- 3. Occupancy predictions table - stores all predictions with metadata
CREATE TABLE IF NOT EXISTS occupancy_predictions (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    room VARCHAR(100) NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    
    -- Prediction results
    probability FLOAT NOT NULL,
    uncertainty FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    
    -- Model information
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    ensemble_weights JSONB,
    contributing_models JSONB,
    
    -- Feature attribution
    feature_importance JSONB,
    top_features JSONB,
    
    -- Actual outcome (for learning)
    actual_outcome BOOLEAN,
    prediction_error FLOAT,
    
    -- Metadata
    processing_time_ms INTEGER,
    explanation TEXT,
    
    -- Indexing
    INDEX idx_occupancy_predictions_timestamp ON occupancy_predictions (timestamp DESC),
    INDEX idx_occupancy_predictions_room ON occupancy_predictions (room, timestamp DESC),
    INDEX idx_occupancy_predictions_horizon ON occupancy_predictions (horizon_minutes, timestamp DESC),
    INDEX idx_occupancy_predictions_accuracy ON occupancy_predictions (prediction_error, timestamp DESC) WHERE actual_outcome IS NOT NULL
);

-- Convert to hypertable
SELECT create_hypertable('occupancy_predictions', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- 4. Discovered patterns table - stores patterns found by pattern discovery
CREATE TABLE IF NOT EXISTS discovered_patterns (
    id BIGSERIAL PRIMARY KEY,
    discovered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    room VARCHAR(100) NOT NULL,
    
    -- Pattern description
    pattern_type VARCHAR(100) NOT NULL, -- 'temporal', 'sequential', 'contextual', etc.
    pattern_data JSONB NOT NULL,
    
    -- Pattern statistics
    frequency INTEGER NOT NULL,
    confidence FLOAT NOT NULL,
    support FLOAT NOT NULL,
    significance_score FLOAT NOT NULL,
    
    -- Temporal information
    time_window_minutes INTEGER,
    valid_from TIMESTAMPTZ,
    valid_to TIMESTAMPTZ,
    
    -- Pattern metadata
    sensor_entities TEXT[],
    conditions JSONB,
    
    -- Performance tracking
    prediction_improvement FLOAT,
    usage_count INTEGER DEFAULT 0,
    last_used TIMESTAMPTZ,
    
    -- Indexing
    INDEX idx_discovered_patterns_room ON discovered_patterns (room, discovered_at DESC),
    INDEX idx_discovered_patterns_type ON discovered_patterns (pattern_type, discovered_at DESC),
    INDEX idx_discovered_patterns_significance ON discovered_patterns (significance_score DESC),
    INDEX idx_discovered_patterns_gin ON discovered_patterns USING gin (pattern_data)
);

-- 5. Model performance table - tracks model accuracy over time
CREATE TABLE IF NOT EXISTS model_performance (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    room VARCHAR(100) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    
    -- Performance metrics
    accuracy FLOAT NOT NULL,
    precision_score FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    roc_auc FLOAT,
    
    -- Time-based metrics
    window_size INTEGER,
    evaluation_period INTERVAL,
    
    -- Model metadata
    model_version VARCHAR(50),
    hyperparameters JSONB,
    training_data_points INTEGER,
    
    -- Drift detection
    drift_score FLOAT,
    drift_detected BOOLEAN DEFAULT FALSE,
    
    -- Indexing
    INDEX idx_model_performance_timestamp ON model_performance (timestamp DESC),
    INDEX idx_model_performance_room_model ON model_performance (room, model_name, timestamp DESC),
    INDEX idx_model_performance_accuracy ON model_performance (accuracy DESC, timestamp DESC),
    INDEX idx_model_performance_drift ON model_performance (drift_detected, timestamp DESC) WHERE drift_detected = TRUE
);

-- Convert to hypertable
SELECT create_hypertable('model_performance', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- 6. Anomaly events table - stores detected anomalies (cat activity, etc.)
CREATE TABLE IF NOT EXISTS anomaly_events (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    room VARCHAR(100),
    
    -- Anomaly classification
    anomaly_type VARCHAR(100) NOT NULL, -- 'cat_activity', 'impossible_movement', 'sensor_malfunction'
    anomaly_score FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    
    -- Anomaly details
    sensor_sequence JSONB,
    triggering_events JSONB,
    explanation TEXT,
    
    -- Detection metadata
    detector_version VARCHAR(50),
    detection_method VARCHAR(100),
    
    -- Human feedback (for learning)
    human_verified BOOLEAN,
    human_feedback TEXT,
    feedback_timestamp TIMESTAMPTZ,
    
    -- Indexing
    INDEX idx_anomaly_events_timestamp ON anomaly_events (timestamp DESC),
    INDEX idx_anomaly_events_type ON anomaly_events (anomaly_type, timestamp DESC),
    INDEX idx_anomaly_events_score ON anomaly_events (anomaly_score DESC),
    INDEX idx_anomaly_events_room ON anomaly_events (room, timestamp DESC) WHERE room IS NOT NULL
);

-- Convert to hypertable
SELECT create_hypertable('anomaly_events', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- 7. System metrics table - tracks system performance and resource usage
CREATE TABLE IF NOT EXISTS system_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- System resource metrics
    cpu_usage FLOAT,
    memory_usage FLOAT,
    disk_usage FLOAT,
    
    -- Processing metrics
    events_processed_per_second INTEGER,
    predictions_per_second INTEGER,
    average_prediction_time_ms FLOAT,
    
    -- Model metrics
    active_models INTEGER,
    model_updates_per_hour INTEGER,
    feature_count INTEGER,
    
    -- Data metrics
    sensor_events_per_hour INTEGER,
    data_ingestion_rate_mb_per_second FLOAT,
    
    -- Error tracking
    error_count INTEGER DEFAULT 0,
    warning_count INTEGER DEFAULT 0,
    
    -- Indexing
    INDEX idx_system_metrics_timestamp ON system_metrics (timestamp DESC)
);

-- Convert to hypertable
SELECT create_hypertable('system_metrics', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- 8. Model configurations table - stores model hyperparameters and versions
CREATE TABLE IF NOT EXISTS model_configurations (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    room VARCHAR(100) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    
    -- Model configuration
    hyperparameters JSONB NOT NULL,
    features_used TEXT[],
    training_config JSONB,
    
    -- Performance summary
    validation_accuracy FLOAT,
    cross_validation_score FLOAT,
    
    -- Deployment info
    deployed_at TIMESTAMPTZ,
    deprecated_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Indexing
    INDEX idx_model_configurations_room ON model_configurations (room, created_at DESC),
    INDEX idx_model_configurations_active ON model_configurations (is_active, created_at DESC) WHERE is_active = TRUE
);

-- Create compression policies for data retention (180 days as specified in CLAUDE.md)
SELECT add_compression_policy('sensor_events', INTERVAL '7 days');
SELECT add_compression_policy('computed_features', INTERVAL '7 days');
SELECT add_compression_policy('occupancy_predictions', INTERVAL '7 days');
SELECT add_compression_policy('model_performance', INTERVAL '7 days');
SELECT add_compression_policy('anomaly_events', INTERVAL '7 days');
SELECT add_compression_policy('system_metrics', INTERVAL '1 day');

-- Create retention policies (180 days)
SELECT add_retention_policy('sensor_events', INTERVAL '180 days');
SELECT add_retention_policy('computed_features', INTERVAL '180 days');
SELECT add_retention_policy('occupancy_predictions', INTERVAL '180 days');
SELECT add_retention_policy('discovered_patterns', INTERVAL '365 days'); -- Keep patterns longer
SELECT add_retention_policy('model_performance', INTERVAL '180 days');
SELECT add_retention_policy('anomaly_events', INTERVAL '180 days');
SELECT add_retention_policy('system_metrics', INTERVAL '90 days');

-- Create continuous aggregates for performance monitoring
CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_events_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', timestamp) AS bucket,
    entity_id,
    room,
    sensor_type,
    COUNT(*) AS event_count,
    COUNT(DISTINCT entity_id) AS unique_sensors,
    AVG(CASE WHEN is_anomaly THEN 1.0 ELSE 0.0 END) AS anomaly_rate
FROM sensor_events
GROUP BY bucket, entity_id, room, sensor_type;

CREATE MATERIALIZED VIEW IF NOT EXISTS prediction_accuracy_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', timestamp) AS bucket,
    room,
    model_name,
    horizon_minutes,
    AVG(probability) AS avg_probability,
    AVG(uncertainty) AS avg_uncertainty,
    AVG(prediction_error) AS avg_error,
    COUNT(*) AS prediction_count
FROM occupancy_predictions
WHERE actual_outcome IS NOT NULL
GROUP BY bucket, room, model_name, horizon_minutes;

-- Create refresh policy for continuous aggregates
SELECT add_continuous_aggregate_policy('sensor_events_hourly',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('prediction_accuracy_hourly',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

-- Create functions for common queries
CREATE OR REPLACE FUNCTION get_recent_sensor_events(
    p_room VARCHAR,
    p_hours INTEGER DEFAULT 24
) RETURNS TABLE (
    timestamp TIMESTAMPTZ,
    entity_id VARCHAR,
    state VARCHAR,
    zone_type VARCHAR,
    anomaly_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        se.timestamp,
        se.entity_id,
        se.state,
        se.zone_type,
        se.anomaly_score
    FROM sensor_events se
    WHERE se.room = p_room
      AND se.timestamp > NOW() - INTERVAL '1 hour' * p_hours
    ORDER BY se.timestamp DESC;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_prediction_accuracy(
    p_room VARCHAR,
    p_model_name VARCHAR,
    p_days INTEGER DEFAULT 7
) RETURNS TABLE (
    accuracy FLOAT,
    precision_score FLOAT,
    recall FLOAT,
    prediction_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        AVG(op.accuracy) as accuracy,
        AVG(op.precision_score) as precision_score,
        AVG(op.recall) as recall,
        COUNT(*) as prediction_count
    FROM model_performance op
    WHERE op.room = p_room
      AND op.model_name = p_model_name
      AND op.timestamp > NOW() - INTERVAL '1 day' * p_days;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic data enrichment
CREATE OR REPLACE FUNCTION enrich_sensor_event() RETURNS TRIGGER AS $$
BEGIN
    -- Calculate time since last change for this entity
    SELECT 
        NEW.timestamp - MAX(timestamp) INTO NEW.time_since_last_change
    FROM sensor_events 
    WHERE entity_id = NEW.entity_id 
      AND timestamp < NEW.timestamp;
    
    -- Determine state transition
    IF NEW.previous_state IS NOT NULL THEN
        NEW.state_transition = NEW.previous_state || '_to_' || NEW.state;
    END IF;
    
    -- Count concurrent events (within 5 seconds)
    SELECT COUNT(*) INTO NEW.concurrent_events
    FROM sensor_events
    WHERE timestamp BETWEEN NEW.timestamp - INTERVAL '5 seconds' 
                       AND NEW.timestamp + INTERVAL '5 seconds'
      AND entity_id != NEW.entity_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_enrich_sensor_event
    BEFORE INSERT ON sensor_events
    FOR EACH ROW EXECUTE FUNCTION enrich_sensor_event();

-- Create indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_sensor_events_composite 
ON sensor_events (room, timestamp DESC, entity_id, state);

CREATE INDEX IF NOT EXISTS idx_predictions_composite 
ON occupancy_predictions (room, horizon_minutes, timestamp DESC, probability);

CREATE INDEX IF NOT EXISTS idx_performance_composite 
ON model_performance (room, model_name, horizon_minutes, timestamp DESC);

-- Grant appropriate permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO ha_predictor;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO ha_predictor;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO ha_predictor;

-- Create user for application
CREATE USER IF NOT EXISTS ha_predictor WITH PASSWORD 'ha_predictor_password';
GRANT ALL PRIVILEGES ON DATABASE ha_predictor TO ha_predictor;