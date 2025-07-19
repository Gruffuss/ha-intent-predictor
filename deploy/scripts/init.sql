-- HA Intent Predictor Database Initialization
-- TimescaleDB optimized schema for time-series data

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Enable other useful extensions
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- Create schemas for organization
CREATE SCHEMA IF NOT EXISTS raw_data;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS models;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Raw sensor events table
CREATE TABLE IF NOT EXISTS raw_data.sensor_events (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    state VARCHAR(50),
    old_state VARCHAR(50),
    attributes JSONB,
    room VARCHAR(100),
    sensor_type VARCHAR(100),
    zone_info JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Room occupancy states (ground truth)
CREATE TABLE IF NOT EXISTS raw_data.room_occupancy (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    room VARCHAR(100) NOT NULL,
    occupied BOOLEAN NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    detected_by VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Computed features table
CREATE TABLE IF NOT EXISTS features.computed_features (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    room VARCHAR(100) NOT NULL,
    feature_set VARCHAR(100) NOT NULL,
    features JSONB NOT NULL,
    computation_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Pattern discoveries
CREATE TABLE IF NOT EXISTS features.discovered_patterns (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    room VARCHAR(100) NOT NULL,
    pattern_type VARCHAR(100) NOT NULL,
    pattern_data JSONB NOT NULL,
    significance_score FLOAT NOT NULL,
    occurrences INTEGER DEFAULT 1,
    first_seen TIMESTAMPTZ DEFAULT NOW(),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Model predictions
CREATE TABLE IF NOT EXISTS models.predictions (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    room VARCHAR(100) NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    predicted_for TIMESTAMPTZ NOT NULL,
    probability FLOAT NOT NULL,
    uncertainty FLOAT NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    features_used JSONB,
    prediction_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Model performance metrics
CREATE TABLE IF NOT EXISTS models.performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    room VARCHAR(100) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    metric_type VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    horizon_minutes INTEGER,
    sample_size INTEGER,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Model states and configurations
CREATE TABLE IF NOT EXISTS models.model_states (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    room VARCHAR(100) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    state_data BYTEA,
    hyperparameters JSONB,
    feature_importance JSONB,
    training_samples INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Anomaly detection results
CREATE TABLE IF NOT EXISTS analytics.anomalies (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    room VARCHAR(100),
    anomaly_type VARCHAR(100) NOT NULL,
    anomaly_score FLOAT NOT NULL,
    description TEXT,
    related_events JSONB,
    is_cat_activity BOOLEAN DEFAULT FALSE,
    human_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- System events and logs
CREATE TABLE IF NOT EXISTS analytics.system_events (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    context JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertables for time-series optimization
SELECT create_hypertable('raw_data.sensor_events', 'timestamp', 
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

SELECT create_hypertable('raw_data.room_occupancy', 'timestamp', 
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

SELECT create_hypertable('features.computed_features', 'timestamp', 
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

SELECT create_hypertable('features.discovered_patterns', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

SELECT create_hypertable('models.predictions', 'timestamp', 
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

SELECT create_hypertable('models.performance_metrics', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

SELECT create_hypertable('models.model_states', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

SELECT create_hypertable('analytics.anomalies', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

SELECT create_hypertable('analytics.system_events', 'timestamp', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Create indexes for optimal query performance
CREATE INDEX IF NOT EXISTS idx_sensor_events_entity_time ON raw_data.sensor_events (entity_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_events_room_time ON raw_data.sensor_events (room, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_events_type_time ON raw_data.sensor_events (sensor_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_events_zone_info ON raw_data.sensor_events USING GIN (zone_info);
CREATE INDEX IF NOT EXISTS idx_sensor_events_attributes ON raw_data.sensor_events USING GIN (attributes);

CREATE INDEX IF NOT EXISTS idx_room_occupancy_room_time ON raw_data.room_occupancy (room, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_room_occupancy_occupied ON raw_data.room_occupancy (occupied, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_features_room_time ON features.computed_features (room, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_features_set_time ON features.computed_features (feature_set, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_features_data ON features.computed_features USING GIN (features);

CREATE INDEX IF NOT EXISTS idx_patterns_room_type ON features.discovered_patterns (room, pattern_type);
CREATE INDEX IF NOT EXISTS idx_patterns_significance ON features.discovered_patterns (significance_score DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_data ON features.discovered_patterns USING GIN (pattern_data);

CREATE INDEX IF NOT EXISTS idx_predictions_room_time ON models.predictions (room, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_horizon ON models.predictions (horizon_minutes, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_for_time ON models.predictions (predicted_for);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON models.predictions (model_name, model_version);

CREATE INDEX IF NOT EXISTS idx_performance_model_time ON models.performance_metrics (model_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_performance_room_metric ON models.performance_metrics (room, metric_type);
CREATE INDEX IF NOT EXISTS idx_performance_horizon ON models.performance_metrics (horizon_minutes);

CREATE INDEX IF NOT EXISTS idx_model_states_room_model ON models.model_states (room, model_name);
CREATE INDEX IF NOT EXISTS idx_model_states_version ON models.model_states (model_version, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_anomalies_room_time ON analytics.anomalies (room, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_anomalies_type_score ON analytics.anomalies (anomaly_type, anomaly_score DESC);
CREATE INDEX IF NOT EXISTS idx_anomalies_cat ON analytics.anomalies (is_cat_activity, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_system_events_type ON analytics.system_events (event_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_events_severity ON analytics.system_events (severity, timestamp DESC);

-- Create compression policies for older data
SELECT add_compression_policy('raw_data.sensor_events', INTERVAL '7 days');
SELECT add_compression_policy('features.computed_features', INTERVAL '7 days');
SELECT add_compression_policy('models.predictions', INTERVAL '7 days');
SELECT add_compression_policy('models.performance_metrics', INTERVAL '30 days');
SELECT add_compression_policy('analytics.anomalies', INTERVAL '30 days');
SELECT add_compression_policy('analytics.system_events', INTERVAL '30 days');

-- Create retention policies for very old data
SELECT add_retention_policy('raw_data.sensor_events', INTERVAL '1 year');
SELECT add_retention_policy('features.computed_features', INTERVAL '1 year');
SELECT add_retention_policy('models.predictions', INTERVAL '6 months');
SELECT add_retention_policy('models.performance_metrics', INTERVAL '2 years');
SELECT add_retention_policy('analytics.system_events', INTERVAL '1 year');

-- Create continuous aggregates for common queries
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.hourly_room_occupancy
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', timestamp) AS hour,
    room,
    AVG(CASE WHEN occupied THEN 1 ELSE 0 END) AS occupancy_rate,
    COUNT(*) AS measurements,
    AVG(confidence) AS avg_confidence
FROM raw_data.room_occupancy
GROUP BY hour, room
WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.daily_prediction_accuracy
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', p.timestamp) AS day,
    p.room,
    p.horizon_minutes,
    p.model_name,
    AVG(ABS(p.probability - CASE WHEN o.occupied THEN 1 ELSE 0 END)) AS mae,
    COUNT(*) AS predictions_made
FROM models.predictions p
JOIN raw_data.room_occupancy o ON o.room = p.room 
    AND o.timestamp >= p.predicted_for - INTERVAL '5 minutes'
    AND o.timestamp <= p.predicted_for + INTERVAL '5 minutes'
GROUP BY day, p.room, p.horizon_minutes, p.model_name
WITH NO DATA;

-- Add refresh policies for continuous aggregates
SELECT add_continuous_aggregate_policy('analytics.hourly_room_occupancy',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('analytics.daily_prediction_accuracy',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');

-- Create useful views for common queries
CREATE VIEW analytics.current_room_states AS
SELECT DISTINCT ON (room) 
    room, 
    occupied, 
    confidence,
    timestamp,
    detected_by
FROM raw_data.room_occupancy
ORDER BY room, timestamp DESC;

CREATE VIEW analytics.latest_predictions AS
SELECT DISTINCT ON (room, horizon_minutes)
    room,
    horizon_minutes,
    predicted_for,
    probability,
    uncertainty,
    model_name,
    timestamp
FROM models.predictions
ORDER BY room, horizon_minutes, timestamp DESC;

-- Create functions for common operations
CREATE OR REPLACE FUNCTION get_room_occupancy_probability(
    room_name VARCHAR(100),
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ
) RETURNS FLOAT AS $$
BEGIN
    RETURN (
        SELECT AVG(CASE WHEN occupied THEN 1 ELSE 0 END)
        FROM raw_data.room_occupancy
        WHERE room = room_name 
        AND timestamp >= start_time 
        AND timestamp <= end_time
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_model_accuracy(
    model_name_param VARCHAR(100),
    room_name VARCHAR(100),
    horizon_minutes_param INTEGER,
    days_back INTEGER DEFAULT 7
) RETURNS FLOAT AS $$
BEGIN
    RETURN (
        SELECT AVG(ABS(p.probability - CASE WHEN o.occupied THEN 1 ELSE 0 END))
        FROM models.predictions p
        JOIN raw_data.room_occupancy o ON o.room = p.room 
            AND o.timestamp >= p.predicted_for - INTERVAL '2 minutes'
            AND o.timestamp <= p.predicted_for + INTERVAL '2 minutes'
        WHERE p.model_name = model_name_param
        AND p.room = room_name
        AND p.horizon_minutes = horizon_minutes_param
        AND p.timestamp >= NOW() - (days_back || ' days')::INTERVAL
    );
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic data quality checks
CREATE OR REPLACE FUNCTION validate_sensor_event() RETURNS TRIGGER AS $$
BEGIN
    -- Check for reasonable timestamps
    IF NEW.timestamp > NOW() + INTERVAL '1 hour' OR NEW.timestamp < NOW() - INTERVAL '1 year' THEN
        RAISE WARNING 'Sensor event timestamp seems unreasonable: %', NEW.timestamp;
    END IF;
    
    -- Check for required fields
    IF NEW.entity_id IS NULL OR NEW.room IS NULL THEN
        RAISE EXCEPTION 'Missing required fields in sensor event';
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_validate_sensor_event
    BEFORE INSERT ON raw_data.sensor_events
    FOR EACH ROW EXECUTE FUNCTION validate_sensor_event();

-- Create user for the application
-- Note: Password should be set via environment variable or secure method
CREATE USER ha_predictor_app;

-- Grant necessary permissions
GRANT CONNECT ON DATABASE ha_predictor TO ha_predictor_app;
GRANT USAGE ON SCHEMA raw_data, features, models, analytics TO ha_predictor_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA raw_data TO ha_predictor_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA features TO ha_predictor_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA models TO ha_predictor_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA analytics TO ha_predictor_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA raw_data TO ha_predictor_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA features TO ha_predictor_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA models TO ha_predictor_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA analytics TO ha_predictor_app;

-- Create read-only user for analytics/monitoring
-- Note: Password should be set via environment variable or secure method  
CREATE USER ha_predictor_readonly;
GRANT CONNECT ON DATABASE ha_predictor TO ha_predictor_readonly;
GRANT USAGE ON SCHEMA raw_data, features, models, analytics TO ha_predictor_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA raw_data TO ha_predictor_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA features TO ha_predictor_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA models TO ha_predictor_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO ha_predictor_readonly;

-- Insert initial system event
INSERT INTO analytics.system_events (timestamp, event_type, severity, message, context)
VALUES (NOW(), 'database_init', 'INFO', 'Database schema initialized successfully', 
        jsonb_build_object('version', '1.0.0', 'tables_created', 9, 'views_created', 2));

-- Display initialization summary
DO $$
BEGIN
    RAISE NOTICE 'HA Intent Predictor Database Initialization Complete';
    RAISE NOTICE 'Tables created: %', (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema IN ('raw_data', 'features', 'models', 'analytics'));
    RAISE NOTICE 'Hypertables created: %', (SELECT COUNT(*) FROM timescaledb_information.hypertables);
    RAISE NOTICE 'Continuous aggregates created: %', (SELECT COUNT(*) FROM timescaledb_information.continuous_aggregates);
    RAISE NOTICE 'Database ready for adaptive learning system';
END $$;