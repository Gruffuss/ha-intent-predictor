-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Main sensor events table (based on what timeseries_db.py actually uses)
CREATE TABLE IF NOT EXISTS sensor_events (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    entity_id VARCHAR(255) NOT NULL,
    state VARCHAR(50),
    numeric_value DOUBLE PRECISION,
    attributes JSONB,
    room VARCHAR(100),
    sensor_type VARCHAR(50),
    zone_type VARCHAR(50),
    zone_info JSONB,
    person VARCHAR(50),
    enriched_data JSONB,
    processed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('sensor_events', 'timestamp', 
                       if_not_exists => TRUE,
                       chunk_time_interval => INTERVAL '1 day');

-- Predictions table (matching what the code expects)
CREATE TABLE IF NOT EXISTS predictions (
    timestamp TIMESTAMPTZ NOT NULL,
    room TEXT NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    probability FLOAT NOT NULL,
    uncertainty FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    model_name TEXT,
    features JSONB,
    metadata JSONB,
    PRIMARY KEY (timestamp, room, horizon_minutes)
);

-- Convert to hypertable
SELECT create_hypertable('predictions', 'timestamp',
                       if_not_exists => TRUE,
                       chunk_time_interval => INTERVAL '1 day');

-- Pattern discoveries table (note: code uses "pattern_discoveries" not "discovered_patterns")
CREATE TABLE IF NOT EXISTS pattern_discoveries (
    timestamp TIMESTAMPTZ NOT NULL,
    room TEXT NOT NULL,
    pattern_type TEXT NOT NULL,
    pattern_data JSONB NOT NULL,
    significance_score FLOAT NOT NULL,
    frequency INTEGER NOT NULL,
    metadata JSONB,
    PRIMARY KEY (timestamp, room, pattern_type)
);

-- Convert to hypertable
SELECT create_hypertable('pattern_discoveries', 'timestamp',
                       if_not_exists => TRUE,
                       chunk_time_interval => INTERVAL '1 day');

-- Model performance table
CREATE TABLE IF NOT EXISTS model_performance (
    timestamp TIMESTAMPTZ NOT NULL,
    model_name TEXT NOT NULL,
    room TEXT NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    accuracy FLOAT,
    auc_score FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    sample_count INTEGER,
    metadata JSONB,
    PRIMARY KEY (timestamp, model_name, room, horizon_minutes)
);

-- Convert to hypertable
SELECT create_hypertable('model_performance', 'timestamp',
                       if_not_exists => TRUE,
                       chunk_time_interval => INTERVAL '1 day');

-- Room occupancy table (used by historical_import.py)
CREATE TABLE IF NOT EXISTS room_occupancy (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    room VARCHAR(100) NOT NULL,
    occupied BOOLEAN NOT NULL,
    confidence DOUBLE PRECISION,
    inference_method VARCHAR(100),
    supporting_evidence JSONB,
    person VARCHAR(50),
    duration_minutes INTEGER,
    processed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('room_occupancy', 'timestamp',
                       if_not_exists => TRUE,
                       chunk_time_interval => INTERVAL '1 day');

-- Discovered patterns table (used by bootstrap_complete.py for initial patterns)
CREATE TABLE IF NOT EXISTS discovered_patterns (
    id BIGSERIAL PRIMARY KEY,
    room VARCHAR(100) NOT NULL,
    pattern_type VARCHAR(100) NOT NULL,
    pattern_data JSONB NOT NULL,
    significance_score DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    support_count INTEGER,
    discovered_at TIMESTAMPTZ DEFAULT NOW(),
    last_validated TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create all necessary indexes
CREATE INDEX IF NOT EXISTS idx_sensor_events_timestamp ON sensor_events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_events_entity_id ON sensor_events (entity_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_events_room ON sensor_events (room, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_events_sensor_type ON sensor_events (sensor_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_events_person ON sensor_events (person, timestamp DESC) WHERE person IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sensor_events_zone_type ON sensor_events (zone_type, timestamp DESC) WHERE zone_type IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_predictions_room_time ON predictions (room, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_horizon ON predictions (horizon_minutes, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_pattern_discoveries_room ON pattern_discoveries (room, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_pattern_discoveries_type ON pattern_discoveries (pattern_type, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_model_performance_room_model ON model_performance (room, model_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_model_performance_accuracy ON model_performance (accuracy DESC, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_room_occupancy_room ON room_occupancy (room, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_room_occupancy_timestamp ON room_occupancy (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_discovered_patterns_room ON discovered_patterns (room, discovered_at DESC);
CREATE INDEX IF NOT EXISTS idx_discovered_patterns_type ON discovered_patterns (pattern_type, discovered_at DESC);