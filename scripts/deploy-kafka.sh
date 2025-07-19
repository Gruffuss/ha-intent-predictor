#!/bin/bash

# Deploy complete architecture with Kafka
# Run this inside the container

cd /opt/ha-intent-predictor

echo "=== Backing up current docker-compose.yml ==="
cp docker-compose.yml docker-compose.yml.backup.$(date +%Y%m%d_%H%M%S)

echo "=== Creating complete docker-compose.yml with Kafka ==="
cat > docker-compose.yml << 'EOF'
services:
  postgres:
    image: timescale/timescaledb:2.11.0-pg15
    container_name: ha-predictor-postgres
    environment:
      POSTGRES_DB: ha_predictor
      POSTGRES_USER: ha_predictor
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-hapredictor_db_pass}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ha_predictor"]
      interval: 10s
      timeout: 5s
      retries: 5
    
  redis:
    image: redis:7-alpine
    container_name: ha-predictor-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    container_name: ha-predictor-zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"
    volumes:
      - zookeeper_data:/var/lib/zookeeper/data
      - zookeeper_logs:/var/lib/zookeeper/log
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "nc", "-z", "localhost", "2181"]
      interval: 10s
      timeout: 5s
      retries: 5

  kafka:
    image: confluentinc/cp-kafka:7.4.0
    container_name: ha-predictor-kafka
    depends_on:
      zookeeper:
        condition: service_healthy
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092,PLAINTEXT_INTERNAL://kafka:29092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_INTERNAL:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT_INTERNAL
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: 'true'
    ports:
      - "9092:9092"
    volumes:
      - kafka_data:/var/lib/kafka/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "kafka-topics", "--bootstrap-server", "localhost:9092", "--list"]
      interval: 30s
      timeout: 10s
      retries: 5

volumes:
  postgres_data:
  redis_data:
  kafka_data:
  zookeeper_data:
  zookeeper_logs:

networks:
  default:
    name: ha-predictor-network
EOF

echo "=== Creating database init script ==="
mkdir -p scripts
cat > scripts/init.sql << 'EOF'
-- Enable TimescaleDB extension (already enabled in the image)
-- CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create tables for HA Intent Predictor
CREATE TABLE IF NOT EXISTS sensor_events (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    state VARCHAR(50),
    attributes JSONB,
    room VARCHAR(100),
    sensor_type VARCHAR(100),
    zone_info JSONB,
    source VARCHAR(50) DEFAULT 'home_assistant'
);

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    room VARCHAR(100) NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    probability FLOAT NOT NULL,
    uncertainty FLOAT NOT NULL,
    model_info JSONB,
    features JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_performance (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    room VARCHAR(100) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS room_states (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    room VARCHAR(100) NOT NULL,
    occupied BOOLEAN NOT NULL,
    confidence FLOAT,
    source VARCHAR(50) DEFAULT 'inference',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertables for time-series data
SELECT create_hypertable('sensor_events', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('predictions', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('model_performance', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('room_states', 'timestamp', if_not_exists => TRUE);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_sensor_events_entity_time ON sensor_events (entity_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_events_room_time ON sensor_events (room, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sensor_events_type_time ON sensor_events (sensor_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_room_time ON predictions (room, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_horizon ON predictions (horizon_minutes, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_model_performance_room_model ON model_performance (room, model_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_room_states_room_time ON room_states (room, timestamp DESC);

-- Create some useful views
CREATE OR REPLACE VIEW latest_predictions AS
SELECT DISTINCT ON (room, horizon_minutes) 
    room, horizon_minutes, probability, uncertainty, timestamp, model_info
FROM predictions 
ORDER BY room, horizon_minutes, timestamp DESC;

CREATE OR REPLACE VIEW room_occupancy_summary AS
SELECT 
    room,
    COUNT(*) as total_events,
    COUNT(*) FILTER (WHERE occupied = true) as occupied_events,
    ROUND(100.0 * COUNT(*) FILTER (WHERE occupied = true) / COUNT(*), 2) as occupancy_rate,
    MAX(timestamp) as last_updated
FROM room_states 
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY room;

-- Insert some test data
INSERT INTO sensor_events (timestamp, entity_id, state, room, sensor_type) VALUES
(NOW(), 'binary_sensor.test_presence', 'on', 'test_room', 'presence'),
(NOW() - INTERVAL '1 minute', 'binary_sensor.test_presence', 'off', 'test_room', 'presence');

COMMIT;
EOF

echo "=== Updating .env file ==="
if [ ! -f .env ]; then
    cat > .env << 'EOF'
POSTGRES_PASSWORD=hapredictor_db_pass
GRAFANA_PASSWORD=hapredictor_grafana_pass
EOF
else
    # Add missing variables if they don't exist
    if ! grep -q POSTGRES_PASSWORD .env; then
        echo "POSTGRES_PASSWORD=hapredictor_db_pass" >> .env
    fi
    if ! grep -q GRAFANA_PASSWORD .env; then
        echo "GRAFANA_PASSWORD=hapredictor_grafana_pass" >> .env
    fi
fi

echo "=== Deploying new architecture ==="
docker compose down
echo "Waiting for services to stop..."
sleep 5
docker compose up -d

echo "=== Waiting for services to start ==="
sleep 15

echo "=== Checking service health ==="
for i in {1..30}; do
    echo "Health check attempt $i/30..."
    
    # Check PostgreSQL
    if docker exec ha-predictor-postgres pg_isready -U ha_predictor >/dev/null 2>&1; then
        echo "✓ PostgreSQL is ready"
        postgres_ready=true
    else
        echo "⏳ PostgreSQL not ready yet..."
        postgres_ready=false
    fi
    
    # Check Redis
    if docker exec ha-predictor-redis redis-cli ping >/dev/null 2>&1; then
        echo "✓ Redis is ready"
        redis_ready=true
    else
        echo "⏳ Redis not ready yet..."
        redis_ready=false
    fi
    
    # Check Zookeeper
    if docker exec ha-predictor-zookeeper nc -z localhost 2181 >/dev/null 2>&1; then
        echo "✓ Zookeeper is ready"
        zk_ready=true
    else
        echo "⏳ Zookeeper not ready yet..."
        zk_ready=false
    fi
    
    # Check Kafka
    if docker exec ha-predictor-kafka kafka-topics --bootstrap-server localhost:9092 --list >/dev/null 2>&1; then
        echo "✓ Kafka is ready"
        kafka_ready=true
    else
        echo "⏳ Kafka not ready yet..."
        kafka_ready=false
    fi
    
    if [ "$postgres_ready" = true ] && [ "$redis_ready" = true ] && [ "$zk_ready" = true ] && [ "$kafka_ready" = true ]; then
        echo ""
        echo "🎉 All services are ready!"
        break
    fi
    
    echo "Waiting 10 seconds before next check..."
    sleep 10
done

echo ""
echo "=== Final Status ==="
docker compose ps

echo ""
echo "=== Testing Database Schema ==="
docker exec ha-predictor-postgres psql -U ha_predictor -d ha_predictor -c "\dt"

echo ""
echo "=== Testing Kafka Topics ==="
docker exec ha-predictor-kafka kafka-topics --bootstrap-server localhost:9092 --list

echo ""
echo "=== Deployment Complete ==="
echo "PostgreSQL: localhost:5432 (user: ha_predictor, db: ha_predictor)"
echo "Redis: localhost:6379"
echo "Kafka: localhost:9092"
echo "Zookeeper: localhost:2181"
echo ""
echo "Monitor with: /opt/ha-intent-predictor/scripts/remote-monitor.sh"