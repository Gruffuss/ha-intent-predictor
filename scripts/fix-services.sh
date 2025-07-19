#!/bin/bash

# HA Intent Predictor Service Fix Script
# Run this inside the container to fix common service issues

echo "=== HA Intent Predictor Service Fix Script ==="
echo "Timestamp: $(date)"
echo

# Change to application directory
cd /opt/ha-intent-predictor 2>/dev/null || {
    echo "ERROR: Application directory /opt/ha-intent-predictor not found"
    exit 1
}

echo "=== Current Status ==="
docker compose ps 2>/dev/null || echo "Docker compose not working"
echo

echo "=== Stopping All Services ==="
docker compose down 2>/dev/null || echo "No services to stop"

echo "=== Checking Docker ==="
systemctl status docker --no-pager --lines=3
if ! systemctl is-active docker >/dev/null; then
    echo "Starting Docker service..."
    systemctl start docker
    sleep 5
fi

echo "=== Checking for docker-compose.yml ==="
if [ ! -f docker-compose.yml ]; then
    echo "ERROR: docker-compose.yml not found"
    ls -la
    exit 1
fi

echo "=== Checking .env file ==="
if [ ! -f .env ]; then
    echo "Creating missing .env file..."
    cat > .env << EOF
POSTGRES_PASSWORD=$(openssl rand -base64 32)
GRAFANA_PASSWORD=$(openssl rand -base64 16)
EOF
    echo ".env file created"
fi

echo "=== Starting Services ==="
docker compose up -d

echo "=== Waiting for Services to Start ==="
sleep 30

echo "=== Checking Service Status ==="
docker compose ps

echo "=== Testing Connectivity ==="
echo "Testing Grafana (port 3000):"
for i in {1..10}; do
    if curl -s -o /dev/null http://localhost:3000; then
        echo "✅ Grafana is responding on port 3000"
        break
    else
        echo "⏳ Attempt $i/10: Grafana not yet ready..."
        sleep 5
    fi
done

echo "Testing PostgreSQL (port 5432):"
if docker compose exec postgres pg_isready -U ha_predictor >/dev/null 2>&1; then
    echo "✅ PostgreSQL is ready"
else
    echo "❌ PostgreSQL is not ready"
fi

echo "Testing Redis (port 6379):"
if docker compose exec redis redis-cli ping >/dev/null 2>&1; then
    echo "✅ Redis is ready"
else
    echo "❌ Redis is not ready"
fi

echo
echo "=== Service URLs ==="
container_ip=$(hostname -I | awk '{print $1}')
echo "Grafana: http://$container_ip:3000"
echo "API: http://$container_ip:8000"
echo "Health: http://$container_ip/health"

echo
echo "=== Grafana Credentials ==="
if [ -f .env ]; then
    echo "Username: admin"
    echo "Password: $(grep GRAFANA_PASSWORD .env | cut -d= -f2)"
else
    echo "Check .env file for Grafana password"
fi

echo
echo "=== Next Steps ==="
echo "1. Test Grafana: curl http://localhost:3000"
echo "2. Check logs if issues persist: docker compose logs grafana"
echo "3. Access Grafana web UI: http://$container_ip:3000"

echo
echo "=== Fix Complete ==="