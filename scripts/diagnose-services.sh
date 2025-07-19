#!/bin/bash

# HA Intent Predictor Service Diagnostic Script
# Run this inside the container to diagnose service issues

echo "=== HA Intent Predictor Service Diagnostic ==="
echo "Timestamp: $(date)"
echo

echo "=== Docker Containers Status ==="
docker ps -a
echo

echo "=== Docker Compose Status ==="
cd /opt/ha-intent-predictor 2>/dev/null || echo "Application directory not found"
if [ -f docker-compose.yml ]; then
    docker compose ps
    echo
    echo "=== Docker Compose Logs (last 20 lines) ==="
    docker compose logs --tail=20
else
    echo "docker-compose.yml not found"
fi
echo

echo "=== Port Usage ==="
netstat -tlnp | grep -E ":(3000|5432|6379|8000|9092|2181)" || echo "No services found on expected ports"
echo

echo "=== Service Status ==="
systemctl status ha-intent-predictor.service --no-pager --lines=5 2>/dev/null || echo "Main service not found"
echo
systemctl status nginx --no-pager --lines=5 2>/dev/null || echo "Nginx not found"
echo

echo "=== Network Connectivity ==="
echo "Testing localhost connectivity:"
curl -s -o /dev/null -w "HTTP %{http_code} - %{url_effective}\n" http://localhost:3000 || echo "Port 3000: FAILED"
curl -s -o /dev/null -w "HTTP %{http_code} - %{url_effective}\n" http://localhost:5432 || echo "Port 5432: FAILED"
curl -s -o /dev/null -w "HTTP %{http_code} - %{url_effective}\n" http://localhost:6379 || echo "Port 6379: FAILED"
curl -s -o /dev/null -w "HTTP %{http_code} - %{url_effective}\n" http://localhost:8000 || echo "Port 8000: FAILED"
echo

echo "=== Docker Network ==="
docker network ls
echo

echo "=== Application Directory ==="
ls -la /opt/ha-intent-predictor/ 2>/dev/null || echo "Application directory not found"
echo

echo "=== Recent Logs ==="
echo "--- Docker Compose Services ---"
if [ -f /opt/ha-intent-predictor/docker-compose.yml ]; then
    cd /opt/ha-intent-predictor
    docker compose logs grafana --tail=10 2>/dev/null || echo "Grafana logs not available"
    echo
    docker compose logs postgres --tail=10 2>/dev/null || echo "Postgres logs not available"
    echo
    docker compose logs redis --tail=10 2>/dev/null || echo "Redis logs not available"
fi

echo
echo "=== Disk Space ==="
df -h
echo

echo "=== Memory Usage ==="
free -h
echo

echo "=== Process List ==="
ps aux | grep -E "(docker|grafana|postgres|redis)" | grep -v grep || echo "No relevant processes found"
echo

echo "=== Environment Variables ==="
if [ -f /opt/ha-intent-predictor/.env ]; then
    echo ".env file exists"
    wc -l /opt/ha-intent-predictor/.env
else
    echo ".env file not found"
fi

echo
echo "=== Firewall Status ==="
ufw status 2>/dev/null || echo "UFW not installed or configured"

echo
echo "=== Suggested Fixes ==="
echo "1. If containers are not running:"
echo "   cd /opt/ha-intent-predictor && docker compose up -d"
echo
echo "2. If containers are failing to start:"
echo "   cd /opt/ha-intent-predictor && docker compose logs"
echo
echo "3. If ports are not accessible:"
echo "   Check if services are bound to localhost only"
echo "   Verify docker-compose.yml port mappings"
echo
echo "4. Restart all services:"
echo "   cd /opt/ha-intent-predictor && docker compose down && docker compose up -d"

echo
echo "=== Diagnostic Complete ==="