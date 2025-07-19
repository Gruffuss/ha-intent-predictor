#!/bin/bash

# Remote Monitoring Script for HA Intent Predictor
# This script outputs status to a file that can be read remotely

MONITOR_FILE="/tmp/ha-predictor-status.json"
MONITOR_LOG="/tmp/ha-predictor-monitor.log"

# Function to get container status
get_container_status() {
    local container_name="$1"
    if docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
        echo "running"
    elif docker ps -a --format "{{.Names}}" | grep -q "^${container_name}$"; then
        echo "stopped"
    else
        echo "missing"
    fi
}

# Function to test service health
test_service_health() {
    local service="$1"
    case $service in
        postgres)
            if docker exec ha-predictor-postgres pg_isready -U ha_predictor >/dev/null 2>&1; then
                echo "healthy"
            else
                echo "unhealthy"
            fi
            ;;
        redis)
            if docker exec ha-predictor-redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
                echo "healthy"
            else
                echo "unhealthy"
            fi
            ;;
        kafka)
            if docker exec ha-predictor-kafka kafka-topics --bootstrap-server localhost:9092 --list >/dev/null 2>&1; then
                echo "healthy"
            else
                echo "unhealthy"
            fi
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Function to get resource usage
get_resource_usage() {
    # Get container resource usage
    docker stats --no-stream --format "{{.Container}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}" 2>/dev/null || echo "no_data"
}

# Function to get system info
get_system_info() {
    echo "{"
    echo "  \"timestamp\": \"$(date -Iseconds)\","
    echo "  \"uptime\": \"$(uptime -p)\","
    echo "  \"load_avg\": \"$(uptime | awk -F'load average:' '{print $2}' | xargs)\","
    echo "  \"disk_usage\": \"$(df -h / | awk 'NR==2 {print $5}' | tr -d '%')\","
    echo "  \"memory_usage\": \"$(free | awk 'NR==2{printf \"%.1f\", $3*100/$2}')\","
    echo "  \"containers\": {"
    echo "    \"postgres\": {"
    echo "      \"status\": \"$(get_container_status ha-predictor-postgres)\","
    echo "      \"health\": \"$(test_service_health postgres)\""
    echo "    },"
    echo "    \"redis\": {"
    echo "      \"status\": \"$(get_container_status ha-predictor-redis)\","
    echo "      \"health\": \"$(test_service_health redis)\""
    echo "    },"
    echo "    \"kafka\": {"
    echo "      \"status\": \"$(get_container_status ha-predictor-kafka)\","
    echo "      \"health\": \"$(test_service_health kafka)\""
    echo "    },"
    echo "    \"zookeeper\": {"
    echo "      \"status\": \"$(get_container_status ha-predictor-zookeeper)\","
    echo "      \"health\": \"unknown\""
    echo "    }"
    echo "  },"
    echo "  \"resource_usage\": ["
    get_resource_usage | while IFS=',' read -r container cpu mem_usage mem_perc; do
        if [ "$container" != "no_data" ]; then
            echo "    {"
            echo "      \"container\": \"$container\","
            echo "      \"cpu_percent\": \"$cpu\","
            echo "      \"memory_usage\": \"$mem_usage\","
            echo "      \"memory_percent\": \"$mem_perc\""
            echo "    },"
        fi
    done | sed '$ s/,$//'
    echo "  ],"
    echo "  \"services\": {"
    echo "    \"nginx\": \"$(systemctl is-active nginx 2>/dev/null || echo inactive)\","
    echo "    \"ssh\": \"$(systemctl is-active ssh 2>/dev/null || echo inactive)\","
    echo "    \"docker\": \"$(systemctl is-active docker 2>/dev/null || echo inactive)\""
    echo "  },"
    echo "  \"application\": {"
    echo "    \"directory_exists\": $([ -d /opt/ha-intent-predictor ] && echo true || echo false),"
    echo "    \"venv_exists\": $([ -d /opt/ha-intent-predictor/venv ] && echo true || echo false),"
    echo "    \"compose_file_exists\": $([ -f /opt/ha-intent-predictor/docker-compose.yml ] && echo true || echo false)"
    echo "  }"
    echo "}"
}

# Function to get recent logs
get_recent_logs() {
    echo "=== RECENT DOCKER LOGS ===" >> "$MONITOR_LOG"
    echo "Timestamp: $(date)" >> "$MONITOR_LOG"
    echo "" >> "$MONITOR_LOG"
    
    for container in ha-predictor-postgres ha-predictor-redis ha-predictor-kafka ha-predictor-zookeeper; do
        if docker ps -a --format "{{.Names}}" | grep -q "^${container}$"; then
            echo "--- $container ---" >> "$MONITOR_LOG"
            docker logs "$container" --tail 10 --since 5m 2>&1 >> "$MONITOR_LOG"
            echo "" >> "$MONITOR_LOG"
        fi
    done
    
    # Keep only last 1000 lines
    tail -1000 "$MONITOR_LOG" > "${MONITOR_LOG}.tmp" && mv "${MONITOR_LOG}.tmp" "$MONITOR_LOG"
}

# Main monitoring function
monitor_once() {
    get_system_info > "$MONITOR_FILE"
    get_recent_logs
    
    # Also create a human-readable summary
    cat > "/tmp/ha-predictor-summary.txt" << EOF
HA Intent Predictor Status Summary
Generated: $(date)

=== CONTAINER STATUS ===
PostgreSQL: $(get_container_status ha-predictor-postgres) - $(test_service_health postgres)
Redis: $(get_container_status ha-predictor-redis) - $(test_service_health redis)  
Kafka: $(get_container_status ha-predictor-kafka) - $(test_service_health kafka)
Zookeeper: $(get_container_status ha-predictor-zookeeper)

=== SYSTEM HEALTH ===
Disk Usage: $(df -h / | awk 'NR==2 {print $5}')
Memory Usage: $(free | awk 'NR==2{printf "%.1f%%", $3*100/$2}')
Load Average: $(uptime | awk -F'load average:' '{print $2}' | xargs)

=== SERVICES ===
Docker: $(systemctl is-active docker 2>/dev/null || echo inactive)
Nginx: $(systemctl is-active nginx 2>/dev/null || echo inactive)
SSH: $(systemctl is-active ssh 2>/dev/null || echo inactive)

=== QUICK TESTS ===
PostgreSQL Connection: $(docker exec ha-predictor-postgres pg_isready -U ha_predictor 2>/dev/null && echo "OK" || echo "FAIL")
Redis Connection: $(docker exec ha-predictor-redis redis-cli ping 2>/dev/null || echo "FAIL")
Web Server: $(curl -s http://localhost/health >/dev/null && echo "OK" || echo "FAIL")

=== DOCKER COMPOSE STATUS ===
$(cd /opt/ha-intent-predictor && docker compose ps 2>/dev/null || echo "Docker compose not available")
EOF
}

# Continuous monitoring mode
monitor_continuous() {
    echo "Starting continuous monitoring... (Ctrl+C to stop)"
    echo "Monitor files: $MONITOR_FILE and /tmp/ha-predictor-summary.txt"
    
    while true; do
        monitor_once
        sleep 30
    done
}

# Handle command line arguments
case "${1:-once}" in
    "continuous"|"loop"|"watch")
        monitor_continuous
        ;;
    "once"|"")
        monitor_once
        echo "Status written to: $MONITOR_FILE"
        echo "Summary written to: /tmp/ha-predictor-summary.txt"
        echo "Logs written to: $MONITOR_LOG"
        ;;
    "show"|"cat"|"view")
        if [ -f "/tmp/ha-predictor-summary.txt" ]; then
            cat /tmp/ha-predictor-summary.txt
        else
            echo "No status file found. Run monitor first."
        fi
        ;;
    "json")
        if [ -f "$MONITOR_FILE" ]; then
            cat "$MONITOR_FILE"
        else
            echo "No JSON status file found. Run monitor first."
        fi
        ;;
    *)
        echo "Usage: $0 [once|continuous|show|json]"
        echo "  once       - Generate status once (default)"
        echo "  continuous - Monitor continuously every 30s"
        echo "  show       - Show latest human-readable status"
        echo "  json       - Show latest JSON status"
        ;;
esac