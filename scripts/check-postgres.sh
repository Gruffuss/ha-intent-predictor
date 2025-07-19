#!/bin/bash

# Quick PostgreSQL diagnostic script
# Run this to check what's happening with PostgreSQL during installation

echo "=== PostgreSQL Container Diagnostic ==="
echo "Timestamp: $(date)"
echo

echo "=== Container Status ==="
if which pct >/dev/null 2>&1; then
    # We're on Proxmox host
    echo "Running on Proxmox host"
    CTID=$(pct list | grep ha-intent-predictor | awk '{print $1}' | head -1)
    if [ -n "$CTID" ]; then
        echo "Found container: $CTID"
        echo "Container status: $(pct status $CTID)"
        
        echo
        echo "=== Docker Containers ==="
        pct exec "$CTID" -- docker ps -a | grep ha-predictor || echo "No ha-predictor containers found"
        
        echo
        echo "=== PostgreSQL Container Details ==="
        if pct exec "$CTID" -- docker ps -a | grep -q ha-predictor-postgres; then
            echo "PostgreSQL container exists"
            echo "Status:"
            pct exec "$CTID" -- docker ps -a | grep ha-predictor-postgres
            
            echo
            echo "=== PostgreSQL Logs (last 20 lines) ==="
            pct exec "$CTID" -- docker logs ha-predictor-postgres --tail 20 2>/dev/null || echo "Could not get logs"
            
            echo
            echo "=== Testing Connection ==="
            if pct exec "$CTID" -- docker exec ha-predictor-postgres pg_isready -U ha_predictor 2>/dev/null; then
                echo "✅ PostgreSQL is ready"
            else
                echo "❌ PostgreSQL is not ready"
                echo "Trying to connect..."
                pct exec "$CTID" -- docker exec ha-predictor-postgres pg_isready -U ha_predictor
            fi
        else
            echo "❌ PostgreSQL container not found"
        fi
        
        echo
        echo "=== Docker Compose Status ==="
        pct exec "$CTID" -- bash -c "cd /opt/ha-intent-predictor && docker compose ps" 2>/dev/null || echo "Could not get compose status"
        
        echo
        echo "=== Environment File ==="
        if pct exec "$CTID" -- test -f /opt/ha-intent-predictor/.env; then
            echo "✅ .env file exists"
            pct exec "$CTID" -- wc -l /opt/ha-intent-predictor/.env
        else
            echo "❌ .env file missing"
        fi
        
    else
        echo "❌ No ha-intent-predictor container found"
        echo "Available containers:"
        pct list
    fi
else
    # We're inside the container
    echo "Running inside container"
    
    echo "=== Docker Containers ==="
    docker ps -a | grep ha-predictor || echo "No ha-predictor containers found"
    
    echo
    echo "=== PostgreSQL Container Details ==="
    if docker ps -a | grep -q ha-predictor-postgres; then
        echo "PostgreSQL container exists"
        echo "Status:"
        docker ps -a | grep ha-predictor-postgres
        
        echo
        echo "=== PostgreSQL Logs (last 20 lines) ==="
        docker logs ha-predictor-postgres --tail 20 2>/dev/null || echo "Could not get logs"
        
        echo
        echo "=== Testing Connection ==="
        if docker exec ha-predictor-postgres pg_isready -U ha_predictor 2>/dev/null; then
            echo "✅ PostgreSQL is ready"
        else
            echo "❌ PostgreSQL is not ready"
            echo "Trying to connect..."
            docker exec ha-predictor-postgres pg_isready -U ha_predictor
        fi
    else
        echo "❌ PostgreSQL container not found"
    fi
    
    echo
    echo "=== Docker Compose Status ==="
    if [ -d /opt/ha-intent-predictor ]; then
        cd /opt/ha-intent-predictor
        docker compose ps 2>/dev/null || echo "Could not get compose status"
    else
        echo "Application directory not found"
    fi
fi

echo
echo "=== Suggested Actions ==="
echo "If PostgreSQL is not starting:"
echo "1. Check if there's enough disk space: df -h"
echo "2. Check PostgreSQL logs: docker logs ha-predictor-postgres"
echo "3. Restart the container: docker restart ha-predictor-postgres"
echo "4. Recreate services: cd /opt/ha-intent-predictor && docker compose down && docker compose up -d"

echo
echo "=== Diagnostic Complete ==="