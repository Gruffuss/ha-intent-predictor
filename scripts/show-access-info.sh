#!/bin/bash

# Show access information for HA Intent Predictor container
# Usage: ./show-access-info.sh [container_id]

# Default password from installation
DEFAULT_PASSWORD="hapredictor123"

# Find container if not specified
if [ -n "$1" ]; then
    CTID="$1"
else
    CTID=$(pct list | grep ha-intent-predictor | awk '{print $1}' | head -1)
    if [ -z "$CTID" ]; then
        echo "No ha-intent-predictor container found. Please specify container ID:"
        echo "Usage: $0 <container_id>"
        exit 1
    fi
fi

# Check if container exists and is running
if ! pct status "$CTID" &>/dev/null; then
    echo "Container $CTID not found"
    exit 1
fi

if ! pct status "$CTID" | grep -q "running"; then
    echo "Container $CTID is not running. Starting it..."
    pct start "$CTID"
    sleep 5
fi

# Get container info
HOSTNAME=$(pct exec "$CTID" -- hostname 2>/dev/null || echo "ha-intent-predictor")
CONTAINER_IP=$(pct exec "$CTID" -- hostname -I 2>/dev/null | awk '{print $1}' || echo "checking...")

echo "🔐 HA Intent Predictor Access Information"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "📋 Container Details:"
echo "   • Container ID: $CTID"
echo "   • Hostname: $HOSTNAME"
echo "   • IP Address: $CONTAINER_IP"
echo "   • Status: $(pct status "$CTID")"
echo
echo "🔐 Access Information:"
echo "   • SSH: ssh root@$CONTAINER_IP"
echo "   • Root Password: $DEFAULT_PASSWORD"
echo "   • Console: pct enter $CTID (from Proxmox host)"
echo
echo "🔗 Service URLs:"
echo "   • Health Check: http://$CONTAINER_IP/health"
echo "   • API: http://$CONTAINER_IP:8000"
echo "   • Grafana: http://$CONTAINER_IP:3000"
echo
echo "🔧 Quick Commands:"
echo "   • Enter container: pct enter $CTID"
echo "   • Check services: pct exec $CTID -- docker ps"
echo "   • View logs: pct exec $CTID -- docker logs ha-predictor-postgres"
echo "   • Restart services: pct exec $CTID -- 'cd /opt/ha-intent-predictor && docker compose restart'"
echo
echo "💡 Troubleshooting:"
echo "   • Test SSH: ssh root@$CONTAINER_IP (password: $DEFAULT_PASSWORD)"
echo "   • Fix services: pct exec $CTID -- /opt/ha-intent-predictor/scripts/fix-services.sh"
echo "   • Diagnose issues: pct exec $CTID -- /opt/ha-intent-predictor/scripts/diagnose-services.sh"
echo

# Test if SSH is working
echo "🧪 Testing SSH connectivity..."
if timeout 5 ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no root@$CONTAINER_IP "echo 'SSH test successful'" 2>/dev/null; then
    echo "✅ SSH is working!"
else
    echo "❌ SSH not accessible. Try:"
    echo "   1. Check if SSH service is running: pct exec $CTID -- systemctl status ssh"
    echo "   2. Restart SSH: pct exec $CTID -- systemctl restart ssh"
    echo "   3. Use console access: pct enter $CTID"
fi

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"