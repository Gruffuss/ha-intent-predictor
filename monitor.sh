#!/bin/bash

# Simple monitoring script
echo "=== Intent Predictor Status ==="
echo "Service Status:"
systemctl is-active intent-predictor

echo -e "\nHealth Check:"
curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || echo "Health check failed"

echo -e "\nResource Usage:"
ps aux | grep "python src/main.py" | grep -v grep

echo -e "\nLast 10 log entries:"
tail -n 10 /home/intent-predictor/ha-intent-predictor/logs/intent_predictor.log 2>/dev/null || echo "No logs yet"

echo -e "\nDisk Usage:"
df -h /home/intent-predictor/ha-intent-predictor/
