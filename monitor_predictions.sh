#!/bin/bash

# Monitor the prediction system
echo "=== Monitoring HA Intent Prediction System ==="
echo "Current time: $(date)"

# Check if process is running
PID=$(pgrep -f "python src/main.py")
if [ -n "$PID" ]; then
    echo "✅ System is running (PID: $PID)"
else
    echo "❌ System is not running"
    exit 1
fi

# Check training status from logs
echo ""
echo "=== Training Status ==="
COMPLETED=$(tail -50 /home/intent-predictor/ha-intent-predictor/logs/intent_predictor.log | grep -c "Initial training completed successfully")
if [ "$COMPLETED" -gt 0 ]; then
    echo "✅ Training completed successfully"
    
    # Show validation scores if available
    echo "Validation scores:"
    tail -50 /home/intent-predictor/ha-intent-predictor/logs/intent_predictor.log | grep "val R²" | tail -6
    
    AVG_SCORE=$(tail -50 /home/intent-predictor/ha-intent-predictor/logs/intent_predictor.log | grep "Average validation R²" | tail -1 | grep -o "R².*" | cut -d' ' -f2)
    if [ -n "$AVG_SCORE" ]; then
        echo "Average validation R² score: $AVG_SCORE"
    fi
else
    echo "🔄 Training in progress..."
    
    # Check feature extraction progress
    FEATURE_PROGRESS=$(tail -10 /home/intent-predictor/ha-intent-predictor/logs/intent_predictor.log | grep "Feature extraction progress" | tail -1)
    if [ -n "$FEATURE_PROGRESS" ]; then
        echo "Feature extraction: $FEATURE_PROGRESS"
    fi
    
    # Check target generation progress
    TARGET_PROGRESS=$(tail -10 /home/intent-predictor/ha-intent-predictor/logs/intent_predictor.log | grep "Processing target" | tail -1)
    if [ -n "$TARGET_PROGRESS" ]; then
        echo "Target generation: $TARGET_PROGRESS"
    fi
    
    # Check model fitting progress
    MODEL_PROGRESS=$(tail -10 /home/intent-predictor/ha-intent-predictor/logs/intent_predictor.log | grep "Fitted.*model" | tail -1)
    if [ -n "$MODEL_PROGRESS" ]; then
        echo "Model fitting: $MODEL_PROGRESS"
    fi
fi

# Check predictions
echo ""
echo "=== Current Predictions ==="
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4ZGIzNTgzNWVlZDc0MDNlODRiZmY1OWU5MDIxNDYyNSIsImlhdCI6MTc1MjMzNjQ2NCwiZXhwIjoyMDY3Njk2NDY0fQ.-Z8FIr1SE8MjHECFUG9hcTW2ha1zMDdELbanvBHAkyU"
HA_URL="http://192.168.51.247:8123"

# Check key sensors
echo "Office 2h occupancy: $(curl -s -H "Authorization: Bearer $TOKEN" "$HA_URL/api/states/sensor.office_occupancy_probability_2h" | grep -o '"state":"[^"]*"' | cut -d'"' -f4)%"
echo "Bedroom 2h occupancy: $(curl -s -H "Authorization: Bearer $TOKEN" "$HA_URL/api/states/sensor.bedroom_occupancy_probability_2h" | grep -o '"state":"[^"]*"' | cut -d'"' -f4)%"
echo "Kitchen 2h occupancy: $(curl -s -H "Authorization: Bearer $TOKEN" "$HA_URL/api/states/sensor.kitchen_livingroom_occupancy_probability_2h" | grep -o '"state":"[^"]*"' | cut -d'"' -f4)%"
echo "Current activity: $(curl -s -H "Authorization: Bearer $TOKEN" "$HA_URL/api/states/sensor.current_activity" | grep -o '"state":"[^"]*"' | cut -d'"' -f4)"

# Check last update times
echo ""
echo "=== Last Updates ==="
LAST_PREDICTION=$(tail -20 /home/intent-predictor/ha-intent-predictor/logs/intent_predictor.log | grep "Predictions updated successfully" | tail -1 | cut -d' ' -f1-2)
if [ -n "$LAST_PREDICTION" ]; then
    echo "Last prediction: $LAST_PREDICTION"
else
    echo "No predictions made yet"
fi

echo ""
echo "=== Recent Errors ==="
ERRORS=$(tail -50 /home/intent-predictor/ha-intent-predictor/logs/intent_predictor.log | grep -c "ERROR\|WARNING")
if [ "$ERRORS" -gt 0 ]; then
    echo "⚠️  Found $ERRORS recent errors/warnings:"
    tail -50 /home/intent-predictor/ha-intent-predictor/logs/intent_predictor.log | grep "ERROR\|WARNING" | tail -3
else
    echo "✅ No recent errors"
fi