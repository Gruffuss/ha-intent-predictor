# Home Assistant Intent Prediction System - Setup Guide

## Overview

This system automatically learns your behavioral patterns from Home Assistant sensor data and predicts:
- **Current Activity**: WORKING, COOKING, RELAXING, etc.
- **Room Occupancy Intent**: Likelihood you'll be in each room over next 2 hours
- **Activity Duration**: How long current activity will continue
- **Next Room**: Most likely next room you'll visit

Perfect for **smart precooling** and **proactive automation**!

## Hardware Requirements

✅ **Your Setup is Perfect!**
- Intel N200 4C/4T @ 3.7GHz ✅
- 16GB RAM ✅  
- Docker support ✅

Expected resource usage:
- **Training**: ~50-80% CPU for 1-2 hours initially, then 5-10 min daily
- **Running**: ~5-10% CPU, ~500MB RAM
- **Storage**: ~500MB for models and cache

## Quick Start

### 1. Get Your Home Assistant Token

1. Go to your Home Assistant: `Settings → People → Long-lived access tokens`
2. Click `CREATE TOKEN`
3. Name it: `Intent Prediction System`
4. Copy the token (you'll need this in step 3)

### 2. Download the System

```bash
# Create project directory
mkdir ha-intent-predictor
cd ha-intent-predictor

# Download all files (you'll need to save each artifact manually)
# Or clone from your repo if you put this there
```

### 3. Configure the System

Edit `config/config.yaml`:

```yaml
# REQUIRED: Update these with your details
home_assistant:
  url: "http://YOUR-HA-IP:8123"          # Your HA URL
  token: "YOUR_LONG_LIVED_ACCESS_TOKEN"  # Token from step 1

# OPTIONAL: Update entity names if different
data:
  presence_entities:
    - binary_sensor.bedroom_floor
    - binary_sensor.bedroom_vladimir_bed_side
    - binary_sensor.bedroom_presence_sensor_anca_bed_side
    # ... update with your exact entity names
```

**Important**: Make sure the entity names in `presence_entities` match your actual Home Assistant entities exactly!

### 4. Create Required Directories

```bash
mkdir -p data/models data/cache logs
chmod 755 data logs
```

### 5. Deploy with Docker

```bash
# Build and start the system
docker-compose up -d

# Check if it's working
docker-compose logs -f intent-predictor
```

### 6. Verify It's Working

1. **Check the logs**: `docker-compose logs intent-predictor`
   - Should show "Application initialized successfully"
   - Initial training takes 1-2 hours with 180 days of data

2. **Check Home Assistant**: New sensors should appear:
   - `sensor.current_activity`
   - `sensor.office_occupancy_probability_2h`
   - `sensor.bedroom_occupancy_probability_2h`
   - `sensor.kitchen_livingroom_occupancy_probability_2h`
   - And more...

3. **Health check**: Visit `http://YOUR-SERVER-IP:8000/health`

## What the System Does

### Phase 1: Initial Training (1-2 hours)
- Downloads 180 days of your sensor history
- Automatically discovers your activity patterns
- Trains ML models on your specific behavior
- No manual configuration needed!

### Phase 2: Real-time Prediction (ongoing)
- Updates predictions every 5 minutes
- Learns continuously from new data
- Retrains models daily at 3 AM

### Phase 3: Use in Automations

Example automation using the predictions:

```yaml
# Precool office when high probability of use
automation:
  - alias: "Smart Office Precooling"
    trigger:
      - platform: numeric_state
        entity_id: sensor.office_occupancy_probability_2h
        above: 70
    condition:
      - condition: numeric_state
        entity_id: sensor.office_temperature
        above: 24
    action:
      - service: climate.turn_on
        target:
          entity_id: climate.office_ac
        data:
          temperature: 22
```

## Troubleshooting

### System Won't Start
```bash
# Check logs for errors
docker-compose logs intent-predictor

# Common issues:
# 1. Wrong HA URL/token → Update config/config.yaml
# 2. Wrong entity names → Check HA Developer Tools
# 3. Insufficient data → Needs at least 7 days of history
```

### No Predictions Generated
```bash
# Check if sensors exist in HA
# Check if entities in config match HA exactly
# Verify presence sensors are actually triggering
```

### Poor Prediction Accuracy
- **Wait 7+ days** for the system to learn your patterns
- Check if presence sensors are properly configured
- Verify cats aren't triggering false positives excessively
- Look at `sensor.ml_model_accuracy` for performance metrics

### High Resource Usage
```bash
# Reduce training frequency
# Edit config.yaml:
ml:
  retrain_schedule: "03:00"  # Only once per day

# Reduce historical data
data:
  historical_days: 90  # Instead of 180
```

## Monitoring

### Health Check
- URL: `http://your-server:8000/health`
- Shows: training status, HA connection, last prediction time

### Key Sensors to Watch
- `sensor.current_activity` - What you're doing now
- `sensor.ml_model_accuracy` - How well predictions are working
- `sensor.ml_last_retrain` - When models were last updated

### Log Files
- Location: `./logs/intent_predictor.log`
- Rotates automatically, keeps 5 backups

## Advanced Configuration

### Adjust Activity Types
The system automatically discovers activities, but you can monitor what it learns by watching `sensor.current_activity`.

### Custom Room Groupings
Edit `config/config.yaml` to group sensors differently:

```yaml
rooms:
  office:
    - binary_sensor.office_presence_vladimir_desk
    - binary_sensor.office_presence_anca_desk
  
  bedroom:
    - binary_sensor.bedroom_vladimir_bed_side
    - binary_sensor.bedroom_presence_sensor_anca_bed_side
    
  # Add custom rooms here
  gameroom:
    - binary_sensor.gameroom_sensor_1
    - binary_sensor.gameroom_sensor_2
```

### Performance Tuning
```yaml
# For more frequent updates (uses more CPU)
data:
  update_interval: 180  # 3 minutes instead of 5

# For less frequent updates (uses less CPU)  
data:
  update_interval: 600  # 10 minutes
```

## Integration Examples

### Smart Climate Control
```yaml
# Precool rooms before occupancy
- alias: "Predictive Climate Control"
  trigger:
    - platform: numeric_state
      entity_id: sensor.bedroom_occupancy_probability_2h
      above: 60
  action:
    - service: climate.set_temperature
      target:
        entity_id: climate.bedroom_ac
      data:
        temperature: 21
```

### Energy Optimization
```yaml
# Turn off devices when room won't be used
- alias: "Smart Power Down"
  trigger:
    - platform: numeric_state
      entity_id: sensor.office_occupancy_probability_2h
      below: 20
  condition:
    - condition: state
      entity_id: sensor.current_activity
      state: "WORKING"
      for:
        minutes: 30
  action:
    - service: switch.turn_off
      target:
        entity_id: switch.office_monitors
```

### Adaptive Lighting
```yaml
# Prepare lighting based on predicted activity
- alias: "Predictive Lighting"
  trigger:
    - platform: state
      entity_id: sensor.current_activity
      to: "COOKING"
  action:
    - service: light.turn_on
      target:
        entity_id: light.kitchen_lights
      data:
        brightness: 255
        color_temp: 4000
```

## Support

### Common Questions

**Q: How long until it starts working well?**
A: Initial predictions start immediately, but accuracy improves over 7-14 days as it learns your patterns.

**Q: Does it work with irregular schedules?**  
A: Yes! It learns multiple patterns and adapts to changes. Works great for shift work, freelancers, etc.

**Q: What about guests and unusual days?**
A: The system detects when patterns don't match learned behavior and adjusts confidence accordingly.

**Q: Can I see what it learned about my patterns?**
A: Check the logs during training to see discovered activity types and patterns.

### Getting Help

1. **Check logs first**: `docker-compose logs intent-predictor`
2. **Verify configuration**: Ensure entity names match exactly
3. **Monitor sensors**: Watch predictions in HA for a few days
4. **Health check**: Visit the health endpoint

The system is designed to work "out of the box" with minimal configuration - it should start learning your patterns automatically!