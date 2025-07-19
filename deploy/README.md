# HA Intent Predictor - Deployment Guide

This deployment system provides a complete, automated setup for the HA Intent Predictor ML occupancy system on Proxmox.

## 🚀 Quick Start

### Option 1: One-Command Deployment (Recommended)

Run this command on your Proxmox host:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/yourusername/ha-intent-predictor/main/deploy/proxmox-install.sh)
```

### Option 2: Manual Download and Run

```bash
# Download the installer
wget https://raw.githubusercontent.com/yourusername/ha-intent-predictor/main/deploy/proxmox-install.sh

# Make it executable
chmod +x proxmox-install.sh

# Run the installer
./proxmox-install.sh
```

## 📋 Prerequisites

- **Proxmox VE** 7.0 or later
- **Root access** on Proxmox host
- **Available storage**: 50GB+ recommended
- **Memory**: 8GB+ available for the container
- **CPU**: 4+ cores recommended
- **Home Assistant** instance with:
  - Long-lived access token
  - Network access to the Proxmox container

## 🏗️ What Gets Installed

### Infrastructure Services
- **PostgreSQL 15 + TimescaleDB**: Time-series optimized database
- **Redis 7**: Real-time feature caching
- **Apache Kafka**: Event streaming (optional, can be simplified)
- **Grafana**: Analytics dashboard
- **Prometheus**: Metrics collection
- **Nginx**: Reverse proxy

### Application Components
- **Python ML Application**: Adaptive learning system
- **Data Ingestion Service**: Continuous HA data streaming
- **Model Training Service**: Online learning and pattern discovery
- **API Service**: REST API for predictions and management
- **HA Integration**: Automatic entity creation

### System Services
- **Systemd services**: Automatic startup and monitoring
- **Log rotation**: Automatic log management
- **Health monitoring**: System health checks
- **Backup system**: Automated model and data backups

## 🔧 Configuration

### During Installation

The installer will prompt you for:

1. **Container Configuration**:
   - Container ID (auto-suggested)
   - Hostname
   - CPU cores (default: 4)
   - RAM (default: 8192MB)
   - Storage pool
   - Disk size (default: 32GB)

2. **Home Assistant Integration**:
   - HA URL (e.g., `http://YOUR_HA_IP:8123`)
   - Long-lived access token

### Post-Installation Configuration

After installation, customize these files:

```bash
# Enter the container
pct enter <container_id>

# Edit main configuration
nano /opt/ha-intent-predictor/config/app.yaml

# Update sensor mappings (customize for your sensors)
nano /opt/ha-intent-predictor/config/sensors.yaml

# Review room definitions
nano /opt/ha-intent-predictor/config/rooms.yaml
```

## 📊 Accessing the System

After installation, you can access:

- **API Documentation**: `http://<container-ip>/api/docs`
- **Grafana Dashboard**: `http://<container-ip>/grafana/` (admin/auto-generated-password)
- **Health Check**: `http://<container-ip>/health`
- **Prometheus Metrics**: `http://<container-ip>:9090`

## 🔍 System Management

### Service Management

```bash
# Check service status
systemctl status ha-intent-predictor.service
systemctl status ha-predictor-ingestion.service
systemctl status ha-predictor-training.service
systemctl status ha-predictor-api.service

# View logs
journalctl -u ha-intent-predictor.service -f
journalctl -u ha-predictor-ingestion.service -f

# Restart services
systemctl restart ha-intent-predictor.service
```

### Docker Services

```bash
# Check Docker services
cd /opt/ha-intent-predictor
docker compose ps

# View Docker logs
docker compose logs -f postgres
docker compose logs -f redis
docker compose logs -f kafka

# Restart Docker services
docker compose restart
```

### Application Logs

```bash
# Main application logs
tail -f /var/log/ha-intent-predictor/app.log

# API access logs
tail -f /var/log/ha-intent-predictor/api-access.log

# Error logs
tail -f /var/log/ha-intent-predictor/api-error.log
```

## 🏠 Home Assistant Integration

### Automatic Entity Creation

The system automatically creates these entities in Home Assistant:

#### Prediction Entities
- `sensor.occupancy_prediction_{room}_{horizon}min`
  - Example: `sensor.occupancy_prediction_bedroom_15min`
  - Attributes: probability, uncertainty, confidence, model info

#### Helper Entities
- `sensor.occupancy_trend_{room}` - Occupancy trend analysis
- `binary_sensor.occupancy_anomaly_{room}` - Anomaly detection
- `sensor.model_performance_{room}` - Model accuracy metrics

#### System Entities
- `sensor.ha_intent_predictor_status` - Overall system status
- `sensor.pattern_discovery_status` - Pattern discovery progress

### Example Automations

The system creates automation templates in `/opt/ha-intent-predictor/config/automation_templates.yaml`:

```yaml
# Preheating automation
- alias: "Preheat Bedroom"
  trigger:
    - platform: numeric_state
      entity_id: sensor.occupancy_prediction_bedroom_120min
      above: 0.7
  condition:
    - condition: numeric_state
      entity_id: sensor.occupancy_prediction_bedroom_120min
      attribute: confidence
      above: 0.6
  action:
    - service: climate.set_temperature
      target:
        entity_id: climate.bedroom
      data:
        temperature: 22
```

### Dashboard Configuration

A complete Lovelace dashboard configuration is generated at `/opt/ha-intent-predictor/config/dashboard.yaml`.

## 🧠 Understanding the Learning System

### How It Works

1. **Data Ingestion**: Continuously streams data from all your HA sensors
2. **Pattern Discovery**: Automatically discovers occupancy patterns without assumptions
3. **Online Learning**: Models update with every new observation
4. **Multi-Horizon Prediction**: Predicts occupancy 15min, 30min, 60min, and 120min ahead
5. **Anomaly Detection**: Identifies unusual patterns (including cat activity)
6. **Person-Specific Learning**: Learns individual behavior patterns

### No Configuration Required

The system is designed to learn everything from your data:
- **No schedules to define** - discovers your routines automatically
- **No patterns to configure** - finds what actually exists in your behavior
- **No thresholds to set** - adapts to your specific occupancy patterns
- **No room-specific rules** - learns each room's unique characteristics

### Monitoring Learning Progress

Check the learning progress:

```bash
# View pattern discovery status
curl http://<container-ip>/api/patterns/status

# Check model performance
curl http://<container-ip>/api/models/performance

# View current predictions
curl http://<container-ip>/api/predictions/current
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Container Won't Start
```bash
# Check container status
pct status <container_id>

# Check container logs
pct enter <container_id>
journalctl -xe
```

#### 2. Services Not Starting
```bash
# Check service dependencies
systemctl list-dependencies ha-intent-predictor.service

# Check Docker services
docker compose ps
docker compose logs
```

#### 3. No Historical Data Imported
```bash
# Check HA connection
cd /opt/ha-intent-predictor
python scripts/ha_integration_setup.py --ha-url http://YOUR_HA_IP:8123 --ha-token YOUR_TOKEN --test-only

# Manual data import
python scripts/bootstrap.py --days 30 --batch-size 500
```

#### 4. Predictions Not Appearing in HA
```bash
# Check HA integration
python scripts/ha_integration_setup.py --ha-url http://YOUR_HA_IP:8123 --ha-token YOUR_TOKEN

# Verify entities were created
curl -H "Authorization: Bearer YOUR_TOKEN" http://YOUR_HA_IP:8123/api/states | grep occupancy_prediction
```

### Performance Issues

#### High Memory Usage
```bash
# Check memory usage
free -h
docker stats

# Adjust memory limits in systemd services
systemctl edit ha-intent-predictor.service
```

#### High CPU Usage
```bash
# Check CPU usage
top
htop

# Reduce model update frequency
nano /opt/ha-intent-predictor/config/app.yaml
# Increase model_update_interval
```

### Log Analysis

#### Enable Debug Logging
```bash
# Edit configuration
nano /opt/ha-intent-predictor/config/app.yaml

# Change logging level to DEBUG
logging:
  level: DEBUG

# Restart services
systemctl restart ha-intent-predictor.service
```

#### Common Log Patterns
```bash
# Connection issues
grep -i "connection" /var/log/ha-intent-predictor/app.log

# Model training issues
grep -i "training\|model" /var/log/ha-intent-predictor/app.log

# Database issues
grep -i "database\|postgres" /var/log/ha-intent-predictor/app.log
```

## 🔄 Updates and Maintenance

### Updating the System

```bash
# Enter container
pct enter <container_id>

# Update system packages
apt update && apt upgrade -y

# Update Python packages
cd /opt/ha-intent-predictor
source venv/bin/activate
pip install --upgrade -r requirements.txt

# Restart services
systemctl restart ha-intent-predictor.service
```

### Database Maintenance

```bash
# Access PostgreSQL
docker exec -it ha-predictor-postgres psql -U ha_predictor -d ha_predictor

# Check database size
SELECT pg_size_pretty(pg_database_size('ha_predictor'));

# View table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname IN ('raw_data', 'features', 'models', 'analytics')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Backup and Recovery

```bash
# Database backup
docker exec ha-predictor-postgres pg_dump -U ha_predictor ha_predictor > backup_$(date +%Y%m%d).sql

# Configuration backup
tar -czf config_backup_$(date +%Y%m%d).tar.gz /opt/ha-intent-predictor/config/

# Full system backup
tar -czf full_backup_$(date +%Y%m%d).tar.gz /opt/ha-intent-predictor/
```

## 📞 Support

### Getting Help

1. **Check the logs** first for error messages
2. **Review the configuration** files for correct sensor mappings
3. **Test HA connectivity** using the provided scripts
4. **Check system resources** (memory, CPU, disk)

### Reporting Issues

When reporting issues, include:
- Container configuration details
- Relevant log excerpts
- Home Assistant version and configuration
- Error messages and stack traces
- Steps to reproduce the issue

### Community Resources

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Detailed technical documentation
- **Example Configurations**: Sample configurations for common setups

## 🎯 Performance Optimization

### For Limited Resources

If running on limited hardware:

```yaml
# In app.yaml, reduce resource usage:
ml:
  model_update_interval: 600  # 10 minutes instead of 5
  prediction_horizons: [15, 60]  # Fewer horizons
  
performance:
  batch_size: 500  # Smaller batches
  max_concurrent_predictions: 5  # Fewer concurrent operations
```

### For High Performance

If you have more resources:

```yaml
# In app.yaml, increase performance:
ml:
  model_update_interval: 60  # 1 minute updates
  prediction_horizons: [5, 15, 30, 60, 120, 240]  # More horizons
  
performance:
  batch_size: 2000  # Larger batches
  max_concurrent_predictions: 20  # More concurrent operations
```

## 🎉 Success Indicators

The system is working correctly when you see:

1. **Entities in HA**: All prediction entities showing up with reasonable values
2. **Learning Progress**: Pattern discovery status showing progress
3. **Stable Predictions**: Predictions stabilizing after initial learning period
4. **Performance Metrics**: Model accuracy improving over time
5. **Resource Usage**: Stable memory and CPU usage within limits

The system typically takes 24-48 hours to start showing reliable predictions as it learns your patterns.