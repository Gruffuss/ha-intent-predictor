# HA Intent Predictor

Adaptive ML-based occupancy prediction system for Home Assistant. Learns occupancy patterns without assumptions and provides 2-hour preheating and 15-minute precooling predictions.

## Features

- **Pure Learning**: No hardcoded patterns or schedules
- **Multi-Zone Support**: Full zones + subzones with person-specific detection
- **Real-time Adaptation**: Continuous learning with every observation
- **Uncertainty Quantification**: Confidence estimates for all predictions
- **Cat Activity Detection**: Physics-based anomaly detection
- **Ensemble Learning**: Multiple models with meta-learning weights
- **Production Ready**: Docker, systemd, monitoring included

## Quick Start

### 1. Environment Setup
```bash
# Clone and setup
git clone <repository>
cd ha-intent-predictor
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Copy environment template
cp systemd/environment.template .env

# Edit .env with your Home Assistant details
# Required: HA_URL, HA_TOKEN, DB_PASSWORD
```

### 3. Infrastructure (Docker)
```bash
# Start infrastructure services
docker-compose up -d timescaledb redis kafka

# Wait for services to be ready
docker-compose logs -f
```

### 4. Database Setup
```bash
# Create database schema
psql -h localhost -U postgres -f database/schema.sql
```

### 5. Start Application
```bash
python main.py
```

## API Access

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics

## Configuration

### Sensors (config/sensors.yaml)
Maps all 98 sensors to rooms and zones without assumptions.

### Rooms (config/rooms.yaml)
Defines room structure with full/subzone relationships.

### System (config/system.yaml)
Complete system configuration with environment variables.

## Deployment

### Development
```bash
python main.py
```

### Production
```bash
# Install system service
sudo python scripts/deploy.py install

# Start service
sudo systemctl start ha-intent-predictor
sudo systemctl enable ha-intent-predictor
```

### Docker
```bash
docker-compose up -d
```

## Monitoring

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Kafka UI**: http://localhost:8080

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Home Assistant │───▶│  HA Data Stream │───▶│   Kafka Queue   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   TimescaleDB   │◀───│ Adaptive ML     │◀───│ Event Processor │
│   (Storage)     │    │ Predictor       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Redis       │◀───│   FastAPI       │───▶│  HA Integration │
│  (Features)     │    │   (API)         │    │   (Entities)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Room Setup

The system supports your exact sensor configuration:

### Multi-Zone Rooms
- **Living/Kitchen**: Dual full zones + subzones (couch, stove, sink, dining)
- **Bedroom**: Full zone + person-specific bed sides (Anca/Vladimir)
- **Office**: Full zone + dual desks (Anca/Vladimir)

### Bathroom Inference
- **Bathrooms**: Entrance + door sensors for occupancy inference
- **No Interior Sensors**: Uses entry/exit patterns and door states

### Anomaly Detection
- **Cat Activity**: Detects impossible human movements
- **Sensor Validation**: Identifies malfunctioning sensors
- **Pattern Learning**: Adapts to your specific environment

## Learning Process

1. **Data Collection**: Aggressive sensor data streaming
2. **Feature Discovery**: Dynamic feature engineering
3. **Pattern Mining**: Temporal pattern discovery
4. **Model Selection**: Automatic algorithm selection
5. **Ensemble Learning**: Multiple models with meta-weights
6. **Uncertainty Estimation**: Confidence quantification
7. **Continuous Adaptation**: Real-time learning

## Troubleshooting

### Common Issues

**Database Connection Error**
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Check database exists
psql -h localhost -U postgres -l | grep ha_predictor
```

**Home Assistant Connection**
```bash
# Test HA connection
curl -H "Authorization: Bearer YOUR_TOKEN" http://YOUR_HA_URL/api/states
```

**Service Won't Start**
```bash
# Check logs
journalctl -u ha-intent-predictor -f

# Check configuration
python -c "from config.config_loader import ConfigLoader; print(ConfigLoader().get('home_assistant'))"
```

### Performance Tuning

**High Memory Usage**
- Reduce feature cache size in config
- Increase Redis memory limits
- Optimize model ensemble size

**Slow Predictions**
- Check database query performance
- Optimize feature computation
- Reduce ensemble size

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- **Issues**: https://github.com/your-org/ha-intent-predictor/issues
- **Documentation**: See `IMPLEMENTATION_SUMMARY.md`
- **CLAUDE.md**: Original specification document