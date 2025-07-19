# ML Occupancy Prediction System - Implementation Guide

## ⚠️ CRITICAL DEVELOPMENT RULES

**1. NEVER MAKE CODE CHANGES WITHOUT EXPLICIT PERMISSION**
- Always ask before modifying existing functions, classes, or parameters
- Never simplify, alter, or "improve" code without user approval  
- Only fix import errors and references when explicitly requested
- Preserve all existing functionality and architecture

**2. ALWAYS CHECK IF FILES/FEATURES ALREADY EXIST BEFORE CREATING ANYTHING NEW**
- Use `find`, `ls`, `grep`, or `Read` tools to check existing files
- Check repository structure before implementing
- Avoid duplicating existing functionality
- Build upon existing codebase, don't recreate

## 🚀 DEPLOYED SYSTEM ACCESS

**Current Installation**: Container ID 200 on Proxmox, IP: `192.168.51.10`

### Quick Access Commands (Password-Free)
```bash
# Monitor system status
ssh ha-predictor '/opt/ha-intent-predictor/scripts/remote-monitor.sh show'

# Get JSON status for programmatic access
ssh ha-predictor '/opt/ha-intent-predictor/scripts/remote-monitor.sh json'

# Check Docker services
ssh ha-predictor 'cd /opt/ha-intent-predictor && docker compose ps'

# View service logs
ssh ha-predictor 'cd /opt/ha-intent-predictor && docker compose logs --tail 20'

# SSH into container directly
ssh ha-predictor

# Restart services if needed
ssh ha-predictor 'cd /opt/ha-intent-predictor && docker compose restart'
```

### Service Endpoints
- **PostgreSQL**: `192.168.51.10:5432` (user: `ha_predictor`, db: `ha_predictor`, password: `hapredictor_db_pass`)
- **Redis**: `192.168.51.10:6379`
- **Kafka**: `192.168.51.10:9092`
- **Zookeeper**: `192.168.51.10:2181`
- **Web Health Check**: `http://192.168.51.10/health`

### SSH Configuration
- **SSH Config**: `~/.ssh/config` contains `ha-predictor` host entry
- **Private Key**: `~/.ssh/ha-predictor`
- **No Password Required**: Key-based authentication configured

### Container Credentials
- **SSH User**: `root`
- **SSH Password**: `hapredictor123` (backup, key auth preferred)
- **Container Management**: `pct enter 200` (from Proxmox host)

## Executive Summary

This guide details the implementation of an adaptive ML-based occupancy prediction system designed to run on a Proxmox LXC container (Intel N200, 4 cores, 8GB RAM). The system learns occupancy patterns dynamically without any hardcoded schedules, predicting room occupancy 2 hours ahead for preheating and 15 minutes ahead for precooling.

## Core Philosophy

**No Assumptions, Pure Learning**: This system makes NO assumptions about daily routines, schedules, or patterns. It learns entirely from observed behavior, adapting continuously as patterns change or don't exist at all.

### Why This Approach?

You mentioned that you don't follow strict patterns that you can see - this is exactly why this system:
- Never assumes "people wake up at 7am" or "bathrooms are used for 5-10 minutes"
- Discovers if patterns exist in your data, no matter how subtle or complex
- Adapts to completely random behavior by providing appropriate uncertainty estimates
- Learns person-specific behaviors (Anca vs Vladimir) without assumptions
- Handles the combined living/kitchen space as one fluid area with multiple activity zones
- Deals with cat interference by learning what cat movement looks like in YOUR home

### Understanding Your Zone System

Your presence detection uses two complementary zone types:
- **Full zones**: Detect presence anywhere in the room (including uncovered areas)
- **Subzones**: Detect presence at specific locations (couch, desk, stove, etc.)

This dual system provides rich information:
- Full zone ON + Subzones OFF = Person in transit or in uncovered area
- Full zone ON + Specific subzones ON = Person at known location doing specific activity
- Movement patterns from general → specific → different specific tell a story

## System Architecture

### Components

1. **Continuous Data Ingestion** - Aggressive sensor data collection from Home Assistant
2. **Dynamic Feature Discovery** - Automatically identifies relevant patterns
3. **Adaptive Prediction Engine** - Self-adjusting models that learn behavior
4. **Real-time Model Updates** - Continuous learning from new observations
5. **HA Integration Service** - Publishes predictions back to Home Assistant

### Technology Stack

- **Language**: Python 3.11
- **ML Framework**: scikit-learn + LightGBM + River (for online learning)
- **Data Storage**: PostgreSQL with TimescaleDB extension (time-series optimized)
- **Cache**: Redis for real-time feature computation
- **API**: FastAPI for REST endpoints
- **Processing**: Apache Kafka for event streaming

## ML Strategy - Fully Adaptive Approach

### Core Principles

1. **No Hardcoded Patterns**: The system discovers all patterns through observation
2. **Continuous Learning**: Models update with every new data point
3. **Multi-timescale Analysis**: Automatically detects relevant time windows (could be 17 minutes, 2.3 hours, or 5.7 days)
4. **Probabilistic Everything**: All predictions include uncertainty estimates

### Model Architecture

```python
class AdaptiveOccupancyPredictor:
    def __init__(self):
        # Online learning models that update with each observation
        self.short_term_models = {}  # Per-room adaptive models
        self.long_term_models = {}   # Per-room pattern discoverers
        self.cat_detector = OnlineAnomalyDetector()
        self.feature_selector = AutoML(max_features=50)
        
    def learn_from_observation(self, room_id, sensor_data, outcome):
        # Every single observation updates the model
        # No batching, no waiting for "enough data"
        pass
```

### Dynamic Feature Engineering

Instead of predefined features, the system discovers what matters:

```python
class DynamicFeatureDiscovery:
    def __init__(self):
        self.feature_importance_tracker = {}
        self.interaction_detector = InteractionDiscovery()
        self.temporal_pattern_miner = TemporalMiner()
        
    def discover_features(self, sensor_stream):
        # Extract different types of features from zone combinations
        zone_features = self.extract_zone_combination_features(sensor_stream)
        
        # Automatically generate and test feature combinations
        # Keep only statistically significant features
        # Detect non-linear interactions between sensors
        # Find variable-length temporal patterns
        
        return self.select_significant_features(zone_features)
    
    def extract_zone_combination_features(self, sensor_stream):
        """
        Extract features from full zone + subzone combinations
        """
        features = {
            'full_without_sub': self.detect_general_area_presence(sensor_stream),
            'specific_locations': self.extract_subzone_patterns(sensor_stream),
            'zone_coverage': self.calculate_zone_coverage(sensor_stream),
            'movement_precision': self.analyze_movement_granularity(sensor_stream)
        }
        
        return features
    
    def detect_general_area_presence(self, stream):
        """
        Detect when someone is in general area but not in any subzone
        This indicates transitional movement or areas without subzone coverage
        """
        patterns = []
        for event in stream:
            if event['zone_type'] == 'full':
                # Check if any related subzones are active
                room_state = self.get_room_state(event['room'], event['timestamp'])
                subzones_active = any(
                    sz['state'] == 'on' 
                    for sz in room_state['subzones']
                )
                
                if not subzones_active:
                    # Person in general area, not in specific location
                    patterns.append({
                        'type': 'general_presence',
                        'room': event['room'],
                        'timestamp': event['timestamp'],
                        'indicates': 'transitional_movement'
                    })
        
        return patterns
    
    def analyze_movement_granularity(self, stream):
        """
        Analyze movement patterns using both full and subzone data
        """
        movement_patterns = {
            'precise_movements': [],  # Subzone to subzone
            'general_movements': [],  # Full zone without subzone detail
            'hybrid_movements': []    # Combinations
        }
        
        for i in range(len(stream) - 1):
            curr = stream[i]
            next = stream[i + 1]
            
            if curr['zone_type'] == 'subzone' and next['zone_type'] == 'subzone':
                # Precise movement tracking
                movement_patterns['precise_movements'].append({
                    'from': curr['specific_location'],
                    'to': next['specific_location'],
                    'duration': next['timestamp'] - curr['timestamp']
                })
            elif curr['zone_type'] == 'full' and next['zone_type'] == 'full':
                # General movement between rooms
                movement_patterns['general_movements'].append({
                    'from_room': curr['room'],
                    'to_room': next['room'],
                    'precision': 'low'
                })
            else:
                # Hybrid - entering/leaving specific zones
                movement_patterns['hybrid_movements'].append({
                    'transition_type': f"{curr['zone_type']}_to_{next['zone_type']}",
                    'details': (curr, next)
                })
        
        return movement_patterns
```

## Detailed Implementation

### File Structure
```
occupancy_predictor/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── ha_stream.py         # Continuous HA data streaming
│   │   ├── event_processor.py   # Real-time event processing
│   │   └── data_enricher.py     # Dynamic feature computation
│   ├── learning/
│   │   ├── __init__.py
│   │   ├── online_models.py     # Continuously updating models
│   │   ├── pattern_discovery.py # Automatic pattern mining
│   │   ├── anomaly_detection.py # Cat and anomaly detection
│   │   └── meta_learner.py      # Learns which models work when
│   ├── prediction/
│   │   ├── __init__.py
│   │   ├── ensemble.py          # Combines multiple predictors
│   │   ├── uncertainty.py       # Confidence estimation
│   │   └── explainer.py         # Explains predictions
│   ├── adaptation/
│   │   ├── __init__.py
│   │   ├── drift_detection.py   # Detects behavior changes
│   │   ├── model_evolution.py   # Evolves model architecture
│   │   └── hyperopt.py          # Continuous hyperparameter tuning
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── ha_publisher.py      # HA integration
│   │   ├── api.py               # REST endpoints
│   │   └── monitoring.py        # Performance tracking
│   └── storage/
│       ├── __init__.py
│       ├── timeseries_db.py     # TimescaleDB operations
│       ├── feature_store.py     # Redis feature cache
│       └── model_store.py       # Model versioning
├── config/
│   ├── sensors.yaml             # Sensor mappings only
│   └── rooms.yaml               # Room definitions only
├── scripts/
│   ├── bootstrap.py             # Initial system setup
│   ├── historical_import.py     # Import 180 days of data
│   └── deploy.py                # Deployment automation
└── requirements.txt
```

### Core Implementation Details

#### 1. Continuous Data Streaming

```python
class HADataStream:
    def __init__(self, ha_url, token):
        self.ha = HomeAssistantAPI(ha_url, token)
        self.kafka_producer = KafkaProducer()
        
    async def stream_all_sensors(self):
        """
        Aggressively pull all sensor data continuously
        """
        # Subscribe to ALL state changes
        async for event in self.ha.subscribe_events():
            # Transform to standardized format
            enriched_event = self.enrich_event(event)
            
            # Send to Kafka for processing
            await self.kafka_producer.send('sensor_events', enriched_event)
            
            # Immediate feature computation
            await self.compute_streaming_features(enriched_event)
    
    def enrich_event(self, event):
        """
        Add context without making assumptions
        """
        # Special handling for combined living/kitchen space
        room = self.identify_room(event.entity_id)
        if room in ['livingroom', 'kitchen']:
            room = 'living_kitchen'  # Unified space
            
        return {
            'timestamp': event.timestamp,
            'entity_id': event.entity_id,
            'state': event.state,
            'attributes': event.attributes,
            'room': room,
            'sensor_type': self.identify_sensor_type(event.entity_id),
            'derived': {
                'time_since_last_change': self.calculate_time_delta(event),
                'state_transition': self.get_transition_type(event),
                'concurrent_events': self.find_concurrent_events(event),
                'zone_info': self.extract_zone_info(event.entity_id)
            }
        }
    
    def extract_zone_info(self, entity_id):
        """
        Extract zone information from multi-zone rooms
        Full zones and subzones provide complementary information
        """
        # Define zone relationships
        zone_config = {
            'bedroom': {
                'full_zone': 'full_bedroom',
                'subzones': ['anca_bed_side', 'vladimir_bed_side', 'floor', 'entrance']
            },
            'office': {
                'full_zone': 'full_office', 
                'subzones': ['anca_desk', 'vladimir_desk', 'entrance']
            },
            'living_kitchen': {
                'full_zones': ['livingroom_full', 'kitchen_full'],
                'subzones': {
                    'livingroom': ['livingroom_couch'],
                    'kitchen': ['kitchen_stove', 'kitchen_sink', 'kitchen_dining']
                }
            }
        }
        
        # Identify zone type
        for room, config in zone_config.items():
            if room == 'living_kitchen':
                # Handle dual full zones
                if 'livingroom_full' in entity_id:
                    return {
                        'room': room,
                        'zone': 'livingroom_full',
                        'zone_type': 'full',
                        'coverage': 'general_living_area',
                        'related_subzones': ['livingroom_couch']
                    }
                elif 'kitchen_full' in entity_id:
                    return {
                        'room': room,
                        'zone': 'kitchen_full',
                        'zone_type': 'full',
                        'coverage': 'general_kitchen_area',
                        'related_subzones': ['kitchen_stove', 'kitchen_sink', 'kitchen_dining']
                    }
                # Check subzones
                for area, subzones in config['subzones'].items():
                    for subzone in subzones:
                        if subzone in entity_id:
                            return {
                                'room': room,
                                'zone': subzone,
                                'zone_type': 'subzone',
                                'area': area,
                                'specific_location': subzone
                            }
            else:
                # Single full zone rooms
                if config['full_zone'] in entity_id:
                    return {
                        'room': room,
                        'zone': config['full_zone'],
                        'zone_type': 'full',
                        'coverage': 'general_room_area'
                    }
                for subzone in config['subzones']:
                    if subzone in entity_id:
                        return {
                            'room': room,
                            'zone': subzone,
                            'zone_type': 'subzone',
                            'specific_location': subzone
                        }
        
        return {'room': self.identify_room(entity_id), 'zone': 'main', 'zone_type': 'unknown'}
```

#### 2. Adaptive Pattern Discovery

```python
class PatternDiscovery:
    def __init__(self):
        self.pattern_library = {}
        self.statistical_tests = StatisticalTestSuite()
        
    def discover_patterns(self, event_stream, room_id):
        """
        No assumptions - let data speak for itself
        """
        # Variable-length sequence mining
        sequences = self.extract_sequences(event_stream, 
                                         min_length=2, 
                                         max_length=100)
        
        # Test every possible time window
        for window in self.generate_time_windows():
            pattern_strength = self.test_pattern_significance(
                sequences, window, room_id
            )
            
            if pattern_strength > self.adaptive_threshold:
                self.pattern_library[room_id].add(
                    Pattern(sequences, window, pattern_strength)
                )
        
        # Detect anti-patterns (what DOESN'T happen)
        self.discover_negative_patterns(event_stream, room_id)
    
    def generate_time_windows(self):
        """
        Test all possible time windows, not just "hourly" or "daily"
        """
        # Windows from 1 minute to 30 days, with variable steps
        windows = []
        for minutes in range(1, 43200):  # Up to 30 days
            if self.is_promising_window(minutes):
                windows.append(minutes)
        return windows
```

#### 3. Online Learning Models

```python
from river import ensemble, preprocessing, metrics

class ContinuousLearningModel:
    def __init__(self, room_id):
        self.room_id = room_id
        
        # Ensemble of online learners
        self.models = {
            'gradient_boost': ensemble.AdaptiveRandomForestClassifier(
                n_models=10,
                max_features="sqrt",
                lambda_value=6,
                grace_period=10
            ),
            'hoeffding_tree': tree.ExtremelyFastDecisionTreeClassifier(
                grace_period=10,
                split_confidence=1e-5,
                nominal_attributes=['sensor_type', 'room']
            ),
            'neural': neural_net.MLPClassifier(
                hidden_dims=(20, 10),
                activations=('relu', 'relu', 'identity'),
                learning_rate=0.001
            )
        }
        
        # Meta-learner decides which model to trust
        self.meta_learner = MetaLearner()
        
        # Track each model's performance
        self.model_performance = {
            name: metrics.Rolling(metrics.ROCAUC(), window_size=1000)
            for name in self.models
        }
    
    def learn_one(self, features, y_true):
        """
        Update all models with single observation
        """
        # Update each model
        for name, model in self.models.items():
            y_pred = model.predict_proba_one(features)
            
            # Update performance tracking
            if y_pred is not None:
                self.model_performance[name].update(y_true, y_pred[True])
            
            # Learn from this observation
            model.learn_one(features, y_true)
        
        # Update meta-learner
        self.meta_learner.update(
            self.get_model_predictions(features),
            y_true
        )
    
    def predict_proba_one(self, features):
        """
        Ensemble prediction with uncertainty
        """
        predictions = self.get_model_predictions(features)
        
        # Meta-learner weights each model
        weights = self.meta_learner.get_weights(
            predictions, 
            self.model_performance
        )
        
        # Weighted ensemble
        final_pred = sum(
            pred * weight 
            for pred, weight in zip(predictions.values(), weights)
        )
        
        # Calculate uncertainty
        uncertainty = self.calculate_prediction_uncertainty(predictions, weights)
        
        return {
            'probability': final_pred,
            'uncertainty': uncertainty,
            'contributing_models': self.get_contribution_explanation(predictions, weights)
        }
```

#### 4. Cat Activity Detection Without Assumptions

```python
class AdaptiveCatDetector:
    def __init__(self):
        self.movement_clusterer = IncrementalDBSCAN()
        self.impossible_sequences = set()
        self.person_specific_patterns = {
            'anca': {'common_zones': set(), 'movement_speed': RunningStats()},
            'vladimir': {'common_zones': set(), 'movement_speed': RunningStats()}
        }
        
    def learn_movement_patterns(self, sensor_sequence):
        """
        Learn what's normal vs abnormal without hardcoding
        Special handling for multi-person households
        """
        # Extract movement velocity between sensors
        velocities = self.calculate_velocities(sensor_sequence)
        
        # Check if movement is between person-specific zones
        person = self.identify_person_from_zones(sensor_sequence)
        if person:
            self.person_specific_patterns[person]['movement_speed'].update(velocities)
        
        # Cluster movements - outliers might be cats
        cluster_label = self.movement_clusterer.partial_fit_predict(velocities)
        
        if cluster_label == -1:  # Outlier
            # This might be cat movement
            self.analyze_outlier(sensor_sequence)
    
    def identify_person_from_zones(self, sequence):
        """
        Identify if movement is person-specific based on zones
        """
        zones = [event['zone_info']['zone'] for event in sequence]
        
        if any('anca' in zone for zone in zones):
            return 'anca'
        elif any('vladimir' in zone for zone in zones):
            return 'vladimir'
        
        return None
    
    def analyze_outlier(self, sequence):
        """
        Determine if outlier is cat vs human edge case
        """
        # Look for patterns like:
        # - Simultaneous triggers in distant rooms
        # - Repeated fast transitions
        # - Sensors that only trigger with other cat-like patterns
        # - Movement through multiple zones too quickly
        # - Presence without door opening (for rooms with doors)
        
        features = self.extract_anomaly_features(sequence)
        
        # Special check for impossible human movements
        if self.is_physically_impossible(sequence):
            self.tag_as_cat_activity(sequence)
            return
        
        # Let the system learn what's cat vs unusual human
        if self.is_consistent_with_previous_cats(features):
            self.tag_as_cat_activity(sequence)
            
    def is_physically_impossible(self, sequence):
        """
        Check for movements that violate physics for humans
        Uses both full and subzone data for better accuracy
        """
        for i in range(len(sequence) - 1):
            curr = sequence[i]
            next = sequence[i + 1]
            
            time_diff = next['timestamp'] - curr['timestamp']
            
            # Check door constraints for bathrooms
            if next['room'] in ['bathroom', 'small_bathroom']:
                door_entity = f"binary_sensor.{next['room']}_door_sensor_contact"
                if not self.was_door_opened(door_entity, curr['timestamp'], next['timestamp']):
                    return True
            
            # Analyze movement pattern
            movement = self.analyze_movement(curr, next, time_diff)
            
            if movement['impossible']:
                return True
                
        return False
    
    def analyze_movement(self, curr_event, next_event, time_diff):
        """
        Analyze if movement is possible for humans
        """
        curr_zone = curr_event.get('zone_info', {})
        next_zone = next_event.get('zone_info', {})
        
        # Different room transitions
        if curr_zone['room'] != next_zone['room']:
            # Check if transition time is reasonable
            min_transition_time = self.get_min_transition_time(
                curr_zone['room'], next_zone['room']
            )
            
            if time_diff < min_transition_time:
                # Too fast for human movement
                return {'impossible': True, 'reason': 'inter_room_speed'}
        
        # Same room but different zones
        elif curr_zone['zone'] != next_zone['zone']:
            # Subzone to subzone movement
            if (curr_zone['zone_type'] == 'subzone' and 
                next_zone['zone_type'] == 'subzone'):
                
                # Check specific subzone transitions
                if self.is_subzone_transition_impossible(
                    curr_zone['zone'], next_zone['zone'], time_diff
                ):
                    return {'impossible': True, 'reason': 'subzone_speed'}
            
            # Full zone to subzone or vice versa
            elif (curr_zone['zone_type'] != next_zone['zone_type']):
                # This is normal - person moving from general area to specific location
                return {'impossible': False, 'reason': 'normal_precision_change'}
        
        return {'impossible': False}
    
    def is_subzone_transition_impossible(self, zone1, zone2, time_seconds):
        """
        Check if specific subzone transition is too fast
        """
        # Define minimum transition times between subzones
        transition_times = {
            ('bedroom_anca_side', 'kitchen_stove'): 5,  # Across house
            ('bedroom_vladimir_side', 'kitchen_sink'): 5,
            ('office_anca_desk', 'bedroom_vladimir_side'): 3,  # Different rooms
            ('office_vladimir_desk', 'bedroom_anca_side'): 3,
            ('livingroom_couch', 'bedroom_floor'): 4,
            # Same room transitions are faster
            ('kitchen_stove', 'kitchen_sink'): 1,
            ('bedroom_anca_side', 'bedroom_vladimir_side'): 2,
            ('office_anca_desk', 'office_vladimir_desk'): 2,
        }
        
        # Check both directions
        min_time = transition_times.get((zone1, zone2)) or \
                  transition_times.get((zone2, zone1))
        
        if min_time and time_seconds < min_time:
            return True
            
        # Check for physically impossible same-instant triggers
        if time_seconds < 0.5 and zone1 != zone2:
            # Different zones triggered within 0.5 seconds - likely cat
            return True
            
        return False
```

#### 5. Multi-Horizon Prediction

```python
class AdaptiveHorizonPredictor:
    def __init__(self):
        # Don't assume 15 min and 2 hours are optimal
        self.horizon_optimizer = HorizonOptimizer()
        self.predictors = {}
        
    def optimize_prediction_horizons(self, historical_accuracy):
        """
        Find the actual optimal prediction horizons
        """
        # Test predictions from 1 minute to 4 hours
        test_horizons = range(1, 240)
        
        results = {}
        for horizon in test_horizons:
            accuracy = self.test_horizon_accuracy(horizon, historical_accuracy)
            results[horizon] = accuracy
        
        # Find natural breakpoints where accuracy drops
        optimal_horizons = self.find_accuracy_breakpoints(results)
        
        # Might discover that 23 minutes and 1.7 hours are optimal
        return optimal_horizons
    
    def create_predictor_for_horizon(self, horizon_minutes):
        """
        Create specialized predictor for discovered horizon
        """
        return HorizonSpecificPredictor(
            horizon=horizon_minutes,
            feature_lookback=self.optimize_lookback(horizon_minutes),
            model_type=self.select_best_model_type(horizon_minutes)
        )
```

#### 6. Integration with Home Assistant

```python
class DynamicHAIntegration:
    def __init__(self, ha_api):
        self.ha = ha_api
        self.prediction_entities = {}
        
    def publish_predictions(self, room_id, predictions):
        """
        Create dynamic entities based on discovered patterns
        """
        for horizon, prediction in predictions.items():
            entity_id = f"sensor.occupancy_{room_id}_{horizon}min"
            
            # Create entity if it doesn't exist
            if entity_id not in self.prediction_entities:
                self.create_prediction_entity(entity_id, room_id, horizon)
            
            # Update with prediction and metadata
            self.ha.set_state(entity_id, {
                'state': prediction['probability'],
                'attributes': {
                    'uncertainty': prediction['uncertainty'],
                    'confidence': 1 - prediction['uncertainty'],
                    'contributing_factors': prediction['factors'],
                    'model_agreement': prediction['model_agreement'],
                    'last_updated': datetime.now().isoformat(),
                    'horizon_minutes': horizon,
                    'explanation': prediction.get('explanation', '')
                }
            })
    
    def create_automation_helpers(self, room_id):
        """
        Create helper entities for complex automations
        """
        # Trends
        self.ha.set_state(f"sensor.occupancy_trend_{room_id}", {
            'state': self.calculate_occupancy_trend(room_id),
            'attributes': {
                'trend_strength': self.trend_strength,
                'change_probability': self.change_probability
            }
        })
        
        # Anomalies
        self.ha.set_state(f"binary_sensor.occupancy_anomaly_{room_id}", {
            'state': self.detect_anomalous_pattern(room_id),
            'attributes': {
                'anomaly_score': self.anomaly_score,
                'anomaly_type': self.anomaly_type
            }
        })
```

### Deployment and Operation

#### Initial Setup with Your Sensor Data

```python
# bootstrap.py
async def bootstrap_system():
    # Import your 180 days of historical data
    print("Importing historical data from all 98 sensors...")
    importer = HistoricalDataImporter()
    
    # Define sensor groups for easier processing
    sensor_groups = {
        'presence_zones': [
            'binary_sensor.presence_livingroom_full',
            'binary_sensor.presence_livingroom_couch',
            'binary_sensor.kitchen_pressence_full_kitchen',
            'binary_sensor.kitchen_pressence_stove',
            'binary_sensor.kitchen_pressence_sink',
            'binary_sensor.kitchen_pressence_dining_table',
            'binary_sensor.bedroom_presence_sensor_full_bedroom',
            'binary_sensor.bedroom_presence_sensor_anca_bed_side',
            'binary_sensor.bedroom_vladimir_bed_side',
            'binary_sensor.office_presence_full_office',
            'binary_sensor.office_presence_anca_desk',
            'binary_sensor.office_presence_vladimir_desk',
            # ... all other presence sensors
        ],
        'doors': [
            'binary_sensor.bathroom_door_sensor_contact',
            'binary_sensor.bedroom_door_sensor_contact',
            'binary_sensor.office_door_sensor_contact',
            'binary_sensor.guest_bedroom_door_sensor_contact',
            'binary_sensor.small_bathroom_door_sensor_contact'
        ],
        'climate': [
            # All temperature and humidity sensors
        ]
    }
    
    await importer.import_from_ha(days=180, sensor_groups=sensor_groups)
    
    # Special handling for combined living/kitchen space
    print("Configuring unified living/kitchen space...")
    await importer.merge_room_data('livingroom', 'kitchen', new_name='living_kitchen')
    
    # Let system discover all patterns - no preconceptions
    print("Discovering patterns... this may take a while")
    discoverer = PatternDiscovery()
    
    # Discover patterns for each room type
    discoveries = await asyncio.gather(
        discoverer.discover_multizone_patterns('living_kitchen'),
        discoverer.discover_multizone_patterns('bedroom'),
        discoverer.discover_multizone_patterns('office'),
        discoverer.discover_bathroom_patterns(['bathroom', 'small_bathroom']),
        discoverer.discover_transition_patterns('hallways')
    )
    
    # Initialize person-specific learning
    print("Initializing person-specific pattern learning...")
    person_learner = PersonSpecificLearner()
    await person_learner.initialize(['anca', 'vladimir'])
    
    # Start continuous learning
    print("Starting adaptive learning system...")
    learner = ContinuousLearningSystem()
    await learner.start()
    
    # Set up Home Assistant integration
    print("Creating Home Assistant entities...")
    ha_integration = DynamicHAIntegration()
    
    # Create prediction entities for actual rooms
    rooms_to_predict = [
        'living_kitchen',  # Combined space
        'bedroom',
        'office',
        'bathroom',
        'small_bathroom',
        'guest_bedroom'
    ]
    
    for room in rooms_to_predict:
        await ha_integration.create_room_predictors(room)
    
    print("System ready - all patterns will be learned from observation")
    print("No schedules or patterns are assumed - everything is data-driven")
```

#### Configuration Files (Minimal - No Patterns)

```yaml
# sensors.yaml - Just mappings, no patterns
sensors:
  presence:
    # Living room/Kitchen combined space
    - binary_sensor.presence_livingroom_full
    - binary_sensor.presence_livingroom_couch
    - binary_sensor.kitchen_pressence_full_kitchen
    - binary_sensor.kitchen_pressence_stove
    - binary_sensor.kitchen_pressence_sink
    - binary_sensor.kitchen_pressence_dining_table
    
    # Bedroom zones
    - binary_sensor.bedroom_presence_sensor_full_bedroom
    - binary_sensor.bedroom_presence_sensor_anca_bed_side
    - binary_sensor.bedroom_vladimir_bed_side
    - binary_sensor.bedroom_floor
    - binary_sensor.bedroom_entrance
    
    # Office zones
    - binary_sensor.office_presence_full_office
    - binary_sensor.office_presence_anca_desk
    - binary_sensor.office_presence_vladimir_desk
    - binary_sensor.office_entrance
    
    # Bathroom entrances (no presence inside)
    - binary_sensor.bathroom_entrance
    - binary_sensor.presence_small_bathroom_entrance
    
    # Other areas
    - binary_sensor.guest_bedroom_entrance
    - binary_sensor.presence_ground_floor_hallway
    - binary_sensor.upper_hallway
    - binary_sensor.upper_hallway_upstairs
    - binary_sensor.upper_hallway_downstairs
    - binary_sensor.presence_stairs_up_ground_floor
  
  doors:
    - binary_sensor.bathroom_door_sensor_contact
    - binary_sensor.bedroom_door_sensor_contact
    - binary_sensor.office_door_sensor_contact
    - binary_sensor.guest_bedroom_door_sensor_contact
    - binary_sensor.small_bathroom_door_sensor_contact
    
  climate:
    - sensor.livingroom_env_sensor_temperature
    - sensor.livingroom_env_sensor_humidity
    - sensor.bedroom_env_sensor_temperature
    - sensor.bedroom_env_sensor_humidity
    - sensor.office_env_sensor_temperature
    - sensor.office_env_sensor_humidity
    - sensor.bathroom_env_sensor_temperature
    - sensor.bathroom_env_sensor_humidity
    - sensor.guest_bedroom_env_sensor_temperature
    - sensor.guest_bedroom_env_sensor_humidity
    - sensor.upper_hallway_env_sensor_temperature
    - sensor.upper_hallway_env_sensor_humidity
    - sensor.attic_env_sensor_temperature
    - sensor.attic_env_sensor_humidity
    - sensor.big_bath_env_sensor_temperature
    - sensor.big_bath_env_sensor_humidity
    
  light_levels:
    - sensor.bedroom_presence_light_level
    - sensor.kitchen_pressence_light_level
    - sensor.livingroom_pressence_light_level
    - sensor.office_presence_light_level
    - sensor.upper_hallway_pressence_light_level

# rooms.yaml - Just structure, no schedules
rooms:
  living_kitchen:  # Combined space
    name: "Living Room & Kitchen"
    sensors:
      presence:
        livingroom_full: binary_sensor.presence_livingroom_full
        livingroom_couch: binary_sensor.presence_livingroom_couch
        kitchen_full: binary_sensor.kitchen_pressence_full_kitchen
        kitchen_stove: binary_sensor.kitchen_pressence_stove
        kitchen_sink: binary_sensor.kitchen_pressence_sink
        kitchen_dining: binary_sensor.kitchen_pressence_dining_table
      climate:
        temperature: sensor.livingroom_env_sensor_temperature
        humidity: sensor.livingroom_env_sensor_humidity
      light:
        livingroom: sensor.livingroom_pressence_light_level
        kitchen: sensor.kitchen_pressence_light_level
  
  bedroom:
    sensors:
      presence:
        full_room: binary_sensor.bedroom_presence_sensor_full_bedroom
        anca_side: binary_sensor.bedroom_presence_sensor_anca_bed_side
        vladimir_side: binary_sensor.bedroom_vladimir_bed_side
        floor: binary_sensor.bedroom_floor
        entrance: binary_sensor.bedroom_entrance
      door: binary_sensor.bedroom_door_sensor_contact
      climate:
        temperature: sensor.bedroom_env_sensor_temperature
        humidity: sensor.bedroom_env_sensor_humidity
      light: sensor.bedroom_presence_light_level
  
  office:
    sensors:
      presence:
        full_room: binary_sensor.office_presence_full_office
        anca_desk: binary_sensor.office_presence_anca_desk
        vladimir_desk: binary_sensor.office_presence_vladimir_desk
        entrance: binary_sensor.office_entrance
      door: binary_sensor.office_door_sensor_contact
      climate:
        temperature: sensor.office_env_sensor_temperature
        humidity: sensor.office_env_sensor_humidity
      light: sensor.office_presence_light_level
  
  bathroom:
    sensors:
      entrance: binary_sensor.bathroom_entrance
      door: binary_sensor.bathroom_door_sensor_contact
      climate:
        temperature: sensor.bathroom_env_sensor_temperature
        humidity: sensor.bathroom_env_sensor_humidity
  
  small_bathroom:
    sensors:
      entrance: binary_sensor.presence_small_bathroom_entrance
      door: binary_sensor.small_bathroom_door_sensor_contact
  
  guest_bedroom:
    sensors:
      entrance: binary_sensor.guest_bedroom_entrance
      door: binary_sensor.guest_bedroom_door_sensor_contact
      climate:
        temperature: sensor.guest_bedroom_env_sensor_temperature
        humidity: sensor.guest_bedroom_env_sensor_humidity
  
  hallways:
    ground_floor:
      presence: binary_sensor.presence_ground_floor_hallway
      stairs: binary_sensor.presence_stairs_up_ground_floor
    upper:
      presence: binary_sensor.upper_hallway
      upstairs: binary_sensor.upper_hallway_upstairs
      downstairs: binary_sensor.upper_hallway_downstairs
      climate:
        temperature: sensor.upper_hallway_env_sensor_temperature
        humidity: sensor.upper_hallway_env_sensor_humidity
      light: sensor.upper_hallway_pressence_light_level
```

### Bathroom Occupancy Prediction Strategy

Since bathrooms only have entrance zones and door sensors, the system needs special logic:

```python
class BathroomOccupancyPredictor:
    def __init__(self):
        self.bathroom_state = {
            'bathroom': {'occupied': False, 'entry_time': None, 'door_closed': False},
            'small_bathroom': {'occupied': False, 'entry_time': None, 'door_closed': False}
        }
        self.duration_learner = DurationPatternLearner()
        
    def process_bathroom_event(self, event):
        """
        Infer bathroom occupancy from entrance zones and door sensors
        """
        room = self.identify_bathroom(event.entity_id)
        if not room:
            return
            
        if 'entrance' in event.entity_id:
            # Someone triggered entrance zone
            if event.state == 'on':
                self.handle_entrance_trigger(room, event.timestamp)
            else:
                # Entrance zone cleared - might be leaving
                self.handle_entrance_clear(room, event.timestamp)
                
        elif 'door_sensor_contact' in event.entity_id:
            # Door state changed
            self.bathroom_state[room]['door_closed'] = (event.state == 'off')
            self.update_occupancy_logic(room, event.timestamp)
    
    def handle_entrance_trigger(self, room, timestamp):
        """
        Someone approached bathroom entrance
        """
        state = self.bathroom_state[room]
        
        if not state['occupied']:
            # Likely entering
            state['occupied'] = True
            state['entry_time'] = timestamp
            
            # Learn typical duration for this time of day
            predicted_duration = self.duration_learner.predict_duration(
                room, timestamp, self.get_context_features()
            )
            
            return {
                'room': room,
                'occupied': True,
                'predicted_duration': predicted_duration,
                'confidence': self.calculate_entry_confidence(room, timestamp)
            }
    
    def update_occupancy_logic(self, room, timestamp):
        """
        Complex logic combining entrance and door states
        """
        state = self.bathroom_state[room]
        
        # If door just closed after entrance trigger, high confidence occupied
        if state['door_closed'] and state['occupied']:
            return {'occupied': True, 'confidence': 0.95}
            
        # If door opened after being closed, might be leaving
        if not state['door_closed'] and state['occupied']:
            # Check if enough time passed for typical bathroom use
            duration = timestamp - state['entry_time']
            typical_duration = self.duration_learner.get_typical_duration(room, timestamp)
            
            if duration > typical_duration * 0.8:
                # Likely leaving
                state['occupied'] = False
                return {'occupied': False, 'confidence': 0.8}
    
    def learn_bathroom_patterns(self, historical_data):
        """
        Learn patterns specific to bathroom usage
        """
        # Learn:
        # - Typical duration by time of day
        # - Patterns of door usage (some people don't close doors)
        # - Correlation with other room activities
        # - Morning/evening routine patterns
        
        patterns = {
            'duration_by_hour': defaultdict(RunningStats),
            'door_usage_probability': defaultdict(float),
            'pre_bathroom_activity': defaultdict(Counter),
            'post_bathroom_activity': defaultdict(Counter)
        }
        
        return patterns
```

### Temporal Pattern Mining Without Assumptions

```python
class UnbiasedPatternMiner:
    def mine_patterns(self, event_stream):
        # Use suffix trees for efficient pattern matching
        suffix_tree = SuffixTree()
        
        for event_sequence in event_stream:
            suffix_tree.add(event_sequence)
        
        # Find statistically significant patterns
        patterns = []
        for pattern in suffix_tree.get_all_patterns():
            if self.is_significant(pattern):
                patterns.append(pattern)
        
        return patterns
    
    def is_significant(self, pattern):
        # Use statistical tests, not hardcoded thresholds
        # Check if pattern occurs more than random chance
        return chi_squared_test(pattern.frequency, pattern.expected_frequency)
```

### Dynamic Feature Importance

```python
class AdaptiveFeatureSelector:
    def __init__(self):
        self.feature_scores = defaultdict(lambda: RunningStats())
        
    def update_importance(self, features, prediction_error):
        # Use SHAP values in online manner
        for feature, value in features.items():
            contribution = self.estimate_contribution(feature, value, prediction_error)
            self.feature_scores[feature].update(contribution)
        
        # Prune features that don't contribute
        self.prune_useless_features()
    
    def get_top_features(self, n=50):
        # Return dynamically selected best features
        return sorted(self.feature_scores.items(), 
                     key=lambda x: x[1].mean(), 
                     reverse=True)[:n]
```

## Performance Optimization

### Efficient Stream Processing

```python
class StreamProcessor:
    def __init__(self):
        self.window_manager = SlidingWindowManager()
        self.feature_cache = TTLCache(maxsize=10000, ttl=300)
        
    async def process_event(self, event):
        # Compute features incrementally
        features = await self.incremental_feature_computation(event)
        
        # Update all relevant models asynchronously
        await asyncio.gather(
            self.update_short_term_models(features),
            self.update_long_term_models(features),
            self.update_pattern_miners(features)
        )
```

### Resource Management

```python
class ResourceOptimizer:
    def __init__(self, cpu_limit=80, memory_limit=6000):  # 6GB limit
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        
    def optimize_models(self, models):
        # Dynamically adjust model complexity based on resources
        current_usage = self.get_resource_usage()
        
        if current_usage['memory'] > self.memory_limit * 0.9:
            # Prune least important features
            self.prune_features(models)
            
        if current_usage['cpu'] > self.cpu_limit:
            # Reduce model update frequency for stable rooms
            self.adjust_update_frequency(models)
```

## Monitoring and Adaptation

### Continuous Performance Monitoring

```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'prediction_accuracy': RunningWindowMetric(window=1000),
            'false_positive_rate': RunningWindowMetric(window=1000),
            'model_drift': DriftDetector(),
            'feature_relevance': FeatureRelevanceTracker()
        }
    
    def log_prediction(self, room_id, prediction, actual):
        # Track everything
        self.metrics['prediction_accuracy'].update(prediction == actual)
        
        # Detect when models need updating
        if self.metrics['model_drift'].detect_drift():
            self.trigger_model_adaptation(room_id)
```

## Error Handling and Recovery

```python
class ResilientPredictor:
    def predict_with_fallback(self, room_id, horizon):
        try:
            # Try primary prediction
            return self.primary_predictor.predict(room_id, horizon)
        except Exception as e:
            logger.error(f"Primary prediction failed: {e}")
            
            # Fallback to simpler model
            try:
                return self.fallback_predictor.predict(room_id, horizon)
            except:
                # Last resort - return uncertain prediction
                return {
                    'probability': 0.5,
                    'uncertainty': 1.0,
                    'error': 'Prediction system temporarily unavailable'
                }
```

## Summary

This system is designed to learn everything from scratch without any assumptions about your behavior patterns. It continuously adapts to whatever patterns exist (or don't exist) in your actual occupancy data. The key principles:

1. **No hardcoded patterns or schedules**
2. **Continuous learning from every observation**
3. **Dynamic discovery of relevant features and time horizons**
4. **Adaptive models that evolve with your behavior**
5. **Full uncertainty quantification for all predictions**

The system will discover whether you have patterns or not, and adapt its prediction strategy accordingly. If your behavior is truly random, it will learn that too and provide appropriate uncertainty estimates.