"""
Event-Based Pattern Discovery System - STUMPY Replacement

Comprehensive event-based pattern discovery system that replaces the STUMPY matrix profile
approach with event-driven learning. Designed specifically for sparse occupancy data 
(84% zeros) and focuses on state transition events rather than time-sampled data.

Key Features:
1. HMM-based state transition learning (replaces STUMPY time series analysis)
2. Integrated data cleaning pipeline with event deduplication
3. Behavioral change point detection for model retraining triggers
4. Event sequence analysis instead of time-sampled windows
5. Sparse data optimization with 84% zero handling
6. Drop-in replacement for existing PatternDiscovery class

Components Integrated:
- HMMPredictor: Core state transition modeling
- BehavioralChangeDetector: Change point detection and retraining triggers  
- EventDeduplicator: Data cleaning for event sequences
- Event sequence processors: Direct event analysis without time sampling

Performance Target: >85% accuracy on historical validation data
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta, timezone
from collections import defaultdict, Counter, deque
from dataclasses import dataclass
import json
import warnings

# Core ML components
from .hmm_predictor import (
    HMMPredictor, 
    OccupancySequence, 
    HMMPrediction,
    MultiRoomHMMEnsemble
)
from .change_point_detector import BehavioralChangeDetector, ChangePoint
from ..data_cleaning.event_deduplicator import EventDeduplicator, DeduplicationMetrics

# Storage and configuration
from src.storage.timeseries_db import TimescaleDBManager
from src.storage.feature_store import RedisFeatureStore
from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


@dataclass
class EventPattern:
    """Represents a discovered event-based pattern"""
    pattern_id: str
    room_id: str
    pattern_type: str  # 'transition', 'duration', 'frequency', 'routine'
    confidence: float
    occurrences: int
    typical_duration_minutes: float
    state_sequence: List[str]
    time_features: Dict[str, Any]
    description: str


@dataclass
class EventSequenceAnalysis:
    """Analysis results for event sequences"""
    total_events: int
    clean_events: int
    duplicate_rate: float
    state_transitions: int
    occupancy_sessions: int
    avg_session_duration: float
    transition_matrix: Dict[str, Dict[str, float]]
    patterns_discovered: List[EventPattern]


class EventBasedPatternDiscovery:
    """
    Event-based pattern discovery system that replaces STUMPY matrix profiles.
    
    Uses HMM state modeling and event sequence analysis instead of time-series
    pattern matching. Designed for sparse occupancy data with focus on actual
    occupancy events rather than uniform time sampling.
    
    Maintains same interface as original PatternDiscovery class for compatibility.
    """
    
    def __init__(self):
        """Initialize the event-based pattern discovery system"""
        self.pattern_library = {}
        self.memory_limit_mb = 4000  # 4GB safety limit
        
        # Initialize core components
        self.event_deduplicator = None
        self.behavioral_change_detector = None
        self.hmm_ensemble = None
        
        # Event processing configuration
        self.event_config = {
            'deduplication_window_seconds': 5,
            'min_state_duration_seconds': 1,
            'min_session_length': 2,  # Minimum events per session
            'change_detection_sensitivity': 0.7,
            'hmm_states': 4,  # away, arriving, occupied, leaving
            'min_training_sequences': 20
        }
        
        # Room definitions (same as original PatternDiscovery)
        self.FULL_ROOM_SENSORS = {
            'office': ['binary_sensor.office_presence_full_office'],
            'bedroom': ['binary_sensor.bedroom_presence_sensor_full_bedroom'],
            'living_kitchen': [
                'binary_sensor.presence_livingroom_full',
                'binary_sensor.kitchen_pressence_full_kitchen'
            ],
            'bathroom': [
                'binary_sensor.bathroom_door_sensor_contact',
                'binary_sensor.bathroom_entrance'
            ],
            'small_bathroom': [
                'binary_sensor.small_bathroom_door_sensor_contact',
                'binary_sensor.presence_small_bathroom_entrance'
            ]
        }
        
        # Performance tracking
        self.performance_metrics = {}
        self.processing_stats = {}
        
        logger.info("Event-based pattern discovery system initialized")
    
    async def initialize_components(self):
        """Initialize all sub-components with proper configuration"""
        try:
            # Load configuration
            config = ConfigLoader()
            
            # Initialize database connections
            db_config = config.get("database.timescale")
            db_connection_string = (
                f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            
            redis_config = config.get("redis")
            redis_url = f"redis://{redis_config['host']}:{redis_config['port']}/{redis_config.get('db', 0)}"
            
            # Initialize event deduplicator
            self.event_deduplicator = EventDeduplicator(
                connection_string=db_connection_string,
                max_time_window_seconds=self.event_config['deduplication_window_seconds'],
                min_state_duration_seconds=self.event_config['min_state_duration_seconds']
            )
            await self.event_deduplicator.initialize()
            
            # Initialize TimescaleDB manager for change detector
            db_manager = TimescaleDBManager(db_connection_string)
            await db_manager.initialize()
            
            # Initialize Redis feature store
            feature_store = RedisFeatureStore(redis_url)
            await feature_store.initialize()
            
            # Initialize behavioral change detector
            change_detector_config = {
                'penalty': 10,
                'min_segment_size': 24,
                'window_size': 168,
                'max_changes': 10,
                'retraining_threshold': self.event_config['change_detection_sensitivity']
            }
            
            self.behavioral_change_detector = BehavioralChangeDetector(change_detector_config)
            await self.behavioral_change_detector.initialize(db_manager, feature_store)
            
            # Initialize multi-room HMM ensemble
            room_ids = list(self.FULL_ROOM_SENSORS.keys())
            hmm_config = {
                room_id: {
                    'n_states': self.event_config['hmm_states'],
                    'min_observations': self.event_config['min_training_sequences']
                }
                for room_id in room_ids
            }
            
            self.hmm_ensemble = MultiRoomHMMEnsemble(room_ids, hmm_config)
            
            # Register retraining callback
            self.behavioral_change_detector.register_retraining_callback(
                self._handle_behavioral_change
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def discover_multizone_patterns(self, room_name: str, historical_events: List[Dict] = None) -> Dict:
        """
        Main pattern discovery method - drop-in replacement for original PatternDiscovery.
        
        Uses event-based HMM analysis instead of STUMPY matrix profiles.
        Maintains same interface for compatibility with adaptive_predictor.py.
        """
        logger.info(f"🔍 Starting event-based pattern discovery for {room_name}")
        
        # Initialize components if not already done
        if self.event_deduplicator is None:
            await self.initialize_components()
        
        # Load events if not provided
        if historical_events is None:
            historical_events = await self._load_events_with_monitoring(room_name)
        
        if not historical_events:
            logger.warning(f"No events found for {room_name}")
            return {'room': room_name, 'patterns': {}, 'total_events': 0}
        
        logger.info(f"📊 Analyzing {len(historical_events):,} events for {room_name}")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Clean and deduplicate events
            logger.info("🧹 Cleaning and deduplicating event data...")
            clean_events = await self._clean_event_sequence(historical_events, room_name)
            
            # Step 2: Analyze event sequences
            logger.info("🔄 Analyzing event sequences...")
            sequence_analysis = await self._analyze_event_sequences(clean_events, room_name)
            
            # Step 3: Train HMM models on clean sequences
            logger.info("🤖 Training HMM models...")
            hmm_patterns = await self._discover_hmm_patterns(clean_events, room_name)
            
            # Step 4: Detect behavioral change points
            logger.info("📈 Detecting behavioral change points...")
            change_points = await self._detect_behavioral_changes(clean_events, room_name)
            
            # Step 5: Extract event-based patterns
            logger.info("🎯 Extracting event-based patterns...")
            event_patterns = await self._extract_event_patterns(sequence_analysis, hmm_patterns)
            
            # Combine all discoveries
            patterns = {
                'event_sequences': sequence_analysis,
                'hmm_states': hmm_patterns,
                'change_points': change_points,
                'discovered_patterns': event_patterns,
                'metadata': {
                    'total_events': len(historical_events),
                    'clean_events': len(clean_events),
                    'duplicate_rate': sequence_analysis.duplicate_rate,
                    'processing_time_seconds': (datetime.now() - start_time).total_seconds(),
                    'discovery_method': 'event_based_hmm'
                }
            }
            
            # Store patterns in library and Redis
            self.pattern_library[room_name] = patterns
            await self.store_patterns_in_redis(room_name, patterns)
            
            # Track performance
            self.performance_metrics[room_name] = {
                'events_processed': len(historical_events),
                'patterns_discovered': len(event_patterns.patterns_discovered),
                'hmm_accuracy': hmm_patterns.get('validation_accuracy', 0.0),
                'change_points_detected': len(change_points),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
            logger.info(f"✅ Event-based pattern discovery complete for {room_name}")
            logger.info(f"   Discovered {len(event_patterns.patterns_discovered)} patterns")
            logger.info(f"   HMM validation accuracy: {hmm_patterns.get('validation_accuracy', 0.0):.3f}")
            logger.info(f"   Change points detected: {len(change_points)}")
            
            return {
                'room': room_name,
                'patterns': patterns,
                'total_events': len(historical_events),
                'summary': self._generate_pattern_summary(patterns),
                'performance_metrics': self.performance_metrics[room_name]
            }
            
        except Exception as e:
            logger.error(f"Error in event-based pattern discovery for {room_name}: {e}")
            return {
                'room': room_name,
                'patterns': {'error': str(e)},
                'total_events': len(historical_events),
                'summary': {'has_patterns': False, 'error': str(e)}
            }
    
    async def _clean_event_sequence(self, events: List[Dict], room_name: str) -> List[Dict]:
        """Clean and deduplicate event sequence for better HMM training"""
        try:
            logger.info(f"🧹 Cleaning {len(events):,} events for {room_name}")
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(events)
            if df.empty:
                return []
            
            # Ensure proper timestamp parsing
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Add occupancy flag
            df['occupied'] = df['state'].isin(['on', 'occupied', '1', 1]).astype(int)
            
            # Filter to only state changes (remove consecutive duplicates)
            df['state_changed'] = df['occupied'] != df['occupied'].shift(1)
            state_changes = df[df['state_changed'] | (df.index == 0)].copy()
            
            # Apply minimum duration filter
            min_duration = timedelta(seconds=self.event_config['min_state_duration_seconds'])
            filtered_events = []
            
            for i, row in state_changes.iterrows():
                if i == len(state_changes) - 1:  # Last event
                    filtered_events.append(row.to_dict())
                else:
                    next_row = state_changes.iloc[state_changes.index.get_loc(i) + 1]
                    duration = next_row['timestamp'] - row['timestamp']
                    
                    if duration >= min_duration:
                        filtered_events.append(row.to_dict())
            
            logger.info(f"   Filtered from {len(df):,} to {len(filtered_events):,} events")
            logger.info(f"   Duplicate reduction: {((len(df) - len(filtered_events)) / len(df) * 100):.1f}%")
            
            return filtered_events
            
        except Exception as e:
            logger.error(f"Error cleaning event sequence for {room_name}: {e}")
            return events  # Return original events if cleanup fails
    
    async def _analyze_event_sequences(self, events: List[Dict], room_name: str) -> EventSequenceAnalysis:
        """Analyze event sequences to extract statistical patterns"""
        try:
            if not events:
                return EventSequenceAnalysis(
                    total_events=0,
                    clean_events=0,
                    duplicate_rate=0.0,
                    state_transitions=0,
                    occupancy_sessions=0,
                    avg_session_duration=0.0,
                    transition_matrix={},
                    patterns_discovered=[]
                )
            
            df = pd.DataFrame(events)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Count state transitions
            state_transitions = len(df)
            
            # Identify occupancy sessions
            sessions = []
            current_session = None
            
            for _, row in df.iterrows():
                if row['occupied'] and current_session is None:
                    # Start new session
                    current_session = {'start': row['timestamp'], 'events': [row]}
                elif row['occupied'] and current_session is not None:
                    # Continue session
                    current_session['events'].append(row)
                elif not row['occupied'] and current_session is not None:
                    # End session
                    current_session['end'] = row['timestamp']
                    current_session['duration'] = (
                        current_session['end'] - current_session['start']
                    ).total_seconds() / 60.0  # Minutes
                    
                    if len(current_session['events']) >= self.event_config['min_session_length']:
                        sessions.append(current_session)
                    current_session = None
            
            # Handle last session if still open
            if current_session is not None:
                current_session['end'] = df.iloc[-1]['timestamp']
                current_session['duration'] = (
                    current_session['end'] - current_session['start']
                ).total_seconds() / 60.0
                
                if len(current_session['events']) >= self.event_config['min_session_length']:
                    sessions.append(current_session)
            
            # Calculate transition matrix
            transition_counts = defaultdict(lambda: defaultdict(int))
            for i in range(1, len(df)):
                prev_state = 'occupied' if df.iloc[i-1]['occupied'] else 'empty'
                curr_state = 'occupied' if df.iloc[i]['occupied'] else 'empty'
                transition_counts[prev_state][curr_state] += 1
            
            # Normalize to probabilities
            transition_matrix = {}
            for from_state, to_states in transition_counts.items():
                total = sum(to_states.values())
                if total > 0:
                    transition_matrix[from_state] = {
                        to_state: count / total
                        for to_state, count in to_states.items()
                    }
            
            # Calculate averages
            avg_session_duration = np.mean([s['duration'] for s in sessions]) if sessions else 0.0
            
            return EventSequenceAnalysis(
                total_events=len(events),
                clean_events=len(df),
                duplicate_rate=0.0,  # Already cleaned
                state_transitions=state_transitions,
                occupancy_sessions=len(sessions),
                avg_session_duration=avg_session_duration,
                transition_matrix=transition_matrix,
                patterns_discovered=[]  # Will be filled by other methods
            )
            
        except Exception as e:
            logger.error(f"Error analyzing event sequences for {room_name}: {e}")
            return EventSequenceAnalysis(
                total_events=len(events),
                clean_events=0,
                duplicate_rate=0.0,
                state_transitions=0,
                occupancy_sessions=0,
                avg_session_duration=0.0,
                transition_matrix={},
                patterns_discovered=[]
            )
    
    async def _discover_hmm_patterns(self, events: List[Dict], room_name: str) -> Dict[str, Any]:
        """Use HMM to discover state transition patterns"""
        try:
            if not events:
                return {'error': 'No events for HMM training'}
            
            # Get room-specific HMM predictor
            if room_name not in self.hmm_ensemble.room_predictors:
                logger.warning(f"No HMM predictor for room {room_name}")
                return {'error': f'No HMM predictor configured for {room_name}'}
            
            hmm_predictor = self.hmm_ensemble.room_predictors[room_name]
            
            # Convert events to HMM training sequences
            training_sequences = await self._events_to_hmm_sequences(events, room_name)
            
            logger.info(f"   Created {len(training_sequences)} training sequences for HMM")
            
            # Train HMM with sequences
            for sequence in training_sequences:
                await hmm_predictor.add_training_sequence(sequence)
            
            if len(training_sequences) >= self.event_config['min_training_sequences']:
                await hmm_predictor.train()
                
                # Get performance metrics
                performance = await hmm_predictor.get_model_performance()
                
                logger.info(f"   HMM training complete: {performance.get('validation_accuracy', 0.0):.3f} accuracy")
                
                return {
                    'model_performance': performance,
                    'training_sequences': len(training_sequences),
                    'states': {
                        name: {
                            'description': state.description,
                            'typical_duration': state.typical_duration_minutes
                        }
                        for name, state in hmm_predictor.states.items()
                    },
                    'validation_accuracy': performance.get('validation_accuracy', 0.0)
                }
            else:
                logger.warning(f"   Insufficient sequences for HMM training: {len(training_sequences)}")
                return {
                    'error': 'Insufficient training data',
                    'training_sequences': len(training_sequences),
                    'required': self.event_config['min_training_sequences']
                }
            
        except Exception as e:
            logger.error(f"Error in HMM pattern discovery for {room_name}: {e}")
            return {'error': str(e)}
    
    async def _events_to_hmm_sequences(self, events: List[Dict], room_name: str) -> List[OccupancySequence]:
        """Convert event list to HMM training sequences"""
        try:
            if not events:
                return []
            
            # Group events into sessions (periods of activity)
            df = pd.DataFrame(events)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            sequences = []
            current_sequence = None
            max_gap_minutes = 60  # Max gap between events in same sequence
            
            for _, row in df.iterrows():
                timestamp = row['timestamp']
                occupied = bool(row['occupied'])
                
                # Create features for this event
                features = {
                    'hour': timestamp.hour,
                    'day_of_week': timestamp.dayofweek,
                    'recent_room_activity': 1.0 if occupied else 0.0,
                    'sensor_activations_1h': 1,
                    'time_since_last_activity': 0.0,
                    'movement_pattern_score': 0.5,
                    'occupancy_momentum': 1.0 if occupied else 0.0
                }
                
                # Check if this continues current sequence or starts new one
                if current_sequence is None:
                    # Start new sequence
                    current_sequence = {
                        'timestamps': [timestamp],
                        'occupancy': [occupied],
                        'features': [features]
                    }
                else:
                    # Check time gap
                    last_timestamp = current_sequence['timestamps'][-1]
                    gap_minutes = (timestamp - last_timestamp).total_seconds() / 60.0
                    
                    if gap_minutes <= max_gap_minutes:
                        # Continue current sequence
                        current_sequence['timestamps'].append(timestamp)
                        current_sequence['occupancy'].append(occupied)
                        current_sequence['features'].append(features)
                    else:
                        # Save current sequence and start new one
                        if len(current_sequence['timestamps']) >= 2:
                            sequences.append(OccupancySequence(
                                timestamps=current_sequence['timestamps'],
                                occupancy=current_sequence['occupancy'],
                                features=current_sequence['features'],
                                room_id=room_name
                            ))
                        
                        current_sequence = {
                            'timestamps': [timestamp],
                            'occupancy': [occupied],
                            'features': [features]
                        }
            
            # Save final sequence
            if current_sequence and len(current_sequence['timestamps']) >= 2:
                sequences.append(OccupancySequence(
                    timestamps=current_sequence['timestamps'],
                    occupancy=current_sequence['occupancy'],
                    features=current_sequence['features'],
                    room_id=room_name
                ))
            
            logger.debug(f"Created {len(sequences)} HMM sequences from {len(events)} events")
            return sequences
            
        except Exception as e:
            logger.error(f"Error converting events to HMM sequences: {e}")
            return []
    
    async def _detect_behavioral_changes(self, events: List[Dict], room_name: str) -> List[Dict]:
        """Detect behavioral change points in event data"""
        try:
            if len(events) < 100:  # Need minimum data for change detection
                logger.warning(f"Insufficient data for change detection: {len(events)} events")
                return []
            
            # Get date range for analysis
            df = pd.DataFrame(events)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            start_date = df['timestamp'].min()
            end_date = df['timestamp'].max()
            
            logger.info(f"   Detecting changes from {start_date} to {end_date}")
            
            # Use behavioral change detector
            change_points = await self.behavioral_change_detector.detect_historical_changes(
                room_id=room_name,
                start_date=start_date,
                end_date=end_date,
                change_types=['frequency', 'duration', 'timing']
            )
            
            # Convert ChangePoint objects to dictionaries
            change_dicts = []
            for cp in change_points:
                change_dict = {
                    'timestamp': cp.timestamp.isoformat(),
                    'room_id': cp.room_id,
                    'change_type': cp.change_type,
                    'confidence': cp.confidence,
                    'algorithm': cp.algorithm,
                    'description': f"{cp.change_type.title()} change detected with {cp.confidence:.2f} confidence",
                    'metadata': cp.metadata
                }
                change_dicts.append(change_dict)
            
            logger.info(f"   Detected {len(change_points)} behavioral change points")
            return change_dicts
            
        except Exception as e:
            logger.error(f"Error detecting behavioral changes for {room_name}: {e}")
            return []
    
    async def _extract_event_patterns(self, sequence_analysis: EventSequenceAnalysis, 
                                    hmm_patterns: Dict[str, Any]) -> EventSequenceAnalysis:
        """Extract meaningful patterns from event sequence analysis"""
        try:
            patterns = []
            
            # Pattern 1: State transition patterns
            if sequence_analysis.transition_matrix:
                for from_state, transitions in sequence_analysis.transition_matrix.items():
                    for to_state, probability in transitions.items():
                        if probability > 0.1:  # Significant transition
                            pattern = EventPattern(
                                pattern_id=f"transition_{from_state}_to_{to_state}",
                                room_id='',  # Will be set by caller
                                pattern_type='transition',
                                confidence=probability,
                                occurrences=int(probability * sequence_analysis.state_transitions),
                                typical_duration_minutes=0.0,  # Instantaneous
                                state_sequence=[from_state, to_state],
                                time_features={},
                                description=f"Transition from {from_state} to {to_state} "
                                          f"({probability:.2f} probability)"
                            )
                            patterns.append(pattern)
            
            # Pattern 2: Session duration patterns
            if sequence_analysis.avg_session_duration > 0:
                pattern = EventPattern(
                    pattern_id="session_duration",
                    room_id='',
                    pattern_type='duration',
                    confidence=0.8,  # High confidence for statistical measure
                    occurrences=sequence_analysis.occupancy_sessions,
                    typical_duration_minutes=sequence_analysis.avg_session_duration,
                    state_sequence=['occupied'],
                    time_features={'avg_duration': sequence_analysis.avg_session_duration},
                    description=f"Average occupancy session: "
                              f"{sequence_analysis.avg_session_duration:.1f} minutes"
                )
                patterns.append(pattern)
            
            # Pattern 3: HMM state patterns
            if hmm_patterns.get('states'):
                for state_name, state_info in hmm_patterns['states'].items():
                    pattern = EventPattern(
                        pattern_id=f"hmm_state_{state_name}",
                        room_id='',
                        pattern_type='routine',
                        confidence=hmm_patterns.get('validation_accuracy', 0.5),
                        occurrences=hmm_patterns.get('training_sequences', 0),
                        typical_duration_minutes=state_info.get('typical_duration', 0.0),
                        state_sequence=[state_name],
                        time_features=state_info,
                        description=f"HMM state: {state_info.get('description', state_name)}"
                    )
                    patterns.append(pattern)
            
            # Update sequence analysis with discovered patterns
            sequence_analysis.patterns_discovered = patterns
            
            logger.info(f"   Extracted {len(patterns)} event-based patterns")
            return sequence_analysis
            
        except Exception as e:
            logger.error(f"Error extracting event patterns: {e}")
            sequence_analysis.patterns_discovered = []
            return sequence_analysis
    
    async def _handle_behavioral_change(self, retraining_info: Dict[str, Any]):
        """Handle behavioral change detection by triggering model updates"""
        try:
            room_id = retraining_info['room_id']
            change_point = retraining_info['change_point']
            
            logger.info(f"🔄 Behavioral change detected for {room_id}: {change_point['change_type']}")
            logger.info(f"   Confidence: {change_point['confidence']:.3f}")
            logger.info(f"   Triggering pattern rediscovery...")
            
            # Trigger pattern rediscovery for this room
            await self.discover_multizone_patterns(room_id)
            
            logger.info(f"✅ Pattern rediscovery completed for {room_id}")
            
        except Exception as e:
            logger.error(f"Error handling behavioral change: {e}")
    
    async def _load_events_with_monitoring(self, room_name: str) -> List[Dict]:
        """Load events with memory monitoring - same interface as original"""
        try:
            from src.storage.timeseries_db import TimescaleDBManager
            from config.config_loader import ConfigLoader
            
            config = ConfigLoader()
            db_config = config.get("database.timescale")
            db = TimescaleDBManager(
                f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@"
                f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            
            await db.initialize()
            
            try:
                logger.info(f"Loading events for {room_name}...")
                
                # Get sensors for this room
                room_sensors = self.FULL_ROOM_SENSORS.get(room_name, [])
                
                if not room_sensors:
                    logger.warning(f"No sensors defined for {room_name}")
                    return []
                
                async with db.engine.begin() as conn:
                    from sqlalchemy import text
                    
                    # Build sensor filter
                    sensor_conditions = []
                    for sensor in room_sensors:
                        sensor_conditions.append(f"entity_id = '{sensor}'")
                    
                    sensor_filter = " OR ".join(sensor_conditions)
                    
                    query = f"""
                        SELECT timestamp, entity_id, state, attributes, room, sensor_type
                        FROM sensor_events 
                        WHERE ({sensor_filter})
                        AND timestamp >= NOW() - INTERVAL '180 days'
                        ORDER BY timestamp DESC
                        LIMIT 100000
                    """
                    
                    result = await conn.execute(text(query))
                    events = [dict(row._mapping) for row in result.fetchall()]
                    
                    logger.info(f"Loaded {len(events):,} events for {room_name}")
                    return events
                    
            finally:
                await db.close()
                
        except Exception as e:
            logger.error(f"Error loading events for {room_name}: {e}")
            return []
    
    async def store_patterns_in_redis(self, room_id: str, patterns: Dict):
        """Store patterns in Redis - same interface as original"""
        try:
            from src.storage.feature_store import RedisFeatureStore
            from config.config_loader import ConfigLoader
            
            config = ConfigLoader()
            redis_config = config.get("redis")
            redis_url = f"redis://{redis_config['host']}:{redis_config['port']}/{redis_config.get('db', 0)}"
            
            feature_store = RedisFeatureStore(redis_url)
            await feature_store.initialize()
            
            # Prepare pattern data for storage
            pattern_data = {
                'room_id': room_id,
                'discovery_time': datetime.now(timezone.utc).isoformat(),
                'patterns': patterns,
                'summary': self._generate_pattern_summary(patterns),
                'discovery_method': 'event_based_hmm'
            }
            
            # Store in Redis
            await feature_store.cache_pattern(room_id, 'discovered_patterns', pattern_data)
            
            logger.info(f"✅ Stored event-based patterns for {room_id} in Redis")
            
            await feature_store.close()
            
        except Exception as e:
            logger.error(f"Error storing patterns in Redis for {room_id}: {e}")
    
    def _generate_pattern_summary(self, patterns: Dict) -> Dict:
        """Generate pattern summary - same interface as original"""
        summary = {
            'has_patterns': False,
            'pattern_types': [],
            'key_insights': [],
            'discovery_method': 'event_based_hmm'
        }
        
        try:
            # Check event sequence patterns
            if patterns.get('event_sequences'):
                seq_analysis = patterns['event_sequences']
                
                if isinstance(seq_analysis, dict):
                    sessions = seq_analysis.get('occupancy_sessions', 0)
                    avg_duration = seq_analysis.get('avg_session_duration', 0)
                else:
                    sessions = seq_analysis.occupancy_sessions
                    avg_duration = seq_analysis.avg_session_duration
                
                if sessions > 0:
                    summary['has_patterns'] = True
                    summary['pattern_types'].append('event_sequences')
                    summary['key_insights'].append(
                        f"Found {sessions} occupancy sessions "
                        f"(avg: {avg_duration:.1f} minutes)"
                    )
            
            # Check HMM patterns
            if patterns.get('hmm_states') and not patterns['hmm_states'].get('error'):
                hmm_data = patterns['hmm_states']
                accuracy = hmm_data.get('validation_accuracy', 0)
                sequences = hmm_data.get('training_sequences', 0)
                
                if accuracy > 0.5:
                    summary['has_patterns'] = True
                    summary['pattern_types'].append('hmm_states')
                    summary['key_insights'].append(
                        f"HMM model trained with {accuracy:.3f} accuracy "
                        f"({sequences} sequences)"
                    )
            
            # Check change points
            if patterns.get('change_points'):
                change_count = len(patterns['change_points'])
                if change_count > 0:
                    summary['pattern_types'].append('behavioral_changes')
                    summary['key_insights'].append(
                        f"Detected {change_count} behavioral change points"
                    )
            
            # Check discovered patterns
            if patterns.get('discovered_patterns'):
                if isinstance(patterns['discovered_patterns'], dict):
                    pattern_list = patterns['discovered_patterns'].get('patterns_discovered', [])
                else:
                    pattern_list = patterns['discovered_patterns'].patterns_discovered
                
                if pattern_list:
                    summary['has_patterns'] = True
                    summary['pattern_types'].append('event_patterns')
                    summary['key_insights'].append(
                        f"Discovered {len(pattern_list)} distinct event patterns"
                    )
            
            # Overall assessment
            if not summary['has_patterns']:
                summary['key_insights'].append("No significant patterns detected in event data")
            
        except Exception as e:
            logger.error(f"Error generating pattern summary: {e}")
            summary['key_insights'].append(f"Error in summary generation: {str(e)}")
        
        return summary
    
    async def discover_bathroom_patterns(self, bathroom_rooms: List[str]) -> Dict:
        """Discover bathroom patterns - maintains original interface"""
        logger.info(f"Discovering bathroom patterns for {bathroom_rooms}")
        
        bathroom_patterns = {}
        
        for bathroom_room in bathroom_rooms:
            patterns = await self.discover_multizone_patterns(bathroom_room)
            bathroom_patterns[bathroom_room] = patterns
        
        logger.info(f"Bathroom pattern discovery completed for {len(bathroom_patterns)} bathrooms")
        return {'patterns': bathroom_patterns}
    
    async def discover_patterns_from_events(self, room_name: str, events: List[Dict]) -> Dict:
        """
        Discover patterns directly from provided events.
        
        New method specifically for clean event sequences without database loading.
        Useful for testing and when events are already available.
        """
        logger.info(f"🎯 Discovering patterns from {len(events)} provided events for {room_name}")
        
        return await self.discover_multizone_patterns(room_name, events)
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the event-based system"""
        return {
            'rooms_analyzed': list(self.performance_metrics.keys()),
            'total_rooms': len(self.performance_metrics),
            'average_accuracy': np.mean([
                metrics.get('hmm_accuracy', 0.0) 
                for metrics in self.performance_metrics.values()
            ]) if self.performance_metrics else 0.0,
            'total_patterns_discovered': sum([
                metrics.get('patterns_discovered', 0)
                for metrics in self.performance_metrics.values()
            ]),
            'total_events_processed': sum([
                metrics.get('events_processed', 0)
                for metrics in self.performance_metrics.values()
            ]),
            'processing_stats': self.processing_stats,
            'system_type': 'event_based_hmm'
        }
    
    async def close(self):
        """Clean up resources"""
        try:
            if self.event_deduplicator:
                await self.event_deduplicator.close()
            
            # Close other components as needed
            logger.info("Event-based pattern discovery system closed")
            
        except Exception as e:
            logger.error(f"Error closing event-based pattern discovery: {e}")


# Backward compatibility functions
async def discover_bathroom_patterns(room_names: List[str]) -> Dict:
    """Backward compatibility function for bathroom pattern discovery"""
    discoverer = EventBasedPatternDiscovery()
    return await discoverer.discover_bathroom_patterns(room_names)


async def discover_transition_patterns(area_name: str) -> Dict:
    """Backward compatibility function for transition pattern discovery"""
    discoverer = EventBasedPatternDiscovery()
    return await discoverer.discover_multizone_patterns(area_name)