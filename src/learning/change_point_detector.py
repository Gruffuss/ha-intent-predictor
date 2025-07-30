"""
Change Point Detection for Behavioral Pattern Analysis
Implements ruptures-based change point detection to enhance HMM system adaptability

Uses multiple algorithms to detect changes in:
- Occupancy frequency patterns
- Duration distributions  
- Timing behaviors
- Routine shifts over time

Integrates with TimescaleDB for historical analysis and triggers model retraining
when significant behavioral changes are detected.
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
import json

# Ruptures library for change point detection
import ruptures as rpt

# Statistical analysis
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Internal imports
from src.storage.timeseries_db import TimescaleDBManager
from src.storage.feature_store import RedisFeatureStore

logger = logging.getLogger(__name__)


@dataclass
class ChangePoint:
    """Represents a detected change point in occupancy patterns"""
    timestamp: datetime
    room_id: str
    change_type: str  # 'frequency', 'duration', 'timing', 'routine'
    confidence: float
    algorithm: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'room_id': self.room_id,
            'change_type': self.change_type,
            'confidence': self.confidence,
            'algorithm': self.algorithm,
            'metadata': self.metadata
        }


@dataclass 
class PatternSegment:
    """Represents a behavioral pattern segment between change points"""
    start_time: datetime
    end_time: datetime
    room_id: str
    pattern_id: str
    characteristics: Dict[str, Any]
    stability_score: float


class BehavioralChangeDetector:
    """
    Ruptures-based change point detection for occupancy behavior analysis
    
    Implements multiple algorithms:
    - PELT (Pruned Exact Linear Time) for online detection
    - Window-based methods for real-time monitoring
    - DYNP (Dynamic Programming) for offline analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_manager = None
        self.feature_store = None
        
        # Algorithm configurations
        self.algorithms = {
            'pelt': {
                'model': 'rbf',  # Gaussian kernel for non-linear patterns
                'min_size': config.get('min_segment_size', 24),  # 24 hours minimum
                'jump': 1,
                'pen': config.get('penalty', 10)
            },
            'window': {
                'width': config.get('window_size', 168),  # 1 week window
                'model': 'l2',
                'min_size': 12  # 12 hours minimum
            },
            'dynp': {
                'model': 'rbf',
                'n_bkps': config.get('max_changes', 10),
                'min_size': 24
            }
        }
        
        # Feature extractors for different change types
        self.feature_extractors = {
            'frequency': self._extract_frequency_features,
            'duration': self._extract_duration_features,
            'timing': self._extract_timing_features,
            'routine': self._extract_routine_features
        }
        
        # Change point history for each room
        self.change_history: Dict[str, List[ChangePoint]] = defaultdict(list)
        
        # Pattern segments
        self.pattern_segments: Dict[str, List[PatternSegment]] = defaultdict(list)
        
        # Real-time monitoring buffers
        self.monitoring_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.get('buffer_size', 1000))
        )
        
        # Retraining triggers
        self.retraining_callbacks = []
        
        logger.info("Behavioral change detector initialized")
    
    async def initialize(self, db_manager: TimescaleDBManager, feature_store: RedisFeatureStore):
        """Initialize with storage backends"""
        self.db_manager = db_manager
        self.feature_store = feature_store
        
        # Load existing change point history
        await self._load_change_history()
        
        logger.info("Change point detector connected to storage backends")
    
    def register_retraining_callback(self, callback):
        """Register callback for triggering model retraining"""
        self.retraining_callbacks.append(callback)
    
    async def detect_historical_changes(
        self, 
        room_id: str, 
        start_date: datetime, 
        end_date: datetime,
        change_types: List[str] = None
    ) -> List[ChangePoint]:
        """
        Detect change points in historical occupancy data
        
        Args:
            room_id: Room to analyze
            start_date: Analysis start date
            end_date: Analysis end date
            change_types: Types of changes to detect ['frequency', 'duration', 'timing', 'routine']
        
        Returns:
            List of detected change points
        """
        if change_types is None:
            change_types = ['frequency', 'duration', 'timing', 'routine']
        
        logger.info(f"Analyzing historical changes for {room_id} from {start_date} to {end_date}")
        
        # Fetch historical occupancy data
        occupancy_data = await self._fetch_occupancy_data(room_id, start_date, end_date)
        
        if len(occupancy_data) < self.algorithms['pelt']['min_size']:
            logger.warning(f"Insufficient data for change point detection: {len(occupancy_data)} points")
            return []
        
        detected_changes = []
        
        for change_type in change_types:
            try:
                # Extract features for this change type
                features = await self.feature_extractors[change_type](occupancy_data, room_id)
                
                if len(features) < self.algorithms['pelt']['min_size']:
                    logger.warning(f"Insufficient features for {change_type} analysis")
                    continue
                
                # Detect changes using multiple algorithms
                pelt_changes = self._detect_with_pelt(features, change_type, room_id)
                dynp_changes = self._detect_with_dynp(features, change_type, room_id)
                
                # Combine and validate changes
                validated_changes = self._validate_and_merge_changes(
                    pelt_changes + dynp_changes, occupancy_data
                )
                
                detected_changes.extend(validated_changes)
                
            except Exception as e:
                logger.error(f"Error detecting {change_type} changes for {room_id}: {e}")
        
        # Sort by timestamp and remove duplicates
        detected_changes = self._deduplicate_changes(detected_changes)
        
        # Store changes in database
        await self._store_change_points(detected_changes)
        
        # Update change history
        self.change_history[room_id].extend(detected_changes)
        
        logger.info(f"Detected {len(detected_changes)} change points for {room_id}")
        return detected_changes
    
    async def monitor_real_time_changes(self, room_id: str, occupancy_event: Dict[str, Any]):
        """
        Monitor for change points in real-time streaming data
        
        Args:
            room_id: Room being monitored
            occupancy_event: New occupancy event
        """
        # Add event to monitoring buffer
        self.monitoring_buffers[room_id].append(occupancy_event)
        
        buffer = self.monitoring_buffers[room_id]
        
        # Check if we have enough data for analysis
        min_size = self.algorithms['window']['width']
        if len(buffer) < min_size:
            return
        
        # Extract recent features
        recent_data = list(buffer)
        
        try:
            # Analyze for frequency changes (most responsive)
            frequency_features = await self._extract_frequency_features(recent_data, room_id)
            
            if len(frequency_features) >= self.algorithms['window']['min_size']:
                # Use window-based detection for real-time monitoring
                changes = self._detect_with_window(frequency_features, 'frequency', room_id)
                
                if changes:
                    logger.info(f"Real-time change detected for {room_id}: {len(changes)} points")
                    
                    # Validate and store
                    validated_changes = self._validate_and_merge_changes(changes, recent_data)
                    await self._store_change_points(validated_changes)
                    
                    # Trigger retraining if significant change
                    for change in validated_changes:
                        if change.confidence > self.config.get('retraining_threshold', 0.7):
                            await self._trigger_retraining(room_id, change)
        
        except Exception as e:
            logger.error(f"Error in real-time change monitoring for {room_id}: {e}")
    
    def _detect_with_pelt(self, features: np.ndarray, change_type: str, room_id: str) -> List[ChangePoint]:
        """Detect change points using PELT algorithm"""
        try:
            # Configure PELT detector
            config = self.algorithms['pelt']
            detector = rpt.Pelt(
                model=config['model'],
                min_size=config['min_size'],
                jump=config['jump']
            ).fit(features)
            
            # Detect change points
            change_indices = detector.predict(pen=config['pen'])
            
            # Convert indices to ChangePoint objects
            changes = []
            for idx in change_indices[:-1]:  # Last index is end of series
                if 0 < idx < len(features):
                    # Calculate confidence based on cost reduction
                    confidence = self._calculate_confidence(features, idx, 'pelt')
                    
                    change = ChangePoint(
                        timestamp=self._index_to_timestamp(idx, features),
                        room_id=room_id,
                        change_type=change_type,
                        confidence=confidence,
                        algorithm='pelt',
                        metadata={
                            'index': int(idx),
                            'penalty': config['pen'],
                            'model': config['model']
                        }
                    )
                    changes.append(change)
            
            return changes
            
        except Exception as e:
            logger.error(f"PELT detection failed for {room_id}/{change_type}: {e}")
            return []
    
    def _detect_with_dynp(self, features: np.ndarray, change_type: str, room_id: str) -> List[ChangePoint]:
        """Detect change points using Dynamic Programming"""
        try:
            config = self.algorithms['dynp']
            detector = rpt.Dynp(
                model=config['model'],
                min_size=config['min_size']
            ).fit(features)
            
            change_indices = detector.predict(n_bkps=config['n_bkps'])
            
            changes = []
            for idx in change_indices[:-1]:
                if 0 < idx < len(features):
                    confidence = self._calculate_confidence(features, idx, 'dynp')
                    
                    change = ChangePoint(
                        timestamp=self._index_to_timestamp(idx, features),
                        room_id=room_id,
                        change_type=change_type,
                        confidence=confidence,
                        algorithm='dynp',
                        metadata={
                            'index': int(idx),
                            'n_bkps': config['n_bkps'],
                            'model': config['model']
                        }
                    )
                    changes.append(change)
            
            return changes
            
        except Exception as e:
            logger.error(f"DYNP detection failed for {room_id}/{change_type}: {e}")
            return []
    
    def _detect_with_window(self, features: np.ndarray, change_type: str, room_id: str) -> List[ChangePoint]:
        """Detect change points using sliding window approach"""
        try:
            config = self.algorithms['window']
            detector = rpt.Window(
                width=config['width'],
                model=config['model'],
                min_size=config['min_size']
            ).fit(features)
            
            # Use automatic penalty selection
            change_indices = detector.predict(pen='BIC')
            
            changes = []
            for idx in change_indices[:-1]:
                if 0 < idx < len(features):
                    confidence = self._calculate_confidence(features, idx, 'window')
                    
                    change = ChangePoint(
                        timestamp=self._index_to_timestamp(idx, features),
                        room_id=room_id,
                        change_type=change_type,
                        confidence=confidence,
                        algorithm='window',
                        metadata={
                            'index': int(idx),
                            'width': config['width'],
                            'model': config['model']
                        }
                    )
                    changes.append(change)
            
            return changes
            
        except Exception as e:
            logger.error(f"Window detection failed for {room_id}/{change_type}: {e}")
            return []
    
    async def _extract_frequency_features(self, data: List[Dict], room_id: str) -> np.ndarray:
        """Extract features for frequency change detection"""
        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            # Resample to hourly occupancy rates
            hourly_occupancy = df['occupied'].resample('1H').mean().fillna(0)
            
            # Create sliding window features
            features = []
            window_size = 24  # 24-hour windows
            
            for i in range(window_size, len(hourly_occupancy)):
                window_data = hourly_occupancy.iloc[i-window_size:i]
                
                feature_vector = [
                    window_data.mean(),  # Average occupancy rate
                    window_data.std(),   # Variability
                    window_data.sum(),   # Total occupied hours
                    (window_data > 0).sum(),  # Number of occupied hours
                    window_data.max() - window_data.min(),  # Range
                ]
                
                features.append(feature_vector)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting frequency features for {room_id}: {e}")
            return np.array([])
    
    async def _extract_duration_features(self, data: List[Dict], room_id: str) -> np.ndarray:
        """Extract features for duration change detection"""
        try:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            # Identify occupancy sessions
            sessions = []
            in_session = False
            session_start = None
            
            for timestamp, occupied in df['occupied'].items():
                if occupied and not in_session:
                    session_start = timestamp
                    in_session = True
                elif not occupied and in_session:
                    if session_start:
                        duration = (timestamp - session_start).total_seconds() / 3600  # Hours
                        sessions.append(duration)
                    in_session = False
            
            if len(sessions) < 10:  # Need minimum sessions for analysis
                return np.array([])
            
            # Create windowed duration statistics
            features = []
            window_size = max(10, len(sessions) // 20)  # Adaptive window size
            
            for i in range(window_size, len(sessions)):
                window_sessions = sessions[i-window_size:i]
                
                feature_vector = [
                    np.mean(window_sessions),
                    np.std(window_sessions),
                    np.median(window_sessions),
                    np.percentile(window_sessions, 75) - np.percentile(window_sessions, 25),  # IQR
                    len([s for s in window_sessions if s < 0.5]),  # Short sessions
                    len([s for s in window_sessions if s > 4]),    # Long sessions
                ]
                
                features.append(feature_vector)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting duration features for {room_id}: {e}")
            return np.array([])
    
    async def _extract_timing_features(self, data: List[Dict], room_id: str) -> np.ndarray:
        """Extract features for timing pattern change detection"""
        try:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            # Extract time-based features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = df['day_of_week'] >= 5
            
            # Create daily timing profiles
            daily_profiles = []
            
            for date in df.index.date:
                day_data = df[df.index.date == date]
                if len(day_data) < 12:  # Skip days with insufficient data
                    continue
                
                # Hourly occupancy profile
                hourly_profile = day_data.groupby('hour')['occupied'].mean()
                full_profile = [hourly_profile.get(h, 0) for h in range(24)]
                
                # Add day type information
                is_weekend = day_data['is_weekend'].iloc[0]
                profile_with_context = full_profile + [is_weekend]
                
                daily_profiles.append(profile_with_context)
            
            if len(daily_profiles) < 7:  # Need at least a week
                return np.array([])
            
            # Sliding window over daily profiles
            features = []
            window_size = 7  # Week-long windows
            
            for i in range(window_size, len(daily_profiles)):
                window_profiles = daily_profiles[i-window_size:i]
                
                # Statistical features across the week
                profile_array = np.array([p[:-1] for p in window_profiles])  # Exclude weekend flag
                
                feature_vector = [
                    profile_array.mean(axis=0).tolist(),  # Average daily profile
                    profile_array.std(axis=0).tolist(),   # Variability across days
                    np.mean([p[-1] for p in window_profiles])  # Weekend ratio
                ]
                
                # Flatten the feature vector
                flattened = []
                for f in feature_vector:
                    if isinstance(f, list):
                        flattened.extend(f)
                    else:
                        flattened.append(f)
                
                features.append(flattened)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting timing features for {room_id}: {e}")
            return np.array([])
    
    async def _extract_routine_features(self, data: List[Dict], room_id: str) -> np.ndarray:
        """Extract features for routine change detection"""
        try:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            # Extract routine-related features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            
            # Weekly routine patterns
            weekly_features = []
            
            # Group by week
            df['week'] = df.index.isocalendar().week
            df['year'] = df.index.year
            
            for (year, week), week_data in df.groupby(['year', 'week']):
                if len(week_data) < 50:  # Skip weeks with insufficient data
                    continue
                
                # Routine consistency metrics
                routine_features = []
                
                # Daily activity patterns
                for day in range(7):
                    day_data = week_data[week_data['day_of_week'] == day]
                    if len(day_data) > 0:
                        day_activity = day_data['occupied'].mean()
                        day_peak_hour = day_data.groupby('hour')['occupied'].mean().idxmax() if day_data['occupied'].sum() > 0 else 12
                    else:
                        day_activity = 0
                        day_peak_hour = 12
                    
                    routine_features.extend([day_activity, day_peak_hour])
                
                # Overall week characteristics
                total_activity = week_data['occupied'].mean()
                activity_variance = week_data.groupby(['day_of_week', 'hour'])['occupied'].mean().std()
                
                routine_features.extend([total_activity, activity_variance])
                
                weekly_features.append(routine_features)
            
            if len(weekly_features) < 4:  # Need at least a month
                return np.array([])
            
            return np.array(weekly_features)
            
        except Exception as e:
            logger.error(f"Error extracting routine features for {room_id}: {e}")
            return np.array([])
    
    def _calculate_confidence(self, features: np.ndarray, change_index: int, algorithm: str) -> float:
        """Calculate confidence score for a detected change point"""
        try:
            # Split data at change point
            before = features[:change_index]
            after = features[change_index:]
            
            if len(before) < 2 or len(after) < 2:
                return 0.0
            
            # Statistical test for difference
            if features.ndim == 1:
                statistic, p_value = stats.ks_2samp(before, after)
            else:
                # For multivariate data, use mean difference normalized by variance
                mean_before = np.mean(before, axis=0)
                mean_after = np.mean(after, axis=0)
                
                combined_std = np.std(features, axis=0)
                combined_std[combined_std == 0] = 1  # Avoid division by zero
                
                normalized_diff = np.abs(mean_after - mean_before) / combined_std
                statistic = np.mean(normalized_diff)
                
                # Convert to pseudo p-value (higher statistic = more significant)
                p_value = 1.0 / (1.0 + statistic)
            
            # Convert p-value to confidence (lower p-value = higher confidence)
            confidence = 1.0 - p_value
            
            # Ensure confidence is in [0, 1]
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence for {algorithm}: {e}")
            return 0.5  # Default moderate confidence
    
    def _index_to_timestamp(self, index: int, features: np.ndarray) -> datetime:
        """Convert feature index to timestamp"""
        # This is a simplified implementation
        # In practice, you'd maintain timestamp mapping with features
        base_time = datetime.now() - timedelta(hours=len(features))
        return base_time + timedelta(hours=index)
    
    def _validate_and_merge_changes(
        self, 
        changes: List[ChangePoint], 
        data: List[Dict]
    ) -> List[ChangePoint]:
        """Validate and merge similar change points"""
        if not changes:
            return []
        
        # Sort by timestamp
        changes.sort(key=lambda x: x.timestamp)
        
        # Remove low-confidence changes
        min_confidence = self.config.get('min_confidence', 0.3)
        validated = [c for c in changes if c.confidence >= min_confidence]
        
        # Merge changes that are too close together
        merge_window = timedelta(hours=self.config.get('merge_window_hours', 24))
        merged = []
        
        for change in validated:
            if not merged or (change.timestamp - merged[-1].timestamp) > merge_window:
                merged.append(change)
            elif change.confidence > merged[-1].confidence:
                # Replace with higher confidence change
                merged[-1] = change
        
        return merged
    
    def _deduplicate_changes(self, changes: List[ChangePoint]) -> List[ChangePoint]:
        """Remove duplicate change points"""
        if not changes:
            return []
        
        # Sort by timestamp and confidence
        changes.sort(key=lambda x: (x.timestamp, -x.confidence))
        
        deduplicated = []
        dedup_window = timedelta(hours=1)  # Consider changes within 1 hour as duplicates
        
        for change in changes:
            is_duplicate = False
            for existing in deduplicated:
                if (abs(change.timestamp - existing.timestamp) < dedup_window and
                    change.room_id == existing.room_id and
                    change.change_type == existing.change_type):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(change)
        
        return deduplicated
    
    async def _fetch_occupancy_data(
        self, 
        room_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict]:
        """Fetch occupancy data from TimescaleDB"""
        try:
            query = """
            SELECT 
                timestamp,
                room_id,
                CASE WHEN occupancy_probability > 0.5 THEN true ELSE false END as occupied,
                occupancy_probability
            FROM occupancy_predictions 
            WHERE room_id = %s 
                AND timestamp BETWEEN %s AND %s
                AND prediction_horizon = '15min'
            ORDER BY timestamp
            """
            
            async with self.db_manager.session_factory() as session:
                from sqlalchemy import text
                result = await session.execute(
                    text(query), 
                    [room_id, start_date, end_date]
                )
                
                rows = result.fetchall()
                return [
                    {
                        'timestamp': row[0],
                        'room_id': row[1], 
                        'occupied': row[2],
                        'probability': row[3]
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Error fetching occupancy data for {room_id}: {e}")
            return []
    
    async def _store_change_points(self, changes: List[ChangePoint]):
        """Store detected change points in database"""
        if not changes:
            return
        
        try:
            async with self.db_manager.session_factory() as session:
                from sqlalchemy import text
                
                insert_query = """
                INSERT INTO discovered_patterns (
                    room_id, pattern_type, confidence, 
                    start_time, end_time, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """
                
                for change in changes:
                    await session.execute(text(insert_query), [
                        change.room_id,
                        f"change_point_{change.change_type}",
                        change.confidence,
                        change.timestamp,
                        change.timestamp + timedelta(hours=1),  # End time estimation
                        json.dumps(change.metadata)
                    ])
                
                await session.commit()
                logger.info(f"Stored {len(changes)} change points in database")
                
        except Exception as e:
            logger.error(f"Error storing change points: {e}")
    
    async def _load_change_history(self):
        """Load existing change point history from database"""
        try:
            query = """
            SELECT room_id, pattern_type, confidence, start_time, metadata
            FROM discovered_patterns 
            WHERE pattern_type LIKE 'change_point_%'
            ORDER BY start_time DESC
            LIMIT 1000
            """
            
            async with self.db_manager.session_factory() as session:
                from sqlalchemy import text
                result = await session.execute(text(query))
                
                for row in result.fetchall():
                    room_id = row[0]
                    change_type = row[1].replace('change_point_', '')
                    confidence = row[2]
                    timestamp = row[3]
                    metadata = json.loads(row[4]) if row[4] else {}
                    
                    change = ChangePoint(
                        timestamp=timestamp,
                        room_id=room_id,
                        change_type=change_type,
                        confidence=confidence,
                        algorithm=metadata.get('algorithm', 'unknown'),
                        metadata=metadata
                    )
                    
                    self.change_history[room_id].append(change)
                
                logger.info(f"Loaded change history for {len(self.change_history)} rooms")
                
        except Exception as e:
            logger.error(f"Error loading change history: {e}")
    
    async def _trigger_retraining(self, room_id: str, change_point: ChangePoint):
        """Trigger model retraining for significant behavioral changes"""
        try:
            logger.info(f"Triggering retraining for {room_id} due to {change_point.change_type} change")
            
            retraining_info = {
                'room_id': room_id,
                'change_point': change_point.to_dict(),
                'trigger_reason': f"Behavioral change detected: {change_point.change_type}",
                'confidence': change_point.confidence
            }
            
            # Call registered callbacks
            for callback in self.retraining_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(retraining_info)
                    else:
                        callback(retraining_info)
                except Exception as e:
                    logger.error(f"Error in retraining callback: {e}")
            
            # Store retraining trigger in Redis for monitoring
            if self.feature_store:
                await self.feature_store.set(
                    f"retraining_trigger:{room_id}:{int(change_point.timestamp.timestamp())}",
                    retraining_info,
                    ttl=7 * 24 * 3600  # Keep for 7 days
                )
                
        except Exception as e:
            logger.error(f"Error triggering retraining for {room_id}: {e}")
    
    async def get_pattern_segments(self, room_id: str) -> List[PatternSegment]:
        """Get behavioral pattern segments for a room"""
        change_points = self.change_history.get(room_id, [])
        if not change_points:
            return []
        
        # Sort by timestamp
        change_points.sort(key=lambda x: x.timestamp)
        
        segments = []
        start_time = change_points[0].timestamp - timedelta(days=30)  # Start before first change
        
        for i, change_point in enumerate(change_points):
            # Create segment ending at this change point
            segment = PatternSegment(
                start_time=start_time,
                end_time=change_point.timestamp,
                room_id=room_id,
                pattern_id=f"{room_id}_segment_{i}",
                characteristics={
                    'change_types': [cp.change_type for cp in change_points[:i+1]],
                    'duration_days': (change_point.timestamp - start_time).days
                },
                stability_score=1.0 - change_point.confidence  # More stable = lower change confidence
            )
            segments.append(segment)
            start_time = change_point.timestamp
        
        return segments
    
    async def analyze_change_patterns(self, room_id: str = None) -> Dict[str, Any]:
        """Analyze change patterns across rooms or for specific room"""
        analysis = {}
        
        rooms = [room_id] if room_id else list(self.change_history.keys())
        
        for room in rooms:
            changes = self.change_history.get(room, [])
            if not changes:
                continue
            
            # Change frequency analysis
            change_types = defaultdict(int)
            algorithms_used = defaultdict(int)
            confidence_scores = []
            
            for change in changes:
                change_types[change.change_type] += 1
                algorithms_used[change.algorithm] += 1
                confidence_scores.append(change.confidence)
            
            # Time series analysis
            timestamps = [c.timestamp for c in changes]
            if len(timestamps) > 1:
                time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() / 3600 
                             for i in range(1, len(timestamps))]
                avg_interval = np.mean(time_diffs) if time_diffs else 0
            else:
                avg_interval = 0
            
            analysis[room] = {
                'total_changes': len(changes),
                'change_types': dict(change_types),
                'algorithms_used': dict(algorithms_used),
                'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
                'avg_interval_hours': avg_interval,
                'most_common_change': max(change_types.items(), key=lambda x: x[1])[0] if change_types else None,
                'latest_change': max(timestamps) if timestamps else None
            }
        
        return analysis


class ChangePointVisualizer:
    """Visualization utilities for change point analysis"""
    
    def __init__(self, detector: BehavioralChangeDetector):
        self.detector = detector
    
    async def generate_change_timeline(self, room_id: str) -> Dict[str, Any]:
        """Generate timeline data for change point visualization"""
        changes = self.detector.change_history.get(room_id, [])
        
        timeline_data = {
            'room_id': room_id,
            'changes': [
                {
                    'timestamp': change.timestamp.isoformat(),
                    'type': change.change_type,
                    'confidence': change.confidence,
                    'algorithm': change.algorithm
                }
                for change in sorted(changes, key=lambda x: x.timestamp)
            ],
            'segments': [
                {
                    'start': segment.start_time.isoformat(),
                    'end': segment.end_time.isoformat(),
                    'stability': segment.stability_score,
                    'pattern_id': segment.pattern_id
                }
                for segment in await self.detector.get_pattern_segments(room_id)
            ]
        }
        
        return timeline_data
    
    async def export_change_report(self, output_path: str):
        """Export comprehensive change point analysis report"""
        analysis = await self.detector.analyze_change_patterns()
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_rooms_analyzed': len(analysis),
                'total_changes_detected': sum(room_data['total_changes'] for room_data in analysis.values()),
                'avg_changes_per_room': np.mean([room_data['total_changes'] for room_data in analysis.values()]) if analysis else 0
            },
            'room_analysis': analysis,
            'change_type_distribution': self._aggregate_change_types(analysis),
            'algorithm_performance': self._aggregate_algorithm_performance(analysis)
        }
        
        # Save to file
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Change point analysis report exported to {output_path}")
        return report
    
    def _aggregate_change_types(self, analysis: Dict[str, Any]) -> Dict[str, int]:
        """Aggregate change types across all rooms"""
        aggregated = defaultdict(int)
        for room_data in analysis.values():
            for change_type, count in room_data.get('change_types', {}).items():
                aggregated[change_type] += count
        return dict(aggregated)
    
    def _aggregate_algorithm_performance(self, analysis: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Aggregate algorithm performance metrics"""
        algorithm_stats = defaultdict(lambda: {'total_detections': 0, 'avg_confidence': []})
        
        for room_data in analysis.values():
            for algo, count in room_data.get('algorithms_used', {}).items():
                algorithm_stats[algo]['total_detections'] += count
                # Note: This is simplified - in practice you'd track confidence per algorithm
                algorithm_stats[algo]['avg_confidence'].append(room_data.get('avg_confidence', 0))
        
        # Calculate averages
        performance = {}
        for algo, stats in algorithm_stats.items():
            performance[algo] = {
                'total_detections': stats['total_detections'],
                'avg_confidence': np.mean(stats['avg_confidence']) if stats['avg_confidence'] else 0
            }
        
        return performance