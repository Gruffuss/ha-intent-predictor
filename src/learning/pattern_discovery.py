"""
Advanced Pattern Discovery using Modern Libraries - Full Implementation
Uses STUMPY, mlxtend, pyod, ruptures for sophisticated pattern discovery
CRITICAL: Only uses FULL room sensors for meaningful behavioral patterns
"""

import logging
import psutil
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta, timezone
import json

# Pattern discovery libraries
import stumpy
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from statsmodels.tsa.seasonal import seasonal_decompose
import ruptures as rpt
from river import anomaly, preprocessing, compose

logger = logging.getLogger(__name__)


class PatternDiscovery:
    """
    Advanced pattern discovery using specialized libraries
    CRITICAL: Only uses FULL room sensors for meaningful behavioral patterns
    No assumptions, pure data-driven discovery using modern ML techniques
    """
    
    def __init__(self):
        self.pattern_library = {}
        self.memory_limit_mb = 4000  # 4GB safety limit
        
        # Define FULL room sensors only - critical for meaningful patterns
        self.FULL_ROOM_SENSORS = {
            'office': ['binary_sensor.office_presence_full_office'],
            'bedroom': ['binary_sensor.bedroom_presence_sensor_full_bedroom'],
            'living_kitchen': [
                'binary_sensor.presence_livingroom_full',
                'binary_sensor.kitchen_pressence_full_kitchen'
            ],
            'bathroom': [    # Entry/exit detection sensors
                'binary_sensor.bathroom_door_sensor_contact',
                'binary_sensor.bathroom_entrance'
            ],
            'small_bathroom': [  # Entry/exit detection sensors  
                'binary_sensor.small_bathroom_door_sensor_contact',
                'binary_sensor.presence_small_bathroom_entrance'
            ]
        }
        
    async def discover_multizone_patterns(self, room_name: str, historical_events: List[Dict] = None) -> Dict:
        """Discover patterns for multi-zone rooms using modern ML libraries - FULL SENSORS ONLY"""
        logger.info(f"🔍 Starting advanced pattern discovery for {room_name}")
        
        # Load events if not provided - CRITICAL: Only full room sensors
        if historical_events is None:
            historical_events = await self._load_events_with_monitoring(room_name)
        
        if not historical_events:
            logger.warning(f"No events found for {room_name}")
            return {'room': room_name, 'patterns': {}, 'total_events': 0}
        
        logger.info(f"📊 Analyzing {len(historical_events):,} events for {room_name}")
        
        # Run comprehensive pattern discovery
        patterns = await self._comprehensive_discovery(historical_events, room_name)
        
        # Store patterns
        self.pattern_library[room_name] = patterns
        
        # Store in Redis for model access
        await self.store_patterns_in_redis(room_name, patterns)
        
        logger.info(f"✅ Pattern discovery complete for {room_name}")
        
        return {
            'room': room_name,
            'patterns': patterns,
            'total_events': len(historical_events),
            'summary': self._generate_pattern_summary(patterns)
        }
    
    async def _comprehensive_discovery(self, events: List[Dict], room_name: str) -> Dict:
        """Run all pattern discovery methods"""
        patterns = {}
        
        # Convert events to DataFrame for easier manipulation
        df = self._events_to_dataframe(events)
        
        if df.empty:
            logger.warning(f"No valid events to analyze for {room_name}")
            return patterns
        
        # 1. Time Series Pattern Discovery with STUMPY
        logger.info("🕐 Discovering recurring time patterns...")
        patterns['recurring'] = await self._discover_recurring_patterns(df)
        
        # 2. Sequential Pattern Mining
        logger.info("🔄 Mining sequential patterns...")
        patterns['sequential'] = await self._discover_sequential_patterns(df)
        
        # 3. Anomaly Detection
        logger.info("⚠️ Detecting anomalies...")
        patterns['anomalies'] = await self._detect_anomalies(df)
        
        # 4. Seasonal Patterns
        logger.info("📅 Finding seasonal patterns...")
        patterns['seasonal'] = await self._discover_seasonal_patterns(df)
        
        # 5. Behavioral Change Points
        logger.info("📈 Detecting behavioral changes...")
        patterns['change_points'] = await self._detect_change_points(df)
        
        # 6. Occupancy Duration Patterns
        logger.info("⏱️ Analyzing occupancy durations...")
        patterns['durations'] = await self._analyze_duration_patterns(df)
        
        # 7. Transition Patterns
        logger.info("🚶 Analyzing movement transitions...")
        patterns['transitions'] = await self._analyze_transitions(df)
        
        return patterns
    
    def _events_to_dataframe(self, events: List[Dict]) -> pd.DataFrame:
        """Convert events to pandas DataFrame with proper types"""
        df = pd.DataFrame(events)
        
        if df.empty:
            return df
        
        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Create binary occupancy column
        df['occupied'] = df['state'].isin(['on', 'occupied', '1', 1]).astype(int)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    async def _discover_recurring_patterns(self, df: pd.DataFrame) -> Dict:
        """Find ACTUAL behavioral patterns, not noise"""
        try:
            logger.info("🔄 Starting recurring pattern discovery...")
            
            # Create regular time series (5-minute intervals)
            logger.info("📊 Creating 5-minute time series from raw events...")
            ts = df.set_index('timestamp')['occupied'].resample('5min').max().fillna(0)
            logger.info(f"✅ Time series created: {len(ts)} data points spanning {ts.index[0]} to {ts.index[-1]}")
            
            if len(ts) < 24:
                logger.warning(f"❌ Insufficient data: {len(ts)} points < 24 minimum")
                return {'error': 'Insufficient data for pattern discovery'}
            
            patterns = {}
            
            # Try different window sizes
            window_configs = [
                (6, '30min'),
                (12, '1hour'),
                (24, '2hours'),
                (48, '4hours'),
                (144, '12hours'),
                (288, '24hours')
            ]
            
            logger.info(f"🔍 Testing {len(window_configs)} window configurations...")
            
            for i, (window_size, label) in enumerate(window_configs):
                logger.info(f"🎯 Processing window {i+1}/{len(window_configs)}: {label} ({window_size} intervals)")
                
                if len(ts) < window_size * 2:
                    logger.info(f"⏭️  Skipping {label}: need {window_size * 2} points, have {len(ts)}")
                    continue
                
                # Log memory before STUMPY
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                logger.info(f"🧠 Memory before STUMPY: {current_memory:.1f} MB")
                
                # CRITICAL: Validate data before STUMPY to prevent the numpy warnings!
                data_values = ts.values
                data_shape = data_values.shape
                data_std = np.std(data_values)
                data_mean = np.mean(data_values)
                unique_values = len(np.unique(data_values))
                
                logger.info(f"📏 STUMPY input validation:")
                logger.info(f"   Shape: {data_shape}, Window: {window_size}")
                logger.info(f"   Mean: {data_mean:.3f}, Std: {data_std:.3f}")
                logger.info(f"   Unique values: {unique_values}")
                logger.info(f"   Value range: {data_values.min():.3f} to {data_values.max():.3f}")
                
                # Check for problematic data that causes numpy warnings
                if data_std < 1e-10:
                    logger.warning(f"🚨 SKIPPING {label}: Zero standard deviation (constant data) - this causes STUMPY to hang!")
                    patterns[label] = {
                        'found': False,
                        'pattern_count': 0,
                        'reason': f'Constant data (std={data_std:.2e}) - all values are {data_mean:.1f}'
                    }
                    continue
                
                # CRITICAL FIX: Binary sensors (0/1) are valid for STUMPY - check temporal variation instead
                if unique_values < 2:
                    logger.warning(f"🚨 SKIPPING {label}: Constant data - no variation at all")
                    patterns[label] = {
                        'found': False, 
                        'pattern_count': 0,
                        'reason': f'Constant data - no variation'
                    }
                    continue
                
                # For binary data, check if there's meaningful temporal variation
                if unique_values == 2:
                    # Calculate transition rate - how often does state change?
                    transitions = np.sum(np.abs(np.diff(data_values)))
                    transition_rate = transitions / len(data_values)
                    
                    # Binary data needs some transitions to be meaningful for pattern discovery
                    if transition_rate < 0.01:  # Less than 1% of samples have transitions
                        logger.warning(f"🚨 SKIPPING {label}: Binary data with minimal transitions ({transition_rate:.1%})")
                        patterns[label] = {
                            'found': False,
                            'pattern_count': 0,
                            'reason': f'Binary data with insufficient temporal variation ({transition_rate:.1%} transition rate)'
                        }
                        continue
                    else:
                        logger.info(f"✅ Binary data validation: {transition_rate:.1%} transition rate - suitable for STUMPY analysis")
                
                # CRITICAL NEW VALIDATION: Check sliding windows for constant data
                # This is the ROOT CAUSE - STUMPY fails when individual windows have zero std
                logger.info(f"🔍 Checking sliding windows for constant data...")
                constant_windows = 0
                total_windows = len(data_values) - window_size + 1
                
                # Sample check: test every 10th window to avoid performance issues
                sample_step = max(1, total_windows // 100)  # Check at most 100 windows
                for i in range(0, total_windows, sample_step):
                    window_data = data_values[i:i+window_size]
                    window_std = np.std(window_data)
                    if window_std < 1e-12:  # Even stricter for windows
                        constant_windows += 1
                
                # Calculate percentage of constant windows from sample
                sampled_windows = len(range(0, total_windows, sample_step))
                constant_percentage = (constant_windows / sampled_windows * sample_step) if sampled_windows > 0 else 0
                
                logger.info(f"   Sampled {sampled_windows} windows, found {constant_windows} constant windows")
                logger.info(f"   Estimated {constant_percentage:.1f}% constant windows")
                
                # Skip if too many constant windows (this causes STUMPY division by zero)
                if constant_percentage > 50:  # More than 50% constant windows
                    logger.warning(f"🚨 SKIPPING {label}: {constant_percentage:.1f}% constant sliding windows - causes STUMPY division by zero!")
                    patterns[label] = {
                        'found': False,
                        'pattern_count': 0,
                        'reason': f'{constant_percentage:.1f}% constant sliding windows (causes numpy division by zero)'
                    }
                    continue
                
                # Additional validation: Check for insufficient variation across windows
                if data_mean < 0.001 or data_mean > 0.999:  # Nearly all zeros or all ones
                    occupancy_rate = data_mean
                    logger.warning(f"🚨 SKIPPING {label}: Extreme occupancy rate {occupancy_rate:.1%} - insufficient pattern variation")
                    patterns[label] = {
                        'found': False,
                        'pattern_count': 0,
                        'reason': f'Extreme occupancy rate ({occupancy_rate:.1%}) - insufficient variation for patterns'
                    }
                    continue
                
                logger.info(f"✅ Data validation passed for {label}")
                logger.info(f"⚡ Starting STUMPY matrix profile computation for {label}...")
                
                # This is where it hangs - add timeout handling
                import signal
                import asyncio
                from concurrent.futures import ThreadPoolExecutor
                
                def stumpy_worker():
                    """Run STUMPY in separate thread with detailed logging and numpy warning suppression"""
                    try:
                        logger.info(f"🚀 STUMPY thread started for window {window_size}")
                        
                        # Suppress only specific STUMPY-related warnings, not all numpy warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', message='divide by zero encountered in scalar divide', module='stumpy')
                            warnings.filterwarnings('ignore', message='invalid value encountered in divide', module='stumpy')
                            warnings.filterwarnings('ignore', message='divide by zero encountered in true_divide', module='stumpy')
                            
                            # Pre-validate one more time right before STUMPY
                            data_for_stumpy = ts.values
                            if len(data_for_stumpy) < window_size * 2:
                                raise ValueError(f"Insufficient data: {len(data_for_stumpy)} < {window_size * 2}")
                            
                            # Final check for problematic data patterns
                            overall_std = np.std(data_for_stumpy)
                            if overall_std < 1e-12:
                                raise ValueError(f"Data has zero variance: std={overall_std:.2e}")
                            
                            # CRITICAL: For binary data, use deterministic epsilon to prevent division by zero
                            # while preserving the binary semantic meaning (0/1 occupancy states)
                            if unique_values == 2:  # Binary data - preserve 0/1 values
                                logger.info(f"🔧 Applying deterministic epsilon for binary data preservation")
                                epsilon = np.finfo(float).eps * 10  # Minimal deterministic offset
                                data_with_noise = data_for_stumpy.astype(float) + np.arange(len(data_for_stumpy)) * epsilon
                            else:
                                # For non-binary data, add minimal random noise to break ties
                                noise_level = overall_std * 1e-6  # 0.0001% of signal std
                                if noise_level < 1e-15:
                                    noise_level = 1e-15
                                data_with_noise = data_for_stumpy + np.random.normal(0, noise_level, size=len(data_for_stumpy))
                            
                            logger.info(f"🎯 STUMPY input: {len(data_with_noise)} points, std={np.std(data_with_noise):.8f}")
                            
                            # Run STUMPY with preprocessed data
                            mp = stumpy.stump(data_with_noise, m=window_size)
                        
                        logger.info(f"✅ STUMPY completed for {label}: matrix profile shape {mp.shape}")
                        return mp
                        
                    except Exception as e:
                        logger.error(f"💥 STUMPY failed for {label}: {e}")
                        # Don't re-raise - return None to continue with other window sizes
                        return None
                
                # Run STUMPY with timeout
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(stumpy_worker)
                        # Wait up to 10 minutes for each window size (was 30min timeout causing issues)
                        mp = future.result(timeout=600)  
                    
                    # Check if STUMPY failed gracefully
                    if mp is None:
                        logger.error(f"❌ STUMPY returned None for {label} - computation failed")
                        patterns[label] = {
                            'found': False,
                            'pattern_count': 0,
                            'reason': 'STUMPY computation failed due to data characteristics'
                        }
                        continue
                    
                    # Log completion
                    final_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    logger.info(f"🎉 STUMPY completed for {label}! Memory: {final_memory:.1f} MB (+{final_memory-current_memory:.1f} MB)")
                    
                except Exception as e:
                    logger.error(f"❌ STUMPY failed for {label} after timeout/error: {e}")
                    patterns[label] = {
                        'found': False,
                        'pattern_count': 0,
                        'reason': f'STUMPY execution failed: {str(e)}'
                    }
                    continue
                
                # CRITICAL: For occupancy data, patterns should be VERY similar
                # Distance of 0.1 for normalized data means 99% similar
                # For binary data, even stricter
                STRICT_THRESHOLD = 0.05 * np.sqrt(window_size)
                
                # Find all potential matches
                potential_patterns = []
                for idx in range(len(mp)):
                    if mp[idx, 0] < STRICT_THRESHOLD:
                        potential_patterns.append({
                            'idx': idx,
                            'match_idx': int(mp[idx, 1]),
                            'distance': mp[idx, 0],
                            'pattern': ts.iloc[idx:idx+window_size].values
                        })
                
                # CRITICAL FIX: Cluster similar patterns together
                pattern_clusters = []
                used_indices = set()
                
                for pattern in potential_patterns:
                    if pattern['idx'] in used_indices:
                        continue
                    
                    # Start new cluster
                    cluster = {
                        'representative_idx': pattern['idx'],
                        'pattern_data': pattern['pattern'],
                        'occurrences': [pattern['idx']],
                        'distances': [pattern['distance']]
                    }
                    
                    # Find all similar patterns
                    for other in potential_patterns:
                        if other['idx'] in used_indices:
                            continue
                        
                        # Check if patterns are similar enough to be in same cluster
                        pattern_similarity = np.corrcoef(
                            pattern['pattern'], 
                            other['pattern']
                        )[0, 1]
                        
                        if pattern_similarity > 0.95:  # 95% correlation
                            cluster['occurrences'].append(other['idx'])
                            cluster['distances'].append(other['distance'])
                            used_indices.add(other['idx'])
                    
                    # Only keep clusters with multiple occurrences
                    if len(cluster['occurrences']) >= 3:
                        used_indices.update(cluster['occurrences'])
                        pattern_clusters.append(cluster)
                
                # Analyze each pattern cluster
                verified_patterns = []
                for cluster in pattern_clusters:
                    # Get timestamps of occurrences
                    timestamps = [ts.index[idx] for idx in cluster['occurrences']]
                    timestamps.sort()
                    
                    # Analyze pattern characteristics
                    pattern_data = cluster['pattern_data']
                    
                    # Skip trivial patterns
                    occupancy_rate = pattern_data.mean()
                    if occupancy_rate == 0 or occupancy_rate == 1:
                        continue  # All empty or all occupied
                    
                    # Check for meaningful transitions
                    transitions = np.sum(np.abs(np.diff(pattern_data)))
                    if transitions == 0:
                        continue  # No state changes
                    
                    # Analyze temporal consistency
                    hours = [t.hour for t in timestamps]
                    hour_counts = Counter(hours)
                    most_common_hour = hour_counts.most_common(1)[0]
                    hour_consistency = most_common_hour[1] / len(timestamps)
                    
                    # Check day of week consistency
                    days = [t.dayofweek for t in timestamps]
                    day_counts = Counter(days)
                    
                    # Determine pattern type
                    pattern_type = 'irregular'
                    typical_time = None
                    typical_days = []
                    
                    if hour_consistency > 0.7:
                        pattern_type = 'time_specific'
                        typical_time = most_common_hour[0]
                        
                        # Check if it's weekday/weekend specific
                        weekday_count = sum(1 for d in days if d < 5)
                        weekend_count = len(days) - weekday_count
                        
                        if weekday_count > weekend_count * 3:
                            pattern_type = 'weekday_routine'
                            typical_days = [0, 1, 2, 3, 4]
                        elif weekend_count > weekday_count * 2:
                            pattern_type = 'weekend_routine'
                            typical_days = [5, 6]
                    
                    # Calculate true frequency (occurrences per week)
                    total_days = (timestamps[-1] - timestamps[0]).days
                    occurrences_per_week = len(timestamps) / (total_days / 7) if total_days > 0 else 0
                    
                    # Only keep patterns that occur regularly
                    if occurrences_per_week < 0.5:  # Less than once every 2 weeks
                        continue
                    
                    verified_patterns.append({
                        'pattern_id': f"{label}_{len(verified_patterns)}",
                        'occurrences': len(cluster['occurrences']),
                        'occurrences_per_week': round(occurrences_per_week, 1),
                        'pattern_type': pattern_type,
                        'typical_time': typical_time,
                        'typical_days': typical_days,
                        'hour_consistency': round(hour_consistency, 2),
                        'occupancy_transitions': int(transitions),
                        'sample_timestamps': [t.strftime('%Y-%m-%d %H:%M') for t in timestamps[:3]],
                        'description': self._describe_pattern(
                            pattern_data, pattern_type, typical_time, typical_days
                        )
                    })
                
                # Sort by frequency and importance
                verified_patterns.sort(key=lambda x: x['occurrences_per_week'], reverse=True)
                
                # Log what we found
                if verified_patterns:
                    logger.info(f"{label}: Found {len(verified_patterns)} distinct patterns "
                              f"(from {len(potential_patterns)} similar subsequences)")
                    
                    patterns[label] = {
                        'found': True,
                        'pattern_count': len(verified_patterns),
                        'patterns': verified_patterns[:5],  # Top 5 most frequent
                        'total_subsequences_analyzed': len(mp),
                        'similar_subsequences_found': len(potential_patterns),
                        'distinct_patterns': len(verified_patterns)
                    }
                else:
                    patterns[label] = {
                        'found': False,
                        'pattern_count': 0,
                        'reason': 'No meaningful recurring patterns found'
                    }
            
            # Overall analysis
            total_patterns = sum(
                p.get('pattern_count', 0) 
                for p in patterns.values() 
                if isinstance(p, dict)
            )
            
            patterns['analysis'] = {
                'total_distinct_patterns': total_patterns,
                'has_routines': total_patterns > 0,
                'routine_strength': 'strong' if total_patterns > 10 else 'weak' if total_patterns > 0 else 'none',
                'most_consistent_scale': self._find_most_consistent_scale(patterns)
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in pattern discovery: {e}")
            return {'error': str(e)}
    
    async def _discover_sequential_patterns(self, df: pd.DataFrame) -> Dict:
        """Mine sequential patterns using mlxtend"""
        try:
            # Create sessions (continuous occupancy periods)
            sessions = []
            current_session = []
            last_occupied = False
            
            for _, row in df.iterrows():
                if row['occupied']:
                    if not last_occupied:
                        # New session started
                        if current_session:
                            sessions.append(current_session)
                        current_session = [row['entity_id']]
                    else:
                        # Continue session
                        if row['entity_id'] not in current_session:
                            current_session.append(row['entity_id'])
                else:
                    if last_occupied and current_session:
                        # Session ended
                        sessions.append(current_session)
                        current_session = []
                
                last_occupied = row['occupied']
            
            if current_session:
                sessions.append(current_session)
            
            if len(sessions) < 10:
                return {'error': 'Insufficient sessions for pattern mining'}
            
            # Transaction encoding
            te = TransactionEncoder()
            te_ary = te.fit(sessions).transform(sessions)
            transactions_df = pd.DataFrame(te_ary, columns=te.columns_)
            
            # Find frequent itemsets
            frequent_itemsets = fpgrowth(
                transactions_df, 
                min_support=0.05,  # At least 5% support
                use_colnames=True,
                max_len=4  # Max 4 items in pattern
            )
            
            if frequent_itemsets.empty:
                return {'no_patterns': True}
            
            # Generate association rules
            rules = association_rules(
                frequent_itemsets, 
                metric="confidence", 
                min_threshold=0.5  # At least 50% confidence
            )
            
            # Format results
            patterns = {
                'frequent_sets': [],
                'rules': [],
                'session_count': len(sessions)
            }
            
            # Top frequent itemsets
            for _, row in frequent_itemsets.nlargest(10, 'support').iterrows():
                patterns['frequent_sets'].append({
                    'items': list(row['itemsets']),
                    'support': float(row['support']),
                    'frequency': int(row['support'] * len(sessions))
                })
            
            # Top rules
            if not rules.empty:
                for _, row in rules.nlargest(10, 'lift').iterrows():
                    patterns['rules'].append({
                        'if': list(row['antecedents']),
                        'then': list(row['consequents']),
                        'confidence': float(row['confidence']),
                        'lift': float(row['lift']),
                        'support': float(row['support'])
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in sequential pattern mining: {e}")
            return {'error': str(e)}
    
    async def _detect_anomalies(self, df: pd.DataFrame) -> Dict:
        """Detect anomalous patterns using PyOD"""
        try:
            # Create feature matrix
            features = []
            
            # Group by hour to create hourly features
            hourly_occupancy = df.groupby(df['timestamp'].dt.floor('h')).agg({
                'occupied': ['sum', 'mean'],
                'entity_id': 'count'
            }).reset_index()
            
            if len(hourly_occupancy) < 50:
                return {'error': 'Insufficient data for anomaly detection'}
            
            # Create feature matrix
            for _, hour_data in hourly_occupancy.iterrows():
                timestamp = hour_data['timestamp']
                features.append([
                    timestamp.hour,
                    timestamp.dayofweek,
                    hour_data['occupied']['sum'],
                    hour_data['occupied']['mean'],
                    hour_data['entity_id']['count']
                ])
            
            features = np.array(features)
            
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Apply multiple anomaly detectors
            results = {}
            
            # Isolation Forest
            iforest = IForest(contamination=0.1, random_state=42)
            iforest.fit(features_scaled)
            if_scores = iforest.decision_scores_
            if_labels = iforest.labels_
            
            # Local Outlier Factor
            lof = LOF(contamination=0.1)
            lof.fit(features_scaled)
            lof_scores = lof.decision_scores_
            lof_labels = lof.labels_
            
            # Find consensus anomalies
            consensus_anomalies = np.where((if_labels == 1) & (lof_labels == 1))[0]
            
            # Get anomaly details
            anomaly_details = []
            for idx in consensus_anomalies[:10]:  # Top 10 anomalies
                timestamp = hourly_occupancy.iloc[idx]['timestamp']
                anomaly_details.append({
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M'),
                    'hour': int(timestamp.hour),
                    'day_of_week': int(timestamp.dayofweek),
                    'occupancy_sum': int(hourly_occupancy.iloc[idx]['occupied']['sum']),
                    'occupancy_mean': float(hourly_occupancy.iloc[idx]['occupied']['mean']),
                    'event_count': int(hourly_occupancy.iloc[idx]['entity_id']['count']),
                    'if_score': float(if_scores[idx]),
                    'lof_score': float(lof_scores[idx])
                })
            
            results = {
                'anomaly_count': len(consensus_anomalies),
                'anomaly_percentage': float(len(consensus_anomalies) / len(hourly_occupancy) * 100),
                'top_anomalies': anomaly_details,
                'methods_agree': True
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {'error': str(e)}
    
    async def _discover_seasonal_patterns(self, df: pd.DataFrame) -> Dict:
        """Discover daily/weekly seasonal patterns"""
        try:
            # Create regular time series
            ts = df.set_index('timestamp')['occupied'].resample('15min').max().fillna(0)
            
            if len(ts) < 96 * 7:  # Need at least 1 week of data
                return {'error': 'Insufficient data for seasonal analysis'}
            
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                ts, 
                model='additive', 
                period=96  # Daily period (96 * 15min = 24h)
            )
            
            # Extract patterns
            seasonal = decomposition.seasonal
            trend = decomposition.trend.dropna()
            
            # Find peak hours
            hourly_pattern = seasonal.groupby(seasonal.index.hour).mean()
            peak_hours = hourly_pattern.nlargest(3).index.tolist()
            quiet_hours = hourly_pattern.nsmallest(3).index.tolist()
            
            # Calculate pattern strength
            pattern_strength = float(seasonal.std())
            has_pattern = pattern_strength > 0.1
            
            # Day of week patterns
            daily_pattern = ts.groupby(ts.index.dayofweek).mean()
            weekday_avg = daily_pattern[0:5].mean()
            weekend_avg = daily_pattern[5:7].mean()
            
            results = {
                'has_daily_pattern': has_pattern,
                'pattern_strength': pattern_strength,
                'peak_hours': peak_hours,
                'quiet_hours': quiet_hours,
                'weekday_occupancy': float(weekday_avg),
                'weekend_occupancy': float(weekend_avg),
                'weekend_different': abs(weekday_avg - weekend_avg) > 0.1,
                'trend': 'increasing' if trend.iloc[-1] > trend.iloc[0] else 'decreasing',
                'hourly_pattern': {
                    str(hour): float(value) 
                    for hour, value in hourly_pattern.items()
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in seasonal pattern discovery: {e}")
            return {'error': str(e)}
    
    async def _detect_change_points(self, df: pd.DataFrame) -> Dict:
        """Detect points where behavior changes significantly"""
        try:
            # Create daily aggregates
            daily = df.set_index('timestamp').resample('D').agg({
                'occupied': ['sum', 'mean'],
                'entity_id': 'count'
            })
            
            if len(daily) < 14:  # Need at least 2 weeks
                return {'error': 'Insufficient data for change detection'}
            
            # Use occupancy sum as signal
            signal = daily['occupied']['sum'].values
            
            # Detect change points
            algo = rpt.Pelt(model="rbf", min_size=3, jump=1).fit(signal)
            change_points = algo.predict(pen=10)
            
            # Remove the last point (end of signal)
            change_points = change_points[:-1]
            
            # Get details about each segment
            segments = []
            start_idx = 0
            
            for cp in change_points + [len(signal)]:
                if cp > start_idx:
                    segment_data = signal[start_idx:cp]
                    segments.append({
                        'start_date': daily.index[start_idx].strftime('%Y-%m-%d'),
                        'end_date': daily.index[cp-1].strftime('%Y-%m-%d'),
                        'duration_days': cp - start_idx,
                        'mean_occupancy': float(np.mean(segment_data)),
                        'std_occupancy': float(np.std(segment_data))
                    })
                start_idx = cp
            
            results = {
                'change_points_found': len(change_points),
                'behavior_stable': len(change_points) <= 2,
                'segments': segments,
                'dates': [daily.index[cp].strftime('%Y-%m-%d') for cp in change_points]
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in change point detection: {e}")
            return {'error': str(e)}
    
    async def _analyze_duration_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze how long occupancy periods typically last"""
        try:
            durations = []
            start_time = None
            
            for _, row in df.iterrows():
                if row['occupied'] and start_time is None:
                    start_time = row['timestamp']
                elif not row['occupied'] and start_time is not None:
                    duration = (row['timestamp'] - start_time).total_seconds() / 60  # Minutes
                    durations.append({
                        'duration': duration,
                        'start_hour': start_time.hour,
                        'day_of_week': start_time.dayofweek,
                        'is_weekend': start_time.dayofweek >= 5
                    })
                    start_time = None
            
            if not durations:
                return {'no_occupancy_periods': True}
            
            durations_df = pd.DataFrame(durations)
            
            # Overall statistics
            results = {
                'total_periods': len(durations),
                'mean_duration': float(durations_df['duration'].mean()),
                'median_duration': float(durations_df['duration'].median()),
                'std_duration': float(durations_df['duration'].std()),
                'min_duration': float(durations_df['duration'].min()),
                'max_duration': float(durations_df['duration'].max())
            }
            
            # Duration by hour of day
            hourly_durations = durations_df.groupby('start_hour')['duration'].agg(['mean', 'count'])
            results['hourly_patterns'] = {
                str(hour): {
                    'mean_duration': float(row['mean']),
                    'frequency': int(row['count'])
                }
                for hour, row in hourly_durations.iterrows()
            }
            
            # Weekend vs weekday
            results['weekday_mean'] = float(durations_df[~durations_df['is_weekend']]['duration'].mean())
            results['weekend_mean'] = float(durations_df[durations_df['is_weekend']]['duration'].mean())
            
            # Common durations (clustering)
            from sklearn.cluster import KMeans
            
            if len(durations) > 10:
                durations_array = durations_df['duration'].values.reshape(-1, 1)
                
                # Find optimal number of clusters (up to 5)
                max_clusters = min(5, len(durations) // 3)
                if max_clusters >= 2:
                    inertias = []
                    for k in range(2, max_clusters + 1):
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans.fit(durations_array)
                        inertias.append(kmeans.inertia_)
                    
                    # Use elbow method
                    optimal_k = 2  # Default
                    if len(inertias) > 1:
                        deltas = np.diff(inertias)
                        if len(deltas) > 0:
                            optimal_k = np.argmin(deltas) + 2
                    
                    # Cluster with optimal k
                    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(durations_array)
                    
                    # Get cluster info
                    cluster_info = []
                    for i in range(optimal_k):
                        cluster_durations = durations_array[clusters == i]
                        cluster_info.append({
                            'mean_duration': float(cluster_durations.mean()),
                            'size': len(cluster_durations),
                            'percentage': float(len(cluster_durations) / len(durations) * 100)
                        })
                    
                    results['duration_clusters'] = sorted(cluster_info, key=lambda x: x['mean_duration'])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in duration analysis: {e}")
            return {'error': str(e)}
    
    async def _analyze_transitions(self, df: pd.DataFrame) -> Dict:
        """Analyze transition patterns between states"""
        try:
            transitions = []
            
            # Track state changes
            prev_state = None
            prev_entity = None
            prev_time = None
            
            for _, row in df.iterrows():
                current_state = 'occupied' if row['occupied'] else 'empty'
                current_entity = row['entity_id']
                current_time = row['timestamp']
                
                if prev_state is not None:
                    # Record transition
                    time_diff = (current_time - prev_time).total_seconds()
                    
                    if time_diff < 3600:  # Within 1 hour
                        transitions.append({
                            'from_state': prev_state,
                            'to_state': current_state,
                            'from_entity': prev_entity,
                            'to_entity': current_entity,
                            'time_diff': time_diff,
                            'hour': current_time.hour,
                            'day_of_week': current_time.dayofweek
                        })
                
                prev_state = current_state
                prev_entity = current_entity
                prev_time = current_time
            
            if not transitions:
                return {'no_transitions': True}
            
            transitions_df = pd.DataFrame(transitions)
            
            # Transition matrix
            transition_counts = transitions_df.groupby(['from_state', 'to_state']).size().unstack(fill_value=0)
            
            # Normalize to get probabilities
            transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)
            
            results = {
                'total_transitions': len(transitions),
                'transition_matrix': transition_probs.to_dict(),
                'mean_transition_time': float(transitions_df['time_diff'].mean()),
                'median_transition_time': float(transitions_df['time_diff'].median())
            }
            
            # Most common entity transitions
            entity_transitions = transitions_df[
                transitions_df['from_entity'] != transitions_df['to_entity']
            ].groupby(['from_entity', 'to_entity']).size()
            
            if not entity_transitions.empty:
                top_transitions = entity_transitions.nlargest(10)
                results['top_entity_transitions'] = [
                    {
                        'from': from_e,
                        'to': to_e,
                        'count': int(count),
                        'percentage': float(count / len(transitions) * 100)
                    }
                    for (from_e, to_e), count in top_transitions.items()
                ]
            
            return results
            
        except Exception as e:
            logger.error(f"Error in transition analysis: {e}")
            return {'error': str(e)}
    
    async def _load_events_with_monitoring(self, room_name: str) -> List[Dict]:
        """Load events with memory monitoring - CRITICAL: Only FULL room sensors"""
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
            # Monitor memory during loading
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Loading FULL SENSORS ONLY for {room_name}, initial memory: {initial_memory:.1f} MB")
            
            # Get full room sensors for this room
            full_sensors = self.FULL_ROOM_SENSORS.get(room_name, [])
            
            if not full_sensors:
                logger.warning(f"No full room sensors defined for {room_name}")
                return []
            
            async with db.engine.begin() as conn:
                from sqlalchemy import text
                
                # Build exact sensor filter for FULL room sensors only
                sensor_conditions = []
                for sensor in full_sensors:
                    sensor_conditions.append(f"entity_id = '{sensor}'")
                
                sensor_filter = " OR ".join(sensor_conditions)
                
                query = f"""
                    SELECT timestamp, entity_id, state, attributes, room, sensor_type, derived_features
                    FROM sensor_events 
                    WHERE ({sensor_filter})
                    AND timestamp >= NOW() - INTERVAL '180 days'
                    ORDER BY timestamp DESC
                    LIMIT 100000
                """
                
                logger.info(f"📍 Querying ONLY full sensors: {full_sensors}")
                
                result = await conn.execute(text(query))
                events = [dict(row._mapping) for row in result.fetchall()]
                
                # Check memory after loading
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                logger.info(f"Loaded {len(events):,} FULL SENSOR events, memory: {final_memory:.1f} MB "
                          f"(+{final_memory-initial_memory:.1f} MB)")
                
                # Memory safety check
                if final_memory > self.memory_limit_mb:
                    logger.warning(f"High memory usage ({final_memory:.1f} MB), limiting events")
                    events = events[:50000]  # Limit to 50k events if memory is high
                
                return events
                
        finally:
            await db.close()
    
    async def store_patterns_in_redis(self, room_id: str, patterns: Dict):
        """Store discovered patterns in Redis for model access"""
        try:
            from src.storage.feature_store import RedisFeatureStore
            from config.config_loader import ConfigLoader
            
            config = ConfigLoader()
            redis_config = config.get("redis")
            redis_url = f"redis://{redis_config['host']}:{redis_config['port']}/{redis_config.get('db', 0)}"
            
            feature_store = RedisFeatureStore(redis_url)
            await feature_store.initialize()
            
            # Prepare pattern data for storage
            from datetime import timezone
            pattern_data = {
                'room_id': room_id,
                'discovery_time': datetime.now(timezone.utc).isoformat(),
                'patterns': patterns,
                'summary': self._generate_pattern_summary(patterns)
            }
            
            # Store in Redis with proper await
            await feature_store.cache_pattern(room_id, 'discovered_patterns', pattern_data)
            
            logger.info(f"✅ Stored patterns for {room_id} in Redis")
            
            await feature_store.close()
            
        except Exception as e:
            logger.error(f"Error storing patterns in Redis for {room_id}: {e}")
    
    def _generate_pattern_summary(self, patterns: Dict) -> Dict:
        """Generate a summary of discovered patterns"""
        summary = {
            'has_patterns': False,
            'pattern_types': [],
            'key_insights': []
        }
        
        # Check recurring patterns
        if patterns.get('recurring'):
            for window, data in patterns['recurring'].items():
                if data.get('found') and data.get('pattern_count', 0) > 0:
                    summary['has_patterns'] = True
                    summary['pattern_types'].append(f'recurring_{window}')
                    summary['key_insights'].append(
                        f"Found {data['pattern_count']} recurring patterns at {window} scale"
                    )
        
        # Check sequential patterns
        if patterns.get('sequential') and not patterns['sequential'].get('no_patterns'):
            summary['has_patterns'] = True
            summary['pattern_types'].append('sequential')
            if patterns['sequential'].get('rules'):
                summary['key_insights'].append(
                    f"Found {len(patterns['sequential']['rules'])} sequential rules"
                )
        
        # Check anomalies
        if patterns.get('anomalies') and patterns['anomalies'].get('anomaly_count', 0) > 0:
            summary['pattern_types'].append('anomalies')
            summary['key_insights'].append(
                f"Detected {patterns['anomalies']['anomaly_count']} anomalous periods"
            )
        
        # Check seasonal patterns
        if patterns.get('seasonal') and patterns['seasonal'].get('has_daily_pattern'):
            summary['has_patterns'] = True
            summary['pattern_types'].append('seasonal')
            peak_hours = patterns['seasonal'].get('peak_hours', [])
            if peak_hours:
                summary['key_insights'].append(
                    f"Peak activity at hours: {', '.join(map(str, peak_hours))}"
                )
        
        # Check behavior changes
        if patterns.get('change_points') and patterns['change_points'].get('change_points_found', 0) > 0:
            summary['pattern_types'].append('behavioral_changes')
            summary['key_insights'].append(
                f"Behavior changed {patterns['change_points']['change_points_found']} times"
            )
        
        # Duration insights
        if patterns.get('durations') and patterns['durations'].get('mean_duration'):
            avg_duration = patterns['durations']['mean_duration']
            summary['key_insights'].append(
                f"Average occupancy duration: {avg_duration:.0f} minutes"
            )
        
        return summary
    
    def _describe_pattern(self, pattern_data: np.ndarray, pattern_type: str, 
                         typical_time: int, typical_days: list) -> str:
        """Generate human-readable pattern description"""
        
        # Analyze pattern shape
        start_occupied = pattern_data[0] == 1
        end_occupied = pattern_data[-1] == 1
        
        if pattern_type == 'weekday_routine':
            time_str = f"{typical_time:02d}:00" if typical_time is not None else "consistent time"
            if start_occupied and not end_occupied:
                return f"Weekday departure around {time_str}"
            elif not start_occupied and end_occupied:
                return f"Weekday arrival around {time_str}"
            else:
                return f"Weekday activity pattern around {time_str}"
        
        elif pattern_type == 'weekend_routine':
            time_str = f"{typical_time:02d}:00" if typical_time is not None else "consistent time"
            return f"Weekend pattern around {time_str}"
        
        elif pattern_type == 'time_specific':
            time_str = f"{typical_time:02d}:00" if typical_time is not None else "specific time"
            return f"Daily pattern around {time_str}"
        
        else:
            return "Irregular recurring pattern"
    
    def _find_most_consistent_scale(self, patterns: Dict) -> str:
        """Find which time scale has the most consistent patterns"""
        best_scale = None
        best_consistency = 0
        
        for scale, data in patterns.items():
            if isinstance(data, dict) and data.get('patterns'):
                # Average hour consistency across patterns
                avg_consistency = np.mean([
                    p.get('hour_consistency', 0) 
                    for p in data['patterns']
                ])
                
                if avg_consistency > best_consistency:
                    best_consistency = avg_consistency
                    best_scale = scale
        
        return best_scale or 'none'

    async def discover_bathroom_patterns(self, bathroom_rooms: List[str]) -> Dict:
        """Discover patterns for bathroom rooms using advanced algorithms"""
        logger.info(f"Discovering bathroom patterns for {bathroom_rooms}")
        
        bathroom_patterns = {}
        
        for bathroom_room in bathroom_rooms:
            # Note: Bathrooms use door sensors only, not interior full sensors
            patterns = await self.discover_multizone_patterns(bathroom_room)
            bathroom_patterns[bathroom_room] = patterns
        
        logger.info(f"Bathroom pattern discovery completed for {len(bathroom_patterns)} bathrooms")
        return {'patterns': bathroom_patterns}


# Additional helper functions for backward compatibility
async def discover_bathroom_patterns(room_names: List[str]) -> Dict:
    """Specialized pattern discovery for bathrooms"""
    discoverer = PatternDiscovery()
    bathroom_patterns = {}
    
    for room in room_names:
        patterns = await discoverer.discover_multizone_patterns(room)
        bathroom_patterns[room] = patterns
    
    return bathroom_patterns


async def discover_transition_patterns(area_name: str) -> Dict:
    """Discover transition patterns for hallways and common areas"""
    discoverer = PatternDiscovery()
    return await discoverer.discover_multizone_patterns(area_name)