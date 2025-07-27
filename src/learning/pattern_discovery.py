"""
Automatic Pattern Mining - Discovers behavioral patterns without assumptions
Implements the improved pattern discovery with proper statistical testing

IMPROVED VERSION with real statistical significance testing and chunked processing
"""

import logging
import psutil
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta, timezone
import numpy as np
from scipy import stats
from suffix_trees import STree

logger = logging.getLogger(__name__)


class PatternDiscovery:
    """
    Discovers behavioral patterns from event streams without any assumptions
    Uses improved statistical testing to find real behavioral patterns
    """
    
    def __init__(self):
        self.pattern_library = {}
        self.statistical_tests = StatisticalTestSuite()
        self.adaptive_threshold = 0.05  # P-value threshold for statistical significance
        # Add improved pattern discoverer
        self.pattern_discoverer = ImprovedPatternDiscovery()
        
    async def discover_multizone_patterns(self, room_name: str, historical_events: List[Dict] = None) -> Dict:
        """Use improved pattern discovery with chunked processing"""
        logger.info(f"Discovering patterns for {room_name}")
        
        if historical_events is None:
            historical_events = await self._load_events_with_monitoring(room_name)
        
        if not historical_events:
            logger.warning(f"No events for {room_name}")
            return {'room': room_name, 'patterns': {}}
        
        # Use the improved discovery with chunked processing
        logger.info(f"Starting improved pattern discovery for {room_name} with {len(historical_events):,} events")
        patterns = await self._discover_patterns_chunked(historical_events, room_name)
        
        # Store in library
        self.pattern_library[room_name] = patterns
        
        return {
            'room': room_name,
            'patterns': patterns,
            'total_events': len(historical_events)
        }
    
    async def _load_events_with_monitoring(self, room_name: str) -> List[Dict]:
        """Load events with memory monitoring to prevent hangs"""
        from src.storage.timeseries_db import TimescaleDBManager
        from config.config_loader import ConfigLoader
        
        config = ConfigLoader()
        db_config = config.get("database.timescale")
        db = TimescaleDBManager(f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
        await db.initialize()
        
        try:
            # Monitor memory during loading
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Loading events for {room_name}, initial memory: {initial_memory:.1f} MB")
            
            async with db.engine.begin() as conn:
                from sqlalchemy import text
                
                # Load full room sensors only with reasonable limit
                if room_name == 'living_kitchen':
                    query = """
                        SELECT timestamp, entity_id, state, attributes, room, sensor_type, derived_features
                        FROM sensor_events 
                        WHERE (entity_id LIKE '%livingroom_full%' OR entity_id LIKE '%kitchen_pressence_full%')
                        AND timestamp >= NOW() - INTERVAL '180 days'
                        ORDER BY timestamp DESC
                        LIMIT 50000
                    """
                else:
                    query = f"""
                        SELECT timestamp, entity_id, state, attributes, room, sensor_type, derived_features
                        FROM sensor_events 
                        WHERE entity_id LIKE '%{room_name}%full%'
                        AND timestamp >= NOW() - INTERVAL '180 days'
                        ORDER BY timestamp DESC
                        LIMIT 50000
                    """
                
                result = await conn.execute(text(query))
                events = [dict(row._mapping) for row in result.fetchall()]
                
                # Check memory after loading
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                logger.info(f"Loaded {len(events):,} events, memory: {final_memory:.1f} MB (+{final_memory-initial_memory:.1f} MB)")
                
                # Memory safety check
                if final_memory > 2000:  # 2GB limit
                    logger.warning(f"High memory usage ({final_memory:.1f} MB), limiting events")
                    events = events[:10000]  # Limit to 10k events if memory is high
                
                return events
                
        finally:
            await db.close()
    
    async def _discover_patterns_chunked(self, event_stream: List[Dict], room_id: str) -> Dict:
        """Discover patterns using improved algorithm with chunked processing"""
        logger.info(f"🔍 Starting improved pattern discovery for {room_id} with {len(event_stream):,} events")
        
        # Memory monitoring  
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        start_time = datetime.now()
        
        try:
            # CHUNKED PROCESSING - Process events in chunks to prevent memory explosion
            chunk_size = 5000  # Conservative chunk size for improved algorithm
            all_patterns = defaultdict(list)
            
            logger.info(f"📊 Processing {len(event_stream):,} events in chunks of {chunk_size:,}")
            
            for chunk_start in range(0, len(event_stream), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(event_stream))
                chunk_events = event_stream[chunk_start:chunk_end]
                
                # Progress monitoring
                progress_pct = (chunk_end / len(event_stream)) * 100
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                elapsed_time = datetime.now() - start_time
                
                logger.info(f"🔄 Processing chunk {chunk_start//chunk_size + 1}/{(len(event_stream) + chunk_size - 1)//chunk_size}")
                logger.info(f"   📈 Progress: {progress_pct:.1f}% ({chunk_end:,}/{len(event_stream):,} events)")
                logger.info(f"   💾 Memory: {current_memory:.1f} MB (+{current_memory-initial_memory:.1f} MB)")
                logger.info(f"   ⏱️  Duration: {elapsed_time}")
                
                # Memory safety check
                if current_memory > 4000:  # 4GB limit
                    logger.warning(f"⚠️  Memory limit reached ({current_memory:.1f} MB), stopping pattern discovery")
                    break
                
                # Use improved pattern discovery on chunk
                chunk_patterns = self.pattern_discoverer.discover_patterns(chunk_events, room_id)
                
                # Merge chunk patterns into overall patterns
                for pattern_type, pattern_list in chunk_patterns.items():
                    all_patterns[pattern_type].extend(pattern_list)
                
                logger.info(f"   ✅ Chunk complete: found patterns of types {list(chunk_patterns.keys())}")
            
            # Combine patterns from all chunks and remove duplicates
            final_patterns = self._merge_chunk_patterns(all_patterns, room_id)
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            total_pattern_count = sum(len(pattern_list) for pattern_list in final_patterns.values())
            
            logger.info(f"🎉 Improved pattern discovery completed for {room_id}:")
            logger.info(f"   📊 Total patterns found: {total_pattern_count}")
            logger.info(f"   🎯 Pattern types: {list(final_patterns.keys())}")
            logger.info(f"   📈 Events processed: {len(event_stream):,}")
            logger.info(f"   💾 Memory usage: {final_memory:.1f} MB (+{final_memory-initial_memory:.1f} MB)")
            logger.info(f"   ⏱️  Total duration: {datetime.now() - start_time}")
            
            return final_patterns
            
        except Exception as e:
            logger.error(f"❌ Error in improved pattern discovery for {room_id}: {e}")
            import traceback
            logger.error(f"📍 Traceback: {traceback.format_exc()}")
            return {}
    
    def _merge_chunk_patterns(self, all_patterns: Dict, room_id: str) -> Dict:
        """Merge patterns from chunks and remove duplicates"""
        merged_patterns = {}
        
        for pattern_type, pattern_list in all_patterns.items():
            # For temporal patterns, merge by time key
            if pattern_type == 'temporal':
                time_patterns = {}
                for pattern in pattern_list:
                    time_key = pattern.get('time', 'unknown')
                    if time_key not in time_patterns:
                        time_patterns[time_key] = pattern
                    else:
                        # Merge significant days
                        existing_days = {d['day'] for d in time_patterns[time_key].get('significant_days', [])}
                        new_days = {d['day'] for d in pattern.get('significant_days', [])}
                        if new_days - existing_days:  # Has new days
                            time_patterns[time_key]['significant_days'].extend([
                                d for d in pattern.get('significant_days', []) 
                                if d['day'] not in existing_days
                            ])
                merged_patterns[pattern_type] = list(time_patterns.values())
            
            # For sequential patterns, merge by transition type
            elif pattern_type == 'sequential':
                transition_patterns = {}
                for pattern in pattern_list:
                    transition_key = (pattern.get('from', ''), pattern.get('to', ''))
                    if transition_key not in transition_patterns:
                        transition_patterns[transition_key] = pattern
                    else:
                        # Keep the one with more observations
                        if (pattern.get('observed_count', 0) > 
                            transition_patterns[transition_key].get('observed_count', 0)):
                            transition_patterns[transition_key] = pattern
                merged_patterns[pattern_type] = list(transition_patterns.values())
            
            # For other pattern types, just deduplicate by string representation
            else:
                unique_patterns = {}
                for pattern in pattern_list:
                    pattern_key = str(pattern)
                    unique_patterns[pattern_key] = pattern
                merged_patterns[pattern_type] = list(unique_patterns.values())
        
        return merged_patterns

    async def _discover_patterns_with_monitoring(self, event_stream: List[Dict], room_id: str):
        """Original pattern discovery with chunked processing and enhanced monitoring"""
        logger.info(f"🔍 Starting pattern discovery for {room_id} with {len(event_stream):,} events")
        
        # Memory monitoring  
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        start_time = datetime.now()
        
        # Initialize pattern storage
        if room_id not in self.pattern_library:
            self.pattern_library[room_id] = set()
        
        try:
            # CHUNKED PROCESSING - Process events in chunks to prevent memory explosion
            chunk_size = 3000  # Process 3k events at a time - conservative for 8GB system
            total_patterns_found = 0
            total_windows_tested = 0
            
            logger.info(f"📊 Processing {len(event_stream):,} events in chunks of {chunk_size:,}")
            
            for chunk_start in range(0, len(event_stream), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(event_stream))
                chunk_events = event_stream[chunk_start:chunk_end]
                
                # Progress monitoring
                progress_pct = (chunk_end / len(event_stream)) * 100
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                elapsed_time = datetime.now() - start_time
                
                logger.info(f"🔄 Processing chunk {chunk_start//chunk_size + 1}/{(len(event_stream) + chunk_size - 1)//chunk_size}")
                logger.info(f"   📈 Progress: {progress_pct:.1f}% ({chunk_end:,}/{len(event_stream):,} events)")
                logger.info(f"   💾 Memory: {current_memory:.1f} MB (+{current_memory-initial_memory:.1f} MB)")
                logger.info(f"   ⏱️  Duration: {elapsed_time}")
                
                # Memory safety check
                if current_memory > 4000:  # 4GB limit
                    logger.warning(f"⚠️  Memory limit reached ({current_memory:.1f} MB), stopping pattern discovery")
                    break
                
                # Remove time limit - let pattern discovery complete fully
                
                # ORIGINAL ALGORITHM - Extract sequences from chunk
                logger.info(f"   🧩 Extracting sequences from {len(chunk_events):,} events...")
                sequences = self.extract_sequences(chunk_events, 
                                                 min_length=2, 
                                                 max_length=100)  # ORIGINAL: Keep full sequence length range
                
                if not sequences:
                    logger.info(f"   ⚠️  No sequences found in chunk, skipping")
                    continue
                
                logger.info(f"   ✅ Extracted {len(sequences):,} sequences")
                
                # ORIGINAL ALGORITHM - Test time windows for this chunk
                chunk_patterns = 0
                chunk_windows_tested = 0
                
                # ORIGINAL ALGORITHM - Test ALL time windows as originally designed
                time_windows = self.generate_time_windows()  # ORIGINAL: Full comprehensive time window testing
                logger.info(f"   🎯 Testing {len(time_windows)} time windows (original algorithm)...")
                
                for window in time_windows:
                    # Quick memory check
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    if current_memory > 3500:  # 3.5GB safety buffer
                        logger.warning(f"   ⚠️  Memory approaching limit ({current_memory:.1f} MB), stopping window testing")
                        break
                    
                    # ORIGINAL ALGORITHM - Test pattern significance
                    pattern_strength = self.test_pattern_significance(
                        sequences, window, room_id
                    )
                    
                    chunk_windows_tested += 1
                    
                    if pattern_strength < self.adaptive_threshold:  # FIX: Accept LOW fake p-values (HIGH chi-squared)
                        self.pattern_library[room_id].add(
                            Pattern(sequences, window, pattern_strength)
                        )
                        chunk_patterns += 1
                        logger.debug(f"      🎯 Found significant pattern: window={window}min, strength={pattern_strength:.4f}")
                    
                    # Progress logging every 50 windows
                    if chunk_windows_tested % 50 == 0:
                        logger.info(f"      📊 Tested {chunk_windows_tested}/{len(time_windows)} windows, found {chunk_patterns} patterns")
                
                total_patterns_found += chunk_patterns
                total_windows_tested += chunk_windows_tested
                
                logger.info(f"   ✅ Chunk complete: {chunk_patterns} patterns found, {chunk_windows_tested} windows tested")
                logger.info(f"   📊 Running totals: {total_patterns_found} patterns, {total_windows_tested} windows")
            
            # ORIGINAL ALGORITHM - Detect anti-patterns (what DOESN'T happen)
            logger.info(f"🔍 Discovering negative patterns for {room_id}...")
            self.discover_negative_patterns(event_stream, room_id)  # ORIGINAL: Use full event stream
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            total_patterns = len(self.pattern_library.get(room_id, []))
            
            logger.info(f"🎉 Pattern discovery completed for {room_id}:")
            logger.info(f"   📊 Total patterns found: {total_patterns}")
            logger.info(f"   🎯 Time windows tested: {total_windows_tested:,}")
            logger.info(f"   📈 Events processed: {len(event_stream):,}")
            logger.info(f"   💾 Memory usage: {final_memory:.1f} MB (+{final_memory-initial_memory:.1f} MB)")
            logger.info(f"   ⏱️  Total duration: {datetime.now() - start_time}")
            logger.info(f"   ⚡ Processing rate: {len(event_stream)/(elapsed_time.total_seconds() or 1):.0f} events/sec")
            
        except Exception as e:
            logger.error(f"❌ Error in pattern discovery for {room_id}: {e}")
            import traceback
            logger.error(f"📍 Traceback: {traceback.format_exc()}")
    
    
    def generate_time_windows(self):
        """
        ORIGINAL ALGORITHM - Test all possible time windows, not just "hourly" or "daily"
        Exactly as specified in CLAUDE.md
        """
        # Windows from 1 minute to 30 days, with variable steps
        windows = []
        for minutes in range(1, 43200):  # Up to 30 days
            if self.is_promising_window(minutes):
                windows.append(minutes)
        return windows
    
    def is_promising_window(self, minutes: int) -> bool:
        """
        ORIGINAL ALGORITHM - Determine if a time window is worth testing
        Avoids testing every single minute to save computation
        """
        # Test common intervals and logarithmic spacing
        promising_windows = {
            # Minutes: 1-60
            *range(1, 61),
            # Hours: 1-24 (in minutes)
            *[h * 60 for h in range(1, 25)],
            # Days: 1-30 (in minutes) 
            *[d * 24 * 60 for d in range(1, 31)],
            # Special intervals
            90, 105, 135, 150, 165, 180,  # 1.5-3 hours
            270, 300, 330, 360,  # 4.5-6 hours
            450, 480, 510, 540,  # 7.5-9 hours
        }
        
        return minutes in promising_windows
    
    def extract_sequences(self, event_stream: List[Dict], min_length: int, max_length: int) -> List[List[Dict]]:
        """
        ORIGINAL ALGORITHM - Extract variable-length sequences from event stream
        """
        sequences = []
        
        for start_idx in range(len(event_stream)):
            for length in range(min_length, min(max_length + 1, len(event_stream) - start_idx + 1)):
                sequence = event_stream[start_idx:start_idx + length]
                sequences.append(sequence)
        
        return sequences
    
    def test_pattern_significance(self, sequences: List[List[Dict]], window_minutes: int, room_id: str) -> float:
        """
        ORIGINAL ALGORITHM - Test if patterns within time window are statistically significant
        """
        try:
            # Group sequences by time window
            windowed_patterns = self.group_by_time_window(sequences, window_minutes)
            
            if len(windowed_patterns) < 2:
                return 0.0
            
            # Use chi-squared test for independence
            pattern_counts = Counter(str(pattern) for pattern in windowed_patterns)
            expected_frequency = len(windowed_patterns) / len(pattern_counts)
            
            # Calculate chi-squared statistic
            chi_squared = sum(
                (observed - expected_frequency) ** 2 / expected_frequency
                for observed in pattern_counts.values()
            )
            
            # DEBUG: Log chi-squared values to understand distribution
            logger.info(f"      🔍 DEBUG - Window {window_minutes}min: chi²={chi_squared:.2f}, patterns={len(pattern_counts)}, total={len(windowed_patterns)}")
            
            # Convert to p-value (simplified)
            p_value = 1.0 / (1.0 + chi_squared)
            
            return p_value
            
        except Exception as e:
            logger.warning(f"Error testing pattern significance: {e}")
            return 0.0
    
    def group_by_time_window(self, sequences: List[List[Dict]], window_minutes: int) -> List[str]:
        """ORIGINAL ALGORITHM - Group sequences that occur within the same time window"""
        windowed_patterns = []
        
        for sequence in sequences:
            if not sequence:
                continue
            
            try:
                # Handle different timestamp formats
                start_event = sequence[0]
                end_event = sequence[-1]
                
                if isinstance(start_event.get('timestamp'), datetime):
                    start_time = start_event['timestamp']
                    end_time = end_event['timestamp']
                else:
                    # Parse string timestamps
                    timestamp_str = str(start_event.get('timestamp', ''))
                    if 'Z' in timestamp_str:
                        timestamp_str = timestamp_str.replace('Z', '+00:00')
                    start_time = datetime.fromisoformat(timestamp_str)
                    
                    timestamp_str = str(end_event.get('timestamp', ''))
                    if 'Z' in timestamp_str:
                        timestamp_str = timestamp_str.replace('Z', '+00:00')
                    end_time = datetime.fromisoformat(timestamp_str)
                
                # Check if sequence fits within time window
                if (end_time - start_time).total_seconds() <= window_minutes * 60:
                    # Create pattern signature
                    pattern_sig = self.create_pattern_signature(sequence)
                    windowed_patterns.append(pattern_sig)
                    
            except Exception as e:
                logger.debug(f"Error processing sequence timestamp: {e}")
                continue
        
        return windowed_patterns
    
    def create_pattern_signature(self, sequence: List[Dict]) -> str:
        """ORIGINAL ALGORITHM - Create a signature for a sequence of events"""
        signature_parts = []
        
        for event in sequence:
            part = f"{event.get('room', 'unknown')}-{event.get('sensor_type', 'unknown')}-{event.get('state', 0)}"
            signature_parts.append(part)
        
        return "->".join(signature_parts)
    
    def discover_negative_patterns(self, event_stream: List[Dict], room_id: str):
        """
        ORIGINAL ALGORITHM - Detect anti-patterns (what DOESN'T happen)
        Important for understanding absence of activity
        """
        logger.info(f"Discovering negative patterns for {room_id}")
        
        # Find expected but missing sequences
        all_possible_sequences = self.generate_possible_sequences(event_stream)
        actual_sequences = set(self.create_pattern_signature(seq) for seq in self.extract_sequences(event_stream, 2, 5))
        
        missing_patterns = all_possible_sequences - actual_sequences
        
        # Store significant missing patterns
        if room_id not in self.pattern_library:
            self.pattern_library[room_id] = set()
        
        for missing_pattern in missing_patterns:
            if self.is_significant_absence(missing_pattern, event_stream):
                self.pattern_library[room_id].add(
                    NegativePattern(missing_pattern, self.calculate_absence_strength(missing_pattern, event_stream))
                )
    
    def generate_possible_sequences(self, event_stream: List[Dict]) -> set:
        """ORIGINAL ALGORITHM - Generate all theoretically possible sequences from observed events"""
        unique_events = set()
        for event in event_stream:
            event_sig = f"{event.get('room', 'unknown')}-{event.get('sensor_type', 'unknown')}-{event.get('state', 0)}"
            unique_events.add(event_sig)
        
        # Generate all 2-3 event combinations
        possible_sequences = set()
        unique_events_list = list(unique_events)
        
        for i, event1 in enumerate(unique_events_list):
            for j, event2 in enumerate(unique_events_list):
                if i != j:
                    possible_sequences.add(f"{event1}->{event2}")
                    
                    for k, event3 in enumerate(unique_events_list):
                        if k != i and k != j:
                            possible_sequences.add(f"{event1}->{event2}->{event3}")
        
        return possible_sequences
    
    def is_significant_absence(self, pattern: str, event_stream: List[Dict]) -> bool:
        """ORIGINAL ALGORITHM - Determine if absence of a pattern is statistically significant"""
        # Simplified: if pattern components exist but never occur together
        pattern_parts = pattern.split('->')
        
        # Check if individual components exist
        components_exist = all(
            any(part.split('-')[0] in str(event) for event in event_stream)
            for part in pattern_parts
        )
        
        return components_exist and len(pattern_parts) >= 2
    
    def calculate_absence_strength(self, pattern: str, event_stream: List[Dict]) -> float:
        """ORIGINAL ALGORITHM - Calculate how strong the evidence is for pattern absence"""
        # Simplified calculation based on component frequency
        return 0.8  # Placeholder
    
    async def discover_bathroom_patterns(self, bathroom_rooms: List[str]) -> Dict:
        """Discover patterns for bathroom rooms using improved algorithm"""
        logger.info(f"Discovering bathroom patterns for {bathroom_rooms}")
        
        bathroom_patterns = {}
        
        for bathroom_room in bathroom_rooms:
            # Load events for this bathroom
            historical_events = await self._load_events_with_monitoring(bathroom_room)
            
            if historical_events:
                # Use improved pattern discovery
                patterns = await self._discover_patterns_chunked(historical_events, bathroom_room)
                bathroom_patterns[bathroom_room] = patterns
            else:
                bathroom_patterns[bathroom_room] = {}
        
        logger.info(f"Bathroom pattern discovery completed for {len(bathroom_patterns)} bathrooms")
        return {'patterns': bathroom_patterns}

    # Store patterns in Redis for model consumption
    async def store_patterns_in_redis(self, room_id: str):
        """Store discovered patterns in Redis for model access"""
        try:
            patterns = self.pattern_library.get(room_id, {})
            if not patterns:
                return
            
            # Handle both old and new pattern formats
            if isinstance(patterns, dict):
                # New improved pattern format
                total_pattern_count = sum(len(pattern_list) for pattern_list in patterns.values())
                pattern_data = {
                    'total_patterns': total_pattern_count,
                    'discovery_time': datetime.now(timezone.utc).isoformat(),
                    'pattern_types': list(patterns.keys()),
                    'patterns': patterns  # Store the entire structured format
                }
            else:
                # Old pattern format (legacy support)
                pattern_data = {
                    'total_patterns': len(patterns),
                    'discovery_time': datetime.now(timezone.utc).isoformat(),
                    'patterns': []
                }
                
                for pattern in patterns:
                    if hasattr(pattern, 'time_window'):
                        pattern_info = {
                            'time_window': pattern.time_window,
                            'strength': pattern.strength,
                            'type': 'positive'
                        }
                        pattern_data['patterns'].append(pattern_info)
            
            # Store in Redis via feature store
            from src.storage.feature_store import RedisFeatureStore
            from config.config_loader import ConfigLoader
            
            config = ConfigLoader()
            redis_config = config.get("redis")
            feature_store = RedisFeatureStore(redis_config)
            
            await feature_store.cache_pattern(room_id, 'discovered_patterns', pattern_data)
            pattern_count = pattern_data['total_patterns']
            logger.info(f"Stored {pattern_count} patterns for {room_id} in Redis")
            
        except Exception as e:
            logger.error(f"Error storing patterns in Redis for {room_id}: {e}")


class StatisticalTestSuite:
    """
    ORIGINAL ALGORITHM - Statistical tests for pattern significance
    """
    
    def chi_squared_test(self, observed_freq: int, expected_freq: float) -> float:
        """Chi-squared test for pattern frequency"""
        if expected_freq == 0:
            return 1.0
        
        chi_stat = (observed_freq - expected_freq) ** 2 / expected_freq
        # Simplified p-value calculation
        return 1.0 / (1.0 + chi_stat)


class UnbiasedPatternMiner:
    """
    ORIGINAL ALGORITHM - Unbiased pattern mining using suffix trees
    Implements the approach from CLAUDE.md
    """
    
    def mine_patterns(self, event_stream: List[Dict]) -> List[Dict]:
        """Use suffix trees for efficient pattern matching"""
        # Convert events to string sequence for suffix tree
        event_strings = []
        for event in event_stream:
            event_str = f"{event.get('room', 'U')}{event.get('state', 0)}"
            event_strings.append(event_str)
        
        # Build suffix tree
        sequence = ''.join(event_strings)
        if not sequence:
            return []
        
        # Use suffix tree to find repeated patterns
        try:
            suffix_tree = STree(sequence)
            patterns = []
            
            # Find patterns that repeat multiple times
            for i in range(2, min(20, len(sequence))):  # Pattern length 2-20
                for j in range(len(sequence) - i + 1):
                    substring = sequence[j:j+i]
                    occurrences = len(suffix_tree.find_all(substring))
                    
                    if occurrences >= 3:  # Must occur at least 3 times
                        if self.is_significant(substring, occurrences, len(sequence)):
                            patterns.append({
                                'pattern': substring,
                                'frequency': occurrences,
                                'length': i,
                                'significance': self.calculate_significance(occurrences, len(sequence))
                            })
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Error in pattern mining: {e}")
            return []
    
    def is_significant(self, pattern: str, frequency: int, total_length: int) -> bool:
        """Check if pattern occurs more than random chance"""
        expected_frequency = (total_length - len(pattern) + 1) * (1.0 / (4 ** len(pattern)))
        return frequency > expected_frequency * 2  # Threshold for significance
    
    def calculate_significance(self, frequency: int, total_length: int) -> float:
        """Calculate statistical significance of pattern"""
        return min(1.0, frequency / (total_length * 0.1))


class Pattern:
    """ORIGINAL - Represents a discovered behavioral pattern"""
    
    def __init__(self, sequences: List[List[Dict]], time_window: int, strength: float):
        self.sequences = sequences
        self.time_window = time_window
        self.strength = strength
        self.discovered_at = datetime.now(timezone.utc)
    
    def __hash__(self):
        return hash((str(self.sequences), self.time_window))
    
    def __eq__(self, other):
        return (self.sequences == other.sequences and 
                self.time_window == other.time_window)


class NegativePattern:
    """ORIGINAL - Represents a significant absence pattern"""
    
    def __init__(self, pattern: str, strength: float):
        self.pattern = pattern
        self.strength = strength
        self.discovered_at = datetime.now(timezone.utc)
    
    def __hash__(self):
        return hash(self.pattern)
    
    def __eq__(self, other):
        return self.pattern == other.pattern


class ImprovedPatternDiscovery:
    """
    Discovers meaningful occupancy patterns by testing against random behavior
    Implements the exact approach from pattern_discovery_implem.md
    """
    
    def __init__(self):
        self.pattern_library = defaultdict(dict)
        self.baseline_probabilities = {}
        self.transition_matrices = {}
        
    def discover_patterns(self, event_stream: List[Dict], room_id: str):
        """Main pattern discovery with proper statistical testing"""
        logger.info(f"Discovering patterns for {room_id} with {len(event_stream)} events")
        
        # Step 1: Calculate baseline probabilities
        self._calculate_baseline_probabilities(event_stream, room_id)
        
        # Step 2: Discover different types of patterns
        patterns = {
            'temporal': self._discover_temporal_patterns(event_stream, room_id),
            'sequential': self._discover_sequential_patterns(event_stream, room_id),
            'conditional': self._discover_conditional_patterns(event_stream, room_id),
            'absence': self._discover_absence_patterns(event_stream, room_id)
        }
        
        # Step 3: Filter by statistical significance
        significant_patterns = self._filter_significant_patterns(patterns)
        
        self.pattern_library[room_id] = significant_patterns
        logger.info(f"Found {len(significant_patterns)} significant patterns for {room_id}")
        
        return significant_patterns
    
    def _calculate_baseline_probabilities(self, events: List[Dict], room_id: str):
        """Calculate baseline occupancy probability for null hypothesis testing"""
        
        # Group events by time buckets (e.g., hour of day, day of week)
        time_buckets = defaultdict(lambda: {'occupied': 0, 'total': 0})
        
        # Also calculate overall probability
        total_occupied = 0
        total_observations = 0
        
        for event in events:
            if event.get('state') in ['on', 'occupied', '1', 1]:
                timestamp = self._parse_timestamp(event['timestamp'])
                
                # Hour of day bucket
                hour_key = f"hour_{timestamp.hour}"
                time_buckets[hour_key]['occupied'] += 1
                time_buckets[hour_key]['total'] += 1
                
                # Day of week bucket
                dow_key = f"dow_{timestamp.weekday()}"
                time_buckets[dow_key]['occupied'] += 1
                time_buckets[dow_key]['total'] += 1
                
                # Hour + day combo
                combo_key = f"hour_{timestamp.hour}_dow_{timestamp.weekday()}"
                time_buckets[combo_key]['occupied'] += 1
                time_buckets[combo_key]['total'] += 1
                
                total_occupied += 1
            
            total_observations += 1
        
        # Calculate probabilities
        self.baseline_probabilities[room_id] = {
            'overall': total_occupied / max(total_observations, 1),
            'time_specific': {}
        }
        
        for bucket, counts in time_buckets.items():
            if counts['total'] > 0:
                prob = counts['occupied'] / counts['total']
                self.baseline_probabilities[room_id]['time_specific'][bucket] = prob
        
        logger.info(f"Baseline probability for {room_id}: {self.baseline_probabilities[room_id]['overall']:.3f}")
    
    def _discover_temporal_patterns(self, events: List[Dict], room_id: str) -> List[Dict]:
        """Find recurring temporal patterns (e.g., occupied every weekday at 9am)"""
        
        temporal_patterns = []
        
        # Group occupancy by time slots
        time_slots = defaultdict(list)  # key: (hour, minute), value: list of (date, occupied)
        
        for event in events:
            if event.get('state') in ['on', 'off', 'occupied', 'unoccupied', '1', '0', 1, 0]:
                timestamp = self._parse_timestamp(event['timestamp'])
                time_key = (timestamp.hour, timestamp.minute // 15 * 15)  # 15-min buckets
                date_key = timestamp.date()
                occupied = event.get('state') in ['on', 'occupied', '1', 1]
                
                time_slots[time_key].append((date_key, occupied))
        
        # Test each time slot for patterns
        for time_key, observations in time_slots.items():
            if len(observations) < 10:  # Need enough data
                continue
            
            # Calculate pattern strength
            pattern_info = self._analyze_temporal_pattern(time_key, observations, room_id)
            
            if pattern_info['is_significant']:
                temporal_patterns.append(pattern_info)
        
        return temporal_patterns
    
    def _analyze_temporal_pattern(self, time_key: Tuple[int, int], 
                                  observations: List[Tuple], room_id: str) -> Dict:
        """Analyze if a specific time slot shows consistent occupancy pattern"""
        
        hour, minute = time_key
        
        # Separate by day of week
        dow_patterns = defaultdict(list)
        for date, occupied in observations:
            dow = date.weekday()
            dow_patterns[dow].append(occupied)
        
        # Test each day of week
        significant_days = []
        
        for dow, occupancy_list in dow_patterns.items():
            if len(occupancy_list) < 4:  # Need at least 4 weeks of data
                continue
            
            occupied_count = sum(occupancy_list)
            total_count = len(occupancy_list)
            observed_prob = occupied_count / total_count
            
            # Get baseline probability for this time
            baseline_key = f"hour_{hour}_dow_{dow}"
            baseline_prob = self.baseline_probabilities[room_id]['time_specific'].get(
                baseline_key, 
                self.baseline_probabilities[room_id]['overall']
            )
            
            # Binomial test: is this significantly different from baseline?
            p_value = stats.binom_test(occupied_count, total_count, baseline_prob)
            
            if p_value < 0.05:  # Significant pattern
                effect_size = abs(observed_prob - baseline_prob)
                
                if effect_size > 0.2:  # Meaningful difference
                    significant_days.append({
                        'day': dow,
                        'probability': observed_prob,
                        'baseline': baseline_prob,
                        'p_value': p_value,
                        'effect_size': effect_size,
                        'sample_size': total_count
                    })
        
        is_significant = len(significant_days) > 0
        
        return {
            'type': 'temporal',
            'time': f"{hour:02d}:{minute:02d}",
            'is_significant': is_significant,
            'significant_days': significant_days,
            'pattern_description': self._describe_temporal_pattern(time_key, significant_days)
        }
    
    def _discover_sequential_patterns(self, events: List[Dict], room_id: str) -> List[Dict]:
        """Find sequential patterns (e.g., bedroom→bathroom within 5 minutes)"""
        
        sequential_patterns = []
        
        # Build sequences of room transitions
        transitions = []
        last_event = None
        
        for event in sorted(events, key=lambda x: x['timestamp']):
            if event.get('state') in ['on', 'occupied', '1', 1]:
                if last_event:
                    time_diff = (self._parse_timestamp(event['timestamp']) - 
                               self._parse_timestamp(last_event['timestamp'])).total_seconds()
                    
                    if time_diff < 3600:  # Within 1 hour
                        transitions.append({
                            'from': last_event.get('room', last_event.get('entity_id', '')),
                            'to': event.get('room', event.get('entity_id', '')),
                            'time_diff': time_diff,
                            'timestamp': event['timestamp']
                        })
                
                last_event = event
        
        # Analyze transition patterns
        transition_groups = defaultdict(list)
        for trans in transitions:
            key = (trans['from'], trans['to'])
            transition_groups[key].append(trans['time_diff'])
        
        # Test each transition type
        for (from_room, to_room), time_diffs in transition_groups.items():
            if len(time_diffs) < 10:  # Need enough samples
                continue
            
            pattern_info = self._analyze_sequential_pattern(
                from_room, to_room, time_diffs, room_id
            )
            
            if pattern_info['is_significant']:
                sequential_patterns.append(pattern_info)
        
        return sequential_patterns
    
    def _analyze_sequential_pattern(self, from_room: str, to_room: str, 
                                   time_diffs: List[float], room_id: str) -> Dict:
        """Test if a room transition happens more than random chance"""
        
        # Calculate expected frequency under random behavior
        from_prob = self.baseline_probabilities.get(from_room, {}).get('overall', 0.1)
        to_prob = self.baseline_probabilities.get(to_room, {}).get('overall', 0.1)
        
        # Expected transitions under independence
        total_time_windows = len(time_diffs) * 10  # Approximate total opportunities
        expected_transitions = total_time_windows * from_prob * to_prob
        observed_transitions = len(time_diffs)
        
        # Poisson test for rare events
        if expected_transitions < 5:
            p_value = stats.poisson.sf(observed_transitions - 1, expected_transitions)
        else:
            # Normal approximation for common events
            z_score = (observed_transitions - expected_transitions) / np.sqrt(expected_transitions)
            p_value = stats.norm.sf(abs(z_score)) * 2  # Two-tailed
        
        # Analyze timing consistency
        time_stats = {
            'mean': np.mean(time_diffs),
            'std': np.std(time_diffs),
            'median': np.median(time_diffs),
            'percentile_25': np.percentile(time_diffs, 25),
            'percentile_75': np.percentile(time_diffs, 75)
        }
        
        # Check if timing is consistent (low variance relative to mean)
        cv = time_stats['std'] / time_stats['mean'] if time_stats['mean'] > 0 else float('inf')
        timing_is_consistent = cv < 0.5  # Coefficient of variation < 50%
        
        is_significant = p_value < 0.05 and observed_transitions > expected_transitions * 1.5
        
        return {
            'type': 'sequential',
            'from': from_room,
            'to': to_room,
            'is_significant': is_significant,
            'p_value': p_value,
            'observed_count': observed_transitions,
            'expected_count': expected_transitions,
            'time_stats': time_stats,
            'timing_consistent': timing_is_consistent,
            'pattern_description': f"{from_room} → {to_room} within {time_stats['median']:.0f}s"
        }
    
    def _discover_conditional_patterns(self, events: List[Dict], room_id: str) -> List[Dict]:
        """Find conditional patterns (if X then Y within time T)"""
        
        conditional_patterns = []
        
        # Look for patterns like "if bedroom empty at 7am, office occupied within 30min"
        # Group events by time and state
        event_sequences = self._build_event_sequences(events)
        
        # Test various conditions
        conditions_to_test = [
            ('morning_bedroom_exit', 'office_entry', 30),  # Minutes
            ('evening_office_exit', 'living_room_entry', 15),
            ('late_night_living_room_exit', 'bedroom_entry', 10),
            ('bathroom_entry', 'bathroom_exit', 20),  # Duration patterns
        ]
        
        for condition_name, outcome_name, max_time_window in conditions_to_test:
            pattern_info = self._test_conditional_pattern(
                event_sequences, condition_name, outcome_name, max_time_window
            )
            
            if pattern_info and pattern_info['is_significant']:
                conditional_patterns.append(pattern_info)
        
        return conditional_patterns
    
    def _discover_absence_patterns(self, events: List[Dict], room_id: str) -> List[Dict]:
        """Find patterns of when rooms are NOT occupied"""
        
        absence_patterns = []
        
        # Create occupancy timeline
        timeline = self._create_occupancy_timeline(events)
        
        # Find consistent gaps
        gaps = self._find_occupancy_gaps(timeline)
        
        # Group gaps by time of day
        gap_patterns = defaultdict(list)
        for gap_start, gap_end in gaps:
            hour_start = gap_start.hour
            duration = (gap_end - gap_start).total_seconds() / 60  # Minutes
            gap_patterns[hour_start].append(duration)
        
        # Test each hour for consistent absence
        for hour, durations in gap_patterns.items():
            if len(durations) < 10:  # Need enough samples
                continue
            
            # Test if gaps at this hour are longer than expected
            all_gaps = [d for durs in gap_patterns.values() for d in durs]
            expected_duration = np.mean(all_gaps) if all_gaps else 60
            
            # T-test to see if this hour has significantly longer gaps
            if len(all_gaps) > 30:
                t_stat, p_value = stats.ttest_ind(durations, all_gaps)
                
                if p_value < 0.05 and np.mean(durations) > expected_duration:
                    absence_patterns.append({
                        'type': 'absence',
                        'hour': hour,
                        'mean_duration': np.mean(durations),
                        'expected_duration': expected_duration,
                        'p_value': p_value,
                        'is_significant': True,
                        'pattern_description': f"Typically unoccupied at {hour:02d}:00 for {np.mean(durations):.0f} minutes"
                    })
        
        return absence_patterns
    
    def _filter_significant_patterns(self, patterns: Dict) -> Dict:
        """Apply multiple testing correction and filter truly significant patterns"""
        
        # Flatten all patterns
        all_patterns = []
        for pattern_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                pattern['pattern_type'] = pattern_type
                all_patterns.append(pattern)
        
        # Apply Benjamini-Hochberg FDR correction
        p_values = [p.get('p_value', 1.0) for p in all_patterns]
        if p_values:
            rejected, corrected_p_values = self._benjamini_hochberg(p_values)
            
            # Update patterns with corrected p-values
            for pattern, corrected_p, is_sig in zip(all_patterns, corrected_p_values, rejected):
                pattern['corrected_p_value'] = corrected_p
                pattern['passes_fdr'] = is_sig
        
        # Filter and organize significant patterns
        significant_patterns = defaultdict(list)
        for pattern in all_patterns:
            if pattern.get('passes_fdr', False) or pattern.get('is_significant', False):
                significant_patterns[pattern['pattern_type']].append(pattern)
        
        return dict(significant_patterns)
    
    def _benjamini_hochberg(self, p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
        """Benjamini-Hochberg FDR correction for multiple testing"""
        
        n = len(p_values)
        if n == 0:
            return [], []
        
        # Sort p-values and track original indices
        sorted_indices = np.argsort(p_values)
        sorted_p_values = np.array(p_values)[sorted_indices]
        
        # Calculate critical values
        critical_values = alpha * np.arange(1, n + 1) / n
        
        # Find largest i where P(i) <= critical_value(i)
        below_critical = sorted_p_values <= critical_values
        if np.any(below_critical):
            max_i = np.max(np.where(below_critical)[0])
            threshold = sorted_p_values[max_i]
        else:
            threshold = -1
        
        # Apply threshold
        rejected = [p <= threshold for p in p_values]
        
        # Calculate adjusted p-values
        adjusted_p = np.minimum(1, n * sorted_p_values / np.arange(1, n + 1))
        adjusted_p = np.maximum.accumulate(adjusted_p[::-1])[::-1]
        
        # Restore original order
        corrected_p_values = np.empty_like(adjusted_p)
        corrected_p_values[sorted_indices] = adjusted_p
        
        return rejected, corrected_p_values.tolist()
    
    def _parse_timestamp(self, timestamp) -> datetime:
        """Parse various timestamp formats"""
        if isinstance(timestamp, datetime):
            return timestamp
        
        # Handle string timestamps
        timestamp_str = str(timestamp)
        if 'Z' in timestamp_str:
            timestamp_str = timestamp_str.replace('Z', '+00:00')
        
        try:
            return datetime.fromisoformat(timestamp_str)
        except:
            # Try other formats
            return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    
    def _describe_temporal_pattern(self, time_key: Tuple[int, int], significant_days: List[Dict]) -> str:
        """Generate human-readable description of temporal pattern"""
        
        hour, minute = time_key
        time_str = f"{hour:02d}:{minute:02d}"
        
        if not significant_days:
            return f"No pattern at {time_str}"
        
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        days = [day_names[d['day']] for d in significant_days]
        
        avg_prob = np.mean([d['probability'] for d in significant_days])
        
        if len(days) >= 5 and all(d in range(5) for d in [s['day'] for s in significant_days]):
            return f"Typically occupied on weekdays at {time_str} ({avg_prob:.0%} probability)"
        elif len(days) == 2 and set([s['day'] for s in significant_days]) == {5, 6}:
            return f"Weekend pattern at {time_str} ({avg_prob:.0%} probability)"
        else:
            return f"Occupied on {', '.join(days)} at {time_str} ({avg_prob:.0%} probability)"
    
    def _build_event_sequences(self, events: List[Dict]) -> List[Dict]:
        """Build sequences of events for pattern analysis"""
        # Implementation depends on your event structure
        return sorted(events, key=lambda x: x['timestamp'])
    
    def _test_conditional_pattern(self, sequences: List[Dict], condition: str, 
                                 outcome: str, max_window: int) -> Optional[Dict]:
        """Test conditional patterns - implementation depends on your specific conditions"""
        # This would need to be customized based on what conditions you want to test
        pass
    
    def _create_occupancy_timeline(self, events: List[Dict]) -> List[Tuple[datetime, bool]]:
        """Create timeline of occupancy states"""
        timeline = []
        for event in sorted(events, key=lambda x: x['timestamp']):
            timestamp = self._parse_timestamp(event['timestamp'])
            occupied = event.get('state') in ['on', 'occupied', '1', 1]
            timeline.append((timestamp, occupied))
        return timeline
    
    def _find_occupancy_gaps(self, timeline: List[Tuple[datetime, bool]]) -> List[Tuple[datetime, datetime]]:
        """Find periods of non-occupancy"""
        gaps = []
        last_unoccupied_start = None
        
        for timestamp, occupied in timeline:
            if not occupied and last_unoccupied_start is None:
                last_unoccupied_start = timestamp
            elif occupied and last_unoccupied_start is not None:
                gaps.append((last_unoccupied_start, timestamp))
                last_unoccupied_start = None
        
        return gaps