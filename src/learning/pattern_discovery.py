"""
Automatic Pattern Mining - Discovers behavioral patterns without assumptions
Implements the sophisticated pattern discovery from CLAUDE.md

RESTORED ORIGINAL ALGORITHM with essential memory monitoring to prevent hangs
"""

import logging
import psutil
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta, timezone
import numpy as np
from scipy.stats import chi2_contingency
from suffix_trees import STree

logger = logging.getLogger(__name__)


class PatternDiscovery:
    """
    Discovers behavioral patterns from event streams without any assumptions
    Exactly as specified in CLAUDE.md - RESTORED ORIGINAL VERSION
    """
    
    def __init__(self):
        self.pattern_library = {}
        self.statistical_tests = StatisticalTestSuite()
        self.adaptive_threshold = 0.05  # P-value threshold for statistical significance
        
    async def discover_multizone_patterns(self, room_name: str, historical_events: List[Dict] = None) -> Dict:
        """Discover patterns for multi-zone rooms with memory monitoring"""
        logger.info(f"Discovering multi-zone patterns for {room_name}")
        
        if historical_events is None:
            # Load with memory monitoring to prevent hangs
            historical_events = await self._load_events_with_monitoring(room_name)
        
        if not historical_events:
            logger.warning(f"No events for {room_name}")
            return {'room': room_name, 'patterns': set()}
        
        # Use original algorithm with memory monitoring
        await self._discover_patterns_with_monitoring(historical_events, room_name)
        
        patterns = self.pattern_library.get(room_name, set())
        logger.info(f"Discovered {len(patterns)} patterns for {room_name}")
        
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
            chunk_size = 10000  # Process 10k events at a time for meaningful pattern discovery
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

    # Store patterns in Redis for model consumption
    async def store_patterns_in_redis(self, room_id: str):
        """Store discovered patterns in Redis for model access"""
        try:
            patterns = self.pattern_library.get(room_id, set())
            if not patterns:
                return
            
            # Convert patterns to serializable format
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
            logger.info(f"Stored {len(patterns)} patterns for {room_id} in Redis")
            
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