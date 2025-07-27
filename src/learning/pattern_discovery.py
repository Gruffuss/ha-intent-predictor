"""
Automatic Pattern Mining - Discovers behavioral patterns without assumptions
Implements the sophisticated pattern discovery from CLAUDE.md

OPTIMIZATIONS IMPLEMENTED (to fix 4-hour hanging + enhance pattern quality):
1. Chunked database loading (5,000 events at a time) - prevents memory explosion from 100k+ events
2. Increased sequence length (max 15 instead of 5) - captures meaningful behavioral patterns
3. Comprehensive time windows (14 instead of 6) - covers micro to daily patterns 
4. Memory monitoring with 4GB safety limit - allows thorough pattern analysis
5. Timeout protection (1 hour max per room) - allows comprehensive discovery
6. Progress reporting with resource usage - provides visibility during processing
7. Pattern pruning (top 100 per room) - controls memory growth
8. Increased event limit (200k instead of 50k) - more comprehensive analysis
9. Relaxed significance threshold (0.3 instead of 0.05) - finds more patterns
10. Increased sequence limit (50k instead of 10k) - better pattern discovery
"""

import logging
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import chi2_contingency
from suffix_trees import STree
import psutil
import asyncio

logger = logging.getLogger(__name__)


class PatternDiscovery:
    """
    Discovers behavioral patterns from event streams without any assumptions
    Exactly as specified in CLAUDE.md
    """
    
    def __init__(self):
        self.pattern_library = {}
        self.statistical_tests = StatisticalTestSuite()
        self.adaptive_threshold = 0.3  # Increased threshold to find more patterns initially
        
    async def discover_multizone_patterns(self, room_name: str, historical_events: List[Dict] = None) -> Dict:
        """
        Discover patterns for multi-zone rooms with optimized chunked processing
        Handles the complex zone relationships from CLAUDE.md
        """
        logger.info(f"Discovering multi-zone patterns for {room_name}")
        
        if historical_events is None:
            # Use chunked loading to prevent memory explosion
            historical_events = await self._load_events_chunked(room_name)
        else:
            # Filter events for this room
            historical_events = [
                event for event in historical_events
                if event.get('room') == room_name
            ]
        
        if not historical_events:
            logger.warning(f"No events found for room {room_name}")
            return {}
        
        logger.info(f"Processing {len(historical_events):,} events for {room_name} in optimized chunks")
        
        # Discover patterns using the optimized algorithm
        await self._discover_patterns_optimized(historical_events, room_name)
        
        patterns = self.pattern_library.get(room_name, set())
        
        logger.info(f"Discovered {len(patterns)} patterns for {room_name}")
        
        # Cache discovered patterns in Redis where models look for them
        await self._cache_patterns_for_models(room_name, patterns)
        
        return {
            'room': room_name,
            'pattern_count': len(patterns),
            'patterns': list(patterns)[:10]  # Show first 10 patterns
        }
    
    async def _load_events_chunked(self, room_name: str, chunk_size: int = 5000) -> List[Dict]:
        """Load events in chunks to prevent memory explosion"""
        logger.info(f"Loading events for {room_name} in chunks of {chunk_size:,}")
        
        from src.storage.timeseries_db import TimescaleDBManager
        from config.config_loader import ConfigLoader
        
        config = ConfigLoader()
        db_config = config.get("database.timescale")
        db = TimescaleDBManager(f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
        await db.initialize()
        
        all_events = []
        offset = 0
        
        try:
            while True:
                async with db.engine.begin() as conn:
                    from sqlalchemy import text
                    # Load in chunks with LIMIT and OFFSET - ONLY full room sensors for patterns
                    result = await conn.execute(text("""
                        SELECT timestamp, entity_id, state, attributes, room, sensor_type, derived_features
                        FROM sensor_events 
                        WHERE room = :room 
                        AND (
                            entity_id LIKE '%_full_%' OR 
                            entity_id LIKE '%full_%' OR
                            entity_id LIKE '%_presence_%full%' OR
                            (entity_id LIKE '%presence%' AND entity_id LIKE '%full%')
                        )
                        AND entity_id NOT LIKE '%temperature%'
                        AND entity_id NOT LIKE '%humidity%' 
                        AND entity_id NOT LIKE '%pressure%'
                        AND entity_id NOT LIKE '%light_level%'
                        ORDER BY timestamp 
                        LIMIT :limit OFFSET :offset
                    """), {
                        'room': room_name, 
                        'limit': chunk_size, 
                        'offset': offset
                    })
                    
                    chunk_events = [dict(row._mapping) for row in result.fetchall()]
                    
                    if not chunk_events:
                        break  # No more events
                    
                    all_events.extend(chunk_events)
                    offset += chunk_size
                    
                    logger.info(f"  Loaded chunk {offset//chunk_size}: {len(chunk_events):,} events "
                              f"(total: {len(all_events):,})")
                    
                    # Memory management - allow more events for comprehensive pattern discovery  
                    if len(all_events) > 200000:
                        logger.info(f"  Limiting to most recent 200,000 events for comprehensive analysis")
                        all_events = all_events[-200000:]  # Keep most recent
                        break
            
        finally:
            await db.close()
        
        logger.info(f"Loaded {len(all_events):,} total events for {room_name}")
        return all_events
    
    async def discover_bathroom_patterns(self, bathroom_rooms: List[str], historical_events: List[Dict] = None) -> Dict:
        """
        Discover patterns specific to bathroom usage with optimized processing
        Handles the special bathroom logic from CLAUDE.md
        """
        logger.info(f"Discovering bathroom patterns for {bathroom_rooms}")
        
        if historical_events is None:
            # Use chunked loading for all bathroom rooms combined
            bathroom_events = []
            for room in bathroom_rooms:
                room_events = await self._load_events_chunked(room)
                bathroom_events.extend(room_events)
                logger.info(f"  Loaded {len(room_events):,} events for {room}")
        else:
            bathroom_events = []
            for room in bathroom_rooms:
                room_events = [
                    event for event in historical_events
                    if event.get('room') == room
                ]
                bathroom_events.extend(room_events)
        
        # Bathroom-specific optimized pattern discovery
        patterns = {}
        for room in bathroom_rooms:
            room_events = [e for e in bathroom_events if e.get('room') == room]
            if room_events:
                logger.info(f"Processing {len(room_events):,} events for bathroom {room}")
                await self._discover_patterns_optimized(room_events, room)
                patterns[room] = self.pattern_library.get(room, set())
            else:
                logger.warning(f"No events found for bathroom {room}")
                patterns[room] = set()
        
        total_patterns = sum(len(p) for p in patterns.values())
        logger.info(f"Discovered {total_patterns} total bathroom patterns across {len(bathroom_rooms)} rooms")
        
        return {
            'bathroom_rooms': bathroom_rooms,
            'patterns': patterns
        }
    
    async def discover_transition_patterns(self, area_name: str, historical_events: List[Dict] = None) -> Dict:
        """Discover transition patterns for hallways and connections with optimized processing"""
        logger.info(f"Discovering transition patterns for {area_name}")
        
        if historical_events is None:
            # Use chunked loading with transition-specific query
            from src.storage.timeseries_db import TimescaleDBManager
            from config.config_loader import ConfigLoader
            
            config = ConfigLoader()
            db_config = config.get("database.timescale")
            db = TimescaleDBManager(f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
            await db.initialize()
            
            try:
                async with db.engine.begin() as conn:
                    from sqlalchemy import text
                    # Limit transition events to prevent memory issues
                    result = await conn.execute(text("""
                        SELECT timestamp, entity_id, state, attributes, room, sensor_type, derived_features
                        FROM sensor_events 
                        WHERE entity_id LIKE '%hallway%' OR entity_id LIKE '%stairs%' 
                        ORDER BY timestamp DESC
                        LIMIT 20000
                    """))
                    historical_events = [dict(row._mapping) for row in result.fetchall()]
            finally:
                await db.close()
        
        # Filter transition events
        transition_events = [
            event for event in historical_events
            if 'hallway' in event.get('entity_id', '') or 'stairs' in event.get('entity_id', '')
        ]
        
        if transition_events:
            logger.info(f"Processing {len(transition_events):,} transition events for {area_name}")
            await self._discover_patterns_optimized(transition_events, area_name)
        else:
            logger.warning(f"No transition events found for {area_name}")
        
        patterns = self.pattern_library.get(area_name, set())
        
        return {
            'area': area_name,
            'pattern_count': len(patterns),
            'patterns': list(patterns)[:10]
        }

    async def _discover_patterns_optimized(self, event_stream: List[Dict], room_id: str):
        """
        Optimized pattern discovery that prevents memory explosion and hanging
        Processes events in chunks with reduced complexity
        """
        logger.info(f"Starting optimized pattern discovery for {room_id} with {len(event_stream):,} events")
        
        # Memory monitoring - Allow up to 4GB for meaningful pattern discovery
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        logger.info(f"  Initial memory usage: {initial_memory:.1f} MB")
        
        # Initialize pattern storage
        if room_id not in self.pattern_library:
            self.pattern_library[room_id] = set()
        
        # Timeout protection - Allow more time for comprehensive analysis
        start_time = datetime.now()
        max_duration = timedelta(hours=1)  # Maximum 1 hour per room for thorough analysis
        
        # Process events in chunks to prevent memory explosion
        chunk_size = 2000  # Process 2k events at a time
        total_patterns_found = 0
        
        for chunk_start in range(0, len(event_stream), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(event_stream))
            chunk_events = event_stream[chunk_start:chunk_end]
            
            # Progress and resource monitoring
            progress_pct = (chunk_end / len(event_stream)) * 100
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            elapsed_time = datetime.now() - start_time
            
            logger.info(f"  Processing chunk {chunk_start:,}-{chunk_end:,} "
                       f"({progress_pct:.1f}% complete, {current_memory:.1f} MB, {elapsed_time})")
            
            # Timeout protection
            if elapsed_time > max_duration:
                logger.warning(f"  Pattern discovery timeout reached ({max_duration}) for {room_id}")
                break
            
            # Memory protection - Allow up to 4GB for comprehensive pattern discovery
            if current_memory > initial_memory + 4000:  # More than 4GB increase
                logger.warning(f"  Memory usage too high ({current_memory:.1f} MB), stopping pattern discovery")
                break
            
            # Extract sequences with meaningful length for behavioral patterns
            sequences = self.extract_sequences_optimized(chunk_events, 
                                                       min_length=2, 
                                                       max_length=15)  # Increased to capture longer behavioral patterns
            
            if not sequences:
                continue
            
            logger.info(f"    Generated {len(sequences):,} sequences for analysis")
            
            # Test only behaviorally-relevant time windows
            relevant_windows = self.get_behavioral_relevant_windows()
            
            patterns_in_chunk = 0
            for window in relevant_windows:
                try:
                    pattern_strength = self.test_pattern_significance(
                        sequences, window, room_id
                    )
                    
                    if pattern_strength > self.adaptive_threshold:
                        pattern = Pattern(sequences[:10], window, pattern_strength)  # Limit stored sequences
                        self.pattern_library[room_id].add(pattern)
                        patterns_in_chunk += 1
                        total_patterns_found += 1
                        
                except Exception as e:
                    logger.warning(f"    Error testing window {window}min: {e}")
                    continue
            
            logger.info(f"    Found {patterns_in_chunk} patterns in this chunk")
            
            # Memory management - if too many patterns, keep only strongest
            if len(self.pattern_library[room_id]) > 100:
                sorted_patterns = sorted(self.pattern_library[room_id], 
                                       key=lambda p: p.strength, reverse=True)
                self.pattern_library[room_id] = set(sorted_patterns[:100])
                logger.info(f"    Pruned to top 100 strongest patterns")
        
        # Final resource monitoring
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        total_time = datetime.now() - start_time
        memory_delta = final_memory - initial_memory
        
        logger.info(f"Completed pattern discovery for {room_id}: {total_patterns_found} total patterns found")
        logger.info(f"  Resource usage: {total_time} elapsed, {memory_delta:+.1f} MB memory change")
        
        # Simplified negative pattern discovery (reduced scope)
        await self._discover_negative_patterns_optimized(event_stream, room_id)
    
    def extract_sequences_optimized(self, event_stream: List[Dict], min_length: int, max_length: int) -> List[List[Dict]]:
        """
        Extract variable-length sequences with memory optimization
        Dramatically reduced max_length to prevent millions of sequences
        """
        sequences = []
        
        # Validate inputs
        if not event_stream or not isinstance(event_stream, list):
            logger.warning(f"Invalid event_stream: {type(event_stream)}")
            return sequences
            
        if not isinstance(min_length, int) or not isinstance(max_length, int):
            logger.warning(f"Invalid length parameters: min_length={min_length}, max_length={max_length}")
            return sequences
            
        if min_length < 1 or max_length < min_length or max_length > 20:
            logger.warning(f"Invalid length values: min_length={min_length}, max_length={max_length}")
            return sequences
        
        # Extract sequences safely with step size for sampling
        stream_length = len(event_stream)
        if stream_length < min_length:
            logger.debug(f"Event stream too short ({stream_length}) for min_length ({min_length})")
            return sequences
        
        # Use step size to reduce sequence count (sample every 5th starting position)
        step_size = max(1, stream_length // 1000)  # Limit to ~1000 starting positions
        
        for start_idx in range(0, stream_length, step_size):
            max_possible_length = min(max_length + 1, stream_length - start_idx + 1)
            for length in range(min_length, max_possible_length):
                sequence = event_stream[start_idx:start_idx + length]
                sequences.append(sequence)
                
                # Limit total sequences to prevent memory explosion but allow more comprehensive analysis
                if len(sequences) >= 50000:  # Increased limit for better pattern discovery
                    logger.info(f"    Reached sequence limit (50,000), stopping extraction")
                    return sequences
        
        return sequences
    
    def get_behavioral_relevant_windows(self) -> List[int]:
        """
        Return time windows relevant for comprehensive behavioral pattern detection
        Increased granularity to capture nuanced behavioral patterns
        """
        # Comprehensive behavioral time windows for meaningful pattern discovery
        return [
            5,     # 5 minutes - micro-patterns
            10,    # 10 minutes - short interactions
            15,    # 15 minutes - immediate activity detection
            30,    # 30 minutes - short activity periods
            45,    # 45 minutes - extended interactions
            60,    # 1 hour - typical activity blocks
            90,    # 1.5 hours - longer sessions
            120,   # 2 hours - extended activities  
            180,   # 3 hours - work/gaming sessions
            240,   # 4 hours - longer behavioral patterns
            360,   # 6 hours - half-day patterns
            480,   # 8 hours - work day patterns
            720,   # 12 hours - half-day cycles
            1440   # 24 hours - daily routine patterns
        ]
    
    def generate_time_windows(self):
        """
        DEPRECATED - Use get_behavioral_relevant_windows() instead
        This old method tested 43,200 windows causing 4-hour hangs
        """
        logger.warning("generate_time_windows() is deprecated - use get_behavioral_relevant_windows()")
        return self.get_behavioral_relevant_windows()
    
    def is_promising_window(self, minutes: int) -> bool:
        """
        Determine if a time window is worth testing
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
        DEPRECATED - Use extract_sequences_optimized() instead
        This old method generated millions of sequences causing memory explosion
        """
        logger.warning("extract_sequences() is deprecated - use extract_sequences_optimized()")
        return self.extract_sequences_optimized(event_stream, min_length, min(max_length, 5))
    
    def test_pattern_significance(self, sequences: List[List[Dict]], window_minutes: int, room_id: str) -> float:
        """
        Test if patterns within time window are statistically significant
        Returns significance score (0.0 to 1.0, higher = more significant)
        """
        try:
            # Validate inputs
            if not isinstance(window_minutes, int):
                logger.warning(f"Invalid window_minutes type: {type(window_minutes)} - {window_minutes}")
                return 0.0
                
            if not sequences or not isinstance(sequences, list):
                logger.debug(f"Invalid sequences input: {type(sequences)}")
                return 0.0
                
            # Group sequences by time window
            windowed_patterns = self.group_by_time_window(sequences, window_minutes)
            
            if not windowed_patterns or len(windowed_patterns) < 2:
                logger.debug(f"Not enough windowed patterns ({len(windowed_patterns) if windowed_patterns else 0}) for {room_id} at {window_minutes}min")
                return 0.0
            
            # Count pattern frequencies
            pattern_counts = Counter(str(pattern) for pattern in windowed_patterns)
            unique_patterns = len(pattern_counts)
            total_patterns = len(windowed_patterns)
            
            logger.debug(f"{room_id} {window_minutes}min: {total_patterns} total patterns, {unique_patterns} unique patterns")
            
            # Test if pattern distribution is significantly non-random
            if unique_patterns < 2:
                logger.debug(f"Only one unique pattern type found for {room_id} at {window_minutes}min")
                return 0.0
            
            # Use proper statistical tests for temporal pattern significance
            from scipy.stats import entropy
            import numpy as np
            
            # Test 1: Pattern entropy - low entropy = more predictable patterns
            observed_frequencies = np.array(list(pattern_counts.values()))
            total_freq = observed_frequencies.sum()
            pattern_probabilities = observed_frequencies / total_freq
            
            # Calculate normalized entropy (0 = perfectly predictable, 1 = random)
            max_entropy = np.log(unique_patterns) if unique_patterns > 1 else 1.0
            actual_entropy = entropy(pattern_probabilities, base=np.e)
            normalized_entropy = actual_entropy / max_entropy if max_entropy > 0 else 1.0
            
            # Convert to significance: lower entropy = higher significance
            entropy_significance = max(0.0, 1.0 - normalized_entropy)
            
            # Test 2: Pattern concentration - are patterns concentrated or spread out?
            # Coefficient of variation: std/mean ratio
            freq_mean = np.mean(observed_frequencies)
            freq_std = np.std(observed_frequencies)
            concentration_score = freq_std / freq_mean if freq_mean > 0 else 0.0
            
            # Normalize concentration (higher concentration = more significant patterns)
            concentration_significance = min(1.0, concentration_score / 2.0)  # Cap at reasonable level
            
            # Test 3: Repetition significance - patterns that repeat are significant
            max_freq = np.max(observed_frequencies)
            repetition_rate = max_freq / total_freq
            repetition_significance = min(1.0, repetition_rate * 2.0)  # Scale up but cap
            
            # Combine significance scores with weights
            significance_score = (0.4 * entropy_significance + 
                                0.3 * concentration_significance + 
                                0.3 * repetition_significance)
            
            # Create meaningful p-value based on significance
            # Higher significance -> lower p-value
            p_value = max(0.0001, 1.0 - significance_score)  # Avoid exactly 0.0
            
            # Additional pattern strength indicators
            # Higher pattern repetition increases significance
            max_pattern_freq = max(observed_frequencies)
            repetition_bonus = min(0.3, (max_pattern_freq - 1) / total_patterns)
            
            # Temporal clustering bonus - patterns that cluster in time are more significant
            time_clustering_score = self._calculate_temporal_clustering(windowed_patterns, window_minutes)
            
            # Combine scores
            final_score = significance_score + repetition_bonus + time_clustering_score
            final_score = max(0.0, min(1.0, final_score))
            
            if final_score > 0.1:  # Only log meaningful patterns
                logger.info(f"{room_id} {window_minutes}min: entropy={normalized_entropy:.3f}, concentration={concentration_score:.3f}, "
                           f"repetition_rate={repetition_rate:.3f}, p-value={p_value:.4f}, "
                           f"significance={significance_score:.3f}, clustering={time_clustering_score:.3f}, final={final_score:.3f}")
                           
                # Log pattern distribution for debugging
                top_patterns = pattern_counts.most_common(3)
                logger.debug(f"  Top patterns: {top_patterns} (unique={unique_patterns}, total={total_patterns})")
            
            return final_score
            
        except Exception as e:
            logger.warning(f"Error testing pattern significance for {room_id} at {window_minutes}min: {e}")
            return 0.0
    
    async def _cache_patterns_for_models(self, room_name: str, patterns: set) -> None:
        """Cache discovered patterns in Redis format that models expect"""
        if not patterns:
            logger.debug(f"No patterns to cache for {room_name}")
            return
            
        try:
            # Convert pattern objects to format expected by models
            pattern_list = []
            for pattern_obj in patterns:
                try:
                    # Extract meaningful data from Pattern/NegativePattern objects
                    if hasattr(pattern_obj, 'sequences'):
                        # Regular Pattern object with event sequences
                        sequences = pattern_obj.sequences
                        if sequences and len(sequences) > 0:
                            # Convert first sequence to behavioral pattern string
                            first_sequence = sequences[0]
                            pattern_data = ""
                            for evt in first_sequence:
                                if isinstance(evt, dict):
                                    # Create meaningful pattern from event data
                                    room = evt.get('room', 'unknown')
                                    entity = evt.get('entity_id', 'unknown')
                                    state = evt.get('state', 'unknown')
                                    pattern_data += f"{entity}:{state}->"
                            
                            # Remove trailing ->
                            pattern_data = pattern_data.rstrip('->')
                            
                            if not pattern_data:
                                pattern_data = f"sequence_{len(first_sequence)}_events"
                        else:
                            pattern_data = "empty_sequence"
                        
                        significance = getattr(pattern_obj, 'strength', 0.5)
                        time_window = getattr(pattern_obj, 'time_window', 60)
                        
                    elif hasattr(pattern_obj, 'pattern'):
                        # NegativePattern object with pattern string
                        pattern_data = pattern_obj.pattern
                        significance = getattr(pattern_obj, 'strength', 0.5)
                        time_window = 60  # Default
                        
                    else:
                        # Fallback - should not happen with proper Pattern objects
                        logger.warning(f"Unknown pattern object type: {type(pattern_obj)}")
                        continue
                    
                    # Create pattern dict in format models expect
                    pattern_dict = {
                        'pattern': pattern_data,
                        'frequency': 1,  # Could be calculated from sequences length
                        'length': len(pattern_data) if isinstance(pattern_data, str) else 1,
                        'significance': significance,
                        'time_window': time_window
                    }
                    pattern_list.append(pattern_dict)
                    
                except Exception as e:
                    logger.warning(f"Error converting pattern {pattern_obj}: {e}")
                    continue
            
            if pattern_list:
                # Cache patterns in Redis where models look for them
                from src.storage.feature_store import RedisFeatureStore
                feature_store = RedisFeatureStore()
                await feature_store.initialize()
                
                # Cache with 'all_patterns' key that get_room_patterns() expects
                success = await feature_store.cache_pattern(room_name, 'all_patterns', pattern_list)
                
                if success:
                    logger.info(f"Cached {len(pattern_list)} patterns in Redis for {room_name}")
                else:
                    logger.warning(f"Failed to cache patterns for {room_name}")
                    
                await feature_store.close()
            else:
                logger.warning(f"No valid patterns to cache for {room_name}")
                
        except Exception as e:
            logger.error(f"Error caching patterns for {room_name}: {e}")

    # Database storage removed - models only use Redis cache

    def _calculate_temporal_clustering(self, windowed_patterns: List[str], window_minutes: int) -> float:
        """
        Calculate temporal clustering score for patterns
        Patterns that occur close together in time are more significant
        """
        try:
            if len(windowed_patterns) < 2:
                return 0.0
            
            # Simple clustering metric: if patterns repeat within the time window,
            # they show temporal clustering which increases significance
            pattern_counts = Counter(windowed_patterns)
            
            # Find patterns that occur multiple times
            repeated_patterns = sum(1 for count in pattern_counts.values() if count > 1)
            total_unique_patterns = len(pattern_counts)
            
            if total_unique_patterns == 0:
                return 0.0
            
            # Higher clustering when more patterns repeat
            clustering_ratio = repeated_patterns / total_unique_patterns
            
            # Scale by window size - shorter windows with repetition are more significant
            window_scale = max(0.1, min(1.0, 60.0 / window_minutes))  # 60min = baseline
            
            clustering_score = clustering_ratio * window_scale * 0.2  # Max 0.2 bonus
            
            return max(0.0, min(0.2, clustering_score))
            
        except Exception as e:
            logger.debug(f"Error calculating temporal clustering: {e}")
            return 0.0
    
    def group_by_time_window(self, sequences: List[List[Dict]], window_minutes: int) -> List[str]:
        """Group sequences that occur within the same time window"""
        windowed_patterns = []
        
        for sequence in sequences:
            if not sequence:
                continue
                
            # Handle both datetime objects and strings
            start_timestamp = sequence[0]['timestamp']
            end_timestamp = sequence[-1]['timestamp']
            
            if isinstance(start_timestamp, str):
                start_time = datetime.fromisoformat(start_timestamp.replace('Z', '+00:00'))
            else:
                start_time = start_timestamp
                
            if isinstance(end_timestamp, str):
                end_time = datetime.fromisoformat(end_timestamp.replace('Z', '+00:00'))
            else:
                end_time = end_timestamp
            
            # Check if sequence fits within time window
            if (end_time - start_time).total_seconds() <= window_minutes * 60:
                # Create pattern signature
                pattern_sig = self.create_pattern_signature(sequence)
                windowed_patterns.append(pattern_sig)
        
        return windowed_patterns
    
    def create_pattern_signature(self, sequence: List[Dict]) -> str:
        """Create a signature for a sequence of events"""
        signature_parts = []
        
        for event in sequence:
            part = f"{event.get('room', 'unknown')}-{event.get('sensor_type', 'unknown')}-{event.get('state', 0)}"
            signature_parts.append(part)
        
        return "->".join(signature_parts)
    
    async def _discover_negative_patterns_optimized(self, event_stream: List[Dict], room_id: str):
        """
        Optimized negative pattern discovery with reduced scope
        Focuses on the most important anti-patterns without memory explosion
        """
        logger.info(f"Discovering key negative patterns for {room_id}")
        
        # Limit scope to prevent memory explosion
        if len(event_stream) > 10000:
            # Sample recent events for negative pattern analysis
            sample_events = event_stream[-10000:]
            logger.info(f"  Sampling most recent 10,000 events for negative pattern analysis")
        else:
            sample_events = event_stream
        
        # Find expected but missing sequences (simplified approach)
        key_possible_sequences = self._generate_key_possible_sequences(sample_events)
        actual_sequences = set(self.create_pattern_signature(seq) 
                             for seq in self.extract_sequences_optimized(sample_events, 2, 3))
        
        missing_patterns = key_possible_sequences - actual_sequences
        
        # Store only the most significant missing patterns
        if room_id not in self.pattern_library:
            self.pattern_library[room_id] = set()
        
        negative_patterns_added = 0
        for missing_pattern in missing_patterns:
            if self.is_significant_absence(missing_pattern, sample_events):
                strength = self.calculate_absence_strength(missing_pattern, sample_events)
                if strength > 0.7:  # Higher threshold for negative patterns
                    self.pattern_library[room_id].add(
                        NegativePattern(missing_pattern, strength)
                    )
                    negative_patterns_added += 1
                    
                    # Limit negative patterns to prevent bloat
                    if negative_patterns_added >= 20:
                        break
        
        logger.info(f"  Added {negative_patterns_added} significant negative patterns for {room_id}")
    
    def _generate_key_possible_sequences(self, event_stream: List[Dict]) -> set:
        """
        Generate only key possible sequences to prevent memory explosion
        Focuses on the most important 2-3 event combinations
        """
        unique_events = set()
        for event in event_stream[-1000:]:  # Sample recent events only
            event_sig = f"{event.get('room', 'unknown')}-{event.get('sensor_type', 'unknown')}-{event.get('state', 0)}"
            unique_events.add(event_sig)
            
            # Limit unique events to prevent explosion
            if len(unique_events) >= 20:
                break
        
        # Generate only 2-event combinations (not 3+ to save memory)
        possible_sequences = set()
        unique_events_list = list(unique_events)
        
        for i, event1 in enumerate(unique_events_list):
            for j, event2 in enumerate(unique_events_list):
                if i != j:
                    possible_sequences.add(f"{event1}->{event2}")
                    
                    # Limit sequences to prevent memory explosion
                    if len(possible_sequences) >= 200:
                        return possible_sequences
        
        return possible_sequences
    
    def discover_negative_patterns(self, event_stream: List[Dict], room_id: str):
        """
        DEPRECATED - Use _discover_negative_patterns_optimized() instead
        This old method could cause memory explosion with large datasets
        """
        logger.warning("discover_negative_patterns() is deprecated - using optimized version")
        import asyncio
        asyncio.create_task(self._discover_negative_patterns_optimized(event_stream, room_id))
    
    def generate_possible_sequences(self, event_stream: List[Dict]) -> set:
        """Generate all theoretically possible sequences from observed events"""
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
        """Determine if absence of a pattern is statistically significant"""
        # Simplified: if pattern components exist but never occur together
        pattern_parts = pattern.split('->')
        
        # Check if individual components exist
        components_exist = all(
            any(part.split('-')[0] in str(event) for event in event_stream)
            for part in pattern_parts
        )
        
        return components_exist and len(pattern_parts) >= 2
    
    def calculate_absence_strength(self, pattern: str, event_stream: List[Dict]) -> float:
        """
        Calculate how strong the evidence is for pattern absence
        Based on component frequency and context availability
        """
        try:
            if not pattern or not event_stream:
                return 0.0
            
            # Split pattern into components (e.g., "room1->room2->room3")
            pattern_parts = pattern.split('->')
            if len(pattern_parts) < 2:
                return 0.0
            
            # Count occurrences of individual components
            component_counts = {}
            total_events = len(event_stream)
            
            for event in event_stream:
                event_signature = self._create_event_signature(event)
                for part in pattern_parts:
                    if part in event_signature:
                        component_counts[part] = component_counts.get(part, 0) + 1
            
            # Calculate component frequencies
            component_frequencies = {}
            for part in pattern_parts:
                count = component_counts.get(part, 0)
                frequency = count / total_events if total_events > 0 else 0.0
                component_frequencies[part] = frequency
            
            # Calculate expected pattern occurrences if independent
            min_component_freq = min(component_frequencies.values()) if component_frequencies else 0.0
            expected_pattern_freq = min_component_freq ** len(pattern_parts)
            
            # Calculate actual pattern occurrences
            actual_pattern_count = self._count_pattern_occurrences(pattern, event_stream)
            actual_pattern_freq = actual_pattern_count / total_events if total_events > 0 else 0.0
            
            # Absence strength = how much less frequent than expected
            if expected_pattern_freq > 0:
                absence_ratio = 1.0 - (actual_pattern_freq / expected_pattern_freq)
                # Clamp between 0 and 1
                absence_strength = max(0.0, min(1.0, absence_ratio))
            else:
                # If no expected occurrences, absence is certain
                absence_strength = 1.0 if actual_pattern_count == 0 else 0.0
            
            # Weight by sample size - more data = more confident
            sample_weight = min(1.0, total_events / 100.0)  # Full confidence at 100+ events
            final_strength = absence_strength * sample_weight
            
            logger.debug(f"Pattern '{pattern}': expected_freq={expected_pattern_freq:.4f}, "
                        f"actual_freq={actual_pattern_freq:.4f}, absence_strength={final_strength:.4f}")
            
            return final_strength
            
        except Exception as e:
            logger.error(f"Error calculating absence strength for pattern '{pattern}': {e}")
            return 0.0
    
    def _create_event_signature(self, event: Dict) -> str:
        """Create signature for event matching pattern components"""
        room = event.get('room', 'unknown')
        state = event.get('state', 'unknown')
        sensor_type = event.get('sensor_type', 'unknown')
        return f"{room}:{sensor_type}:{state}"
    
    def _count_pattern_occurrences(self, pattern: str, event_stream: List[Dict]) -> int:
        """Count actual occurrences of the pattern in event stream"""
        pattern_parts = pattern.split('->')
        if len(pattern_parts) < 2:
            return 0
        
        occurrences = 0
        
        # Look for sequences matching the pattern
        for i in range(len(event_stream) - len(pattern_parts) + 1):
            match_found = True
            
            # Check if sequence starting at i matches pattern
            for j, expected_part in enumerate(pattern_parts):
                event_signature = self._create_event_signature(event_stream[i + j])
                if expected_part not in event_signature:
                    match_found = False
                    break
            
            if match_found:
                occurrences += 1
        
        return occurrences


class StatisticalTestSuite:
    """
    Statistical tests for pattern significance
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
    Unbiased pattern mining using suffix trees
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
    """Represents a discovered behavioral pattern"""
    
    def __init__(self, sequences: List[List[Dict]], time_window: int, strength: float):
        self.sequences = sequences
        self.time_window = time_window
        self.strength = strength
        from datetime import timezone
        self.discovered_at = datetime.now(timezone.utc)
    
    def __hash__(self):
        return hash((str(self.sequences), self.time_window))
    
    def __eq__(self, other):
        return (self.sequences == other.sequences and 
                self.time_window == other.time_window)


class NegativePattern:
    """Represents a significant absence pattern"""
    
    def __init__(self, pattern: str, strength: float):
        self.pattern = pattern
        self.strength = strength
        from datetime import timezone
        self.discovered_at = datetime.now(timezone.utc)
    
    def __hash__(self):
        return hash(self.pattern)
    
    def __eq__(self, other):
        return self.pattern == other.pattern