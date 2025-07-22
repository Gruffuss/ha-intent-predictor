"""
Automatic Pattern Mining - Discovers behavioral patterns without assumptions
Implements the sophisticated pattern discovery from CLAUDE.md
"""

import logging
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import chi2_contingency
from suffix_trees import STree

logger = logging.getLogger(__name__)


class PatternDiscovery:
    """
    Discovers behavioral patterns from event streams without any assumptions
    Exactly as specified in CLAUDE.md
    """
    
    def __init__(self):
        self.pattern_library = {}
        self.statistical_tests = StatisticalTestSuite()
        self.adaptive_threshold = 0.05  # P-value threshold for statistical significance
        
    async def discover_multizone_patterns(self, room_name: str, historical_events: List[Dict] = None) -> Dict:
        """
        Discover patterns for multi-zone rooms
        Handles the complex zone relationships from CLAUDE.md
        """
        logger.info(f"Discovering multi-zone patterns for {room_name}")
        
        if historical_events is None:
            # If no events provided, get from database
            from src.storage.timeseries_db import TimescaleDBManager
            from config.config_loader import ConfigLoader
            
            config = ConfigLoader()
            db_config = config.get("database.timescale")
            db = TimescaleDBManager(f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
            await db.initialize()
            
            async with db.engine.begin() as conn:
                from sqlalchemy import text
                result = await conn.execute(text("SELECT * FROM sensor_events WHERE room = :room ORDER BY timestamp"), {'room': room_name})
                historical_events = [dict(row._mapping) for row in result.fetchall()]
            
            await db.close()
        
        # Filter events for this room
        room_events = [
            event for event in historical_events
            if event.get('room') == room_name
        ]
        
        if not room_events:
            logger.warning(f"No events found for room {room_name}")
            return {}
        
        # Discover patterns using the sophisticated algorithm
        self.discover_patterns(room_events, room_name)
        
        patterns = self.pattern_library.get(room_name, set())
        
        logger.info(f"Discovered {len(patterns)} patterns for {room_name}")
        
        return {
            'room': room_name,
            'pattern_count': len(patterns),
            'patterns': list(patterns)[:10]  # Show first 10 patterns
        }
    
    async def discover_bathroom_patterns(self, bathroom_rooms: List[str], historical_events: List[Dict] = None) -> Dict:
        """
        Discover patterns specific to bathroom usage
        Handles the special bathroom logic from CLAUDE.md
        """
        logger.info(f"Discovering bathroom patterns for {bathroom_rooms}")
        
        if historical_events is None:
            # If no events provided, get from database
            from src.storage.timeseries_db import TimescaleDBManager
            from config.config_loader import ConfigLoader
            
            config = ConfigLoader()
            db_config = config.get("database.timescale")
            db = TimescaleDBManager(f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
            await db.initialize()
            
            bathroom_events = []
            async with db.engine.begin() as conn:
                from sqlalchemy import text
                for room in bathroom_rooms:
                    result = await conn.execute(text("SELECT * FROM sensor_events WHERE room = :room ORDER BY timestamp"), {'room': room})
                    room_events = [dict(row._mapping) for row in result.fetchall()]
                    bathroom_events.extend(room_events)
            
            await db.close()
            historical_events = bathroom_events
        else:
            bathroom_events = []
            for room in bathroom_rooms:
                room_events = [
                    event for event in historical_events
                    if event.get('room') == room
                ]
                bathroom_events.extend(room_events)
        
        # Bathroom-specific pattern discovery
        patterns = {}
        for room in bathroom_rooms:
            room_events = [e for e in bathroom_events if e.get('room') == room]
            if room_events:
                self.discover_patterns(room_events, room)
                patterns[room] = self.pattern_library.get(room, set())
        
        return {
            'bathroom_rooms': bathroom_rooms,
            'patterns': patterns
        }
    
    async def discover_transition_patterns(self, area_name: str, historical_events: List[Dict] = None) -> Dict:
        """Discover transition patterns for hallways and connections"""
        logger.info(f"Discovering transition patterns for {area_name}")
        
        if historical_events is None:
            # If no events provided, get from database
            from src.storage.timeseries_db import TimescaleDBManager
            from config.config_loader import ConfigLoader
            
            config = ConfigLoader()
            db_config = config.get("database.timescale")
            db = TimescaleDBManager(f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
            await db.initialize()
            
            async with db.engine.begin() as conn:
                from sqlalchemy import text
                result = await conn.execute(text("SELECT * FROM sensor_events WHERE entity_id LIKE '%hallway%' OR entity_id LIKE '%stairs%' ORDER BY timestamp"))
                historical_events = [dict(row._mapping) for row in result.fetchall()]
            
            await db.close()
        
        # Filter transition events
        transition_events = [
            event for event in historical_events
            if 'hallway' in event.get('entity_id', '') or 'stairs' in event.get('entity_id', '')
        ]
        
        if transition_events:
            self.discover_patterns(transition_events, area_name)
        
        patterns = self.pattern_library.get(area_name, set())
        
        return {
            'area': area_name,
            'pattern_count': len(patterns),
            'patterns': list(patterns)[:10]
        }

    def discover_patterns(self, event_stream: List[Dict], room_id: str):
        """
        No assumptions - let data speak for itself
        Implements exact algorithm from CLAUDE.md
        """
        logger.info(f"Discovering patterns for {room_id}")
        
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
                if room_id not in self.pattern_library:
                    self.pattern_library[room_id] = set()
                    
                self.pattern_library[room_id].add(
                    Pattern(sequences, window, pattern_strength)
                )
        
        # Detect anti-patterns (what DOESN'T happen)
        self.discover_negative_patterns(event_stream, room_id)
        
        logger.info(f"Discovered {len(self.pattern_library.get(room_id, []))} patterns for {room_id}")
    
    def generate_time_windows(self):
        """
        Test all possible time windows, not just "hourly" or "daily"
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
        Extract variable-length sequences from event stream
        """
        sequences = []
        
        for start_idx in range(len(event_stream)):
            for length in range(min_length, min(max_length + 1, len(event_stream) - start_idx + 1)):
                sequence = event_stream[start_idx:start_idx + length]
                sequences.append(sequence)
        
        return sequences
    
    def test_pattern_significance(self, sequences: List[List[Dict]], window_minutes: int, room_id: str) -> float:
        """
        Test if patterns within time window are statistically significant
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
                (float(observed) - expected_frequency) ** 2 / expected_frequency
                for observed in pattern_counts.values()
                if isinstance(observed, (int, float)) and observed > 0
            )
            
            # Convert to p-value (simplified)
            p_value = 1.0 / (1.0 + chi_squared)
            
            return p_value
            
        except Exception as e:
            logger.warning(f"Error testing pattern significance: {e}")
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
    
    def discover_negative_patterns(self, event_stream: List[Dict], room_id: str):
        """
        Detect anti-patterns (what DOESN'T happen)
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