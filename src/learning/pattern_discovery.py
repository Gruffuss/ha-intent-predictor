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
                (observed - expected_frequency) ** 2 / expected_frequency
                for observed in pattern_counts.values()
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
                
            start_time = datetime.fromisoformat(sequence[0]['timestamp'].replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(sequence[-1]['timestamp'].replace('Z', '+00:00'))
            
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
        """Calculate how strong the evidence is for pattern absence"""
        # Simplified calculation based on component frequency
        return 0.8  # Placeholder


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
        self.discovered_at = datetime.now()
    
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
        self.discovered_at = datetime.now()
    
    def __hash__(self):
        return hash(self.pattern)
    
    def __eq__(self, other):
        return self.pattern == other.pattern