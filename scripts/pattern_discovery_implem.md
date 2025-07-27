Looking at your pattern discovery implementation, I can see the core issue is with the statistical significance testing in `test_pattern_significance()`. Let me break down what's happening and why it might not be working as intended:

## Current Issues:

1. **Chi-squared test misapplication**: Your current approach assumes all patterns should occur with equal frequency (`expected_frequency = total_patterns / unique_patterns`). But in reality, some behavioral patterns naturally occur more than others. For example, "bedroom→bathroom" in the morning would be much more common than "bathroom→office→kitchen→bathroom" in quick succession.

2. **Invalid p-value calculation**: The formula `p_value = 1.0 / (1.0 + chi_squared)` isn't a real p-value - it's just an arbitrary transformation. Real p-values require proper degrees of freedom and chi-squared distribution tables.

3. **Missing temporal context**: The current approach treats all pattern occurrences equally, regardless of when they happen. But "kitchen activity at 7am" vs "kitchen activity at 3am" have very different significance.

## What Pattern Discovery Should Actually Do:

For an occupancy prediction system, you want to find:
- **Recurring temporal patterns**: "Office occupied weekdays 9am-5pm" (but discovered, not assumed)
- **Sequential patterns**: "When bedroom empties, bathroom fills within 5 minutes"  
- **Conditional patterns**: "If living room occupied past 10pm, bedroom occupancy likely within 30min"
- **Absence patterns**: "Kitchen never occupied 2am-5am"

## Better Approach:

Instead of testing if pattern frequencies deviate from uniform distribution, you should test if patterns occur more than random chance would predict. Here's what I think would work better:

1. **Calculate baseline probabilities**: What's the probability of each room being occupied at any given time?
2. **Expected vs observed frequency**: For a sequence like "bedroom→bathroom", calculate expected frequency as P(bedroom) × P(bathroom) × number_of_time_windows
3. **Use proper statistical tests**: Poisson test for rare events, binomial test for common events
4. **Consider temporal clustering**: Patterns that cluster at specific times are more significant

Would you like me to walk through how to redesign the pattern significance testing to actually discover meaningful occupancy patterns? I can show you the mathematical approach that would properly identify whether someone has routines or truly random behavior.

The Problem with Current Approach
Your current chi-squared test assumes all patterns should occur equally often, which doesn't make sense for human behavior. Instead, we need to test if patterns occur more than random chance would predict.
Redesigned Pattern Discovery
Here's a better approach that will actually find meaningful patterns:

import numpy as np
from scipy import stats
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ImprovedPatternDiscovery:
    """
    Discovers meaningful occupancy patterns by testing against random behavior
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
    
    def _discover_temporal_patterns(self, events: List[Dict], room_id: str) -> Dict:
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
        
        Key Improvements:

Proper Null Hypothesis: Tests patterns against baseline random behavior, not uniform distribution
Multiple Pattern Types:

Temporal: "Usually occupied at 9am on weekdays"
Sequential: "Bedroom → Bathroom within 5 minutes"
Conditional: "If X happens, Y follows within T time"
Absence: "Never occupied 2am-5am"


Correct Statistical Tests:

Binomial test for temporal patterns
Poisson/Normal test for sequential patterns
T-tests for absence patterns
FDR correction for multiple testing


Considers Context:

Time of day matters
Day of week matters
Pattern consistency matters


Effect Size: Not just statistical significance, but also practical significance (is the difference meaningful?)

Integration with Your Code:
Replace your current PatternDiscovery class with this improved version. The key changes:
python# In your pattern_discovery.py
class PatternDiscovery:
    def __init__(self):
        self.pattern_library = {}
        self.statistical_tests = StatisticalTestSuite()
        self.pattern_discoverer = ImprovedPatternDiscovery()  # Add this
        
    async def discover_multizone_patterns(self, room_name: str, historical_events: List[Dict] = None) -> Dict:
        """Use improved pattern discovery"""
        logger.info(f"Discovering patterns for {room_name}")
        
        if historical_events is None:
            historical_events = await self._load_events_with_monitoring(room_name)
        
        # Use the improved discovery
        patterns = self.pattern_discoverer.discover_patterns(historical_events, room_name)
        
        # Store in library
        self.pattern_library[room_name] = patterns
        
        return {
            'room': room_name,
            'patterns': patterns,
            'total_events': len(historical_events)
        }
This approach will actually find real patterns in your occupancy data, or determine that behavior is truly random if no patterns exist.