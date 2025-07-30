"""
Performance Comparison: STUMPY-based vs Event-based HMM Pattern Discovery

Comprehensive benchmarking suite to compare the old STUMPY matrix profile system
with the new event-based HMM pattern discovery system.

Test Scenarios:
1. Bedroom (84% sparse data) - Most challenging case
2. Office (moderate activity) - Baseline performance  
3. Living Kitchen (multi-zone) - Complexity test
4. Bathroom (inference-based) - Special case
5. Full System (all rooms) - Complete system test

Metrics Measured:
- Accuracy: Pattern discovery success, prediction accuracy, false positive/negative rates
- Performance: Processing time, memory usage, CPU utilization
- Reliability: System stability, no hangs/crashes
- Data Handling: Sparse data processing, large dataset handling

Performance Target: >85% accuracy improvement with HMM system
"""

import asyncio
import logging
import time
import psutil
import tracemalloc
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
import json
import gc
import warnings

# Test framework imports
import pytest
import unittest
from unittest.mock import patch, MagicMock

# Import both pattern discovery systems
from src.learning.pattern_discovery import PatternDiscovery as StumpyPatternDiscovery
from src.learning.event_based_pattern_discovery import EventBasedPatternDiscovery

# Import supporting components
from src.storage.timeseries_db import TimescaleDBManager
from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance measurement results"""
    system_name: str
    room_name: str
    test_scenario: str
    
    # Timing metrics
    processing_time_seconds: float
    pattern_discovery_time: float
    data_loading_time: float
    
    # Memory metrics
    peak_memory_mb: float
    memory_increase_mb: float
    
    # CPU metrics
    avg_cpu_percent: float
    peak_cpu_percent: float
    
    # Accuracy metrics
    patterns_discovered: int
    pattern_success_rate: float
    validation_accuracy: float
    
    # Data handling metrics
    events_processed: int
    duplicate_events_removed: int
    sparse_data_handled: bool
    
    # Reliability metrics
    completed_successfully: bool
    errors_encountered: List[str]
    warnings_count: int
    
    # System-specific metrics
    system_specific_metrics: Dict[str, Any]


@dataclass
class ComparisonResult:
    """Comprehensive comparison between two systems"""
    stumpy_metrics: PerformanceMetrics
    hmm_metrics: PerformanceMetrics
    
    # Improvement calculations
    processing_time_improvement: float  # Negative = slower
    memory_improvement: float  # Negative = more memory
    accuracy_improvement: float  # Positive = better accuracy
    reliability_improvement: float  # 0-1 scale
    
    # Overall assessment
    recommended_system: str
    improvement_summary: str
    deployment_recommendation: str


class PerformanceMonitor:
    """Monitor system performance during pattern discovery"""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.peak_memory = 0
        self.cpu_samples = []
        self.warnings_count = 0
        self.errors = []
    
    def start_monitoring(self):
        """Start performance monitoring"""
        tracemalloc.start()
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        self.cpu_samples = []
        self.warnings_count = 0
        self.errors = []
        
        # Set up warnings capture
        def warning_handler(message, category, filename, lineno, file=None, line=None):
            self.warnings_count += 1
        
        warnings.showwarning = warning_handler
        
        logger.info(f"Performance monitoring started - Memory: {self.start_memory:.1f} MB")
    
    def sample_resources(self):
        """Sample current resource usage"""
        try:
            # Memory
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            self.peak_memory = max(self.peak_memory, current_memory)
            
            # CPU
            cpu_percent = psutil.Process().cpu_percent()
            self.cpu_samples.append(cpu_percent)
            
        except Exception as e:
            logger.warning(f"Error sampling resources: {e}")
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return metrics"""
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        metrics = {
            'processing_time': end_time - self.start_time,
            'start_memory_mb': self.start_memory,
            'end_memory_mb': end_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': end_memory - self.start_memory,
            'avg_cpu_percent': np.mean(self.cpu_samples) if self.cpu_samples else 0.0,
            'peak_cpu_percent': np.max(self.cpu_samples) if self.cpu_samples else 0.0,
            'warnings_count': self.warnings_count
        }
        
        tracemalloc.stop()
        logger.info(f"Performance monitoring stopped - Total time: {metrics['processing_time']:.2f}s")
        return metrics


class PatternDiscoveryPerformanceTest:
    """Comprehensive performance testing for pattern discovery systems"""
    
    def __init__(self):
        self.config = ConfigLoader()
        self.test_rooms = ['bedroom', 'office', 'living_kitchen', 'bathroom', 'small_bathroom']
        self.test_results = {}
        
        # Test data configuration
        self.test_config = {
            'max_events_per_room': 50000,  # Limit for testing
            'test_date_range_days': 30,     # Shorter range for focused testing
            'validation_split': 0.2,       # 20% for validation
            'sparse_data_threshold': 0.8    # 80% zeros = sparse
        }
    
    async def initialize_database(self):
        """Initialize database connection for testing"""
        db_config = self.config.get("database.timescale")
        self.db = TimescaleDBManager(
            f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        await self.db.initialize()
        logger.info("Database connection initialized for testing")
    
    async def load_test_data(self, room_name: str) -> Tuple[List[Dict], Dict[str, Any]]:
        """Load test data for a specific room with metadata"""
        try:
            # Room sensor mapping
            room_sensors = {
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
            
            sensors = room_sensors.get(room_name, [])
            if not sensors:
                return [], {'error': f'No sensors defined for {room_name}'}
            
            async with self.db.engine.begin() as conn:
                from sqlalchemy import text
                
                # Build sensor filter
                sensor_conditions = [f"entity_id = '{sensor}'" for sensor in sensors]
                sensor_filter = " OR ".join(sensor_conditions)
                
                query = f"""
                    SELECT timestamp, entity_id, state, attributes, room, sensor_type
                    FROM sensor_events 
                    WHERE ({sensor_filter})
                    AND timestamp >= NOW() - INTERVAL '{self.test_config["test_date_range_days"]} days'
                    ORDER BY timestamp DESC
                    LIMIT {self.test_config['max_events_per_room']}
                """
                
                result = await conn.execute(text(query))
                events = [dict(row._mapping) for row in result.fetchall()]
                
                # Calculate data characteristics
                df = pd.DataFrame(events)
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['occupied'] = df['state'].isin(['on', 'occupied', '1', 1]).astype(int)
                    
                    metadata = {
                        'total_events': len(events),
                        'date_range': {
                            'start': df['timestamp'].min().isoformat(),
                            'end': df['timestamp'].max().isoformat()
                        },
                        'occupancy_rate': df['occupied'].mean(),
                        'sparse_data': df['occupied'].mean() < (1 - self.test_config['sparse_data_threshold']),
                        'sensors_involved': len(sensors),
                        'unique_sensors_active': df['entity_id'].nunique()
                    }
                else:
                    metadata = {'total_events': 0, 'error': 'No events found'}
                
                logger.info(f"Loaded {len(events):,} events for {room_name}")
                logger.info(f"  Occupancy rate: {metadata.get('occupancy_rate', 0):.3f}")
                logger.info(f"  Sparse data: {metadata.get('sparse_data', False)}")
                
                return events, metadata
                
        except Exception as e:
            logger.error(f"Error loading test data for {room_name}: {e}")
            return [], {'error': str(e)}
    
    async def test_stumpy_system(self, room_name: str, test_events: List[Dict]) -> PerformanceMetrics:
        """Test the STUMPY-based pattern discovery system"""
        logger.info(f"🔬 Testing STUMPY system for {room_name}")
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        errors = []
        patterns_discovered = 0
        validation_accuracy = 0.0
        system_metrics = {}
        
        try:
            # Initialize STUMPY pattern discovery
            stumpy_discovery = StumpyPatternDiscovery()
            
            # Track data loading time
            data_load_start = time.time()
            
            # Sample resources during processing
            monitor.sample_resources()
            
            # Run pattern discovery
            pattern_start = time.time()
            
            try:
                result = await stumpy_discovery.discover_multizone_patterns(
                    room_name, test_events
                )
                
                # Extract metrics from results
                if result and 'patterns' in result:
                    patterns = result['patterns']
                    
                    # Count successful pattern discoveries
                    if 'recurring' in patterns:
                        for window, data in patterns['recurring'].items():
                            if isinstance(data, dict) and data.get('found'):
                                patterns_discovered += data.get('pattern_count', 0)
                    
                    # Get other pattern types
                    for pattern_type in ['sequential', 'seasonal', 'durations']:
                        if pattern_type in patterns and not patterns[pattern_type].get('error'):
                            patterns_discovered += 1
                    
                    # Calculate success rate
                    total_attempted = len(patterns.get('recurring', {}))
                    if total_attempted > 0:
                        successful = sum(1 for p in patterns['recurring'].values() 
                                       if isinstance(p, dict) and p.get('found'))
                        validation_accuracy = successful / total_attempted
                    
                    system_metrics = {
                        'stumpy_windows_tested': total_attempted,
                        'stumpy_windows_successful': sum(1 for p in patterns['recurring'].values() 
                                                       if isinstance(p, dict) and p.get('found')),
                        'preprocessing_applied': any(
                            p.get('preprocessing_info') for p in patterns['recurring'].values()
                            if isinstance(p, dict)
                        )
                    }
                
            except Exception as e:
                errors.append(f"STUMPY pattern discovery failed: {str(e)}")
                logger.error(f"STUMPY pattern discovery error: {e}")
            
            pattern_end = time.time()
            data_load_time = pattern_start - data_load_start
            pattern_time = pattern_end - pattern_start
            
            monitor.sample_resources()
            
        except Exception as e:
            errors.append(f"STUMPY system error: {str(e)}")
            logger.error(f"STUMPY system error: {e}")
            data_load_time = 0
            pattern_time = 0
        
        # Stop monitoring and collect metrics
        performance_data = monitor.stop_monitoring()
        
        # Determine if sparse data was handled successfully
        df = pd.DataFrame(test_events) if test_events else pd.DataFrame()
        sparse_data_handled = True
        if not df.empty:
            df['occupied'] = df['state'].isin(['on', 'occupied', '1', 1]).astype(int)
            occupancy_rate = df['occupied'].mean()
            sparse_data_handled = (occupancy_rate < 0.2 and len(errors) == 0)  # Successfully handled sparse data
        
        return PerformanceMetrics(
            system_name="STUMPY",
            room_name=room_name,
            test_scenario=f"room_{room_name}",
            processing_time_seconds=performance_data['processing_time'],
            pattern_discovery_time=pattern_time,
            data_loading_time=data_load_time,
            peak_memory_mb=performance_data['peak_memory_mb'],
            memory_increase_mb=performance_data['memory_increase_mb'],
            avg_cpu_percent=performance_data['avg_cpu_percent'],
            peak_cpu_percent=performance_data['peak_cpu_percent'],
            patterns_discovered=patterns_discovered,
            pattern_success_rate=validation_accuracy,
            validation_accuracy=validation_accuracy,
            events_processed=len(test_events),
            duplicate_events_removed=0,  # STUMPY doesn't do deduplication
            sparse_data_handled=sparse_data_handled,
            completed_successfully=len(errors) == 0,
            errors_encountered=errors,
            warnings_count=performance_data['warnings_count'],
            system_specific_metrics=system_metrics
        )
    
    async def test_hmm_system(self, room_name: str, test_events: List[Dict]) -> PerformanceMetrics:
        """Test the event-based HMM pattern discovery system"""
        logger.info(f"🔬 Testing HMM system for {room_name}")
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        errors = []
        patterns_discovered = 0
        validation_accuracy = 0.0
        duplicate_events_removed = 0
        system_metrics = {}
        
        try:
            # Initialize HMM pattern discovery
            hmm_discovery = EventBasedPatternDiscovery()
            
            # Track data loading time
            data_load_start = time.time()
            
            # Sample resources during processing
            monitor.sample_resources()
            
            # Run pattern discovery
            pattern_start = time.time()
            
            try:
                result = await hmm_discovery.discover_multizone_patterns(
                    room_name, test_events
                )
                
                # Extract metrics from results
                if result and 'patterns' in result:
                    patterns = result['patterns']
                    
                    # Get HMM validation accuracy
                    if 'hmm_states' in patterns and not patterns['hmm_states'].get('error'):
                        validation_accuracy = patterns['hmm_states'].get('validation_accuracy', 0.0)
                    
                    # Count discovered patterns
                    if 'discovered_patterns' in patterns:
                        dp = patterns['discovered_patterns']
                        if hasattr(dp, 'patterns_discovered'):
                            patterns_discovered = len(dp.patterns_discovered)
                        elif isinstance(dp, dict) and 'patterns_discovered' in dp:
                            patterns_discovered = len(dp['patterns_discovered'])
                    
                    # Get duplicate removal metrics
                    if 'event_sequences' in patterns:
                        es = patterns['event_sequences']
                        if hasattr(es, 'total_events') and hasattr(es, 'clean_events'):
                            duplicate_events_removed = es.total_events - es.clean_events
                        elif isinstance(es, dict):
                            duplicate_events_removed = es.get('total_events', 0) - es.get('clean_events', 0)
                    
                    # Get metadata
                    metadata = patterns.get('metadata', {})
                    system_metrics = {
                        'hmm_training_sequences': patterns.get('hmm_states', {}).get('training_sequences', 0),
                        'change_points_detected': len(patterns.get('change_points', [])),
                        'event_deduplication_rate': duplicate_events_removed / len(test_events) if test_events else 0,
                        'discovery_method': metadata.get('discovery_method', 'event_based_hmm')
                    }
                
            except Exception as e:
                errors.append(f"HMM pattern discovery failed: {str(e)}")
                logger.error(f"HMM pattern discovery error: {e}")
            
            pattern_end = time.time()
            data_load_time = pattern_start - data_load_start
            pattern_time = pattern_end - pattern_start
            
            monitor.sample_resources()
            
        except Exception as e:
            errors.append(f"HMM system error: {str(e)}")
            logger.error(f"HMM system error: {e}")
            data_load_time = 0
            pattern_time = 0
        
        # Stop monitoring and collect metrics
        performance_data = monitor.stop_monitoring()
        
        # Determine if sparse data was handled successfully
        df = pd.DataFrame(test_events) if test_events else pd.DataFrame()
        sparse_data_handled = True
        if not df.empty:
            df['occupied'] = df['state'].isin(['on', 'occupied', '1', 1]).astype(int)
            occupancy_rate = df['occupied'].mean()
            sparse_data_handled = (occupancy_rate < 0.2 and len(errors) == 0)  # Successfully handled sparse data
        
        return PerformanceMetrics(
            system_name="HMM",
            room_name=room_name,
            test_scenario=f"room_{room_name}",
            processing_time_seconds=performance_data['processing_time'],
            pattern_discovery_time=pattern_time,
            data_loading_time=data_load_time,
            peak_memory_mb=performance_data['peak_memory_mb'],
            memory_increase_mb=performance_data['memory_increase_mb'],
            avg_cpu_percent=performance_data['avg_cpu_percent'],
            peak_cpu_percent=performance_data['peak_cpu_percent'],
            patterns_discovered=patterns_discovered,
            pattern_success_rate=validation_accuracy,
            validation_accuracy=validation_accuracy,
            events_processed=len(test_events),
            duplicate_events_removed=duplicate_events_removed,
            sparse_data_handled=sparse_data_handled,
            completed_successfully=len(errors) == 0,
            errors_encountered=errors,
            warnings_count=performance_data['warnings_count'],
            system_specific_metrics=system_metrics
        )
    
    def compare_systems(self, stumpy_metrics: PerformanceMetrics, 
                       hmm_metrics: PerformanceMetrics) -> ComparisonResult:
        """Compare performance between STUMPY and HMM systems"""
        
        # Calculate improvements (positive = HMM is better)
        processing_time_improvement = (
            (stumpy_metrics.processing_time_seconds - hmm_metrics.processing_time_seconds) 
            / stumpy_metrics.processing_time_seconds * 100
            if stumpy_metrics.processing_time_seconds > 0 else 0
        )
        
        memory_improvement = (
            (stumpy_metrics.peak_memory_mb - hmm_metrics.peak_memory_mb) 
            / stumpy_metrics.peak_memory_mb * 100
            if stumpy_metrics.peak_memory_mb > 0 else 0
        )
        
        accuracy_improvement = (
            hmm_metrics.validation_accuracy - stumpy_metrics.validation_accuracy
        )
        
        # Reliability improvement (based on completion and errors)
        stumpy_reliability = 1.0 if stumpy_metrics.completed_successfully else 0.5
        hmm_reliability = 1.0 if hmm_metrics.completed_successfully else 0.5
        
        if stumpy_metrics.errors_encountered:
            stumpy_reliability -= len(stumpy_metrics.errors_encountered) * 0.1
        if hmm_metrics.errors_encountered:
            hmm_reliability -= len(hmm_metrics.errors_encountered) * 0.1
        
        reliability_improvement = hmm_reliability - stumpy_reliability
        
        # Determine recommended system
        score_stumpy = (
            stumpy_metrics.validation_accuracy * 0.4 +
            (1.0 if stumpy_metrics.completed_successfully else 0.0) * 0.3 +
            (1.0 if stumpy_metrics.sparse_data_handled else 0.0) * 0.2 +
            (1.0 - min(stumpy_metrics.processing_time_seconds / 600, 1.0)) * 0.1  # Penalty for >10min
        )
        
        score_hmm = (
            hmm_metrics.validation_accuracy * 0.4 +
            (1.0 if hmm_metrics.completed_successfully else 0.0) * 0.3 +
            (1.0 if hmm_metrics.sparse_data_handled else 0.0) * 0.2 +
            (1.0 - min(hmm_metrics.processing_time_seconds / 600, 1.0)) * 0.1
        )
        
        recommended_system = "HMM" if score_hmm > score_stumpy else "STUMPY"
        
        # Generate improvement summary
        improvements = []
        if processing_time_improvement > 5:
            improvements.append(f"Processing time: {processing_time_improvement:.1f}% faster")
        elif processing_time_improvement < -5:
            improvements.append(f"Processing time: {abs(processing_time_improvement):.1f}% slower")
        
        if memory_improvement > 5:
            improvements.append(f"Memory usage: {memory_improvement:.1f}% less")
        elif memory_improvement < -5:
            improvements.append(f"Memory usage: {abs(memory_improvement):.1f}% more")
        
        if accuracy_improvement > 0.05:
            improvements.append(f"Accuracy: {accuracy_improvement:.3f} improvement")
        elif accuracy_improvement < -0.05:
            improvements.append(f"Accuracy: {abs(accuracy_improvement):.3f} degradation")
        
        if reliability_improvement > 0.1:
            improvements.append(f"Reliability: {reliability_improvement:.2f} improvement")
        elif reliability_improvement < -0.1:
            improvements.append(f"Reliability: {abs(reliability_improvement):.2f} degradation")
        
        improvement_summary = "; ".join(improvements) if improvements else "Performance comparable"
        
        # Deployment recommendation
        if recommended_system == "HMM":
            if score_hmm - score_stumpy > 0.2:
                deployment_recommendation = "Strongly recommend HMM system deployment"
            else:
                deployment_recommendation = "Recommend HMM system deployment with monitoring"
        else:
            if score_stumpy - score_hmm > 0.2:
                deployment_recommendation = "Continue with STUMPY system"
            else:
                deployment_recommendation = "Consider gradual migration to HMM system"
        
        return ComparisonResult(
            stumpy_metrics=stumpy_metrics,
            hmm_metrics=hmm_metrics,
            processing_time_improvement=processing_time_improvement,
            memory_improvement=memory_improvement,
            accuracy_improvement=accuracy_improvement,
            reliability_improvement=reliability_improvement,
            recommended_system=recommended_system,
            improvement_summary=improvement_summary,
            deployment_recommendation=deployment_recommendation
        )
    
    async def run_room_comparison(self, room_name: str) -> ComparisonResult:
        """Run complete comparison for a single room"""
        logger.info(f"🏠 Running comparison for room: {room_name}")
        
        # Load test data
        test_events, metadata = await self.load_test_data(room_name)
        
        if not test_events:
            logger.warning(f"No test data available for {room_name}")
            return None
        
        logger.info(f"Testing with {len(test_events):,} events")
        logger.info(f"Data characteristics: {metadata}")
        
        # Force garbage collection before tests
        gc.collect()
        
        # Test STUMPY system
        logger.info("Testing STUMPY system...")
        stumpy_metrics = await self.test_stumpy_system(room_name, test_events)
        
        # Force garbage collection between tests
        gc.collect()
        await asyncio.sleep(1)  # Brief pause
        
        # Test HMM system
        logger.info("Testing HMM system...")
        hmm_metrics = await self.test_hmm_system(room_name, test_events)
        
        # Compare results
        comparison = self.compare_systems(stumpy_metrics, hmm_metrics)
        
        # Store results
        self.test_results[room_name] = {
            'metadata': metadata,
            'comparison': comparison
        }
        
        logger.info(f"✅ Comparison complete for {room_name}")
        logger.info(f"   Recommended system: {comparison.recommended_system}")
        logger.info(f"   Key improvements: {comparison.improvement_summary}")
        
        return comparison
    
    async def run_full_comparison(self) -> Dict[str, Any]:
        """Run comprehensive comparison across all test rooms"""
        logger.info("🚀 Starting comprehensive pattern discovery performance comparison")
        
        await self.initialize_database()
        
        overall_results = {
            'test_config': self.test_config,
            'room_comparisons': {},
            'summary': {},
            'recommendations': {}
        }
        
        successful_tests = []
        failed_tests = []
        
        # Test each room
        for room_name in self.test_rooms:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"TESTING ROOM: {room_name.upper()}")
                logger.info(f"{'='*60}")
                
                comparison = await self.run_room_comparison(room_name)
                
                if comparison:
                    overall_results['room_comparisons'][room_name] = asdict(comparison)
                    successful_tests.append((room_name, comparison))
                    
                    # Log immediate results
                    logger.info(f"\n📊 RESULTS FOR {room_name.upper()}:")
                    logger.info(f"   STUMPY: {comparison.stumpy_metrics.validation_accuracy:.3f} accuracy, "
                              f"{comparison.stumpy_metrics.processing_time_seconds:.1f}s, "
                              f"{comparison.stumpy_metrics.peak_memory_mb:.1f}MB")
                    logger.info(f"   HMM:    {comparison.hmm_metrics.validation_accuracy:.3f} accuracy, "
                              f"{comparison.hmm_metrics.processing_time_seconds:.1f}s, "
                              f"{comparison.hmm_metrics.peak_memory_mb:.1f}MB")
                    logger.info(f"   Winner: {comparison.recommended_system}")
                    logger.info(f"   Improvements: {comparison.improvement_summary}")
                else:
                    failed_tests.append(room_name)
                    logger.warning(f"❌ Test failed for {room_name}")
                
            except Exception as e:
                logger.error(f"❌ Error testing {room_name}: {e}")
                failed_tests.append(room_name)
        
        # Generate overall summary
        if successful_tests:
            stumpy_scores = [c.stumpy_metrics.validation_accuracy for _, c in successful_tests]
            hmm_scores = [c.hmm_metrics.validation_accuracy for _, c in successful_tests]
            
            stumpy_times = [c.stumpy_metrics.processing_time_seconds for _, c in successful_tests]
            hmm_times = [c.hmm_metrics.processing_time_seconds for _, c in successful_tests]
            
            stumpy_memory = [c.stumpy_metrics.peak_memory_mb for _, c in successful_tests]
            hmm_memory = [c.hmm_metrics.peak_memory_mb for _, c in successful_tests]
            
            hmm_wins = sum(1 for _, c in successful_tests if c.recommended_system == "HMM")
            
            overall_results['summary'] = {
                'tests_completed': len(successful_tests),
                'tests_failed': len(failed_tests),
                'hmm_wins': hmm_wins,
                'stumpy_wins': len(successful_tests) - hmm_wins,
                'average_accuracy': {
                    'stumpy': np.mean(stumpy_scores),
                    'hmm': np.mean(hmm_scores),
                    'improvement': np.mean(hmm_scores) - np.mean(stumpy_scores)
                },
                'average_processing_time': {
                    'stumpy': np.mean(stumpy_times),
                    'hmm': np.mean(hmm_times),
                    'improvement_percent': (np.mean(stumpy_times) - np.mean(hmm_times)) / np.mean(stumpy_times) * 100
                },
                'average_memory_usage': {
                    'stumpy': np.mean(stumpy_memory),
                    'hmm': np.mean(hmm_memory),
                    'improvement_percent': (np.mean(stumpy_memory) - np.mean(hmm_memory)) / np.mean(stumpy_memory) * 100
                }
            }
            
            # Generate final recommendations
            overall_results['recommendations'] = self._generate_deployment_recommendations(successful_tests)
        
        await self.db.close()
        
        logger.info(f"\n{'='*80}")
        logger.info("📋 COMPREHENSIVE COMPARISON COMPLETE")
        logger.info(f"{'='*80}")
        
        if successful_tests:
            summary = overall_results['summary']
            logger.info(f"✅ Tests completed: {summary['tests_completed']}")
            logger.info(f"🏆 HMM wins: {summary['hmm_wins']}, STUMPY wins: {summary['stumpy_wins']}")
            logger.info(f"📈 Average accuracy improvement: {summary['average_accuracy']['improvement']:.3f}")
            logger.info(f"⏱️  Average processing time improvement: {summary['average_processing_time']['improvement_percent']:.1f}%")
            logger.info(f"💾 Average memory improvement: {summary['average_memory_usage']['improvement_percent']:.1f}%")
        
        return overall_results
    
    def _generate_deployment_recommendations(self, successful_tests: List[Tuple[str, ComparisonResult]]) -> Dict[str, Any]:
        """Generate deployment recommendations based on test results"""
        
        # Analyze results by room type
        room_analysis = {}
        for room_name, comparison in successful_tests:
            room_analysis[room_name] = {
                'winner': comparison.recommended_system,
                'accuracy_improvement': comparison.accuracy_improvement,
                'processing_improvement': comparison.processing_time_improvement,
                'reliability': comparison.hmm_metrics.completed_successfully,
                'sparse_data_handling': comparison.hmm_metrics.sparse_data_handled
            }
        
        # Overall recommendation
        hmm_wins = sum(1 for _, c in successful_tests if c.recommended_system == "HMM")
        total_tests = len(successful_tests)
        
        if hmm_wins / total_tests >= 0.8:
            overall_recommendation = "DEPLOY_HMM"
            confidence = "HIGH"
        elif hmm_wins / total_tests >= 0.6:
            overall_recommendation = "GRADUAL_MIGRATION"
            confidence = "MEDIUM"
        else:
            overall_recommendation = "KEEP_STUMPY"
            confidence = "LOW"
        
        # Specific recommendations by scenario
        scenario_recommendations = {}
        
        # Sparse data handling
        sparse_rooms = [room for room, comp in successful_tests 
                       if self.test_results[room]['metadata'].get('sparse_data', False)]
        if sparse_rooms:
            sparse_success = sum(1 for room in sparse_rooms 
                               if room_analysis[room]['winner'] == 'HMM')
            scenario_recommendations['sparse_data'] = {
                'recommendation': 'HMM' if sparse_success > len(sparse_rooms) / 2 else 'STUMPY',
                'rooms_tested': sparse_rooms,
                'success_rate': sparse_success / len(sparse_rooms)
            }
        
        return {
            'overall_recommendation': overall_recommendation,
            'confidence': confidence,
            'room_analysis': room_analysis,
            'scenario_recommendations': scenario_recommendations,
            'deployment_strategy': self._get_deployment_strategy(overall_recommendation),
            'monitoring_requirements': self._get_monitoring_requirements(),
            'rollback_plan': self._get_rollback_plan()
        }
    
    def _get_deployment_strategy(self, recommendation: str) -> Dict[str, Any]:
        """Get deployment strategy based on recommendation"""
        
        strategies = {
            'DEPLOY_HMM': {
                'approach': 'Full replacement',
                'timeline': '2-4 weeks',
                'phases': [
                    'Deploy HMM system in parallel',
                    'Monitor performance for 1 week',
                    'Switch traffic to HMM system',
                    'Decommission STUMPY system'
                ],
                'risk_level': 'LOW'
            },
            'GRADUAL_MIGRATION': {
                'approach': 'Phased migration',
                'timeline': '4-8 weeks',
                'phases': [
                    'Deploy HMM for best-performing rooms',
                    'A/B test for 2 weeks',
                    'Evaluate and expand to more rooms',
                    'Complete migration over 6-8 weeks'
                ],
                'risk_level': 'MEDIUM'
            },
            'KEEP_STUMPY': {
                'approach': 'Status quo with improvements',
                'timeline': 'Ongoing',
                'phases': [
                    'Improve STUMPY system reliability',
                    'Monitor HMM system development',
                    'Re-evaluate in 3-6 months'
                ],
                'risk_level': 'LOW'
            }
        }
        
        return strategies.get(recommendation, strategies['KEEP_STUMPY'])
    
    def _get_monitoring_requirements(self) -> List[str]:
        """Get monitoring requirements for deployment"""
        return [
            'Pattern discovery success rates by room',
            'Processing time and memory usage trends',
            'Prediction accuracy validation',
            'System error rates and warnings',
            'Event processing throughput',
            'HMM model training convergence',
            'Change point detection accuracy',
            'Overall system stability metrics'
        ]
    
    def _get_rollback_plan(self) -> Dict[str, Any]:
        return {
            'triggers': [
                'Accuracy drops below 80% of baseline',
                'Processing time increases >50%',
                'System errors exceed 5% of requests',
                'Memory usage exceeds available resources'
            ],
            'procedure': [
                'Immediately switch traffic back to STUMPY',
                'Capture detailed error logs and metrics',
                'Analyze root cause of performance issues',
                'Implement fixes and re-test before retry'
            ],
            'rollback_time': '< 30 minutes',
            'data_safety': 'All pattern discoveries stored in Redis with timestamps'
        }


# Test execution functions
async def run_performance_comparison():
    """Main function to run the performance comparison"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run tests
    test_runner = PatternDiscoveryPerformanceTest()
    
    try:
        results = await test_runner.run_full_comparison()
        
        # Save results to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'pattern_discovery_comparison_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📁 Complete results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Performance comparison failed: {e}")
        raise


if __name__ == "__main__":
    # Run the performance comparison
    asyncio.run(run_performance_comparison())