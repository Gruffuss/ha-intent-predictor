"""
Data Cleaning Module for Home Assistant Sensor Events

This module provides comprehensive data cleaning capabilities for Home Assistant
occupancy sensor data, specifically designed for HMM training preparation.

Components:
- EventDeduplicator: High-performance deduplication engine
- Database migrations: Schema enhancements for efficient processing
- Utility functions: Analysis and batch processing tools

The main goal is to transform raw sensor events into clean, transition-only
data suitable for Hidden Markov Model training.
"""

from .event_deduplicator import EventDeduplicator, DeduplicationMetrics, analyze_duplicates, clean_historical_data

__all__ = [
    'EventDeduplicator',
    'DeduplicationMetrics', 
    'analyze_duplicates',
    'clean_historical_data'
]