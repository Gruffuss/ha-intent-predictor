#!/usr/bin/env python3
"""
HMM Accuracy Validation Script

Tests the new event-based HMM pattern learning system on historical Home Assistant data
to validate >85% accuracy target, focusing on the challenging bedroom sparse data case.

Critical Success Criteria:
- Bedroom accuracy >85% (hardest case with 84% sparse data)
- All rooms show improved accuracy vs baseline
- System completes without hangs or errors  
- Processing time reasonable (<30 minutes for full dataset)
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
import json
import sys
import os

# Add src to path for imports
sys.path.append('/opt/ha-intent-predictor/src')
sys.path.append('/opt/ha-intent-predictor')

from src.learning.hmm_predictor import HMMPredictor, OccupancySequence
from src.learning.event_based_pattern_discovery import EventBasedPatternDiscovery
from src.data_cleaning.event_deduplicator import EventDeduplicator
from src.storage.timeseries_db import TimescaleDBManager
from config.config_loader import ConfigLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HMMAccuracyValidator:
    """Validates HMM system accuracy on historical Home Assistant data"""
    
    def __init__(self):
        self.config = ConfigLoader()
        self.rooms = ['bedroom', 'office', 'living_kitchen', 'bathroom', 'small_bathroom']
        self.results = {}
        
        # Initialize components
        self.db_manager = None
        self.deduplicator = None
        self.pattern_discovery = None
        
    async def initialize(self):
        """Initialize database connections and components"""
        try:
            # Initialize TimescaleDB connection
            db_config = self.config.get("database.timescale")
            connection_string = f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            
            self.db_manager = TimescaleDBManager(connection_string)
            await self.db_manager.initialize()
            
            # Initialize event deduplicator
            self.deduplicator = EventDeduplicator(connection_string)
            await self.deduplicator.initialize()
            
            # Initialize event-based pattern discovery
            self.pattern_discovery = EventBasedPatternDiscovery()
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
    
    async def validate_bedroom_accuracy(self) -> Dict[str, Any]:
        """
        Validate HMM accuracy on bedroom data (84% sparse - hardest case)
        This is the critical test for >85% accuracy target
        """
        logger.info("🎯 Starting bedroom accuracy validation (84% sparse data)")
        
        try:
            # Fetch bedroom occupancy data (last 30 days for manageable test)
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=30)
            
            bedroom_events = await self._fetch_room_events('bedroom', start_date, end_date)
            logger.info(f"📊 Bedroom data: {len(bedroom_events)} events")
            
            if len(bedroom_events) < 100:
                logger.warning("⚠️ Insufficient bedroom data for validation")
                return {'accuracy': 0.0, 'reason': 'insufficient_data'}
            
            # Clean the data first
            logger.info("🧹 Cleaning bedroom events...")
            clean_events = await self._clean_events(bedroom_events)
            logger.info(f"✨ Clean bedroom events: {len(clean_events)}")
            
            # Calculate sparsity
            occupied_count = sum(1 for event in clean_events if event.get('occupied', False))
            sparsity = 1.0 - (occupied_count / len(clean_events))
            logger.info(f"📈 Bedroom sparsity: {sparsity:.1%} (expecting ~84%)")
            
            # Train HMM on bedroom data
            logger.info("🧠 Training HMM on bedroom data...")
            hmm_predictor = HMMPredictor('bedroom', {'sparse_threshold': 0.8})
            
            # Convert events to sequences
            sequences = await self._events_to_sequences(clean_events, 'bedroom')
            logger.info(f"🔄 Created {len(sequences)} training sequences")
            
            if len(sequences) < 5:
                logger.warning("⚠️ Insufficient sequences for HMM training")
                return {'accuracy': 0.0, 'reason': 'insufficient_sequences'}
            
            # Add sequences to HMM
            for sequence in sequences:
                await hmm_predictor.add_training_sequence(sequence)
            
            # Train the model
            await hmm_predictor.train()
            
            # Validate accuracy using time-series cross validation
            accuracy = await self._validate_hmm_accuracy(hmm_predictor, sequences)
            
            result = {
                'room': 'bedroom',
                'accuracy': accuracy,
                'sparsity': sparsity,
                'events_processed': len(bedroom_events),
                'clean_events': len(clean_events),
                'training_sequences': len(sequences),
                'target_met': accuracy >= 0.85,
                'validation_method': 'time_series_cross_validation'
            }
            
            logger.info(f"🎯 Bedroom Validation Results:")
            logger.info(f"   Accuracy: {accuracy:.1%}")
            logger.info(f"   Target (>85%): {'✅ MET' if accuracy >= 0.85 else '❌ NOT MET'}")
            logger.info(f"   Sparsity: {sparsity:.1%}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Bedroom validation failed: {e}")
            return {'accuracy': 0.0, 'error': str(e)}
    
    async def validate_all_rooms(self) -> Dict[str, Any]:
        """Validate HMM accuracy across all room types"""
        logger.info("🏠 Starting validation for all rooms")
        
        all_results = {}
        
        for room in self.rooms:
            logger.info(f"🚪 Validating room: {room}")
            
            try:
                if room == 'bedroom':
                    # Use detailed bedroom validation
                    result = await self.validate_bedroom_accuracy()
                else:
                    # Simplified validation for other rooms
                    result = await self._validate_room_accuracy(room)
                
                all_results[room] = result
                
                accuracy = result.get('accuracy', 0.0)
                logger.info(f"   {room}: {accuracy:.1%} accuracy")
                
            except Exception as e:
                logger.error(f"❌ Validation failed for {room}: {e}")
                all_results[room] = {'accuracy': 0.0, 'error': str(e)}
        
        # Calculate overall statistics
        accuracies = [r.get('accuracy', 0.0) for r in all_results.values() if 'error' not in r]
        overall_accuracy = np.mean(accuracies) if accuracies else 0.0
        
        summary = {
            'overall_accuracy': overall_accuracy,
            'rooms_tested': len(all_results),
            'successful_tests': len(accuracies),
            'target_met': overall_accuracy >= 0.85,
            'bedroom_target_met': all_results.get('bedroom', {}).get('target_met', False),
            'room_results': all_results
        }
        
        logger.info(f"🏆 Overall Validation Summary:")
        logger.info(f"   Overall Accuracy: {overall_accuracy:.1%}")
        logger.info(f"   Bedroom Target (>85%): {'✅ MET' if summary['bedroom_target_met'] else '❌ NOT MET'}")
        logger.info(f"   Successful Tests: {len(accuracies)}/{len(all_results)}")
        
        return summary
    
    async def _validate_room_accuracy(self, room: str) -> Dict[str, Any]:
        """Validate accuracy for a specific room (simplified version)"""
        try:
            # Fetch recent data
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=14)  # 2 weeks for other rooms
            
            events = await self._fetch_room_events(room, start_date, end_date)
            
            if len(events) < 50:
                return {'accuracy': 0.0, 'reason': 'insufficient_data'}
            
            # Clean events
            clean_events = await self._clean_events(events)
            
            # Create HMM predictor
            hmm_predictor = HMMPredictor(room)
            
            # Convert to sequences
            sequences = await self._events_to_sequences(clean_events, room)
            
            if len(sequences) < 3:
                return {'accuracy': 0.0, 'reason': 'insufficient_sequences'}
            
            # Train and validate
            for sequence in sequences:
                await hmm_predictor.add_training_sequence(sequence)
            
            await hmm_predictor.train()
            accuracy = await self._validate_hmm_accuracy(hmm_predictor, sequences)
            
            return {
                'room': room,
                'accuracy': accuracy,
                'events_processed': len(events),
                'clean_events': len(clean_events),
                'training_sequences': len(sequences),
                'target_met': accuracy >= 0.85
            }
            
        except Exception as e:
            logger.error(f"❌ Room validation failed for {room}: {e}")
            return {'accuracy': 0.0, 'error': str(e)}
    
    async def _fetch_room_events(self, room: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch occupancy events for a specific room"""
        try:
            query = """
            SELECT timestamp, entity_id, state, room, sensor_type, attributes
            FROM sensor_events 
            WHERE room = :room 
                AND timestamp BETWEEN :start_date AND :end_date
                AND state IN ('on', 'off', 'occupied', 'unoccupied', '1', '0')
            ORDER BY timestamp
            """
            
            async with self.db_manager.session_factory() as session:
                from sqlalchemy import text
                result = await session.execute(
                    text(query), 
                    {
                        'room': room, 
                        'start_date': start_date, 
                        'end_date': end_date
                    }
                )
                
                events = []
                for row in result.fetchall():
                    events.append({
                        'timestamp': row[0],
                        'entity_id': row[1],
                        'state': row[2],
                        'room': row[3],
                        'sensor_type': row[4],
                        'attributes': row[5] or {},
                        'occupied': row[2] in ['on', 'occupied', '1']
                    })
                
                return events
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch events for {room}: {e}")
            return []
    
    async def _clean_events(self, events: List[Dict]) -> List[Dict]:
        """Clean events using the event deduplicator"""
        try:
            # Simple in-memory deduplication for validation
            cleaned = []
            last_state_by_entity = {}
            
            for event in events:
                entity_id = event['entity_id']
                current_state = event['state']
                
                # Only keep state changes
                if entity_id not in last_state_by_entity or last_state_by_entity[entity_id] != current_state:
                    cleaned.append(event)
                    last_state_by_entity[entity_id] = current_state
            
            return cleaned
            
        except Exception as e:
            logger.error(f"❌ Event cleaning failed: {e}")
            return events  # Return original events if cleaning fails
    
    async def _events_to_sequences(self, events: List[Dict], room: str) -> List[OccupancySequence]:
        """Convert cleaned events to HMM training sequences"""
        try:
            if not events:
                return []
            
            # Group events by day to create daily sequences
            events_by_day = {}
            for event in events:
                day = event['timestamp'].date()
                if day not in events_by_day:
                    events_by_day[day] = []
                events_by_day[day].append(event)
            
            sequences = []
            
            for day, day_events in events_by_day.items():
                if len(day_events) < 5:  # Need minimum events for sequence
                    continue
                
                # Sort by timestamp
                day_events.sort(key=lambda x: x['timestamp'])
                
                # Create sequence
                timestamps = [event['timestamp'] for event in day_events]
                occupancy = [event['occupied'] for event in day_events]
                
                # Create simple features
                features = []
                for event in day_events:
                    feature_dict = {
                        'hour': event['timestamp'].hour,
                        'day_of_week': event['timestamp'].weekday(),
                        'recent_room_activity': 1 if event['occupied'] else 0,
                        'sensor_activations_1h': 1,
                        'time_since_last_activity': 0
                    }
                    features.append(feature_dict)
                
                sequence = OccupancySequence(
                    timestamps=timestamps,
                    occupancy=occupancy,
                    features=features,
                    room_id=room
                )
                
                sequences.append(sequence)
            
            return sequences
            
        except Exception as e:
            logger.error(f"❌ Failed to convert events to sequences: {e}")
            return []
    
    async def _validate_hmm_accuracy(self, hmm_predictor: HMMPredictor, sequences: List[OccupancySequence]) -> float:
        """Validate HMM accuracy using time-series cross validation"""
        try:
            if len(sequences) < 3:
                return 0.0
            
            correct_predictions = 0
            total_predictions = 0
            
            # Use time-series cross validation
            for i in range(2, len(sequences)):  # Start from 3rd sequence
                try:
                    # Use previous sequences as training
                    train_sequences = sequences[:i]
                    test_sequence = sequences[i]
                    
                    # Create fresh HMM for this fold
                    fold_hmm = HMMPredictor(hmm_predictor.room_id, hmm_predictor.config)
                    
                    # Train on previous sequences
                    for seq in train_sequences:
                        await fold_hmm.add_training_sequence(seq)
                    
                    await fold_hmm.train()
                    
                    # Test on current sequence (use first 80% to predict last 20%)
                    test_split = int(0.8 * len(test_sequence.occupancy))
                    if test_split < 2:
                        continue
                    
                    test_features = test_sequence.features[test_split-1]  # Use last training feature
                    
                    # Make prediction
                    prediction = await fold_hmm.predict_occupancy(test_features, 15)
                    predicted_occupied = prediction.probability > 0.5
                    
                    # Check actual occupancy in test portion
                    actual_occupied = any(test_sequence.occupancy[test_split:])
                    
                    if predicted_occupied == actual_occupied:
                        correct_predictions += 1
                    total_predictions += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️ Validation fold failed: {e}")
                    continue
            
            if total_predictions == 0:
                return 0.0
            
            accuracy = correct_predictions / total_predictions
            return accuracy
            
        except Exception as e:
            logger.error(f"❌ HMM accuracy validation failed: {e}")
            return 0.0
    
    async def save_results(self, results: Dict[str, Any]):
        """Save validation results to file"""
        try:
            results['validation_timestamp'] = datetime.now(timezone.utc).isoformat()
            results['validation_type'] = 'hmm_accuracy_validation'
            
            filename = f"/opt/ha-intent-predictor/hmm_validation_results_{int(time.time())}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"💾 Results saved to: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
    
    async def close(self):
        """Clean up resources"""
        try:
            if self.db_manager:
                await self.db_manager.close()
            if self.deduplicator:
                await self.deduplicator.close()
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

async def main():
    """Main validation workflow"""
    logger.info("🚀 Starting HMM Accuracy Validation")
    
    validator = HMMAccuracyValidator()
    
    try:
        # Initialize
        await validator.initialize()
        
        # Run bedroom validation (critical test)
        logger.info("=" * 60)
        logger.info("🎯 CRITICAL TEST: Bedroom Accuracy (84% sparse data)")
        logger.info("=" * 60)
        
        bedroom_result = await validator.validate_bedroom_accuracy()
        
        # Check if bedroom meets target
        bedroom_target_met = bedroom_result.get('target_met', False)
        bedroom_accuracy = bedroom_result.get('accuracy', 0.0)
        
        if bedroom_target_met:
            logger.info("🎉 BEDROOM TARGET MET! Proceeding with full validation...")
            
            # Run full validation
            logger.info("=" * 60)
            logger.info("🏠 FULL SYSTEM VALIDATION")
            logger.info("=" * 60)
            
            full_results = await validator.validate_all_rooms()
            
            # Save results
            await validator.save_results(full_results)
            
            # Final summary
            overall_accuracy = full_results.get('overall_accuracy', 0.0)
            logger.info("=" * 60)
            logger.info("🏆 FINAL VALIDATION RESULTS")
            logger.info("=" * 60)
            logger.info(f"Overall Accuracy: {overall_accuracy:.1%}")
            logger.info(f"Bedroom Accuracy: {bedroom_accuracy:.1%}")
            logger.info(f"Target Achievement: {'✅ SUCCESS' if bedroom_target_met else '❌ FAILED'}")
            
        else:
            logger.warning("⚠️ BEDROOM TARGET NOT MET - System needs improvement")
            logger.info(f"Bedroom Accuracy: {bedroom_accuracy:.1%} (target: >85%)")
            
            # Save bedroom results
            await validator.save_results({'bedroom_test': bedroom_result})
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return 1
    
    finally:
        await validator.close()
    
    logger.info("✅ Validation completed")
    return 0

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)