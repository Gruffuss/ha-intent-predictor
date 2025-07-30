#!/usr/bin/env python3
"""
Performance Comparison Execution Script

Runs comprehensive performance comparison between STUMPY and HMM pattern discovery systems.
Designed to run in the container environment with proper error handling and logging.

Usage:
    python scripts/run_performance_comparison.py [--rooms bedroom,office] [--output results.json]
"""

import asyncio
import argparse
import logging
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import test framework
from tests.test_pattern_discovery_performance import PatternDiscoveryPerformanceTest


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup comprehensive logging configuration"""
    
    # Create logs directory if it doesn't exist
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Default log file
    if not log_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"performance_comparison_{timestamp}.log"
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Setup handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Console: {log_level}, File: {log_file}")
    
    return logger


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run performance comparison between STUMPY and HMM pattern discovery systems"
    )
    
    parser.add_argument(
        '--rooms',
        type=str,
        default='bedroom,office,living_kitchen,bathroom,small_bathroom',
        help='Comma-separated list of rooms to test (default: all rooms)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for results (default: auto-generated with timestamp)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path (default: auto-generated with timestamp)'
    )
    
    parser.add_argument(
        '--max-events',
        type=int,
        default=50000,
        help='Maximum events per room for testing (default: 50000)'
    )
    
    parser.add_argument(
        '--test-days',
        type=int,
        default=30,
        help='Number of days of historical data to test (default: 30)'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with limited data (for development)'
    )
    
    parser.add_argument(
        '--room-only',
        type=str,
        help='Test only a specific room (for debugging)'
    )
    
    return parser.parse_args()


class PerformanceComparisonRunner:
    """Runner for performance comparison with enhanced error handling"""
    
    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configure test parameters
        self.test_config = {
            'max_events_per_room': args.max_events if not args.quick_test else 1000,
            'test_date_range_days': args.test_days if not args.quick_test else 7,
            'validation_split': 0.2,
            'sparse_data_threshold': 0.8
        }
        
        # Parse rooms to test
        if args.room_only:
            self.rooms_to_test = [args.room_only]
        else:
            self.rooms_to_test = [room.strip() for room in args.rooms.split(',')]
        
        self.logger.info(f"Configured to test rooms: {self.rooms_to_test}")
        self.logger.info(f"Test configuration: {self.test_config}")
    
    async def run_comparison(self):
        """Run the comprehensive performance comparison"""
        
        self.logger.info("🚀 Starting Performance Comparison")
        self.logger.info(f"Testing rooms: {', '.join(self.rooms_to_test)}")
        
        start_time = datetime.now()
        
        try:
            # Initialize test runner
            test_runner = PatternDiscoveryPerformanceTest()
            
            # Override test configuration
            test_runner.test_config.update(self.test_config)
            test_runner.test_rooms = self.rooms_to_test
            
            self.logger.info("🔧 Test runner initialized")
            
            # Run comprehensive comparison
            results = await test_runner.run_full_comparison()
            
            # Calculate total runtime
            end_time = datetime.now()
            total_runtime = (end_time - start_time).total_seconds()
            
            # Add runtime metadata
            results['execution_metadata'] = {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_runtime_seconds': total_runtime,
                'rooms_tested': self.rooms_to_test,
                'test_configuration': self.test_config,
                'command_line_args': vars(self.args)
            }
            
            self.logger.info(f"✅ Comparison completed successfully in {total_runtime:.1f} seconds")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Performance comparison failed: {e}")
            self.logger.error(f"Error details:", exc_info=True)
            raise
    
    def save_results(self, results: dict, output_file: str = None):
        """Save results to JSON file with proper formatting"""
        
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"performance_comparison_results_{timestamp}.json"
        
        # Ensure results directory exists
        output_path = Path(output_file)
        if not output_path.is_absolute():
            results_dir = project_root / "results"
            results_dir.mkdir(exist_ok=True)
            output_path = results_dir / output_file
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            
            self.logger.info(f"📁 Results saved to: {output_path}")
            self.logger.info(f"📊 File size: {output_path.stat().st_size / 1024:.1f} KB")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {e}")
            raise
    
    def print_summary(self, results: dict):
        """Print a comprehensive summary of results"""
        
        print("\n" + "="*80)
        print("📋 PERFORMANCE COMPARISON SUMMARY")
        print("="*80)
        
        # Execution summary
        metadata = results.get('execution_metadata', {})
        print(f"🕐 Execution time: {metadata.get('total_runtime_seconds', 0):.1f} seconds")
        print(f"🏠 Rooms tested: {', '.join(metadata.get('rooms_tested', []))}")
        
        # Overall results
        summary = results.get('summary', {})
        if summary:
            print(f"\n📊 OVERALL RESULTS:")
            print(f"   Tests completed: {summary.get('tests_completed', 0)}")
            print(f"   Tests failed: {summary.get('tests_failed', 0)}")
            print(f"   HMM wins: {summary.get('hmm_wins', 0)}")
            print(f"   STUMPY wins: {summary.get('stumpy_wins', 0)}")
            
            # Accuracy comparison
            accuracy = summary.get('average_accuracy', {})
            if accuracy:
                print(f"\n🎯 ACCURACY COMPARISON:")
                print(f"   STUMPY average: {accuracy.get('stumpy', 0):.3f}")
                print(f"   HMM average: {accuracy.get('hmm', 0):.3f}")
                print(f"   Improvement: {accuracy.get('improvement', 0):.3f}")
            
            # Performance comparison
            timing = summary.get('average_processing_time', {})
            if timing:
                print(f"\n⏱️  PROCESSING TIME COMPARISON:")
                print(f"   STUMPY average: {timing.get('stumpy', 0):.1f}s")
                print(f"   HMM average: {timing.get('hmm', 0):.1f}s")
                print(f"   Improvement: {timing.get('improvement_percent', 0):.1f}%")
            
            # Memory comparison
            memory = summary.get('average_memory_usage', {})
            if memory:
                print(f"\n💾 MEMORY USAGE COMPARISON:")
                print(f"   STUMPY average: {memory.get('stumpy', 0):.1f} MB")
                print(f"   HMM average: {memory.get('hmm', 0):.1f} MB")
                print(f"   Improvement: {memory.get('improvement_percent', 0):.1f}%")
        
        # Room-by-room results
        room_comparisons = results.get('room_comparisons', {})
        if room_comparisons:
            print(f"\n🏠 ROOM-BY-ROOM RESULTS:")
            for room_name, comparison in room_comparisons.items():
                winner = comparison.get('recommended_system', 'Unknown')
                accuracy_imp = comparison.get('accuracy_improvement', 0)
                time_imp = comparison.get('processing_time_improvement', 0)
                
                print(f"   {room_name.ljust(15)}: {winner.ljust(8)} "
                      f"(accuracy: {accuracy_imp:+.3f}, "
                      f"time: {time_imp:+.1f}%)")
        
        # Recommendations
        recommendations = results.get('recommendations', {})
        if recommendations:
            print(f"\n🎯 DEPLOYMENT RECOMMENDATIONS:")
            overall_rec = recommendations.get('overall_recommendation', 'Unknown')
            confidence = recommendations.get('confidence', 'Unknown')
            print(f"   Overall: {overall_rec} (confidence: {confidence})")
            
            strategy = recommendations.get('deployment_strategy', {})
            if strategy:
                print(f"   Approach: {strategy.get('approach', 'Unknown')}")
                print(f"   Timeline: {strategy.get('timeline', 'Unknown')}")
                print(f"   Risk level: {strategy.get('risk_level', 'Unknown')}")
        
        print("="*80)


async def main():
    """Main execution function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    logger.info("Performance comparison script started")
    logger.info(f"Command line arguments: {vars(args)}")
    
    try:
        # Initialize runner
        runner = PerformanceComparisonRunner(args)
        
        # Run comparison
        results = await runner.run_comparison()
        
        # Save results
        output_file = runner.save_results(results, args.output)
        
        # Print summary
        runner.print_summary(results)
        
        logger.info("🎉 Performance comparison completed successfully")
        logger.info(f"📁 Results saved to: {output_file}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Performance comparison interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Performance comparison failed: {e}")
        logger.error("Error details:", exc_info=True)
        return 1


if __name__ == "__main__":
    # Ensure we're running with asyncio
    exit_code = asyncio.run(main())
    sys.exit(exit_code)