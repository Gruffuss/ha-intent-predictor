#!/usr/bin/env python3
"""
Validation Script for Performance Comparison Setup

Validates that all required components are available and functional before
running the comprehensive performance comparison.

Checks:
1. Required imports and dependencies
2. Database connectivity
3. Pattern discovery systems availability
4. Sample data access
5. Basic functionality tests

Usage:
    python scripts/validate_comparison_setup.py
"""

import asyncio
import logging
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComparisonSetupValidator:
    """Validates setup for performance comparison"""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.errors = []
    
    def log_check(self, check_name: str, success: bool, details: str = ""):
        """Log the result of a validation check"""
        if success:
            self.checks_passed += 1
            logger.info(f"✅ {check_name}")
            if details:
                logger.info(f"   {details}")
        else:
            self.checks_failed += 1
            logger.error(f"❌ {check_name}")
            if details:
                logger.error(f"   {details}")
    
    def validate_imports(self):
        """Validate that all required imports are available"""
        logger.info("🔍 Validating imports...")
        
        import_checks = [
            ('numpy', 'import numpy as np'),
            ('pandas', 'import pandas as pd'),
            ('psutil', 'import psutil'),
            ('sqlalchemy', 'from sqlalchemy import text'),
            ('asyncio', 'import asyncio'),
            ('hmmlearn', 'from hmmlearn import hmm'),
            ('sklearn', 'from sklearn.preprocessing import StandardScaler'),
            ('scipy', 'from scipy.ndimage import gaussian_filter1d'),
        ]
        
        for name, import_statement in import_checks:
            try:
                exec(import_statement)
                self.log_check(f"Import {name}", True)
            except ImportError as e:
                self.log_check(f"Import {name}", False, str(e))
                self.errors.append(f"Missing dependency: {name}")
    
    def validate_project_structure(self):
        """Validate project structure and required files"""
        logger.info("🔍 Validating project structure...")
        
        required_files = [
            'src/learning/pattern_discovery.py',
            'src/learning/event_based_pattern_discovery.py',
            'src/learning/hmm_predictor.py',
            'src/storage/timeseries_db.py',
            'config/config_loader.py',
            'tests/test_pattern_discovery_performance.py'
        ]
        
        for file_path in required_files:
            full_path = project_root / file_path
            exists = full_path.exists()
            self.log_check(f"File exists: {file_path}", exists)
            if not exists:
                self.errors.append(f"Missing required file: {file_path}")
    
    def validate_pattern_discovery_imports(self):
        """Validate that pattern discovery systems can be imported"""
        logger.info("🔍 Validating pattern discovery system imports...")
        
        try:
            from src.learning.pattern_discovery import PatternDiscovery
            self.log_check("STUMPY PatternDiscovery import", True)
        except Exception as e:
            self.log_check("STUMPY PatternDiscovery import", False, str(e))
            self.errors.append(f"Cannot import STUMPY PatternDiscovery: {e}")
        
        try:
            from src.learning.event_based_pattern_discovery import EventBasedPatternDiscovery
            self.log_check("HMM EventBasedPatternDiscovery import", True)
        except Exception as e:
            self.log_check("HMM EventBasedPatternDiscovery import", False, str(e))
            self.errors.append(f"Cannot import HMM EventBasedPatternDiscovery: {e}")
    
    async def validate_database_connection(self):
        """Validate database connectivity"""
        logger.info("🔍 Validating database connection...")
        
        try:
            from config.config_loader import ConfigLoader
            from src.storage.timeseries_db import TimescaleDBManager
            
            config = ConfigLoader()
            db_config = config.get("database.timescale")
            
            db = TimescaleDBManager(
                f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@"
                f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            
            await db.initialize()
            
            # Test basic query
            async with db.engine.begin() as conn:
                from sqlalchemy import text
                result = await conn.execute(text("SELECT COUNT(*) FROM sensor_events LIMIT 1"))
                count = result.fetchone()[0]
                
                await db.close()
                
                self.log_check("Database connection", True, f"Found {count:,} sensor events")
                
        except Exception as e:
            self.log_check("Database connection", False, str(e))
            self.errors.append(f"Database connection failed: {e}")
    
    async def validate_sample_data_access(self):
        """Validate that sample data can be accessed for testing"""
        logger.info("🔍 Validating sample data access...")
        
        try:
            from config.config_loader import ConfigLoader
            from src.storage.timeseries_db import TimescaleDBManager
            
            config = ConfigLoader()
            db_config = config.get("database.timescale")
            
            db = TimescaleDBManager(
                f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@"
                f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            
            await db.initialize()
            
            # Test room-specific data access
            test_rooms = ['bedroom', 'office', 'living_kitchen']
            room_sensors = {
                'bedroom': ['binary_sensor.bedroom_presence_sensor_full_bedroom'],
                'office': ['binary_sensor.office_presence_full_office'],
                'living_kitchen': [
                    'binary_sensor.presence_livingroom_full',
                    'binary_sensor.kitchen_pressence_full_kitchen'
                ]
            }
            
            async with db.engine.begin() as conn:
                from sqlalchemy import text
                
                for room_name in test_rooms:
                    sensors = room_sensors[room_name]
                    sensor_conditions = [f"entity_id = '{sensor}'" for sensor in sensors]
                    sensor_filter = " OR ".join(sensor_conditions)
                    
                    query = f"""
                        SELECT COUNT(*) 
                        FROM sensor_events 
                        WHERE ({sensor_filter})
                        AND timestamp >= NOW() - INTERVAL '7 days'
                    """
                    
                    result = await conn.execute(text(query))
                    count = result.fetchone()[0]
                    
                    has_data = count > 0
                    self.log_check(f"Sample data for {room_name}", has_data, 
                                 f"{count:,} events in last 7 days")
                    
                    if not has_data:
                        self.errors.append(f"No recent data for {room_name}")
            
            await db.close()
            
        except Exception as e:
            self.log_check("Sample data access", False, str(e))
            self.errors.append(f"Sample data access failed: {e}")
    
    async def validate_basic_functionality(self):
        """Validate basic functionality of both systems"""
        logger.info("🔍 Validating basic functionality...")
        
        # Test STUMPY system initialization
        try:
            from src.learning.pattern_discovery import PatternDiscovery
            stumpy_discovery = PatternDiscovery()
            self.log_check("STUMPY system initialization", True)
        except Exception as e:
            self.log_check("STUMPY system initialization", False, str(e))
            self.errors.append(f"STUMPY initialization failed: {e}")
        
        # Test HMM system initialization
        try:
            from src.learning.event_based_pattern_discovery import EventBasedPatternDiscovery
            hmm_discovery = EventBasedPatternDiscovery()
            self.log_check("HMM system initialization", True)
        except Exception as e:
            self.log_check("HMM system initialization", False, str(e))
            self.errors.append(f"HMM initialization failed: {e}")
    
    def validate_performance_test_framework(self):
        """Validate the performance test framework"""
        logger.info("🔍 Validating performance test framework...")
        
        try:
            from tests.test_pattern_discovery_performance import (
                PatternDiscoveryPerformanceTest,
                PerformanceMetrics,
                ComparisonResult
            )
            
            # Test initialization
            test_runner = PatternDiscoveryPerformanceTest()
            self.log_check("Performance test framework import", True)
            
            # Validate test configuration
            if hasattr(test_runner, 'test_config') and hasattr(test_runner, 'test_rooms'):
                self.log_check("Performance test configuration", True, 
                             f"Testing {len(test_runner.test_rooms)} rooms")
            else:
                self.log_check("Performance test configuration", False, 
                             "Missing test configuration")
                self.errors.append("Performance test framework not properly configured")
                
        except Exception as e:
            self.log_check("Performance test framework", False, str(e))
            self.errors.append(f"Performance test framework failed: {e}")
    
    async def run_all_validations(self):
        """Run all validation checks"""
        logger.info("🚀 Starting validation checks...")
        
        self.validate_imports()
        self.validate_project_structure()
        self.validate_pattern_discovery_imports()
        await self.validate_database_connection()
        await self.validate_sample_data_access()
        await self.validate_basic_functionality()
        self.validate_performance_test_framework()
        
        # Summary
        total_checks = self.checks_passed + self.checks_failed
        logger.info(f"\n📊 VALIDATION SUMMARY:")
        logger.info(f"   Total checks: {total_checks}")
        logger.info(f"   Passed: {self.checks_passed}")
        logger.info(f"   Failed: {self.checks_failed}")
        
        if self.checks_failed == 0:
            logger.info("✅ All validation checks passed! System ready for performance comparison.")
            return True
        else:
            logger.error(f"❌ {self.checks_failed} validation checks failed:")
            for error in self.errors:
                logger.error(f"   - {error}")
            logger.error("Please fix these issues before running the performance comparison.")
            return False


async def main():
    """Main validation function"""
    
    logger.info("🧪 Performance Comparison Setup Validation")
    logger.info("=" * 60)
    
    try:
        validator = ComparisonSetupValidator()
        success = await validator.run_all_validations()
        
        logger.info("=" * 60)
        
        if success:
            logger.info("🎉 Validation completed successfully!")
            logger.info("You can now run the performance comparison with:")
            logger.info("   python scripts/run_performance_comparison.py")
            return 0
        else:
            logger.error("💥 Validation failed!")
            logger.error("Please fix the issues above before proceeding.")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Validation script failed: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)