#!/usr/bin/env python3
"""
HA Intent Predictor - Installation Test Suite
Comprehensive testing and validation of the deployment
"""

import asyncio
import aiohttp
import asyncpg
import redis.asyncio as redis
import logging
import json
import yaml
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import sys
import click
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float
    details: str
    error: Optional[str] = None


class InstallationTester:
    """Comprehensive installation testing suite"""
    
    def __init__(self, container_id: str, config_path: str = "/opt/ha-intent-predictor/config/app.yaml"):
        self.container_id = container_id
        self.config_path = config_path
        self.config = None
        self.results: List[TestResult] = []
        
    async def load_config(self) -> bool:
        """Load configuration from container"""
        try:
            result = subprocess.run([
                "pct", "exec", self.container_id, "--",
                "cat", self.config_path
            ], capture_output=True, text=True, check=True)
            
            self.config = yaml.safe_load(result.stdout)
            logger.info("Configuration loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False
    
    def run_container_command(self, command: List[str]) -> Tuple[bool, str, str]:
        """Run command in container and return success, stdout, stderr"""
        try:
            full_command = ["pct", "exec", self.container_id, "--"] + command
            result = subprocess.run(
                full_command, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
    
    async def test_container_basic_functionality(self) -> TestResult:
        """Test basic container functionality"""
        start_time = time.time()
        
        try:
            # Test container is running
            success, stdout, stderr = self.run_container_command(["echo", "test"])
            if not success:
                return TestResult(
                    "Container Basic Functionality",
                    False,
                    time.time() - start_time,
                    "Container not responding",
                    stderr
                )
            
            # Test basic tools
            tools_to_test = ["python3", "pip", "curl", "systemctl"]
            missing_tools = []
            
            for tool in tools_to_test:
                success, _, _ = self.run_container_command(["which", tool])
                if not success:
                    missing_tools.append(tool)
            
            if missing_tools:
                return TestResult(
                    "Container Basic Functionality",
                    False,
                    time.time() - start_time,
                    f"Missing tools: {', '.join(missing_tools)}"
                )
            
            return TestResult(
                "Container Basic Functionality",
                True,
                time.time() - start_time,
                "All basic tools available"
            )
            
        except Exception as e:
            return TestResult(
                "Container Basic Functionality",
                False,
                time.time() - start_time,
                "Exception occurred",
                str(e)
            )
    
    async def test_docker_services(self) -> TestResult:
        """Test Docker services are running"""
        start_time = time.time()
        
        try:
            # Check Docker daemon
            success, _, stderr = self.run_container_command(["systemctl", "is-active", "docker"])
            if not success:
                return TestResult(
                    "Docker Services",
                    False,
                    time.time() - start_time,
                    "Docker daemon not running",
                    stderr
                )
            
            # Check Docker Compose services
            success, stdout, stderr = self.run_container_command([
                "bash", "-c", 
                "cd /opt/ha-intent-predictor && docker compose ps --format json"
            ])
            
            if not success:
                return TestResult(
                    "Docker Services",
                    False,
                    time.time() - start_time,
                    "Docker Compose not working",
                    stderr
                )
            
            # Parse service status
            try:
                services_info = []
                for line in stdout.strip().split('\n'):
                    if line.strip():
                        service = json.loads(line)
                        services_info.append(f"{service['Name']}: {service['State']}")
                
                # Check for critical services
                critical_services = ["postgres", "redis"]
                running_services = [s for s in services_info if "running" in s.lower()]
                
                if len(running_services) >= len(critical_services):
                    return TestResult(
                        "Docker Services",
                        True,
                        time.time() - start_time,
                        f"Services running: {len(running_services)}"
                    )
                else:
                    return TestResult(
                        "Docker Services",
                        False,
                        time.time() - start_time,
                        f"Not enough services running: {services_info}"
                    )
                    
            except json.JSONDecodeError:
                return TestResult(
                    "Docker Services",
                    False,
                    time.time() - start_time,
                    "Could not parse service status",
                    stdout
                )
            
        except Exception as e:
            return TestResult(
                "Docker Services",
                False,
                time.time() - start_time,
                "Exception occurred",
                str(e)
            )
    
    async def test_database_connectivity(self) -> TestResult:
        """Test database connectivity and schema"""
        start_time = time.time()
        
        try:
            if not self.config:
                return TestResult(
                    "Database Connectivity",
                    False,
                    time.time() - start_time,
                    "Configuration not loaded"
                )
            
            db_config = self.config['database']
            
            # Test connection using docker exec
            success, stdout, stderr = self.run_container_command([
                "docker", "exec", "ha-predictor-postgres",
                "pg_isready", "-U", db_config['user'], "-d", db_config['name']
            ])
            
            if not success:
                return TestResult(
                    "Database Connectivity",
                    False,
                    time.time() - start_time,
                    "Database not accepting connections",
                    stderr
                )
            
            # Test schema exists
            success, stdout, stderr = self.run_container_command([
                "docker", "exec", "ha-predictor-postgres",
                "psql", "-U", db_config['user'], "-d", db_config['name'],
                "-t", "-c", 
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema IN ('raw_data', 'features', 'models', 'analytics');"
            ])
            
            if not success:
                return TestResult(
                    "Database Connectivity",
                    False,
                    time.time() - start_time,
                    "Could not query database schema",
                    stderr
                )
            
            table_count = int(stdout.strip())
            if table_count > 0:
                return TestResult(
                    "Database Connectivity",
                    True,
                    time.time() - start_time,
                    f"Database connected, {table_count} tables found"
                )
            else:
                return TestResult(
                    "Database Connectivity",
                    False,
                    time.time() - start_time,
                    "Database connected but no tables found"
                )
                
        except Exception as e:
            return TestResult(
                "Database Connectivity",
                False,
                time.time() - start_time,
                "Exception occurred",
                str(e)
            )
    
    async def test_redis_connectivity(self) -> TestResult:
        """Test Redis connectivity"""
        start_time = time.time()
        
        try:
            # Test Redis ping
            success, stdout, stderr = self.run_container_command([
                "docker", "exec", "ha-predictor-redis",
                "redis-cli", "ping"
            ])
            
            if not success:
                return TestResult(
                    "Redis Connectivity",
                    False,
                    time.time() - start_time,
                    "Redis not responding to ping",
                    stderr
                )
            
            if "PONG" in stdout:
                return TestResult(
                    "Redis Connectivity",
                    True,
                    time.time() - start_time,
                    "Redis responding to ping"
                )
            else:
                return TestResult(
                    "Redis Connectivity",
                    False,
                    time.time() - start_time,
                    "Redis ping returned unexpected response",
                    stdout
                )
                
        except Exception as e:
            return TestResult(
                "Redis Connectivity",
                False,
                time.time() - start_time,
                "Exception occurred",
                str(e)
            )
    
    async def test_python_environment(self) -> TestResult:
        """Test Python environment and dependencies"""
        start_time = time.time()
        
        try:
            # Test virtual environment
            success, stdout, stderr = self.run_container_command([
                "test", "-d", "/opt/ha-intent-predictor/venv"
            ])
            
            if not success:
                return TestResult(
                    "Python Environment",
                    False,
                    time.time() - start_time,
                    "Virtual environment not found"
                )
            
            # Test key dependencies
            success, stdout, stderr = self.run_container_command([
                "/opt/ha-intent-predictor/venv/bin/python", "-c",
                "import river, pandas, numpy, scikit_learn, redis, psycopg2, fastapi; print('OK')"
            ])
            
            if not success:
                return TestResult(
                    "Python Environment",
                    False,
                    time.time() - start_time,
                    "Key dependencies not importable",
                    stderr
                )
            
            if "OK" in stdout:
                return TestResult(
                    "Python Environment",
                    True,
                    time.time() - start_time,
                    "All key dependencies importable"
                )
            else:
                return TestResult(
                    "Python Environment",
                    False,
                    time.time() - start_time,
                    "Dependency import test failed",
                    stdout
                )
                
        except Exception as e:
            return TestResult(
                "Python Environment",
                False,
                time.time() - start_time,
                "Exception occurred",
                str(e)
            )
    
    async def test_systemd_services(self) -> TestResult:
        """Test systemd services"""
        start_time = time.time()
        
        try:
            services = [
                "ha-intent-predictor.service",
                "ha-predictor-ingestion.service", 
                "ha-predictor-training.service",
                "ha-predictor-api.service"
            ]
            
            service_status = {}
            
            for service in services:
                success, stdout, stderr = self.run_container_command([
                    "systemctl", "is-active", service
                ])
                
                if success and "active" in stdout:
                    service_status[service] = "active"
                else:
                    success2, stdout2, _ = self.run_container_command([
                        "systemctl", "is-enabled", service
                    ])
                    if success2 and "enabled" in stdout2:
                        service_status[service] = "enabled"
                    else:
                        service_status[service] = "inactive"
            
            active_services = [s for s, status in service_status.items() if status == "active"]
            enabled_services = [s for s, status in service_status.items() if status in ["active", "enabled"]]
            
            if len(active_services) >= 2:  # At least some services should be running
                return TestResult(
                    "Systemd Services",
                    True,
                    time.time() - start_time,
                    f"Services active: {len(active_services)}, enabled: {len(enabled_services)}"
                )
            elif len(enabled_services) >= 3:  # Services configured but maybe not started yet
                return TestResult(
                    "Systemd Services",
                    True,
                    time.time() - start_time,
                    f"Services configured: {len(enabled_services)} (may be starting up)"
                )
            else:
                return TestResult(
                    "Systemd Services",
                    False,
                    time.time() - start_time,
                    f"Not enough services configured: {service_status}"
                )
                
        except Exception as e:
            return TestResult(
                "Systemd Services",
                False,
                time.time() - start_time,
                "Exception occurred",
                str(e)
            )
    
    async def test_api_endpoints(self) -> TestResult:
        """Test API endpoints"""
        start_time = time.time()
        
        try:
            # Get container IP
            success, stdout, stderr = self.run_container_command([
                "hostname", "-I"
            ])
            
            if not success:
                return TestResult(
                    "API Endpoints",
                    False,
                    time.time() - start_time,
                    "Could not get container IP",
                    stderr
                )
            
            container_ip = stdout.strip().split()[0]
            
            # Test health endpoint
            success, stdout, stderr = self.run_container_command([
                "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                f"http://localhost:8000/health"
            ])
            
            if success and "200" in stdout:
                # Test API docs endpoint
                success2, stdout2, _ = self.run_container_command([
                    "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                    f"http://localhost:8000/docs"
                ])
                
                api_status = "Health endpoint OK"
                if success2 and "200" in stdout2:
                    api_status += ", Docs endpoint OK"
                
                return TestResult(
                    "API Endpoints",
                    True,
                    time.time() - start_time,
                    api_status
                )
            else:
                return TestResult(
                    "API Endpoints",
                    False,
                    time.time() - start_time,
                    f"Health endpoint returned: {stdout}",
                    stderr
                )
                
        except Exception as e:
            return TestResult(
                "API Endpoints",
                False,
                time.time() - start_time,
                "Exception occurred",
                str(e)
            )
    
    async def test_home_assistant_connectivity(self) -> TestResult:
        """Test Home Assistant connectivity"""
        start_time = time.time()
        
        try:
            if not self.config:
                return TestResult(
                    "Home Assistant Connectivity",
                    False,
                    time.time() - start_time,
                    "Configuration not loaded"
                )
            
            ha_config = self.config.get('home_assistant', {})
            ha_url = ha_config.get('url', '')
            ha_token = ha_config.get('token', '')
            
            if not ha_url or not ha_token:
                return TestResult(
                    "Home Assistant Connectivity",
                    False,
                    time.time() - start_time,
                    "HA URL or token not configured"
                )
            
            # Test HA API endpoint
            success, stdout, stderr = self.run_container_command([
                "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                "-H", f"Authorization: Bearer {ha_token}",
                f"{ha_url}/api/"
            ])
            
            if success and "200" in stdout:
                return TestResult(
                    "Home Assistant Connectivity",
                    True,
                    time.time() - start_time,
                    "HA API accessible"
                )
            else:
                return TestResult(
                    "Home Assistant Connectivity",
                    False,
                    time.time() - start_time,
                    f"HA API returned: {stdout}",
                    stderr
                )
                
        except Exception as e:
            return TestResult(
                "Home Assistant Connectivity",
                False,
                time.time() - start_time,
                "Exception occurred",
                str(e)
            )
    
    async def test_file_permissions(self) -> TestResult:
        """Test file permissions and ownership"""
        start_time = time.time()
        
        try:
            # Check key directories exist and are writable
            directories_to_check = [
                "/opt/ha-intent-predictor",
                "/var/log/ha-intent-predictor",
                "/opt/ha-intent-predictor/config",
                "/opt/ha-intent-predictor/venv"
            ]
            
            issues = []
            
            for directory in directories_to_check:
                success, stdout, stderr = self.run_container_command([
                    "test", "-d", directory
                ])
                
                if not success:
                    issues.append(f"Directory missing: {directory}")
                    continue
                
                # Test if writable
                success, _, _ = self.run_container_command([
                    "test", "-w", directory
                ])
                
                if not success:
                    issues.append(f"Directory not writable: {directory}")
            
            # Check key files
            files_to_check = [
                "/opt/ha-intent-predictor/config/app.yaml",
                "/etc/systemd/system/ha-intent-predictor.service"
            ]
            
            for file_path in files_to_check:
                success, _, _ = self.run_container_command([
                    "test", "-f", file_path
                ])
                
                if not success:
                    issues.append(f"File missing: {file_path}")
            
            if issues:
                return TestResult(
                    "File Permissions",
                    False,
                    time.time() - start_time,
                    f"Issues found: {'; '.join(issues)}"
                )
            else:
                return TestResult(
                    "File Permissions",
                    True,
                    time.time() - start_time,
                    "All directories and files accessible"
                )
                
        except Exception as e:
            return TestResult(
                "File Permissions",
                False,
                time.time() - start_time,
                "Exception occurred",
                str(e)
            )
    
    async def run_all_tests(self) -> List[TestResult]:
        """Run all tests and return results"""
        logger.info(f"Starting comprehensive testing for container {self.container_id}")
        
        # Load configuration first
        if not await self.load_config():
            return [TestResult(
                "Configuration Loading",
                False,
                0.0,
                "Failed to load configuration file"
            )]
        
        # Define test sequence
        tests = [
            self.test_container_basic_functionality,
            self.test_file_permissions,
            self.test_docker_services,
            self.test_database_connectivity,
            self.test_redis_connectivity,
            self.test_python_environment,
            self.test_systemd_services,
            self.test_api_endpoints,
            self.test_home_assistant_connectivity
        ]
        
        results = []
        
        for test in tests:
            logger.info(f"Running test: {test.__name__}")
            result = await test()
            results.append(result)
            
            if result.passed:
                logger.info(f"✓ {result.name}: {result.details}")
            else:
                logger.error(f"✗ {result.name}: {result.details}")
                if result.error:
                    logger.error(f"  Error: {result.error}")
        
        self.results = results
        return results
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report"""
        if not self.results:
            return "No test results available"
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        success_rate = (passed / total) * 100
        
        report = []
        report.append("=" * 80)
        report.append("HA INTENT PREDICTOR - INSTALLATION TEST REPORT")
        report.append("=" * 80)
        report.append(f"Container ID: {self.container_id}")
        report.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
        report.append("")
        
        # Summary by status
        report.append("TEST RESULTS:")
        report.append("-" * 40)
        
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            duration = f"{result.duration:.2f}s"
            report.append(f"[{status}] {result.name:<30} ({duration})")
            report.append(f"      {result.details}")
            if result.error:
                report.append(f"      Error: {result.error}")
            report.append("")
        
        # Overall assessment
        report.append("OVERALL ASSESSMENT:")
        report.append("-" * 40)
        
        if success_rate >= 90:
            report.append("🎉 EXCELLENT: Installation is working perfectly!")
        elif success_rate >= 75:
            report.append("✅ GOOD: Installation is mostly working with minor issues")
        elif success_rate >= 50:
            report.append("⚠️  WARNING: Installation has significant issues that need attention")
        else:
            report.append("❌ CRITICAL: Installation has major problems and requires troubleshooting")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


@click.command()
@click.option('--container-id', '-c', required=True, help='Container ID to test')
@click.option('--config-path', '-p', default='/opt/ha-intent-predictor/config/app.yaml', 
              help='Path to configuration file inside container')
@click.option('--output', '-o', help='Output file for test report')
@click.option('--json-output', is_flag=True, help='Output results in JSON format')
async def main(container_id: str, config_path: str, output: str, json_output: bool):
    """Run comprehensive installation tests for HA Intent Predictor"""
    
    tester = InstallationTester(container_id, config_path)
    results = await tester.run_all_tests()
    
    if json_output:
        # Output JSON format
        json_results = {
            "container_id": container_id,
            "test_date": datetime.now().isoformat(),
            "tests": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "duration": r.duration,
                    "details": r.details,
                    "error": r.error
                }
                for r in results
            ],
            "summary": {
                "total_tests": len(results),
                "passed_tests": sum(1 for r in results if r.passed),
                "success_rate": (sum(1 for r in results if r.passed) / len(results)) * 100
            }
        }
        
        output_text = json.dumps(json_results, indent=2)
    else:
        # Output human-readable format
        output_text = tester.generate_report()
    
    if output:
        with open(output, 'w') as f:
            f.write(output_text)
        logger.info(f"Test report saved to {output}")
    else:
        print(output_text)


if __name__ == "__main__":
    asyncio.run(main())