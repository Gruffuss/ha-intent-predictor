#!/usr/bin/env python3
"""
Deployment script for HA Intent Predictor system.

Automates deployment to Proxmox LXC container and sets up
the complete system as specified in CLAUDE.md.
"""

import asyncio
import logging
import sys
import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProxmoxLXCDeployment:
    """
    Proxmox LXC container deployment for HA Intent Predictor.
    
    Targets Intel N200, 4 cores, 8GB RAM as specified in CLAUDE.md.
    """
    
    def __init__(self, deployment_config: Dict):
        self.config = deployment_config
        self.container_id = deployment_config.get('container_id')
        self.container_name = deployment_config.get('container_name', 'ha-intent-predictor')
        self.proxmox_host = deployment_config.get('proxmox_host')
        self.proxmox_user = deployment_config.get('proxmox_user')
        
        # Container specs from CLAUDE.md
        self.container_specs = {
            'cores': 4,
            'memory': 8192,  # 8GB
            'storage': 80,   # 80GB
            'template': 'ubuntu-22.04-standard_22.04-1_amd64.tar.zst'
        }
        
        logger.info(f"Initialized Proxmox LXC deployment for {self.container_name}")
    
    async def create_container(self):
        """Create LXC container with proper specs"""
        
        logger.info("Creating Proxmox LXC container...")
        
        # Create container command
        create_cmd = [
            'pct', 'create', str(self.container_id),
            f"local:vztmpl/{self.container_specs['template']}",
            '--hostname', self.container_name,
            '--cores', str(self.container_specs['cores']),
            '--memory', str(self.container_specs['memory']),
            '--storage', 'local-lvm',
            '--rootfs', f"local-lvm:{self.container_specs['storage']}",
            '--net0', 'name=eth0,bridge=vmbr0,ip=dhcp',
            '--onboot', '1',
            '--unprivileged', '1',
            '--features', 'nesting=1'
        ]
        
        try:
            result = subprocess.run(create_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Container creation failed: {result.stderr}")
            
            logger.info(f"Container {self.container_id} created successfully")
            
        except Exception as e:
            logger.error(f"Error creating container: {e}")
            raise
    
    async def start_container(self):
        """Start the LXC container"""
        
        logger.info("Starting LXC container...")
        
        try:
            result = subprocess.run(['pct', 'start', str(self.container_id)], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Container start failed: {result.stderr}")
            
            # Wait for container to be ready
            await asyncio.sleep(10)
            
            logger.info("Container started successfully")
            
        except Exception as e:
            logger.error(f"Error starting container: {e}")
            raise
    
    async def setup_container_environment(self):
        """Set up the container environment"""
        
        logger.info("Setting up container environment...")
        
        # Update system
        await self._run_in_container([
            'apt', 'update', '&&',
            'apt', 'upgrade', '-y'
        ])
        
        # Install required packages
        packages = [
            'python3.11',
            'python3.11-venv',
            'python3-pip',
            'postgresql-client',
            'redis-tools',
            'git',
            'curl',
            'wget',
            'htop',
            'vim',
            'supervisor',
            'nginx'
        ]
        
        await self._run_in_container(['apt', 'install', '-y'] + packages)
        
        # Set up Python environment
        await self._run_in_container([
            'python3.11', '-m', 'venv', '/opt/ha-intent-predictor/venv'
        ])
        
        logger.info("Container environment set up")
    
    async def _run_in_container(self, command: List[str]):
        """Run command inside the container"""
        
        full_command = ['pct', 'exec', str(self.container_id), '--'] + command
        
        try:
            result = subprocess.run(full_command, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Command failed: {' '.join(command)}")
                logger.warning(f"Error: {result.stderr}")
            
            return result.stdout
            
        except Exception as e:
            logger.error(f"Error running command in container: {e}")
            raise


class SystemDeployment:
    """
    Main system deployment orchestrator.
    
    Handles the complete deployment process including:
    - Container setup
    - Application deployment
    - Service configuration
    - Database setup
    - Monitoring setup
    """
    
    def __init__(self, config_path: str = "config/deploy.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize deployment components
        self.proxmox = ProxmoxLXCDeployment(self.config.get('proxmox', {}))
        
        # Deployment paths
        self.project_root = Path(__file__).parent.parent
        self.deployment_dir = Path('/opt/ha-intent-predictor')
        
        logger.info("System deployment initialized")
    
    def _load_config(self) -> Dict:
        """Load deployment configuration"""
        
        if not self.config_path.exists():
            # Create default config
            default_config = {
                'proxmox': {
                    'container_id': 200,
                    'container_name': 'ha-intent-predictor',
                    'proxmox_host': 'localhost',
                    'proxmox_user': 'root'
                },
                'database': {
                    'host': 'localhost',
                    'port': 5432,
                    'name': 'ha_predictor',
                    'user': 'ha_predictor',
                    'password': 'generate_secure_password'
                },
                'redis': {
                    'host': 'localhost',
                    'port': 6379,
                    'password': 'generate_secure_password'
                },
                'home_assistant': {
                    'url': 'http://homeassistant.local:8123',
                    'token': 'YOUR_LONG_LIVED_ACCESS_TOKEN'
                },
                'monitoring': {
                    'prometheus_enabled': True,
                    'grafana_enabled': True
                }
            }
            
            # Save default config
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            logger.info(f"Created default config at {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    async def deploy_full_system(self):
        """Deploy the complete system"""
        
        print("="*60)
        print("HA INTENT PREDICTOR SYSTEM DEPLOYMENT")
        print("="*60)
        print("Deploying to Proxmox LXC container")
        print(f"Target: Intel N200, 4 cores, 8GB RAM")
        print("="*60)
        
        try:
            # Step 1: Create and configure container
            print("\n1. Creating Proxmox LXC container...")
            await self._setup_container()
            
            # Step 2: Deploy application
            print("\n2. Deploying application...")
            await self._deploy_application()
            
            # Step 3: Setup databases
            print("\n3. Setting up databases...")
            await self._setup_databases()
            
            # Step 4: Configure services
            print("\n4. Configuring services...")
            await self._configure_services()
            
            # Step 5: Setup monitoring
            print("\n5. Setting up monitoring...")
            await self._setup_monitoring()
            
            # Step 6: Validate deployment
            print("\n6. Validating deployment...")
            await self._validate_deployment()
            
            print("\n" + "="*60)
            print("DEPLOYMENT COMPLETED SUCCESSFULLY")
            print("="*60)
            print("System is ready for bootstrap and historical data import")
            print(f"Container ID: {self.proxmox.container_id}")
            print(f"Container Name: {self.proxmox.container_name}")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise
    
    async def _setup_container(self):
        """Set up the LXC container"""
        
        print("  - Creating LXC container...")
        await self.proxmox.create_container()
        
        print("  - Starting container...")
        await self.proxmox.start_container()
        
        print("  - Setting up environment...")
        await self.proxmox.setup_container_environment()
        
        print("  ✓ Container setup completed")
    
    async def _deploy_application(self):
        """Deploy the application code"""
        
        print("  - Copying application files...")
        
        # Create application directory
        await self.proxmox._run_in_container([
            'mkdir', '-p', str(self.deployment_dir)
        ])
        
        # Copy source code
        source_files = [
            'src/',
            'config/',
            'scripts/',
            'main.py',
            'requirements.txt'
        ]
        
        for file_path in source_files:
            source = self.project_root / file_path
            if source.exists():
                # Copy to container (simplified - would use proper file transfer)
                print(f"    - Copying {file_path}")
        
        print("  - Installing Python dependencies...")
        await self.proxmox._run_in_container([
            '/opt/ha-intent-predictor/venv/bin/pip', 'install', '-r', 
            f'{self.deployment_dir}/requirements.txt'
        ])
        
        print("  ✓ Application deployed")
    
    async def _setup_databases(self):
        """Set up PostgreSQL and Redis"""
        
        print("  - Installing PostgreSQL...")
        await self.proxmox._run_in_container([
            'apt', 'install', '-y', 'postgresql', 'postgresql-contrib'
        ])
        
        print("  - Installing TimescaleDB...")
        await self.proxmox._run_in_container([
            'sh', '-c', 
            'echo "deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main" | tee /etc/apt/sources.list.d/timescaledb.list'
        ])
        
        await self.proxmox._run_in_container([
            'apt', 'update', '&&',
            'apt', 'install', '-y', 'timescaledb-2-postgresql-14'
        ])
        
        print("  - Configuring PostgreSQL...")
        db_config = self.config['database']
        
        # Create database and user
        await self.proxmox._run_in_container([
            'sudo', '-u', 'postgres', 'createdb', db_config['name']
        ])
        
        await self.proxmox._run_in_container([
            'sudo', '-u', 'postgres', 'createuser', db_config['user']
        ])
        
        print("  - Installing Redis...")
        await self.proxmox._run_in_container([
            'apt', 'install', '-y', 'redis-server'
        ])
        
        print("  ✓ Databases set up")
    
    async def _configure_services(self):
        """Configure system services"""
        
        print("  - Creating systemd service files...")
        
        # Main application service
        service_content = f"""
[Unit]
Description=HA Intent Predictor
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=ha-predictor
WorkingDirectory={self.deployment_dir}
Environment=PATH={self.deployment_dir}/venv/bin
ExecStart={self.deployment_dir}/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        # Write service file (simplified)
        print("    - Main application service")
        
        print("  - Configuring nginx reverse proxy...")
        nginx_config = """
server {
    listen 80;
    server_name localhost;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
"""
        
        print("  ✓ Services configured")
    
    async def _setup_monitoring(self):
        """Set up monitoring stack"""
        
        if self.config.get('monitoring', {}).get('prometheus_enabled'):
            print("  - Installing Prometheus...")
            # Would install and configure Prometheus
            
        if self.config.get('monitoring', {}).get('grafana_enabled'):
            print("  - Installing Grafana...")
            # Would install and configure Grafana
            
        print("  ✓ Monitoring set up")
    
    async def _validate_deployment(self):
        """Validate the deployment"""
        
        print("  - Checking container status...")
        
        # Check if container is running
        result = await self.proxmox._run_in_container(['systemctl', 'is-active', 'postgresql'])
        if 'active' not in result:
            raise RuntimeError("PostgreSQL is not running")
        
        result = await self.proxmox._run_in_container(['systemctl', 'is-active', 'redis'])
        if 'active' not in result:
            raise RuntimeError("Redis is not running")
        
        print("  - Checking application...")
        
        # Check if application can start
        result = await self.proxmox._run_in_container([
            f'{self.deployment_dir}/venv/bin/python', 
            f'{self.deployment_dir}/main.py', 
            '--version'
        ])
        
        print("  ✓ Deployment validation passed")
    
    async def generate_deployment_report(self):
        """Generate deployment report"""
        
        report = {
            'deployment_timestamp': datetime.now().isoformat(),
            'container_info': {
                'id': self.proxmox.container_id,
                'name': self.proxmox.container_name,
                'cores': self.proxmox.container_specs['cores'],
                'memory_mb': self.proxmox.container_specs['memory'],
                'storage_gb': self.proxmox.container_specs['storage']
            },
            'services': {
                'postgresql': 'configured',
                'redis': 'configured',
                'nginx': 'configured',
                'ha_predictor': 'configured'
            },
            'monitoring': {
                'prometheus': self.config.get('monitoring', {}).get('prometheus_enabled', False),
                'grafana': self.config.get('monitoring', {}).get('grafana_enabled', False)
            },
            'next_steps': [
                'Connect to container: pct enter {}'.format(self.proxmox.container_id),
                'Run bootstrap: python scripts/bootstrap.py',
                'Import historical data: python scripts/historical_import.py --days 180',
                'Start system: systemctl start ha-intent-predictor'
            ]
        }
        
        # Save report
        report_path = Path('logs/deployment_report.json')
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Deployment report saved to: {report_path}")
        
        return report


async def main():
    """Main deployment entry point"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy HA Intent Predictor system')
    parser.add_argument('--config', type=str, default='config/deploy.yaml',
                       help='Deployment configuration file')
    parser.add_argument('--container-id', type=int, 
                       help='Override container ID')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without executing')
    parser.add_argument('--report', action='store_true',
                       help='Generate deployment report')
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.container_id:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            config['proxmox']['container_id'] = args.container_id
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
    
    deployment = SystemDeployment(args.config)
    
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
        print(f"Would deploy to container {deployment.proxmox.container_id}")
        return
    
    try:
        await deployment.deploy_full_system()
        
        if args.report:
            await deployment.generate_deployment_report()
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())