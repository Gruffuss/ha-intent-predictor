#!/usr/bin/env python3
"""
Home Assistant Integration Setup Script
Creates all necessary entities and automations for the occupancy prediction system
"""

import asyncio
import logging
import yaml
import json
from pathlib import Path
from typing import Dict, List
import aiohttp
import click

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HAIntegrationSetup:
    """Setup Home Assistant integration for occupancy predictions"""
    
    def __init__(self, ha_url: str, ha_token: str):
        self.ha_url = ha_url.rstrip('/')
        self.ha_token = ha_token
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={'Authorization': f'Bearer {self.ha_token}'}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_connection(self) -> bool:
        """Test connection to Home Assistant"""
        try:
            async with self.session.get(f"{self.ha_url}/api/") as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    async def create_sensor_entity(self, entity_id: str, config: Dict):
        """Create a sensor entity in Home Assistant"""
        payload = {
            "state": config.get("initial_state", "unknown"),
            "attributes": config.get("attributes", {})
        }
        
        try:
            async with self.session.post(
                f"{self.ha_url}/api/states/{entity_id}",
                json=payload
            ) as resp:
                if resp.status in [200, 201]:
                    logger.info(f"Created/updated entity: {entity_id}")
                    return True
                else:
                    logger.error(f"Failed to create entity {entity_id}: {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"Error creating entity {entity_id}: {e}")
            return False
    
    async def create_prediction_entities(self, rooms: List[str], horizons: List[int]):
        """Create all prediction entities"""
        logger.info("Creating prediction entities...")
        
        entities_created = 0
        
        for room in rooms:
            for horizon in horizons:
                entity_id = f"sensor.occupancy_prediction_{room}_{horizon}min"
                
                config = {
                    "initial_state": "0.5",
                    "attributes": {
                        "friendly_name": f"{room.replace('_', ' ').title()} Occupancy Prediction ({horizon}min)",
                        "unit_of_measurement": "probability",
                        "device_class": "None",
                        "state_class": "measurement",
                        "icon": "mdi:home-analytics",
                        "room": room,
                        "horizon_minutes": horizon,
                        "uncertainty": 1.0,
                        "confidence": 0.0,
                        "model_name": "initializing",
                        "last_updated": None,
                        "prediction_for": None,
                        "contributing_factors": [],
                        "explanation": "Prediction system initializing..."
                    }
                }
                
                if await self.create_sensor_entity(entity_id, config):
                    entities_created += 1
        
        logger.info(f"Created {entities_created} prediction entities")
        return entities_created
    
    async def create_helper_entities(self, rooms: List[str]):
        """Create helper entities for trends, anomalies, etc."""
        logger.info("Creating helper entities...")
        
        entities_created = 0
        
        for room in rooms:
            # Occupancy trend sensor
            trend_entity = f"sensor.occupancy_trend_{room}"
            trend_config = {
                "initial_state": "stable",
                "attributes": {
                    "friendly_name": f"{room.replace('_', ' ').title()} Occupancy Trend",
                    "icon": "mdi:trending-up",
                    "room": room,
                    "trend_strength": 0.0,
                    "change_probability": 0.0,
                    "trend_direction": "stable",
                    "confidence": 0.0
                }
            }
            
            if await self.create_sensor_entity(trend_entity, trend_config):
                entities_created += 1
            
            # Anomaly detection binary sensor
            anomaly_entity = f"binary_sensor.occupancy_anomaly_{room}"
            anomaly_config = {
                "initial_state": "off",
                "attributes": {
                    "friendly_name": f"{room.replace('_', ' ').title()} Occupancy Anomaly",
                    "device_class": "problem",
                    "icon": "mdi:alert-circle",
                    "room": room,
                    "anomaly_score": 0.0,
                    "anomaly_type": None,
                    "description": None,
                    "is_cat_activity": False
                }
            }
            
            if await self.create_sensor_entity(anomaly_entity, anomaly_config):
                entities_created += 1
            
            # Model performance sensor
            performance_entity = f"sensor.model_performance_{room}"
            performance_config = {
                "initial_state": "unknown",
                "attributes": {
                    "friendly_name": f"{room.replace('_', ' ').title()} Model Performance",
                    "unit_of_measurement": "accuracy",
                    "icon": "mdi:chart-line",
                    "room": room,
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "false_positive_rate": 0.0,
                    "samples_trained": 0,
                    "last_training": None
                }
            }
            
            if await self.create_sensor_entity(performance_entity, performance_config):
                entities_created += 1
        
        logger.info(f"Created {entities_created} helper entities")
        return entities_created
    
    async def create_system_entities(self):
        """Create system-wide entities"""
        logger.info("Creating system entities...")
        
        entities_created = 0
        
        # System status sensor
        system_entity = "sensor.ha_intent_predictor_status"
        system_config = {
            "initial_state": "initializing",
            "attributes": {
                "friendly_name": "HA Intent Predictor System Status",
                "icon": "mdi:brain",
                "version": "1.0.0",
                "uptime": 0,
                "total_predictions": 0,
                "total_events_processed": 0,
                "active_models": 0,
                "system_health": "unknown",
                "last_model_update": None,
                "memory_usage": "unknown",
                "cpu_usage": "unknown"
            }
        }
        
        if await self.create_sensor_entity(system_entity, system_config):
            entities_created += 1
        
        # Pattern discovery status
        patterns_entity = "sensor.pattern_discovery_status"
        patterns_config = {
            "initial_state": "discovering",
            "attributes": {
                "friendly_name": "Pattern Discovery Status",
                "icon": "mdi:puzzle",
                "patterns_discovered": 0,
                "rooms_analyzed": 0,
                "discovery_progress": 0,
                "significant_patterns": 0,
                "last_discovery": None
            }
        }
        
        if await self.create_sensor_entity(patterns_entity, patterns_config):
            entities_created += 1
        
        logger.info(f"Created {entities_created} system entities")
        return entities_created
    
    async def create_automation_templates(self, rooms: List[str], horizons: List[int]):
        """Create automation templates for users"""
        logger.info("Creating automation templates...")
        
        automations = []
        
        # Preheating automation template
        for room in rooms:
            if room in ['living_kitchen', 'bedroom', 'office', 'guest_bedroom']:
                automation = {
                    "alias": f"Preheat {room.replace('_', ' ').title()}",
                    "description": f"Automatically preheat {room} based on occupancy prediction",
                    "trigger": [
                        {
                            "platform": "numeric_state",
                            "entity_id": f"sensor.occupancy_prediction_{room}_120min",
                            "above": 0.7
                        }
                    ],
                    "condition": [
                        {
                            "condition": "numeric_state",
                            "entity_id": f"sensor.occupancy_prediction_{room}_120min",
                            "attribute": "confidence",
                            "above": 0.6
                        },
                        {
                            "condition": "template",
                            "value_template": "{{ states('climate." + room + "') != 'unavailable' }}"
                        }
                    ],
                    "action": [
                        {
                            "service": "climate.set_temperature",
                            "target": {
                                "entity_id": f"climate.{room}"
                            },
                            "data": {
                                "temperature": "{{ state_attr('climate." + room + "', 'target_temperature') or 22 }}"
                            }
                        },
                        {
                            "service": "notify.persistent_notification",
                            "data": {
                                "title": f"Preheating {room.replace('_', ' ').title()}",
                                "message": f"High occupancy probability detected for {room}. Preheating started."
                            }
                        }
                    ],
                    "mode": "single"
                }
                automations.append(automation)
        
        # Precooling automation template
        for room in rooms:
            if room in ['living_kitchen', 'bedroom', 'office']:
                automation = {
                    "alias": f"Precool {room.replace('_', ' ').title()}",
                    "description": f"Automatically precool {room} based on occupancy prediction",
                    "trigger": [
                        {
                            "platform": "numeric_state",
                            "entity_id": f"sensor.occupancy_prediction_{room}_15min",
                            "above": 0.8
                        }
                    ],
                    "condition": [
                        {
                            "condition": "numeric_state",
                            "entity_id": f"sensor.occupancy_prediction_{room}_15min",
                            "attribute": "confidence",
                            "above": 0.7
                        },
                        {
                            "condition": "template",
                            "value_template": "{{ state_attr('sensor." + room + "_env_sensor_temperature', 'state') | float > 25 }}"
                        }
                    ],
                    "action": [
                        {
                            "service": "climate.set_temperature",
                            "target": {
                                "entity_id": f"climate.{room}"
                            },
                            "data": {
                                "temperature": "{{ (state_attr('climate." + room + "', 'target_temperature') or 22) - 2 }}"
                            }
                        }
                    ],
                    "mode": "single"
                }
                automations.append(automation)
        
        # Anomaly notification automation
        anomaly_automation = {
            "alias": "Occupancy Anomaly Detected",
            "description": "Notify when anomalous occupancy patterns are detected",
            "trigger": [
                {
                    "platform": "state",
                    "entity_id": [f"binary_sensor.occupancy_anomaly_{room}" for room in rooms],
                    "to": "on"
                }
            ],
            "condition": [
                {
                    "condition": "template",
                    "value_template": "{{ not state_attr(trigger.entity_id, 'is_cat_activity') }}"
                }
            ],
            "action": [
                {
                    "service": "notify.persistent_notification",
                    "data": {
                        "title": "Occupancy Anomaly Detected",
                        "message": "Unusual occupancy pattern detected in {{ state_attr(trigger.entity_id, 'room') }}. Type: {{ state_attr(trigger.entity_id, 'anomaly_type') }}"
                    }
                }
            ],
            "mode": "queued",
            "max": 10
        }
        automations.append(anomaly_automation)
        
        # Save automation templates to file
        automations_file = Path("/opt/ha-intent-predictor/config/automation_templates.yaml")
        automations_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(automations_file, 'w') as f:
            yaml.dump(automations, f, default_flow_style=False, indent=2)
        
        logger.info(f"Created {len(automations)} automation templates")
        logger.info(f"Automation templates saved to: {automations_file}")
        
        return len(automations)
    
    async def create_dashboard_config(self, rooms: List[str]):
        """Create Lovelace dashboard configuration"""
        logger.info("Creating dashboard configuration...")
        
        dashboard = {
            "title": "HA Intent Predictor",
            "views": [
                {
                    "title": "Overview",
                    "path": "overview",
                    "icon": "mdi:home-analytics",
                    "cards": [
                        {
                            "type": "entities",
                            "title": "System Status",
                            "entities": [
                                "sensor.ha_intent_predictor_status",
                                "sensor.pattern_discovery_status"
                            ]
                        }
                    ]
                }
            ]
        }
        
        # Add room-specific views
        for room in rooms:
            room_view = {
                "title": room.replace('_', ' ').title(),
                "path": room,
                "icon": "mdi:home-outline",
                "cards": [
                    {
                        "type": "entities",
                        "title": f"{room.replace('_', ' ').title()} Predictions",
                        "entities": [
                            f"sensor.occupancy_prediction_{room}_15min",
                            f"sensor.occupancy_prediction_{room}_30min",
                            f"sensor.occupancy_prediction_{room}_60min",
                            f"sensor.occupancy_prediction_{room}_120min"
                        ]
                    },
                    {
                        "type": "entities",
                        "title": f"{room.replace('_', ' ').title()} Status",
                        "entities": [
                            f"sensor.occupancy_trend_{room}",
                            f"binary_sensor.occupancy_anomaly_{room}",
                            f"sensor.model_performance_{room}"
                        ]
                    },
                    {
                        "type": "history-graph",
                        "title": f"{room.replace('_', ' ').title()} Prediction History",
                        "entities": [
                            f"sensor.occupancy_prediction_{room}_15min",
                            f"sensor.occupancy_prediction_{room}_60min"
                        ],
                        "hours_to_show": 24,
                        "refresh_interval": 60
                    }
                ]
            }
            dashboard["views"].append(room_view)
        
        # Add analytics view
        analytics_view = {
            "title": "Analytics",
            "path": "analytics",
            "icon": "mdi:chart-line",
            "cards": [
                {
                    "type": "entities",
                    "title": "Model Performance",
                    "entities": [f"sensor.model_performance_{room}" for room in rooms]
                },
                {
                    "type": "entities",
                    "title": "Anomaly Detection",
                    "entities": [f"binary_sensor.occupancy_anomaly_{room}" for room in rooms]
                }
            ]
        }
        dashboard["views"].append(analytics_view)
        
        # Save dashboard configuration
        dashboard_file = Path("/opt/ha-intent-predictor/config/dashboard.yaml")
        with open(dashboard_file, 'w') as f:
            yaml.dump(dashboard, f, default_flow_style=False, indent=2)
        
        logger.info(f"Dashboard configuration saved to: {dashboard_file}")
        return dashboard
    
    async def setup_complete_integration(self, config_file: str):
        """Setup complete Home Assistant integration"""
        logger.info("Setting up complete Home Assistant integration...")
        
        # Load configuration
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        rooms = list(config['rooms'].keys())
        horizons = config['ml']['prediction_horizons']
        
        # Create all entities
        prediction_entities = await self.create_prediction_entities(rooms, horizons)
        helper_entities = await self.create_helper_entities(rooms)
        system_entities = await self.create_system_entities()
        
        # Create automation templates
        automations = await self.create_automation_templates(rooms, horizons)
        
        # Create dashboard configuration
        dashboard = await self.create_dashboard_config(rooms)
        
        total_entities = prediction_entities + helper_entities + system_entities
        
        logger.info("🎉 Home Assistant integration setup completed!")
        logger.info(f"Total entities created: {total_entities}")
        logger.info(f"Automation templates created: {automations}")
        logger.info("Next steps:")
        logger.info("1. Import automation templates into Home Assistant")
        logger.info("2. Configure dashboard using the generated configuration")
        logger.info("3. Customize automations based on your specific needs")
        
        return {
            "entities_created": total_entities,
            "automations_created": automations,
            "rooms_configured": len(rooms),
            "horizons_configured": len(horizons)
        }


@click.command()
@click.option('--ha-url', required=True, help='Home Assistant URL')
@click.option('--ha-token', required=True, help='Home Assistant Long-Lived Access Token')
@click.option('--config', '-c', default='/opt/ha-intent-predictor/config/app.yaml',
              help='Path to configuration file')
@click.option('--test-only', is_flag=True, help='Only test connection')
async def main(ha_url: str, ha_token: str, config: str, test_only: bool):
    """Setup Home Assistant integration for HA Intent Predictor"""
    
    async with HAIntegrationSetup(ha_url, ha_token) as setup:
        # Test connection
        logger.info("Testing Home Assistant connection...")
        if not await setup.test_connection():
            logger.error("Failed to connect to Home Assistant")
            return
        
        logger.info("✅ Connected to Home Assistant successfully")
        
        if test_only:
            logger.info("Connection test completed")
            return
        
        # Setup complete integration
        result = await setup.setup_complete_integration(config)
        
        logger.info("Integration setup summary:")
        for key, value in result.items():
            logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())