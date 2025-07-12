"""
Home Assistant API Client for data collection and sensor updates
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)

class HomeAssistantClient:
    """Client for interacting with Home Assistant API"""
    
    def __init__(self, url: str, token: str, timeout: int = 30):
        self.url = url.rstrip('/')
        self.token = token
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        })
        
    def test_connection(self) -> bool:
        """Test connection to Home Assistant"""
        try:
            response = self.session.get(f'{self.url}/api/', timeout=self.timeout)
            response.raise_for_status()
            logger.info("Successfully connected to Home Assistant")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Home Assistant: {e}")
            return False
    
    def get_historical_data(self, entity_ids: List[str], days: int = 7) -> pd.DataFrame:
        """
        Fetch historical data for specified entities
        
        Args:
            entity_ids: List of entity IDs to fetch
            days: Number of days of history to fetch
            
        Returns:
            DataFrame with columns: entity_id, state, last_changed
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        logger.info(f"Fetching {days} days of historical data for {len(entity_ids)} entities")
        
        all_data = []
        
        # Fetch data in chunks to avoid overwhelming the API
        chunk_size = 10
        for i in range(0, len(entity_ids), chunk_size):
            chunk_entities = entity_ids[i:i + chunk_size]
            
            for entity_id in chunk_entities:
                try:
                    data = self._fetch_entity_history(entity_id, start_time, end_time)
                    all_data.extend(data)
                    
                    # Be nice to the API
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {entity_id}: {e}")
                    continue
        
        if not all_data:
            logger.warning("No historical data retrieved")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df['last_changed'] = pd.to_datetime(df['last_changed'])
        df = df.sort_values('last_changed').reset_index(drop=True)
        
        logger.info(f"Retrieved {len(df)} historical records")
        return df
    
    def _fetch_entity_history(self, entity_id: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Fetch history for a single entity"""
        
        start_iso = start_time.isoformat()
        end_iso = end_time.isoformat()
        
        url = f'{self.url}/api/history/period/{start_iso}?filter_entity_id={entity_id}&end_time={end_iso}'
        
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        
        data = response.json()
        
        if not data or not data[0]:
            return []
        
        entity_data = data[0]  # First (and only) entity in response
        
        records = []
        for state_obj in entity_data:
            # Only include binary sensor state changes
            if state_obj['state'] in ['on', 'off', 'unavailable']:
                records.append({
                    'entity_id': entity_id,
                    'state': state_obj['state'],
                    'last_changed': state_obj['last_changed']
                })
        
        return records
    
    def get_current_states(self, entity_ids: List[str]) -> Dict[str, str]:
        """Get current states of specified entities"""
        
        current_states = {}
        
        for entity_id in entity_ids:
            try:
                response = self.session.get(f'{self.url}/api/states/{entity_id}', timeout=self.timeout)
                response.raise_for_status()
                
                state_data = response.json()
                current_states[entity_id] = state_data.get('state', 'unknown')
                
            except Exception as e:
                logger.warning(f"Failed to get current state for {entity_id}: {e}")
                current_states[entity_id] = 'unknown'
        
        return current_states
    
    def update_sensor(self, entity_id: str, state: Any, attributes: Optional[Dict] = None) -> bool:
        """Update a sensor state in Home Assistant"""
        
        payload = {
            'state': state,
            'attributes': attributes or {}
        }
        
        try:
            response = self.session.post(
                f'{self.url}/api/states/{entity_id}',
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            logger.debug(f"Updated sensor {entity_id}: {state}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update sensor {entity_id}: {e}")
            return False
    
    def update_multiple_sensors(self, sensor_updates: Dict[str, Dict]) -> Dict[str, bool]:
        """
        Update multiple sensors at once
        
        Args:
            sensor_updates: Dict mapping entity_id to {'state': value, 'attributes': dict}
            
        Returns:
            Dict mapping entity_id to success status
        """
        
        results = {}
        
        for entity_id, update_data in sensor_updates.items():
            state = update_data.get('state')
            attributes = update_data.get('attributes', {})
            
            success = self.update_sensor(entity_id, state, attributes)
            results[entity_id] = success
            
            # Small delay between updates
            time.sleep(0.05)
        
        successful_updates = sum(results.values())
        logger.info(f"Updated {successful_updates}/{len(sensor_updates)} sensors successfully")
        
        return results
    
    def create_sensor_if_not_exists(self, entity_id: str, initial_state: Any = 'unknown') -> bool:
        """Create a sensor if it doesn't exist"""
        
        # Try to get current state first
        try:
            response = self.session.get(f'{self.url}/api/states/{entity_id}', timeout=self.timeout)
            if response.status_code == 200:
                logger.debug(f"Sensor {entity_id} already exists")
                return True
        except:
            pass
        
        # Create the sensor
        return self.update_sensor(entity_id, initial_state, {
            'friendly_name': entity_id.replace('sensor.', '').replace('_', ' ').title(),
            'device_class': None,
            'unit_of_measurement': None
        })
