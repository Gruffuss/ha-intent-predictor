"""
Model Versioning Store for HA Intent Predictor.

Manages model persistence, versioning, and backup as specified in CLAUDE.md.
Ensures continuous learning models can be restored and evolved.
"""

import asyncio
import json
import logging
import pickle
import gzip
import hashlib
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a stored model"""
    model_id: str
    room_id: str
    model_type: str
    version: int
    created_at: datetime
    updated_at: datetime
    size_bytes: int
    checksum: str
    performance_metrics: Dict[str, float]
    training_data_range: Tuple[datetime, datetime]
    parent_version: Optional[int] = None
    is_active: bool = True
    backup_path: Optional[str] = None
    compression_enabled: bool = True


@dataclass
class ModelVersion:
    """A versioned model instance"""
    metadata: ModelMetadata
    model_data: bytes
    feature_schema: Dict[str, Any]
    training_config: Dict[str, Any]


class ModelStore:
    """
    Model versioning and storage system.
    
    Handles model persistence, versioning, compression, and backup
    for the continuous learning system.
    """
    
    def __init__(self, storage_config: Dict):
        self.storage_config = storage_config
        self.storage_type = storage_config.get('type', 'local')
        self.storage_path = Path(storage_config.get('path', './models'))
        self.versioning_enabled = storage_config.get('versioning_enabled', True)
        self.max_versions = storage_config.get('max_versions', 10)
        self.backup_enabled = storage_config.get('backup_enabled', True)
        self.backup_interval = storage_config.get('backup_interval', 3600)
        self.compression_enabled = storage_config.get('compression_enabled', True)
        self.compression_level = storage_config.get('compression_level', 6)
        
        # Model registry - in-memory cache of model metadata
        self.model_registry: Dict[str, Dict[int, ModelMetadata]] = {}
        
        # Active models cache
        self.active_models: Dict[str, ModelVersion] = {}
        
        # Backup task
        self.backup_task = None
        
        logger.info(f"Model store initialized with {self.storage_type} storage")
    
    async def initialize(self):
        """Initialize the model store"""
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.storage_path / 'models').mkdir(exist_ok=True)
        (self.storage_path / 'backups').mkdir(exist_ok=True)
        (self.storage_path / 'metadata').mkdir(exist_ok=True)
        
        # Load existing model registry
        await self._load_model_registry()
        
        # Start backup task if enabled
        if self.backup_enabled:
            self.backup_task = asyncio.create_task(self._backup_loop())
        
        logger.info("Model store initialized successfully")
    
    async def _load_model_registry(self):
        """Load model registry from storage"""
        try:
            registry_path = self.storage_path / 'metadata' / 'registry.json'
            
            if registry_path.exists():
                async with aiofiles.open(registry_path, 'r') as f:
                    registry_data = json.loads(await f.read())
                
                # Convert to proper format
                for room_id, versions in registry_data.items():
                    self.model_registry[room_id] = {}
                    for version_str, metadata_dict in versions.items():
                        version = int(version_str)
                        
                        # Convert datetime strings back to datetime objects
                        metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
                        metadata_dict['updated_at'] = datetime.fromisoformat(metadata_dict['updated_at'])
                        
                        if metadata_dict['training_data_range']:
                            metadata_dict['training_data_range'] = (
                                datetime.fromisoformat(metadata_dict['training_data_range'][0]),
                                datetime.fromisoformat(metadata_dict['training_data_range'][1])
                            )
                        
                        self.model_registry[room_id][version] = ModelMetadata(**metadata_dict)
                
                logger.info(f"Loaded {len(self.model_registry)} room models from registry")
            
        except Exception as e:
            logger.error(f"Error loading model registry: {e}")
            self.model_registry = {}
    
    async def _save_model_registry(self):
        """Save model registry to storage"""
        try:
            registry_path = self.storage_path / 'metadata' / 'registry.json'
            
            # Convert to JSON-serializable format
            registry_data = {}
            for room_id, versions in self.model_registry.items():
                registry_data[room_id] = {}
                for version, metadata in versions.items():
                    metadata_dict = asdict(metadata)
                    
                    # Convert datetime objects to strings
                    metadata_dict['created_at'] = metadata.created_at.isoformat()
                    metadata_dict['updated_at'] = metadata.updated_at.isoformat()
                    
                    if metadata.training_data_range:
                        metadata_dict['training_data_range'] = [
                            metadata.training_data_range[0].isoformat(),
                            metadata.training_data_range[1].isoformat()
                        ]
                    
                    registry_data[room_id][str(version)] = metadata_dict
            
            # Write to file
            async with aiofiles.open(registry_path, 'w') as f:
                await f.write(json.dumps(registry_data, indent=2))
            
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")
    
    async def store_model(self, room_id: str, model_type: str, model_data: Any,
                         feature_schema: Dict[str, Any], training_config: Dict[str, Any],
                         performance_metrics: Dict[str, float] = None,
                         training_data_range: Tuple[datetime, datetime] = None) -> str:
        """
        Store a new model version.
        
        Args:
            room_id: Room identifier
            model_type: Type of model (e.g., 'gradient_boost', 'neural_net')
            model_data: The actual model object
            feature_schema: Schema of features used
            training_config: Training configuration
            performance_metrics: Performance metrics
            training_data_range: Range of training data
            
        Returns:
            Model ID for the stored model
        """
        try:
            # Get next version number
            if room_id not in self.model_registry:
                self.model_registry[room_id] = {}
            
            version = len(self.model_registry[room_id]) + 1
            
            # Generate model ID
            model_id = f"{room_id}_{model_type}_v{version}"
            
            # Serialize model data
            serialized_data = pickle.dumps(model_data)
            
            # Compress if enabled
            if self.compression_enabled:
                serialized_data = gzip.compress(serialized_data, compresslevel=self.compression_level)
            
            # Calculate checksum
            checksum = hashlib.sha256(serialized_data).hexdigest()
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                room_id=room_id,
                model_type=model_type,
                version=version,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                size_bytes=len(serialized_data),
                checksum=checksum,
                performance_metrics=performance_metrics or {},
                training_data_range=training_data_range,
                parent_version=version - 1 if version > 1 else None,
                is_active=True,
                compression_enabled=self.compression_enabled
            )
            
            # Store model file
            model_path = self.storage_path / 'models' / f"{model_id}.pkl"
            if self.compression_enabled:
                model_path = model_path.with_suffix('.pkl.gz')
            
            async with aiofiles.open(model_path, 'wb') as f:
                await f.write(serialized_data)
            
            # Store metadata
            metadata_path = self.storage_path / 'metadata' / f"{model_id}.json"
            async with aiofiles.open(metadata_path, 'w') as f:
                metadata_dict = asdict(metadata)
                metadata_dict['created_at'] = metadata.created_at.isoformat()
                metadata_dict['updated_at'] = metadata.updated_at.isoformat()
                if metadata.training_data_range:
                    metadata_dict['training_data_range'] = [
                        metadata.training_data_range[0].isoformat(),
                        metadata.training_data_range[1].isoformat()
                    ]
                
                await f.write(json.dumps(metadata_dict, indent=2))
            
            # Store feature schema and training config
            schema_path = self.storage_path / 'metadata' / f"{model_id}_schema.json"
            async with aiofiles.open(schema_path, 'w') as f:
                await f.write(json.dumps({
                    'feature_schema': feature_schema,
                    'training_config': training_config
                }, indent=2))
            
            # Update registry
            self.model_registry[room_id][version] = metadata
            
            # Save registry
            await self._save_model_registry()
            
            # Clean up old versions if necessary
            if self.versioning_enabled and len(self.model_registry[room_id]) > self.max_versions:
                await self._cleanup_old_versions(room_id)
            
            logger.info(f"Stored model {model_id} ({len(serialized_data)} bytes)")
            
            return model_id
            
        except Exception as e:
            logger.error(f"Error storing model: {e}")
            raise
    
    async def load_model(self, model_id: str) -> Optional[ModelVersion]:
        """
        Load a model by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            ModelVersion object or None if not found
        """
        try:
            # Check if model is in cache
            if model_id in self.active_models:
                return self.active_models[model_id]
            
            # Find metadata
            metadata = None
            for room_id, versions in self.model_registry.items():
                for version, meta in versions.items():
                    if meta.model_id == model_id:
                        metadata = meta
                        break
                if metadata:
                    break
            
            if not metadata:
                logger.warning(f"Model {model_id} not found in registry")
                return None
            
            # Load model data
            model_filename = f"{model_id}.pkl"
            if metadata.compression_enabled:
                model_filename += ".gz"
            
            model_path = self.storage_path / 'models' / model_filename
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return None
            
            async with aiofiles.open(model_path, 'rb') as f:
                compressed_data = await f.read()
            
            # Decompress if needed
            if metadata.compression_enabled:
                serialized_data = gzip.decompress(compressed_data)
            else:
                serialized_data = compressed_data
            
            # Verify checksum on serialized data (as originally intended)
            checksum = hashlib.sha256(serialized_data).hexdigest()
            if checksum != metadata.checksum:
            
            # Deserialize model
            model_data = pickle.loads(serialized_data)
            
            # Load feature schema and training config
            schema_path = self.storage_path / 'metadata' / f"{model_id}_schema.json"
            if schema_path.exists():
                async with aiofiles.open(schema_path, 'r') as f:
                    schema_data = json.loads(await f.read())
                
                feature_schema = schema_data.get('feature_schema', {})
                training_config = schema_data.get('training_config', {})
            else:
                feature_schema = {}
                training_config = {}
            
            # Create model version
            model_version = ModelVersion(
                metadata=metadata,
                model_data=model_data,
                feature_schema=feature_schema,
                training_config=training_config
            )
            
            # Cache the model
            self.active_models[model_id] = model_version
            
            logger.info(f"Loaded model {model_id}")
            
            return model_version
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return None
    
    async def get_latest_model(self, room_id: str, model_type: str = None) -> Optional[ModelVersion]:
        """
        Get the latest model for a room.
        
        Args:
            room_id: Room identifier
            model_type: Optional model type filter
            
        Returns:
            Latest ModelVersion or None
        """
        try:
            if room_id not in self.model_registry:
                return None
            
            # Find latest version
            latest_version = 0
            latest_metadata = None
            
            for version, metadata in self.model_registry[room_id].items():
                if metadata.is_active and version > latest_version:
                    if model_type is None or metadata.model_type == model_type:
                        latest_version = version
                        latest_metadata = metadata
            
            if latest_metadata:
                return await self.load_model(latest_metadata.model_id)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest model for {room_id}: {e}")
            return None
    
    async def list_models(self, room_id: str = None) -> List[ModelMetadata]:
        """
        List all models, optionally filtered by room.
        
        Args:
            room_id: Optional room filter
            
        Returns:
            List of ModelMetadata objects
        """
        models = []
        
        for room, versions in self.model_registry.items():
            if room_id is None or room == room_id:
                for version, metadata in versions.items():
                    models.append(metadata)
        
        # Sort by creation time
        models.sort(key=lambda m: m.created_at, reverse=True)
        
        return models
    
    async def delete_model(self, model_id: str) -> bool:
        """
        Delete a model and its associated files.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find and remove from registry
            model_metadata = None
            for room_id, versions in self.model_registry.items():
                for version, metadata in versions.items():
                    if metadata.model_id == model_id:
                        model_metadata = metadata
                        del self.model_registry[room_id][version]
                        break
                if model_metadata:
                    break
            
            if not model_metadata:
                logger.warning(f"Model {model_id} not found in registry")
                return False
            
            # Delete model file
            model_filename = f"{model_id}.pkl"
            if model_metadata.compression_enabled:
                model_filename += ".gz"
            
            model_path = self.storage_path / 'models' / model_filename
            if model_path.exists():
                model_path.unlink()
            
            # Delete metadata file
            metadata_path = self.storage_path / 'metadata' / f"{model_id}.json"
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Delete schema file
            schema_path = self.storage_path / 'metadata' / f"{model_id}_schema.json"
            if schema_path.exists():
                schema_path.unlink()
            
            # Remove from cache
            if model_id in self.active_models:
                del self.active_models[model_id]
            
            # Save registry
            await self._save_model_registry()
            
            logger.info(f"Deleted model {model_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {e}")
            return False
    
    async def _cleanup_old_versions(self, room_id: str):
        """Clean up old model versions"""
        try:
            if room_id not in self.model_registry:
                return
            
            versions = list(self.model_registry[room_id].keys())
            versions.sort(reverse=True)  # Latest first
            
            # Keep only max_versions
            versions_to_delete = versions[self.max_versions:]
            
            for version in versions_to_delete:
                metadata = self.model_registry[room_id][version]
                await self.delete_model(metadata.model_id)
                
                logger.info(f"Cleaned up old model version {version} for room {room_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up old versions: {e}")
    
    async def _backup_loop(self):
        """Background task for periodic backups"""
        while True:
            try:
                await asyncio.sleep(self.backup_interval)
                await self._create_backup()
            except Exception as e:
                logger.error(f"Error in backup loop: {e}")
    
    async def _create_backup(self):
        """Create a backup of all models"""
        try:
            backup_dir = self.storage_path / 'backups' / datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy models directory
            models_backup = backup_dir / 'models'
            shutil.copytree(self.storage_path / 'models', models_backup)
            
            # Copy metadata directory
            metadata_backup = backup_dir / 'metadata'
            shutil.copytree(self.storage_path / 'metadata', metadata_backup)
            
            # Create backup manifest
            manifest = {
                'created_at': datetime.now().isoformat(),
                'models_count': len(list((self.storage_path / 'models').glob('*'))),
                'total_size': sum(f.stat().st_size for f in (self.storage_path / 'models').rglob('*') if f.is_file())
            }
            
            async with aiofiles.open(backup_dir / 'manifest.json', 'w') as f:
                await f.write(json.dumps(manifest, indent=2))
            
            logger.info(f"Created backup at {backup_dir}")
            
            # Clean up old backups (keep last 7 days)
            await self._cleanup_old_backups()
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
    
    async def _cleanup_old_backups(self):
        """Clean up old backup directories"""
        try:
            backup_base = self.storage_path / 'backups'
            cutoff_date = datetime.now() - timedelta(days=7)
            
            for backup_dir in backup_base.iterdir():
                if backup_dir.is_dir():
                    try:
                        # Parse backup directory name
                        backup_date = datetime.strptime(backup_dir.name, '%Y%m%d_%H%M%S')
                        
                        if backup_date < cutoff_date:
                            shutil.rmtree(backup_dir)
                            logger.info(f"Cleaned up old backup: {backup_dir}")
                    except ValueError:
                        # Skip directories that don't match the expected format
                        continue
            
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            stats = {
                'total_models': 0,
                'total_size_bytes': 0,
                'rooms': {},
                'storage_path': str(self.storage_path),
                'compression_enabled': self.compression_enabled
            }
            
            # Count models and calculate sizes
            for room_id, versions in self.model_registry.items():
                room_stats = {
                    'model_count': len(versions),
                    'latest_version': max(versions.keys()) if versions else 0,
                    'total_size_bytes': 0
                }
                
                for version, metadata in versions.items():
                    room_stats['total_size_bytes'] += metadata.size_bytes
                    stats['total_models'] += 1
                    stats['total_size_bytes'] += metadata.size_bytes
                
                stats['rooms'][room_id] = room_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}
    
    async def health_check(self) -> str:
        """Health check for model store"""
        try:
            # Check if storage directory is accessible
            if not self.storage_path.exists():
                return "storage_unavailable"
            
            # Check if we can write to storage
            test_file = self.storage_path / '.health_check'
            test_file.write_text("test")
            test_file.unlink()
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"Model store health check failed: {e}")
            return "error"
    
    async def shutdown(self):
        """Shutdown the model store"""
        logger.info("Shutting down model store...")
        
        # Cancel backup task
        if self.backup_task:
            self.backup_task.cancel()
        
        # Save registry one final time
        await self._save_model_registry()
        
        # Clear caches
        self.active_models.clear()
        self.model_registry.clear()
        
        logger.info("Model store shut down")