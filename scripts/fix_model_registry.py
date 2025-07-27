#!/usr/bin/env python3
"""
Fix Model Registry Script
Recreates the registry.json file from existing model metadata files
"""

import json
import os
import sys
from pathlib import Path

def fix_model_registry():
    """Recreate registry.json from existing model metadata"""
    
    # Path to metadata directory
    metadata_dir = Path('models/metadata')
    
    if not metadata_dir.exists():
        print("Error: models/metadata directory not found")
        sys.exit(1)
    
    registry = {}
    
    # Process all model metadata files
    for json_file in metadata_dir.glob('*_v1.json'):
        if json_file.name == 'registry.json' or json_file.name.endswith('_schema.json'):
            continue
            
        print(f"Processing {json_file.name}")
        
        try:
            with open(json_file) as f:
                metadata = json.load(f)
            
            room_id = metadata['room_id']
            version = metadata['version']
            
            # Use room_id as the key for registry lookup (model_type is separate)
            registry_key = room_id
            
            if registry_key not in registry:
                registry[registry_key] = {}
            
            registry[registry_key][str(version)] = metadata
            
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
    
    # Write registry file
    registry_file = metadata_dir / 'registry.json'
    
    try:
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
        
        print(f"✅ Created registry with {len(registry)} room models")
        print(f"Registry keys: {list(registry.keys())}")
        
        # Verify file size
        file_size = registry_file.stat().st_size
        print(f"Registry file size: {file_size} bytes")
        
        if file_size == 0:
            print("❌ Warning: Registry file is empty!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error writing registry file: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Fixing model registry...")
    success = fix_model_registry()
    
    if success:
        print("✅ Model registry fixed successfully")
    else:
        print("❌ Failed to fix model registry")
        sys.exit(1)