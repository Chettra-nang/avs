#!/usr/bin/env python3
"""Fix all scenario files to use the correct SynchronizedCollector interface"""

import re
from pathlib import Path

def fix_scenario_file(file_path):
    """Fix a single scenario file"""
    print(f"Fixing {file_path.name}...")
    
    content = file_path.read_text()
    
    # Replace the old constructor call
    old_pattern = r'collector = SynchronizedCollector\(\s*modalities=\["kinematics", "occupancy_grid", "grayscale"\],\s*n_agents=collection_config\["n_agents"\]\s*\)'
    new_replacement = '''# Configure complete dataset collection
    modality_manager = ModalityConfigManager()
    collector = SynchronizedCollector(
        n_agents=collection_config["n_agents"],
        modality_config_manager=modality_manager
    )'''
    
    # Add the import
    if "from highway_datacollection.collection.modality_config import ModalityConfigManager" not in content:
        import_line = "from highway_datacollection.collection.collector import SynchronizedCollector"
        new_import = """from highway_datacollection.collection.collector import SynchronizedCollector
from highway_datacollection.collection.modality_config import ModalityConfigManager"""
        content = content.replace(import_line, new_import)
    
    # Replace the constructor
    content = re.sub(old_pattern, new_replacement, content, flags=re.MULTILINE | re.DOTALL)
    
    # Also handle the simpler case
    simple_pattern = r'collector = SynchronizedCollector\(\s*modalities=\["kinematics", "occupancy_grid", "grayscale"\]\s*\)'
    simple_replacement = '''modality_manager = ModalityConfigManager()
    collector = SynchronizedCollector(
        modality_config_manager=modality_manager
    )'''
    
    content = re.sub(simple_pattern, simple_replacement, content, flags=re.MULTILINE)
    
    file_path.write_text(content)
    print(f"✅ Fixed {file_path.name}")

def main():
    """Fix all scenario files"""
    collecting_data_dir = Path("/home/chettra/ITC/Research/AVs/collecting_data")
    
    scenario_files = list(collecting_data_dir.glob("scenario_*.py"))
    
    for file_path in scenario_files:
        try:
            fix_scenario_file(file_path)
        except Exception as e:
            print(f"❌ Error fixing {file_path.name}: {e}")
    
    print(f"\n🎉 Fixed {len(scenario_files)} scenario files!")

if __name__ == "__main__":
    main()