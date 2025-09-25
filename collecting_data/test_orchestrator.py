#!/usr/bin/env python3
"""Test that mimics what the scenarios do to identify the issue"""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from highway_datacollection.collection.orchestrator import run_full_collection
from highway_datacollection.collection.modality_config import ModalityConfigManager

def test_orchestrator_collection():
    """Test collection using the orchestrator like the scenarios do"""
    print("Testing orchestrator-based collection...")
    
    # Storage path
    storage_path = Path("data/test_orchestrator_output")
    storage_path.mkdir(parents=True, exist_ok=True)
    
    # Test configuration mimicking scenario files
    collection_config = {
        "base_storage_path": storage_path,
        "episodes_per_scenario": 2,  # Very small for testing
        "n_agents": 2,
        "max_steps_per_episode": 30,
        "scenarios": ["free_flow"],  # Only one scenario
        "base_seed": 1001,
        "batch_size": 1,  # Single episode batches
        "modality_config_manager": ModalityConfigManager()
    }
    
    print("Configuration:")
    for key, value in collection_config.items():
        if key != "modality_config_manager":
            print(f"  {key}: {value}")
    
    try:
        print("Starting orchestrator collection...")
        result = run_full_collection(**collection_config)
        
        print(f"✅ Test PASSED!")
        print(f"Total episodes: {result.total_episodes}")
        print(f"Successful episodes: {result.successful_episodes}")
        print(f"Failed episodes: {result.failed_episodes}")
        print(f"Collection time: {result.collection_time:.2f}s")
        
        if result.errors:
            print(f"Errors: {len(result.errors)}")
            for error in result.errors[:3]:
                print(f"  - {error}")
                
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_orchestrator_collection()
    exit(0 if success else 1)