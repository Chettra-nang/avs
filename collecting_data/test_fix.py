#!/usr/bin/env python3
"""Quick test to verify the array boolean fix"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from highway_datacollection.collection.collector import SynchronizedCollector
from highway_datacollection.collection.modality_config import ModalityConfigManager
from highway_datacollection.environments.factory import MultiAgentEnvFactory
import highway_env

def test_data_collection():
    """Test data collection with the numpy array fix"""
    print("Testing data collection with array boolean fix...")
    
    try:
        # Create environment factory  
        env_factory = MultiAgentEnvFactory()
        
        # Create modality config manager and enable desired modalities
        modality_manager = ModalityConfigManager()
        
        # Create synchronized collector
        collector = SynchronizedCollector(
            n_agents=2,
            modality_config_manager=modality_manager
        )
        
        # Try to collect a small sample
        print("Starting test collection...")
        
        results = collector.collect_episode_batch(
            scenario_name="free_flow",
            episodes=2,
            seed=42,
            max_steps=30
        )
        
        print(f"✅ Test PASSED! Collected {results.successful_episodes}/{results.total_episodes} episodes")
        print(f"Collection time: {results.collection_time:.2f}s")
        if results.errors:
            print(f"Errors encountered: {len(results.errors)}")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {str(e)}")
        print("Error traceback:")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_collection()
    exit(0 if success else 1)