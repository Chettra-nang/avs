#!/usr/bin/env python3
"""Debug script to trace the exact location of the array truth value error."""

import sys
import traceback
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from highway_datacollection.collection.collector import SynchronizedCollector

def debug_collection():
    """Test data collection with detailed error tracing."""
    print("Creating SynchronizedCollector...")
    collector = SynchronizedCollector(n_agents=1)
    
    print("Setting up environments...")
    collector._setup_environments("free_flow")
    
    print("Attempting to collect single episode directly...")
    try:
        # Call the internal method directly to bypass error handling
        episode_data = collector._collect_single_episode(
            scenario_name="free_flow",
            seed=42,
            max_steps=5,
            episode_idx=0
        )
        print(f"Success! Collected episode: {episode_data.episode_id}")
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Full traceback:")
        traceback.print_exc()
        
        # Print the exact line that caused the issue
        tb = traceback.extract_tb(e.__traceback__)
        for frame in tb:
            print(f"File: {frame.filename}, Line: {frame.lineno}, Function: {frame.name}")
            print(f"Code: {frame.line}")
            print()

if __name__ == "__main__":
    debug_collection()