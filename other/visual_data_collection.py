#!/usr/bin/env python3
"""
Visual Data Collection Demo - Watch data collection in action
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from highway_datacollection.collection.collector import SynchronizedCollector
from highway_datacollection.environments.factory import MultiAgentEnvFactory
from highway_datacollection.collection.action_samplers import RandomActionSampler

def visual_collection_demo():
    """Run data collection with visual display."""
    print("Visual Data Collection Demo")
    print("=" * 40)
    print("Watch the data collection process!")
    print("Press Ctrl+C to stop")
    print("=" * 40)
    
    try:
        # Create components
        env_factory = MultiAgentEnvFactory()
        action_sampler = RandomActionSampler()
        collector = SynchronizedCollector(env_factory, action_sampler)
        
        # Collect a few episodes with visual display
        print("Starting data collection...")
        result = collector.collect_episode_batch(
            scenario_name="free_flow",
            episodes=3,  # Just 3 episodes for demo
            seed=42,
            max_steps=100  # Shorter episodes for demo
        )
        
        print(f"\nCollection completed!")
        print(f"Episodes collected: {len(result.episode_metadata)}")
        print(f"Total transitions: {len(result.transitions)}")
        print(f"Collection time: {result.collection_time:.2f}s")
        
        # Show some sample data
        if result.transitions:
            print(f"\nSample transition data:")
            sample = result.transitions[0]
            print(f"  Episode ID: {sample.get('episode_id', 'N/A')}")
            print(f"  Step: {sample.get('step', 'N/A')}")
            print(f"  Action: {sample.get('action', 'N/A')}")
            print(f"  Reward: {sample.get('reward', 'N/A')}")
        
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: This demo requires the full data collection system to be working")

if __name__ == "__main__":
    visual_collection_demo()