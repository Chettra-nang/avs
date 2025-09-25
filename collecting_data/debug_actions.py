#!/usr/bin/env python3
"""Debug the action issue in environment stepping"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from highway_datacollection.collection.collector import SynchronizedCollector
from highway_datacollection.collection.modality_config import ModalityConfigManager
import highway_env

def debug_actions():
    """Debug what actions are being generated and how environment expects them"""
    print("Debugging action format issues...")
    
    # Create collector 
    modality_manager = ModalityConfigManager()
    collector = SynchronizedCollector(
        n_agents=2,
        modality_config_manager=modality_manager
    )
    
    # Set up environments
    collector._setup_environments("free_flow")
    print("✅ Environments set up successfully")
    
    # Check what environments expect
    first_env = next(iter(collector._environments.values()))
    print(f"Environment type: {type(first_env)}")
    print(f"Action space: {first_env.action_space}")
    
    # Sample actions
    actions = collector.sample_actions(None, 0, "test_episode")
    print(f"Sampled actions: {actions}")
    print(f"Actions type: {type(actions)}")
    print(f"Actions length: {len(actions) if hasattr(actions, '__len__') else 'N/A'}")
    
    # Try to understand what the environment step expects
    if hasattr(first_env.action_space, 'sample'):
        sample_action = first_env.action_space.sample()
        print(f"Environment sample action: {sample_action}")
        print(f"Sample action type: {type(sample_action)}")
        
    # Try stepping with different action formats
    print("\n--- Testing different action formats ---")
    
    # Reset environment first
    initial_obs = collector.reset_parallel_envs(42)
    print("✅ Environments reset successfully")
    
    try:
        # Test 1: Original actions as tuple
        print(f"\nTest 1: Stepping with actions={actions} (type: {type(actions)})")
        step_results = collector.step_parallel_envs(actions)
        print("✅ Test 1 PASSED!")
        return True
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        
    try:
        # Test 2: Convert to list  
        actions_list = list(actions)
        print(f"\nTest 2: Stepping with actions={actions_list} (type: {type(actions_list)})")
        step_results = collector.step_parallel_envs(actions_list)
        print("✅ Test 2 PASSED!")
        return True
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        
    try:
        # Test 3: First action only
        first_action = actions[0] if hasattr(actions, '__getitem__') else actions
        print(f"\nTest 3: Stepping with single action={first_action} (type: {type(first_action)})")
        step_results = collector.step_parallel_envs((first_action,) * collector._n_agents)
        print("✅ Test 3 PASSED!")
        return True
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
    
    return False

if __name__ == "__main__":
    try:
        debug_actions()
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()