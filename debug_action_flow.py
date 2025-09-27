#!/usr/bin/env python3
"""
Debug Action Flow

This script helps debug the action flow to identify where the tuple issue occurs.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_action_types():
    """Debug action types at each step of the process."""
    print("🔍 Debugging Action Flow")
    print("=" * 30)
    
    try:
        from highway_datacollection.collection.action_samplers import RandomActionSampler
        from collecting_ambulance_data.scenarios.ambulance_scenarios import get_scenario_by_name
        from highway_datacollection.environments.factory import MultiAgentEnvFactory
        
        # Step 1: Test action sampler
        print("\n1️⃣ Testing Action Sampler")
        sampler = RandomActionSampler(action_space_size=5, seed=42)
        actions = sampler.sample_actions({}, n_agents=4)
        print(f"   Actions: {actions}")
        print(f"   Types: {[type(a) for a in actions]}")
        print(f"   All Python ints: {all(isinstance(a, int) and not isinstance(a, bool) for a in actions)}")
        
        # Step 2: Test environment creation
        print("\n2️⃣ Testing Environment Creation")
        env_factory = MultiAgentEnvFactory()
        env = env_factory.create_ambulance_env(
            scenario_name="highway_emergency_light",
            obs_type="Kinematics",
            n_agents=4
        )
        print(f"   Environment created: {type(env)}")
        print(f"   Action space: {env.action_space}")
        
        # Step 3: Test environment reset
        print("\n3️⃣ Testing Environment Reset")
        obs, info = env.reset(seed=42)
        print(f"   Reset successful")
        print(f"   Observation type: {type(obs)}")
        
        # Step 4: Test single step with detailed action tracking
        print("\n4️⃣ Testing Environment Step with Action Tracking")
        
        # Create actions with explicit type checking
        test_actions = tuple(int(i) for i in [1, 2, 0, 3])  # Explicit Python ints
        print(f"   Test actions: {test_actions}")
        print(f"   Action types: {[type(a).__name__ for a in test_actions]}")
        print(f"   Action isinstance checks:")
        for i, action in enumerate(test_actions):
            print(f"     Action {i}: {action} -> int: {isinstance(action, int)}, bool: {isinstance(action, bool)}")
        
        # Try the step
        print(f"   Attempting env.step({test_actions})...")
        try:
            obs, reward, terminated, truncated, info = env.step(test_actions)
            print(f"   ✅ Step successful!")
            print(f"   Reward: {reward}")
        except Exception as e:
            print(f"   ❌ Step failed: {e}")
            print(f"   Error type: {type(e)}")
            
            # Try alternative action formats
            print(f"\n   Trying alternative action formats...")
            
            # Try list instead of tuple
            try:
                list_actions = list(test_actions)
                print(f"   Trying list: {list_actions}")
                obs, reward, terminated, truncated, info = env.step(list_actions)
                print(f"   ✅ List format worked!")
            except Exception as e2:
                print(f"   ❌ List format failed: {e2}")
            
            # Try numpy array
            try:
                import numpy as np
                array_actions = np.array(test_actions, dtype=np.int32)
                print(f"   Trying numpy array: {array_actions}")
                obs, reward, terminated, truncated, info = env.step(array_actions)
                print(f"   ✅ Numpy array worked!")
            except Exception as e3:
                print(f"   ❌ Numpy array failed: {e3}")
        
        env.close()
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

def test_collector_step_by_step():
    """Test the collector step by step to isolate the issue."""
    print("\n🚑 Testing Collector Step by Step")
    print("=" * 35)
    
    try:
        from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector
        
        # Create collector
        collector = AmbulanceDataCollector(n_agents=4)
        
        # Set up environments
        setup_info = collector.setup_ambulance_environments("highway_emergency_light")
        print(f"✅ Environment setup: {setup_info}")
        
        # Access the underlying synchronized collector
        sync_collector = collector._collector
        
        # Test reset
        sync_collector.reset_parallel_envs(seed=42)
        print("✅ Environments reset")
        
        # Test observation sampling
        observations = sync_collector.get_current_observations()
        print(f"✅ Observations obtained: {type(observations)}")
        
        # Test action sampling with detailed tracking
        print("\n🎯 Testing Action Sampling in Collector Context")
        actions = sync_collector.sample_actions(observations, step=0)
        print(f"   Sampled actions: {actions}")
        print(f"   Action types: {[type(a).__name__ for a in actions]}")
        
        # Test step with these actions
        print(f"\n🚶 Testing Step with Sampled Actions")
        try:
            step_results = sync_collector.step_parallel_envs(actions)
            print(f"✅ Step successful!")
        except Exception as e:
            print(f"❌ Step failed: {e}")
            
            # Try to identify which environment is causing the issue
            print(f"\n🔍 Testing Individual Environments")
            for obs_type, env in sync_collector._environments.items():
                try:
                    print(f"   Testing {obs_type} environment...")
                    obs, reward, terminated, truncated, info = env.step(actions)
                    print(f"   ✅ {obs_type} environment step successful")
                except Exception as env_error:
                    print(f"   ❌ {obs_type} environment failed: {env_error}")
        
        collector.cleanup()
        
    except Exception as e:
        print(f"❌ Collector test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main debug function."""
    print("🐛 Action Flow Debugging")
    print("=" * 25)
    
    # Test 1: Basic action flow
    debug_action_types()
    
    # Test 2: Collector integration
    test_collector_step_by_step()
    
    print("\n" + "=" * 50)
    print("🔍 Debug completed. Check the output above for clues.")

if __name__ == "__main__":
    main()