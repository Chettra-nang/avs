#!/usr/bin/env python3
"""
Simple Step Test

This script directly tests the step function to isolate the tuple issue.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_direct_step():
    """Test stepping environments directly like the collector does."""
    print("🧪 Direct Step Test")
    print("=" * 20)
    
    try:
        from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector
        from highway_datacollection.collection.action_samplers import RandomActionSampler
        
        # Create collector
        collector = AmbulanceDataCollector(n_agents=4)
        
        # Set up environments
        collector.setup_ambulance_environments("highway_emergency_light")
        
        # Get the underlying synchronized collector
        sync_collector = collector._collector
        
        # Reset environments
        sync_collector.reset_parallel_envs(seed=42)
        print("✅ Environments reset")
        
        # Create action sampler
        action_sampler = RandomActionSampler(action_space_size=5, seed=42)
        
        # Sample actions
        dummy_observations = {"Kinematics": [], "OccupancyGrid": [], "GrayscaleObservation": []}
        actions = action_sampler.sample_actions(dummy_observations, n_agents=4)
        print(f"✅ Actions sampled: {actions}")
        print(f"   Types: {[type(a).__name__ for a in actions]}")
        
        # Test step_parallel_envs directly
        print(f"🚶 Testing step_parallel_envs with actions: {actions}")
        try:
            step_results = sync_collector.step_parallel_envs(actions)
            print("✅ step_parallel_envs successful!")
            
            # Check results
            for obs_type, result in step_results.items():
                print(f"   {obs_type}: reward={result.get('reward', 'N/A')}")
                
        except Exception as e:
            print(f"❌ step_parallel_envs failed: {e}")
            print(f"   Error type: {type(e)}")
            
            # Let's try to step each environment individually to see which one fails
            print(f"\n🔍 Testing individual environments:")
            for obs_type, env in sync_collector._environments.items():
                try:
                    print(f"   Testing {obs_type}...")
                    obs, reward, terminated, truncated, info = env.step(actions)
                    print(f"   ✅ {obs_type} successful")
                except Exception as env_error:
                    print(f"   ❌ {obs_type} failed: {env_error}")
                    print(f"      Error type: {type(env_error)}")
                    
                    # Try different action formats for the failing environment
                    print(f"      Trying different action formats...")
                    
                    # Try as list
                    try:
                        obs, reward, terminated, truncated, info = env.step(list(actions))
                        print(f"      ✅ List format worked for {obs_type}")
                    except Exception as list_error:
                        print(f"      ❌ List format failed: {list_error}")
                    
                    # Try individual integers
                    try:
                        single_actions = [int(a) for a in actions]
                        obs, reward, terminated, truncated, info = env.step(single_actions)
                        print(f"      ✅ Individual int conversion worked for {obs_type}")
                    except Exception as int_error:
                        print(f"      ❌ Individual int conversion failed: {int_error}")
        
        collector.cleanup()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_action_validation():
    """Test action validation and conversion."""
    print("\n🔍 Action Validation Test")
    print("=" * 25)
    
    from highway_datacollection.collection.action_samplers import RandomActionSampler
    
    # Test different action formats
    sampler = RandomActionSampler(action_space_size=5, seed=42)
    
    # Test 1: Normal sampling
    actions1 = sampler.sample_actions({}, n_agents=4)
    print(f"Normal sampling: {actions1} (types: {[type(a).__name__ for a in actions1]})")
    
    # Test 2: Manual tuple creation
    actions2 = tuple(int(i) for i in [1, 2, 3, 4])
    print(f"Manual tuple: {actions2} (types: {[type(a).__name__ for a in actions2]})")
    
    # Test 3: Check for any hidden numpy types
    import numpy as np
    for i, action in enumerate(actions1):
        is_numpy = isinstance(action, np.integer)
        is_python_int = isinstance(action, int) and not isinstance(action, bool)
        print(f"Action {i}: {action} -> numpy: {is_numpy}, python int: {is_python_int}")

def main():
    """Main test function."""
    print("🔧 Simple Step Test")
    print("=" * 20)
    
    # Test action validation first
    test_action_validation()
    
    # Test direct stepping
    test_direct_step()

if __name__ == "__main__":
    main()