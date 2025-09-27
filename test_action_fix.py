#!/usr/bin/env python3
"""
Test Action Sampler Fix

This script tests that the action sampler fix resolves the tuple issue.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_action_sampler():
    """Test the fixed action sampler."""
    print("🧪 Testing Action Sampler Fix")
    print("=" * 30)
    
    try:
        from highway_datacollection.collection.action_samplers import RandomActionSampler
        
        # Create action sampler
        sampler = RandomActionSampler(action_space_size=5, seed=42)
        print("✅ Action sampler created successfully")
        
        # Test action sampling
        observations = {"Kinematics": []}  # Dummy observations
        actions = sampler.sample_actions(observations, n_agents=4, step=0)
        
        print(f"✅ Actions sampled: {actions}")
        print(f"   Action types: {[type(a).__name__ for a in actions]}")
        
        # Verify all actions are Python integers
        all_python_ints = all(isinstance(a, int) and not isinstance(a, bool) for a in actions)
        if all_python_ints:
            print("✅ All actions are Python integers - fix successful!")
            return True
        else:
            print("❌ Some actions are not Python integers")
            for i, action in enumerate(actions):
                print(f"   Action {i}: {action} (type: {type(action)})")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ambulance_collector():
    """Test the ambulance collector with fixed actions."""
    print("\n🚑 Testing Ambulance Collector")
    print("=" * 30)
    
    try:
        from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector
        
        # Create collector
        collector = AmbulanceDataCollector(n_agents=4)
        print("✅ Ambulance collector created successfully")
        
        # Test a very short episode
        print("🧪 Testing short episode collection...")
        
        result = collector.collect_single_ambulance_scenario(
            scenario_name="highway_emergency_light",
            episodes=1,
            max_steps=5,  # Very short episode
            seed=42
        )
        
        if result.successful_episodes > 0:
            print("✅ Short episode collection successful!")
            print(f"   Episodes: {result.successful_episodes}/{result.total_episodes}")
            return True
        else:
            print("❌ Episode collection failed")
            if result.errors:
                for error in result.errors:
                    print(f"   Error: {error}")
            return False
            
    except Exception as e:
        print(f"❌ Ambulance collector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            collector.cleanup()
        except:
            pass

def main():
    """Main test function."""
    print("🔧 Testing Action Sampler Fixes")
    print("=" * 40)
    print()
    
    # Test 1: Action sampler
    sampler_ok = test_action_sampler()
    
    # Test 2: Ambulance collector (only if sampler test passed)
    if sampler_ok:
        collector_ok = test_ambulance_collector()
    else:
        collector_ok = False
    
    print("\n" + "=" * 40)
    if sampler_ok and collector_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✅ The action sampler fix should resolve the tuple error")
        print()
        print("You can now run your data collection:")
        print("  python collecting_ambulance_data/examples/basic_ambulance_collection.py \\")
        print("    --episodes 50 --max-steps 100 --output-dir data/ambulance_dataset")
    else:
        print("❌ Some tests failed")
        if not sampler_ok:
            print("   - Action sampler test failed")
        if not collector_ok:
            print("   - Ambulance collector test failed")
    
    return 0 if (sampler_ok and collector_ok) else 1

if __name__ == "__main__":
    sys.exit(main())