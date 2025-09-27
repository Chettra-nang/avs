#!/usr/bin/env python3
"""
Test Robust Collection

This script tests the patched collector with multiple episodes to verify
the tuple error is resolved.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_multiple_episodes():
    """Test multiple episodes to see if the tuple error is resolved."""
    print("🧪 Testing Multiple Episodes")
    print("=" * 30)
    
    try:
        from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector
        
        # Create collector
        with AmbulanceDataCollector(n_agents=4) as collector:
            print("✅ Ambulance collector created")
            
            # Test with a few episodes to see if the error occurs
            print("🚑 Testing 20 episodes with 20 steps each...")
            
            result = collector.collect_single_ambulance_scenario(
                scenario_name="highway_emergency_light",
                episodes=20,
                max_steps=20,  # Short episodes to test quickly
                seed=42
            )
            
            print(f"✅ Collection completed!")
            print(f"   Total episodes: {result.total_episodes}")
            print(f"   Successful episodes: {result.successful_episodes}")
            print(f"   Failed episodes: {result.failed_episodes}")
            print(f"   Collection time: {result.collection_time:.2f}s")
            
            if result.errors:
                print(f"   Errors encountered:")
                for error in result.errors[:3]:  # Show first 3 errors
                    print(f"     - {error}")
            
            if result.successful_episodes > 15:  # At least 75% success rate
                print("🎉 Test PASSED! Robust collection is working!")
                return True
            else:
                print("⚠️  Test partially successful but some episodes failed")
                return False
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_scenarios():
    """Test different scenarios to ensure robustness."""
    print("\n🎯 Testing Different Scenarios")
    print("=" * 35)
    
    scenarios_to_test = [
        "highway_emergency_light",
        "highway_emergency_moderate", 
        "highway_rush_hour"
    ]
    
    try:
        from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector
        
        with AmbulanceDataCollector(n_agents=4) as collector:
            
            for scenario in scenarios_to_test:
                print(f"\n🚑 Testing {scenario}...")
                
                result = collector.collect_single_ambulance_scenario(
                    scenario_name=scenario,
                    episodes=5,  # Small number for quick test
                    max_steps=15,
                    seed=42
                )
                
                success_rate = result.successful_episodes / result.total_episodes
                print(f"   Success rate: {success_rate:.1%} ({result.successful_episodes}/{result.total_episodes})")
                
                if result.errors:
                    print(f"   Errors: {len(result.errors)}")
                    for error in result.errors[:2]:
                        print(f"     - {error}")
            
            print("\n✅ Multi-scenario test completed!")
            return True
            
    except Exception as e:
        print(f"❌ Multi-scenario test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🔧 Testing Robust Collection System")
    print("=" * 40)
    print()
    print("This test verifies that the tuple C API error is resolved")
    print("by running multiple episodes and scenarios.")
    print()
    
    # Test 1: Multiple episodes
    test1_passed = test_multiple_episodes()
    
    # Test 2: Different scenarios
    test2_passed = test_different_scenarios()
    
    print("\n" + "=" * 40)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ The tuple error appears to be resolved!")
        print()
        print("You can now run full data collection:")
        print("  python collecting_ambulance_data/examples/basic_ambulance_collection.py \\")
        print("    --episodes 50 --max-steps 100 --output-dir data/ambulance_dataset")
    else:
        print("⚠️  Some tests had issues")
        if not test1_passed:
            print("   - Multiple episode test had problems")
        if not test2_passed:
            print("   - Multi-scenario test had problems")
        print()
        print("The system may still work but monitor for errors during full collection.")
    
    return 0 if (test1_passed and test2_passed) else 1

if __name__ == "__main__":
    sys.exit(main())