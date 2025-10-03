#!/usr/bin/env python3
"""
Test script to verify environment selection for different scenario types
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from highway_datacollection.environments.factory import MultiAgentEnvFactory
from collecting_ambulance_data.scenarios.ambulance_scenarios import get_all_ambulance_scenarios


def test_environment_selection():
    """Test that correct environment types are selected for different scenarios"""
    
    print("=" * 80)
    print("🧪 Testing Environment Selection for Different Scenario Types")
    print("=" * 80)
    print()
    
    # Get all scenarios
    all_scenarios = get_all_ambulance_scenarios()
    
    # Create factory
    factory = MultiAgentEnvFactory()
    
    # Test scenarios by type
    test_scenarios = {
        'Highway': ['highway_emergency_light', 'highway_rush_hour'],
        'Roundabout': ['roundabout_single_lane', 'roundabout_multi_lane'],
        'Intersection': ['intersection_four_way', 'intersection_t_junction'],
        'Corner': ['corner_sharp_turn', 'corner_urban_crossing'],
        'Merge': ['merge_highway_entry', 'merge_heavy_traffic']
    }
    
    results = {}
    
    for category, scenario_names in test_scenarios.items():
        print(f"\n{'─' * 80}")
        print(f"Testing {category} Scenarios")
        print(f"{'─' * 80}")
        
        for scenario_name in scenario_names:
            if scenario_name not in all_scenarios:
                print(f"  ⚠️  {scenario_name}: NOT FOUND in scenarios")
                continue
            
            # Get environment ID
            env_id = factory._get_env_id_for_scenario(scenario_name)
            
            # Try to create environment
            try:
                env = factory.create_ambulance_env(
                    scenario_name=scenario_name,
                    obs_type='Kinematics',
                    n_agents=4
                )
                
                # Get environment type from created env
                actual_env_id = env.unwrapped.spec.id if hasattr(env.unwrapped, 'spec') else 'unknown'
                
                print(f"  ✅ {scenario_name}")
                print(f"     Expected: {env_id}")
                print(f"     Actual: {actual_env_id}")
                print(f"     Road type: {type(env.unwrapped.road).__name__ if hasattr(env.unwrapped, 'road') else 'unknown'}")
                
                results[scenario_name] = {
                    'category': category,
                    'expected_env': env_id,
                    'actual_env': actual_env_id,
                    'success': True
                }
                
                env.close()
                
            except Exception as e:
                print(f"  ❌ {scenario_name}: ERROR - {e}")
                results[scenario_name] = {
                    'category': category,
                    'expected_env': env_id,
                    'success': False,
                    'error': str(e)
                }
    
    # Summary
    print(f"\n{'=' * 80}")
    print("📊 SUMMARY")
    print(f"{'=' * 80}")
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"\n✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")
    
    # Show environment type mapping
    print(f"\n{'=' * 80}")
    print("🗺️  Environment Type Mapping")
    print(f"{'=' * 80}")
    
    env_mapping = {}
    for scenario_name, result in results.items():
        if result['success']:
            env_type = result['expected_env']
            if env_type not in env_mapping:
                env_mapping[env_type] = []
            env_mapping[env_type].append(scenario_name)
    
    for env_type, scenarios in sorted(env_mapping.items()):
        print(f"\n{env_type}:")
        for scenario in scenarios:
            print(f"  - {scenario}")
    
    print(f"\n{'=' * 80}")
    
    if successful == total:
        print("✅ ALL TESTS PASSED! Environment selection working correctly.")
        print("\n🎉 Data collection will use:")
        print("   - highway-v0 for highway scenarios (straight roads)")
        print("   - roundabout-v0 for roundabout scenarios (circular roads)")
        print("   - intersection-v0 for corner/intersection scenarios (crossing roads)")
        print("   - merge-v0 for merge scenarios (merging lanes)")
    else:
        print("⚠️  Some tests failed. Check errors above.")
    
    print(f"{'=' * 80}\n")
    
    return successful == total


if __name__ == "__main__":
    success = test_environment_selection()
    sys.exit(0 if success else 1)
