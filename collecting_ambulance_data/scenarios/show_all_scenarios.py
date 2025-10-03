#!/usr/bin/env python3
"""
Show all 30 ambulance scenarios and their environment types
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collecting_ambulance_data.scenarios.ambulance_scenarios import get_all_ambulance_scenarios
from highway_datacollection.environments.factory import MultiAgentEnvFactory


def show_all_scenarios():
    """Display all 30 scenarios with their environment types"""
    
    print("\n" + "=" * 80)
    print("🚗 ALL 30 AMBULANCE SCENARIOS - Environment Types")
    print("=" * 80 + "\n")
    
    # Get all scenarios
    all_scenarios = get_all_ambulance_scenarios()
    factory = MultiAgentEnvFactory()
    
    # Group by environment type
    env_groups = {
        'highway-v0': [],
        'roundabout-v0': [],
        'intersection-v0': [],
        'merge-v0': []
    }
    
    # Categorize scenarios
    for scenario_name in all_scenarios.keys():
        env_id = factory._get_env_id_for_scenario(scenario_name)
        env_groups[env_id].append(scenario_name)
    
    # Display by environment type
    emoji_map = {
        'highway-v0': '🛣️',
        'roundabout-v0': '⭕',
        'intersection-v0': '🌐',
        'merge-v0': '🔀'
    }
    
    desc_map = {
        'highway-v0': 'Straight Multi-Lane Highway',
        'roundabout-v0': 'Circular Roundabout',
        'intersection-v0': 'Crossing Roads / Intersection',
        'merge-v0': 'Highway with Merging Lanes'
    }
    
    total_count = 0
    
    for env_id in ['highway-v0', 'roundabout-v0', 'intersection-v0', 'merge-v0']:
        scenarios = env_groups[env_id]
        emoji = emoji_map[env_id]
        desc = desc_map[env_id]
        
        print(f"\n{emoji} {env_id.upper()}")
        print(f"   Type: {desc}")
        print(f"   Count: {len(scenarios)} scenarios")
        print(f"   {'-' * 76}")
        
        for i, scenario_name in enumerate(sorted(scenarios), 1):
            config = all_scenarios[scenario_name]
            print(f"   {i:2d}. {scenario_name}")
            print(f"       {config.get('description', 'N/A')}")
            print(f"       Vehicles: {config.get('vehicles_count', 'N/A')} | "
                  f"Duration: {config.get('duration', 'N/A')} steps | "
                  f"Density: {config.get('traffic_density', 'N/A')}")
        
        total_count += len(scenarios)
    
    # Summary
    print(f"\n{'=' * 80}")
    print(f"📊 SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n{'🛣️  Highway (straight roads):':<40} {len(env_groups['highway-v0']):>2} scenarios")
    print(f"{'⭕ Roundabout (circular roads):':<40} {len(env_groups['roundabout-v0']):>2} scenarios")
    print(f"{'🌐 Intersection/Corner (crossing roads):':<40} {len(env_groups['intersection-v0']):>2} scenarios")
    print(f"{'🔀 Merge (merging lanes):':<40} {len(env_groups['merge-v0']):>2} scenarios")
    print(f"{'-' * 80}")
    print(f"{'✅ TOTAL:':<40} {total_count:>2} scenarios")
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    show_all_scenarios()
