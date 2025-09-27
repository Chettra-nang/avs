#!/usr/bin/env python3
"""
Demonstration of multi-modal data collection with ambulance scenarios.

This script shows how ambulance scenarios now support all 3 observation types:
- Kinematics: Vehicle state data (position, velocity, heading)
- OccupancyGrid: Spatial grid representation of the environment  
- GrayscaleObservation: Visual/image observations of the driving scene
"""

import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from highway_datacollection.scenarios.registry import ScenarioRegistry
from highway_datacollection.environments.factory import MultiAgentEnvFactory


def main():
    """Demonstrate multi-modal data collection with ambulance scenarios."""
    
    print("Ambulance Multi-Modal Data Collection Demonstration")
    print("=" * 55)
    
    # Initialize components
    registry = ScenarioRegistry()
    factory = MultiAgentEnvFactory()
    
    # Get ambulance scenarios
    ambulance_scenarios = registry.list_ambulance_scenarios()
    if not ambulance_scenarios:
        print("❌ No ambulance scenarios found!")
        return
    
    print(f"\n📋 Found {len(ambulance_scenarios)} ambulance scenarios")
    
    # Show supported observation types
    print("\n🔍 Supported Observation Types:")
    print("-" * 35)
    obs_types = factory.get_supported_observation_types()
    for i, obs_type in enumerate(obs_types, 1):
        print(f"{i}. {obs_type}")
        if obs_type == "Kinematics":
            print("   📊 Vehicle state data (position, velocity, heading)")
        elif obs_type == "OccupancyGrid":
            print("   🗺️  Spatial grid representation of the environment")
        elif obs_type == "GrayscaleObservation":
            print("   📷 Visual/image observations of the driving scene")
    
    # Test ambulance scenario with different observation types
    test_scenario = ambulance_scenarios[0]
    print(f"\n🚑 Testing scenario: '{test_scenario}'")
    print("-" * 50)
    
    # Get scenario configuration
    config = registry.get_ambulance_scenario_config(test_scenario)
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"Traffic density: {config.get('traffic_density', 'N/A')}")
    print(f"Controlled vehicles: {config.get('controlled_vehicles', 'N/A')}")
    
    # Show that observation config is not hardcoded
    if 'observation' not in config:
        print("✅ Observation config is NOT hardcoded - supports multi-modal collection!")
    else:
        print("❌ Observation config is hardcoded - limited to single modality")
    
    # Test environment creation with different observation types
    print(f"\n🏗️  Testing Environment Creation:")
    print("-" * 35)
    
    n_agents = 4  # Ambulance scenarios use 4 controlled agents
    
    for obs_type in obs_types:
        try:
            print(f"\n🔧 Creating environment with {obs_type} observations...")
            
            # Create environment using the factory
            env = factory.create_ambulance_env(test_scenario, obs_type, n_agents)
            
            # Get observation space info
            obs_space = env.observation_space
            if hasattr(obs_space, 'spaces'):
                # Multi-agent observation space
                agent_obs_space = obs_space.spaces[0] if obs_space.spaces else obs_space
                print(f"   ✅ Success! Observation space: {type(agent_obs_space).__name__}")
                if hasattr(agent_obs_space, 'shape'):
                    print(f"   📐 Shape: {agent_obs_space.shape}")
            else:
                print(f"   ✅ Success! Observation space: {type(obs_space).__name__}")
                if hasattr(obs_space, 'shape'):
                    print(f"   📐 Shape: {obs_space.shape}")
            
            # Clean up
            env.close()
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Test parallel environment creation (multi-modal)
    print(f"\n🔄 Testing Parallel Multi-Modal Environment Creation:")
    print("-" * 52)
    
    try:
        # Create environments for all observation types simultaneously
        parallel_envs = factory.create_parallel_ambulance_envs(test_scenario, n_agents)
        
        print(f"✅ Successfully created {len(parallel_envs)} parallel environments:")
        for obs_type, env in parallel_envs.items():
            obs_space = env.observation_space
            if hasattr(obs_space, 'spaces'):
                agent_obs_space = obs_space.spaces[0] if obs_space.spaces else obs_space
                space_info = f"{type(agent_obs_space).__name__}"
                if hasattr(agent_obs_space, 'shape'):
                    space_info += f" {agent_obs_space.shape}"
            else:
                space_info = f"{type(obs_space).__name__}"
                if hasattr(obs_space, 'shape'):
                    space_info += f" {obs_space.shape}"
            
            print(f"   🔹 {obs_type}: {space_info}")
            env.close()
            
    except Exception as e:
        print(f"❌ Parallel environment creation failed: {e}")
    
    # Show ambulance-specific configuration
    print(f"\n🚑 Ambulance-Specific Configuration:")
    print("-" * 38)
    ambulance_config = config.get('_ambulance_config', {})
    print(f"Ambulance agent index: {ambulance_config.get('ambulance_agent_index', 'N/A')}")
    print(f"Emergency priority: {ambulance_config.get('emergency_priority', 'N/A')}")
    print(f"Ambulance behavior: {ambulance_config.get('ambulance_behavior', 'N/A')}")
    
    # Show supported observation types for ambulance scenarios
    print(f"\n📊 Ambulance Scenario Observation Support:")
    print("-" * 42)
    try:
        ambulance_obs_types = registry.get_supported_observation_types_for_ambulance()
        if ambulance_obs_types:
            print("✅ Ambulance scenarios support all observation types:")
            for obs_type in ambulance_obs_types:
                print(f"   • {obs_type}")
        else:
            print("❌ No observation types found for ambulance scenarios")
    except Exception as e:
        print(f"❌ Error getting ambulance observation types: {e}")
    
    print("\n" + "=" * 55)
    print("🎉 Multi-modal ambulance data collection is now supported!")
    print("   Ambulance scenarios can collect:")
    print("   📊 Kinematics data (vehicle states)")
    print("   🗺️  OccupancyGrid data (spatial information)")
    print("   📷 GrayscaleObservation data (visual information)")


if __name__ == "__main__":
    main()