#!/usr/bin/env python3
"""
Demonstration of the MultiAgentEnvFactory functionality.

This script shows how to create and use multi-agent highway environments
with different observation modalities.
"""

from highway_datacollection.environments.factory import MultiAgentEnvFactory


def main():
    """Demonstrate the MultiAgentEnvFactory functionality."""
    print("=== MultiAgentEnvFactory Demo ===\n")
    
    # Create factory instance
    factory = MultiAgentEnvFactory()
    
    # Show supported observation types
    print("Supported observation types:")
    for obs_type in factory.get_supported_observation_types():
        print(f"  - {obs_type}")
    print()
    
    # Show available scenarios
    print("Available scenarios:")
    for scenario in factory._scenario_registry.list_scenarios():
        print(f"  - {scenario}")
    print()
    
    # Demonstrate single agent environment
    print("=== Single Agent Environment ===")
    env_single = factory.create_env("free_flow", "Kinematics", 1)
    obs, info = env_single.reset(seed=42)
    print(f"Observation type: {type(obs)}")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env_single.action_space}")
    
    # Take a step
    action = 0  # LANE_LEFT
    obs, reward, terminated, truncated, info = env_single.step(action)
    print(f"After step - Reward: {reward:.3f}, Terminated: {terminated}")
    env_single.close()
    print()
    
    # Demonstrate multi-agent environment
    print("=== Multi-Agent Environment ===")
    env_multi = factory.create_env("free_flow", "Kinematics", 2)
    obs, info = env_multi.reset(seed=42)
    print(f"Observation type: {type(obs)}")
    print(f"Number of agents: {len(obs)}")
    print(f"Each agent obs shape: {obs[0].shape}")
    print(f"Action space: {env_multi.action_space}")
    
    # Take a step with multiple agents
    actions = (0, 1)  # LANE_LEFT, IDLE
    obs, reward, terminated, truncated, info = env_multi.step(actions)
    print(f"After step - Reward: {reward:.3f}, Terminated: {terminated}")
    env_multi.close()
    print()
    
    # Demonstrate parallel environments for all observation types
    print("=== Parallel Multi-Modal Environments ===")
    parallel_envs = factory.create_parallel_envs("free_flow", 2)
    
    for obs_type, env in parallel_envs.items():
        obs, info = env.reset(seed=42)
        print(f"{obs_type}:")
        print(f"  Observation type: {type(obs)}")
        print(f"  Number of agents: {len(obs)}")
        if hasattr(obs[0], 'shape'):
            print(f"  Agent obs shape: {obs[0].shape}")
        else:
            print(f"  Agent obs type: {type(obs[0])}")
        env.close()
    print()
    
    # Demonstrate configuration validation
    print("=== Configuration Validation ===")
    valid_configs = [
        ("free_flow", "Kinematics", 2),
        ("dense_commuting", "OccupancyGrid", 3),
        ("lane_closure", "GrayscaleObservation", 1),
    ]
    
    invalid_configs = [
        ("invalid_scenario", "Kinematics", 2),
        ("free_flow", "InvalidObsType", 2),
        ("free_flow", "Kinematics", 0),
    ]
    
    print("Valid configurations:")
    for scenario, obs_type, n_agents in valid_configs:
        is_valid = factory.validate_configuration(scenario, obs_type, n_agents)
        print(f"  {scenario}, {obs_type}, {n_agents} agents: {is_valid}")
    
    print("Invalid configurations:")
    for scenario, obs_type, n_agents in invalid_configs:
        is_valid = factory.validate_configuration(scenario, obs_type, n_agents)
        print(f"  {scenario}, {obs_type}, {n_agents} agents: {is_valid}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()