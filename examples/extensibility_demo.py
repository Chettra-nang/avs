#!/usr/bin/env python3
"""
Demonstration of extensibility features and policy integration.

This example shows how to:
1. Use different action sampling strategies
2. Configure modalities per scenario
3. Create custom observation processors
4. Integrate trained policies (mock example)
"""

import numpy as np
from pathlib import Path
from unittest.mock import Mock

from highway_datacollection.collection import (
    SynchronizedCollector,
    RandomActionSampler,
    PolicyActionSampler,
    HybridActionSampler,
    ModalityConfigManager,
    create_kinematics_only_config,
    create_vision_only_config,
    ObservationProcessor,
    ModalityConfig
)


class CustomKinematicsProcessor(ObservationProcessor):
    """Custom processor that adds velocity magnitude to kinematics data."""
    
    def process_observation(self, observation: np.ndarray, metadata: dict) -> np.ndarray:
        """Add velocity magnitude as additional feature."""
        obs_array = np.array(observation)
        
        if len(obs_array.shape) == 2 and obs_array.shape[1] >= 5:
            # Extract velocity components (assuming columns 3,4 are vx,vy)
            vx = obs_array[:, 3]
            vy = obs_array[:, 4]
            
            # Calculate velocity magnitude
            v_mag = np.sqrt(vx**2 + vy**2)
            
            # Add as new column
            obs_with_vmag = np.column_stack([obs_array, v_mag])
            return obs_with_vmag
        
        return obs_array
    
    def get_output_schema(self) -> dict:
        """Schema includes velocity magnitude."""
        return {
            'kinematics_raw': list,
            'velocity_magnitude': float
        }


def demo_action_sampling():
    """Demonstrate different action sampling strategies."""
    print("=== Action Sampling Demo ===")
    
    # 1. Random action sampling
    random_sampler = RandomActionSampler(seed=42)
    collector = SynchronizedCollector(n_agents=2, action_sampler=random_sampler)
    print(f"Random sampler: {type(collector.get_action_sampler()).__name__}")
    
    # 2. Mock policy action sampling
    mock_policy = Mock()
    mock_policy.predict = Mock(return_value=(1, None))  # Always return action 1
    
    policy_sampler = PolicyActionSampler(mock_policy, observation_key="Kinematics")
    collector.set_action_sampler(policy_sampler)
    print(f"Policy sampler: {type(collector.get_action_sampler()).__name__}")
    
    # 3. Hybrid action sampling (different strategies per agent)
    agent_0_sampler = RandomActionSampler(seed=100)
    agent_1_sampler = PolicyActionSampler(mock_policy)
    
    hybrid_sampler = HybridActionSampler(
        samplers={0: agent_0_sampler, 1: agent_1_sampler}
    )
    collector.set_action_sampler(hybrid_sampler)
    print(f"Hybrid sampler: {type(collector.get_action_sampler()).__name__}")
    
    # Test action sampling
    mock_observations = {"Kinematics": {"observation": [[1, 2, 3, 4, 5]]}}
    actions = collector.sample_actions(mock_observations)
    print(f"Sample actions: {actions}")


def demo_modality_configuration():
    """Demonstrate modality configuration and toggles."""
    print("\n=== Modality Configuration Demo ===")
    
    # Create modality configuration manager
    manager = ModalityConfigManager()
    
    # 1. Scenario with only kinematics
    kinematics_config = create_kinematics_only_config("highway_simple")
    manager.set_scenario_modality_config("highway_simple", kinematics_config)
    
    enabled = manager.get_enabled_modalities("highway_simple")
    print(f"Highway simple enabled modalities: {enabled}")
    
    # 2. Scenario with only vision modalities
    vision_config = create_vision_only_config("highway_complex")
    manager.set_scenario_modality_config("highway_complex", vision_config)
    
    enabled = manager.get_enabled_modalities("highway_complex")
    print(f"Highway complex enabled modalities: {enabled}")
    
    # 3. Custom scenario configuration
    custom_config = manager.create_scenario_config(
        scenario_name="highway_custom",
        enabled_modalities=["Kinematics", "OccupancyGrid"],
        disabled_modalities=["GrayscaleObservation"]
    )
    manager.set_scenario_modality_config("highway_custom", custom_config)
    
    enabled = manager.get_enabled_modalities("highway_custom")
    print(f"Highway custom enabled modalities: {enabled}")
    
    # 4. Global modality disable
    manager.disable_modality_globally("GrayscaleObservation")
    enabled_default = manager.get_enabled_modalities("new_scenario")
    print(f"New scenario (after global disable): {enabled_default}")
    
    return manager


def demo_custom_processors():
    """Demonstrate custom observation processors."""
    print("\n=== Custom Processors Demo ===")
    
    # Create manager and register custom processor
    manager = ModalityConfigManager()
    custom_processor = CustomKinematicsProcessor()
    manager.register_processor("Kinematics", custom_processor)
    
    # Create scenario config with custom processor
    config = manager.create_scenario_config(
        scenario_name="custom_processing",
        enabled_modalities=["Kinematics"],
        custom_processors={"Kinematics": custom_processor}
    )
    manager.set_scenario_modality_config("custom_processing", config)
    
    # Test processor
    sample_obs = np.array([[1, 2, 3, 4, 5]])  # [presence, x, y, vx, vy]
    processed = custom_processor.process_observation(sample_obs, {})
    print(f"Original observation shape: {sample_obs.shape}")
    print(f"Processed observation shape: {processed.shape}")
    print(f"Added velocity magnitude: {processed[0, -1]:.2f}")
    
    return manager


def demo_integration():
    """Demonstrate full integration of extensibility features."""
    print("\n=== Integration Demo ===")
    
    # Set up modality configuration
    modality_manager = ModalityConfigManager()
    
    # Configure different scenarios
    scenarios_config = {
        "free_flow": create_kinematics_only_config("free_flow"),
        "dense_commuting": create_vision_only_config("dense_commuting"),
        "lane_closure": modality_manager.create_scenario_config(
            "lane_closure", 
            enabled_modalities=["Kinematics", "OccupancyGrid"]
        )
    }
    
    for scenario, config in scenarios_config.items():
        modality_manager.set_scenario_modality_config(scenario, config)
    
    # Set up action sampling
    mock_policy = Mock()
    mock_policy.predict = Mock(return_value=(2, None))
    
    # Different sampling strategies for different agents
    hybrid_sampler = HybridActionSampler(
        samplers={
            0: RandomActionSampler(seed=42),  # Agent 0: random
            1: PolicyActionSampler(mock_policy)  # Agent 1: policy
        }
    )
    
    # Create collector with all configurations
    collector = SynchronizedCollector(
        n_agents=2,
        action_sampler=hybrid_sampler,
        modality_config_manager=modality_manager
    )
    
    # Test different scenarios
    for scenario in scenarios_config.keys():
        enabled = collector.get_enabled_modalities(scenario)
        print(f"Scenario '{scenario}': {enabled}")
    
    print("Integration demo completed successfully!")


def main():
    """Run all demonstrations."""
    print("Highway Data Collection - Extensibility Features Demo")
    print("=" * 60)
    
    try:
        demo_action_sampling()
        modality_manager = demo_modality_configuration()
        custom_manager = demo_custom_processors()
        demo_integration()
        
        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("\nKey features demonstrated:")
        print("✓ Multiple action sampling strategies")
        print("✓ Per-scenario modality configuration")
        print("✓ Custom observation processors")
        print("✓ Policy integration hooks")
        print("✓ Hybrid configurations")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()