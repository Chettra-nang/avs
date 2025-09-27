#!/usr/bin/env python3
"""
Debug script for ambulance scenario validation issues.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from collecting_ambulance_data.validation import AmbulanceScenarioValidator
from highway_datacollection.environments.factory import MultiAgentEnvFactory
from highway_datacollection.collection.action_samplers import RandomActionSampler


def debug_single_scenario():
    """Debug a single scenario to understand the issue."""
    # Set up detailed logging
    logging.basicConfig(level=logging.DEBUG)
    
    scenario_name = "highway_emergency_light"
    obs_type = "Kinematics"
    n_agents = 4
    
    print(f"Debugging scenario: {scenario_name} with {obs_type} observation")
    
    try:
        # Create environment factory and action sampler
        env_factory = MultiAgentEnvFactory()
        action_sampler = RandomActionSampler()
        
        print("Creating ambulance environment...")
        env = env_factory.create_ambulance_env(scenario_name, obs_type, n_agents)
        
        print("Resetting environment...")
        obs, info = env.reset()
        
        print(f"Initial observation type: {type(obs)}")
        print(f"Initial observation shape/length: {len(obs) if hasattr(obs, '__len__') else 'N/A'}")
        print(f"Action space: {env.action_space}")
        
        # Try a few steps
        for step in range(3):
            print(f"\nStep {step}:")
            
            # Sample actions
            dummy_observations = {"Kinematics": {"observation": obs}}
            actions = action_sampler.sample_actions(dummy_observations, n_agents, step)
            
            print(f"Actions: {actions}")
            
            # Execute step
            try:
                obs, rewards, dones, truncated, info = env.step(actions)
                
                print(f"Rewards: {rewards}")
                print(f"Dones: {dones}")
                print(f"Truncated: {truncated}")
                print(f"Info keys: {list(info.keys()) if isinstance(info, dict) else type(info)}")
                
                # Check if episode is done
                if isinstance(dones, (list, tuple)):
                    if any(dones) or any(truncated if isinstance(truncated, (list, tuple)) else [truncated]):
                        print("Episode terminated")
                        break
                else:
                    if dones or truncated:
                        print("Episode terminated")
                        break
                        
            except Exception as e:
                print(f"Step execution failed: {e}")
                import traceback
                traceback.print_exc()
                break
        
        env.close()
        print("Environment closed successfully")
        
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_single_scenario()