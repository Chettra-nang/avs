"""
Environment factory for creating multi-agent HighwayEnv instances.
"""

from typing import Dict, Any, List, Optional
import gymnasium as gym
import highway_env
import copy

from ..scenarios.registry import ScenarioRegistry
from .config import OBSERVATION_CONFIGS


class MultiAgentEnvFactory:
    """
    Factory class for creating and configuring HighwayEnv instances 
    with different observation modalities.
    
    Ensures consistent multi-agent configurations across different observation
    types while maintaining synchronization through identical base configurations.
    """
    
    def __init__(self):
        self._scenario_registry = ScenarioRegistry()
        self._supported_obs_types = list(OBSERVATION_CONFIGS.keys())
    
    def create_env(self, scenario_name: str, obs_type: str, n_agents: int) -> gym.Env:
        """
        Create a single environment instance.
        
        Args:
            scenario_name: Name of the scenario configuration
            obs_type: Type of observation (Kinematics, OccupancyGrid, GrayscaleObservation)
            n_agents: Number of controlled agents
            
        Returns:
            Configured gymnasium environment
            
        Raises:
            ValueError: If obs_type is not supported or n_agents is invalid
            KeyError: If scenario_name is not found
        """
        if obs_type not in self._supported_obs_types:
            raise ValueError(f"Unsupported observation type: {obs_type}. "
                           f"Supported types: {self._supported_obs_types}")
        
        if n_agents < 1:
            raise ValueError(f"Number of agents must be >= 1, got {n_agents}")
        
        # Get base configuration
        base_config = self.get_base_config(scenario_name, n_agents)
        
        # Add observation-specific configuration
        if n_agents > 1:
            # For multi-agent environments, wrap the observation and action configs
            obs_config = {
                "type": "MultiAgentObservation",
                "observation_config": copy.deepcopy(OBSERVATION_CONFIGS[obs_type])
            }
            action_config = {
                "type": "MultiAgentAction",
                "action_config": {"type": "DiscreteMetaAction"}
            }
        else:
            obs_config = copy.deepcopy(OBSERVATION_CONFIGS[obs_type])
            action_config = {"type": "DiscreteMetaAction"}
        
        base_config["observation"] = obs_config
        base_config["action"] = action_config
        
        # Create and configure environment
        env = gym.make("highway-v0", config=base_config)
        
        return env
    
    def create_parallel_envs(self, scenario_name: str, n_agents: int, 
                           enabled_modalities: Optional[List[str]] = None) -> Dict[str, gym.Env]:
        """
        Create parallel environments for specified observation modalities.
        
        Args:
            scenario_name: Name of the scenario configuration
            n_agents: Number of controlled agents
            enabled_modalities: List of modalities to create environments for (None for all)
            
        Returns:
            Dictionary mapping observation types to environment instances
        """
        parallel_envs = {}
        
        # Use specified modalities or all supported types
        modalities_to_create = enabled_modalities or self._supported_obs_types
        
        for obs_type in modalities_to_create:
            if obs_type in self._supported_obs_types:
                parallel_envs[obs_type] = self.create_env(scenario_name, obs_type, n_agents)
            else:
                raise ValueError(f"Unsupported modality: {obs_type}. "
                               f"Supported modalities: {self._supported_obs_types}")
        
        return parallel_envs
    
    def get_base_config(self, scenario_name: str, n_agents: int) -> Dict[str, Any]:
        """
        Get base configuration for environment creation.
        
        Creates consistent multi-agent configuration with DiscreteMetaAction
        that can be used across different observation modalities.
        
        Args:
            scenario_name: Name of the scenario configuration
            n_agents: Number of controlled agents
            
        Returns:
            Base configuration dictionary
            
        Raises:
            KeyError: If scenario_name is not found
            ValueError: If n_agents is invalid
        """
        if n_agents < 1:
            raise ValueError(f"Number of agents must be >= 1, got {n_agents}")
        
        # Get scenario configuration
        scenario_config = self._scenario_registry.get_scenario_config(scenario_name)
        
        # Build base configuration for multi-agent setup
        base_config = {
            # Environment setup
            "lanes_count": scenario_config["lanes_count"],
            "vehicles_count": scenario_config["vehicles_count"],
            "duration": scenario_config["duration"],
            "initial_lane_id": scenario_config.get("initial_lane_id"),
            "ego_spacing": scenario_config.get("ego_spacing", 2),
            "other_vehicles_type": scenario_config.get("other_vehicles_type", 
                                                     "highway_env.vehicle.behavior.IDMVehicle"),
            
            # Multi-agent configuration
            "controlled_vehicles": n_agents,
            
            # Simulation parameters
            "simulation_frequency": scenario_config.get("simulation_frequency", 15),
            "policy_frequency": scenario_config.get("policy_frequency", 1),
            
            # Rendering configuration
            "render_mode": None,
            "real_time_rendering": scenario_config.get("real_time_rendering", False),
            "show_trajectories": scenario_config.get("show_trajectories", False),
            "render_agent": scenario_config.get("render_agent", True),
            "offscreen_rendering": scenario_config.get("offscreen_rendering", False),
            "manual_control": scenario_config.get("manual_control", False),
            
            # Reward configuration
            "high_speed_reward": scenario_config.get("high_speed_reward", 0.4),
            "right_lane_reward": scenario_config.get("right_lane_reward", 0.1),
            "lane_change_reward": scenario_config.get("lane_change_reward", 0),
            "reward_speed_range": scenario_config.get("reward_speed_range", [20, 30]),
            "normalize_reward": scenario_config.get("normalize_reward", True),
            
            # Visual configuration (used by some observation types)
            "centering_position": scenario_config.get("centering_position", [0.3, 0.5]),
            "scaling": scenario_config.get("scaling", 5.5),
        }
        
        # Add scenario-specific parameters
        if "initial_spacing" in scenario_config:
            base_config["initial_spacing"] = scenario_config["initial_spacing"]
        
        return base_config
    
    def get_supported_observation_types(self) -> List[str]:
        """
        Get list of supported observation types.
        
        Returns:
            List of supported observation type names
        """
        return self._supported_obs_types.copy()
    
    def validate_configuration(self, scenario_name: str, obs_type: str, n_agents: int) -> bool:
        """
        Validate configuration parameters before environment creation.
        
        Args:
            scenario_name: Name of the scenario configuration
            obs_type: Type of observation
            n_agents: Number of controlled agents
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check if scenario exists
            self._scenario_registry.get_scenario_config(scenario_name)
            
            # Check if observation type is supported
            if obs_type not in self._supported_obs_types:
                return False
            
            # Check if n_agents is valid
            if n_agents < 1:
                return False
            
            return True
        except (KeyError, ValueError):
            return False