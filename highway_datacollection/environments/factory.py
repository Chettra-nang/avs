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
    
    def create_ambulance_env(self, scenario_name: str, obs_type: str, n_agents: int) -> gym.Env:
        """
        Create a single ambulance environment instance with ambulance ego vehicle.
        
        Args:
            scenario_name: Name of the ambulance scenario configuration
            obs_type: Type of observation (Kinematics, OccupancyGrid, GrayscaleObservation)
            n_agents: Number of controlled agents (first agent will be ambulance)
            
        Returns:
            Configured gymnasium environment with ambulance ego vehicle
            
        Raises:
            ValueError: If obs_type is not supported or n_agents is invalid
            KeyError: If scenario_name is not found
        """
        if obs_type not in self._supported_obs_types:
            raise ValueError(f"Unsupported observation type: {obs_type}. "
                           f"Supported types: {self._supported_obs_types}")
        
        if n_agents < 1:
            raise ValueError(f"Number of agents must be >= 1, got {n_agents}")
        
        # Get ambulance-specific base configuration
        base_config = self.get_ambulance_base_config(scenario_name, n_agents)
        
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
        
        # Configure ambulance-specific behavior after environment creation
        self._configure_ambulance_environment(env, base_config)
        
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
    
    def create_parallel_ambulance_envs(self, scenario_name: str, n_agents: int, 
                                     enabled_modalities: Optional[List[str]] = None) -> Dict[str, gym.Env]:
        """
        Create parallel ambulance environments for specified observation modalities.
        
        Args:
            scenario_name: Name of the ambulance scenario configuration
            n_agents: Number of controlled agents (first will be ambulance)
            enabled_modalities: List of modalities to create environments for (None for all)
            
        Returns:
            Dictionary mapping observation types to ambulance environment instances
        """
        parallel_envs = {}
        
        # Use specified modalities or all supported types
        modalities_to_create = enabled_modalities or self._supported_obs_types
        
        for obs_type in modalities_to_create:
            if obs_type in self._supported_obs_types:
                parallel_envs[obs_type] = self.create_ambulance_env(scenario_name, obs_type, n_agents)
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
        
        # Check if this is an ambulance scenario
        is_ambulance_scenario = scenario_config.get("ego_vehicle_type") == "ambulance"
        
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
            
            # Vehicle type configuration (handle ambulance if specified)
            "ego_vehicle_type": scenario_config.get("ego_vehicle_type", "normal"),
            
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
    
    def get_ambulance_base_config(self, scenario_name: str, n_agents: int) -> Dict[str, Any]:
        """
        Get base configuration for ambulance environment creation.
        
        Creates ambulance-specific multi-agent configuration where the first
        controlled vehicle (index 0) is configured as an ambulance type.
        
        Args:
            scenario_name: Name of the ambulance scenario configuration
            n_agents: Number of controlled agents (first will be ambulance)
            
        Returns:
            Base configuration dictionary with ambulance-specific settings
            
        Raises:
            KeyError: If scenario_name is not found
            ValueError: If n_agents is invalid
        """
        if n_agents < 1:
            raise ValueError(f"Number of agents must be >= 1, got {n_agents}")
        
        # Validate n_agents for ambulance scenarios (3-4 agents recommended)
        if n_agents > 4:
            raise ValueError(f"Ambulance scenarios support maximum 4 agents, got {n_agents}")
        
        # Get scenario configuration - try ambulance scenarios first
        try:
            # Import ambulance scenarios to get configuration
            from collecting_ambulance_data.scenarios.ambulance_scenarios import get_scenario_by_name
            scenario_config = get_scenario_by_name(scenario_name)
        except (ImportError, KeyError):
            # Fall back to regular scenario registry
            scenario_config = self._scenario_registry.get_scenario_config(scenario_name)
        
        # Build base configuration for ambulance multi-agent setup
        base_config = {
            # Environment setup - ensure 4-lane highway
            "lanes_count": 4,  # Fixed 4-lane highway for all ambulance scenarios
            "vehicles_count": scenario_config.get("vehicles_count", 20),
            "duration": scenario_config.get("duration", 40),
            "initial_lane_id": scenario_config.get("initial_lane_id"),
            "ego_spacing": scenario_config.get("ego_spacing", 2),
            "other_vehicles_type": scenario_config.get("other_vehicles_type", 
                                                     "highway_env.vehicle.behavior.IDMVehicle"),
            
            # Multi-agent configuration with ambulance support (3-4 controlled agents)
            "controlled_vehicles": n_agents,
            
            # Ambulance-specific configuration metadata
            # Note: highway-env doesn't have built-in ambulance type, so we use metadata
            # to track ambulance behavior and implement custom logic in the collector
            "_ambulance_config": {
                "ambulance_agent_index": 0,  # First controlled vehicle is ambulance
                "emergency_priority": scenario_config.get("_ambulance_config", {}).get("emergency_priority", "high"),
                "ambulance_behavior": scenario_config.get("_ambulance_config", {}).get("ambulance_behavior", "emergency_response"),
                "other_agents_type": "normal"  # Other controlled agents are normal vehicles
            },
            
            # Simulation parameters
            "simulation_frequency": scenario_config.get("simulation_frequency", 15),
            "policy_frequency": scenario_config.get("policy_frequency", 1),
            
            # Rendering configuration (horizontal orientation for ambulance scenarios)
            "render_mode": None,
            "real_time_rendering": scenario_config.get("real_time_rendering", False),
            "show_trajectories": scenario_config.get("show_trajectories", False),
            "render_agent": scenario_config.get("render_agent", True),
            "offscreen_rendering": scenario_config.get("offscreen_rendering", True),
            "manual_control": scenario_config.get("manual_control", False),
            
            # Screen configuration for horizontal orientation (800x600)
            "screen_width": 800,  # Horizontal orientation
            "screen_height": 600,  # Horizontal orientation
            
            # Reward configuration
            "high_speed_reward": scenario_config.get("high_speed_reward", 0.4),
            "right_lane_reward": scenario_config.get("right_lane_reward", 0.1),
            "lane_change_reward": scenario_config.get("lane_change_reward", 0),
            "reward_speed_range": scenario_config.get("reward_speed_range", [20, 30]),
            "normalize_reward": scenario_config.get("normalize_reward", True),
            "collision_reward": scenario_config.get("collision_reward", -1),
            
            # Visual configuration (horizontal orientation)
            "centering_position": scenario_config.get("centering_position", [0.3, 0.5]),
            "scaling": scenario_config.get("scaling", 5.5),
        }
        
        # Add scenario-specific parameters
        if "initial_spacing" in scenario_config:
            base_config["initial_spacing"] = scenario_config["initial_spacing"]
        
        if "spawn_probability" in scenario_config:
            base_config["spawn_probability"] = scenario_config["spawn_probability"]
            
        if "speed_limit" in scenario_config:
            base_config["speed_limit"] = scenario_config["speed_limit"]
            
        if "highway_conditions" in scenario_config:
            base_config["highway_conditions"] = scenario_config["highway_conditions"]
        
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
            # Check if scenario exists (try both regular and ambulance scenarios)
            try:
                self._scenario_registry.get_scenario_config(scenario_name)
            except KeyError:
                # Try ambulance scenarios
                from collecting_ambulance_data.scenarios.ambulance_scenarios import get_scenario_by_name
                get_scenario_by_name(scenario_name)
            
            # Check if observation type is supported
            if obs_type not in self._supported_obs_types:
                return False
            
            # Check if n_agents is valid
            if n_agents < 1:
                return False
            
            return True
        except (KeyError, ValueError, ImportError):
            return False
    
    def validate_ambulance_configuration(self, scenario_name: str, obs_type: str, n_agents: int) -> bool:
        """
        Validate ambulance-specific configuration parameters.
        
        Args:
            scenario_name: Name of the ambulance scenario configuration
            obs_type: Type of observation
            n_agents: Number of controlled agents
            
        Returns:
            True if ambulance configuration is valid, False otherwise
        """
        try:
            # Import and validate ambulance scenario
            from collecting_ambulance_data.scenarios.ambulance_scenarios import get_scenario_by_name, validate_ambulance_scenario
            scenario_config = get_scenario_by_name(scenario_name)
            
            # Validate ambulance-specific constraints
            if not validate_ambulance_scenario(scenario_config):
                return False
            
            # Check if observation type is supported
            if obs_type not in self._supported_obs_types:
                return False
            
            # Check if n_agents is valid (ambulance scenarios typically use 3-4 agents)
            if n_agents < 1 or n_agents > 4:
                return False
            
            # Ensure first agent is ambulance
            if scenario_config.get("ambulance_agent_index", 0) != 0:
                return False
            
            return True
        except (KeyError, ValueError, ImportError):
            return False
    
    def _configure_ambulance_environment(self, env: gym.Env, config: Dict[str, Any]) -> None:
        """
        Configure ambulance-specific behavior in the environment.
        
        This method sets up the first controlled vehicle as an ambulance with
        emergency response characteristics and visual distinction.
        
        Args:
            env: The highway environment to configure
            config: Configuration dictionary with ambulance settings
        """
        try:
            # Import ambulance vehicle classes
            from .ambulance_vehicle import AmbulanceVehicle, is_ambulance_vehicle, HIGHWAY_ENV_AVAILABLE
            
            if not HIGHWAY_ENV_AVAILABLE:
                # If highway-env ambulance classes aren't available, use metadata approach
                self._configure_ambulance_metadata(env, config)
                return
            
            # Wait for environment reset to configure vehicles
            # This will be called after the environment creates its vehicles
            if hasattr(env, 'unwrapped'):
                env_unwrapped = env.unwrapped
                
                # Store ambulance configuration for later use
                if not hasattr(env_unwrapped, '_ambulance_config'):
                    env_unwrapped._ambulance_config = config.get('_ambulance_config', {})
                    env_unwrapped._is_ambulance_env = True
                    
                    # Add hook for vehicle creation
                    original_reset = env_unwrapped.reset
                    
                    def ambulance_reset(*args, **kwargs):
                        result = original_reset(*args, **kwargs)
                        self._setup_ambulance_vehicles(env_unwrapped)
                        return result
                    
                    env_unwrapped.reset = ambulance_reset
                    
        except ImportError:
            # Fall back to metadata-based configuration
            self._configure_ambulance_metadata(env, config)
    
    def _configure_ambulance_metadata(self, env: gym.Env, config: Dict[str, Any]) -> None:
        """
        Configure ambulance behavior using metadata approach.
        
        This method stores ambulance configuration metadata that can be used
        by data collection and analysis tools to identify ambulance behavior.
        
        Args:
            env: The highway environment to configure
            config: Configuration dictionary with ambulance settings
        """
        if hasattr(env, 'unwrapped'):
            env_unwrapped = env.unwrapped
            
            # Store ambulance metadata
            env_unwrapped._ambulance_config = config.get('_ambulance_config', {})
            env_unwrapped._is_ambulance_env = True
            env_unwrapped._ambulance_agent_index = config.get('_ambulance_config', {}).get('ambulance_agent_index', 0)
            
            # Mark environment as ambulance-enabled
            env_unwrapped.ambulance_enabled = True
    
    def _setup_ambulance_vehicles(self, env) -> None:
        """
        Set up ambulance vehicles after environment reset.
        
        This method configures the first controlled vehicle as an ambulance
        with appropriate visual and behavioral characteristics.
        
        Args:
            env: The unwrapped highway environment
        """
        try:
            from .ambulance_vehicle import AmbulanceVehicle, is_ambulance_vehicle
            
            if hasattr(env, 'controlled_vehicles') and env.controlled_vehicles:
                ambulance_config = getattr(env, '_ambulance_config', {})
                ambulance_index = ambulance_config.get('ambulance_agent_index', 0)
                
                # Configure first controlled vehicle as ambulance
                if len(env.controlled_vehicles) > ambulance_index:
                    ambulance_vehicle = env.controlled_vehicles[ambulance_index]
                    
                    # Add ambulance properties to the vehicle
                    ambulance_vehicle.is_emergency_vehicle = True
                    ambulance_vehicle.vehicle_type = "ambulance"
                    ambulance_vehicle.emergency_priority = ambulance_config.get('emergency_priority', 'high')
                    
                    # Set ambulance visual properties (red color)
                    if hasattr(ambulance_vehicle, 'color'):
                        ambulance_vehicle.color = (255, 0, 0)  # Red color
                    
                    # Add ambulance identification method
                    ambulance_vehicle.is_ambulance = lambda: True
                    
        except (ImportError, AttributeError):
            # If vehicle modification fails, continue with metadata approach
            pass