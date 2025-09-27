"""
Ambulance Highway Scenarios Configuration

This module defines 15 distinct highway scenarios for ambulance data collection.
Each scenario is configured with 4-lane highways, varying traffic densities,
and emergency vehicle situations. All scenarios use horizontal image orientation
and have 4 controlled vehicles with the first agent representing an ambulance.

The scenarios support multi-modal data collection with all 3 observation types:
- Kinematics: Vehicle state data (position, velocity, heading)
- OccupancyGrid: Spatial grid representation of the environment  
- GrayscaleObservation: Visual/image observations of the driving scene

Note: highway-env doesn't have a built-in "ambulance" vehicle type. The ambulance
behavior is implemented through custom environment logic that treats the first
controlled vehicle as an ambulance with emergency response characteristics.
"""

from typing import Dict, Any


def get_base_ambulance_config() -> Dict[str, Any]:
    """
    Get the base configuration shared across all ambulance scenarios.
    
    Note: observation and action configs are set dynamically by MultiAgentEnvFactory
    to support all 3 observation types (Kinematics, OccupancyGrid, GrayscaleObservation)
    
    Returns:
        Dict[str, Any]: Base configuration for ambulance scenarios
    """
    return {
        "lanes_count": 4,  # Fixed 4-lane highway for all scenarios
        "controlled_vehicles": 4,  # 4 controlled agents total
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",  # Rule-based traffic
        "screen_width": 800,  # Horizontal image orientation
        "screen_height": 600,  # Horizontal image orientation
        "centering_position": [0.3, 0.5],  # Horizontal view centering
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent_views": True,
        "offscreen_rendering": True,
        "simulation_frequency": 15,  # Hz
        "policy_frequency": 1,  # Hz
        "real_time_rendering": False,
        "manual_control": False,
        # Ambulance-specific metadata (for custom environment logic)
        "_ambulance_config": {
            "ambulance_agent_index": 0,  # First controlled vehicle is ambulance
            "emergency_priority": "high",
            "ambulance_behavior": "emergency_response"
        }
    }


def get_ambulance_scenarios() -> Dict[str, Dict[str, Any]]:
    """
    Returns a dictionary of 15 ambulance highway scenarios.
    
    Each scenario includes:
    - 4-lane highway configuration
    - Varying traffic densities and conditions
    - 4 controlled vehicles (first agent represents ambulance)
    - Horizontal image orientation settings
    - Support for multi-modal data collection (Kinematics, OccupancyGrid, GrayscaleObservation)
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping scenario names to configurations
    """
    
    base_config = get_base_ambulance_config()
    
    scenarios = {
        # Scenario 1: Light traffic flow
        "highway_emergency_light": {
            **base_config,
            "scenario_name": "highway_emergency_light",
            "description": "Ambulance on highway with light traffic flow",
            "traffic_density": "light",
            "vehicles_count": 15,
            "initial_lane_id": None,
            "duration": 40,
            "highway_conditions": "normal",
            "speed_limit": 30,
            "spawn_probability": 0.3,
            "collision_reward": -1,
            "_ambulance_config": {
                **base_config["_ambulance_config"],
                "emergency_priority": "high"
            }
        },
        
        # Scenario 2: Moderate highway traffic
        "highway_emergency_moderate": {
            **base_config,
            "scenario_name": "highway_emergency_moderate",
            "description": "Ambulance navigating moderate highway traffic",
            "traffic_density": "moderate",
            "vehicles_count": 25,
            "initial_lane_id": None,
            "duration": 45,
            "highway_conditions": "normal",
            "speed_limit": 30,
            "spawn_probability": 0.5,
            "collision_reward": -1,
            "_ambulance_config": {
                **base_config["_ambulance_config"],
                "emergency_priority": "high"
            }
        },
        
        # Scenario 3: Heavy highway congestion
        "highway_emergency_dense": {
            **base_config,
            "scenario_name": "highway_emergency_dense",
            "description": "Ambulance in heavy highway congestion",
            "traffic_density": "heavy",
            "vehicles_count": 40,
            "initial_lane_id": None,
            "duration": 50,
            "highway_conditions": "congested",
            "speed_limit": 25,
            "spawn_probability": 0.8,
            "collision_reward": -1,
            "_ambulance_config": {
                **base_config["_ambulance_config"],
                "emergency_priority": "high"
            }
        },
        
        # Scenario 4: Lane closure situation
        "highway_lane_closure": {
            **base_config,
            "scenario_name": "highway_lane_closure",
            "description": "Ambulance navigating highway with lane closure",
            "traffic_density": "moderate",
            "vehicles_count": 30,
            "initial_lane_id": None,
            "duration": 45,
            "highway_conditions": "construction",
            "speed_limit": 20,
            "spawn_probability": 0.6,
            "collision_reward": -1,
            "lanes_count": 3,  # Reduced lanes due to closure
            "_ambulance_config": {
                **base_config["_ambulance_config"],
                "emergency_priority": "high"
            }
        },
        
        # Scenario 5: Rush hour traffic
        "highway_rush_hour": {
            **base_config,
            "scenario_name": "highway_rush_hour",
            "description": "Ambulance during peak highway rush hour",
            "traffic_density": "heavy",
            "vehicles_count": 45,
            "initial_lane_id": None,
            "duration": 55,
            "highway_conditions": "rush_hour",
            "speed_limit": 25,
            "spawn_probability": 0.9,
            "collision_reward": -1,
            "_ambulance_config": {
                **base_config["_ambulance_config"],
                "emergency_priority": "high"
            }
        },
        
        # Scenario 6: Accident scene approach
        "highway_accident_scene": {
            **base_config,
            "scenario_name": "highway_accident_scene",
            "description": "Ambulance approaching highway accident location",
            "traffic_density": "moderate",
            "vehicles_count": 25,
            "initial_lane_id": None,
            "duration": 40,
            "highway_conditions": "accident",
            "speed_limit": 15,
            "spawn_probability": 0.4,
            "collision_reward": -2,
            "_ambulance_config": {
                **base_config["_ambulance_config"],
                "emergency_priority": "high"
            }
        },
        
        # Scenario 7: Construction zone
        "highway_construction": {
            **base_config,
            "scenario_name": "highway_construction",
            "description": "Ambulance through highway construction zone",
            "traffic_density": "moderate",
            "vehicles_count": 20,
            "initial_lane_id": None,
            "duration": 50,
            "highway_conditions": "construction",
            "speed_limit": 15,
            "spawn_probability": 0.3,
            "collision_reward": -1,
            "_ambulance_config": {
                **base_config["_ambulance_config"],
                "emergency_priority": "medium"
            }
        },
        
        # Scenario 8: Weather conditions
        "highway_weather_conditions": {
            **base_config,
            "scenario_name": "highway_weather_conditions",
            "description": "Ambulance on highway with weather challenges",
            "traffic_density": "light",
            "vehicles_count": 18,
            "initial_lane_id": None,
            "duration": 45,
            "highway_conditions": "weather",
            "speed_limit": 20,
            "spawn_probability": 0.4,
            "collision_reward": -1,
            "_ambulance_config": {
                **base_config["_ambulance_config"],
                "emergency_priority": "high"
            }
        },
        
        # Scenario 9: Stop-and-go traffic
        "highway_stop_and_go": {
            **base_config,
            "scenario_name": "highway_stop_and_go",
            "description": "Ambulance in stop-and-go highway traffic",
            "traffic_density": "heavy",
            "vehicles_count": 35,
            "initial_lane_id": None,
            "duration": 60,
            "highway_conditions": "stop_and_go",
            "speed_limit": 10,
            "spawn_probability": 0.7,
            "collision_reward": -1,
            "_ambulance_config": {
                **base_config["_ambulance_config"],
                "emergency_priority": "high"
            }
        },
        
        # Scenario 10: Aggressive drivers
        "highway_aggressive_drivers": {
            **base_config,
            "scenario_name": "highway_aggressive_drivers",
            "description": "Ambulance with aggressive highway drivers",
            "traffic_density": "moderate",
            "vehicles_count": 28,
            "initial_lane_id": None,
            "duration": 45,
            "highway_conditions": "aggressive",
            "speed_limit": 30,
            "other_vehicles_type": "highway_env.vehicle.behavior.AggressiveVehicle",
            "spawn_probability": 0.6,
            "collision_reward": -1,
            "_ambulance_config": {
                **base_config["_ambulance_config"],
                "emergency_priority": "high"
            }
        }
    }
    
    return scenarios


def get_additional_ambulance_scenarios() -> Dict[str, Dict[str, Any]]:
    """
    Returns the remaining 5 ambulance highway scenarios (11-15).
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping scenario names to configurations
    """
    
    base_config = get_base_ambulance_config()
    
    additional_scenarios = {
        # Scenario 11: Heavy merge areas
        "highway_merge_heavy": {
            **base_config,
            "scenario_name": "highway_merge_heavy",
            "description": "Ambulance navigating heavy highway merge areas",
            "traffic_density": "heavy",
            "vehicles_count": 38,
            "initial_lane_id": None,
            "duration": 50,
            "highway_conditions": "merge_heavy",
            "speed_limit": 25,
            "spawn_probability": 0.8,
            "collision_reward": -1,
            "_ambulance_config": {
                **base_config["_ambulance_config"],
                "emergency_priority": "high"
            }
        },
        
        # Scenario 12: Varying speed zones
        "highway_speed_variation": {
            **base_config,
            "scenario_name": "highway_speed_variation",
            "description": "Ambulance with varying highway speed zones",
            "traffic_density": "moderate",
            "vehicles_count": 25,
            "initial_lane_id": None,
            "duration": 45,
            "highway_conditions": "speed_variation",
            "speed_limit": 35,
            "spawn_probability": 0.5,
            "collision_reward": -1,
            "_ambulance_config": {
                **base_config["_ambulance_config"],
                "emergency_priority": "medium"
            }
        },
        
        # Scenario 13: Shoulder use scenario
        "highway_shoulder_use": {
            **base_config,
            "scenario_name": "highway_shoulder_use",
            "description": "Ambulance using highway shoulder when needed",
            "traffic_density": "heavy",
            "vehicles_count": 35,
            "initial_lane_id": None,
            "duration": 40,
            "highway_conditions": "shoulder_available",
            "speed_limit": 20,
            "spawn_probability": 0.7,
            "collision_reward": -1,
            "_ambulance_config": {
                **base_config["_ambulance_config"],
                "emergency_priority": "high"
            }
        },
        
        # Scenario 14: Heavy truck traffic
        "highway_truck_heavy": {
            **base_config,
            "scenario_name": "highway_truck_heavy",
            "description": "Ambulance on highway with heavy truck traffic",
            "traffic_density": "moderate",
            "vehicles_count": 22,
            "initial_lane_id": None,
            "duration": 50,
            "highway_conditions": "truck_heavy",
            "speed_limit": 25,
            "spawn_probability": 0.4,
            "collision_reward": -1,
            "_ambulance_config": {
                **base_config["_ambulance_config"],
                "emergency_priority": "high"
            }
        },
        
        # Scenario 15: High time pressure
        "highway_time_pressure": {
            **base_config,
            "scenario_name": "highway_time_pressure",
            "description": "Ambulance with high time pressure scenario",
            "traffic_density": "moderate",
            "vehicles_count": 30,
            "initial_lane_id": None,
            "duration": 35,
            "highway_conditions": "time_critical",
            "speed_limit": 35,
            "spawn_probability": 0.6,
            "collision_reward": -2,
            "_ambulance_config": {
                **base_config["_ambulance_config"],
                "emergency_priority": "high"
            }
        }
    }
    
    return additional_scenarios


def get_all_ambulance_scenarios() -> Dict[str, Dict[str, Any]]:
    """
    Returns all 15 ambulance highway scenarios combined.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping all scenario names to configurations
    """
    all_scenarios = {}
    all_scenarios.update(get_ambulance_scenarios())
    all_scenarios.update(get_additional_ambulance_scenarios())
    return all_scenarios


def validate_ambulance_scenario(scenario_config: Dict[str, Any]) -> bool:
    """
    Validates an ambulance scenario configuration.
    
    Args:
        scenario_config: The scenario configuration to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_fields = [
        "scenario_name", "controlled_vehicles", "_ambulance_config",
        "lanes_count", "screen_width", "screen_height"
    ]
    
    # Check required fields
    for field in required_fields:
        if field not in scenario_config:
            return False
    
    # Validate specific constraints
    if scenario_config["controlled_vehicles"] != 4:
        return False
    
    # Check ambulance config
    ambulance_config = scenario_config.get("_ambulance_config", {})
    if ambulance_config.get("ambulance_agent_index") != 0:
        return False
    
    # Validate horizontal orientation
    if scenario_config["screen_width"] <= scenario_config["screen_height"]:
        return False
    
    # Note: observation and action validation is handled by the environment factory
    # since these are set dynamically to support multi-modal data collection
    
    return True


def get_scenario_names() -> list:
    """
    Returns a list of all ambulance scenario names.
    
    Returns:
        list: List of scenario names
    """
    return list(get_all_ambulance_scenarios().keys())


def get_scenario_by_name(scenario_name: str) -> Dict[str, Any]:
    """
    Retrieves a specific ambulance scenario by name.
    
    Args:
        scenario_name: Name of the scenario to retrieve
        
    Returns:
        Dict[str, Any]: Scenario configuration
        
    Raises:
        KeyError: If scenario name is not found
    """
    all_scenarios = get_all_ambulance_scenarios()
    if scenario_name not in all_scenarios:
        raise KeyError(f"Scenario '{scenario_name}' not found. Available scenarios: {list(all_scenarios.keys())}")
    
    return all_scenarios[scenario_name]


def get_supported_observation_types() -> list:
    """
    Returns the list of supported observation types for ambulance scenarios.
    
    Returns:
        list: List of supported observation types
    """
    return ["Kinematics", "OccupancyGrid", "GrayscaleObservation"]


# Export the main functions for easy import
__all__ = [
    'get_base_ambulance_config',
    'get_ambulance_scenarios',
    'get_additional_ambulance_scenarios', 
    'get_all_ambulance_scenarios',
    'validate_ambulance_scenario',
    'get_scenario_names',
    'get_scenario_by_name',
    'get_supported_observation_types'
]