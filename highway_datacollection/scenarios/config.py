"""
Configuration constants for highway driving scenarios.
"""

from typing import Dict, Any

# Default parameters for all scenarios
DEFAULT_SCENARIO_PARAMS = {
    "vehicles_count": 50,
    "lanes_count": 4,
    "duration": 40,  # seconds
    "initial_lane_id": None,
    "ego_spacing": 2,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "controlled_vehicles": 2,  # minimum for multi-agent scenarios
    "action": {
        "type": "DiscreteMetaAction",
    },
    "simulation_frequency": 15,  # Hz
    "policy_frequency": 1,  # Hz
    "real_time_rendering": False,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
    "manual_control": False,
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "high_speed_reward": 0.4,
    "right_lane_reward": 0.1,
    "lane_change_reward": 0,
    "reward_speed_range": [20, 30],
    "normalize_reward": True,
}

# Curriculum scenario configurations
SCENARIO_CONFIGS = {
    "free_flow": {
        **DEFAULT_SCENARIO_PARAMS,
        "vehicles_count": 20,
        "lanes_count": 4,
        "duration": 30,
        "initial_spacing": 3,
        "description": "Light traffic with free-flowing conditions"
    },
    
    "dense_commuting": {
        **DEFAULT_SCENARIO_PARAMS,
        "vehicles_count": 80,
        "lanes_count": 4,
        "duration": 45,
        "initial_spacing": 1.5,
        "description": "Heavy commuter traffic with frequent lane changes"
    },
    
    "stop_and_go": {
        **DEFAULT_SCENARIO_PARAMS,
        "vehicles_count": 60,
        "lanes_count": 3,
        "duration": 50,
        "initial_spacing": 1.2,
        "description": "Congested traffic with stop-and-go patterns"
    },
    
    "aggressive_neighbors": {
        **DEFAULT_SCENARIO_PARAMS,
        "vehicles_count": 45,
        "lanes_count": 4,
        "duration": 35,
        "other_vehicles_type": "highway_env.vehicle.behavior.AggressiveVehicle",
        "description": "Moderate traffic with aggressive driving behaviors"
    },
    
    "lane_closure": {
        **DEFAULT_SCENARIO_PARAMS,
        "vehicles_count": 55,
        "lanes_count": 3,  # Simulates lane closure scenario
        "duration": 40,
        "initial_spacing": 1.8,
        "description": "Traffic merging due to lane closure conditions"
    },
    
    "time_budget": {
        **DEFAULT_SCENARIO_PARAMS,
        "vehicles_count": 40,
        "lanes_count": 4,
        "duration": 25,  # Shorter duration for time pressure
        "high_speed_reward": 0.6,  # Higher reward for speed
        "description": "Time-pressured driving with efficiency focus"
    }
}

# Validation constraints for scenario parameters
SCENARIO_CONSTRAINTS = {
    "vehicles_count": {"min": 10, "max": 100},
    "lanes_count": {"min": 2, "max": 6},
    "duration": {"min": 10, "max": 120},
    "controlled_vehicles": {"min": 1, "max": 10},
    "simulation_frequency": {"min": 5, "max": 30},
    "policy_frequency": {"min": 1, "max": 15}
}

# Supported observation types for multi-modal collection
OBSERVATION_TYPES = [
    "Kinematics",
    "OccupancyGrid", 
    "GrayscaleObservation"
]