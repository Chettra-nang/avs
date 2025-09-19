"""
Configuration constants for environment creation.
"""

from typing import Dict, Any

# Environment configuration templates
ENV_CONFIGS = {
    "base": {
        "import_module": "highway_env",
        "id": "highway-v0",
        "render_mode": None,
    }
}

# Observation configuration for different modalities
OBSERVATION_CONFIGS = {
    "Kinematics": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100], 
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False,
        "flatten": False,
        "observe_intentions": False
    },
    
    "OccupancyGrid": {
        "type": "OccupancyGrid",
        "vehicles_count": 15,
        "features": ["presence", "on_road"],
        "grid_size": [[-5.5*5, 5.5*5], [-5.5*5, 5.5*5]],
        "grid_step": [5, 5],
        "absolute": False,
        "flatten": False
    },
    
    "GrayscaleObservation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],  # RGB to grayscale weights
        "scaling": 1.75,
        "centering_position": [0.3, 0.5]
    }
}