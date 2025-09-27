"""
Ambulance Vehicle Implementation

This module provides custom ambulance vehicle classes that extend highway-env's
vehicle types to support ambulance-specific behavior and visual appearance.
"""

from typing import Optional, Tuple, List
import numpy as np

try:
    from highway_env.vehicle.controller import MDPVehicle, ControlledVehicle
    from highway_env.vehicle.behavior import IDMVehicle
    from highway_env.road.road import Road
    from highway_env.vehicle.kinematics import Vehicle
    HIGHWAY_ENV_AVAILABLE = True
except ImportError:
    # Fallback for when highway-env is not available
    HIGHWAY_ENV_AVAILABLE = False
    MDPVehicle = object
    ControlledVehicle = object
    IDMVehicle = object


class AmbulanceVehicle(ControlledVehicle if HIGHWAY_ENV_AVAILABLE else object):
    """
    Ambulance vehicle class that extends ControlledVehicle with emergency behavior.
    
    This class provides ambulance-specific behavior including:
    - Emergency response characteristics
    - Visual distinction (red color)
    - Higher priority in traffic interactions
    - Emergency driving patterns
    """
    
    # Ambulance visual configuration
    DEFAULT_COLOR = (255, 0, 0)  # Red color for ambulance
    LENGTH = 6.0  # Slightly longer than regular vehicles
    WIDTH = 2.2   # Slightly wider than regular vehicles
    
    def __init__(self, road: "Road", position: np.ndarray, heading: float = 0, 
                 speed: float = 0, target_lane_index: Optional[Tuple[str, str, int]] = None,
                 target_speed: Optional[float] = None, route: Optional[List[Tuple[str, str, int]]] = None,
                 emergency_priority: str = "high"):
        """
        Initialize ambulance vehicle.
        
        Args:
            road: The road on which the vehicle is driving
            position: Initial position [x, y]
            heading: Initial heading angle in radians
            speed: Initial speed in m/s
            target_lane_index: Target lane for the vehicle
            target_speed: Target speed for the vehicle
            route: Route for the vehicle to follow
            emergency_priority: Emergency priority level ("high", "medium", "low")
        """
        if not HIGHWAY_ENV_AVAILABLE:
            raise ImportError("highway-env is required for AmbulanceVehicle")
            
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        
        # Ambulance-specific attributes
        self.emergency_priority = emergency_priority
        self.is_emergency_vehicle = True
        self.vehicle_type = "ambulance"
        
        # Override visual properties
        self.color = self.DEFAULT_COLOR
        self.LENGTH = self.LENGTH
        self.WIDTH = self.WIDTH
        
        # Emergency behavior parameters
        self._emergency_speed_factor = 1.2 if emergency_priority == "high" else 1.1
        self._lane_change_urgency = 2.0 if emergency_priority == "high" else 1.5
        
    def act(self, action: Optional[int] = None) -> None:
        """
        Execute ambulance-specific action with emergency behavior modifications.
        
        Args:
            action: Action to execute (if None, uses default behavior)
        """
        # Apply emergency behavior modifications
        if hasattr(self, 'target_speed') and self.target_speed is not None:
            # Ambulances can exceed normal speed limits during emergencies
            emergency_target_speed = self.target_speed * self._emergency_speed_factor
            original_target_speed = self.target_speed
            self.target_speed = emergency_target_speed
            
            # Execute the action
            super().act(action)
            
            # Restore original target speed
            self.target_speed = original_target_speed
        else:
            super().act(action)
    
    def get_emergency_priority(self) -> str:
        """Get the emergency priority level."""
        return self.emergency_priority
    
    def set_emergency_priority(self, priority: str) -> None:
        """
        Set the emergency priority level.
        
        Args:
            priority: Priority level ("high", "medium", "low")
        """
        if priority not in ["high", "medium", "low"]:
            raise ValueError(f"Invalid priority: {priority}. Must be 'high', 'medium', or 'low'")
        
        self.emergency_priority = priority
        self._emergency_speed_factor = 1.2 if priority == "high" else 1.1
        self._lane_change_urgency = 2.0 if priority == "high" else 1.5
    
    def is_ambulance(self) -> bool:
        """Check if this vehicle is an ambulance."""
        return True
    
    @classmethod
    def create_from_config(cls, road: "Road", position: np.ndarray, config: dict) -> "AmbulanceVehicle":
        """
        Create ambulance vehicle from configuration dictionary.
        
        Args:
            road: The road on which the vehicle is driving
            position: Initial position [x, y]
            config: Configuration dictionary with ambulance parameters
            
        Returns:
            Configured AmbulanceVehicle instance
        """
        return cls(
            road=road,
            position=position,
            heading=config.get("heading", 0),
            speed=config.get("speed", 0),
            target_lane_index=config.get("target_lane_index"),
            target_speed=config.get("target_speed"),
            route=config.get("route"),
            emergency_priority=config.get("emergency_priority", "high")
        )


class AmbulanceIDMVehicle(IDMVehicle if HIGHWAY_ENV_AVAILABLE else object):
    """
    Ambulance vehicle with IDM (Intelligent Driver Model) behavior.
    
    This class combines ambulance characteristics with IDM driving behavior
    for use as non-controlled ambulance vehicles in the environment.
    """
    
    DEFAULT_COLOR = (255, 0, 0)  # Red color for ambulance
    LENGTH = 6.0
    WIDTH = 2.2
    
    def __init__(self, road: "Road", position: np.ndarray, heading: float = 0, 
                 speed: float = 0, target_lane_index: Optional[Tuple[str, str, int]] = None,
                 target_speed: Optional[float] = None, route: Optional[List[Tuple[str, str, int]]] = None,
                 emergency_priority: str = "high"):
        """Initialize ambulance IDM vehicle."""
        if not HIGHWAY_ENV_AVAILABLE:
            raise ImportError("highway-env is required for AmbulanceIDMVehicle")
            
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        
        # Ambulance-specific attributes
        self.emergency_priority = emergency_priority
        self.is_emergency_vehicle = True
        self.vehicle_type = "ambulance"
        
        # Override visual properties
        self.color = self.DEFAULT_COLOR
        self.LENGTH = self.LENGTH
        self.WIDTH = self.WIDTH
    
    def is_ambulance(self) -> bool:
        """Check if this vehicle is an ambulance."""
        return True


def create_ambulance_vehicle(road: "Road", position: np.ndarray, 
                           vehicle_class: str = "controlled", **kwargs) -> Vehicle:
    """
    Factory function to create ambulance vehicles.
    
    Args:
        road: The road on which the vehicle is driving
        position: Initial position [x, y]
        vehicle_class: Type of ambulance vehicle ("controlled" or "idm")
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured ambulance vehicle instance
    """
    if not HIGHWAY_ENV_AVAILABLE:
        raise ImportError("highway-env is required for ambulance vehicles")
    
    if vehicle_class == "controlled":
        return AmbulanceVehicle(road, position, **kwargs)
    elif vehicle_class == "idm":
        return AmbulanceIDMVehicle(road, position, **kwargs)
    else:
        raise ValueError(f"Unknown ambulance vehicle class: {vehicle_class}")


def is_ambulance_vehicle(vehicle: Vehicle) -> bool:
    """
    Check if a vehicle is an ambulance.
    
    Args:
        vehicle: Vehicle instance to check
        
    Returns:
        True if the vehicle is an ambulance, False otherwise
    """
    return hasattr(vehicle, 'is_ambulance') and vehicle.is_ambulance()


def get_ambulance_config() -> dict:
    """
    Get default configuration for ambulance vehicles.
    
    Returns:
        Dictionary with default ambulance configuration
    """
    return {
        "emergency_priority": "high",
        "color": (255, 0, 0),  # Red
        "length": 6.0,
        "width": 2.2,
        "emergency_speed_factor": 1.2,
        "lane_change_urgency": 2.0
    }


# Export main classes and functions
__all__ = [
    'AmbulanceVehicle',
    'AmbulanceIDMVehicle', 
    'create_ambulance_vehicle',
    'is_ambulance_vehicle',
    'get_ambulance_config',
    'HIGHWAY_ENV_AVAILABLE'
]