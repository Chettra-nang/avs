"""
Scenario registry for managing curriculum-based highway driving scenarios.
"""

from typing import Dict, Any, List
import copy
from .config import SCENARIO_CONFIGS, SCENARIO_CONSTRAINTS, DEFAULT_SCENARIO_PARAMS

# Import ambulance scenarios
try:
    import sys
    import os
    
    # Try multiple paths to find the ambulance scenarios
    current_dir = os.path.dirname(__file__)
    possible_paths = [
        # From highway_datacollection/scenarios/ go to root and then to collecting_ambulance_data
        os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'collecting_ambulance_data'),
        # From current working directory
        os.path.join(os.getcwd(), 'collecting_ambulance_data'),
        # Relative path
        'collecting_ambulance_data'
    ]
    
    ambulance_path = None
    for path in possible_paths:
        if os.path.exists(path):
            ambulance_path = path
            break
    
    if ambulance_path and ambulance_path not in sys.path:
        sys.path.insert(0, ambulance_path)
    
    from scenarios.ambulance_scenarios import (
        get_all_ambulance_scenarios, 
        validate_ambulance_scenario,
        get_scenario_names as get_ambulance_scenario_names
    )
    AMBULANCE_SCENARIOS_AVAILABLE = True
except ImportError as e:
    AMBULANCE_SCENARIOS_AVAILABLE = False
    # Store the import error for debugging
    _AMBULANCE_IMPORT_ERROR = str(e)


class ScenarioRegistry:
    """
    Centralized registry for managing highway driving scenario configurations.
    
    Provides access to predefined curriculum scenarios with validation and
    customization capabilities. Now includes support for ambulance scenarios.
    """
    
    def __init__(self):
        self._scenarios = copy.deepcopy(SCENARIO_CONFIGS)
        self._constraints = SCENARIO_CONSTRAINTS
        self._ambulance_scenarios = {}
        self._ambulance_constraints = self._get_ambulance_constraints()
        
        # Load ambulance scenarios if available
        if AMBULANCE_SCENARIOS_AVAILABLE:
            self._ambulance_scenarios = get_all_ambulance_scenarios()
    
    def _get_ambulance_constraints(self) -> Dict[str, Dict[str, Any]]:
        """
        Define validation constraints specific to ambulance scenarios.
        
        Returns:
            Dictionary of ambulance-specific constraints
        """
        return {
            "controlled_vehicles": {"min": 4, "max": 4, "required": 4},
            "_ambulance_config": {"required": True},
            "lanes_count": {"min": 3, "max": 4},
            "vehicles_count": {"min": 10, "max": 50},
            "duration": {"min": 20, "max": 80},
            "screen_width": {"min": 600},
            "screen_height": {"max": 700},
            "traffic_density": {"allowed": ["light", "moderate", "heavy"]},
            "highway_conditions": {"allowed": [
                "normal", "congested", "construction", "rush_hour", 
                "accident", "weather", "stop_and_go", "aggressive",
                "merge_heavy", "speed_variation", "shoulder_available",
                "truck_heavy", "time_critical"
            ]}
            # Note: observation and action configs are set dynamically by MultiAgentEnvFactory
            # to support multi-modal data collection (Kinematics, OccupancyGrid, GrayscaleObservation)
        }
    
    def get_scenario_config(self, scenario_name: str) -> Dict[str, Any]:
        """
        Retrieve configuration for a specific scenario (regular or ambulance).
        
        Args:
            scenario_name: Name of the scenario to retrieve
            
        Returns:
            Dictionary containing scenario configuration parameters
            
        Raises:
            KeyError: If scenario_name is not found in registry
        """
        # Check regular scenarios first
        if scenario_name in self._scenarios:
            return copy.deepcopy(self._scenarios[scenario_name])
        
        # Check ambulance scenarios
        if scenario_name in self._ambulance_scenarios:
            return copy.deepcopy(self._ambulance_scenarios[scenario_name])
        
        # Scenario not found
        available_regular = list(self._scenarios.keys())
        available_ambulance = list(self._ambulance_scenarios.keys())
        all_available = available_regular + available_ambulance
        raise KeyError(f"Scenario '{scenario_name}' not found. Available: {all_available}")
    
    def list_scenarios(self) -> List[str]:
        """
        Get list of all available scenario names (regular and ambulance).
        
        Returns:
            List of scenario names in the registry
        """
        regular_scenarios = list(self._scenarios.keys())
        ambulance_scenarios = list(self._ambulance_scenarios.keys())
        return regular_scenarios + ambulance_scenarios
    
    def list_regular_scenarios(self) -> List[str]:
        """
        Get list of regular (non-ambulance) scenario names.
        
        Returns:
            List of regular scenario names
        """
        return list(self._scenarios.keys())
    
    def list_ambulance_scenarios(self) -> List[str]:
        """
        Get list of ambulance scenario names.
        
        Returns:
            List of ambulance scenario names
        """
        return list(self._ambulance_scenarios.keys())
    
    def validate_scenario(self, config: Dict[str, Any]) -> bool:
        """
        Validate scenario configuration against constraints.
        
        Args:
            config: Scenario configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Determine if this is an ambulance scenario
            is_ambulance = "_ambulance_config" in config
            constraints = self._ambulance_constraints if is_ambulance else self._constraints
            
            # Use ambulance-specific validation if available
            if is_ambulance and AMBULANCE_SCENARIOS_AVAILABLE:
                return validate_ambulance_scenario(config)
            
            # Standard validation
            for param, constraint_info in constraints.items():
                if param in config:
                    value = config[param]
                    
                    # Check required values
                    if "required" in constraint_info:
                        if constraint_info["required"] is True:
                            # Just check presence for boolean required
                            continue
                        elif isinstance(constraint_info["required"], (int, str)) and value != constraint_info["required"]:
                            return False
                    
                    # Check required types for nested objects
                    if "required_type" in constraint_info:
                        if isinstance(value, dict) and value.get("type") != constraint_info["required_type"]:
                            return False
                    
                    # Check allowed values
                    if "allowed" in constraint_info:
                        if value not in constraint_info["allowed"]:
                            return False
                    
                    # Check min/max constraints (ensure numeric values)
                    if "min" in constraint_info and isinstance(value, (int, float)) and value < constraint_info["min"]:
                        return False
                    if "max" in constraint_info and isinstance(value, (int, float)) and value > constraint_info["max"]:
                        return False
            
            return True
        except (TypeError, KeyError):
            return False
    
    def customize_scenario(self, base_scenario: str, **overrides) -> Dict[str, Any]:
        """
        Create customized scenario configuration based on existing scenario.
        
        Args:
            base_scenario: Name of base scenario to customize
            **overrides: Parameter overrides to apply
            
        Returns:
            Customized scenario configuration
            
        Raises:
            KeyError: If base_scenario is not found
            ValueError: If customized configuration is invalid
        """
        config = self.get_scenario_config(base_scenario)
        config.update(overrides)
        
        if not self.validate_scenario(config):
            raise ValueError(f"Customized scenario configuration is invalid: {overrides}")
        
        return config
    
    def customize_ambulance_scenario(self, base_scenario: str, **overrides) -> Dict[str, Any]:
        """
        Create customized ambulance scenario configuration.
        
        Args:
            base_scenario: Name of base ambulance scenario to customize
            **overrides: Parameter overrides to apply
            
        Returns:
            Customized ambulance scenario configuration
            
        Raises:
            KeyError: If base_scenario is not found in ambulance scenarios
            ValueError: If customized configuration is invalid
        """
        if base_scenario not in self._ambulance_scenarios:
            available = list(self._ambulance_scenarios.keys())
            raise KeyError(f"Ambulance scenario '{base_scenario}' not found. Available: {available}")
        
        config = copy.deepcopy(self._ambulance_scenarios[base_scenario])
        
        # Check for invalid overrides before applying them
        if "controlled_vehicles" in overrides and overrides["controlled_vehicles"] != 4:
            raise ValueError(f"Ambulance scenarios must have exactly 4 controlled vehicles, got: {overrides['controlled_vehicles']}")
        
        config.update(overrides)
        
        # Ensure ambulance-specific constraints are maintained
        if "_ambulance_config" not in config:
            config["_ambulance_config"] = {}
        config["_ambulance_config"]["ambulance_agent_index"] = 0
        config["controlled_vehicles"] = 4  # Always enforce this for ambulance scenarios
        
        if not self.validate_scenario(config):
            raise ValueError(f"Customized ambulance scenario configuration is invalid: {overrides}")
        
        return config
    
    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters used across all scenarios.
        
        Returns:
            Dictionary of default scenario parameters
        """
        return copy.deepcopy(DEFAULT_SCENARIO_PARAMS)
    
    def get_ambulance_scenario_config(self, scenario_name: str) -> Dict[str, Any]:
        """
        Retrieve configuration for a specific ambulance scenario.
        
        Args:
            scenario_name: Name of the ambulance scenario to retrieve
            
        Returns:
            Dictionary containing ambulance scenario configuration parameters
            
        Raises:
            KeyError: If scenario_name is not found in ambulance scenarios
            RuntimeError: If ambulance scenarios are not available
        """
        if not AMBULANCE_SCENARIOS_AVAILABLE:
            raise RuntimeError("Ambulance scenarios are not available. Please ensure collecting_ambulance_data is properly installed.")
        
        if scenario_name not in self._ambulance_scenarios:
            available = list(self._ambulance_scenarios.keys())
            raise KeyError(f"Ambulance scenario '{scenario_name}' not found. Available: {available}")
        
        return copy.deepcopy(self._ambulance_scenarios[scenario_name])
    
    def is_ambulance_scenario(self, scenario_name: str) -> bool:
        """
        Check if a scenario is an ambulance scenario.
        
        Args:
            scenario_name: Name of the scenario to check
            
        Returns:
            True if scenario is an ambulance scenario, False otherwise
        """
        return scenario_name in self._ambulance_scenarios
    
    def get_scenarios_by_type(self, scenario_type: str) -> List[str]:
        """
        Get scenarios filtered by type (regular or ambulance).
        
        Args:
            scenario_type: Type of scenarios to retrieve ("regular" or "ambulance")
            
        Returns:
            List of scenario names of the specified type
            
        Raises:
            ValueError: If scenario_type is not "regular" or "ambulance"
        """
        if scenario_type == "regular":
            return self.list_regular_scenarios()
        elif scenario_type == "ambulance":
            return self.list_ambulance_scenarios()
        else:
            raise ValueError(f"Invalid scenario_type '{scenario_type}'. Must be 'regular' or 'ambulance'")
    
    def get_ambulance_scenarios_by_traffic_density(self, density: str) -> List[str]:
        """
        Get ambulance scenarios filtered by traffic density.
        
        Args:
            density: Traffic density to filter by ("light", "moderate", "heavy")
            
        Returns:
            List of ambulance scenario names with the specified traffic density
            
        Raises:
            ValueError: If density is not valid
        """
        if density not in ["light", "moderate", "heavy"]:
            raise ValueError(f"Invalid density '{density}'. Must be 'light', 'moderate', or 'heavy'")
        
        matching_scenarios = []
        for scenario_name, config in self._ambulance_scenarios.items():
            if config.get("traffic_density") == density:
                matching_scenarios.append(scenario_name)
        
        return matching_scenarios
    
    def get_ambulance_scenarios_by_conditions(self, conditions: str) -> List[str]:
        """
        Get ambulance scenarios filtered by highway conditions.
        
        Args:
            conditions: Highway conditions to filter by
            
        Returns:
            List of ambulance scenario names with the specified conditions
        """
        matching_scenarios = []
        for scenario_name, config in self._ambulance_scenarios.items():
            if config.get("highway_conditions") == conditions:
                matching_scenarios.append(scenario_name)
        
        return matching_scenarios
    
    def validate_ambulance_scenario_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate ambulance scenario configuration using ambulance-specific validation.
        
        Args:
            config: Ambulance scenario configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if not AMBULANCE_SCENARIOS_AVAILABLE:
            return False
        
        return validate_ambulance_scenario(config)
    
    def get_supported_observation_types_for_ambulance(self) -> List[str]:
        """
        Get supported observation types for ambulance scenarios.
        
        Returns:
            List of supported observation types for ambulance scenarios
        """
        if not AMBULANCE_SCENARIOS_AVAILABLE:
            return []
        
        try:
            from scenarios.ambulance_scenarios import get_supported_observation_types
            return get_supported_observation_types()
        except ImportError:
            return ["Kinematics", "OccupancyGrid", "GrayscaleObservation"]  # Default fallback
    
    def get_ambulance_constraints(self) -> Dict[str, Dict[str, Any]]:
        """
        Get ambulance-specific validation constraints.
        
        Returns:
            Dictionary of ambulance scenario constraints
        """
        return copy.deepcopy(self._ambulance_constraints)
    
    def get_ambulance_import_status(self) -> Dict[str, Any]:
        """
        Get information about ambulance scenario import status for debugging.
        
        Returns:
            Dictionary with import status information
        """
        status = {
            "available": AMBULANCE_SCENARIOS_AVAILABLE,
            "scenario_count": len(self._ambulance_scenarios),
            "scenarios_loaded": list(self._ambulance_scenarios.keys())
        }
        
        if not AMBULANCE_SCENARIOS_AVAILABLE and '_AMBULANCE_IMPORT_ERROR' in globals():
            status["import_error"] = _AMBULANCE_IMPORT_ERROR
        
        return status