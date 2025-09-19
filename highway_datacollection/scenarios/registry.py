"""
Scenario registry for managing curriculum-based highway driving scenarios.
"""

from typing import Dict, Any, List
import copy
from .config import SCENARIO_CONFIGS, SCENARIO_CONSTRAINTS, DEFAULT_SCENARIO_PARAMS


class ScenarioRegistry:
    """
    Centralized registry for managing highway driving scenario configurations.
    
    Provides access to predefined curriculum scenarios with validation and
    customization capabilities.
    """
    
    def __init__(self):
        self._scenarios = copy.deepcopy(SCENARIO_CONFIGS)
        self._constraints = SCENARIO_CONSTRAINTS
    
    def get_scenario_config(self, scenario_name: str) -> Dict[str, Any]:
        """
        Retrieve configuration for a specific scenario.
        
        Args:
            scenario_name: Name of the scenario to retrieve
            
        Returns:
            Dictionary containing scenario configuration parameters
            
        Raises:
            KeyError: If scenario_name is not found in registry
        """
        if scenario_name not in self._scenarios:
            available = list(self._scenarios.keys())
            raise KeyError(f"Scenario '{scenario_name}' not found. Available: {available}")
        
        return copy.deepcopy(self._scenarios[scenario_name])
    
    def list_scenarios(self) -> List[str]:
        """
        Get list of all available scenario names.
        
        Returns:
            List of scenario names in the registry
        """
        return list(self._scenarios.keys())
    
    def validate_scenario(self, config: Dict[str, Any]) -> bool:
        """
        Validate scenario configuration against constraints.
        
        Args:
            config: Scenario configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            for param, constraints in self._constraints.items():
                if param in config:
                    value = config[param]
                    if "min" in constraints and value < constraints["min"]:
                        return False
                    if "max" in constraints and value > constraints["max"]:
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
    
    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters used across all scenarios.
        
        Returns:
            Dictionary of default scenario parameters
        """
        return copy.deepcopy(DEFAULT_SCENARIO_PARAMS)