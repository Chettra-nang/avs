"""
Ambulance Scenarios Package

This package contains scenario configurations for ambulance data collection.
"""

from .ambulance_scenarios import (
    get_ambulance_scenarios,
    get_additional_ambulance_scenarios,
    get_all_ambulance_scenarios,
    validate_ambulance_scenario,
    get_scenario_names,
    get_scenario_by_name
)

__all__ = [
    'get_ambulance_scenarios',
    'get_additional_ambulance_scenarios',
    'get_all_ambulance_scenarios',
    'validate_ambulance_scenario',
    'get_scenario_names',
    'get_scenario_by_name'
]