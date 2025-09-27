"""
Ambulance Data Collection System

This package extends the existing highway data collection infrastructure to support
ambulance ego vehicles with specialized scenarios. It maintains full compatibility
with current multi-modal data collection while introducing ambulance-specific
vehicle configurations and emergency response scenarios.
"""

__version__ = "1.0.0"
__author__ = "Highway Data Collection Team"

# Import scenarios module (available components only)
from .scenarios import (
    get_all_ambulance_scenarios,
    get_scenario_names,
    get_scenario_by_name,
    validate_ambulance_scenario
)

# Import collection module
from .collection import AmbulanceDataCollector

__all__ = [
    "get_all_ambulance_scenarios",
    "get_scenario_names", 
    "get_scenario_by_name",
    "validate_ambulance_scenario",
    "AmbulanceDataCollector"
]