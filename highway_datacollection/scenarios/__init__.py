"""
Scenario management module for curriculum-based highway driving scenarios.
"""

from .registry import ScenarioRegistry
from .config import SCENARIO_CONFIGS, DEFAULT_SCENARIO_PARAMS

__all__ = ["ScenarioRegistry", "SCENARIO_CONFIGS", "DEFAULT_SCENARIO_PARAMS"]