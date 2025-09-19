"""
Test basic project structure and imports.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_main_package_import():
    """Test that main package can be imported."""
    import highway_datacollection
    assert highway_datacollection.__version__ == "0.1.0"


def test_scenario_registry_import():
    """Test that ScenarioRegistry can be imported and instantiated."""
    from highway_datacollection.scenarios import ScenarioRegistry
    
    registry = ScenarioRegistry()
    scenarios = registry.list_scenarios()
    
    # Check that all expected scenarios are present
    expected_scenarios = [
        "free_flow", "dense_commuting", "stop_and_go", 
        "aggressive_neighbors", "lane_closure", "time_budget"
    ]
    
    for scenario in expected_scenarios:
        assert scenario in scenarios


def test_scenario_config_retrieval():
    """Test that scenario configurations can be retrieved."""
    from highway_datacollection.scenarios import ScenarioRegistry
    
    registry = ScenarioRegistry()
    
    # Test getting a specific scenario
    config = registry.get_scenario_config("free_flow")
    assert isinstance(config, dict)
    assert "vehicles_count" in config
    assert "lanes_count" in config
    assert "duration" in config


def test_scenario_validation():
    """Test scenario configuration validation."""
    from highway_datacollection.scenarios import ScenarioRegistry
    
    registry = ScenarioRegistry()
    
    # Valid configuration
    valid_config = {
        "vehicles_count": 50,
        "lanes_count": 4,
        "duration": 30
    }
    assert registry.validate_scenario(valid_config) is True
    
    # Invalid configuration (too many vehicles)
    invalid_config = {
        "vehicles_count": 200,  # Exceeds max constraint
        "lanes_count": 4,
        "duration": 30
    }
    assert registry.validate_scenario(invalid_config) is False


def test_package_structure():
    """Test that all expected modules are present."""
    import highway_datacollection.scenarios
    import highway_datacollection.environments
    import highway_datacollection.collection
    import highway_datacollection.features
    import highway_datacollection.storage
    
    # Test that config constants are accessible
    from highway_datacollection.scenarios.config import SCENARIO_CONFIGS, DEFAULT_SCENARIO_PARAMS
    assert isinstance(SCENARIO_CONFIGS, dict)
    assert isinstance(DEFAULT_SCENARIO_PARAMS, dict)