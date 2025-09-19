"""
Unit tests for scenario registry and configuration management.
"""

import pytest
import copy
from highway_datacollection.scenarios.registry import ScenarioRegistry
from highway_datacollection.scenarios.config import SCENARIO_CONFIGS, DEFAULT_SCENARIO_PARAMS


class TestScenarioRegistry:
    """Test suite for ScenarioRegistry class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ScenarioRegistry()
    
    def test_initialization(self):
        """Test registry initialization."""
        assert isinstance(self.registry._scenarios, dict)
        assert isinstance(self.registry._constraints, dict)
        assert len(self.registry._scenarios) == 6  # Six curriculum scenarios
    
    def test_list_scenarios(self):
        """Test listing all available scenarios."""
        scenarios = self.registry.list_scenarios()
        expected_scenarios = [
            "free_flow", "dense_commuting", "stop_and_go", 
            "aggressive_neighbors", "lane_closure", "time_budget"
        ]
        
        assert len(scenarios) == 6
        for scenario in expected_scenarios:
            assert scenario in scenarios
    
    def test_get_scenario_config_valid(self):
        """Test retrieving valid scenario configurations."""
        # Test each curriculum scenario
        for scenario_name in ["free_flow", "dense_commuting", "stop_and_go", 
                             "aggressive_neighbors", "lane_closure", "time_budget"]:
            config = self.registry.get_scenario_config(scenario_name)
            
            # Verify required parameters are present
            assert "vehicles_count" in config
            assert "lanes_count" in config
            assert "duration" in config
            assert "description" in config
            
            # Verify it's a deep copy (modifications don't affect original)
            original_count = config["vehicles_count"]
            config["vehicles_count"] = 999
            new_config = self.registry.get_scenario_config(scenario_name)
            assert new_config["vehicles_count"] == original_count
    
    def test_get_scenario_config_invalid(self):
        """Test retrieving invalid scenario configurations."""
        with pytest.raises(KeyError) as exc_info:
            self.registry.get_scenario_config("nonexistent_scenario")
        
        assert "not found" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)
    
    def test_validate_scenario_valid_configs(self):
        """Test validation of valid scenario configurations."""
        # Test all predefined scenarios are valid
        for scenario_name in self.registry.list_scenarios():
            config = self.registry.get_scenario_config(scenario_name)
            assert self.registry.validate_scenario(config) is True
        
        # Test custom valid configuration
        valid_config = {
            "vehicles_count": 50,
            "lanes_count": 4,
            "duration": 30,
            "controlled_vehicles": 3
        }
        assert self.registry.validate_scenario(valid_config) is True
    
    def test_validate_scenario_invalid_configs(self):
        """Test validation of invalid scenario configurations."""
        # Test vehicles_count constraints
        invalid_config = {"vehicles_count": 5}  # Below minimum of 10
        assert self.registry.validate_scenario(invalid_config) is False
        
        invalid_config = {"vehicles_count": 150}  # Above maximum of 100
        assert self.registry.validate_scenario(invalid_config) is False
        
        # Test lanes_count constraints
        invalid_config = {"lanes_count": 1}  # Below minimum of 2
        assert self.registry.validate_scenario(invalid_config) is False
        
        invalid_config = {"lanes_count": 10}  # Above maximum of 6
        assert self.registry.validate_scenario(invalid_config) is False
        
        # Test duration constraints
        invalid_config = {"duration": 5}  # Below minimum of 10
        assert self.registry.validate_scenario(invalid_config) is False
        
        invalid_config = {"duration": 200}  # Above maximum of 120
        assert self.registry.validate_scenario(invalid_config) is False
        
        # Test invalid data types
        invalid_config = {"vehicles_count": "not_a_number"}
        assert self.registry.validate_scenario(invalid_config) is False
    
    def test_customize_scenario_valid(self):
        """Test customizing scenarios with valid parameters."""
        # Customize free_flow scenario
        custom_config = self.registry.customize_scenario(
            "free_flow",
            vehicles_count=30,
            duration=25
        )
        
        assert custom_config["vehicles_count"] == 30
        assert custom_config["duration"] == 25
        assert custom_config["lanes_count"] == 4  # Original value preserved
        assert "description" in custom_config  # Original fields preserved
        
        # Verify original scenario unchanged
        original = self.registry.get_scenario_config("free_flow")
        assert original["vehicles_count"] == 20  # Original value
    
    def test_customize_scenario_invalid_base(self):
        """Test customizing with invalid base scenario."""
        with pytest.raises(KeyError):
            self.registry.customize_scenario("nonexistent", vehicles_count=30)
    
    def test_customize_scenario_invalid_params(self):
        """Test customizing with invalid parameters."""
        with pytest.raises(ValueError) as exc_info:
            self.registry.customize_scenario(
                "free_flow",
                vehicles_count=5  # Below minimum constraint
            )
        
        assert "invalid" in str(exc_info.value).lower()
    
    def test_get_default_params(self):
        """Test retrieving default parameters."""
        defaults = self.registry.get_default_params()
        
        # Verify key default parameters
        assert defaults["vehicles_count"] == 50
        assert defaults["lanes_count"] == 4
        assert defaults["duration"] == 40
        assert defaults["controlled_vehicles"] == 2
        assert defaults["action"]["type"] == "DiscreteMetaAction"
        
        # Verify it's a deep copy
        defaults["vehicles_count"] = 999
        new_defaults = self.registry.get_default_params()
        assert new_defaults["vehicles_count"] == 50
    
    def test_scenario_specific_configurations(self):
        """Test specific configurations for each curriculum scenario."""
        # Test free_flow scenario
        free_flow = self.registry.get_scenario_config("free_flow")
        assert free_flow["vehicles_count"] == 20
        assert free_flow["duration"] == 30
        assert "free-flowing" in free_flow["description"]
        
        # Test dense_commuting scenario
        dense = self.registry.get_scenario_config("dense_commuting")
        assert dense["vehicles_count"] == 80
        assert dense["duration"] == 45
        assert "commuter" in dense["description"]
        
        # Test stop_and_go scenario
        stop_go = self.registry.get_scenario_config("stop_and_go")
        assert stop_go["vehicles_count"] == 60
        assert stop_go["lanes_count"] == 3
        assert "stop-and-go" in stop_go["description"]
        
        # Test aggressive_neighbors scenario
        aggressive = self.registry.get_scenario_config("aggressive_neighbors")
        assert "AggressiveVehicle" in aggressive["other_vehicles_type"]
        assert "aggressive" in aggressive["description"]
        
        # Test lane_closure scenario
        closure = self.registry.get_scenario_config("lane_closure")
        assert closure["lanes_count"] == 3
        assert "closure" in closure["description"]
        
        # Test time_budget scenario
        time_budget = self.registry.get_scenario_config("time_budget")
        assert time_budget["duration"] == 25  # Shorter duration
        assert time_budget["high_speed_reward"] == 0.6  # Higher speed reward
        assert "Time-pressured" in time_budget["description"]
    
    def test_customizable_parameters(self):
        """Test that all required customizable parameters are supported."""
        # Test vehicles_count customization
        for scenario in self.registry.list_scenarios():
            custom = self.registry.customize_scenario(scenario, vehicles_count=25)
            assert custom["vehicles_count"] == 25
        
        # Test lanes_count customization
        for scenario in self.registry.list_scenarios():
            custom = self.registry.customize_scenario(scenario, lanes_count=5)
            assert custom["lanes_count"] == 5
        
        # Test duration customization
        for scenario in self.registry.list_scenarios():
            custom = self.registry.customize_scenario(scenario, duration=60)
            assert custom["duration"] == 60
    
    def test_multi_agent_support(self):
        """Test that all scenarios support multi-agent configuration."""
        for scenario_name in self.registry.list_scenarios():
            config = self.registry.get_scenario_config(scenario_name)
            
            # Verify minimum controlled vehicles for multi-agent scenarios
            assert config["controlled_vehicles"] >= 2
            
            # Test customizing controlled vehicles
            custom = self.registry.customize_scenario(
                scenario_name, 
                controlled_vehicles=4
            )
            assert custom["controlled_vehicles"] == 4


class TestScenarioConfigIntegration:
    """Integration tests for scenario configuration constants."""
    
    def test_all_scenarios_have_required_fields(self):
        """Test that all predefined scenarios have required fields."""
        required_fields = [
            "vehicles_count", "lanes_count", "duration", 
            "controlled_vehicles", "description"
        ]
        
        for scenario_name, config in SCENARIO_CONFIGS.items():
            for field in required_fields:
                assert field in config, f"Scenario '{scenario_name}' missing field '{field}'"
    
    def test_default_params_completeness(self):
        """Test that default parameters include all necessary fields."""
        required_defaults = [
            "vehicles_count", "lanes_count", "duration", "controlled_vehicles",
            "action", "simulation_frequency", "policy_frequency"
        ]
        
        for field in required_defaults:
            assert field in DEFAULT_SCENARIO_PARAMS, f"Missing default parameter: {field}"
    
    def test_scenario_inheritance(self):
        """Test that scenarios properly inherit from default parameters."""
        for scenario_name, config in SCENARIO_CONFIGS.items():
            # Check that scenario has default parameters as base
            for key, value in DEFAULT_SCENARIO_PARAMS.items():
                if key not in config:
                    # Should inherit from defaults
                    continue
                # If overridden, should be intentional (different from default)
                if config[key] == value:
                    continue  # Same as default, which is fine