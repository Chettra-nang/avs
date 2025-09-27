"""
Test cases for ambulance scenario registry integration.
"""

import unittest
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from highway_datacollection.scenarios.registry import ScenarioRegistry


class TestAmbulanceScenarioRegistry(unittest.TestCase):
    """Test cases for ambulance scenario registry functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ScenarioRegistry()
    
    def test_ambulance_scenarios_loaded(self):
        """Test that ambulance scenarios are loaded correctly."""
        ambulance_scenarios = self.registry.list_ambulance_scenarios()
        self.assertGreater(len(ambulance_scenarios), 0, "No ambulance scenarios loaded")
        self.assertEqual(len(ambulance_scenarios), 15, "Expected 15 ambulance scenarios")
    
    def test_scenario_listing(self):
        """Test scenario listing functionality."""
        all_scenarios = self.registry.list_scenarios()
        regular_scenarios = self.registry.list_regular_scenarios()
        ambulance_scenarios = self.registry.list_ambulance_scenarios()
        
        # Check that all scenarios include both regular and ambulance
        expected_total = len(regular_scenarios) + len(ambulance_scenarios)
        self.assertEqual(len(all_scenarios), expected_total)
        
        # Check that there are no duplicates
        self.assertEqual(len(set(all_scenarios)), len(all_scenarios))
    
    def test_ambulance_scenario_retrieval(self):
        """Test retrieving ambulance scenario configurations."""
        ambulance_scenarios = self.registry.list_ambulance_scenarios()
        self.assertGreater(len(ambulance_scenarios), 0)
        
        # Test retrieving first ambulance scenario
        scenario_name = ambulance_scenarios[0]
        config = self.registry.get_scenario_config(scenario_name)
        
        # Verify ambulance-specific properties
        self.assertIn('_ambulance_config', config)
        self.assertEqual(config.get('controlled_vehicles'), 4)
        ambulance_config = config.get('_ambulance_config', {})
        self.assertEqual(ambulance_config.get('ambulance_agent_index'), 0)
        self.assertIn('traffic_density', config)
        self.assertIn('_ambulance_config', config)
    
    def test_ambulance_scenario_validation(self):
        """Test ambulance scenario validation."""
        ambulance_scenarios = self.registry.list_ambulance_scenarios()
        self.assertGreater(len(ambulance_scenarios), 0)
        
        # Test valid ambulance scenario
        config = self.registry.get_ambulance_scenario_config(ambulance_scenarios[0])
        self.assertTrue(self.registry.validate_scenario(config))
        
        # Test invalid ambulance scenario (wrong controlled vehicles count)
        invalid_config = config.copy()
        invalid_config['controlled_vehicles'] = 2  # Invalid for ambulance scenarios
        self.assertFalse(self.registry.validate_scenario(invalid_config))
        
        # Test invalid ambulance scenario (wrong controlled vehicles count)
        invalid_config = config.copy()
        invalid_config['controlled_vehicles'] = 2
        self.assertFalse(self.registry.validate_scenario(invalid_config))
    
    def test_scenario_type_detection(self):
        """Test scenario type detection."""
        regular_scenarios = self.registry.list_regular_scenarios()
        ambulance_scenarios = self.registry.list_ambulance_scenarios()
        
        if regular_scenarios:
            self.assertFalse(self.registry.is_ambulance_scenario(regular_scenarios[0]))
        
        if ambulance_scenarios:
            self.assertTrue(self.registry.is_ambulance_scenario(ambulance_scenarios[0]))
    
    def test_traffic_density_filtering(self):
        """Test filtering ambulance scenarios by traffic density."""
        light_scenarios = self.registry.get_ambulance_scenarios_by_traffic_density("light")
        moderate_scenarios = self.registry.get_ambulance_scenarios_by_traffic_density("moderate")
        heavy_scenarios = self.registry.get_ambulance_scenarios_by_traffic_density("heavy")
        
        # Verify all scenarios are accounted for
        total_filtered = len(light_scenarios) + len(moderate_scenarios) + len(heavy_scenarios)
        total_ambulance = len(self.registry.list_ambulance_scenarios())
        self.assertEqual(total_filtered, total_ambulance)
        
        # Test invalid density
        with self.assertRaises(ValueError):
            self.registry.get_ambulance_scenarios_by_traffic_density("invalid")
    
    def test_highway_conditions_filtering(self):
        """Test filtering ambulance scenarios by highway conditions."""
        # Test some known conditions
        normal_scenarios = self.registry.get_ambulance_scenarios_by_conditions("normal")
        construction_scenarios = self.registry.get_ambulance_scenarios_by_conditions("construction")
        
        # Verify results are lists
        self.assertIsInstance(normal_scenarios, list)
        self.assertIsInstance(construction_scenarios, list)
    
    def test_ambulance_scenario_customization(self):
        """Test customizing ambulance scenarios."""
        ambulance_scenarios = self.registry.list_ambulance_scenarios()
        self.assertGreater(len(ambulance_scenarios), 0)
        
        base_scenario = ambulance_scenarios[0]
        original_config = self.registry.get_ambulance_scenario_config(base_scenario)
        
        # Test valid customization
        custom_config = self.registry.customize_ambulance_scenario(
            base_scenario,
            duration=60,
            vehicles_count=25
        )
        
        # Verify customization applied
        self.assertEqual(custom_config['duration'], 60)
        self.assertEqual(custom_config['vehicles_count'], 25)
        
        # Verify ambulance constraints maintained
        self.assertIn('_ambulance_config', custom_config)
        ambulance_config = custom_config.get('_ambulance_config', {})
        self.assertEqual(ambulance_config.get('ambulance_agent_index'), 0)
        self.assertEqual(custom_config['controlled_vehicles'], 4)
        
        # Test invalid customization (should raise ValueError)
        try:
            # Test with controlled_vehicles which should definitely fail
            self.registry.customize_ambulance_scenario(
                base_scenario,
                controlled_vehicles=10  # This should definitely be invalid
            )
            self.fail("Expected ValueError was not raised for controlled_vehicles=10")
        except ValueError:
            pass  # This is expected
    
    def test_scenarios_by_type(self):
        """Test getting scenarios by type."""
        regular_by_type = self.registry.get_scenarios_by_type("regular")
        ambulance_by_type = self.registry.get_scenarios_by_type("ambulance")
        
        self.assertEqual(regular_by_type, self.registry.list_regular_scenarios())
        self.assertEqual(ambulance_by_type, self.registry.list_ambulance_scenarios())
        
        # Test invalid type
        with self.assertRaises(ValueError):
            self.registry.get_scenarios_by_type("invalid")
    
    def test_ambulance_constraints(self):
        """Test ambulance-specific constraints."""
        constraints = self.registry.get_ambulance_constraints()
        
        # Verify key constraints exist
        self.assertIn('controlled_vehicles', constraints)
        self.assertIn('_ambulance_config', constraints)
        self.assertIn('traffic_density', constraints)
        self.assertIn('highway_conditions', constraints)
        # Note: observation and action constraints are handled dynamically by MultiAgentEnvFactory
        
        # Verify constraint values
        self.assertEqual(constraints['controlled_vehicles']['required'], 4)
        self.assertEqual(constraints['_ambulance_config']['required'], True)
    
    def test_import_status(self):
        """Test ambulance import status reporting."""
        status = self.registry.get_ambulance_import_status()
        
        self.assertIn('available', status)
        self.assertIn('scenario_count', status)
        self.assertIn('scenarios_loaded', status)
        
        # If available, should have scenarios
        if status['available']:
            self.assertGreater(status['scenario_count'], 0)
            self.assertGreater(len(status['scenarios_loaded']), 0)


if __name__ == '__main__':
    unittest.main()