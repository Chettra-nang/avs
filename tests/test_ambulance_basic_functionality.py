#!/usr/bin/env python3
"""
Basic functionality test for ambulance system.

This is a lightweight test that verifies core ambulance system functionality
without requiring extensive setup. It can be run quickly to verify the system works.

Requirements covered: 1.4, 2.4, 6.4
"""

import unittest
import sys
import os
import logging
from pathlib import Path

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)


class TestAmbulanceBasicFunctionality(unittest.TestCase):
    """Basic functionality tests for ambulance system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the entire test class."""
        # Reduce logging noise
        logging.basicConfig(level=logging.WARNING)
    
    def test_ambulance_scenarios_import(self):
        """Test that ambulance scenarios can be imported and loaded."""
        try:
            from collecting_ambulance_data.scenarios.ambulance_scenarios import (
                get_all_ambulance_scenarios, 
                get_scenario_names, 
                get_supported_observation_types
            )
            
            scenarios = get_all_ambulance_scenarios()
            scenario_names = get_scenario_names()
            obs_types = get_supported_observation_types()
            
            # Basic checks
            self.assertIsInstance(scenarios, dict)
            self.assertIsInstance(scenario_names, list)
            self.assertIsInstance(obs_types, list)
            
            self.assertEqual(len(scenario_names), 15)
            self.assertEqual(len(scenarios), 15)
            self.assertEqual(len(obs_types), 3)
            
            print(f"✓ Successfully loaded {len(scenario_names)} ambulance scenarios")
            print(f"✓ Supported observation types: {obs_types}")
            
        except ImportError as e:
            self.fail(f"Failed to import ambulance scenarios: {e}")
    
    def test_ambulance_collector_import(self):
        """Test that ambulance collector can be imported and initialized."""
        try:
            from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector
            
            collector = AmbulanceDataCollector(n_agents=4)
            
            # Basic checks
            self.assertEqual(collector.n_agents, 4)
            self.assertIsNotNone(collector._collector)
            self.assertIsNotNone(collector._env_factory)
            
            # Test basic methods
            scenarios = collector.get_available_scenarios()
            self.assertEqual(len(scenarios), 15)
            
            stats = collector.get_collection_statistics()
            self.assertIsInstance(stats, dict)
            self.assertIn('ambulance_episodes_collected', stats)
            
            collector.cleanup()
            
            print(f"✓ Successfully initialized AmbulanceDataCollector")
            print(f"✓ Available scenarios: {len(scenarios)}")
            
        except ImportError as e:
            self.fail(f"Failed to import ambulance collector: {e}")
        except Exception as e:
            self.fail(f"Failed to initialize ambulance collector: {e}")
    
    def test_environment_factory_ambulance_methods(self):
        """Test that environment factory has ambulance methods."""
        try:
            from highway_datacollection.environments.factory import MultiAgentEnvFactory
            
            factory = MultiAgentEnvFactory()
            
            # Check that ambulance methods exist
            self.assertTrue(hasattr(factory, 'create_ambulance_env'))
            self.assertTrue(hasattr(factory, 'get_ambulance_base_config'))
            self.assertTrue(hasattr(factory, 'create_parallel_ambulance_envs'))
            
            print("✓ Environment factory has ambulance methods")
            
        except ImportError as e:
            self.fail(f"Failed to import environment factory: {e}")
    
    def test_scenario_registry_ambulance_integration(self):
        """Test that scenario registry includes ambulance scenarios."""
        try:
            from highway_datacollection.scenarios.registry import ScenarioRegistry
            
            registry = ScenarioRegistry()
            
            # Check that ambulance methods exist
            self.assertTrue(hasattr(registry, 'list_ambulance_scenarios'))
            self.assertTrue(hasattr(registry, 'get_ambulance_scenario_config'))
            
            # Test basic functionality
            ambulance_scenarios = registry.list_ambulance_scenarios()
            self.assertIsInstance(ambulance_scenarios, list)
            self.assertEqual(len(ambulance_scenarios), 15)
            
            print(f"✓ Scenario registry includes {len(ambulance_scenarios)} ambulance scenarios")
            
        except ImportError as e:
            self.fail(f"Failed to import scenario registry: {e}")
        except Exception as e:
            self.fail(f"Scenario registry ambulance integration failed: {e}")
    
    def test_ambulance_validation_import(self):
        """Test that ambulance validation can be imported."""
        try:
            from collecting_ambulance_data.validation import AmbulanceScenarioValidator
            
            validator = AmbulanceScenarioValidator(n_agents=4)
            
            # Basic checks
            self.assertEqual(validator.n_agents, 4)
            self.assertEqual(len(validator.scenario_names), 15)
            self.assertEqual(len(validator.supported_obs_types), 3)
            
            print("✓ Ambulance validation system available")
            
        except ImportError as e:
            self.fail(f"Failed to import ambulance validation: {e}")
        except Exception as e:
            self.fail(f"Failed to initialize ambulance validator: {e}")
    
    def test_basic_scenario_configuration_validation(self):
        """Test basic scenario configuration validation."""
        try:
            from collecting_ambulance_data.scenarios.ambulance_scenarios import (
                get_all_ambulance_scenarios,
                validate_ambulance_scenario
            )
            
            scenarios = get_all_ambulance_scenarios()
            
            # Test validation of first scenario
            first_scenario_name = list(scenarios.keys())[0]
            first_scenario_config = scenarios[first_scenario_name]
            
            is_valid = validate_ambulance_scenario(first_scenario_config)
            self.assertTrue(is_valid, f"First scenario {first_scenario_name} should be valid")
            
            # Check required fields
            required_fields = ['scenario_name', 'controlled_vehicles', '_ambulance_config']
            for field in required_fields:
                self.assertIn(field, first_scenario_config, f"Missing field: {field}")
            
            # Check ambulance-specific configuration
            ambulance_config = first_scenario_config['_ambulance_config']
            self.assertEqual(ambulance_config.get('ambulance_agent_index'), 0)
            self.assertEqual(first_scenario_config.get('controlled_vehicles'), 4)
            
            print(f"✓ Basic scenario configuration validation passed")
            
        except Exception as e:
            self.fail(f"Basic scenario validation failed: {e}")
    
    def test_environment_creation_basic(self):
        """Test basic environment creation (without full execution)."""
        try:
            from highway_datacollection.environments.factory import MultiAgentEnvFactory
            from collecting_ambulance_data.scenarios.ambulance_scenarios import get_scenario_names
            
            factory = MultiAgentEnvFactory()
            scenario_names = get_scenario_names()
            
            # Test configuration generation
            test_scenario = scenario_names[0]
            config = factory.get_ambulance_base_config(test_scenario, 4)
            
            # Basic configuration checks
            self.assertIsInstance(config, dict)
            self.assertIn('_ambulance_config', config)
            self.assertEqual(config.get('controlled_vehicles'), 4)
            
            ambulance_config = config['_ambulance_config']
            self.assertEqual(ambulance_config.get('ambulance_agent_index'), 0)
            
            print(f"✓ Basic environment configuration generation works")
            
        except Exception as e:
            self.fail(f"Basic environment creation test failed: {e}")
    
    def test_system_integration_imports(self):
        """Test that all system components can be imported together."""
        try:
            # Import all major components
            from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector
            from collecting_ambulance_data.scenarios.ambulance_scenarios import get_all_ambulance_scenarios
            from collecting_ambulance_data.validation import AmbulanceScenarioValidator
            from highway_datacollection.environments.factory import MultiAgentEnvFactory
            from highway_datacollection.scenarios.registry import ScenarioRegistry
            
            # Basic integration test
            scenarios = get_all_ambulance_scenarios()
            collector = AmbulanceDataCollector(n_agents=4)
            validator = AmbulanceScenarioValidator(n_agents=4)
            factory = MultiAgentEnvFactory()
            registry = ScenarioRegistry()
            
            # Verify they work together
            collector_scenarios = collector.get_available_scenarios()
            validator_scenarios = validator.scenario_names
            registry_scenarios = registry.list_ambulance_scenarios()
            
            self.assertEqual(len(collector_scenarios), 15)
            self.assertEqual(len(validator_scenarios), 15)
            self.assertEqual(len(registry_scenarios), 15)
            
            collector.cleanup()
            
            print("✓ All system components integrate successfully")
            
        except Exception as e:
            self.fail(f"System integration test failed: {e}")


def run_basic_tests():
    """Run basic functionality tests with simple output."""
    print("🚑 AMBULANCE SYSTEM BASIC FUNCTIONALITY TEST")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAmbulanceBasicFunctionality)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    print("BASIC TEST SUMMARY")
    print("="*60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {'✓ PASS' if result.wasSuccessful() else '✗ FAIL'}")
    
    if result.wasSuccessful():
        print("\n🎉 BASIC FUNCTIONALITY TESTS PASSED!")
        print("The ambulance system core functionality is working.")
        print("\nTo run comprehensive tests, use:")
        print("  python tests/run_ambulance_tests.py")
    else:
        print("\n❌ SOME BASIC TESTS FAILED")
        print("Please check the errors above.")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split(chr(10))[0]}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split(chr(10))[0]}")
    
    print("="*60)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    # Set up environment
    os.environ.setdefault('PYTHONPATH', str(Path(__file__).parent.parent))
    
    # Run basic tests
    exit_code = run_basic_tests()
    sys.exit(exit_code)