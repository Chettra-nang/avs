"""
Comprehensive test cases for ambulance scenario validation and testing.

This module tests each of the 15 ambulance scenarios individually for proper
execution, ambulance ego vehicle configuration, and multi-agent setup.
"""

import unittest
import sys
import os
import logging
from typing import Dict, Any, List

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from collecting_ambulance_data.validation import AmbulanceScenarioValidator, run_ambulance_scenario_validation
from collecting_ambulance_data.scenarios.ambulance_scenarios import (
    get_all_ambulance_scenarios, 
    get_scenario_names, 
    validate_ambulance_scenario,
    get_supported_observation_types
)


class TestAmbulanceScenarioValidation(unittest.TestCase):
    """Test cases for ambulance scenario validation and testing."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the entire test class."""
        # Set up logging
        logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
        
        # Initialize validator
        cls.validator = AmbulanceScenarioValidator(n_agents=4)
        
        # Get scenario information
        cls.all_scenarios = get_all_ambulance_scenarios()
        cls.scenario_names = get_scenario_names()
        cls.supported_obs_types = get_supported_observation_types()
        
        print(f"\nTesting {len(cls.scenario_names)} ambulance scenarios:")
        for i, name in enumerate(cls.scenario_names, 1):
            print(f"  {i:2d}. {name}")
        print()
    
    def test_scenario_count(self):
        """Test that we have exactly 15 ambulance scenarios."""
        self.assertEqual(len(self.scenario_names), 15, 
                        f"Expected 15 ambulance scenarios, found {len(self.scenario_names)}")
        self.assertEqual(len(self.all_scenarios), 15,
                        f"Expected 15 scenario configurations, found {len(self.all_scenarios)}")
    
    def test_supported_observation_types(self):
        """Test that all required observation types are supported."""
        expected_obs_types = ["Kinematics", "OccupancyGrid", "GrayscaleObservation"]
        self.assertEqual(set(self.supported_obs_types), set(expected_obs_types),
                        f"Expected observation types {expected_obs_types}, got {self.supported_obs_types}")
    
    def test_all_scenario_configurations(self):
        """Test that all 15 ambulance scenario configurations are valid."""
        print("\nTesting scenario configurations...")
        
        failed_scenarios = []
        
        for scenario_name in self.scenario_names:
            with self.subTest(scenario=scenario_name):
                # Test configuration validation
                config_result = self.validator.validate_scenario_configuration(scenario_name)
                
                if not config_result["valid"]:
                    failed_scenarios.append((scenario_name, config_result["errors"]))
                    print(f"  ❌ {scenario_name}: {', '.join(config_result['errors'])}")
                else:
                    print(f"  ✅ {scenario_name}: {config_result['tests_passed']} tests passed")
                
                # Assert that configuration is valid
                self.assertTrue(config_result["valid"], 
                              f"Scenario {scenario_name} configuration invalid: {config_result['errors']}")
                
                # Check that we have ambulance-specific configuration
                self.assertIn("_ambulance_config", self.all_scenarios[scenario_name],
                            f"Scenario {scenario_name} missing ambulance configuration")
                
                # Verify ambulance agent index is 0
                ambulance_config = self.all_scenarios[scenario_name]["_ambulance_config"]
                self.assertEqual(ambulance_config.get("ambulance_agent_index"), 0,
                               f"Scenario {scenario_name} ambulance agent index should be 0")
                
                # Verify controlled vehicles count is 4
                self.assertEqual(self.all_scenarios[scenario_name].get("controlled_vehicles"), 4,
                               f"Scenario {scenario_name} should have 4 controlled vehicles")
        
        if failed_scenarios:
            self.fail(f"Configuration validation failed for scenarios: {[name for name, _ in failed_scenarios]}")
    
    def test_individual_scenario_highway_emergency_light(self):
        """Test highway_emergency_light scenario individually."""
        scenario_name = "highway_emergency_light"
        self._test_individual_scenario(scenario_name)
    
    def test_individual_scenario_highway_emergency_moderate(self):
        """Test highway_emergency_moderate scenario individually."""
        scenario_name = "highway_emergency_moderate"
        self._test_individual_scenario(scenario_name)
    
    def test_individual_scenario_highway_emergency_dense(self):
        """Test highway_emergency_dense scenario individually."""
        scenario_name = "highway_emergency_dense"
        self._test_individual_scenario(scenario_name)
    
    def test_individual_scenario_highway_lane_closure(self):
        """Test highway_lane_closure scenario individually."""
        scenario_name = "highway_lane_closure"
        self._test_individual_scenario(scenario_name)
    
    def test_individual_scenario_highway_rush_hour(self):
        """Test highway_rush_hour scenario individually."""
        scenario_name = "highway_rush_hour"
        self._test_individual_scenario(scenario_name)
    
    def test_individual_scenario_highway_accident_scene(self):
        """Test highway_accident_scene scenario individually."""
        scenario_name = "highway_accident_scene"
        self._test_individual_scenario(scenario_name)
    
    def test_individual_scenario_highway_construction(self):
        """Test highway_construction scenario individually."""
        scenario_name = "highway_construction"
        self._test_individual_scenario(scenario_name)
    
    def test_individual_scenario_highway_weather_conditions(self):
        """Test highway_weather_conditions scenario individually."""
        scenario_name = "highway_weather_conditions"
        self._test_individual_scenario(scenario_name)
    
    def test_individual_scenario_highway_stop_and_go(self):
        """Test highway_stop_and_go scenario individually."""
        scenario_name = "highway_stop_and_go"
        self._test_individual_scenario(scenario_name)
    
    def test_individual_scenario_highway_aggressive_drivers(self):
        """Test highway_aggressive_drivers scenario individually."""
        scenario_name = "highway_aggressive_drivers"
        self._test_individual_scenario(scenario_name)
    
    def test_individual_scenario_highway_merge_heavy(self):
        """Test highway_merge_heavy scenario individually."""
        scenario_name = "highway_merge_heavy"
        self._test_individual_scenario(scenario_name)
    
    def test_individual_scenario_highway_speed_variation(self):
        """Test highway_speed_variation scenario individually."""
        scenario_name = "highway_speed_variation"
        self._test_individual_scenario(scenario_name)
    
    def test_individual_scenario_highway_shoulder_use(self):
        """Test highway_shoulder_use scenario individually."""
        scenario_name = "highway_shoulder_use"
        self._test_individual_scenario(scenario_name)
    
    def test_individual_scenario_highway_truck_heavy(self):
        """Test highway_truck_heavy scenario individually."""
        scenario_name = "highway_truck_heavy"
        self._test_individual_scenario(scenario_name)
    
    def test_individual_scenario_highway_time_pressure(self):
        """Test highway_time_pressure scenario individually."""
        scenario_name = "highway_time_pressure"
        self._test_individual_scenario(scenario_name)
    
    def _test_individual_scenario(self, scenario_name: str):
        """
        Helper method to test an individual scenario.
        
        Args:
            scenario_name: Name of the scenario to test
        """
        print(f"\nTesting individual scenario: {scenario_name}")
        
        # Validate scenario exists
        self.assertIn(scenario_name, self.scenario_names,
                     f"Scenario {scenario_name} not found in available scenarios")
        
        # Test configuration
        config_result = self.validator.validate_scenario_configuration(scenario_name)
        self.assertTrue(config_result["valid"],
                       f"Configuration validation failed for {scenario_name}: {config_result['errors']}")
        print(f"  ✅ Configuration: {config_result['tests_passed']} tests passed")
        
        # Test environment creation for each observation type
        env_results = {}
        for obs_type in self.supported_obs_types:
            env_result = self.validator.validate_environment_creation(scenario_name, obs_type)
            env_results[obs_type] = env_result
            
            self.assertTrue(env_result["valid"],
                           f"Environment creation failed for {scenario_name}/{obs_type}: {env_result['errors']}")
            print(f"  ✅ Environment ({obs_type}): {env_result['tests_passed']} tests passed")
        
        # Test multi-agent behavior for each observation type
        behavior_results = {}
        for obs_type in self.supported_obs_types:
            behavior_result = self.validator.validate_multi_agent_behavior(scenario_name, obs_type, test_steps=5)
            behavior_results[obs_type] = behavior_result
            
            self.assertTrue(behavior_result["valid"],
                           f"Multi-agent behavior validation failed for {scenario_name}/{obs_type}: {behavior_result['errors']}")
            print(f"  ✅ Behavior ({obs_type}): {behavior_result['tests_passed']} tests passed, "
                  f"{behavior_result['info'].get('steps_completed', 0)} steps completed")
        
        # Verify ambulance-specific properties
        scenario_config = self.all_scenarios[scenario_name]
        
        # Check ambulance configuration
        ambulance_config = scenario_config.get("_ambulance_config", {})
        self.assertEqual(ambulance_config.get("ambulance_agent_index"), 0,
                        f"Ambulance agent index should be 0 for {scenario_name}")
        self.assertIn(ambulance_config.get("emergency_priority"), ["high", "medium", "low"],
                     f"Invalid emergency priority for {scenario_name}")
        
        # Check multi-agent setup (4 controlled vehicles)
        self.assertEqual(scenario_config.get("controlled_vehicles"), 4,
                        f"Should have 4 controlled vehicles for {scenario_name}")
        
        # Check horizontal orientation
        screen_width = scenario_config.get("screen_width", 0)
        screen_height = scenario_config.get("screen_height", 0)
        self.assertGreater(screen_width, screen_height,
                          f"Should use horizontal orientation for {scenario_name}")
        
        print(f"  ✅ {scenario_name} passed all validation tests")
    
    def test_environment_creation_all_scenarios(self):
        """Test environment creation for all scenarios and observation types."""
        print("\nTesting environment creation for all scenarios...")
        
        failed_combinations = []
        
        for scenario_name in self.scenario_names:
            for obs_type in self.supported_obs_types:
                with self.subTest(scenario=scenario_name, obs_type=obs_type):
                    try:
                        env_result = self.validator.validate_environment_creation(scenario_name, obs_type)
                        
                        if not env_result["valid"]:
                            failed_combinations.append((scenario_name, obs_type, env_result["errors"]))
                            print(f"  ❌ {scenario_name}/{obs_type}: {', '.join(env_result['errors'])}")
                        else:
                            print(f"  ✅ {scenario_name}/{obs_type}: {env_result['tests_passed']} tests passed")
                        
                        self.assertTrue(env_result["valid"],
                                      f"Environment creation failed for {scenario_name}/{obs_type}: {env_result['errors']}")
                        
                    except Exception as e:
                        failed_combinations.append((scenario_name, obs_type, [str(e)]))
                        self.fail(f"Environment creation test failed for {scenario_name}/{obs_type}: {e}")
        
        if failed_combinations:
            self.fail(f"Environment creation failed for {len(failed_combinations)} combinations")
    
    def test_multi_agent_behavior_all_scenarios(self):
        """Test multi-agent behavior for all scenarios and observation types."""
        print("\nTesting multi-agent behavior for all scenarios...")
        
        failed_combinations = []
        
        for scenario_name in self.scenario_names:
            for obs_type in self.supported_obs_types:
                with self.subTest(scenario=scenario_name, obs_type=obs_type):
                    try:
                        behavior_result = self.validator.validate_multi_agent_behavior(
                            scenario_name, obs_type, test_steps=3  # Reduced steps for faster testing
                        )
                        
                        if not behavior_result["valid"]:
                            failed_combinations.append((scenario_name, obs_type, behavior_result["errors"]))
                            print(f"  ❌ {scenario_name}/{obs_type}: {', '.join(behavior_result['errors'])}")
                        else:
                            steps_completed = behavior_result["info"].get("steps_completed", 0)
                            print(f"  ✅ {scenario_name}/{obs_type}: {behavior_result['tests_passed']} tests passed, "
                                  f"{steps_completed} steps completed")
                        
                        self.assertTrue(behavior_result["valid"],
                                      f"Multi-agent behavior validation failed for {scenario_name}/{obs_type}: {behavior_result['errors']}")
                        
                        # Verify that we have ambulance-specific information
                        if behavior_result["valid"] and "info" in behavior_result:
                            info = behavior_result["info"]
                            self.assertIn("ambulance_total_reward", info,
                                        f"Missing ambulance reward information for {scenario_name}/{obs_type}")
                            self.assertIn("total_rewards", info,
                                        f"Missing total rewards information for {scenario_name}/{obs_type}")
                            
                            # Check that we have rewards for 4 agents
                            total_rewards = info.get("total_rewards", [])
                            self.assertEqual(len(total_rewards), 4,
                                          f"Expected 4 agent rewards for {scenario_name}/{obs_type}, got {len(total_rewards)}")
                        
                    except Exception as e:
                        failed_combinations.append((scenario_name, obs_type, [str(e)]))
                        self.fail(f"Multi-agent behavior test failed for {scenario_name}/{obs_type}: {e}")
        
        if failed_combinations:
            self.fail(f"Multi-agent behavior validation failed for {len(failed_combinations)} combinations")
    
    def test_ambulance_ego_vehicle_configuration(self):
        """Test that ambulance ego vehicle is correctly configured in each scenario."""
        print("\nTesting ambulance ego vehicle configuration...")
        
        for scenario_name in self.scenario_names:
            with self.subTest(scenario=scenario_name):
                scenario_config = self.all_scenarios[scenario_name]
                
                # Check ambulance configuration exists
                self.assertIn("_ambulance_config", scenario_config,
                            f"Missing ambulance configuration for {scenario_name}")
                
                ambulance_config = scenario_config["_ambulance_config"]
                
                # Check ambulance agent index is 0 (first agent)
                self.assertEqual(ambulance_config.get("ambulance_agent_index"), 0,
                               f"Ambulance agent index should be 0 for {scenario_name}")
                
                # Check emergency priority is valid
                emergency_priority = ambulance_config.get("emergency_priority")
                self.assertIn(emergency_priority, ["high", "medium", "low"],
                            f"Invalid emergency priority '{emergency_priority}' for {scenario_name}")
                
                # Check ambulance behavior is specified
                ambulance_behavior = ambulance_config.get("ambulance_behavior")
                self.assertIsNotNone(ambulance_behavior,
                                   f"Missing ambulance behavior for {scenario_name}")
                
                print(f"  ✅ {scenario_name}: ambulance_agent_index={ambulance_config.get('ambulance_agent_index')}, "
                      f"priority={emergency_priority}, behavior={ambulance_behavior}")
    
    def test_multi_agent_setup_configuration(self):
        """Test that multi-agent setup works with ambulance and normal vehicles."""
        print("\nTesting multi-agent setup configuration...")
        
        for scenario_name in self.scenario_names:
            with self.subTest(scenario=scenario_name):
                scenario_config = self.all_scenarios[scenario_name]
                
                # Check controlled vehicles count
                controlled_vehicles = scenario_config.get("controlled_vehicles")
                self.assertEqual(controlled_vehicles, 4,
                               f"Expected 4 controlled vehicles for {scenario_name}, got {controlled_vehicles}")
                
                # Check lanes count (should be 4 for most scenarios, 3 for lane closure)
                lanes_count = scenario_config.get("lanes_count")
                if scenario_name == "highway_lane_closure":
                    self.assertEqual(lanes_count, 3,
                                   f"Expected 3 lanes for lane closure scenario, got {lanes_count}")
                else:
                    self.assertEqual(lanes_count, 4,
                                   f"Expected 4 lanes for {scenario_name}, got {lanes_count}")
                
                # Check horizontal orientation
                screen_width = scenario_config.get("screen_width", 0)
                screen_height = scenario_config.get("screen_height", 0)
                self.assertGreater(screen_width, screen_height,
                                 f"Expected horizontal orientation for {scenario_name}")
                
                # Check other vehicles type is specified
                other_vehicles_type = scenario_config.get("other_vehicles_type")
                self.assertIsNotNone(other_vehicles_type,
                                   f"Missing other vehicles type for {scenario_name}")
                
                print(f"  ✅ {scenario_name}: {controlled_vehicles} controlled vehicles, "
                      f"{lanes_count} lanes, {screen_width}x{screen_height} screen")
    
    def test_scenario_diversity(self):
        """Test that scenarios have diverse traffic densities and conditions."""
        print("\nTesting scenario diversity...")
        
        traffic_densities = set()
        highway_conditions = set()
        emergency_priorities = set()
        
        for scenario_name in self.scenario_names:
            scenario_config = self.all_scenarios[scenario_name]
            
            # Collect traffic densities
            traffic_density = scenario_config.get("traffic_density")
            if traffic_density:
                traffic_densities.add(traffic_density)
            
            # Collect highway conditions
            highway_condition = scenario_config.get("highway_conditions")
            if highway_condition:
                highway_conditions.add(highway_condition)
            
            # Collect emergency priorities
            ambulance_config = scenario_config.get("_ambulance_config", {})
            emergency_priority = ambulance_config.get("emergency_priority")
            if emergency_priority:
                emergency_priorities.add(emergency_priority)
        
        # Check diversity
        self.assertGreaterEqual(len(traffic_densities), 2,
                               f"Expected diverse traffic densities, found: {traffic_densities}")
        self.assertGreaterEqual(len(highway_conditions), 3,
                               f"Expected diverse highway conditions, found: {highway_conditions}")
        self.assertGreaterEqual(len(emergency_priorities), 1,
                               f"Expected emergency priorities, found: {emergency_priorities}")
        
        print(f"  ✅ Traffic densities: {sorted(traffic_densities)}")
        print(f"  ✅ Highway conditions: {sorted(highway_conditions)}")
        print(f"  ✅ Emergency priorities: {sorted(emergency_priorities)}")
    
    def test_comprehensive_validation_run(self):
        """Test running comprehensive validation on a subset of scenarios."""
        print("\nTesting comprehensive validation run...")
        
        # Test with first 3 scenarios for speed
        test_scenarios = self.scenario_names[:3]
        
        results = run_ambulance_scenario_validation(
            scenarios=test_scenarios,
            test_steps=3,  # Reduced steps for faster testing
            n_agents=4
        )
        
        # Check results structure
        self.assertIn("scenarios", results)
        self.assertIn("summary", results)
        
        # Check that all test scenarios were processed
        for scenario_name in test_scenarios:
            self.assertIn(scenario_name, results["scenarios"],
                         f"Missing results for scenario {scenario_name}")
            
            scenario_result = results["scenarios"][scenario_name]
            self.assertIn("overall_valid", scenario_result,
                         f"Missing overall_valid for scenario {scenario_name}")
        
        # Check summary statistics
        summary = results["summary"]
        self.assertIn("scenarios_passed", summary)
        self.assertIn("scenarios_failed", summary)
        
        total_scenarios = summary["scenarios_passed"] + summary["scenarios_failed"]
        self.assertEqual(total_scenarios, len(test_scenarios),
                        f"Expected {len(test_scenarios)} total scenarios, got {total_scenarios}")
        
        print(f"  ✅ Comprehensive validation completed: "
              f"{summary['scenarios_passed']}/{total_scenarios} scenarios passed")


class TestAmbulanceScenarioValidationIntegration(unittest.TestCase):
    """Integration tests for ambulance scenario validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = AmbulanceScenarioValidator(n_agents=4)
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        self.assertEqual(self.validator.n_agents, 4)
        self.assertIsNotNone(self.validator.env_factory)
        self.assertIsNotNone(self.validator.action_sampler)
        self.assertEqual(len(self.validator.scenario_names), 15)
        self.assertEqual(len(self.validator.supported_obs_types), 3)
    
    def test_validation_statistics(self):
        """Test validation statistics tracking."""
        initial_stats = self.validator.get_validation_statistics()
        
        # Check initial statistics structure
        expected_keys = [
            "scenarios_validated", "scenarios_passed", "scenarios_failed",
            "configuration_errors", "environment_errors", "execution_errors",
            "total_tests_run", "total_tests_passed"
        ]
        
        for key in expected_keys:
            self.assertIn(key, initial_stats)
            self.assertEqual(initial_stats[key], 0)
        
        # Test statistics reset
        self.validator.reset_statistics()
        reset_stats = self.validator.get_validation_statistics()
        self.assertEqual(initial_stats, reset_stats)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)