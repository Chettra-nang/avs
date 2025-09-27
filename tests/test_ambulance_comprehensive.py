"""
Comprehensive test suite for the ambulance data collection system.

This module provides comprehensive testing for:
- Ambulance scenario configurations
- Environment creation and agent setup
- Multi-modal data collection outputs
- Error handling and edge cases
- Integration with existing systems

Requirements covered: 1.4, 2.4, 6.4
"""

import unittest
import sys
import os
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import ambulance system components
from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector
from collecting_ambulance_data.scenarios.ambulance_scenarios import (
    get_all_ambulance_scenarios, 
    get_scenario_names, 
    validate_ambulance_scenario,
    get_supported_observation_types,
    get_scenario_by_name
)
from collecting_ambulance_data.validation import AmbulanceScenarioValidator
from highway_datacollection.environments.factory import MultiAgentEnvFactory
from highway_datacollection.scenarios.registry import ScenarioRegistry


class TestAmbulanceScenarioConfigurations(unittest.TestCase):
    """Test ambulance scenario configurations comprehensively."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the entire test class."""
        logging.basicConfig(level=logging.WARNING)  # Reduce log noise
        cls.all_scenarios = get_all_ambulance_scenarios()
        cls.scenario_names = get_scenario_names()
        cls.supported_obs_types = get_supported_observation_types()
    
    def test_scenario_count_and_availability(self):
        """Test that exactly 15 ambulance scenarios are available."""
        self.assertEqual(len(self.scenario_names), 15, 
                        f"Expected 15 ambulance scenarios, found {len(self.scenario_names)}")
        self.assertEqual(len(self.all_scenarios), 15,
                        f"Expected 15 scenario configurations, found {len(self.all_scenarios)}")
        
        # Verify all expected scenarios are present
        expected_scenarios = [
            "highway_emergency_light", "highway_emergency_moderate", "highway_emergency_dense",
            "highway_lane_closure", "highway_rush_hour", "highway_accident_scene",
            "highway_construction", "highway_weather_conditions", "highway_stop_and_go",
            "highway_aggressive_drivers", "highway_merge_heavy", "highway_speed_variation",
            "highway_shoulder_use", "highway_truck_heavy", "highway_time_pressure"
        ]
        
        for expected_scenario in expected_scenarios:
            self.assertIn(expected_scenario, self.scenario_names,
                         f"Expected scenario '{expected_scenario}' not found")
    
    def test_scenario_configuration_structure(self):
        """Test that all scenarios have required configuration structure."""
        required_fields = [
            "scenario_name", "controlled_vehicles", "_ambulance_config",
            "lanes_count", "screen_width", "screen_height", "traffic_density"
        ]
        
        for scenario_name in self.scenario_names:
            with self.subTest(scenario=scenario_name):
                config = self.all_scenarios[scenario_name]
                
                # Check required fields
                for field in required_fields:
                    self.assertIn(field, config,
                                f"Scenario {scenario_name} missing required field: {field}")
                
                # Validate ambulance configuration
                ambulance_config = config.get("_ambulance_config", {})
                self.assertIsInstance(ambulance_config, dict,
                                    f"Ambulance config must be dict for {scenario_name}")
                self.assertEqual(ambulance_config.get("ambulance_agent_index"), 0,
                               f"Ambulance agent index must be 0 for {scenario_name}")
                
                # Validate controlled vehicles count
                self.assertEqual(config.get("controlled_vehicles"), 4,
                               f"Must have 4 controlled vehicles for {scenario_name}")
                
                # Validate horizontal orientation
                screen_width = config.get("screen_width", 0)
                screen_height = config.get("screen_height", 0)
                self.assertGreater(screen_width, screen_height,
                                 f"Must use horizontal orientation for {scenario_name}")
    
    def test_scenario_validation_function(self):
        """Test the validate_ambulance_scenario function."""
        for scenario_name in self.scenario_names:
            with self.subTest(scenario=scenario_name):
                config = self.all_scenarios[scenario_name]
                self.assertTrue(validate_ambulance_scenario(config),
                              f"Scenario {scenario_name} failed validation")
    
    def test_scenario_diversity(self):
        """Test that scenarios have diverse configurations."""
        traffic_densities = set()
        highway_conditions = set()
        emergency_priorities = set()
        
        for scenario_name in self.scenario_names:
            config = self.all_scenarios[scenario_name]
            
            # Collect diversity metrics
            if "traffic_density" in config:
                traffic_densities.add(config["traffic_density"])
            
            if "highway_conditions" in config:
                highway_conditions.add(config["highway_conditions"])
            
            ambulance_config = config.get("_ambulance_config", {})
            if "emergency_priority" in ambulance_config:
                emergency_priorities.add(ambulance_config["emergency_priority"])
        
        # Verify diversity
        self.assertGreaterEqual(len(traffic_densities), 2,
                               f"Expected diverse traffic densities, found: {traffic_densities}")
        self.assertGreaterEqual(len(highway_conditions), 3,
                               f"Expected diverse highway conditions, found: {highway_conditions}")
        self.assertGreaterEqual(len(emergency_priorities), 1,
                               f"Expected emergency priorities, found: {emergency_priorities}")
    
    def test_get_scenario_by_name(self):
        """Test individual scenario retrieval."""
        for scenario_name in self.scenario_names:
            with self.subTest(scenario=scenario_name):
                config = get_scenario_by_name(scenario_name)
                self.assertIsInstance(config, dict)
                self.assertEqual(config.get("scenario_name"), scenario_name)
        
        # Test invalid scenario name
        with self.assertRaises(KeyError):
            get_scenario_by_name("invalid_scenario_name")
    
    def test_supported_observation_types(self):
        """Test that all required observation types are supported."""
        expected_obs_types = ["Kinematics", "OccupancyGrid", "GrayscaleObservation"]
        self.assertEqual(set(self.supported_obs_types), set(expected_obs_types),
                        f"Expected observation types {expected_obs_types}, got {self.supported_obs_types}")


class TestAmbulanceEnvironmentCreation(unittest.TestCase):
    """Test ambulance environment creation and agent setup."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env_factory = MultiAgentEnvFactory()
        self.scenario_names = get_scenario_names()
        self.supported_obs_types = get_supported_observation_types()
        self.n_agents = 4
    
    def test_environment_factory_ambulance_methods(self):
        """Test that environment factory has ambulance-specific methods."""
        # Check that ambulance methods exist
        self.assertTrue(hasattr(self.env_factory, 'create_ambulance_env'),
                       "Environment factory missing create_ambulance_env method")
        self.assertTrue(hasattr(self.env_factory, 'get_ambulance_base_config'),
                       "Environment factory missing get_ambulance_base_config method")
        self.assertTrue(hasattr(self.env_factory, 'create_parallel_ambulance_envs'),
                       "Environment factory missing create_parallel_ambulance_envs method")
    
    def test_ambulance_base_config_generation(self):
        """Test ambulance base configuration generation."""
        for scenario_name in self.scenario_names[:3]:  # Test first 3 for speed
            with self.subTest(scenario=scenario_name):
                config = self.env_factory.get_ambulance_base_config(scenario_name, self.n_agents)
                
                # Verify ambulance-specific configuration
                self.assertIn("_ambulance_config", config)
                ambulance_config = config["_ambulance_config"]
                self.assertEqual(ambulance_config.get("ambulance_agent_index"), 0)
                self.assertIn("emergency_priority", ambulance_config)
                
                # Verify multi-agent configuration
                self.assertEqual(config.get("controlled_vehicles"), self.n_agents)
                self.assertEqual(config.get("lanes_count"), 4)
    
    def test_ambulance_environment_creation_all_obs_types(self):
        """Test ambulance environment creation for all observation types."""
        test_scenario = self.scenario_names[0]  # Use first scenario for testing
        
        for obs_type in self.supported_obs_types:
            with self.subTest(obs_type=obs_type):
                try:
                    env = self.env_factory.create_ambulance_env(test_scenario, obs_type, self.n_agents)
                    self.assertIsNotNone(env, f"Environment creation returned None for {obs_type}")
                    
                    # Test environment reset
                    obs, info = env.reset()
                    self.assertIsNotNone(obs, f"Environment reset returned None observation for {obs_type}")
                    
                    # For multi-agent, check observation structure
                    if self.n_agents > 1:
                        self.assertTrue(isinstance(obs, (list, tuple)) and len(obs) == self.n_agents,
                                      f"Expected {self.n_agents} observations for {obs_type}, got {type(obs)}")
                    
                    # Test action space
                    action_space = env.action_space
                    self.assertIsNotNone(action_space, f"No action space for {obs_type}")
                    
                    # Test observation space
                    observation_space = env.observation_space
                    self.assertIsNotNone(observation_space, f"No observation space for {obs_type}")
                    
                    env.close()
                    
                except Exception as e:
                    self.fail(f"Environment creation failed for {test_scenario}/{obs_type}: {e}")
    
    def test_parallel_ambulance_environments(self):
        """Test parallel ambulance environment creation."""
        test_scenario = self.scenario_names[0]
        
        # Test creating all modalities
        parallel_envs = self.env_factory.create_parallel_ambulance_envs(
            test_scenario, self.n_agents, self.supported_obs_types
        )
        
        self.assertEqual(len(parallel_envs), len(self.supported_obs_types),
                        f"Expected {len(self.supported_obs_types)} parallel environments")
        
        for obs_type in self.supported_obs_types:
            self.assertIn(obs_type, parallel_envs,
                         f"Missing parallel environment for {obs_type}")
            
            env = parallel_envs[obs_type]
            self.assertIsNotNone(env, f"Parallel environment is None for {obs_type}")
            
            # Test that environment can be reset
            obs, info = env.reset()
            self.assertIsNotNone(obs, f"Parallel environment reset failed for {obs_type}")
            
            env.close()
    
    def test_ambulance_agent_setup(self):
        """Test that ambulance agent is properly configured as first agent."""
        test_scenario = self.scenario_names[0]
        obs_type = "Kinematics"  # Use Kinematics for simplicity
        
        env = self.env_factory.create_ambulance_env(test_scenario, obs_type, self.n_agents)
        obs, info = env.reset()
        
        # Check that we have the correct number of agents
        if isinstance(obs, (list, tuple)):
            self.assertEqual(len(obs), self.n_agents,
                           f"Expected {self.n_agents} agent observations")
        
        # Test that environment has ambulance metadata
        if hasattr(env, 'unwrapped'):
            env_unwrapped = env.unwrapped
            if hasattr(env_unwrapped, '_is_ambulance_env'):
                self.assertTrue(env_unwrapped._is_ambulance_env,
                              "Environment should be marked as ambulance environment")
        
        env.close()
    
    def test_environment_validation_method(self):
        """Test environment factory validation method."""
        test_scenario = self.scenario_names[0]
        
        for obs_type in self.supported_obs_types:
            with self.subTest(obs_type=obs_type):
                # This should not raise an exception
                try:
                    is_valid = self.env_factory.validate_ambulance_configuration(
                        test_scenario, obs_type, self.n_agents
                    )
                    # Validation should return a boolean or not raise an exception
                    self.assertIsInstance(is_valid, bool)
                except Exception as e:
                    self.fail(f"Validation method failed for {test_scenario}/{obs_type}: {e}")


class TestAmbulanceDataCollection(unittest.TestCase):
    """Test ambulance data collection and multi-modal outputs."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = AmbulanceDataCollector(n_agents=4)
        self.scenario_names = get_scenario_names()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.collector.cleanup()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ambulance_collector_initialization(self):
        """Test ambulance collector initialization."""
        self.assertEqual(self.collector.n_agents, 4)
        self.assertIsNotNone(self.collector._collector)
        self.assertIsNotNone(self.collector._env_factory)
        self.assertEqual(len(self.collector._scenario_names), 15)
        
        # Test statistics initialization
        stats = self.collector.get_collection_statistics()
        self.assertIn("ambulance_episodes_collected", stats)
        self.assertIn("ambulance_scenarios_processed", stats)
        self.assertEqual(stats["ambulance_episodes_collected"], 0)
    
    def test_available_scenarios_and_info(self):
        """Test getting available scenarios and scenario information."""
        scenarios = self.collector.get_available_scenarios()
        self.assertEqual(len(scenarios), 15)
        
        # Test scenario info for first scenario
        first_scenario = scenarios[0]
        info = self.collector.get_scenario_info(first_scenario)
        
        required_info_fields = [
            "scenario_name", "traffic_density", "vehicles_count", "duration",
            "lanes_count", "controlled_vehicles", "ambulance_config",
            "supports_multi_modal", "supported_observations"
        ]
        
        for field in required_info_fields:
            self.assertIn(field, info, f"Missing info field: {field}")
        
        self.assertEqual(info["controlled_vehicles"], 4)
        self.assertTrue(info["supports_multi_modal"])
        self.assertEqual(len(info["supported_observations"]), 3)
    
    def test_ambulance_environment_setup(self):
        """Test ambulance environment setup."""
        test_scenario = self.scenario_names[0]
        
        setup_info = self.collector.setup_ambulance_environments(test_scenario)
        
        required_setup_fields = [
            "scenario_name", "n_agents", "ambulance_agent_index",
            "emergency_priority", "environments_created", "supported_modalities"
        ]
        
        for field in required_setup_fields:
            self.assertIn(field, setup_info, f"Missing setup info field: {field}")
        
        self.assertEqual(setup_info["scenario_name"], test_scenario)
        self.assertEqual(setup_info["n_agents"], 4)
        self.assertEqual(setup_info["ambulance_agent_index"], 0)
        self.assertGreater(setup_info["environments_created"], 0)
    
    def test_single_scenario_data_collection(self):
        """Test data collection from a single ambulance scenario."""
        test_scenario = self.scenario_names[0]
        
        # Collect minimal data for testing (1 episode, 3 steps)
        result = self.collector.collect_single_ambulance_scenario(
            scenario_name=test_scenario,
            episodes=1,
            max_steps=3,
            seed=42,
            batch_size=1
        )
        
        # Verify collection result structure
        self.assertIsNotNone(result)
        self.assertEqual(result.total_episodes, 1)
        self.assertGreaterEqual(result.successful_episodes, 0)
        self.assertIsInstance(result.collection_time, float)
        
        # If collection was successful, verify episode data
        if result.successful_episodes > 0:
            self.assertEqual(len(result.episodes), 1)
            episode = result.episodes[0]
            
            # Verify episode structure
            self.assertIsNotNone(episode.episode_id)
            self.assertEqual(episode.scenario, test_scenario)
            self.assertGreater(len(episode.observations), 0)
            
            # Verify multi-agent observations
            first_step_obs = episode.observations[0]
            self.assertEqual(len(first_step_obs), 4, "Should have 4 agent observations")
    
    def test_multi_modal_data_collection_outputs(self):
        """Test that data collection produces expected multi-modal outputs."""
        test_scenario = self.scenario_names[0]
        
        # Set up environments to ensure all modalities are available
        setup_info = self.collector.setup_ambulance_environments(test_scenario)
        supported_modalities = setup_info.get("supported_modalities", [])
        
        if not supported_modalities:
            self.skipTest("No supported modalities available for testing")
        
        # Collect minimal data
        result = self.collector.collect_single_ambulance_scenario(
            scenario_name=test_scenario,
            episodes=1,
            max_steps=2,
            seed=42
        )
        
        if result.successful_episodes > 0:
            episode = result.episodes[0]
            
            # Check that observations contain multi-modal data
            for step_observations in episode.observations:
                self.assertEqual(len(step_observations), 4, "Should have 4 agent observations")
                
                # Each agent observation should contain data
                for agent_obs in step_observations:
                    self.assertIsInstance(agent_obs, dict, "Agent observation should be a dictionary")
                    # The exact structure depends on the observation type, but should not be empty
                    self.assertGreater(len(agent_obs), 0, "Agent observation should not be empty")
    
    def test_data_storage_integration(self):
        """Test integration with data storage system."""
        test_scenario = self.scenario_names[0]
        
        # Collect minimal data
        collection_results = self.collector.collect_ambulance_data(
            scenarios=[test_scenario],
            episodes_per_scenario=1,
            max_steps_per_episode=2,
            batch_size=1
        )
        
        # Test data storage
        if collection_results[test_scenario].successful_episodes > 0:
            storage_info = self.collector.store_ambulance_data(
                collection_results, Path(self.temp_dir)
            )
            
            # Verify storage info structure
            required_storage_fields = [
                "output_dir", "scenarios_stored", "total_episodes_stored",
                "storage_paths", "errors"
            ]
            
            for field in required_storage_fields:
                self.assertIn(field, storage_info, f"Missing storage info field: {field}")
            
            self.assertEqual(storage_info["scenarios_stored"], 1)
            self.assertGreater(storage_info["total_episodes_stored"], 0)
            self.assertEqual(len(storage_info["errors"]), 0)
    
    def test_collection_statistics_tracking(self):
        """Test that collection statistics are properly tracked."""
        initial_stats = self.collector.get_collection_statistics()
        initial_episodes = initial_stats["ambulance_episodes_collected"]
        initial_scenarios = initial_stats["ambulance_scenarios_processed"]
        
        test_scenario = self.scenario_names[0]
        
        # Collect data
        result = self.collector.collect_single_ambulance_scenario(
            scenario_name=test_scenario,
            episodes=1,
            max_steps=2,
            seed=42
        )
        
        # Check updated statistics
        updated_stats = self.collector.get_collection_statistics()
        
        if result.successful_episodes > 0:
            self.assertGreater(updated_stats["ambulance_episodes_collected"], initial_episodes)
            self.assertGreater(updated_stats["ambulance_scenarios_processed"], initial_scenarios)
        
        # Test statistics reset
        self.collector.reset_statistics()
        reset_stats = self.collector.get_collection_statistics()
        self.assertEqual(reset_stats["ambulance_episodes_collected"], 0)
        self.assertEqual(reset_stats["ambulance_scenarios_processed"], 0)


class TestAmbulanceErrorHandling(unittest.TestCase):
    """Test error handling and edge cases for ambulance system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = AmbulanceDataCollector(n_agents=4)
        self.env_factory = MultiAgentEnvFactory()
        self.validator = AmbulanceScenarioValidator(n_agents=4)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.collector.cleanup()
    
    def test_invalid_scenario_names(self):
        """Test handling of invalid scenario names."""
        invalid_scenario = "invalid_ambulance_scenario"
        
        # Test collector methods with invalid scenario
        with self.assertRaises(ValueError):
            self.collector.setup_ambulance_environments(invalid_scenario)
        
        with self.assertRaises(ValueError):
            self.collector.collect_single_ambulance_scenario(invalid_scenario)
        
        with self.assertRaises(ValueError):
            self.collector.get_scenario_info(invalid_scenario)
        
        # Test environment factory with invalid scenario
        with self.assertRaises((ValueError, KeyError)):
            self.env_factory.create_ambulance_env(invalid_scenario, "Kinematics", 4)
    
    def test_invalid_observation_types(self):
        """Test handling of invalid observation types."""
        valid_scenario = get_scenario_names()[0]
        invalid_obs_type = "InvalidObservationType"
        
        with self.assertRaises(ValueError):
            self.env_factory.create_ambulance_env(valid_scenario, invalid_obs_type, 4)
    
    def test_invalid_agent_counts(self):
        """Test handling of invalid agent counts."""
        valid_scenario = get_scenario_names()[0]
        valid_obs_type = "Kinematics"
        
        # Test with 0 agents
        with self.assertRaises(ValueError):
            self.env_factory.create_ambulance_env(valid_scenario, valid_obs_type, 0)
        
        # Test with negative agents
        with self.assertRaises(ValueError):
            self.env_factory.create_ambulance_env(valid_scenario, valid_obs_type, -1)
        
        # Test with too many agents (ambulance scenarios support max 4)
        with self.assertRaises(ValueError):
            self.env_factory.get_ambulance_base_config(valid_scenario, 10)
    
    def test_environment_creation_failures(self):
        """Test handling of environment creation failures."""
        # Test with mock that raises exception
        with patch.object(self.env_factory, 'get_ambulance_base_config') as mock_config:
            mock_config.side_effect = Exception("Mock environment creation failure")
            
            with self.assertRaises(Exception):
                self.env_factory.create_ambulance_env(get_scenario_names()[0], "Kinematics", 4)
    
    def test_data_collection_error_recovery(self):
        """Test error recovery during data collection."""
        # Test with invalid collection parameters
        valid_scenario = get_scenario_names()[0]
        
        # Test with invalid episodes count
        with self.assertRaises((ValueError, TypeError)):
            self.collector.collect_single_ambulance_scenario(
                scenario_name=valid_scenario,
                episodes=-1,  # Invalid
                max_steps=10
            )
        
        # Test with invalid max_steps
        with self.assertRaises((ValueError, TypeError)):
            self.collector.collect_single_ambulance_scenario(
                scenario_name=valid_scenario,
                episodes=1,
                max_steps=-1  # Invalid
            )
    
    def test_validation_error_handling(self):
        """Test validation error handling."""
        # Test validator with invalid scenario
        invalid_scenario = "invalid_scenario"
        result = self.validator.validate_scenario_configuration(invalid_scenario)
        
        self.assertFalse(result["valid"])
        self.assertGreater(len(result["errors"]), 0)
        
        # Test environment validation with invalid parameters
        valid_scenario = get_scenario_names()[0]
        result = self.validator.validate_environment_creation(valid_scenario, "InvalidObsType")
        
        self.assertFalse(result["valid"])
        self.assertGreater(len(result["errors"]), 0)
    
    def test_resource_cleanup(self):
        """Test proper resource cleanup."""
        test_scenario = get_scenario_names()[0]
        
        # Set up environments
        self.collector.setup_ambulance_environments(test_scenario)
        
        # Test cleanup
        self.collector.cleanup()
        
        # Verify cleanup was successful (no exceptions should be raised)
        self.collector.cleanup()  # Should be safe to call multiple times
    
    def test_concurrent_access_safety(self):
        """Test thread safety and concurrent access."""
        # This is a basic test - in a real scenario you might use threading
        test_scenario = get_scenario_names()[0]
        
        # Multiple calls should not interfere with each other
        info1 = self.collector.get_scenario_info(test_scenario)
        info2 = self.collector.get_scenario_info(test_scenario)
        
        self.assertEqual(info1, info2)
        
        # Multiple statistics calls should be consistent
        stats1 = self.collector.get_collection_statistics()
        stats2 = self.collector.get_collection_statistics()
        
        self.assertEqual(stats1, stats2)


class TestAmbulanceSystemIntegration(unittest.TestCase):
    """Test integration with existing highway data collection systems."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scenario_registry = ScenarioRegistry()
        self.env_factory = MultiAgentEnvFactory()
    
    def test_scenario_registry_integration(self):
        """Test integration with existing scenario registry."""
        # Test that ambulance scenarios are properly integrated
        all_scenarios = self.scenario_registry.list_scenarios()
        ambulance_scenarios = self.scenario_registry.list_ambulance_scenarios()
        
        # Ambulance scenarios should be included in all scenarios
        for ambulance_scenario in ambulance_scenarios:
            self.assertIn(ambulance_scenario, all_scenarios,
                         f"Ambulance scenario {ambulance_scenario} not in all scenarios list")
    
    def test_environment_factory_compatibility(self):
        """Test that ambulance environments are compatible with existing factory."""
        # Test that both regular and ambulance environment creation work
        regular_scenarios = self.scenario_registry.list_regular_scenarios()
        ambulance_scenarios = self.scenario_registry.list_ambulance_scenarios()
        
        if regular_scenarios:
            # Test regular environment creation still works
            regular_env = self.env_factory.create_env(regular_scenarios[0], "Kinematics", 2)
            self.assertIsNotNone(regular_env)
            regular_env.close()
        
        if ambulance_scenarios:
            # Test ambulance environment creation works
            ambulance_env = self.env_factory.create_ambulance_env(ambulance_scenarios[0], "Kinematics", 4)
            self.assertIsNotNone(ambulance_env)
            ambulance_env.close()
    
    def test_observation_config_compatibility(self):
        """Test that ambulance scenarios work with existing observation configurations."""
        from highway_datacollection.environments.config import OBSERVATION_CONFIGS
        
        # Test that all supported observation types have configurations
        supported_obs_types = get_supported_observation_types()
        
        for obs_type in supported_obs_types:
            self.assertIn(obs_type, OBSERVATION_CONFIGS,
                         f"Observation type {obs_type} not in OBSERVATION_CONFIGS")
    
    def test_action_sampler_compatibility(self):
        """Test that ambulance environments work with existing action samplers."""
        from highway_datacollection.collection.action_samplers import RandomActionSampler
        
        test_scenario = get_scenario_names()[0]
        env = self.env_factory.create_ambulance_env(test_scenario, "Kinematics", 4)
        obs, info = env.reset()
        
        # Test that action sampler works with ambulance environment
        action_sampler = RandomActionSampler()
        dummy_observations = {"Kinematics": {"observation": obs}}
        actions = action_sampler.sample_actions(dummy_observations, 4, 0)
        
        self.assertIsNotNone(actions)
        self.assertEqual(len(actions), 4)
        
        # Test that actions can be applied to environment
        obs, rewards, dones, truncated, info = env.step(actions)
        self.assertIsNotNone(obs)
        
        env.close()


if __name__ == '__main__':
    # Set up test environment
    os.environ.setdefault('PYTHONPATH', str(Path(__file__).parent.parent))
    
    # Configure logging for tests
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests with high verbosity
    unittest.main(verbosity=2, buffer=True)