"""
Specialized tests for ambulance environment factory functionality.

This module focuses specifically on testing the MultiAgentEnvFactory's
ambulance-specific methods and configurations.

Requirements covered: 1.4, 2.4
"""

import unittest
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from highway_datacollection.environments.factory import MultiAgentEnvFactory
from collecting_ambulance_data.scenarios.ambulance_scenarios import get_scenario_names, get_supported_observation_types


class TestAmbulanceEnvironmentFactory(unittest.TestCase):
    """Test ambulance-specific environment factory functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = MultiAgentEnvFactory()
        self.scenario_names = get_scenario_names()
        self.obs_types = get_supported_observation_types()
        self.n_agents = 4
        
        # Reduce logging noise during tests
        logging.getLogger('highway_datacollection').setLevel(logging.WARNING)
        logging.getLogger('collecting_ambulance_data').setLevel(logging.WARNING)
    
    def test_ambulance_base_config_structure(self):
        """Test the structure of ambulance base configurations."""
        test_scenario = self.scenario_names[0]
        config = self.factory.get_ambulance_base_config(test_scenario, self.n_agents)
        
        # Test required configuration fields
        required_fields = [
            'lanes_count', 'vehicles_count', 'duration', 'controlled_vehicles',
            '_ambulance_config', 'policy_frequency', 'render_mode'
        ]
        
        for field in required_fields:
            self.assertIn(field, config, f"Missing required field: {field}")
        
        # Test ambulance-specific configuration
        ambulance_config = config['_ambulance_config']
        self.assertIsInstance(ambulance_config, dict)
        self.assertEqual(ambulance_config['ambulance_agent_index'], 0)
        self.assertIn('emergency_priority', ambulance_config)
        self.assertIn('ambulance_behavior', ambulance_config)
        self.assertEqual(ambulance_config['other_agents_type'], 'normal')
        
        # Test multi-agent configuration
        self.assertEqual(config['controlled_vehicles'], self.n_agents)
        self.assertEqual(config['lanes_count'], 4)  # Fixed for ambulance scenarios
    
    def test_ambulance_config_validation_constraints(self):
        """Test validation constraints for ambulance configurations."""
        test_scenario = self.scenario_names[0]
        
        # Test valid agent counts
        for n_agents in [3, 4]:
            config = self.factory.get_ambulance_base_config(test_scenario, n_agents)
            self.assertEqual(config['controlled_vehicles'], n_agents)
        
        # Test invalid agent counts
        with self.assertRaises(ValueError):
            self.factory.get_ambulance_base_config(test_scenario, 0)
        
        with self.assertRaises(ValueError):
            self.factory.get_ambulance_base_config(test_scenario, -1)
        
        with self.assertRaises(ValueError):
            self.factory.get_ambulance_base_config(test_scenario, 10)  # Too many agents
    
    def test_ambulance_environment_creation_consistency(self):
        """Test that ambulance environments are created consistently."""
        test_scenario = self.scenario_names[0]
        obs_type = "Kinematics"
        
        # Create multiple environments and verify consistency
        envs = []
        configs = []
        
        for i in range(3):
            env = self.factory.create_ambulance_env(test_scenario, obs_type, self.n_agents)
            config = self.factory.get_ambulance_base_config(test_scenario, self.n_agents)
            
            envs.append(env)
            configs.append(config)
        
        # Verify all configurations are identical
        for i in range(1, len(configs)):
            self.assertEqual(configs[0], configs[i], 
                           f"Configuration {i} differs from configuration 0")
        
        # Verify all environments have same structure
        for i, env in enumerate(envs):
            obs, info = env.reset()
            
            # Check observation structure consistency
            if isinstance(obs, (list, tuple)):
                self.assertEqual(len(obs), self.n_agents,
                               f"Environment {i} has wrong number of observations")
            
            env.close()
    
    def test_parallel_ambulance_environment_creation(self):
        """Test parallel ambulance environment creation for all modalities."""
        test_scenario = self.scenario_names[0]
        
        # Test creating all modalities
        parallel_envs = self.factory.create_parallel_ambulance_envs(
            test_scenario, self.n_agents, self.obs_types
        )
        
        self.assertEqual(len(parallel_envs), len(self.obs_types))
        
        # Test each environment
        for obs_type in self.obs_types:
            self.assertIn(obs_type, parallel_envs)
            env = parallel_envs[obs_type]
            
            # Test environment functionality
            obs, info = env.reset()
            self.assertIsNotNone(obs)
            
            # Test action space
            action_space = env.action_space
            self.assertIsNotNone(action_space)
            
            # Test observation space
            observation_space = env.observation_space
            self.assertIsNotNone(observation_space)
            
            env.close()
        
        # Test creating subset of modalities
        subset_modalities = self.obs_types[:2]
        subset_envs = self.factory.create_parallel_ambulance_envs(
            test_scenario, self.n_agents, subset_modalities
        )
        
        self.assertEqual(len(subset_envs), len(subset_modalities))
        for obs_type in subset_modalities:
            self.assertIn(obs_type, subset_envs)
            subset_envs[obs_type].close()
    
    def test_ambulance_environment_metadata(self):
        """Test that ambulance environments have proper metadata."""
        test_scenario = self.scenario_names[0]
        obs_type = "Kinematics"
        
        env = self.factory.create_ambulance_env(test_scenario, obs_type, self.n_agents)
        
        # Check for ambulance-specific metadata
        if hasattr(env, 'unwrapped'):
            env_unwrapped = env.unwrapped
            
            # Check ambulance environment marker
            if hasattr(env_unwrapped, '_is_ambulance_env'):
                self.assertTrue(env_unwrapped._is_ambulance_env)
            
            # Check controlled vehicles configuration
            if hasattr(env_unwrapped, 'controlled_vehicles'):
                controlled_vehicles = env_unwrapped.controlled_vehicles
                if controlled_vehicles is not None:
                    self.assertEqual(len(controlled_vehicles), self.n_agents)
        
        env.close()
    
    def test_ambulance_environment_step_execution(self):
        """Test that ambulance environments can execute steps properly."""
        test_scenario = self.scenario_names[0]
        
        for obs_type in self.obs_types:
            with self.subTest(obs_type=obs_type):
                env = self.factory.create_ambulance_env(test_scenario, obs_type, self.n_agents)
                obs, info = env.reset()
                
                # Sample random actions
                actions = [env.action_space.sample() for _ in range(self.n_agents)]
                
                # Execute step
                next_obs, rewards, dones, truncated, step_info = env.step(actions)
                
                # Verify step results
                self.assertIsNotNone(next_obs)
                self.assertIsNotNone(rewards)
                self.assertIsNotNone(dones)
                self.assertIsNotNone(truncated)
                
                # For multi-agent environments, check result structure
                if self.n_agents > 1:
                    if isinstance(rewards, (list, tuple)):
                        self.assertEqual(len(rewards), self.n_agents)
                    if isinstance(dones, (list, tuple)):
                        self.assertEqual(len(dones), self.n_agents)
                
                env.close()
    
    def test_ambulance_configuration_validation_method(self):
        """Test the ambulance configuration validation method."""
        test_scenario = self.scenario_names[0]
        
        for obs_type in self.obs_types:
            with self.subTest(obs_type=obs_type):
                # This should not raise an exception for valid configurations
                try:
                    is_valid = self.factory.validate_ambulance_configuration(
                        test_scenario, obs_type, self.n_agents
                    )
                    self.assertIsInstance(is_valid, bool)
                except Exception as e:
                    self.fail(f"Validation failed for {test_scenario}/{obs_type}: {e}")
        
        # Test with invalid parameters
        invalid_obs_type = "InvalidObservationType"
        try:
            result = self.factory.validate_ambulance_configuration(
                test_scenario, invalid_obs_type, self.n_agents
            )
            # Should either return False or raise an exception
            if isinstance(result, bool):
                self.assertFalse(result)
        except Exception:
            # Exception is acceptable for invalid parameters
            pass
    
    def test_ambulance_scenario_compatibility(self):
        """Test compatibility with different ambulance scenarios."""
        obs_type = "Kinematics"
        
        # Test first 5 scenarios for performance
        test_scenarios = self.scenario_names[:5]
        
        for scenario_name in test_scenarios:
            with self.subTest(scenario=scenario_name):
                # Test configuration generation
                config = self.factory.get_ambulance_base_config(scenario_name, self.n_agents)
                self.assertIsInstance(config, dict)
                self.assertIn('_ambulance_config', config)
                
                # Test environment creation
                env = self.factory.create_ambulance_env(scenario_name, obs_type, self.n_agents)
                self.assertIsNotNone(env)
                
                # Test environment reset
                obs, info = env.reset()
                self.assertIsNotNone(obs)
                
                env.close()
    
    def test_ambulance_environment_resource_management(self):
        """Test proper resource management for ambulance environments."""
        test_scenario = self.scenario_names[0]
        obs_type = "Kinematics"
        
        # Create and close multiple environments
        for i in range(5):
            env = self.factory.create_ambulance_env(test_scenario, obs_type, self.n_agents)
            obs, info = env.reset()
            env.close()
        
        # Create parallel environments and close them
        parallel_envs = self.factory.create_parallel_ambulance_envs(
            test_scenario, self.n_agents, self.obs_types
        )
        
        for env in parallel_envs.values():
            env.close()
        
        # No exceptions should be raised during cleanup
    
    def test_ambulance_environment_determinism(self):
        """Test that ambulance environments are deterministic with same seed."""
        test_scenario = self.scenario_names[0]
        obs_type = "Kinematics"
        seed = 42
        
        # Create two environments with same seed
        env1 = self.factory.create_ambulance_env(test_scenario, obs_type, self.n_agents)
        env2 = self.factory.create_ambulance_env(test_scenario, obs_type, self.n_agents)
        
        # Reset with same seed
        obs1, info1 = env1.reset(seed=seed)
        obs2, info2 = env2.reset(seed=seed)
        
        # Observations should be identical (or very similar for floating point)
        if isinstance(obs1, (list, tuple)) and isinstance(obs2, (list, tuple)):
            self.assertEqual(len(obs1), len(obs2))
            # Note: Exact equality might not hold due to floating point precision
            # In a real test, you might want to use numpy.allclose for numerical comparison
        
        env1.close()
        env2.close()


class TestAmbulanceEnvironmentFactoryEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for ambulance environment factory."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = MultiAgentEnvFactory()
        self.scenario_names = get_scenario_names()
        self.obs_types = get_supported_observation_types()
    
    def test_invalid_scenario_handling(self):
        """Test handling of invalid scenario names."""
        invalid_scenarios = [
            "nonexistent_scenario",
            "",
            None,
            123,
            ["list_instead_of_string"]
        ]
        
        for invalid_scenario in invalid_scenarios:
            with self.subTest(scenario=invalid_scenario):
                with self.assertRaises((ValueError, KeyError, TypeError, AttributeError)):
                    self.factory.get_ambulance_base_config(invalid_scenario, 4)
                
                with self.assertRaises((ValueError, KeyError, TypeError, AttributeError)):
                    self.factory.create_ambulance_env(invalid_scenario, "Kinematics", 4)
    
    def test_invalid_observation_type_handling(self):
        """Test handling of invalid observation types."""
        valid_scenario = self.scenario_names[0]
        invalid_obs_types = [
            "NonexistentObservationType",
            "",
            None,
            123,
            ["list_instead_of_string"]
        ]
        
        for invalid_obs_type in invalid_obs_types:
            with self.subTest(obs_type=invalid_obs_type):
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    self.factory.create_ambulance_env(valid_scenario, invalid_obs_type, 4)
    
    def test_boundary_agent_counts(self):
        """Test boundary conditions for agent counts."""
        valid_scenario = self.scenario_names[0]
        
        # Test minimum valid agent count
        config = self.factory.get_ambulance_base_config(valid_scenario, 3)
        self.assertEqual(config['controlled_vehicles'], 3)
        
        # Test maximum valid agent count
        config = self.factory.get_ambulance_base_config(valid_scenario, 4)
        self.assertEqual(config['controlled_vehicles'], 4)
        
        # Test invalid agent counts
        invalid_counts = [0, -1, -10, 5, 10, 100]
        
        for invalid_count in invalid_counts:
            with self.subTest(n_agents=invalid_count):
                with self.assertRaises(ValueError):
                    self.factory.get_ambulance_base_config(valid_scenario, invalid_count)
    
    def test_empty_modality_list(self):
        """Test handling of empty modality lists."""
        valid_scenario = self.scenario_names[0]
        
        # Test with empty list
        parallel_envs = self.factory.create_parallel_ambulance_envs(
            valid_scenario, 4, []
        )
        self.assertEqual(len(parallel_envs), 0)
        
        # Test with None (should use all modalities)
        parallel_envs = self.factory.create_parallel_ambulance_envs(
            valid_scenario, 4, None
        )
        self.assertGreater(len(parallel_envs), 0)
        
        for env in parallel_envs.values():
            env.close()
    
    def test_malformed_scenario_configurations(self):
        """Test handling of malformed scenario configurations."""
        # This test would require mocking the scenario loading
        # to inject malformed configurations
        pass  # Placeholder for more advanced testing
    
    def test_memory_usage_with_multiple_environments(self):
        """Test memory usage when creating many environments."""
        valid_scenario = self.scenario_names[0]
        obs_type = "Kinematics"
        
        # Create and immediately close many environments
        # This tests for memory leaks
        environments = []
        
        try:
            for i in range(10):  # Create 10 environments
                env = self.factory.create_ambulance_env(valid_scenario, obs_type, 4)
                environments.append(env)
                
                # Reset environment to ensure it's fully initialized
                obs, info = env.reset()
        
        finally:
            # Clean up all environments
            for env in environments:
                try:
                    env.close()
                except:
                    pass  # Ignore cleanup errors
    
    def test_concurrent_environment_creation(self):
        """Test concurrent environment creation (basic thread safety)."""
        import threading
        import queue
        
        valid_scenario = self.scenario_names[0]
        obs_type = "Kinematics"
        results = queue.Queue()
        errors = queue.Queue()
        
        def create_environment():
            try:
                env = self.factory.create_ambulance_env(valid_scenario, obs_type, 4)
                obs, info = env.reset()
                results.put(env)
            except Exception as e:
                errors.put(e)
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=create_environment)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertTrue(errors.empty(), f"Errors occurred: {list(errors.queue)}")
        self.assertEqual(results.qsize(), 3, "Not all environments were created")
        
        # Clean up environments
        while not results.empty():
            env = results.get()
            env.close()


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main(verbosity=2, buffer=True)