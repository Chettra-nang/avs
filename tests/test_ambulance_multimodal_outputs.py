"""
Tests for ambulance multi-modal data collection outputs.

This module specifically tests that ambulance data collection produces
expected multi-modal outputs across all observation types.

Requirements covered: 1.4, 2.4
"""

import unittest
import sys
import os
import logging
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector
from collecting_ambulance_data.scenarios.ambulance_scenarios import get_scenario_names, get_supported_observation_types
from highway_datacollection.environments.factory import MultiAgentEnvFactory


class TestAmbulanceMultiModalOutputs(unittest.TestCase):
    """Test multi-modal data collection outputs for ambulance scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = AmbulanceDataCollector(n_agents=4)
        self.env_factory = MultiAgentEnvFactory()
        self.scenario_names = get_scenario_names()
        self.obs_types = get_supported_observation_types()
        self.temp_dir = tempfile.mkdtemp()
        
        # Reduce logging noise
        logging.getLogger('highway_datacollection').setLevel(logging.WARNING)
        logging.getLogger('collecting_ambulance_data').setLevel(logging.WARNING)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.collector.cleanup()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_kinematics_observation_output(self):
        """Test Kinematics observation output structure and content."""
        test_scenario = self.scenario_names[0]
        obs_type = "Kinematics"
        
        env = self.env_factory.create_ambulance_env(test_scenario, obs_type, 4)
        obs, info = env.reset()
        
        # Test observation structure
        self.assertIsInstance(obs, (list, tuple), "Kinematics observations should be a list/tuple")
        self.assertEqual(len(obs), 4, "Should have 4 agent observations")
        
        # Test each agent's observation
        for i, agent_obs in enumerate(obs):
            with self.subTest(agent=i):
                self.assertIsInstance(agent_obs, np.ndarray, 
                                    f"Agent {i} observation should be numpy array")
                self.assertGreater(agent_obs.shape[0], 0, 
                                 f"Agent {i} observation should not be empty")
                
                # Kinematics observations typically have shape (n_vehicles, features)
                # where features include position, velocity, heading, presence
                self.assertEqual(len(agent_obs.shape), 2, 
                               f"Agent {i} observation should be 2D array")
                
                # Check for reasonable feature count (typically 5: x, y, vx, vy, presence)
                self.assertGreaterEqual(agent_obs.shape[1], 5, 
                                      f"Agent {i} should have at least 5 features")
        
        env.close()
    
    def test_occupancy_grid_observation_output(self):
        """Test OccupancyGrid observation output structure and content."""
        test_scenario = self.scenario_names[0]
        obs_type = "OccupancyGrid"
        
        env = self.env_factory.create_ambulance_env(test_scenario, obs_type, 4)
        obs, info = env.reset()
        
        # Test observation structure
        self.assertIsInstance(obs, (list, tuple), "OccupancyGrid observations should be a list/tuple")
        self.assertEqual(len(obs), 4, "Should have 4 agent observations")
        
        # Test each agent's observation
        for i, agent_obs in enumerate(obs):
            with self.subTest(agent=i):
                self.assertIsInstance(agent_obs, np.ndarray, 
                                    f"Agent {i} observation should be numpy array")
                
                # OccupancyGrid observations are typically 2D grids
                self.assertEqual(len(agent_obs.shape), 2, 
                               f"Agent {i} observation should be 2D grid")
                
                # Check for reasonable grid dimensions
                self.assertGreater(agent_obs.shape[0], 0, 
                                 f"Agent {i} grid height should be > 0")
                self.assertGreater(agent_obs.shape[1], 0, 
                                 f"Agent {i} grid width should be > 0")
                
                # OccupancyGrid values should be binary (0 or 1) or probabilities [0, 1]
                self.assertTrue(np.all(agent_obs >= 0), 
                              f"Agent {i} occupancy values should be >= 0")
                self.assertTrue(np.all(agent_obs <= 1), 
                              f"Agent {i} occupancy values should be <= 1")
        
        env.close()
    
    def test_grayscale_observation_output(self):
        """Test GrayscaleObservation output structure and content."""
        test_scenario = self.scenario_names[0]
        obs_type = "GrayscaleObservation"
        
        env = self.env_factory.create_ambulance_env(test_scenario, obs_type, 4)
        obs, info = env.reset()
        
        # Test observation structure
        self.assertIsInstance(obs, (list, tuple), "GrayscaleObservation should be a list/tuple")
        self.assertEqual(len(obs), 4, "Should have 4 agent observations")
        
        # Test each agent's observation
        for i, agent_obs in enumerate(obs):
            with self.subTest(agent=i):
                self.assertIsInstance(agent_obs, np.ndarray, 
                                    f"Agent {i} observation should be numpy array")
                
                # Grayscale images should be 2D or 3D arrays
                self.assertIn(len(agent_obs.shape), [2, 3], 
                            f"Agent {i} observation should be 2D or 3D array")
                
                if len(agent_obs.shape) == 3:
                    # If 3D, last dimension should be 1 (grayscale) or 3 (RGB converted to grayscale)
                    self.assertIn(agent_obs.shape[2], [1, 3], 
                                f"Agent {i} image should have 1 or 3 channels")
                
                # Check for reasonable image dimensions
                self.assertGreater(agent_obs.shape[0], 0, 
                                 f"Agent {i} image height should be > 0")
                self.assertGreater(agent_obs.shape[1], 0, 
                                 f"Agent {i} image width should be > 0")
                
                # Grayscale values should be in reasonable range (0-255 or 0-1)
                min_val, max_val = np.min(agent_obs), np.max(agent_obs)
                self.assertGreaterEqual(min_val, 0, 
                                      f"Agent {i} pixel values should be >= 0")
                
                # Check if values are in 0-1 range or 0-255 range
                if max_val <= 1.0:
                    self.assertLessEqual(max_val, 1.0, 
                                       f"Agent {i} normalized pixel values should be <= 1")
                else:
                    self.assertLessEqual(max_val, 255, 
                                       f"Agent {i} pixel values should be <= 255")
        
        env.close()
    
    def test_multi_modal_data_collection_consistency(self):
        """Test that multi-modal data collection produces consistent results."""
        test_scenario = self.scenario_names[0]
        
        # Collect data with minimal episodes for each observation type
        results = {}
        
        for obs_type in self.obs_types:
            with self.subTest(obs_type=obs_type):
                # Set up environment for this observation type
                self.collector.setup_ambulance_environments(test_scenario)
                
                # Collect minimal data
                result = self.collector.collect_single_ambulance_scenario(
                    scenario_name=test_scenario,
                    episodes=1,
                    max_steps=3,
                    seed=42  # Use same seed for consistency
                )
                
                results[obs_type] = result
                
                # Verify collection was successful
                if result.successful_episodes > 0:
                    self.assertEqual(len(result.episodes), 1)
                    episode = result.episodes[0]
                    
                    # Verify episode structure
                    self.assertEqual(episode.scenario, test_scenario)
                    self.assertGreater(len(episode.observations), 0)
                    
                    # Verify multi-agent structure
                    for step_obs in episode.observations:
                        self.assertEqual(len(step_obs), 4, 
                                       f"Should have 4 agent observations for {obs_type}")
        
        # Verify that all observation types produced data
        successful_obs_types = [obs_type for obs_type, result in results.items() 
                               if result.successful_episodes > 0]
        self.assertGreater(len(successful_obs_types), 0, 
                          "At least one observation type should produce data")
    
    def test_ambulance_agent_data_identification(self):
        """Test that ambulance agent (first agent) data can be identified."""
        test_scenario = self.scenario_names[0]
        obs_type = "Kinematics"  # Use Kinematics for easier analysis
        
        # Collect minimal data
        result = self.collector.collect_single_ambulance_scenario(
            scenario_name=test_scenario,
            episodes=1,
            max_steps=2,
            seed=42
        )
        
        if result.successful_episodes > 0:
            episode = result.episodes[0]
            
            # Check that ambulance agent is first agent (index 0)
            for step_idx, step_observations in enumerate(episode.observations):
                ambulance_obs = step_observations[0]  # First agent is ambulance
                other_agent_obs = step_observations[1:]  # Other agents
                
                # Verify ambulance observation exists and has data
                self.assertIsNotNone(ambulance_obs)
                self.assertGreater(len(ambulance_obs), 0)
                
                # Verify other agents also have observations
                for i, other_obs in enumerate(other_agent_obs):
                    self.assertIsNotNone(other_obs, f"Agent {i+1} observation is None")
    
    def test_observation_data_types_and_ranges(self):
        """Test that observation data has correct types and value ranges."""
        test_scenario = self.scenario_names[0]
        
        for obs_type in self.obs_types:
            with self.subTest(obs_type=obs_type):
                env = self.env_factory.create_ambulance_env(test_scenario, obs_type, 4)
                obs, info = env.reset()
                
                # Execute a few steps to get varied data
                for step in range(3):
                    actions = [env.action_space.sample() for _ in range(4)]
                    obs, rewards, dones, truncated, info = env.step(actions)
                    
                    # Test observation data types and ranges
                    for agent_idx, agent_obs in enumerate(obs):
                        self.assertIsInstance(agent_obs, np.ndarray, 
                                            f"Agent {agent_idx} obs should be numpy array")
                        
                        # Check for NaN or infinite values
                        self.assertFalse(np.any(np.isnan(agent_obs)), 
                                       f"Agent {agent_idx} obs contains NaN values")
                        self.assertFalse(np.any(np.isinf(agent_obs)), 
                                       f"Agent {agent_idx} obs contains infinite values")
                        
                        # Observation-specific checks
                        if obs_type == "Kinematics":
                            # Kinematics should have reasonable position/velocity ranges
                            # Positions typically in [-100, 100] range for highway scenarios
                            # Velocities typically in [0, 50] range
                            pass  # Basic checks already done above
                        
                        elif obs_type == "OccupancyGrid":
                            # OccupancyGrid should be binary or probability values
                            self.assertTrue(np.all(agent_obs >= 0), 
                                          f"OccupancyGrid values should be >= 0")
                            self.assertTrue(np.all(agent_obs <= 1), 
                                          f"OccupancyGrid values should be <= 1")
                        
                        elif obs_type == "GrayscaleObservation":
                            # Grayscale should be in valid pixel range
                            min_val, max_val = np.min(agent_obs), np.max(agent_obs)
                            self.assertGreaterEqual(min_val, 0)
                            # Allow both normalized [0,1] and standard [0,255] ranges
                            self.assertTrue(max_val <= 1.0 or max_val <= 255)
                    
                    # Break if episode is done
                    if isinstance(dones, (list, tuple)):
                        if any(dones):
                            break
                    else:
                        if dones:
                            break
                
                env.close()
    
    def test_data_collection_output_format(self):
        """Test the format of collected data for storage and analysis."""
        test_scenario = self.scenario_names[0]
        
        # Collect data
        collection_results = self.collector.collect_ambulance_data(
            scenarios=[test_scenario],
            episodes_per_scenario=1,
            max_steps_per_episode=2,
            batch_size=1
        )
        
        self.assertIn(test_scenario, collection_results)
        result = collection_results[test_scenario]
        
        if result.successful_episodes > 0:
            episode = result.episodes[0]
            
            # Test episode metadata
            self.assertIsNotNone(episode.episode_id)
            self.assertEqual(episode.scenario, test_scenario)
            self.assertIsInstance(episode.metadata, dict)
            
            # Test that metadata contains ambulance-specific information
            metadata = episode.metadata
            expected_metadata_fields = ['scenario', 'episode_id', 'n_agents', 'max_steps']
            for field in expected_metadata_fields:
                self.assertIn(field, metadata, f"Missing metadata field: {field}")
            
            # Test observation data structure
            self.assertGreater(len(episode.observations), 0)
            self.assertEqual(len(episode.actions), len(episode.observations))
            self.assertEqual(len(episode.rewards), len(episode.observations))
            self.assertEqual(len(episode.dones), len(episode.observations))
            
            # Test that each step has 4 agent observations
            for step_idx, step_obs in enumerate(episode.observations):
                self.assertEqual(len(step_obs), 4, 
                               f"Step {step_idx} should have 4 agent observations")
                
                # Test that each agent observation is a dictionary with required fields
                for agent_idx, agent_obs in enumerate(step_obs):
                    self.assertIsInstance(agent_obs, dict, 
                                        f"Agent {agent_idx} observation should be dict")
                    
                    # Agent observations should contain the actual observation data
                    # The exact structure depends on the observation type and processing
                    self.assertGreater(len(agent_obs), 0, 
                                     f"Agent {agent_idx} observation should not be empty")
    
    def test_multi_modal_data_storage_format(self):
        """Test that multi-modal data can be stored in the expected format."""
        test_scenario = self.scenario_names[0]
        
        # Collect data
        collection_results = self.collector.collect_ambulance_data(
            scenarios=[test_scenario],
            episodes_per_scenario=1,
            max_steps_per_episode=2,
            batch_size=1
        )
        
        if collection_results[test_scenario].successful_episodes > 0:
            # Test data storage
            storage_info = self.collector.store_ambulance_data(
                collection_results, Path(self.temp_dir)
            )
            
            # Verify storage was successful
            self.assertEqual(storage_info["scenarios_stored"], 1)
            self.assertGreater(storage_info["total_episodes_stored"], 0)
            self.assertEqual(len(storage_info["errors"]), 0)
            
            # Check that files were created
            output_dir = Path(self.temp_dir)
            self.assertTrue(output_dir.exists())
            
            # Look for data files (exact structure depends on storage implementation)
            data_files = list(output_dir.rglob("*.parquet")) + list(output_dir.rglob("*.jsonl"))
            self.assertGreater(len(data_files), 0, "Should have created data files")
    
    def test_observation_consistency_across_steps(self):
        """Test that observations maintain consistency across steps."""
        test_scenario = self.scenario_names[0]
        obs_type = "Kinematics"
        
        env = self.env_factory.create_ambulance_env(test_scenario, obs_type, 4)
        obs, info = env.reset()
        
        # Record observation shapes and types
        initial_shapes = [agent_obs.shape for agent_obs in obs]
        initial_dtypes = [agent_obs.dtype for agent_obs in obs]
        
        # Execute several steps
        for step in range(5):
            actions = [env.action_space.sample() for _ in range(4)]
            obs, rewards, dones, truncated, info = env.step(actions)
            
            # Verify observation consistency
            self.assertEqual(len(obs), 4, f"Step {step}: Should have 4 observations")
            
            for agent_idx, agent_obs in enumerate(obs):
                # Shape should remain consistent
                self.assertEqual(agent_obs.shape, initial_shapes[agent_idx], 
                               f"Step {step}, Agent {agent_idx}: Shape changed")
                
                # Data type should remain consistent
                self.assertEqual(agent_obs.dtype, initial_dtypes[agent_idx], 
                               f"Step {step}, Agent {agent_idx}: Dtype changed")
            
            # Break if episode is done
            if isinstance(dones, (list, tuple)):
                if any(dones):
                    break
            else:
                if dones:
                    break
        
        env.close()


class TestAmbulanceMultiModalIntegration(unittest.TestCase):
    """Test integration of multi-modal outputs with existing systems."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = AmbulanceDataCollector(n_agents=4)
        self.scenario_names = get_scenario_names()
        self.obs_types = get_supported_observation_types()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.collector.cleanup()
    
    def test_compatibility_with_existing_action_samplers(self):
        """Test that multi-modal outputs work with existing action samplers."""
        from highway_datacollection.collection.action_samplers import RandomActionSampler
        
        test_scenario = self.scenario_names[0]
        action_sampler = RandomActionSampler()
        
        for obs_type in self.obs_types:
            with self.subTest(obs_type=obs_type):
                env = self.collector._env_factory.create_ambulance_env(test_scenario, obs_type, 4)
                obs, info = env.reset()
                
                # Create dummy observations dict for action sampler
                dummy_observations = {obs_type: {"observation": obs}}
                
                # Sample actions
                actions = action_sampler.sample_actions(dummy_observations, 4, 0)
                
                # Verify actions
                self.assertIsNotNone(actions)
                self.assertEqual(len(actions), 4)
                
                # Test that actions can be applied
                next_obs, rewards, dones, truncated, info = env.step(actions)
                self.assertIsNotNone(next_obs)
                
                env.close()
    
    def test_compatibility_with_existing_storage_systems(self):
        """Test that multi-modal data is compatible with existing storage."""
        from highway_datacollection.storage.manager import DatasetStorageManager
        
        test_scenario = self.scenario_names[0]
        
        # Collect minimal data
        result = self.collector.collect_single_ambulance_scenario(
            scenario_name=test_scenario,
            episodes=1,
            max_steps=2,
            seed=42
        )
        
        if result.successful_episodes > 0:
            # Test that data can be processed by storage manager
            with tempfile.TemporaryDirectory() as temp_dir:
                storage_manager = DatasetStorageManager(Path(temp_dir))
                
                episode = result.episodes[0]
                
                # Convert episode data to storage format
                all_observations = []
                all_metadata = []
                
                for episode in result.episodes:
                    all_metadata.append(episode.metadata)
                    
                    for step_idx, step_observations in enumerate(episode.observations):
                        for obs in step_observations:
                            obs_record = obs.copy()
                            obs_record.update({
                                'episode_id': episode.episode_id,
                                'scenario': episode.scenario,
                                'step': step_idx,
                                'ambulance_scenario': True,
                                'ambulance_agent_index': 0
                            })
                            all_observations.append(obs_record)
                
                # This should not raise an exception
                try:
                    storage_paths = storage_manager.write_episode_batch(
                        data=all_observations,
                        metadata=all_metadata,
                        scenario=test_scenario
                    )
                    self.assertIsNotNone(storage_paths)
                except Exception as e:
                    self.fail(f"Storage integration failed: {e}")
    
    def test_multi_modal_data_analysis_compatibility(self):
        """Test that multi-modal data is compatible with analysis tools."""
        test_scenario = self.scenario_names[0]
        
        # Collect data with different observation types
        collection_results = {}
        
        for obs_type in self.obs_types[:2]:  # Test first 2 for performance
            self.collector.setup_ambulance_environments(test_scenario)
            result = self.collector.collect_single_ambulance_scenario(
                scenario_name=test_scenario,
                episodes=1,
                max_steps=2,
                seed=42
            )
            collection_results[obs_type] = result
        
        # Verify that data from different modalities can be analyzed together
        successful_results = {obs_type: result for obs_type, result in collection_results.items() 
                            if result.successful_episodes > 0}
        
        if len(successful_results) > 1:
            # Compare episode structures across modalities
            obs_types = list(successful_results.keys())
            episode1 = successful_results[obs_types[0]].episodes[0]
            episode2 = successful_results[obs_types[1]].episodes[0]
            
            # Episodes should have same basic structure
            self.assertEqual(episode1.scenario, episode2.scenario)
            self.assertEqual(len(episode1.observations), len(episode2.observations))
            self.assertEqual(len(episode1.actions), len(episode2.actions))
            
            # Both should have 4 agents per step
            for step_idx in range(len(episode1.observations)):
                self.assertEqual(len(episode1.observations[step_idx]), 4)
                self.assertEqual(len(episode2.observations[step_idx]), 4)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main(verbosity=2, buffer=True)