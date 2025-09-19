"""
Unit tests for SynchronizedCollector.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from highway_datacollection.collection.collector import SynchronizedCollector
from highway_datacollection.environments.factory import MultiAgentEnvFactory


class TestSynchronizedCollector:
    """Test cases for SynchronizedCollector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_agents = 2
        self.collector = SynchronizedCollector(n_agents=self.n_agents)
        self.test_scenario = "free_flow"
        self.test_seed = 42
    
    def test_init(self):
        """Test collector initialization."""
        collector = SynchronizedCollector(n_agents=3)
        assert collector._n_agents == 3
        assert hasattr(collector, '_env_factory')
        assert isinstance(collector._env_factory, MultiAgentEnvFactory)
        assert collector._environments == {}
        assert collector._current_seed is None
        assert collector._current_scenario is None
        assert len(collector._obs_types) == 3  # Kinematics, OccupancyGrid, GrayscaleObservation
    
    def test_init_default_agents(self):
        """Test collector initialization with default number of agents."""
        collector = SynchronizedCollector()
        assert collector._n_agents == 2
    
    @patch.object(MultiAgentEnvFactory, 'create_parallel_envs')
    def test_setup_environments_success(self, mock_create_parallel):
        """Test successful environment setup."""
        # Mock environments
        mock_envs = {
            'Kinematics': Mock(),
            'OccupancyGrid': Mock(),
            'GrayscaleObservation': Mock()
        }
        mock_create_parallel.return_value = mock_envs
        
        # Setup environments
        self.collector._setup_environments(self.test_scenario)
        
        # Verify setup
        mock_create_parallel.assert_called_once_with(self.test_scenario, self.n_agents)
        assert self.collector._environments == mock_envs
        assert self.collector._current_scenario == self.test_scenario
    
    @patch.object(MultiAgentEnvFactory, 'create_parallel_envs')
    def test_setup_environments_failure(self, mock_create_parallel):
        """Test environment setup failure."""
        mock_create_parallel.side_effect = Exception("Environment creation failed")
        
        with pytest.raises(Exception, match="Environment creation failed"):
            self.collector._setup_environments(self.test_scenario)
        
        assert self.collector._environments == {}
        assert self.collector._current_scenario is None
    
    @patch.object(MultiAgentEnvFactory, 'create_parallel_envs')
    def test_setup_environments_reuse(self, mock_create_parallel):
        """Test that environments are reused when scenario hasn't changed."""
        # Mock environments
        mock_envs = {
            'Kinematics': Mock(),
            'OccupancyGrid': Mock(),
            'GrayscaleObservation': Mock()
        }
        mock_create_parallel.return_value = mock_envs
        
        # Setup environments twice with same scenario
        self.collector._setup_environments(self.test_scenario)
        self.collector._setup_environments(self.test_scenario)
        
        # Should only be called once
        mock_create_parallel.assert_called_once()
    
    @patch.object(MultiAgentEnvFactory, 'create_parallel_envs')
    def test_setup_environments_scenario_change(self, mock_create_parallel):
        """Test environment setup when scenario changes."""
        # Mock environments
        mock_envs1 = {'Kinematics': Mock(), 'OccupancyGrid': Mock(), 'GrayscaleObservation': Mock()}
        mock_envs2 = {'Kinematics': Mock(), 'OccupancyGrid': Mock(), 'GrayscaleObservation': Mock()}
        mock_create_parallel.side_effect = [mock_envs1, mock_envs2]
        
        # Setup environments for first scenario
        self.collector._setup_environments("free_flow")
        assert self.collector._environments == mock_envs1
        
        # Setup environments for different scenario
        self.collector._setup_environments("dense_commuting")
        assert self.collector._environments == mock_envs2
        
        # Should be called twice
        assert mock_create_parallel.call_count == 2
        
        # Old environments should be closed
        for env in mock_envs1.values():
            env.close.assert_called_once()
    
    def test_cleanup_environments(self):
        """Test environment cleanup."""
        # Setup mock environments
        mock_envs = {
            'Kinematics': Mock(),
            'OccupancyGrid': Mock(),
            'GrayscaleObservation': Mock()
        }
        self.collector._environments = mock_envs
        self.collector._current_scenario = self.test_scenario
        
        # Cleanup
        self.collector._cleanup_environments()
        
        # Verify cleanup
        for env in mock_envs.values():
            env.close.assert_called_once()
        
        assert self.collector._environments == {}
        assert self.collector._current_scenario is None
    
    def test_cleanup_environments_with_errors(self):
        """Test environment cleanup with close errors."""
        # Setup mock environments with close errors
        mock_env1 = Mock()
        mock_env2 = Mock()
        mock_env2.close.side_effect = Exception("Close failed")
        mock_env3 = Mock()
        
        mock_envs = {
            'Kinematics': mock_env1,
            'OccupancyGrid': mock_env2,
            'GrayscaleObservation': mock_env3
        }
        self.collector._environments = mock_envs
        
        # Cleanup should not raise exception
        self.collector._cleanup_environments()
        
        # All close methods should be called
        mock_env1.close.assert_called_once()
        mock_env2.close.assert_called_once()
        mock_env3.close.assert_called_once()
        
        assert self.collector._environments == {}
    
    def test_reset_parallel_envs_success(self):
        """Test successful parallel environment reset."""
        # Setup mock environments
        mock_envs = {}
        for obs_type in ['Kinematics', 'OccupancyGrid', 'GrayscaleObservation']:
            mock_env = Mock()
            mock_env.reset.return_value = ([f"obs_{obs_type}_agent_0", f"obs_{obs_type}_agent_1"], 
                                         {"info": f"info_{obs_type}"})
            mock_envs[obs_type] = mock_env
        
        self.collector._environments = mock_envs
        
        # Reset environments
        observations = self.collector.reset_parallel_envs(self.test_seed)
        
        # Verify reset calls
        for obs_type, env in mock_envs.items():
            env.reset.assert_called_once_with(seed=self.test_seed)
        
        # Verify observations structure
        assert len(observations) == 3
        for obs_type in ['Kinematics', 'OccupancyGrid', 'GrayscaleObservation']:
            assert obs_type in observations
            assert 'observation' in observations[obs_type]
            assert 'info' in observations[obs_type]
            assert observations[obs_type]['observation'] == [f"obs_{obs_type}_agent_0", f"obs_{obs_type}_agent_1"]
            assert observations[obs_type]['info'] == {"info": f"info_{obs_type}"}
        
        assert self.collector._current_seed == self.test_seed
    
    def test_reset_parallel_envs_no_environments(self):
        """Test reset when no environments are set up."""
        with pytest.raises(RuntimeError, match="Environments not set up"):
            self.collector.reset_parallel_envs(self.test_seed)
    
    def test_reset_parallel_envs_failure(self):
        """Test reset failure handling."""
        # Setup mock environment that fails on reset
        mock_env = Mock()
        mock_env.reset.side_effect = Exception("Reset failed")
        self.collector._environments = {'Kinematics': mock_env}
        
        with pytest.raises(RuntimeError, match="Environment reset failed"):
            self.collector.reset_parallel_envs(self.test_seed)
    
    def test_step_parallel_envs_success(self):
        """Test successful parallel environment stepping."""
        actions = (0, 1)  # Actions for 2 agents
        
        # Setup mock environments
        mock_envs = {}
        for obs_type in ['Kinematics', 'OccupancyGrid', 'GrayscaleObservation']:
            mock_env = Mock()
            mock_env.step.return_value = (
                [f"obs_{obs_type}_agent_0", f"obs_{obs_type}_agent_1"],  # observations
                1.0,  # reward
                False,  # terminated
                False,  # truncated
                {"info": f"info_{obs_type}"}  # info
            )
            mock_envs[obs_type] = mock_env
        
        self.collector._environments = mock_envs
        
        # Step environments
        step_results = self.collector.step_parallel_envs(actions)
        
        # Verify step calls
        for obs_type, env in mock_envs.items():
            env.step.assert_called_once_with(actions)
        
        # Verify step results structure
        assert len(step_results) == 3
        for obs_type in ['Kinematics', 'OccupancyGrid', 'GrayscaleObservation']:
            assert obs_type in step_results
            result = step_results[obs_type]
            assert 'observation' in result
            assert 'reward' in result
            assert 'terminated' in result
            assert 'truncated' in result
            assert 'info' in result
            assert result['observation'] == [f"obs_{obs_type}_agent_0", f"obs_{obs_type}_agent_1"]
            assert result['reward'] == 1.0
            assert result['terminated'] is False
            assert result['truncated'] is False
            assert result['info'] == {"info": f"info_{obs_type}"}
    
    def test_step_parallel_envs_no_environments(self):
        """Test step when no environments are set up."""
        actions = (0, 1)
        
        with pytest.raises(RuntimeError, match="Environments not set up"):
            self.collector.step_parallel_envs(actions)
    
    def test_step_parallel_envs_wrong_action_count(self):
        """Test step with wrong number of actions."""
        # Setup mock environment
        self.collector._environments = {'Kinematics': Mock()}
        
        # Wrong number of actions
        actions = (0,)  # Only 1 action for 2 agents
        
        with pytest.raises(ValueError, match="Expected 2 actions, got 1"):
            self.collector.step_parallel_envs(actions)
    
    def test_step_parallel_envs_failure(self):
        """Test step failure handling."""
        actions = (0, 1)
        
        # Setup mock environment that fails on step
        mock_env = Mock()
        mock_env.step.side_effect = Exception("Step failed")
        self.collector._environments = {'Kinematics': mock_env}
        
        with pytest.raises(RuntimeError, match="Environment step failed"):
            self.collector.step_parallel_envs(actions)
    
    def test_verify_synchronization_success(self):
        """Test successful synchronization verification."""
        step_results = {
            'Kinematics': {
                'reward': 1.0,
                'terminated': False,
                'truncated': False
            },
            'OccupancyGrid': {
                'reward': 1.0,
                'terminated': False,
                'truncated': False
            },
            'GrayscaleObservation': {
                'reward': 1.0,
                'terminated': False,
                'truncated': False
            }
        }
        
        assert self.collector.verify_synchronization(step_results) is True
    
    def test_verify_synchronization_empty_results(self):
        """Test synchronization verification with empty results."""
        assert self.collector.verify_synchronization({}) is True
    
    def test_verify_synchronization_reward_mismatch(self):
        """Test synchronization verification with reward mismatch."""
        step_results = {
            'Kinematics': {
                'reward': 1.0,
                'terminated': False,
                'truncated': False
            },
            'OccupancyGrid': {
                'reward': 2.0,  # Different reward
                'terminated': False,
                'truncated': False
            }
        }
        
        assert self.collector.verify_synchronization(step_results) is False
    
    def test_verify_synchronization_terminated_mismatch(self):
        """Test synchronization verification with terminated mismatch."""
        step_results = {
            'Kinematics': {
                'reward': 1.0,
                'terminated': False,
                'truncated': False
            },
            'OccupancyGrid': {
                'reward': 1.0,
                'terminated': True,  # Different terminated
                'truncated': False
            }
        }
        
        assert self.collector.verify_synchronization(step_results) is False
    
    def test_verify_synchronization_truncated_mismatch(self):
        """Test synchronization verification with truncated mismatch."""
        step_results = {
            'Kinematics': {
                'reward': 1.0,
                'terminated': False,
                'truncated': False
            },
            'OccupancyGrid': {
                'reward': 1.0,
                'terminated': False,
                'truncated': True  # Different truncated
            }
        }
        
        assert self.collector.verify_synchronization(step_results) is False
    
    @patch('numpy.random.randint')
    def test_sample_actions(self, mock_randint):
        """Test action sampling."""
        mock_randint.side_effect = [0, 3]  # Actions for 2 agents
        
        observations = {}  # Not used in random sampling
        actions = self.collector.sample_actions(observations)
        
        assert actions == (0, 3)
        assert len(actions) == self.n_agents
        
        # Verify random calls
        assert mock_randint.call_count == self.n_agents
        for call in mock_randint.call_args_list:
            assert call[0] == (0, 5)  # Range for DiscreteMetaAction
    
    def test_sample_actions_different_agent_count(self):
        """Test action sampling with different number of agents."""
        collector = SynchronizedCollector(n_agents=4)
        
        observations = {}
        actions = collector.sample_actions(observations)
        
        assert len(actions) == 4
        assert all(isinstance(action, (int, np.integer)) for action in actions)
        assert all(0 <= action < 5 for action in actions)
    
    @patch.object(SynchronizedCollector, '_collect_single_episode')
    @patch.object(SynchronizedCollector, '_setup_environments')
    def test_collect_episode_batch_success(self, mock_setup, mock_collect_single):
        """Test successful episode batch collection."""
        from highway_datacollection.collection.types import EpisodeData
        
        # Mock single episode collection
        mock_episode_data = EpisodeData(
            episode_id="test_episode",
            scenario=self.test_scenario,
            observations=[],
            actions=[],
            rewards=[],
            dones=[],
            infos=[],
            metadata={}
        )
        mock_collect_single.return_value = mock_episode_data
        
        # Collect batch
        result = self.collector.collect_episode_batch(self.test_scenario, 3, self.test_seed)
        
        # Verify setup was called
        mock_setup.assert_called_once_with(self.test_scenario)
        
        # Verify single episode collection was called for each episode
        assert mock_collect_single.call_count == 3
        
        # Verify result structure
        assert result.total_episodes == 3
        assert result.successful_episodes == 3
        assert result.failed_episodes == 0
        assert len(result.episodes) == 3
        assert len(result.errors) == 0
        assert result.collection_time > 0
    
    @patch.object(SynchronizedCollector, '_collect_single_episode')
    @patch.object(SynchronizedCollector, '_setup_environments')
    def test_collect_episode_batch_with_failures(self, mock_setup, mock_collect_single):
        """Test episode batch collection with some failures."""
        from highway_datacollection.collection.types import EpisodeData
        
        # Mock single episode collection with some failures
        mock_episode_data = EpisodeData(
            episode_id="test_episode",
            scenario=self.test_scenario,
            observations=[],
            actions=[],
            rewards=[],
            dones=[],
            infos=[],
            metadata={}
        )
        
        # First episode succeeds, second fails, third succeeds
        mock_collect_single.side_effect = [
            mock_episode_data,
            Exception("Collection failed"),
            mock_episode_data
        ]
        
        # Collect batch
        result = self.collector.collect_episode_batch(self.test_scenario, 3, self.test_seed)
        
        # Verify result structure
        assert result.total_episodes == 3
        assert result.successful_episodes == 2
        assert result.failed_episodes == 1
        assert len(result.episodes) == 2
        assert len(result.errors) == 1
        assert "Episode 2 failed" in result.errors[0]
    
    @patch('highway_datacollection.features.engine.FeatureDerivationEngine')
    @patch.object(SynchronizedCollector, 'sample_actions')
    @patch.object(SynchronizedCollector, 'step_parallel_envs')
    @patch.object(SynchronizedCollector, 'reset_parallel_envs')
    @patch.object(SynchronizedCollector, 'verify_synchronization')
    def test_collect_single_episode_success(self, mock_verify, mock_reset, mock_step, 
                                          mock_sample, mock_feature_engine):
        """Test successful single episode collection."""
        # Setup mocks
        mock_verify.return_value = True
        mock_reset.return_value = {
            'Kinematics': {'observation': [[1, 0, 0, 0, 0, 1, 0], [0, 5, 0, 0, 0, 1, 0]]},
            'OccupancyGrid': {'observation': [np.zeros((11, 11)), np.zeros((11, 11))]},
            'GrayscaleObservation': {'observation': [np.zeros((84, 84, 3)), np.zeros((84, 84, 3))]}
        }
        
        # Mock step results (episode terminates after 2 steps)
        step_results = [
            {
                'Kinematics': {
                    'observation': [[1, 1, 0, 1, 0, 1, 0], [0, 6, 0, 0, 0, 1, 0]],
                    'reward': 0.5,
                    'terminated': False,
                    'truncated': False,
                    'info': {}
                },
                'OccupancyGrid': {
                    'observation': [np.zeros((11, 11)), np.zeros((11, 11))],
                    'reward': 0.5,
                    'terminated': False,
                    'truncated': False,
                    'info': {}
                },
                'GrayscaleObservation': {
                    'observation': [np.zeros((84, 84, 3)), np.zeros((84, 84, 3))],
                    'reward': 0.5,
                    'terminated': False,
                    'truncated': False,
                    'info': {}
                }
            },
            {
                'Kinematics': {
                    'observation': [[1, 2, 0, 1, 0, 1, 0], [0, 7, 0, 0, 0, 1, 0]],
                    'reward': 1.0,
                    'terminated': True,  # Episode ends
                    'truncated': False,
                    'info': {}
                },
                'OccupancyGrid': {
                    'observation': [np.zeros((11, 11)), np.zeros((11, 11))],
                    'reward': 1.0,
                    'terminated': True,
                    'truncated': False,
                    'info': {}
                },
                'GrayscaleObservation': {
                    'observation': [np.zeros((84, 84, 3)), np.zeros((84, 84, 3))],
                    'reward': 1.0,
                    'terminated': True,
                    'truncated': False,
                    'info': {}
                }
            }
        ]
        mock_step.side_effect = step_results
        mock_sample.return_value = (0, 1)
        
        # Mock feature engine
        mock_engine_instance = Mock()
        mock_engine_instance.derive_kinematics_features.return_value = {'lane_position': 0.5}
        mock_engine_instance.calculate_ttc.return_value = 5.0
        mock_engine_instance.generate_language_summary.return_value = "Test summary"
        mock_engine_instance.estimate_traffic_metrics.return_value = {'traffic_density': 0.3}
        mock_feature_engine.return_value = mock_engine_instance
        
        # Setup environments
        self.collector._environments = {'Kinematics': Mock(), 'OccupancyGrid': Mock(), 'GrayscaleObservation': Mock()}
        
        # Collect single episode
        episode_data = self.collector._collect_single_episode(self.test_scenario, self.test_seed, 100, 0)
        
        # Verify episode data structure
        assert episode_data.episode_id.startswith(f"ep_{self.test_scenario}_{self.test_seed}")
        assert episode_data.scenario == self.test_scenario
        assert len(episode_data.observations) == 2  # Initial + 1 step (terminated after step 1)
        assert len(episode_data.actions) == 2
        assert len(episode_data.rewards) == 2
        assert len(episode_data.dones) == 2
        assert episode_data.dones[-1] is True  # Last step should be done
        assert episode_data.metadata['total_steps'] == 2
        assert episode_data.metadata['terminated_early'] is True
    
    @patch.object(SynchronizedCollector, 'verify_synchronization')
    def test_collect_single_episode_desynchronization_error(self, mock_verify):
        """Test single episode collection with desynchronization error."""
        # Setup environments
        self.collector._environments = {'Kinematics': Mock()}
        
        # Mock reset
        with patch.object(self.collector, 'reset_parallel_envs') as mock_reset:
            mock_reset.return_value = {
                'Kinematics': {'observation': [[1, 0, 0, 0, 0, 1, 0]]}
            }
            
            # Mock step and verify synchronization failure
            with patch.object(self.collector, 'step_parallel_envs') as mock_step:
                mock_step.return_value = {'Kinematics': {'reward': 1.0, 'terminated': False, 'truncated': False, 'info': {}}}
                mock_verify.return_value = False  # Desynchronization detected
                
                with patch.object(self.collector, 'sample_actions') as mock_sample:
                    mock_sample.return_value = (0,)
                    
                    with pytest.raises(RuntimeError, match="Environment desynchronization detected"):
                        self.collector._collect_single_episode(self.test_scenario, self.test_seed, 100, 0)
    
    def test_process_observations_structure(self):
        """Test observation processing structure."""
        from highway_datacollection.features.engine import FeatureDerivationEngine
        
        # Mock step results
        step_results = {
            'Kinematics': {
                'observation': [
                    [[1, 0, 0, 0, 0, 1, 0], [0, 5, 0, 0, 0, 1, 0]],  # Agent 0
                    [[1, 1, 0, 1, 0, 1, 0], [0, 6, 0, 0, 0, 1, 0]]   # Agent 1
                ]
            },
            'OccupancyGrid': {
                'observation': [
                    np.zeros((11, 11)),  # Agent 0
                    np.zeros((11, 11))   # Agent 1
                ]
            },
            'GrayscaleObservation': {
                'observation': [
                    np.zeros((84, 84, 3)),  # Agent 0
                    np.zeros((84, 84, 3))   # Agent 1
                ]
            }
        }
        
        feature_engine = FeatureDerivationEngine()
        
        # Process observations
        processed_obs = self.collector._process_observations(
            step_results, feature_engine, "test_episode", 0
        )
        
        # Verify structure
        assert len(processed_obs) == self.n_agents
        
        for agent_idx, obs in enumerate(processed_obs):
            assert obs['episode_id'] == "test_episode"
            assert obs['step'] == 0
            assert obs['agent_id'] == agent_idx
            assert 'kinematics_raw' in obs
            assert 'ttc' in obs
            assert 'summary_text' in obs
            assert 'occupancy_blob' in obs
            assert 'occupancy_shape' in obs
            assert 'occupancy_dtype' in obs
            assert 'grayscale_blob' in obs
            assert 'grayscale_shape' in obs
            assert 'grayscale_dtype' in obs
    
    def test_destructor_cleanup(self):
        """Test that destructor cleans up environments."""
        # Setup mock environments
        mock_envs = {
            'Kinematics': Mock(),
            'OccupancyGrid': Mock()
        }
        self.collector._environments = mock_envs
        
        # Call destructor
        self.collector.__del__()
        
        # Verify cleanup
        for env in mock_envs.values():
            env.close.assert_called_once()


class TestSynchronizedCollectorIntegration:
    """Integration tests for SynchronizedCollector with real environments."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = SynchronizedCollector(n_agents=2)
        self.test_scenario = "free_flow"
        self.test_seed = 42
    
    @pytest.mark.integration
    def test_full_environment_lifecycle(self):
        """Test complete environment lifecycle with real environments."""
        # Setup environments
        self.collector._setup_environments(self.test_scenario)
        
        assert len(self.collector._environments) == 3
        assert self.collector._current_scenario == self.test_scenario
        
        # Reset environments
        observations = self.collector.reset_parallel_envs(self.test_seed)
        
        assert len(observations) == 3
        for obs_type in ['Kinematics', 'OccupancyGrid', 'GrayscaleObservation']:
            assert obs_type in observations
            assert 'observation' in observations[obs_type]
            assert 'info' in observations[obs_type]
            # Each observation should be a list with 2 agents
            assert isinstance(observations[obs_type]['observation'], (list, tuple))
            assert len(observations[obs_type]['observation']) == 2
        
        # Step environments
        actions = (0, 1)  # Valid actions for 2 agents
        step_results = self.collector.step_parallel_envs(actions)
        
        assert len(step_results) == 3
        for obs_type in ['Kinematics', 'OccupancyGrid', 'GrayscaleObservation']:
            assert obs_type in step_results
            result = step_results[obs_type]
            assert 'observation' in result
            assert 'reward' in result
            assert 'terminated' in result
            assert 'truncated' in result
            assert 'info' in result
            # Each observation should be a list with 2 agents
            assert isinstance(result['observation'], (list, tuple))
            assert len(result['observation']) == 2
        
        # Verify synchronization
        assert self.collector.verify_synchronization(step_results) is True
        
        # Cleanup
        self.collector._cleanup_environments()
    
    @pytest.mark.integration
    def test_seed_consistency(self):
        """Test that identical seeds produce identical initial observations."""
        # Setup environments
        self.collector._setup_environments(self.test_scenario)
        
        # Reset with same seed twice
        obs1 = self.collector.reset_parallel_envs(self.test_seed)
        obs2 = self.collector.reset_parallel_envs(self.test_seed)
        
        # Observations should be identical for Kinematics
        # (Other modalities might have slight differences due to rendering)
        kin_obs1 = obs1['Kinematics']['observation']
        kin_obs2 = obs2['Kinematics']['observation']
        
        # Convert to numpy arrays for comparison
        import numpy as np
        np.testing.assert_array_equal(np.array(kin_obs1), np.array(kin_obs2))
        
        # Cleanup
        self.collector._cleanup_environments()
    
    @pytest.mark.integration
    def test_action_synchronization(self):
        """Test that same actions produce synchronized results."""
        # Setup environments
        self.collector._setup_environments(self.test_scenario)
        
        # Reset environments
        self.collector.reset_parallel_envs(self.test_seed)
        
        # Step with same actions multiple times
        actions = (1, 1)  # IDLE for both agents
        
        for _ in range(5):
            step_results = self.collector.step_parallel_envs(actions)
            
            # Verify synchronization
            assert self.collector.verify_synchronization(step_results) is True
            
            # All environments should have same reward/termination status
            rewards = [result['reward'] for result in step_results.values()]
            terminated = [result['terminated'] for result in step_results.values()]
            truncated = [result['truncated'] for result in step_results.values()]
            
            assert len(set(rewards)) == 1, f"Rewards not synchronized: {rewards}"
            assert len(set(terminated)) == 1, f"Terminated not synchronized: {terminated}"
            assert len(set(truncated)) == 1, f"Truncated not synchronized: {truncated}"
        
        # Cleanup
        self.collector._cleanup_environments()
    
    @pytest.mark.integration
    def test_complete_episode_collection_workflow(self):
        """Test complete episode data collection workflow with real environments."""
        # Test parameters
        episodes = 2
        max_steps = 10
        
        # Collect episode batch
        result = self.collector.collect_episode_batch(
            self.test_scenario, episodes, self.test_seed, max_steps
        )
        
        # Verify collection result structure
        assert result.total_episodes == episodes
        assert result.successful_episodes <= episodes
        assert result.failed_episodes >= 0
        assert result.successful_episodes + result.failed_episodes == episodes
        assert len(result.episodes) == result.successful_episodes
        assert result.collection_time > 0
        
        # Verify episode data structure for successful episodes
        for episode_data in result.episodes:
            # Basic episode metadata
            assert episode_data.episode_id.startswith(f"ep_{self.test_scenario}")
            assert episode_data.scenario == self.test_scenario
            assert episode_data.metadata['n_agents'] == 2
            assert episode_data.metadata['total_steps'] > 0
            assert episode_data.metadata['total_steps'] <= max_steps
            assert episode_data.metadata['seed'] >= self.test_seed
            assert episode_data.metadata['max_steps'] == max_steps
            
            # Episode data consistency
            num_steps = episode_data.metadata['total_steps']
            assert len(episode_data.observations) == num_steps
            assert len(episode_data.actions) == num_steps
            assert len(episode_data.rewards) == num_steps
            assert len(episode_data.dones) == num_steps
            assert len(episode_data.infos) == num_steps
            
            # Verify termination condition
            if episode_data.dones:
                assert episode_data.dones[-1] is True or num_steps == max_steps
            
            # Verify observation structure for each step
            for step_idx, step_observations in enumerate(episode_data.observations):
                assert len(step_observations) == 2  # 2 agents
                
                for agent_idx, obs in enumerate(step_observations):
                    # Basic observation structure
                    assert obs['episode_id'] == episode_data.episode_id
                    assert obs['step'] == step_idx
                    assert obs['agent_id'] == agent_idx
                    
                    # Kinematics data
                    assert 'kinematics_raw' in obs
                    assert isinstance(obs['kinematics_raw'], list)
                    
                    # Derived features
                    assert 'ttc' in obs
                    assert isinstance(obs['ttc'], (int, float, np.number))
                    assert 'summary_text' in obs
                    assert isinstance(obs['summary_text'], str)
                    
                    # Binary array data
                    assert 'occupancy_blob' in obs
                    assert 'occupancy_shape' in obs
                    assert 'occupancy_dtype' in obs
                    assert 'grayscale_blob' in obs
                    assert 'grayscale_shape' in obs
                    assert 'grayscale_dtype' in obs
                    
                    # Verify binary data can be decoded
                    from highway_datacollection.storage.encoders import BinaryArrayEncoder
                    encoder = BinaryArrayEncoder()
                    
                    # Test occupancy grid decoding
                    occ_array = encoder.decode(
                        obs['occupancy_blob'],
                        tuple(obs['occupancy_shape']),
                        obs['occupancy_dtype']
                    )
                    assert occ_array.shape == tuple(obs['occupancy_shape'])
                    assert str(occ_array.dtype) == obs['occupancy_dtype']
                    
                    # Test grayscale image decoding
                    gray_array = encoder.decode(
                        obs['grayscale_blob'],
                        tuple(obs['grayscale_shape']),
                        obs['grayscale_dtype']
                    )
                    assert gray_array.shape == tuple(obs['grayscale_shape'])
                    assert str(gray_array.dtype) == obs['grayscale_dtype']
            
            # Verify action structure
            for action_tuple in episode_data.actions:
                assert isinstance(action_tuple, tuple)
                assert len(action_tuple) == 2  # 2 agents
                for action in action_tuple:
                    assert isinstance(action, (int, np.integer))
                    assert 0 <= action < 5  # Valid HighwayEnv actions
        
        # Cleanup
        self.collector._cleanup_environments()
    
    @pytest.mark.integration
    def test_episode_collection_with_different_parameters(self):
        """Test episode collection with different parameter configurations."""
        test_cases = [
            {"episodes": 1, "max_steps": 5, "n_agents": 1},
            {"episodes": 3, "max_steps": 20, "n_agents": 3},
            {"episodes": 2, "max_steps": 15, "n_agents": 2},
        ]
        
        for case in test_cases:
            collector = SynchronizedCollector(n_agents=case["n_agents"])
            
            result = collector.collect_episode_batch(
                self.test_scenario, 
                case["episodes"], 
                self.test_seed, 
                case["max_steps"]
            )
            
            # Verify basic result structure
            assert result.total_episodes == case["episodes"]
            assert result.successful_episodes <= case["episodes"]
            
            # Verify episode structure for successful episodes
            for episode_data in result.episodes:
                assert episode_data.metadata['n_agents'] == case["n_agents"]
                assert episode_data.metadata['max_steps'] == case["max_steps"]
                
                # Verify observation structure matches agent count
                for step_observations in episode_data.observations:
                    assert len(step_observations) == case["n_agents"]
                
                # Verify action structure matches agent count
                for action_tuple in episode_data.actions:
                    assert len(action_tuple) == case["n_agents"]
            
            collector._cleanup_environments()
    
    @pytest.mark.integration
    def test_episode_collection_deterministic_behavior(self):
        """Test that episode collection produces deterministic results with same seeds."""
        episodes = 2
        max_steps = 10
        
        # Collect episodes twice with same seed
        result1 = self.collector.collect_episode_batch(
            self.test_scenario, episodes, self.test_seed, max_steps
        )
        
        # Reset collector and collect again
        self.collector._cleanup_environments()
        result2 = self.collector.collect_episode_batch(
            self.test_scenario, episodes, self.test_seed, max_steps
        )
        
        # Results should be identical
        assert result1.successful_episodes == result2.successful_episodes
        assert len(result1.episodes) == len(result2.episodes)
        
        # Compare episode data (at least basic structure should match)
        for ep1, ep2 in zip(result1.episodes, result2.episodes):
            assert ep1.metadata['total_steps'] == ep2.metadata['total_steps']
            assert len(ep1.observations) == len(ep2.observations)
            assert len(ep1.actions) == len(ep2.actions)
            
            # Actions should be identical (deterministic sampling with same seed)
            for actions1, actions2 in zip(ep1.actions, ep2.actions):
                assert actions1 == actions2
        
        # Cleanup
        self.collector._cleanup_environments()