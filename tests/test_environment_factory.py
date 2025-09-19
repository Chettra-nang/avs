"""
Unit tests for MultiAgentEnvFactory.
"""

import pytest
import gymnasium as gym
from unittest.mock import Mock, patch

from highway_datacollection.environments.factory import MultiAgentEnvFactory
from highway_datacollection.scenarios.registry import ScenarioRegistry


class TestMultiAgentEnvFactory:
    """Test cases for MultiAgentEnvFactory class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = MultiAgentEnvFactory()
        self.test_scenario = "free_flow"
        self.test_obs_type = "Kinematics"
        self.test_n_agents = 2
    
    def test_init(self):
        """Test factory initialization."""
        factory = MultiAgentEnvFactory()
        assert hasattr(factory, '_scenario_registry')
        assert hasattr(factory, '_supported_obs_types')
        assert isinstance(factory._scenario_registry, ScenarioRegistry)
        assert len(factory._supported_obs_types) == 3
        assert "Kinematics" in factory._supported_obs_types
        assert "OccupancyGrid" in factory._supported_obs_types
        assert "GrayscaleObservation" in factory._supported_obs_types
    
    def test_get_supported_observation_types(self):
        """Test getting supported observation types."""
        obs_types = self.factory.get_supported_observation_types()
        assert isinstance(obs_types, list)
        assert len(obs_types) == 3
        assert "Kinematics" in obs_types
        assert "OccupancyGrid" in obs_types
        assert "GrayscaleObservation" in obs_types
        
        # Test that returned list is a copy
        obs_types.append("TestType")
        original_types = self.factory.get_supported_observation_types()
        assert "TestType" not in original_types
    
    def test_get_base_config_valid_scenario(self):
        """Test getting base configuration for valid scenario."""
        config = self.factory.get_base_config(self.test_scenario, self.test_n_agents)
        
        # Check required fields are present
        assert "lanes_count" in config
        assert "vehicles_count" in config
        assert "duration" in config
        assert "controlled_vehicles" in config
        # Action config is added during environment creation, not in base config
        
        # Check multi-agent configuration
        assert config["controlled_vehicles"] == self.test_n_agents
        
        # Check that configuration is consistent with scenario
        scenario_config = self.factory._scenario_registry.get_scenario_config(self.test_scenario)
        assert config["lanes_count"] == scenario_config["lanes_count"]
        assert config["vehicles_count"] == scenario_config["vehicles_count"]
        assert config["duration"] == scenario_config["duration"]
    
    def test_get_base_config_invalid_scenario(self):
        """Test getting base configuration for invalid scenario."""
        with pytest.raises(KeyError):
            self.factory.get_base_config("invalid_scenario", self.test_n_agents)
    
    def test_get_base_config_invalid_n_agents(self):
        """Test getting base configuration with invalid number of agents."""
        with pytest.raises(ValueError, match="Number of agents must be >= 1"):
            self.factory.get_base_config(self.test_scenario, 0)
        
        with pytest.raises(ValueError, match="Number of agents must be >= 1"):
            self.factory.get_base_config(self.test_scenario, -1)
    
    def test_get_base_config_consistency(self):
        """Test that base configuration is consistent across calls."""
        config1 = self.factory.get_base_config(self.test_scenario, self.test_n_agents)
        config2 = self.factory.get_base_config(self.test_scenario, self.test_n_agents)
        
        # Configurations should be identical
        assert config1 == config2
        
        # But should be separate objects (deep copy)
        config1["test_field"] = "test_value"
        assert "test_field" not in config2
    
    def test_get_base_config_different_agents(self):
        """Test base configuration with different number of agents."""
        config_2_agents = self.factory.get_base_config(self.test_scenario, 2)
        config_4_agents = self.factory.get_base_config(self.test_scenario, 4)
        
        assert config_2_agents["controlled_vehicles"] == 2
        assert config_4_agents["controlled_vehicles"] == 4
        
        # Other parameters should be identical
        config_2_copy = config_2_agents.copy()
        config_4_copy = config_4_agents.copy()
        del config_2_copy["controlled_vehicles"]
        del config_4_copy["controlled_vehicles"]
        assert config_2_copy == config_4_copy
    
    @patch('gymnasium.make')
    def test_create_env_valid_params(self, mock_gym_make):
        """Test creating environment with valid parameters."""
        mock_env = Mock()
        mock_gym_make.return_value = mock_env
        
        env = self.factory.create_env(self.test_scenario, self.test_obs_type, self.test_n_agents)
        
        # Check that gym.make was called
        mock_gym_make.assert_called_once()
        call_args = mock_gym_make.call_args
        assert call_args[0][0] == "highway-v0"
        
        # Check that configuration includes observation config
        # Configuration is passed through the 'config' parameter
        assert "config" in call_args[1]
        config = call_args[1]["config"]
        assert "observation" in config
        assert config["observation"]["type"] == "MultiAgentObservation"  # Multi-agent wrapper
        assert config["controlled_vehicles"] == self.test_n_agents
        
        assert env == mock_env
    
    def test_create_env_invalid_obs_type(self):
        """Test creating environment with invalid observation type."""
        with pytest.raises(ValueError, match="Unsupported observation type"):
            self.factory.create_env(self.test_scenario, "InvalidObsType", self.test_n_agents)
    
    def test_create_env_invalid_n_agents(self):
        """Test creating environment with invalid number of agents."""
        with pytest.raises(ValueError, match="Number of agents must be >= 1"):
            self.factory.create_env(self.test_scenario, self.test_obs_type, 0)
    
    def test_create_env_invalid_scenario(self):
        """Test creating environment with invalid scenario."""
        with pytest.raises(KeyError):
            self.factory.create_env("invalid_scenario", self.test_obs_type, self.test_n_agents)
    
    @patch('gymnasium.make')
    def test_create_env_observation_config_isolation(self, mock_gym_make):
        """Test that observation configurations are isolated between calls."""
        mock_env1 = Mock()
        mock_env2 = Mock()
        mock_gym_make.side_effect = [mock_env1, mock_env2]
        
        # Create two environments with different observation types
        self.factory.create_env(self.test_scenario, "Kinematics", self.test_n_agents)
        self.factory.create_env(self.test_scenario, "OccupancyGrid", self.test_n_agents)
        
        # Check that different observation configs were used
        call1_config = mock_gym_make.call_args_list[0][1]["config"]
        call2_config = mock_gym_make.call_args_list[1][1]["config"]
        
        # Both should be MultiAgentObservation but with different underlying types
        assert call1_config["observation"]["type"] == "MultiAgentObservation"
        assert call2_config["observation"]["type"] == "MultiAgentObservation"
        assert call1_config["observation"]["observation_config"]["type"] == "Kinematics"
        assert call2_config["observation"]["observation_config"]["type"] == "OccupancyGrid"
        
        # Base configurations should be identical except for observation
        call1_base = call1_config.copy()
        call2_base = call2_config.copy()
        del call1_base["observation"]
        del call2_base["observation"]
        assert call1_base == call2_base
    
    @patch('gymnasium.make')
    def test_create_parallel_envs(self, mock_gym_make):
        """Test creating parallel environments for all observation types."""
        mock_envs = [Mock(), Mock(), Mock()]
        mock_gym_make.side_effect = mock_envs
        
        parallel_envs = self.factory.create_parallel_envs(self.test_scenario, self.test_n_agents)
        
        # Check that all observation types are present
        assert len(parallel_envs) == 3
        assert "Kinematics" in parallel_envs
        assert "OccupancyGrid" in parallel_envs
        assert "GrayscaleObservation" in parallel_envs
        
        # Check that gym.make was called for each observation type
        assert mock_gym_make.call_count == 3
        
        # Check that each environment has correct observation type
        call_configs = [call[1]["config"] for call in mock_gym_make.call_args_list]
        obs_types_used = [config["observation"]["observation_config"]["type"] for config in call_configs]
        assert "Kinematics" in obs_types_used
        assert "OccupancyGrid" in obs_types_used
        assert "GrayscaleObservation" in obs_types_used
    
    @patch('gymnasium.make')
    def test_create_parallel_envs_base_config_consistency(self, mock_gym_make):
        """Test that parallel environments use consistent base configurations."""
        mock_envs = [Mock(), Mock(), Mock()]
        mock_gym_make.side_effect = mock_envs
        
        self.factory.create_parallel_envs(self.test_scenario, self.test_n_agents)
        
        # Extract base configurations (without observation)
        call_configs = [call[1]["config"] for call in mock_gym_make.call_args_list]
        base_configs = []
        for config in call_configs:
            base_config = config.copy()
            del base_config["observation"]
            base_configs.append(base_config)
        
        # All base configurations should be identical
        assert base_configs[0] == base_configs[1]
        assert base_configs[1] == base_configs[2]
    
    def test_validate_configuration_valid(self):
        """Test configuration validation with valid parameters."""
        assert self.factory.validate_configuration(
            self.test_scenario, self.test_obs_type, self.test_n_agents
        ) is True
    
    def test_validate_configuration_invalid_scenario(self):
        """Test configuration validation with invalid scenario."""
        assert self.factory.validate_configuration(
            "invalid_scenario", self.test_obs_type, self.test_n_agents
        ) is False
    
    def test_validate_configuration_invalid_obs_type(self):
        """Test configuration validation with invalid observation type."""
        assert self.factory.validate_configuration(
            self.test_scenario, "InvalidObsType", self.test_n_agents
        ) is False
    
    def test_validate_configuration_invalid_n_agents(self):
        """Test configuration validation with invalid number of agents."""
        assert self.factory.validate_configuration(
            self.test_scenario, self.test_obs_type, 0
        ) is False
        
        assert self.factory.validate_configuration(
            self.test_scenario, self.test_obs_type, -1
        ) is False
    
    def test_all_scenarios_supported(self):
        """Test that factory works with all available scenarios."""
        scenarios = self.factory._scenario_registry.list_scenarios()
        
        for scenario in scenarios:
            # Should not raise exceptions
            config = self.factory.get_base_config(scenario, 2)
            assert isinstance(config, dict)
            assert config["controlled_vehicles"] == 2
            
            # Validation should pass
            assert self.factory.validate_configuration(scenario, "Kinematics", 2) is True


class TestMultiAgentEnvFactoryIntegration:
    """Integration tests for MultiAgentEnvFactory with real highway-env."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = MultiAgentEnvFactory()
    
    @pytest.mark.integration
    def test_create_real_environment_kinematics(self):
        """Test creating real highway environment with Kinematics observation."""
        env = self.factory.create_env("free_flow", "Kinematics", 2)
        
        assert isinstance(env, gym.Env)
        
        # Test environment reset
        obs, info = env.reset()
        assert isinstance(obs, (list, tuple))
        assert len(obs) == 2  # Two agents
        
        # Test environment step
        actions = (0, 1)  # Valid actions for DiscreteMetaAction (tuple required for multi-agent)
        obs, rewards, terminated, truncated, info = env.step(actions)
        assert isinstance(obs, (list, tuple))
        assert len(obs) == 2
        # Highway-env returns a single reward even for multi-agent environments
        assert isinstance(rewards, (int, float))
        
        env.close()
    
    @pytest.mark.integration
    def test_create_real_parallel_environments(self):
        """Test creating real parallel environments."""
        parallel_envs = self.factory.create_parallel_envs("free_flow", 2)
        
        assert len(parallel_envs) == 3
        
        # Test that all environments can be reset and stepped
        for obs_type, env in parallel_envs.items():
            obs, info = env.reset(seed=42)
            assert isinstance(obs, (list, tuple))
            assert len(obs) == 2
            
            actions = (0, 1)  # Tuple required for multi-agent
            obs, rewards, terminated, truncated, info = env.step(actions)
            assert isinstance(obs, (list, tuple))
            assert len(obs) == 2
            
            env.close()
    
    @pytest.mark.integration
    def test_environment_synchronization_setup(self):
        """Test that parallel environments are set up for synchronization."""
        parallel_envs = self.factory.create_parallel_envs("free_flow", 2)
        
        # Reset all environments with same seed
        seed = 42
        observations = {}
        for obs_type, env in parallel_envs.items():
            obs, info = env.reset(seed=seed)
            observations[obs_type] = obs
        
        # All environments should have same number of agents
        for obs_type, obs in observations.items():
            assert len(obs) == 2, f"Environment {obs_type} has wrong number of agents"
        
        # Clean up
        for env in parallel_envs.values():
            env.close()