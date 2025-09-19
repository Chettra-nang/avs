"""
Unit tests for action sampling strategies.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from highway_datacollection.collection.action_samplers import (
    ActionSampler,
    RandomActionSampler,
    PolicyActionSampler,
    HybridActionSampler
)


class TestRandomActionSampler:
    """Test cases for RandomActionSampler."""
    
    def test_initialization(self):
        """Test RandomActionSampler initialization."""
        sampler = RandomActionSampler(action_space_size=5, seed=42)
        assert sampler.get_action_space_size() == 5
    
    def test_sample_actions_deterministic(self):
        """Test that actions are deterministic with same seed."""
        sampler1 = RandomActionSampler(seed=42)
        sampler2 = RandomActionSampler(seed=42)
        
        observations = {"Kinematics": {"observation": []}}
        
        actions1 = sampler1.sample_actions(observations, n_agents=3)
        actions2 = sampler2.sample_actions(observations, n_agents=3)
        
        assert actions1 == actions2
        assert len(actions1) == 3
        assert all(0 <= action < 5 for action in actions1)
    
    def test_sample_actions_different_seeds(self):
        """Test that different seeds produce different actions."""
        sampler1 = RandomActionSampler(seed=42)
        sampler2 = RandomActionSampler(seed=43)
        
        observations = {"Kinematics": {"observation": []}}
        
        actions1 = sampler1.sample_actions(observations, n_agents=3)
        actions2 = sampler2.sample_actions(observations, n_agents=3)
        
        # With high probability, actions should be different
        assert actions1 != actions2
    
    def test_reset_with_seed(self):
        """Test reset functionality with seed."""
        sampler = RandomActionSampler()
        observations = {"Kinematics": {"observation": []}}
        
        # Sample some actions
        sampler.reset(42)
        actions1 = sampler.sample_actions(observations, n_agents=2)
        
        # Reset with same seed and sample again
        sampler.reset(42)
        actions2 = sampler.sample_actions(observations, n_agents=2)
        
        assert actions1 == actions2
    
    def test_action_space_bounds(self):
        """Test that actions are within expected bounds."""
        sampler = RandomActionSampler(action_space_size=3, seed=42)
        observations = {"Kinematics": {"observation": []}}
        
        # Sample many actions to test bounds
        for _ in range(100):
            actions = sampler.sample_actions(observations, n_agents=2)
            assert all(0 <= action < 3 for action in actions)


class TestPolicyActionSampler:
    """Test cases for PolicyActionSampler."""
    
    def test_initialization_valid_policy(self):
        """Test PolicyActionSampler initialization with valid policy."""
        mock_policy = Mock()
        mock_policy.predict = Mock(return_value=(1, None))
        
        sampler = PolicyActionSampler(mock_policy, observation_key="Kinematics")
        assert sampler._policy == mock_policy
        assert sampler._observation_key == "Kinematics"
        assert sampler._deterministic == True
    
    def test_initialization_invalid_policy(self):
        """Test PolicyActionSampler initialization with invalid policy."""
        invalid_policy = Mock()
        # Remove predict method
        del invalid_policy.predict
        
        with pytest.raises(ValueError, match="Policy must have a 'predict' method"):
            PolicyActionSampler(invalid_policy)
    
    def test_sample_actions_single_agent(self):
        """Test policy action sampling for single agent."""
        mock_policy = Mock()
        mock_policy.predict = Mock(return_value=(2, None))
        
        sampler = PolicyActionSampler(mock_policy, observation_key="Kinematics")
        
        observations = {
            "Kinematics": {
                "observation": np.array([[1, 2, 3, 4, 5]])
            }
        }
        
        actions = sampler.sample_actions(observations, n_agents=1)
        
        assert actions == (2,)
        mock_policy.predict.assert_called_once()
    
    def test_sample_actions_multi_agent(self):
        """Test policy action sampling for multiple agents."""
        mock_policy = Mock()
        # Return different actions for different calls
        mock_policy.predict = Mock(side_effect=[(1, None), (3, None), (0, None)])
        
        sampler = PolicyActionSampler(mock_policy, observation_key="Kinematics")
        
        observations = {
            "Kinematics": {
                "observation": [
                    np.array([1, 2, 3, 4, 5]),
                    np.array([2, 3, 4, 5, 6]),
                    np.array([3, 4, 5, 6, 7])
                ]
            }
        }
        
        actions = sampler.sample_actions(observations, n_agents=3)
        
        assert actions == (1, 3, 0)
        assert mock_policy.predict.call_count == 3
    
    def test_sample_actions_missing_observation_key(self):
        """Test fallback to random actions when observation key is missing."""
        mock_policy = Mock()
        mock_policy.predict = Mock(return_value=(1, None))
        
        sampler = PolicyActionSampler(mock_policy, observation_key="MissingKey")
        
        observations = {
            "Kinematics": {
                "observation": np.array([[1, 2, 3, 4, 5]])
            }
        }
        
        actions = sampler.sample_actions(observations, n_agents=2)
        
        # Should fallback to random actions
        assert len(actions) == 2
        assert all(0 <= action < 5 for action in actions)
        # Policy should not be called
        mock_policy.predict.assert_not_called()
    
    def test_sample_actions_policy_failure(self):
        """Test fallback to random actions when policy fails."""
        mock_policy = Mock()
        mock_policy.predict = Mock(side_effect=Exception("Policy failed"))
        
        sampler = PolicyActionSampler(mock_policy, observation_key="Kinematics")
        
        observations = {
            "Kinematics": {
                "observation": np.array([[1, 2, 3, 4, 5]])
            }
        }
        
        actions = sampler.sample_actions(observations, n_agents=2)
        
        # Should fallback to random actions
        assert len(actions) == 2
        assert all(0 <= action < 5 for action in actions)
    
    def test_reset_with_seed(self):
        """Test reset functionality for policy sampler."""
        mock_policy = Mock()
        mock_policy.predict = Mock(return_value=(1, None))
        mock_policy.set_random_seed = Mock()
        
        sampler = PolicyActionSampler(mock_policy, deterministic=False)
        sampler.reset(42)
        
        # Should call policy's set_random_seed if available
        mock_policy.set_random_seed.assert_called_once_with(2042)  # seed + 2000
    
    def test_get_policy(self):
        """Test getting the underlying policy object."""
        mock_policy = Mock()
        mock_policy.predict = Mock(return_value=(1, None))
        
        sampler = PolicyActionSampler(mock_policy)
        assert sampler.get_policy() == mock_policy


class TestHybridActionSampler:
    """Test cases for HybridActionSampler."""
    
    def test_initialization(self):
        """Test HybridActionSampler initialization."""
        sampler1 = RandomActionSampler(seed=42)
        sampler2 = RandomActionSampler(seed=43)
        default_sampler = RandomActionSampler(seed=44)
        
        hybrid = HybridActionSampler(
            samplers={0: sampler1, 1: sampler2},
            default_sampler=default_sampler
        )
        
        assert hybrid._samplers[0] == sampler1
        assert hybrid._samplers[1] == sampler2
        assert hybrid._default_sampler == default_sampler
    
    def test_sample_actions_agent_specific(self):
        """Test sampling with agent-specific samplers."""
        # Create mock samplers that return predictable actions
        sampler1 = Mock(spec=ActionSampler)
        sampler1.sample_actions = Mock(return_value=(1,))
        
        sampler2 = Mock(spec=ActionSampler)
        sampler2.sample_actions = Mock(return_value=(2,))
        
        default_sampler = Mock(spec=ActionSampler)
        default_sampler.sample_actions = Mock(return_value=(3,))
        
        hybrid = HybridActionSampler(
            samplers={0: sampler1, 1: sampler2},
            default_sampler=default_sampler
        )
        
        observations = {"Kinematics": {"observation": []}}
        actions = hybrid.sample_actions(observations, n_agents=3)
        
        # Agent 0 uses sampler1, agent 1 uses sampler2, agent 2 uses default
        assert actions == (1, 2, 3)
        
        # Verify each sampler was called correctly
        sampler1.sample_actions.assert_called_once_with(observations, 1, 0, "")
        sampler2.sample_actions.assert_called_once_with(observations, 1, 0, "")
        default_sampler.sample_actions.assert_called_once_with(observations, 1, 0, "")
    
    def test_reset_all_samplers(self):
        """Test that reset calls all samplers."""
        sampler1 = Mock(spec=ActionSampler)
        sampler2 = Mock(spec=ActionSampler)
        default_sampler = Mock(spec=ActionSampler)
        
        hybrid = HybridActionSampler(
            samplers={0: sampler1, 2: sampler2},
            default_sampler=default_sampler
        )
        
        hybrid.reset(42)
        
        # Verify all samplers were reset with appropriate seeds
        sampler1.reset.assert_called_once_with(42)  # agent 0: seed + 0 * 100
        sampler2.reset.assert_called_once_with(242)  # agent 2: seed + 2 * 100
        default_sampler.reset.assert_called_once_with(10042)  # seed + 10000
    
    def test_add_agent_sampler(self):
        """Test adding a new agent sampler."""
        hybrid = HybridActionSampler(samplers={})
        
        new_sampler = Mock(spec=ActionSampler)
        hybrid.add_agent_sampler(1, new_sampler)
        
        assert hybrid._samplers[1] == new_sampler
    
    def test_get_agent_sampler(self):
        """Test getting agent-specific samplers."""
        sampler1 = Mock(spec=ActionSampler)
        default_sampler = Mock(spec=ActionSampler)
        
        hybrid = HybridActionSampler(
            samplers={0: sampler1},
            default_sampler=default_sampler
        )
        
        assert hybrid.get_agent_sampler(0) == sampler1
        assert hybrid.get_agent_sampler(1) == default_sampler  # Uses default


class TestActionSamplerIntegration:
    """Integration tests for action samplers."""
    
    def test_deterministic_behavior_across_resets(self):
        """Test that all samplers maintain deterministic behavior across resets."""
        samplers = [
            RandomActionSampler(seed=42),
            # PolicyActionSampler would need a real policy for this test
        ]
        
        observations = {"Kinematics": {"observation": []}}
        
        for sampler in samplers:
            # Sample actions
            sampler.reset(42)
            actions1 = sampler.sample_actions(observations, n_agents=3)
            
            # Reset and sample again
            sampler.reset(42)
            actions2 = sampler.sample_actions(observations, n_agents=3)
            
            assert actions1 == actions2, f"Sampler {type(sampler).__name__} not deterministic"
    
    def test_action_bounds_all_samplers(self):
        """Test that all samplers respect action space bounds."""
        samplers = [
            RandomActionSampler(action_space_size=4, seed=42),
        ]
        
        observations = {"Kinematics": {"observation": []}}
        
        for sampler in samplers:
            for _ in range(10):  # Test multiple samples
                actions = sampler.sample_actions(observations, n_agents=2)
                assert all(0 <= action < 4 for action in actions), \
                    f"Sampler {type(sampler).__name__} produced out-of-bounds actions"


if __name__ == "__main__":
    pytest.main([__file__])