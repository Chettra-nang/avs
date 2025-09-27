"""
Action sampling strategies for data collection.

This module implements the strategy pattern for action sampling, allowing
flexible integration of random actions, trained policies, or custom sampling
strategies while maintaining deterministic behavior through seed management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ActionSampler(ABC):
    """
    Abstract base class for action sampling strategies.
    
    This interface allows pluggable action sampling strategies that can be
    used during data collection. Implementations must maintain deterministic
    behavior when provided with seeds.
    """
    
    @abstractmethod
    def sample_actions(self, observations: Dict[str, Any], n_agents: int, 
                      step: int = 0, episode_id: str = "") -> Tuple[int, ...]:
        """
        Sample actions for all agents given current observations.
        
        Args:
            observations: Current observations from all modalities
            n_agents: Number of agents to sample actions for
            step: Current step in the episode (for context)
            episode_id: Current episode ID (for context)
            
        Returns:
            Tuple of actions for each agent
        """
        pass
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the action sampler state.
        
        Args:
            seed: Random seed for deterministic behavior
        """
        pass
    
    def get_action_space_size(self) -> int:
        """
        Get the size of the action space.
        
        Returns:
            Number of possible actions (default: 5 for HighwayEnv DiscreteMetaAction)
        """
        return 5


class RandomActionSampler(ActionSampler):
    """
    Random action sampling strategy.
    
    Samples actions uniformly at random from the action space.
    Maintains deterministic behavior through seed management.
    """
    
    def __init__(self, action_space_size: int = 5, seed: Optional[int] = None):
        """
        Initialize random action sampler.
        
        Args:
            action_space_size: Size of the action space
            seed: Initial random seed
        """
        self._action_space_size = action_space_size
        self._rng = np.random.Generator(np.random.PCG64())
        if seed is not None:
            self.reset(seed)
        
        logger.info(f"Initialized RandomActionSampler with action space size {action_space_size}")
    
    def sample_actions(self, observations: Dict[str, Any], n_agents: int, 
                      step: int = 0, episode_id: str = "") -> Tuple[int, ...]:
        """
        Sample random actions for all agents.
        
        Args:
            observations: Current observations (not used for random sampling)
            n_agents: Number of agents to sample actions for
            step: Current step (not used for random sampling)
            episode_id: Episode ID (not used for random sampling)
            
        Returns:
            Tuple of random actions for each agent
        """
        actions = tuple(
            int(self._rng.integers(0, self._action_space_size))  # Convert to Python int
            for _ in range(n_agents)
        )
        
        logger.debug(f"Sampled random actions: {actions}")
        return actions
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the random number generator.
        
        Args:
            seed: Random seed for deterministic behavior
        """
        if seed is not None:
            # Use seed + offset to avoid correlation with environment seeds
            self._rng = np.random.Generator(np.random.PCG64(seed + 1000))
            logger.debug(f"Reset RandomActionSampler with seed {seed}")
    
    def get_action_space_size(self) -> int:
        """Get the action space size."""
        return self._action_space_size


class PolicyActionSampler(ActionSampler):
    """
    Policy-based action sampling strategy.
    
    Uses a trained policy to sample actions based on observations.
    Supports both deterministic and stochastic policies with seed management.
    """
    
    def __init__(self, policy: Any, observation_key: str = "Kinematics", 
                 deterministic: bool = True, seed: Optional[int] = None):
        """
        Initialize policy action sampler.
        
        Args:
            policy: Trained policy object (must have predict method)
            observation_key: Which observation modality to use for policy input
            deterministic: Whether to use deterministic policy actions
            seed: Initial random seed for stochastic policies
        """
        self._policy = policy
        self._observation_key = observation_key
        self._deterministic = deterministic
        self._rng = np.random.Generator(np.random.PCG64())
        
        if seed is not None:
            self.reset(seed)
        
        # Validate policy interface
        if not hasattr(policy, 'predict'):
            raise ValueError("Policy must have a 'predict' method")
        
        logger.info(f"Initialized PolicyActionSampler with observation key '{observation_key}', "
                   f"deterministic={deterministic}")
    
    def sample_actions(self, observations: Dict[str, Any], n_agents: int, 
                      step: int = 0, episode_id: str = "") -> Tuple[int, ...]:
        """
        Sample actions using the trained policy.
        
        Args:
            observations: Current observations from all modalities
            n_agents: Number of agents to sample actions for
            step: Current step in the episode
            episode_id: Current episode ID
            
        Returns:
            Tuple of policy-generated actions for each agent
        """
        # Extract observations for the specified modality
        if self._observation_key not in observations:
            logger.warning(f"Observation key '{self._observation_key}' not found in observations. "
                          f"Available keys: {list(observations.keys())}")
            # Fallback to random actions
            fallback_sampler = RandomActionSampler()
            return fallback_sampler.sample_actions(observations, n_agents, step, episode_id)
        
        obs_data = observations[self._observation_key].get('observation', [])
        
        try:
            # Handle different observation formats
            if isinstance(obs_data, (list, tuple)):
                # Multi-agent observations
                actions = []
                for agent_idx in range(n_agents):
                    if agent_idx < len(obs_data):
                        agent_obs = np.array(obs_data[agent_idx])
                    else:
                        # Use last available observation if not enough agents
                        agent_obs = np.array(obs_data[-1]) if obs_data else np.zeros((5, 5))
                    
                    # Get action from policy
                    action, _ = self._policy.predict(agent_obs, deterministic=self._deterministic)
                    actions.append(int(action))
                
                actions_tuple = tuple(actions)
            else:
                # Single observation - replicate for all agents
                obs_array = np.array(obs_data)
                action, _ = self._policy.predict(obs_array, deterministic=self._deterministic)
                actions_tuple = tuple(int(action) for _ in range(n_agents))
            
            logger.debug(f"Sampled policy actions: {actions_tuple}")
            return actions_tuple
            
        except Exception as e:
            logger.error(f"Policy prediction failed: {e}. Falling back to random actions.")
            # Fallback to random actions on policy failure
            fallback_sampler = RandomActionSampler()
            return fallback_sampler.sample_actions(observations, n_agents, step, episode_id)
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the policy sampler state.
        
        Args:
            seed: Random seed for stochastic policies
        """
        if seed is not None and not self._deterministic:
            # Set seed for stochastic policy sampling
            self._rng = np.random.Generator(np.random.PCG64(seed + 2000))
            
            # If policy supports seeding, set it
            if hasattr(self._policy, 'set_random_seed'):
                self._policy.set_random_seed(seed + 2000)
            
            logger.debug(f"Reset PolicyActionSampler with seed {seed}")
    
    def get_policy(self) -> Any:
        """Get the underlying policy object."""
        return self._policy


class HybridActionSampler(ActionSampler):
    """
    Hybrid action sampling strategy.
    
    Combines multiple sampling strategies, allowing different agents to use
    different action sampling approaches or switching between strategies
    based on conditions.
    """
    
    def __init__(self, samplers: Dict[int, ActionSampler], 
                 default_sampler: Optional[ActionSampler] = None):
        """
        Initialize hybrid action sampler.
        
        Args:
            samplers: Dictionary mapping agent indices to their action samplers
            default_sampler: Default sampler for agents not in samplers dict
        """
        self._samplers = samplers
        self._default_sampler = default_sampler or RandomActionSampler()
        
        logger.info(f"Initialized HybridActionSampler with {len(samplers)} agent-specific samplers")
    
    def sample_actions(self, observations: Dict[str, Any], n_agents: int, 
                      step: int = 0, episode_id: str = "") -> Tuple[int, ...]:
        """
        Sample actions using agent-specific samplers.
        
        Args:
            observations: Current observations from all modalities
            n_agents: Number of agents to sample actions for
            step: Current step in the episode
            episode_id: Current episode ID
            
        Returns:
            Tuple of actions from respective samplers for each agent
        """
        actions = []
        
        for agent_idx in range(n_agents):
            # Get sampler for this agent
            sampler = self._samplers.get(agent_idx, self._default_sampler)
            
            # Sample single action for this agent
            agent_actions = sampler.sample_actions(observations, 1, step, episode_id)
            actions.append(agent_actions[0])
        
        actions_tuple = tuple(actions)
        logger.debug(f"Sampled hybrid actions: {actions_tuple}")
        return actions_tuple
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset all samplers.
        
        Args:
            seed: Random seed for deterministic behavior
        """
        # Reset all agent-specific samplers
        for agent_idx, sampler in self._samplers.items():
            # Use different seed offsets for each agent
            agent_seed = seed + agent_idx * 100 if seed is not None else None
            sampler.reset(agent_seed)
        
        # Reset default sampler
        default_seed = seed + 10000 if seed is not None else None
        self._default_sampler.reset(default_seed)
        
        if seed is not None:
            logger.debug(f"Reset HybridActionSampler with seed {seed}")
    
    def add_agent_sampler(self, agent_idx: int, sampler: ActionSampler) -> None:
        """
        Add or update sampler for a specific agent.
        
        Args:
            agent_idx: Agent index
            sampler: Action sampler for this agent
        """
        self._samplers[agent_idx] = sampler
        logger.info(f"Added sampler for agent {agent_idx}: {type(sampler).__name__}")
    
    def get_agent_sampler(self, agent_idx: int) -> ActionSampler:
        """
        Get sampler for a specific agent.
        
        Args:
            agent_idx: Agent index
            
        Returns:
            Action sampler for the agent
        """
        return self._samplers.get(agent_idx, self._default_sampler)