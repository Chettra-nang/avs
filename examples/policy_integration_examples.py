#!/usr/bin/env python3
"""
Policy integration examples for HighwayEnv Multi-Modal Data Collection.

This module demonstrates various approaches to integrating trained policies
and custom action sampling strategies into the data collection pipeline.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from highway_datacollection.collection.action_samplers import ActionSampler
from highway_datacollection.collection.collector import SynchronizedCollector
from highway_datacollection.environments.factory import MultiAgentEnvFactory
from highway_datacollection import ScenarioRegistry


class RuleBasedPolicy(ActionSampler):
    """
    Example rule-based policy for highway driving.
    
    This policy implements simple rule-based decision making for demonstration
    of how to integrate custom policies into the data collection system.
    """
    
    def __init__(self, policy_type: str = "safe_following"):
        """
        Initialize rule-based policy.
        
        Args:
            policy_type: Type of rule-based behavior
                - "safe_following": Conservative following behavior
                - "lane_changer": Aggressive lane changing
                - "speed_keeper": Maintains target speed
        """
        self.policy_type = policy_type
        self.step_count = 0
        
        # Policy parameters
        self.safe_ttc_threshold = 3.0
        self.target_speed = 25.0
        self.lane_change_probability = 0.1
    
    def sample_actions(self, observations: Dict[str, Any], n_agents: int, 
                      step: int = 0, episode_id: str = "") -> Tuple[int, ...]:
        """Sample actions using rule-based policy."""
        # Extract kinematics observations
        kin_obs = observations.get('kinematics', np.zeros((n_agents, 5, 5)))
        
        actions = []
        for agent_idx in range(n_agents):
            agent_obs = kin_obs[agent_idx] if len(kin_obs.shape) > 2 else kin_obs
            action = self._get_action(agent_obs, agent_idx)
            actions.append(action)
        
        self.step_count += 1
        return tuple(actions)
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the policy state."""
        self.step_count = 0
        if seed is not None:
            np.random.seed(seed)
    
    def _get_action(self, observation: np.ndarray, agent_idx: int) -> int:
        """Get action for a single agent based on rule-based policy."""
        if observation.shape[0] == 0:
            return 1  # IDLE
        
        ego_vehicle = observation[0]
        other_vehicles = observation[1:] if observation.shape[0] > 1 else np.zeros((0, 5))
        
        if self.policy_type == "safe_following":
            return self._safe_following_action(ego_vehicle, other_vehicles)
        elif self.policy_type == "lane_changer":
            return self._lane_changing_action(ego_vehicle, other_vehicles)
        elif self.policy_type == "speed_keeper":
            return self._speed_keeping_action(ego_vehicle, other_vehicles)
        else:
            return 1  # Default: IDLE
    
    def _safe_following_action(self, ego: np.ndarray, others: np.ndarray) -> int:
        """Conservative following behavior."""
        if ego[0] == 0:  # Not present
            return 1
        
        # Calculate TTC to lead vehicle
        ttc = self._calculate_ttc(ego, others)
        ego_speed = np.sqrt(ego[3]**2 + ego[4]**2)
        
        # Safety-first decision making
        if ttc < self.safe_ttc_threshold:
            return 0  # SLOWER
        elif ego_speed < self.target_speed * 0.8:
            return 2  # FASTER
        else:
            return 1  # IDLE
    
    def _lane_changing_action(self, ego: np.ndarray, others: np.ndarray) -> int:
        """Aggressive lane changing behavior."""
        if ego[0] == 0:
            return 1
        
        # Random lane change attempts
        if np.random.random() < self.lane_change_probability:
            # Choose left or right lane change
            return np.random.choice([3, 4])  # LANE_LEFT, LANE_RIGHT
        
        # Otherwise maintain speed
        ego_speed = np.sqrt(ego[3]**2 + ego[4]**2)
        if ego_speed < self.target_speed:
            return 2  # FASTER
        else:
            return 1  # IDLE
    
    def _speed_keeping_action(self, ego: np.ndarray, others: np.ndarray) -> int:
        """Speed maintenance behavior."""
        if ego[0] == 0:
            return 1
        
        ego_speed = np.sqrt(ego[3]**2 + ego[4]**2)
        speed_diff = ego_speed - self.target_speed
        
        if speed_diff < -2.0:
            return 2  # FASTER
        elif speed_diff > 2.0:
            return 0  # SLOWER
        else:
            return 1  # IDLE
    
    def _calculate_ttc(self, ego: np.ndarray, others: np.ndarray) -> float:
        """Calculate time-to-collision with lead vehicle."""
        if others.shape[0] == 0:
            return float('inf')
        
        # Find vehicles in same lane ahead
        ego_lane = ego[2]
        same_lane_ahead = others[(others[:, 0] > 0.5) & 
                                (abs(others[:, 2] - ego_lane) < 2.0) &
                                (others[:, 1] > ego[1])]
        
        if same_lane_ahead.shape[0] == 0:
            return float('inf')
        
        # Find closest vehicle ahead
        distances = same_lane_ahead[:, 1] - ego[1]
        closest_idx = np.argmin(distances)
        closest_vehicle = same_lane_ahead[closest_idx]
        
        # Calculate relative velocity
        rel_velocity = ego[3] - closest_vehicle[3]
        
        if rel_velocity <= 0:
            return float('inf')
        
        distance = distances[closest_idx]
        return max(distance / rel_velocity, 0.0)


class MLPolicyWrapper(ActionSampler):
    """
    Example wrapper for machine learning policies.
    
    This demonstrates how to integrate trained ML models (e.g., from
    stable-baselines3, PyTorch, TensorFlow) into the data collection pipeline.
    """
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "dummy"):
        """
        Initialize ML policy wrapper.
        
        Args:
            model_path: Path to trained model file
            model_type: Type of model ("dummy", "sb3", "pytorch", "tensorflow")
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.step_count = 0
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        if self.model_type == "dummy":
            # Dummy model for demonstration
            self.model = self._create_dummy_model()
        elif self.model_type == "sb3":
            self.model = self._load_sb3_model()
        elif self.model_type == "pytorch":
            self.model = self._load_pytorch_model()
        elif self.model_type == "tensorflow":
            self.model = self._load_tensorflow_model()
    
    def _create_dummy_model(self):
        """Create a dummy model for demonstration."""
        class DummyModel:
            def predict(self, observation):
                # Simple heuristic: choose action based on observation mean
                obs_mean = np.mean(observation)
                if obs_mean > 0.5:
                    return 2  # FASTER
                elif obs_mean < -0.5:
                    return 0  # SLOWER
                else:
                    return 1  # IDLE
        
        return DummyModel()
    
    def _load_sb3_model(self):
        """Load Stable-Baselines3 model."""
        try:
            from stable_baselines3 import PPO
            if self.model_path and Path(self.model_path).exists():
                return PPO.load(self.model_path)
            else:
                print("Warning: SB3 model path not found, using dummy model")
                return self._create_dummy_model()
        except ImportError:
            print("Warning: stable-baselines3 not available, using dummy model")
            return self._create_dummy_model()
    
    def _load_pytorch_model(self):
        """Load PyTorch model."""
        try:
            import torch
            if self.model_path and Path(self.model_path).exists():
                return torch.load(self.model_path)
            else:
                print("Warning: PyTorch model path not found, using dummy model")
                return self._create_dummy_model()
        except ImportError:
            print("Warning: PyTorch not available, using dummy model")
            return self._create_dummy_model()
    
    def _load_tensorflow_model(self):
        """Load TensorFlow model."""
        try:
            import tensorflow as tf
            if self.model_path and Path(self.model_path).exists():
                return tf.keras.models.load_model(self.model_path)
            else:
                print("Warning: TensorFlow model path not found, using dummy model")
                return self._create_dummy_model()
        except ImportError:
            print("Warning: TensorFlow not available, using dummy model")
            return self._create_dummy_model()
    
    def sample_actions(self, observations: Dict[str, Any], n_agents: int, 
                      step: int = 0, episode_id: str = "") -> Tuple[int, ...]:
        """Sample actions using ML model."""
        # Extract observations for model
        kin_obs = observations.get('kinematics', np.zeros((n_agents, 5, 5)))
        
        actions = []
        for agent_idx in range(n_agents):
            agent_obs = kin_obs[agent_idx] if len(kin_obs.shape) > 2 else kin_obs
            
            # Preprocess observation for model
            processed_obs = self._preprocess_observation(agent_obs)
            
            # Get action from model
            if hasattr(self.model, 'predict'):
                if self.model_type == "sb3":
                    action, _ = self.model.predict(processed_obs, deterministic=True)
                    action = int(action)
                else:
                    action = self.model.predict(processed_obs)
            else:
                action = 1  # Default action
            
            actions.append(action)
        
        self.step_count += 1
        return tuple(actions)
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the ML policy state."""
        self.step_count = 0
        if seed is not None:
            np.random.seed(seed)
    
    def _preprocess_observation(self, observation: np.ndarray) -> np.ndarray:
        """Preprocess observation for model input."""
        # Flatten observation for simple models
        if observation.ndim > 1:
            return observation.flatten()
        return observation


class EnsemblePolicy(ActionSampler):
    """
    Ensemble policy that combines multiple policies.
    
    This demonstrates how to create ensemble policies that combine
    different decision-making strategies.
    """
    
    def __init__(self, policies: Dict[str, ActionSampler], weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble policy.
        
        Args:
            policies: Dictionary of named policies
            weights: Optional weights for each policy (default: equal weights)
        """
        self.policies = policies
        self.weights = weights or {name: 1.0 for name in policies.keys()}
        self.step_count = 0
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {name: weight / total_weight for name, weight in self.weights.items()}
    
    def sample_actions(self, observations: Dict[str, Any], n_agents: int, 
                      step: int = 0, episode_id: str = "") -> Tuple[int, ...]:
        """Sample actions using ensemble of policies."""
        # Get actions from each policy
        policy_actions = {}
        for name, policy in self.policies.items():
            policy_actions[name] = policy.sample_actions(observations, n_agents, step, episode_id)
        
        # Combine actions using voting or weighted selection
        final_actions = []
        for agent_idx in range(n_agents):
            agent_actions = [actions[agent_idx] for actions in policy_actions.values()]
            
            # Weighted voting
            action_votes = {}
            for policy_name, actions in policy_actions.items():
                action = actions[agent_idx]
                weight = self.weights[policy_name]
                action_votes[action] = action_votes.get(action, 0) + weight
            
            # Select action with highest vote
            best_action = max(action_votes.keys(), key=lambda a: action_votes[a])
            final_actions.append(best_action)
        
        self.step_count += 1
        return tuple(final_actions)
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset all component policies."""
        self.step_count = 0
        for policy in self.policies.values():
            policy.reset(seed)


def demonstrate_rule_based_policies():
    """Demonstrate rule-based policy integration."""
    print("Rule-Based Policy Integration")
    print("-" * 40)
    
    # Create different rule-based policies
    policies = {
        "safe_following": RuleBasedPolicy("safe_following"),
        "lane_changer": RuleBasedPolicy("lane_changer"),
        "speed_keeper": RuleBasedPolicy("speed_keeper")
    }
    
    # Test policies with sample observations
    registry = ScenarioRegistry()
    env_factory = MultiAgentEnvFactory()
    
    try:
        # Create test environment
        envs = env_factory.create_parallel_envs("free_flow", n_agents=2)
        
        # Reset environments
        seed = 42
        observations = {}
        for modality, env in envs.items():
            env.seed(seed)
            obs = env.reset()
            observations[modality] = obs
        
        print("Testing rule-based policies:")
        for policy_name, policy in policies.items():
            actions = policy.sample_actions(observations, 2, step=0, episode_id="test")
            print(f"  {policy_name}: actions = {actions}")
        
        print("✓ Rule-based policies tested successfully")
        
    except Exception as e:
        print(f"✗ Rule-based policy test failed: {e}")


def demonstrate_ml_policy_integration():
    """Demonstrate ML policy integration."""
    print("\nML Policy Integration")
    print("-" * 40)
    
    # Create ML policy wrappers
    policies = {
        "dummy_model": MLPolicyWrapper(model_type="dummy"),
        # Uncomment these if you have trained models:
        # "sb3_model": MLPolicyWrapper("path/to/sb3_model.zip", "sb3"),
        # "pytorch_model": MLPolicyWrapper("path/to/pytorch_model.pth", "pytorch"),
    }
    
    print("Testing ML policy wrappers:")
    
    # Create sample observations
    sample_obs = {
        'kinematics': np.random.randn(2, 5, 5),
        'occupancy_grid': np.random.randn(2, 11, 11),
        'grayscale': np.random.randn(2, 84, 84, 4)
    }
    
    for policy_name, policy in policies.items():
        try:
            actions = policy.sample_actions(sample_obs, 2, step=0, episode_id="test")
            print(f"  {policy_name}: actions = {actions}")
        except Exception as e:
            print(f"  {policy_name}: failed with error {e}")
    
    print("✓ ML policy integration demonstrated")


def demonstrate_ensemble_policies():
    """Demonstrate ensemble policy combination."""
    print("\nEnsemble Policy Integration")
    print("-" * 40)
    
    # Create component policies
    component_policies = {
        "safe": RuleBasedPolicy("safe_following"),
        "aggressive": RuleBasedPolicy("lane_changer"),
        "ml": MLPolicyWrapper(model_type="dummy")
    }
    
    # Create ensemble with different weighting strategies
    ensembles = {
        "equal_weight": EnsemblePolicy(component_policies),
        "safety_focused": EnsemblePolicy(component_policies, {"safe": 0.6, "aggressive": 0.2, "ml": 0.2}),
        "ml_focused": EnsemblePolicy(component_policies, {"safe": 0.2, "aggressive": 0.2, "ml": 0.6})
    }
    
    # Test ensembles
    sample_obs = {'kinematics': np.random.randn(2, 5, 5)}
    
    print("Testing ensemble policies:")
    for ensemble_name, ensemble in ensembles.items():
        actions = ensemble.sample_actions(sample_obs, 2, step=0, episode_id="test")
        print(f"  {ensemble_name}: actions = {actions}")
    
    print("✓ Ensemble policies demonstrated")


def demonstrate_policy_based_collection():
    """Demonstrate data collection with custom policies."""
    print("\nPolicy-Based Data Collection")
    print("-" * 40)
    
    try:
        # Create policy and collector
        policy = RuleBasedPolicy("safe_following")
        env_factory = MultiAgentEnvFactory()
        collector = SynchronizedCollector(env_factory, action_sampler=policy)
        
        print("Collecting data with custom policy...")
        
        # Collect small batch for demonstration
        result = collector.collect_episode_batch(
            scenario_name="free_flow",
            episodes=2,
            seed=42,
            max_steps=20
        )
        
        print(f"✓ Collected {len(result.transitions)} transitions")
        print(f"  Episodes: {len(result.episode_metadata)}")
        print(f"  Collection time: {result.collection_time:.3f}s")
        
        # Show sample actions from collected data
        if result.transitions:
            actions = [t['action'] for t in result.transitions[:10]]
            print(f"  Sample actions: {actions}")
        
    except Exception as e:
        print(f"✗ Policy-based collection failed: {e}")


def show_integration_code_examples():
    """Show code examples for policy integration."""
    print("\nPolicy Integration Code Examples")
    print("-" * 40)
    
    print("""
# Example 1: Simple Custom Policy
from highway_datacollection.collection.action_samplers import ActionSampler

class MyCustomPolicy(ActionSampler):
    def sample_actions(self, observations, n_agents, seed=None):
        # Your custom policy logic here
        actions = []
        for agent_idx in range(n_agents):
            # Extract agent observation
            agent_obs = observations['kinematics'][agent_idx]
            
            # Make decision based on observation
            if agent_obs[0, 3] > 20:  # If speed > 20
                action = 0  # SLOWER
            else:
                action = 2  # FASTER
            
            actions.append(action)
        
        return tuple(actions)

# Usage in collection
from highway_datacollection.collection.collector import SynchronizedCollector

policy = MyCustomPolicy()
collector = SynchronizedCollector(env_factory, action_sampler=policy)
result = collector.collect_episode_batch("free_flow", episodes=100, seed=42)
""")
    
    print("""
# Example 2: Stable-Baselines3 Integration
from stable_baselines3 import PPO
from highway_datacollection.collection.action_samplers import ActionSampler

class SB3PolicySampler(ActionSampler):
    def __init__(self, model_path):
        self.model = PPO.load(model_path)
    
    def sample_actions(self, observations, n_agents, seed=None):
        actions = []
        for agent_idx in range(n_agents):
            obs = observations['kinematics'][agent_idx].flatten()
            action, _ = self.model.predict(obs, deterministic=True)
            actions.append(int(action))
        return tuple(actions)

# Usage
policy = SB3PolicySampler("path/to/trained_model.zip")
collector = SynchronizedCollector(env_factory, action_sampler=policy)
""")
    
    print("""
# Example 3: Multi-Policy Ensemble
policies = {
    "conservative": RuleBasedPolicy("safe_following"),
    "aggressive": RuleBasedPolicy("lane_changer"),
    "learned": MLPolicyWrapper("model.pth", "pytorch")
}

weights = {"conservative": 0.5, "aggressive": 0.2, "learned": 0.3}
ensemble = EnsemblePolicy(policies, weights)

collector = SynchronizedCollector(env_factory, action_sampler=ensemble)
""")


def main():
    """Run all policy integration examples."""
    print("HighwayEnv Multi-Modal Data Collection")
    print("Policy Integration Examples")
    print("=" * 50)
    
    examples = [
        demonstrate_rule_based_policies,
        demonstrate_ml_policy_integration,
        demonstrate_ensemble_policies,
        demonstrate_policy_based_collection,
        show_integration_code_examples
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Error in {example_func.__name__}: {e}")
        print()  # Add spacing between examples
    
    print("=" * 50)
    print("Policy integration examples completed!")


if __name__ == "__main__":
    main()