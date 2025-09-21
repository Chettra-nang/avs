#!/usr/bin/env python3
"""
Comprehensive demonstration script for HighwayEnv Multi-Modal Data Collection System.

This script demonstrates:
1. Complete dataset collection workflow
2. Different scenario configurations and modality selections
3. Data loading examples for accessing collected datasets
4. Policy integration and custom feature extraction examples

Usage:
    python main.py                    # Interactive demonstration
    python main.py --demo basic       # Basic functionality demo
    python main.py --demo collection  # Collection workflow demo
    python main.py --demo loading     # Data loading demo
    python main.py --demo policy      # Policy integration demo
    python main.py --demo features    # Custom feature extraction demo
    python main.py --demo all         # Run all demonstrations
"""

import sys
import logging
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from abc import ABC, abstractmethod

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from highway_datacollection import ScenarioRegistry
from highway_datacollection.config import ensure_directories
from highway_datacollection.collection.orchestrator import run_full_collection, CollectionProgress
from highway_datacollection.collection.action_samplers import ActionSampler, RandomActionSampler
from highway_datacollection.collection.collector import SynchronizedCollector
from highway_datacollection.environments.factory import MultiAgentEnvFactory
from highway_datacollection.features.extractors import KinematicsExtractor
from highway_datacollection.storage.manager import DatasetStorageManager


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/collection/main_collection.log')
        ]
    )


# ============================================================================
# Custom Policy Integration Examples
# ============================================================================

class ExamplePolicyActionSampler(ActionSampler):
    """
    Example policy-based action sampler demonstrating integration of trained agents.
    
    This example shows how to replace random action sampling with policy inference
    while maintaining deterministic behavior through seed management.
    """
    
    def __init__(self, policy_type: str = "conservative"):
        """
        Initialize policy sampler.
        
        Args:
            policy_type: Type of policy behavior ("conservative", "aggressive", "adaptive")
        """
        self.policy_type = policy_type
        self.step_count = 0
    
    def sample_actions(self, observations: Dict[str, Any], n_agents: int, 
                      step: int = 0, episode_id: str = "") -> Tuple[int, ...]:
        """
        Sample actions using policy inference.
        
        Args:
            observations: Dictionary of observations for each modality
            n_agents: Number of agents to sample actions for
            step: Current step in the episode
            episode_id: Current episode ID
            
        Returns:
            Tuple of actions for each agent
        """
        # Extract kinematics observations for policy decision
        kin_obs = observations.get('kinematics', np.zeros((n_agents, 5, 5)))
        actions = []
        
        for agent_idx in range(n_agents):
            agent_obs = kin_obs[agent_idx] if len(kin_obs.shape) > 2 else kin_obs
            action = self._policy_inference(agent_obs, agent_idx)
            actions.append(action)
        
        self.step_count += 1
        return tuple(actions)
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the policy sampler state.
        
        Args:
            seed: Random seed for deterministic behavior
        """
        self.step_count = 0
        if seed is not None:
            np.random.seed(seed)
    
    def _policy_inference(self, observation: np.ndarray, agent_idx: int) -> int:
        """
        Perform policy inference for a single agent.
        
        Args:
            observation: Agent's kinematics observation
            agent_idx: Agent index
            
        Returns:
            Selected action
        """
        # Example policy logic based on policy type
        if self.policy_type == "conservative":
            # Conservative policy: prefer maintaining lane and speed
            return np.random.choice([1, 2], p=[0.8, 0.2])  # IDLE, FASTER
        
        elif self.policy_type == "aggressive":
            # Aggressive policy: more lane changes and speed changes
            return np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.2, 0.2, 0.2, 0.2])
        
        elif self.policy_type == "adaptive":
            # Adaptive policy: decision based on traffic density
            ego_vehicle = observation[0] if observation.shape[0] > 0 else np.zeros(5)
            other_vehicles = observation[1:] if observation.shape[0] > 1 else np.zeros((4, 5))
            
            # Count nearby vehicles (simple traffic density estimation)
            nearby_count = np.sum(other_vehicles[:, 0] > 0.5)  # presence > 0.5
            
            if nearby_count > 2:  # Dense traffic
                return np.random.choice([1, 2], p=[0.9, 0.1])  # Stay in lane
            else:  # Light traffic
                return np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.3, 0.3, 0.15, 0.15])
        
        return 1  # Default: IDLE


class CustomFeatureExtractor:
    """
    Example custom feature extractor demonstrating extensible feature derivation.
    
    This example shows how to add custom derived metrics beyond the standard
    TTC, lane estimation, and traffic density calculations.
    """
    
    def extract_features(self, observation: np.ndarray, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract custom features from kinematics observation.
        
        Args:
            observation: Kinematics observation array
            context: Additional context information
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        if observation.shape[0] == 0:
            return self._empty_features()
        
        ego_vehicle = observation[0]
        other_vehicles = observation[1:] if observation.shape[0] > 1 else np.zeros((0, 5))
        
        # Standard features
        features.update(self._extract_standard_features(ego_vehicle, other_vehicles))
        
        # Custom features
        features.update(self._extract_custom_features(ego_vehicle, other_vehicles))
        
        return features
    
    def _extract_standard_features(self, ego: np.ndarray, others: np.ndarray) -> Dict[str, Any]:
        """Extract standard features (TTC, lane position, etc.)"""
        features = {}
        
        # Lane estimation (simplified)
        features['estimated_lane'] = int(ego[2] // 4.0) if ego[0] > 0 else 0
        
        # Speed
        features['speed'] = np.sqrt(ego[3]**2 + ego[4]**2) if ego[0] > 0 else 0.0
        
        # TTC calculation
        features['ttc'] = self._calculate_ttc(ego, others)
        
        return features
    
    def _extract_custom_features(self, ego: np.ndarray, others: np.ndarray) -> Dict[str, Any]:
        """Extract custom features specific to this extractor"""
        features = {}
        
        # Custom feature 1: Lateral acceleration estimate
        features['lateral_acceleration'] = abs(ego[4]) if ego[0] > 0 else 0.0
        
        # Custom feature 2: Relative speed to traffic flow
        if others.shape[0] > 0 and ego[0] > 0:
            other_speeds = np.sqrt(others[:, 3]**2 + others[:, 4]**2)
            valid_others = others[:, 0] > 0.5
            if np.any(valid_others):
                avg_traffic_speed = np.mean(other_speeds[valid_others])
                ego_speed = np.sqrt(ego[3]**2 + ego[4]**2)
                features['relative_speed_to_traffic'] = ego_speed - avg_traffic_speed
            else:
                features['relative_speed_to_traffic'] = 0.0
        else:
            features['relative_speed_to_traffic'] = 0.0
        
        # Custom feature 3: Congestion indicator
        if others.shape[0] > 0:
            nearby_vehicles = np.sum((others[:, 0] > 0.5) & 
                                   (np.abs(others[:, 1] - ego[1]) < 50) &
                                   (np.abs(others[:, 2] - ego[2]) < 8))
            features['congestion_level'] = min(nearby_vehicles / 5.0, 1.0)  # Normalize to [0,1]
        else:
            features['congestion_level'] = 0.0
        
        # Custom feature 4: Lane change opportunity
        features['lane_change_opportunity'] = self._assess_lane_change_opportunity(ego, others)
        
        return features
    
    def _calculate_ttc(self, ego: np.ndarray, others: np.ndarray) -> float:
        """Calculate time-to-collision with lead vehicle"""
        if others.shape[0] == 0 or ego[0] == 0:
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
        rel_velocity = ego[3] - closest_vehicle[3]  # Longitudinal velocity difference
        
        if rel_velocity <= 0:  # Not approaching
            return float('inf')
        
        distance = distances[closest_idx]
        ttc = distance / rel_velocity
        
        return max(ttc, 0.0)
    
    def _assess_lane_change_opportunity(self, ego: np.ndarray, others: np.ndarray) -> float:
        """Assess opportunity for lane change (0=no opportunity, 1=clear opportunity)"""
        if others.shape[0] == 0 or ego[0] == 0:
            return 1.0
        
        ego_lane = ego[2]
        ego_pos = ego[1]
        
        # Check left and right lanes
        left_lane = ego_lane - 4.0
        right_lane = ego_lane + 4.0
        
        left_opportunity = self._check_lane_opportunity(ego_pos, left_lane, others)
        right_opportunity = self._check_lane_opportunity(ego_pos, right_lane, others)
        
        return max(left_opportunity, right_opportunity)
    
    def _check_lane_opportunity(self, ego_pos: float, target_lane: float, others: np.ndarray) -> float:
        """Check if lane change opportunity exists for specific lane"""
        # Find vehicles in target lane
        target_lane_vehicles = others[(others[:, 0] > 0.5) & 
                                    (abs(others[:, 2] - target_lane) < 2.0)]
        
        if target_lane_vehicles.shape[0] == 0:
            return 1.0  # Clear lane
        
        # Check for safe gaps
        positions = target_lane_vehicles[:, 1]
        safe_gap_size = 20.0  # Minimum safe gap
        
        # Check gap behind and ahead
        behind_vehicles = positions[positions < ego_pos]
        ahead_vehicles = positions[positions > ego_pos]
        
        gap_behind = ego_pos - np.max(behind_vehicles) if behind_vehicles.size > 0 else float('inf')
        gap_ahead = np.min(ahead_vehicles) - ego_pos if ahead_vehicles.size > 0 else float('inf')
        
        min_gap = min(gap_behind, gap_ahead)
        
        if min_gap >= safe_gap_size:
            return 1.0
        elif min_gap >= safe_gap_size * 0.5:
            return 0.5
        else:
            return 0.0
    
    def _empty_features(self) -> Dict[str, Any]:
        """Return empty features when no observation available"""
        return {
            'estimated_lane': 0,
            'speed': 0.0,
            'ttc': float('inf'),
            'lateral_acceleration': 0.0,
            'relative_speed_to_traffic': 0.0,
            'congestion_level': 0.0,
            'lane_change_opportunity': 1.0
        }


def progress_callback(progress: CollectionProgress):
    """
    Callback function for collection progress updates.
    
    Args:
        progress: Current collection progress
    """
    elapsed = progress.scenario_start_time - progress.start_time if hasattr(progress, 'scenario_start_time') else 0
    print(f"[{progress.current_scenario}] Episode {progress.current_episode}/{progress.total_episodes} "
          f"(Success: {progress.successful_episodes}, Failed: {progress.failed_episodes}) "
          f"- Scenario {progress.scenario_index + 1}/{progress.total_scenarios}")


# ============================================================================
# Demonstration Functions
# ============================================================================

def demonstrate_basic_functionality():
    """Demonstrate basic system functionality and scenario configurations."""
    print("HighwayEnv Multi-Modal Data Collection System")
    print("=" * 50)
    
    # Ensure necessary directories exist
    ensure_directories()
    print("✓ Created necessary directories")
    
    # Initialize scenario registry
    registry = ScenarioRegistry()
    print("✓ Initialized scenario registry")
    
    # List available scenarios
    scenarios = registry.list_scenarios()
    print(f"✓ Available scenarios ({len(scenarios)}):")
    for scenario in scenarios:
        print(f"  - {scenario}")
    
    # Demonstrate different scenario configurations
    print("\n" + "=" * 50)
    print("Scenario Configuration Examples")
    print("=" * 50)
    
    example_scenarios = ["free_flow", "dense_commuting", "lane_closure"]
    for scenario_name in example_scenarios:
        if scenario_name in scenarios:
            print(f"\n{scenario_name.replace('_', ' ').title()}:")
            config = registry.get_scenario_config(scenario_name)
            for key, value in config.items():
                if key != "description":
                    print(f"  {key}: {value}")
            print(f"  description: {config['description']}")
            
            # Validate configuration
            is_valid = registry.validate_scenario(config)
            print(f"  ✓ Configuration is valid: {is_valid}")
    
    # Demonstrate modality selection
    print("\n" + "=" * 50)
    print("Modality Selection Examples")
    print("=" * 50)
    
    modalities = ["kinematics", "occupancy_grid", "grayscale"]
    print("Available observation modalities:")
    for modality in modalities:
        print(f"  - {modality}: {'Multi-agent vehicle dynamics' if modality == 'kinematics' else 'Spatial grid representation' if modality == 'occupancy_grid' else 'Vision-based grayscale images'}")
    
    print("\nModality combinations for different use cases:")
    print("  - RL Training: kinematics + occupancy_grid")
    print("  - Vision Research: grayscale + kinematics")
    print("  - Complete Dataset: all modalities")
    print("  - Lightweight Collection: kinematics only")


def demonstrate_collection_workflow():
    """Demonstrate the complete dataset collection workflow."""
    print("\n" + "=" * 50)
    print("Dataset Collection Workflow Demonstration")
    print("=" * 50)
    
    # Set up storage path
    storage_path = Path("data/highway_multimodal_dataset")
    
    print(f"Starting collection workflow to: {storage_path}")
    print("Configuration:")
    print("  - Episodes per scenario: 3 (demo)")
    print("  - Agents per scenario: 2")
    print("  - Max steps per episode: 30")
    print("  - Batch size: 2")
    print("  - Scenarios: free_flow, dense_commuting (demo subset)")
    print("  - All modalities: kinematics, occupancy_grid, grayscale")
    
    try:
        # Run collection with demo parameters
        result = run_full_collection(
            base_storage_path=storage_path,
            episodes_per_scenario=3,  # Small number for demo
            n_agents=2,
            max_steps_per_episode=30,
            base_seed=42,
            scenarios=["free_flow", "dense_commuting"],  # Subset for demo
            batch_size=2,
            progress_callback=progress_callback
        )
        
        # Display results
        print("\n" + "=" * 50)
        print("Collection Results")
        print("=" * 50)
        print(f"Total scenarios: {result.total_scenarios}")
        print(f"Successful scenarios: {result.successful_scenarios}")
        print(f"Failed scenarios: {result.failed_scenarios}")
        print(f"Total episodes: {result.total_episodes}")
        print(f"Successful episodes: {result.successful_episodes}")
        print(f"Failed episodes: {result.failed_episodes}")
        print(f"Collection time: {result.collection_time:.2f}s")
        
        if result.dataset_index_path:
            print(f"Dataset index created: {result.dataset_index_path}")
        
        if result.errors:
            print(f"\nErrors encountered ({len(result.errors)}):")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"  - {error}")
            if len(result.errors) > 3:
                print(f"  ... and {len(result.errors) - 3} more")
        
        # Show per-scenario results
        print("\nPer-scenario results:")
        for scenario_name, scenario_result in result.scenario_results.items():
            print(f"  {scenario_name}: {scenario_result.successful_episodes}/{scenario_result.total_episodes} episodes "
                  f"({scenario_result.collection_time:.2f}s)")
        
        print("\n✓ Collection workflow demonstration completed successfully!")
        return storage_path
        
    except Exception as e:
        print(f"✗ Collection failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def demonstrate_data_loading():
    """Demonstrate how to load and access collected datasets."""
    print("\n" + "=" * 50)
    print("Data Loading and Access Demonstration")
    print("=" * 50)
    
    # Check for existing dataset
    storage_path = Path("data/highway_multimodal_dataset")
    index_path = storage_path / "index.json"
    
    if not index_path.exists():
        print("No dataset found. Running mini collection first...")
        storage_path = demonstrate_collection_workflow()
        if storage_path is None:
            print("Failed to create dataset for loading demonstration")
            return
        index_path = storage_path / "index.json"
    
    try:
        # Load dataset index
        print(f"Loading dataset index from: {index_path}")
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        print(f"✓ Dataset index loaded successfully")
        print(f"  - Total scenarios: {len(index_data['scenarios'])}")
        print(f"  - Creation time: {index_data.get('created', 'Unknown')}")
        print(f"  - Total files: {index_data.get('total_files', 0)}")
        
        # Show available scenarios
        print("\nAvailable scenarios:")
        for scenario_name, scenario_files in index_data['scenarios'].items():
            print(f"  - {scenario_name}: {len(scenario_files)} files")
        
        # Load example data from first available scenario
        if index_data['scenarios']:
            scenario_name = list(index_data['scenarios'].keys())[0]
            scenario_files = index_data['scenarios'][scenario_name]
            
            if scenario_files:
                # Load first parquet file
                first_file_info = scenario_files[0]
                parquet_file = first_file_info['transitions_file']
                parquet_path = storage_path / parquet_file
                
                print(f"\nLoading example data from: {parquet_path}")
                
                if parquet_path.exists():
                    # Load with pandas
                    df = pd.read_parquet(parquet_path)
                    print(f"✓ Loaded parquet file with {len(df)} rows and {len(df.columns)} columns")
                    
                    # Show data structure
                    print("\nDataset structure:")
                    print(f"  - Episodes: {df['episode_id'].nunique()}")
                    print(f"  - Agents: {df['agent_id'].nunique()}")
                    print(f"  - Max steps: {df['step'].max()}")
                    print(f"  - Columns: {list(df.columns)}")
                    
                    # Show sample data
                    print("\nSample data (first 3 rows):")
                    sample_cols = ['episode_id', 'step', 'agent_id', 'action', 'reward', 'ttc', 'summary_text']
                    available_cols = [col for col in sample_cols if col in df.columns]
                    print(df[available_cols].head(3).to_string(index=False))
                    
                    # Demonstrate binary data loading
                    if 'occ_blob' in df.columns and 'occ_shape' in df.columns:
                        print("\nDemonstrating binary array reconstruction:")
                        first_row = df.iloc[0]
                        
                        # Reconstruct occupancy grid
                        blob_data = first_row['occ_blob']
                        shape = eval(first_row['occ_shape']) if isinstance(first_row['occ_shape'], str) else first_row['occ_shape']
                        dtype = first_row['occ_dtype']
                        
                        reconstructed_array = np.frombuffer(blob_data, dtype=dtype).reshape(shape)
                        print(f"  - Reconstructed occupancy grid: shape {reconstructed_array.shape}, dtype {reconstructed_array.dtype}")
                        print(f"  - Value range: [{reconstructed_array.min():.3f}, {reconstructed_array.max():.3f}]")
                    
                    # Show feature analysis
                    print("\nFeature analysis:")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        print("  Numeric feature statistics:")
                        stats = df[numeric_cols].describe()
                        for col in ['ttc', 'reward'] if 'ttc' in numeric_cols and 'reward' in numeric_cols else numeric_cols[:2]:
                            if col in stats.columns:
                                print(f"    {col}: mean={stats.loc['mean', col]:.3f}, std={stats.loc['std', col]:.3f}")
                    
                    # Show text summaries
                    if 'summary_text' in df.columns:
                        print("\nExample natural language summaries:")
                        summaries = df['summary_text'].dropna().head(2)
                        for i, summary in enumerate(summaries):
                            print(f"  {i+1}. {summary}")
                
                else:
                    print(f"✗ Parquet file not found: {parquet_path}")
        
        # Demonstrate programmatic data access
        print("\n" + "=" * 30)
        print("Programmatic Data Access Examples")
        print("=" * 30)
        
        print("""
# Example 1: Load specific scenario data
import pandas as pd
from pathlib import Path

def load_scenario_data(dataset_path, scenario_name):
    scenario_path = Path(dataset_path) / scenario_name
    parquet_files = list(scenario_path.glob("*_transitions.parquet"))
    
    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# Usage
df = load_scenario_data("data/highway_multimodal_dataset", "free_flow")
""")
        
        print("""
# Example 2: Reconstruct binary observations
import numpy as np

def reconstruct_observation(row, modality='occ'):
    blob_col = f'{modality}_blob'
    shape_col = f'{modality}_shape'
    dtype_col = f'{modality}_dtype'
    
    if all(col in row for col in [blob_col, shape_col, dtype_col]):
        blob_data = row[blob_col]
        shape = eval(row[shape_col]) if isinstance(row[shape_col], str) else row[shape_col]
        dtype = row[dtype_col]
        
        return np.frombuffer(blob_data, dtype=dtype).reshape(shape)
    return None

# Usage
occ_grid = reconstruct_observation(df.iloc[0], 'occ')
grayscale = reconstruct_observation(df.iloc[0], 'gray')
""")
        
        print("✓ Data loading demonstration completed successfully!")
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_policy_integration():
    """Demonstrate policy integration for custom action sampling."""
    print("\n" + "=" * 50)
    print("Policy Integration Demonstration")
    print("=" * 50)
    
    print("This demonstration shows how to integrate trained policies")
    print("for custom action sampling while maintaining deterministic behavior.")
    
    # Initialize components
    registry = ScenarioRegistry()
    env_factory = MultiAgentEnvFactory()
    
    # Create example policies
    policies = {
        "conservative": ExamplePolicyActionSampler("conservative"),
        "aggressive": ExamplePolicyActionSampler("aggressive"),
        "adaptive": ExamplePolicyActionSampler("adaptive")
    }
    
    print(f"\n✓ Created {len(policies)} example policies")
    
    # Demonstrate policy-based collection
    print("\nDemonstrating policy-based data collection:")
    
    try:
        # Create environments for demonstration
        scenario_name = "free_flow"
        n_agents = 2
        
        envs = env_factory.create_parallel_envs(scenario_name, n_agents)
        print(f"✓ Created environments for scenario: {scenario_name}")
        
        # Test each policy
        for policy_name, policy in policies.items():
            print(f"\nTesting {policy_name} policy:")
            
            # Reset environments
            seed = 42
            observations = {}
            for modality, env in envs.items():
                # Use the newer gymnasium seeding method
                obs, _ = env.reset(seed=seed)
                observations[modality] = obs
            
            # Sample actions using policy
            actions = policy.sample_actions(observations, n_agents, seed)
            print(f"  - Sampled actions: {actions}")
            
            # Take a few steps to show policy behavior
            for step in range(3):
                step_seed = seed + step + 1
                actions = policy.sample_actions(observations, n_agents, step_seed)
                
                # Step environments (just for demonstration)
                for modality, env in envs.items():
                    obs, rewards, terminated, truncated, infos = env.step(actions)
                    observations[modality] = obs
                
                print(f"  - Step {step + 1} actions: {actions}")
        
        # Show policy integration code example
        print("\n" + "=" * 30)
        print("Policy Integration Code Examples")
        print("=" * 30)
        
        print("""
# Example 1: Custom Policy Action Sampler
from highway_datacollection.collection.action_samplers import ActionSampler

class MyPolicyActionSampler(ActionSampler):
    def __init__(self, model_path):
        self.model = load_model(model_path)  # Load your trained model
    
    def sample_actions(self, observations, n_agents, seed=None):
        # Extract relevant observations for your model
        kin_obs = observations.get('kinematics')
        
        # Run inference
        actions = []
        for agent_idx in range(n_agents):
            agent_obs = kin_obs[agent_idx]
            action = self.model.predict(agent_obs)
            actions.append(action)
        
        return tuple(actions)
""")
        
        print("""
# Example 2: Using Policy in Collection
from highway_datacollection.collection.collector import SynchronizedCollector

# Create collector with custom policy
policy_sampler = MyPolicyActionSampler("path/to/model.pkl")
collector = SynchronizedCollector(
    env_factory=env_factory,
    action_sampler=policy_sampler
)

# Collect data using policy
result = collector.collect_episode_batch(
    scenario_name="dense_commuting",
    episodes=100,
    seed=42
)
""")
        
        print("✓ Policy integration demonstration completed successfully!")
        
    except Exception as e:
        print(f"✗ Policy integration demo failed: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_custom_features():
    """Demonstrate custom feature extraction capabilities."""
    print("\n" + "=" * 50)
    print("Custom Feature Extraction Demonstration")
    print("=" * 50)
    
    print("This demonstration shows how to extend the system with")
    print("custom derived metrics and feature extraction.")
    
    # Create custom feature extractor
    custom_extractor = CustomFeatureExtractor()
    print("✓ Created custom feature extractor")
    
    # Generate example observation for demonstration
    print("\nGenerating example observations for feature extraction:")
    
    # Create synthetic observation data
    np.random.seed(42)
    
    # Ego vehicle: [presence, x, y, vx, vy]
    ego_vehicle = np.array([1.0, 100.0, 8.0, 25.0, 0.5])  # Moving forward with slight lateral movement
    
    # Other vehicles
    other_vehicles = np.array([
        [1.0, 120.0, 8.0, 20.0, 0.0],   # Vehicle ahead in same lane
        [1.0, 80.0, 12.0, 30.0, 0.0],   # Vehicle behind in right lane
        [1.0, 110.0, 4.0, 22.0, 0.0],   # Vehicle ahead in left lane
        [1.0, 90.0, 8.0, 15.0, 0.0],    # Slower vehicle behind in same lane
    ])
    
    # Combine into observation
    observation = np.vstack([ego_vehicle, other_vehicles])
    
    print(f"  - Ego vehicle: position=({ego_vehicle[1]:.1f}, {ego_vehicle[2]:.1f}), velocity=({ego_vehicle[3]:.1f}, {ego_vehicle[4]:.1f})")
    print(f"  - Other vehicles: {other_vehicles.shape[0]} vehicles")
    
    # Extract features
    context = {"scenario": "free_flow", "step": 10}
    features = custom_extractor.extract_features(observation, context)
    
    print("\nExtracted features:")
    for feature_name, value in features.items():
        if isinstance(value, float):
            if value == float('inf'):
                print(f"  - {feature_name}: ∞")
            else:
                print(f"  - {feature_name}: {value:.3f}")
        else:
            print(f"  - {feature_name}: {value}")
    
    # Demonstrate feature interpretation
    print("\nFeature interpretation:")
    print(f"  - Lane {features['estimated_lane']}: Ego vehicle is in lane {features['estimated_lane']}")
    print(f"  - Speed {features['speed']:.1f} m/s: Current ego vehicle speed")
    
    if features['ttc'] != float('inf'):
        print(f"  - TTC {features['ttc']:.1f}s: Time to collision with lead vehicle")
    else:
        print("  - TTC ∞: No collision risk detected")
    
    print(f"  - Lateral acceleration {features['lateral_acceleration']:.3f}: Measure of lane changing")
    print(f"  - Relative speed {features['relative_speed_to_traffic']:.1f}: Speed relative to traffic flow")
    print(f"  - Congestion level {features['congestion_level']:.1f}: Traffic density indicator (0-1)")
    print(f"  - Lane change opportunity {features['lane_change_opportunity']:.1f}: Safety for lane changes (0-1)")
    
    # Show different scenarios
    print("\nTesting different traffic scenarios:")
    
    scenarios = [
        ("Dense traffic", np.array([
            [1.0, 100.0, 8.0, 15.0, 0.0],   # Ego: slower in traffic
            [1.0, 105.0, 8.0, 15.0, 0.0],   # Close ahead
            [1.0, 95.0, 8.0, 15.0, 0.0],    # Close behind
            [1.0, 103.0, 4.0, 15.0, 0.0],   # Close left
            [1.0, 97.0, 12.0, 15.0, 0.0],   # Close right
            [1.0, 110.0, 8.0, 15.0, 0.0],   # Another ahead
        ])),
        ("Highway cruising", np.array([
            [1.0, 100.0, 8.0, 30.0, 0.0],   # Ego: cruising speed
            [1.0, 150.0, 8.0, 28.0, 0.0],   # Distant ahead
            [1.0, 50.0, 12.0, 32.0, 0.0],   # Distant right
        ])),
        ("Lane change opportunity", np.array([
            [1.0, 100.0, 8.0, 25.0, 0.0],   # Ego
            [1.0, 110.0, 8.0, 20.0, 0.0],   # Slow ahead in same lane
            [1.0, 80.0, 4.0, 30.0, 0.0],    # Fast behind in left lane
            [1.0, 130.0, 4.0, 25.0, 0.0],   # Clear ahead in left lane
        ]))
    ]
    
    for scenario_name, scenario_obs in scenarios:
        features = custom_extractor.extract_features(scenario_obs, {"scenario": scenario_name})
        print(f"\n  {scenario_name}:")
        print(f"    Congestion: {features['congestion_level']:.2f}, "
              f"Lane change opportunity: {features['lane_change_opportunity']:.2f}, "
              f"TTC: {features['ttc']:.1f}s" if features['ttc'] != float('inf') else "TTC: ∞")
    
    # Show custom feature extractor code example
    print("\n" + "=" * 30)
    print("Custom Feature Extractor Code Examples")
    print("=" * 30)
    
    print("""
# Example 1: Simple Custom Feature Extractor
from highway_datacollection.features.extractors import FeatureExtractor

class MyFeatureExtractor(FeatureExtractor):
    def extract_features(self, observation, context):
        features = {}
        
        if observation.shape[0] > 0:
            ego = observation[0]
            others = observation[1:]
            
            # Custom feature: Acceleration estimate
            features['acceleration'] = self.estimate_acceleration(ego, context)
            
            # Custom feature: Risk assessment
            features['risk_level'] = self.assess_risk(ego, others)
        
        return features
    
    def estimate_acceleration(self, ego, context):
        # Your custom acceleration estimation logic
        return 0.0
    
    def assess_risk(self, ego, others):
        # Your custom risk assessment logic
        return 0.0
""")
    
    print("""
# Example 2: Using Custom Extractor in Collection
from highway_datacollection.collection.collector import SynchronizedCollector

# Create collector with custom feature extractor
collector = SynchronizedCollector(
    env_factory=env_factory,
    feature_extractor=MyFeatureExtractor()
)

# Features will be automatically extracted during collection
result = collector.collect_episode_batch("free_flow", episodes=10, seed=42)
""")
    
    print("✓ Custom feature extraction demonstration completed successfully!")


def run_interactive_demo():
    """Run interactive demonstration with user choices."""
    print("HighwayEnv Multi-Modal Data Collection System")
    print("Interactive Demonstration")
    print("=" * 50)
    
    demos = {
        "1": ("Basic functionality and scenario configurations", demonstrate_basic_functionality),
        "2": ("Complete dataset collection workflow", demonstrate_collection_workflow),
        "3": ("Data loading and access examples", demonstrate_data_loading),
        "4": ("Policy integration for custom action sampling", demonstrate_policy_integration),
        "5": ("Custom feature extraction capabilities", demonstrate_custom_features),
        "6": ("Run all demonstrations", None)
    }
    
    while True:
        print("\nAvailable demonstrations:")
        for key, (description, _) in demos.items():
            print(f"  {key}. {description}")
        print("  q. Quit")
        
        choice = input("\nSelect demonstration (1-6, q): ").strip().lower()
        
        if choice == 'q':
            break
        elif choice in demos:
            if choice == "6":
                # Run all demos
                for demo_key in ["1", "2", "3", "4", "5"]:
                    print(f"\n{'='*60}")
                    print(f"Running Demo {demo_key}: {demos[demo_key][0]}")
                    print(f"{'='*60}")
                    demos[demo_key][1]()
                break
            else:
                demos[choice][1]()
        else:
            print("Invalid choice. Please select 1-6 or q.")
    
    print("\n" + "=" * 50)
    print("Interactive demonstration complete!")
    print("=" * 50)


def main():
    """Main demonstration function with command line argument support."""
    parser = argparse.ArgumentParser(
        description="HighwayEnv Multi-Modal Data Collection System Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Interactive demonstration
  python main.py --demo basic       # Basic functionality demo
  python main.py --demo collection  # Collection workflow demo
  python main.py --demo loading     # Data loading demo
  python main.py --demo policy      # Policy integration demo
  python main.py --demo features    # Custom feature extraction demo
  python main.py --demo all         # Run all demonstrations
        """
    )
    
    parser.add_argument(
        '--demo',
        choices=['basic', 'collection', 'loading', 'policy', 'features', 'all'],
        help='Run specific demonstration'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    if args.demo:
        # Run specific demonstration
        demo_map = {
            'basic': demonstrate_basic_functionality,
            'collection': demonstrate_collection_workflow,
            'loading': demonstrate_data_loading,
            'policy': demonstrate_policy_integration,
            'features': demonstrate_custom_features,
            'all': lambda: [
                demonstrate_basic_functionality(),
                demonstrate_collection_workflow(),
                demonstrate_data_loading(),
                demonstrate_policy_integration(),
                demonstrate_custom_features()
            ]
        }
        
        print(f"Running {args.demo} demonstration...")
        demo_map[args.demo]()
    else:
        # Run interactive demonstration
        run_interactive_demo()
    
    # Show usage examples
    print("\n" + "=" * 50)
    print("Quick Start Examples")
    print("=" * 50)
    
    print("""
# 1. Basic data collection
from highway_datacollection.collection.orchestrator import run_full_collection
from pathlib import Path

result = run_full_collection(
    base_storage_path=Path("data/my_dataset"),
    episodes_per_scenario=100,
    n_agents=3,
    scenarios=["free_flow", "dense_commuting"]
)

# 2. Load collected data
import pandas as pd
import json

# Load dataset index
with open("data/my_dataset/index.json", 'r') as f:
    index = json.load(f)

# Load scenario data
df = pd.read_parquet("data/my_dataset/free_flow/episode_001_transitions.parquet")

# 3. Custom policy integration
from highway_datacollection.collection.action_samplers import ActionSampler

class MyPolicy(ActionSampler):
    def sample_actions(self, observations, n_agents, seed=None):
        # Your policy logic here
        return tuple(np.random.randint(0, 5, n_agents))

# 4. Custom feature extraction
from highway_datacollection.features.extractors import FeatureExtractor

class MyFeatures(FeatureExtractor):
    def extract_features(self, observation, context):
        # Your feature extraction logic here
        return {"custom_metric": 0.5}
""")
    
    print("\nFor more examples, see the 'examples/' directory.")
    print("For detailed documentation, see the 'docs/' directory.")
    print("\n" + "=" * 50)
    print("System demonstration complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()