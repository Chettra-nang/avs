#!/usr/bin/env python3
"""
Data loading examples for HighwayEnv Multi-Modal Data Collection datasets.

This module provides comprehensive examples of how to load, access, and
process collected multi-modal datasets for various use cases.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DatasetLoader:
    """
    Utility class for loading and accessing multi-modal highway datasets.
    
    This class provides convenient methods for loading datasets created by
    the HighwayEnv Multi-Modal Data Collection system.
    """
    
    def __init__(self, dataset_path: Path):
        """
        Initialize dataset loader.
        
        Args:
            dataset_path: Path to the dataset directory containing index.json
        """
        self.dataset_path = Path(dataset_path)
        self.index_path = self.dataset_path / "index.json"
        self.index_data = None
        
        if self.index_path.exists():
            self.load_index()
    
    def load_index(self) -> Dict[str, Any]:
        """Load dataset index file."""
        with open(self.index_path, 'r') as f:
            self.index_data = json.load(f)
        return self.index_data
    
    def list_scenarios(self) -> List[str]:
        """Get list of available scenarios."""
        if self.index_data is None:
            return []
        return list(self.index_data['scenarios'].keys())
    
    def get_scenario_info(self, scenario_name: str) -> Dict[str, Any]:
        """Get information about a specific scenario."""
        if self.index_data is None or scenario_name not in self.index_data['scenarios']:
            return {}
        return self.index_data['scenarios'][scenario_name]
    
    def load_scenario_data(self, scenario_name: str) -> pd.DataFrame:
        """
        Load all data for a specific scenario.
        
        Args:
            scenario_name: Name of the scenario to load
            
        Returns:
            Combined DataFrame with all episodes for the scenario
        """
        scenario_info = self.get_scenario_info(scenario_name)
        if not scenario_info:
            return pd.DataFrame()
        
        scenario_path = self.dataset_path / scenario_name
        dfs = []
        
        for file_name in scenario_info['files']:
            file_path = scenario_path / file_name
            if file_path.exists() and file_name.endswith('.parquet'):
                df = pd.read_parquet(file_path)
                dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def load_episode_data(self, scenario_name: str, episode_id: str) -> pd.DataFrame:
        """
        Load data for a specific episode.
        
        Args:
            scenario_name: Name of the scenario
            episode_id: ID of the episode to load
            
        Returns:
            DataFrame with data for the specified episode
        """
        scenario_df = self.load_scenario_data(scenario_name)
        if scenario_df.empty:
            return pd.DataFrame()
        
        return scenario_df[scenario_df['episode_id'] == episode_id]
    
    def load_agent_trajectory(self, scenario_name: str, episode_id: str, agent_id: int) -> pd.DataFrame:
        """
        Load trajectory for a specific agent in an episode.
        
        Args:
            scenario_name: Name of the scenario
            episode_id: ID of the episode
            agent_id: ID of the agent
            
        Returns:
            DataFrame with trajectory data for the specified agent
        """
        episode_df = self.load_episode_data(scenario_name, episode_id)
        if episode_df.empty:
            return pd.DataFrame()
        
        return episode_df[episode_df['agent_id'] == agent_id].sort_values('step')
    
    def reconstruct_observation(self, row: pd.Series, modality: str = 'occ') -> Optional[np.ndarray]:
        """
        Reconstruct binary observation array from dataset row.
        
        Args:
            row: DataFrame row containing binary blob data
            modality: Modality to reconstruct ('occ' for occupancy grid, 'gray' for grayscale)
            
        Returns:
            Reconstructed numpy array or None if data not available
        """
        blob_col = f'{modality}_blob'
        shape_col = f'{modality}_shape'
        dtype_col = f'{modality}_dtype'
        
        if not all(col in row.index for col in [blob_col, shape_col, dtype_col]):
            return None
        
        if pd.isna(row[blob_col]):
            return None
        
        try:
            blob_data = row[blob_col]
            shape = eval(row[shape_col]) if isinstance(row[shape_col], str) else row[shape_col]
            dtype = row[dtype_col]
            
            return np.frombuffer(blob_data, dtype=dtype).reshape(shape)
        except Exception:
            return None
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the dataset."""
        if self.index_data is None:
            return {}
        
        stats = {
            'total_scenarios': len(self.index_data['scenarios']),
            'total_files': self.index_data['total_files'],
            'created_at': self.index_data['created_at'],
            'scenarios': {}
        }
        
        for scenario_name in self.list_scenarios():
            scenario_df = self.load_scenario_data(scenario_name)
            if not scenario_df.empty:
                stats['scenarios'][scenario_name] = {
                    'episodes': scenario_df['episode_id'].nunique(),
                    'total_steps': len(scenario_df),
                    'agents': scenario_df['agent_id'].nunique(),
                    'avg_reward': scenario_df['reward'].mean(),
                    'avg_episode_length': len(scenario_df) / scenario_df['episode_id'].nunique()
                }
        
        return stats


def example_basic_loading():
    """Example 1: Basic dataset loading and exploration."""
    print("Example 1: Basic Dataset Loading")
    print("-" * 40)
    
    # Initialize loader
    dataset_path = Path("data/highway_multimodal_dataset")
    loader = DatasetLoader(dataset_path)
    
    if not loader.index_path.exists():
        print(f"Dataset not found at {dataset_path}")
        print("Run main.py --demo collection first to create a dataset")
        return
    
    # Get dataset overview
    scenarios = loader.list_scenarios()
    print(f"Available scenarios: {scenarios}")
    
    # Load data for first scenario
    if scenarios:
        scenario_name = scenarios[0]
        df = loader.load_scenario_data(scenario_name)
        
        print(f"\nLoaded scenario '{scenario_name}':")
        print(f"  - Shape: {df.shape}")
        print(f"  - Episodes: {df['episode_id'].nunique()}")
        print(f"  - Agents: {df['agent_id'].nunique()}")
        print(f"  - Columns: {list(df.columns)}")
        
        # Show sample data
        print(f"\nSample data:")
        sample_cols = ['episode_id', 'step', 'agent_id', 'action', 'reward']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head().to_string(index=False))


def example_episode_analysis():
    """Example 2: Episode-level analysis and trajectory extraction."""
    print("\nExample 2: Episode Analysis")
    print("-" * 40)
    
    dataset_path = Path("data/highway_multimodal_dataset")
    loader = DatasetLoader(dataset_path)
    
    scenarios = loader.list_scenarios()
    if not scenarios:
        print("No scenarios available")
        return
    
    scenario_name = scenarios[0]
    scenario_df = loader.load_scenario_data(scenario_name)
    
    if scenario_df.empty:
        print(f"No data available for scenario {scenario_name}")
        return
    
    # Analyze first episode
    first_episode = scenario_df['episode_id'].iloc[0]
    episode_df = loader.load_episode_data(scenario_name, first_episode)
    
    print(f"Episode analysis for {first_episode}:")
    print(f"  - Duration: {episode_df['step'].max()} steps")
    print(f"  - Agents: {sorted(episode_df['agent_id'].unique())}")
    print(f"  - Total reward: {episode_df['reward'].sum():.3f}")
    
    # Analyze each agent's trajectory
    for agent_id in sorted(episode_df['agent_id'].unique()):
        trajectory = loader.load_agent_trajectory(scenario_name, first_episode, agent_id)
        
        print(f"\n  Agent {agent_id} trajectory:")
        print(f"    - Steps: {len(trajectory)}")
        print(f"    - Actions: {sorted(trajectory['action'].unique())}")
        print(f"    - Reward: {trajectory['reward'].sum():.3f}")
        
        if 'ttc' in trajectory.columns:
            finite_ttc = trajectory['ttc'].replace([np.inf, -np.inf], np.nan).dropna()
            if not finite_ttc.empty:
                print(f"    - Avg TTC: {finite_ttc.mean():.3f}s")


def example_binary_data_reconstruction():
    """Example 3: Binary observation data reconstruction."""
    print("\nExample 3: Binary Data Reconstruction")
    print("-" * 40)
    
    dataset_path = Path("data/highway_multimodal_dataset")
    loader = DatasetLoader(dataset_path)
    
    scenarios = loader.list_scenarios()
    if not scenarios:
        print("No scenarios available")
        return
    
    scenario_df = loader.load_scenario_data(scenarios[0])
    if scenario_df.empty:
        print("No data available")
        return
    
    # Find row with binary data
    sample_row = None
    for _, row in scenario_df.head(10).iterrows():
        if 'occ_blob' in row.index and not pd.isna(row['occ_blob']):
            sample_row = row
            break
    
    if sample_row is None:
        print("No binary observation data found in sample")
        return
    
    print(f"Reconstructing observations for episode {sample_row['episode_id']}, step {sample_row['step']}")
    
    # Reconstruct occupancy grid
    occ_grid = loader.reconstruct_observation(sample_row, 'occ')
    if occ_grid is not None:
        print(f"  Occupancy grid: shape {occ_grid.shape}, dtype {occ_grid.dtype}")
        print(f"    Value range: [{occ_grid.min():.3f}, {occ_grid.max():.3f}]")
        print(f"    Non-zero elements: {np.count_nonzero(occ_grid)}")
    
    # Reconstruct grayscale image
    grayscale = loader.reconstruct_observation(sample_row, 'gray')
    if grayscale is not None:
        print(f"  Grayscale image: shape {grayscale.shape}, dtype {grayscale.dtype}")
        print(f"    Value range: [{grayscale.min():.3f}, {grayscale.max():.3f}]")


def example_feature_analysis():
    """Example 4: Feature analysis and correlation study."""
    print("\nExample 4: Feature Analysis")
    print("-" * 40)
    
    dataset_path = Path("data/highway_multimodal_dataset")
    loader = DatasetLoader(dataset_path)
    
    # Load all data
    all_data = []
    for scenario_name in loader.list_scenarios():
        df = loader.load_scenario_data(scenario_name)
        if not df.empty:
            df['scenario'] = scenario_name
            all_data.append(df)
    
    if not all_data:
        print("No data available for analysis")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Feature statistics
    print("Feature statistics across all scenarios:")
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['episode_id', 'step', 'agent_id']]
    
    if feature_cols:
        stats = combined_df[feature_cols].describe()
        print(stats.round(3).to_string())
    
    # Scenario comparison
    print("\nScenario comparison:")
    scenario_stats = combined_df.groupby('scenario').agg({
        'reward': ['mean', 'std'],
        'action': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
    }).round(3)
    print(scenario_stats.to_string())
    
    # Feature correlations
    if len(feature_cols) >= 2:
        print("\nFeature correlations:")
        correlation_cols = [col for col in ['reward', 'ttc'] if col in feature_cols]
        if len(correlation_cols) >= 2:
            # Handle infinite values in TTC
            corr_df = combined_df[correlation_cols].copy()
            if 'ttc' in corr_df.columns:
                corr_df['ttc'] = corr_df['ttc'].replace([np.inf, -np.inf], np.nan)
            
            correlation = corr_df.corr()
            print(correlation.round(3).to_string())


def example_time_series_analysis():
    """Example 5: Time series analysis of agent behavior."""
    print("\nExample 5: Time Series Analysis")
    print("-" * 40)
    
    dataset_path = Path("data/highway_multimodal_dataset")
    loader = DatasetLoader(dataset_path)
    
    scenarios = loader.list_scenarios()
    if not scenarios:
        print("No scenarios available")
        return
    
    scenario_df = loader.load_scenario_data(scenarios[0])
    if scenario_df.empty:
        print("No data available")
        return
    
    # Analyze first episode
    first_episode = scenario_df['episode_id'].iloc[0]
    episode_df = loader.load_episode_data(scenarios[0], first_episode)
    
    print(f"Time series analysis for episode {first_episode}:")
    
    # Aggregate metrics by time step
    time_series = episode_df.groupby('step').agg({
        'reward': ['mean', 'sum'],
        'action': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0,
        'agent_id': 'count'
    }).round(3)
    
    time_series.columns = ['avg_reward', 'total_reward', 'most_common_action', 'active_agents']
    
    print("Time series data (first 10 steps):")
    print(time_series.head(10).to_string())
    
    # Action distribution over time
    if 'action' in episode_df.columns:
        print(f"\nAction distribution over episode:")
        action_counts = episode_df['action'].value_counts().sort_index()
        for action, count in action_counts.items():
            percentage = (count / len(episode_df)) * 100
            print(f"  Action {action}: {count} times ({percentage:.1f}%)")


def example_custom_analysis():
    """Example 6: Custom analysis functions."""
    print("\nExample 6: Custom Analysis Functions")
    print("-" * 40)
    
    def analyze_driving_behavior(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze driving behavior patterns from trajectory data."""
        if df.empty:
            return {}
        
        analysis = {}
        
        # Action patterns
        if 'action' in df.columns:
            action_dist = df['action'].value_counts(normalize=True)
            analysis['action_distribution'] = action_dist.to_dict()
            analysis['most_common_action'] = action_dist.index[0]
        
        # Reward patterns
        if 'reward' in df.columns:
            analysis['total_reward'] = df['reward'].sum()
            analysis['avg_reward'] = df['reward'].mean()
            analysis['reward_volatility'] = df['reward'].std()
        
        # Safety metrics
        if 'ttc' in df.columns:
            finite_ttc = df['ttc'].replace([np.inf, -np.inf], np.nan).dropna()
            if not finite_ttc.empty:
                analysis['avg_ttc'] = finite_ttc.mean()
                analysis['min_ttc'] = finite_ttc.min()
                analysis['dangerous_situations'] = (finite_ttc < 2.0).sum()
        
        return analysis
    
    def compare_agents(df: pd.DataFrame) -> pd.DataFrame:
        """Compare performance across different agents."""
        if df.empty or 'agent_id' not in df.columns:
            return pd.DataFrame()
        
        agent_stats = []
        for agent_id in df['agent_id'].unique():
            agent_df = df[df['agent_id'] == agent_id]
            stats = analyze_driving_behavior(agent_df)
            stats['agent_id'] = agent_id
            stats['total_steps'] = len(agent_df)
            agent_stats.append(stats)
        
        return pd.DataFrame(agent_stats)
    
    # Apply custom analysis
    dataset_path = Path("data/highway_multimodal_dataset")
    loader = DatasetLoader(dataset_path)
    
    scenarios = loader.list_scenarios()
    if scenarios:
        scenario_df = loader.load_scenario_data(scenarios[0])
        
        if not scenario_df.empty:
            # Overall behavior analysis
            overall_analysis = analyze_driving_behavior(scenario_df)
            print("Overall driving behavior analysis:")
            for key, value in overall_analysis.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
                else:
                    print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
            
            # Agent comparison
            agent_comparison = compare_agents(scenario_df)
            if not agent_comparison.empty:
                print(f"\nAgent comparison:")
                print(agent_comparison.round(3).to_string(index=False))


def main():
    """Run all data loading examples."""
    print("HighwayEnv Multi-Modal Dataset Loading Examples")
    print("=" * 50)
    
    examples = [
        example_basic_loading,
        example_episode_analysis,
        example_binary_data_reconstruction,
        example_feature_analysis,
        example_time_series_analysis,
        example_custom_analysis
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Error in {example_func.__name__}: {e}")
        print()  # Add spacing between examples
    
    print("=" * 50)
    print("Data loading examples completed!")


if __name__ == "__main__":
    main()