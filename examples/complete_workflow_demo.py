#!/usr/bin/env python3
"""
Complete workflow demonstration for HighwayEnv Multi-Modal Data Collection.

This example shows the complete end-to-end workflow from data collection
to loading and analysis, demonstrating all major system capabilities.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from highway_datacollection.collection.orchestrator import run_full_collection
from highway_datacollection.storage.manager import DatasetStorageManager
from highway_datacollection import ScenarioRegistry


def demonstrate_complete_workflow():
    """Demonstrate complete data collection and analysis workflow."""
    print("Complete Workflow Demonstration")
    print("=" * 50)
    
    # Step 1: Configure collection parameters
    print("Step 1: Configuring collection parameters")
    storage_path = Path("data/complete_workflow_demo")
    
    collection_config = {
        "episodes_per_scenario": 10,
        "n_agents": 2,
        "max_steps_per_episode": 100,
        "scenarios": ["free_flow", "dense_commuting", "stop_and_go"],
        "base_seed": 12345,
        "batch_size": 5
    }
    
    print(f"  Storage path: {storage_path}")
    for key, value in collection_config.items():
        print(f"  {key}: {value}")
    
    # Step 2: Run data collection
    print("\nStep 2: Running data collection")
    try:
        result = run_full_collection(
            base_storage_path=storage_path,
            **collection_config
        )
        
        print(f"  ✓ Collection completed successfully")
        print(f"  - Total episodes: {result.total_episodes}")
        print(f"  - Successful episodes: {result.successful_episodes}")
        print(f"  - Collection time: {result.collection_time:.2f}s")
        
    except Exception as e:
        print(f"  ✗ Collection failed: {e}")
        return
    
    # Step 3: Load and analyze collected data
    print("\nStep 3: Loading and analyzing collected data")
    
    # Load dataset index
    index_path = storage_path / "index.json"
    with open(index_path, 'r') as f:
        index_data = json.load(f)
    
    print(f"  ✓ Loaded dataset index")
    print(f"  - Scenarios: {list(index_data['scenarios'].keys())}")
    print(f"  - Total files: {index_data['total_files']}")
    
    # Analyze each scenario
    all_data = []
    for scenario_name, scenario_info in index_data['scenarios'].items():
        print(f"\n  Analyzing scenario: {scenario_name}")
        
        scenario_data = []
        for file_name in scenario_info['files']:
            file_path = storage_path / scenario_name / file_name
            if file_path.exists():
                df = pd.read_parquet(file_path)
                scenario_data.append(df)
        
        if scenario_data:
            scenario_df = pd.concat(scenario_data, ignore_index=True)
            all_data.append(scenario_df)
            
            # Basic statistics
            print(f"    - Episodes: {scenario_df['episode_id'].nunique()}")
            print(f"    - Total steps: {len(scenario_df)}")
            print(f"    - Avg reward: {scenario_df['reward'].mean():.3f}")
            print(f"    - Avg TTC: {scenario_df['ttc'].replace([np.inf, -np.inf], np.nan).mean():.3f}s")
    
    # Step 4: Cross-scenario analysis
    if all_data:
        print("\nStep 4: Cross-scenario analysis")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Add scenario labels
        scenario_labels = []
        for _, row in combined_df.iterrows():
            episode_id = row['episode_id']
            # Extract scenario from episode_id (assuming format includes scenario)
            for scenario in collection_config['scenarios']:
                if scenario in episode_id:
                    scenario_labels.append(scenario)
                    break
            else:
                scenario_labels.append('unknown')
        
        combined_df['scenario'] = scenario_labels
        
        # Comparative analysis
        print("  Comparative analysis across scenarios:")
        scenario_stats = combined_df.groupby('scenario').agg({
            'reward': ['mean', 'std'],
            'ttc': lambda x: x.replace([np.inf, -np.inf], np.nan).mean(),
            'episode_id': 'nunique'
        }).round(3)
        
        print(scenario_stats.to_string())
        
        # Feature correlation analysis
        print("\n  Feature correlation analysis:")
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        correlation_cols = [col for col in ['reward', 'ttc'] if col in numeric_cols]
        
        if len(correlation_cols) >= 2:
            correlation = combined_df[correlation_cols].corr()
            print(correlation.to_string())
    
    # Step 5: Demonstrate data access patterns
    print("\nStep 5: Data access patterns")
    
    print("  Pattern 1: Episode-based access")
    if all_data:
        sample_episode = combined_df['episode_id'].iloc[0]
        episode_data = combined_df[combined_df['episode_id'] == sample_episode]
        print(f"    Episode {sample_episode}: {len(episode_data)} steps")
        print(f"    Agents: {sorted(episode_data['agent_id'].unique())}")
        print(f"    Actions taken: {sorted(episode_data['action'].unique())}")
    
    print("  Pattern 2: Agent-based access")
    if all_data:
        agent_stats = combined_df.groupby('agent_id').agg({
            'reward': 'mean',
            'action': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
        }).round(3)
        print("    Per-agent statistics:")
        print(agent_stats.to_string())
    
    print("  Pattern 3: Time-series access")
    if all_data:
        sample_episode_data = combined_df[combined_df['episode_id'] == sample_episode]
        time_series = sample_episode_data.groupby('step').agg({
            'reward': 'mean',
            'ttc': lambda x: x.replace([np.inf, -np.inf], np.nan).mean()
        }).round(3)
        print("    Time series for sample episode (first 5 steps):")
        print(time_series.head().to_string())
    
    print(f"\n✓ Complete workflow demonstration finished successfully!")
    print(f"Dataset saved to: {storage_path}")
    print(f"Use the dataset for training, analysis, or further research.")


if __name__ == "__main__":
    demonstrate_complete_workflow()