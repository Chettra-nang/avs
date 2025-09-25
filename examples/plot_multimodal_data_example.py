#!/usr/bin/env python3
"""
Example: How to Plot Multimodal Parquet Data

This example demonstrates how to visualize your hierarchical multimodal dataset
with images, occupancy grids, and natural language summaries.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.multimodal_parquet_plotter import MultimodalParquetPlotter
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def create_sample_data():
    """Create sample multimodal data for demonstration."""
    print("🔧 Creating sample multimodal data...")
    
    # Create sample dataset structure
    dataset_root = Path("data/highway_multimodal_dataset")
    dataset_root.mkdir(parents=True, exist_ok=True)
    
    # Create sample scenarios
    scenarios = ["free_flow", "dense_commuting", "stop_and_go"]
    
    for scenario in scenarios:
        scenario_dir = dataset_root / scenario
        scenario_dir.mkdir(exist_ok=True)
        
        # Create sample episode data
        episode_data = []
        
        for step in range(100):  # 100 steps per episode
            # Create sample occupancy grid (20x20)
            occ_grid = np.random.randint(0, 2, (20, 20), dtype=np.uint8)
            occ_blob = pickle.dumps(occ_grid)
            
            # Create sample grayscale image (64x64)
            gray_img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
            gray_blob = pickle.dumps(gray_img)
            
            # Sample natural language summary
            summaries = [
                "Vehicle maintaining steady speed in light traffic",
                "Approaching slower vehicle, considering lane change",
                "Heavy traffic detected, reducing speed",
                "Clear road ahead, accelerating to target speed",
                "Following vehicle at safe distance"
            ]
            
            episode_data.append({
                'episode_id': f'ep_20250921_001',
                'step': step,
                'agent_id': 0,
                'action': int(np.random.randint(0, 5)),
                'reward': float(np.random.normal(0.1, 0.5)),
                'ego_x': float(100 + step * 2 + np.random.normal(0, 0.5)),
                'ego_y': float(50 + np.sin(step * 0.1) * 5 + np.random.normal(0, 0.2)),
                'ego_vx': float(20 + np.random.normal(0, 2)),
                'ego_vy': float(np.random.normal(0, 0.5)),
                'speed': float(20 + np.random.normal(0, 2)),
                'lane_position': float(np.random.normal(0, 0.3)),
                'ttc': float(np.random.exponential(5)),
                'traffic_density': float(np.random.uniform(0.1, 0.8)),
                'lead_vehicle_gap': float(np.random.exponential(30)),
                'vehicle_count': int(np.random.randint(3, 15)),
                'occ_blob': occ_blob,
                'occ_shape': str((20, 20)),
                'occ_dtype': 'uint8',
                'gray_blob': gray_blob,
                'gray_shape': str((64, 64)),
                'gray_dtype': 'uint8',
                'summary_text': np.random.choice(summaries)
            })
        
        # Save as Parquet
        df = pd.DataFrame(episode_data)
        parquet_file = scenario_dir / "ep_20250921_001_transitions.parquet"
        df.to_parquet(parquet_file, index=False)
        
        print(f"   ✅ Created sample data for {scenario}")
    
    # Create index file
    index_data = {
        "dataset_version": "1.0",
        "creation_date": "2025-01-21",
        "total_episodes": 3,
        "data_types": ["transitions", "occupancy_grids", "grayscale_images", "natural_language"],
        "scenarios": {
            scenario: {
                "description": f"Sample {scenario.replace('_', ' ')} scenario",
                "episodes": ["ep_20250921_001"]
            } for scenario in scenarios
        }
    }
    
    import json
    with open(dataset_root / "index.json", 'w') as f:
        json.dump(index_data, f, indent=2)
    
    print(f"✅ Sample dataset created at: {dataset_root}")
    return dataset_root

def demonstrate_plotting():
    """Demonstrate different plotting capabilities."""
    print("\n🎨 Demonstrating Multimodal Data Plotting")
    print("==========================================")
    
    # Create sample data if it doesn't exist
    dataset_root = Path("data/highway_multimodal_dataset")
    if not dataset_root.exists() or not list(dataset_root.glob("*/ep_*_transitions.parquet")):
        dataset_root = create_sample_data()
    
    # Initialize plotter
    plotter = MultimodalParquetPlotter(dataset_root)
    
    print("\n1. 📊 Dataset Analysis")
    print("-" * 30)
    plotter.analyze_dataset()
    
    print("\n2. 🎭 Scenario Comparison")
    print("-" * 30)
    plotter.plot_scenario_comparison(save_plots=True)
    
    print("\n3. 🎬 Individual Episode Analysis")
    print("-" * 30)
    
    # Plot individual episodes from each scenario
    parquet_files = plotter.discover_parquet_files()
    
    for scenario_name, files in parquet_files.items():
        if files['transitions']:
            print(f"\n   📊 Plotting episode from {scenario_name}...")
            
            # Load episode data
            df = plotter.load_episode_data(files['transitions'][0])
            if df is not None:
                # Get first episode
                first_episode = df['episode_id'].iloc[0]
                episode_df = df[df['episode_id'] == first_episode]
                
                # Plot comprehensive overview
                plotter.plot_episode_overview(episode_df, first_episode, save_plots=True)
    
    print("\n✅ All plotting demonstrations complete!")
    print("📁 Check the generated PNG files for visualizations.")

def plot_specific_episode():
    """Example of plotting a specific episode by ID."""
    print("\n🎯 Plotting Specific Episode")
    print("-" * 30)
    
    dataset_root = Path("data/highway_multimodal_dataset")
    plotter = MultimodalParquetPlotter(dataset_root)
    
    # Find and plot a specific episode
    parquet_files = plotter.discover_parquet_files()
    
    target_episode = "ep_20250921_001"
    
    for scenario_name, files in parquet_files.items():
        for parquet_file in files['transitions']:
            df = plotter.load_episode_data(parquet_file)
            if df is not None and target_episode in df['episode_id'].values:
                print(f"   📍 Found episode {target_episode} in {scenario_name}")
                
                episode_df = df[df['episode_id'] == target_episode]
                plotter.plot_episode_overview(episode_df, target_episode, save_plots=True)
                return
    
    print(f"   ❌ Episode {target_episode} not found in dataset")

def analyze_natural_language_summaries():
    """Example of analyzing natural language summaries."""
    print("\n📝 Analyzing Natural Language Summaries")
    print("-" * 40)
    
    dataset_root = Path("data/highway_multimodal_dataset")
    plotter = MultimodalParquetPlotter(dataset_root)
    
    parquet_files = plotter.discover_parquet_files()
    
    all_summaries = []
    
    for scenario_name, files in parquet_files.items():
        for parquet_file in files['transitions']:
            df = plotter.load_episode_data(parquet_file)
            if df is not None and 'summary_text' in df.columns:
                summaries = df['summary_text'].dropna().tolist()
                all_summaries.extend(summaries)
                
                print(f"\n   📊 {scenario_name}: {len(summaries)} summaries")
                
                # Show sample summaries
                for i, summary in enumerate(summaries[:3]):
                    print(f"      {i+1}. {summary}")
    
    print(f"\n📈 Total summaries across dataset: {len(all_summaries)}")
    
    # Simple text analysis
    if all_summaries:
        import collections
        
        # Word frequency analysis
        all_words = []
        for summary in all_summaries:
            words = summary.lower().split()
            all_words.extend(words)
        
        word_freq = collections.Counter(all_words)
        print(f"\n🔤 Most common words:")
        for word, count in word_freq.most_common(10):
            print(f"      {word}: {count}")

def main():
    """Main demonstration function."""
    print("🚗 Multimodal Highway Dataset Plotting Examples")
    print("=" * 50)
    
    try:
        # Full demonstration
        demonstrate_plotting()
        
        # Specific episode plotting
        plot_specific_episode()
        
        # Natural language analysis
        analyze_natural_language_summaries()
        
        print("\n🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("1. Adapt the plotter to your actual data schema")
        print("2. Modify decompression logic for your binary data format")
        print("3. Customize visualizations for your specific use case")
        print("4. Add more analysis functions as needed")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()