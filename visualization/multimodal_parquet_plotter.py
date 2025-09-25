#!/usr/bin/env python3
"""
Multimodal Parquet Dataset Plotter

This script visualizes the hierarchical multimodal dataset stored in Parquet format,
including images, occupancy grids, kinematics data, and natural language summaries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import io
import pickle
from datetime import datetime
import glob

class MultimodalParquetPlotter:
    """Visualize multimodal highway dataset stored in Parquet format."""
    
    def __init__(self, dataset_root="data/highway_multimodal_dataset"):
        self.dataset_root = Path(dataset_root)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def load_dataset_index(self):
        """Load the global dataset catalog."""
        index_path = self.dataset_root / "index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                return json.load(f)
        return None
    
    def discover_parquet_files(self):
        """Discover all Parquet files in the dataset."""
        parquet_files = {}
        
        # Look for scenario folders
        for scenario_dir in self.dataset_root.iterdir():
            if scenario_dir.is_dir():
                scenario_name = scenario_dir.name
                parquet_files[scenario_name] = {
                    'transitions': list(scenario_dir.glob("*_transitions.parquet")),
                    'metadata': list(scenario_dir.glob("*_meta.jsonl"))
                }
        
        return parquet_files
    
    def load_episode_data(self, parquet_file):
        """Load episode transition data from Parquet file."""
        try:
            df = pd.read_parquet(parquet_file)
            return df
        except Exception as e:
            print(f"Error loading {parquet_file}: {e}")
            return None
    
    def decompress_binary_data(self, blob, shape_str, dtype_str):
        """Decompress binary data (occupancy grids or images)."""
        try:
            # Parse shape and dtype
            shape = eval(shape_str) if isinstance(shape_str, str) else shape_str
            dtype = np.dtype(dtype_str)
            
            # Decompress data (assuming it's pickled/compressed)
            data = pickle.loads(blob)
            
            # Reshape to original dimensions
            if isinstance(data, np.ndarray):
                return data.reshape(shape).astype(dtype)
            else:
                return np.array(data).reshape(shape).astype(dtype)
                
        except Exception as e:
            print(f"Error decompressing binary data: {e}")
            return None
    
    def plot_episode_overview(self, episode_df, episode_id, save_plots=True):
        """Plot comprehensive overview of a single episode."""
        print(f"📊 Plotting episode overview for {episode_id}...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Episode metadata
        fig.suptitle(f'Episode {episode_id} - Multimodal Analysis', fontsize=16, fontweight='bold')
        
        # 1. Trajectory plot (top-left, spans 2x2)
        ax_traj = fig.add_subplot(gs[0:2, 0:2])
        self.plot_trajectory(episode_df, ax_traj)
        
        # 2. Speed profile (top-right)
        ax_speed = fig.add_subplot(gs[0, 2:])
        self.plot_speed_profile(episode_df, ax_speed)
        
        # 3. Safety metrics (second row, right)
        ax_safety = fig.add_subplot(gs[1, 2:])
        self.plot_safety_metrics(episode_df, ax_safety)
        
        # 4. Sample occupancy grid (third row, left)
        ax_occ = fig.add_subplot(gs[2, 0])
        self.plot_sample_occupancy_grid(episode_df, ax_occ)
        
        # 5. Sample grayscale image (third row, second)
        ax_img = fig.add_subplot(gs[2, 1])
        self.plot_sample_image(episode_df, ax_img)
        
        # 6. Action distribution (third row, right)
        ax_actions = fig.add_subplot(gs[2, 2:])
        self.plot_action_distribution(episode_df, ax_actions)
        
        # 7. Natural language summaries (bottom row)
        ax_text = fig.add_subplot(gs[3, :])
        self.plot_text_summaries(episode_df, ax_text)
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"episode_{episode_id}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   💾 Episode overview saved as: {filename}")
        
        plt.show()
    
    def plot_trajectory(self, df, ax):
        """Plot vehicle trajectory."""
        ax.set_title("Vehicle Trajectory", fontweight='bold')
        
        # Plot trajectory colored by speed
        scatter = ax.scatter(df['ego_x'], df['ego_y'], c=df['speed'], 
                           cmap='viridis', s=20, alpha=0.7)
        
        # Add start and end markers
        ax.plot(df['ego_x'].iloc[0], df['ego_y'].iloc[0], 'go', markersize=10, label='Start')
        ax.plot(df['ego_x'].iloc[-1], df['ego_y'].iloc[-1], 'ro', markersize=10, label='End')
        
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Speed (m/s)")
    
    def plot_speed_profile(self, df, ax):
        """Plot speed over time."""
        ax.set_title("Speed Profile", fontweight='bold')
        
        ax.plot(df['step'], df['speed'], color='steelblue', linewidth=2)
        ax.fill_between(df['step'], df['speed'], alpha=0.3, color='steelblue')
        
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Speed (m/s)")
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_speed = df['speed'].mean()
        max_speed = df['speed'].max()
        ax.axhline(mean_speed, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_speed:.1f} m/s')
        ax.text(0.02, 0.98, f'Max: {max_speed:.1f} m/s', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.legend()
    
    def plot_safety_metrics(self, df, ax):
        """Plot safety metrics (TTC, traffic density)."""
        ax.set_title("Safety Metrics", fontweight='bold')
        
        # Create twin axis for different scales
        ax2 = ax.twinx()
        
        # Plot TTC
        line1 = ax.plot(df['step'], df['ttc'], color='red', linewidth=2, label='TTC (s)')
        ax.set_ylabel("Time to Collision (s)", color='red')
        ax.tick_params(axis='y', labelcolor='red')
        
        # Plot traffic density
        line2 = ax2.plot(df['step'], df['traffic_density'], color='blue', linewidth=2, label='Traffic Density')
        ax2.set_ylabel("Traffic Density", color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
    
    def plot_sample_occupancy_grid(self, df, ax):
        """Plot a sample occupancy grid."""
        ax.set_title("Sample Occupancy Grid", fontweight='bold')
        
        # Get a sample occupancy grid from the middle of the episode
        mid_idx = len(df) // 2
        sample_row = df.iloc[mid_idx]
        
        if 'occ_blob' in sample_row and pd.notna(sample_row['occ_blob']):
            try:
                occ_grid = self.decompress_binary_data(
                    sample_row['occ_blob'], 
                    sample_row['occ_shape'], 
                    sample_row['occ_dtype']
                )
                
                if occ_grid is not None:
                    im = ax.imshow(occ_grid, cmap='binary', origin='lower')
                    ax.set_xlabel("Grid X")
                    ax.set_ylabel("Grid Y")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.text(0.5, 0.5, "Could not\ndecompress\noccupancy grid", 
                           ha='center', va='center', transform=ax.transAxes)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\noccupancy grid:\n{str(e)[:30]}...", 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "No occupancy\ngrid data", 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    def plot_sample_image(self, df, ax):
        """Plot a sample grayscale image."""
        ax.set_title("Sample Camera Image", fontweight='bold')
        
        # Get a sample image from the middle of the episode
        mid_idx = len(df) // 2
        sample_row = df.iloc[mid_idx]
        
        if 'gray_blob' in sample_row and pd.notna(sample_row['gray_blob']):
            try:
                gray_img = self.decompress_binary_data(
                    sample_row['gray_blob'], 
                    sample_row['gray_shape'], 
                    sample_row['gray_dtype']
                )
                
                if gray_img is not None:
                    ax.imshow(gray_img, cmap='gray', origin='upper')
                    ax.set_xlabel("Image Width")
                    ax.set_ylabel("Image Height")
                else:
                    ax.text(0.5, 0.5, "Could not\ndecompress\nimage data", 
                           ha='center', va='center', transform=ax.transAxes)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading\nimage:\n{str(e)[:30]}...", 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "No image\ndata available", 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    def plot_action_distribution(self, df, ax):
        """Plot action distribution."""
        ax.set_title("Action Distribution", fontweight='bold')
        
        action_counts = df['action'].value_counts().sort_index()
        
        bars = ax.bar(action_counts.index, action_counts.values, 
                     color='lightcoral', alpha=0.8)
        
        ax.set_xlabel("Action")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, action_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom', fontweight='bold')
    
    def plot_text_summaries(self, df, ax):
        """Display sample natural language summaries."""
        ax.set_title("Natural Language Summaries (Sample)", fontweight='bold')
        
        # Get a few sample summaries
        sample_indices = [0, len(df)//3, 2*len(df)//3, len(df)-1]
        summaries = []
        
        for idx in sample_indices:
            if idx < len(df) and 'summary_text' in df.columns:
                summary = df.iloc[idx]['summary_text']
                if pd.notna(summary) and summary.strip():
                    step = df.iloc[idx]['step']
                    summaries.append(f"Step {step}: {summary[:100]}...")
        
        if summaries:
            summary_text = "\n\n".join(summaries)
        else:
            summary_text = "No natural language summaries available in this episode."
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', wrap=True,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def plot_scenario_comparison(self, save_plots=True):
        """Compare different scenarios in the dataset."""
        print("📊 Plotting scenario comparison...")
        
        parquet_files = self.discover_parquet_files()
        
        if not parquet_files:
            print("No Parquet files found!")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Scenario Comparison Across Dataset', fontsize=16, fontweight='bold')
        
        scenario_data = {}
        
        # Load data from each scenario
        for scenario_name, files in parquet_files.items():
            if files['transitions']:
                # Load first episode from each scenario
                df = self.load_episode_data(files['transitions'][0])
                if df is not None:
                    scenario_data[scenario_name] = df
        
        if not scenario_data:
            print("No valid scenario data found!")
            return
        
        # Plot comparisons
        self.plot_speed_comparison(scenario_data, axes[0, 0])
        self.plot_safety_comparison(scenario_data, axes[0, 1])
        self.plot_trajectory_comparison(scenario_data, axes[0, 2])
        self.plot_action_comparison(scenario_data, axes[1, 0])
        self.plot_episode_length_comparison(scenario_data, axes[1, 1])
        self.plot_reward_comparison(scenario_data, axes[1, 2])
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scenario_comparison_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   💾 Scenario comparison saved as: {filename}")
        
        plt.show()
    
    def plot_speed_comparison(self, scenario_data, ax):
        """Compare speed profiles across scenarios."""
        ax.set_title("Speed Profile Comparison", fontweight='bold')
        
        for i, (scenario, df) in enumerate(scenario_data.items()):
            color = self.colors[i % len(self.colors)]
            ax.plot(df['step'], df['speed'], label=scenario, color=color, alpha=0.8)
        
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Speed (m/s)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_safety_comparison(self, scenario_data, ax):
        """Compare safety metrics across scenarios."""
        ax.set_title("Average TTC Comparison", fontweight='bold')
        
        scenarios = list(scenario_data.keys())
        avg_ttc = [df['ttc'].mean() for df in scenario_data.values()]
        
        bars = ax.bar(scenarios, avg_ttc, color=self.colors[:len(scenarios)], alpha=0.8)
        ax.set_ylabel("Average TTC (s)")
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, avg_ttc):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    def plot_trajectory_comparison(self, scenario_data, ax):
        """Compare trajectory patterns across scenarios."""
        ax.set_title("Trajectory Patterns", fontweight='bold')
        
        for i, (scenario, df) in enumerate(scenario_data.items()):
            color = self.colors[i % len(self.colors)]
            ax.plot(df['ego_x'], df['ego_y'], label=scenario, color=color, alpha=0.7)
        
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_action_comparison(self, scenario_data, ax):
        """Compare action distributions across scenarios."""
        ax.set_title("Action Distribution Comparison", fontweight='bold')
        
        scenarios = list(scenario_data.keys())
        
        # Get all unique actions across scenarios (handle different data types)
        all_actions = set()
        for df in scenario_data.values():
            if 'action' in df.columns:
                # Convert to simple integers/floats to avoid numpy array issues
                unique_actions = df['action'].dropna().astype(int).unique()
                all_actions.update(unique_actions)
        
        if not all_actions:
            ax.text(0.5, 0.5, "No action data\navailable", 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        all_actions = sorted(list(all_actions))
        
        # Create stacked bar chart
        bottom = np.zeros(len(scenarios))
        
        for action in all_actions:
            values = []
            for scenario, df in scenario_data.items():
                if 'action' in df.columns:
                    count = (df['action'].astype(int) == action).sum()
                    values.append(count)
                else:
                    values.append(0)
            
            ax.bar(scenarios, values, bottom=bottom, label=f'Action {action}', alpha=0.8)
            bottom += np.array(values)
        
        ax.set_ylabel("Action Count")
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    
    def plot_episode_length_comparison(self, scenario_data, ax):
        """Compare episode lengths across scenarios."""
        ax.set_title("Episode Length Comparison", fontweight='bold')
        
        scenarios = list(scenario_data.keys())
        lengths = [len(df) for df in scenario_data.values()]
        
        bars = ax.bar(scenarios, lengths, color=self.colors[:len(scenarios)], alpha=0.8)
        ax.set_ylabel("Episode Length (steps)")
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, lengths):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   str(value), ha='center', va='bottom', fontweight='bold')
    
    def plot_reward_comparison(self, scenario_data, ax):
        """Compare reward distributions across scenarios."""
        ax.set_title("Reward Distribution", fontweight='bold')
        
        reward_data = []
        labels = []
        
        for scenario, df in scenario_data.items():
            if 'reward' in df.columns:
                reward_data.append(df['reward'].values)
                labels.append(scenario)
        
        if reward_data:
            ax.boxplot(reward_data, labels=labels)
            ax.set_ylabel("Reward")
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, "No reward data\navailable", 
                   ha='center', va='center', transform=ax.transAxes)
    
    def analyze_dataset(self):
        """Provide comprehensive dataset analysis."""
        print("\n" + "="*80)
        print("📊 MULTIMODAL DATASET ANALYSIS")
        print("="*80)
        
        # Load index
        index_data = self.load_dataset_index()
        if index_data:
            print(f"📁 Dataset Root: {self.dataset_root}")
            print(f"📈 Total Scenarios: {len(index_data.get('scenarios', {}))}")
            print(f"🎬 Total Episodes: {index_data.get('total_episodes', 'Unknown')}")
            print(f"📊 Data Types: {', '.join(index_data.get('data_types', []))}")
        
        # Discover files
        parquet_files = self.discover_parquet_files()
        
        total_episodes = 0
        total_steps = 0
        
        for scenario_name, files in parquet_files.items():
            print(f"\n🎭 Scenario: {scenario_name}")
            print(f"   📄 Transition files: {len(files['transitions'])}")
            print(f"   📝 Metadata files: {len(files['metadata'])}")
            
            # Analyze first file in scenario
            if files['transitions']:
                df = self.load_episode_data(files['transitions'][0])
                if df is not None:
                    episodes_in_file = df['episode_id'].nunique()
                    steps_in_file = len(df)
                    total_episodes += episodes_in_file
                    total_steps += steps_in_file
                    
                    print(f"   🎬 Episodes in first file: {episodes_in_file}")
                    print(f"   👣 Steps in first file: {steps_in_file}")
                    
                    # Data completeness check
                    completeness = {}
                    for col in ['occ_blob', 'gray_blob', 'summary_text']:
                        if col in df.columns:
                            non_null = df[col].notna().sum()
                            completeness[col] = f"{non_null}/{len(df)} ({non_null/len(df)*100:.1f}%)"
                    
                    if completeness:
                        print(f"   ✅ Data completeness:")
                        for data_type, stats in completeness.items():
                            print(f"      {data_type}: {stats}")
        
        print(f"\n📊 TOTAL DATASET SUMMARY:")
        print(f"   🎬 Estimated total episodes: {total_episodes}")
        print(f"   👣 Estimated total steps: {total_steps}")
        print("="*80)


def main():
    """Main function to demonstrate multimodal dataset plotting."""
    plotter = MultimodalParquetPlotter()
    
    print("🎨 Multimodal Parquet Dataset Plotter")
    print("=====================================")
    
    # Analyze dataset
    plotter.analyze_dataset()
    
    # Discover available data
    parquet_files = plotter.discover_parquet_files()
    
    if not parquet_files:
        print("❌ No Parquet files found in the dataset!")
        print(f"   Make sure your dataset is in: {plotter.dataset_root}")
        return
    
    # Plot scenario comparison
    plotter.plot_scenario_comparison(save_plots=True)
    
    # Plot individual episode examples
    for scenario_name, files in list(parquet_files.items())[:2]:  # First 2 scenarios
        if files['transitions']:
            print(f"\n📊 Plotting sample episode from {scenario_name}...")
            df = plotter.load_episode_data(files['transitions'][0])
            if df is not None:
                # Get first episode
                first_episode = df['episode_id'].iloc[0]
                episode_df = df[df['episode_id'] == first_episode]
                plotter.plot_episode_overview(episode_df, first_episode, save_plots=True)
    
    print("\n✅ Multimodal dataset visualization complete!")
    print("📁 Check the generated PNG files for all visualizations.")


if __name__ == "__main__":
    main()