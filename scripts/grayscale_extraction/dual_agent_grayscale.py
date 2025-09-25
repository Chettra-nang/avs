#!/usr/bin/env python3
"""
Dual-Agent Grayscale Image Extractor

This script extracts and visualizes grayscale images for 2 agents from your highway dataset.
It provides multiple viewing options and analysis capabilities.

Usage:
    python dual_agent_grayscale.py [options]

Examples:
    # Extract first 5 steps of first episode
    python dual_agent_grayscale.py
    
    # Extract specific episode and steps
    python dual_agent_grayscale.py --episode ep_dense_commuting_10043_0001 --steps 0 1 2 3
    
    # Create comparison grid
    python dual_agent_grayscale.py --comparison-grid --steps 0 2 4 6 8
    
    # Extract all available steps
    python dual_agent_grayscale.py --all-steps
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from highway_datacollection.storage.encoders import BinaryArrayEncoder


class DualAgentGrayscaleExtractor:
    """Extract and visualize grayscale images for 2 agents."""
    
    def __init__(self, data_path: str, output_dir: Optional[str] = None):
        """
        Initialize the extractor.
        
        Args:
            data_path: Path to parquet file
            output_dir: Output directory (default: same as data file)
        """
        self.data_path = Path(data_path)
        self.decoder = BinaryArrayEncoder()
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.data_path.parent / "grayscale_images"
        
        self.output_dir.mkdir(exist_ok=True)
        self.df = None
        
        print(f"🚗🚗 Dual-Agent Grayscale Extractor")
        print(f"📁 Data: {self.data_path.name}")
        print(f"💾 Output: {self.output_dir}")
    
    def load_data(self):
        """Load and analyze the dataset."""
        print("\\n📊 Loading dataset...")
        self.df = pd.read_parquet(self.data_path)
        
        # Basic analysis
        episodes = self.df['episode_id'].unique()
        agents = sorted(self.df['agent_id'].unique())
        steps_range = (self.df['step'].min(), self.df['step'].max())
        
        print(f"✅ Dataset loaded:")
        print(f"   📊 Total rows: {len(self.df)}")
        print(f"   🎬 Episodes: {len(episodes)}")
        print(f"   🚗 Agents: {agents}")
        print(f"   📈 Steps range: {steps_range[0]} - {steps_range[1]}")
        
        # Check grayscale data availability
        grayscale_count = self.df['grayscale_blob'].notna().sum()
        coverage = 100 * grayscale_count / len(self.df)
        print(f"   🖼️  Grayscale coverage: {grayscale_count}/{len(self.df)} ({coverage:.1f}%)")
        
        return self.df
    
    def decode_grayscale_blob(self, blob: bytes, shape: np.ndarray, dtype: str) -> Dict:
        """
        Decode grayscale blob and create different visualizations.
        
        Returns:
            Dictionary with different image interpretations
        """
        try:
            # Decode using proper decoder
            raw_array = self.decoder.decode(blob, tuple(shape), dtype)
            
            result = {
                'raw_array': raw_array,
                'shape': raw_array.shape,
                'dtype': raw_array.dtype,
                'value_range': (float(raw_array.min()), float(raw_array.max())),
                'mean': float(raw_array.mean()),
                'std': float(raw_array.std())
            }
            
            # Create different image representations
            images = {}
            
            # Assume shape is [channels, height, width] = [4, 128, 64]
            if len(raw_array.shape) == 3 and raw_array.shape[0] <= 4:
                # Individual channels
                channel_names = ['Red', 'Green', 'Blue', 'Alpha']
                for i in range(raw_array.shape[0]):
                    channel_img = self._normalize_for_display(raw_array[i])
                    images[f'channel_{i}'] = {
                        'image': channel_img,
                        'name': channel_names[i] if i < len(channel_names) else f'Channel_{i}',
                        'stats': self._get_image_stats(channel_img)
                    }
                
                # RGB composite (if we have at least 3 channels)
                if raw_array.shape[0] >= 3:
                    rgb_gray = (0.299 * raw_array[0] + 0.587 * raw_array[1] + 0.114 * raw_array[2])
                    rgb_gray = self._normalize_for_display(rgb_gray)
                    images['rgb_grayscale'] = {
                        'image': rgb_gray,
                        'name': 'RGB Grayscale',
                        'stats': self._get_image_stats(rgb_gray)
                    }
                
                # Best grayscale representation (use RGB if available, otherwise first channel)
                if 'rgb_grayscale' in images:
                    images['best'] = images['rgb_grayscale']
                else:
                    images['best'] = images['channel_0']
                    images['best']['name'] = 'Grayscale'
            
            result['images'] = images
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def _normalize_for_display(self, array: np.ndarray) -> np.ndarray:
        """Normalize array for display [0, 1]."""
        array = array.astype(np.float32)
        if array.max() > 1.0:
            array = array / 255.0
        return np.clip(array, 0, 1)
    
    def _get_image_stats(self, image: np.ndarray) -> Dict:
        """Get statistics for an image."""
        return {
            'min': float(image.min()),
            'max': float(image.max()),
            'mean': float(image.mean()),
            'std': float(image.std()),
            'non_zero': float(np.count_nonzero(image))
        }
    
    def extract_step_data(self, episode_id: str, step: int) -> Dict:
        """Extract complete data for both agents at a specific step."""
        step_data = self.df[(self.df['episode_id'] == episode_id) & (self.df['step'] == step)]
        
        agents_data = {}
        for _, row in step_data.iterrows():
            agent_id = row['agent_id']
            
            # Extract grayscale data
            grayscale_data = None
            if pd.notna(row['grayscale_blob']):
                grayscale_data = self.decode_grayscale_blob(
                    row['grayscale_blob'],
                    row['grayscale_shape'],
                    row['grayscale_dtype']
                )
            
            # Compile agent data
            agents_data[agent_id] = {
                'ego_x': row['ego_x'],
                'ego_y': row['ego_y'],
                'ego_vx': row['ego_vx'], 
                'ego_vy': row['ego_vy'],
                'lane_position': row['lane_position'],
                'speed': np.sqrt(row['ego_vx']**2 + row['ego_vy']**2),
                'action': row['action'],
                'reward': row['reward'],
                'grayscale': grayscale_data
            }
        
        return agents_data
    
    def create_dual_view_comparison(self, agents_data: Dict, episode_id: str, step: int, 
                                  view_type: str = 'best') -> plt.Figure:
        """Create side-by-side comparison of both agents."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Dual-Agent Highway Views\\nEpisode: {episode_id}, Step: {step}', 
                    fontsize=16, fontweight='bold')
        
        agent_colors = ['#FF4444', '#4444FF']  # Red and Blue
        agent_names = ['Agent 0', 'Agent 1']
        
        for i, (agent_id, data) in enumerate(sorted(agents_data.items())):
            ax = axes[i]
            color = agent_colors[i]
            
            if data['grayscale'] and 'error' not in data['grayscale'] and 'images' in data['grayscale']:
                # Get the specified view
                if view_type in data['grayscale']['images']:
                    img_data = data['grayscale']['images'][view_type]
                    image = img_data['image']
                    
                    # Display the image
                    im = ax.imshow(image, cmap='gray', aspect='auto', origin='upper')
                    
                    # Title and styling
                    ax.set_title(f'{agent_names[i]} - {img_data["name"]}', 
                               fontsize=14, color=color, fontweight='bold', pad=20)
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('Intensity', rotation=270, labelpad=15)
                    
                    # Agent info box
                    info_text = (f'🚗 {agent_names[i]}\\n'
                               f'Position: ({data["ego_x"]:.3f}, {data["ego_y"]:.3f})\\n'
                               f'Velocity: ({data["ego_vx"]:.3f}, {data["ego_vy"]:.3f})\\n'
                               f'Speed: {data["speed"]:.3f}\\n'
                               f'Lane: {data["lane_position"]}\\n'
                               f'Action: {data["action"]}\\n'
                               f'Reward: {data["reward"]:.4f}')
                    
                    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                           fontsize=10, verticalalignment='top', 
                           bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.9),
                           color='white', fontweight='bold')
                    
                    # Image stats
                    stats = img_data['stats']
                    stats_text = (f'Range: [{stats["min"]:.3f}, {stats["max"]:.3f}]\\n'
                                f'Mean: {stats["mean"]:.3f}\\n'
                                f'Non-zero: {stats["non_zero"]:.0f}')
                    
                    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                           fontsize=9, horizontalalignment='right',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                    
                    # Axis labels
                    ax.set_xlabel('Width (pixels)', fontsize=12)
                    ax.set_ylabel('Height (pixels)', fontsize=12)
                    
                else:
                    ax.text(0.5, 0.5, f'No {view_type} view\\navailable for {agent_names[i]}', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14)
                    ax.set_title(f'{agent_names[i]} - No Data', fontsize=14, color='gray')
            
            else:
                error_msg = 'No grayscale data'
                if data['grayscale'] and 'error' in data['grayscale']:
                    error_msg = f'Error: {data["grayscale"]["error"]}'
                
                ax.text(0.5, 0.5, f'{error_msg}\\nfor {agent_names[i]}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
                ax.set_title(f'{agent_names[i]} - Error', fontsize=14, color='red')
        
        plt.tight_layout()
        return fig
    
    def create_comparison_grid(self, episode_id: str, steps: List[int], 
                             view_type: str = 'best') -> plt.Figure:
        """Create a grid showing both agents across multiple time steps."""
        n_steps = len(steps)
        n_agents = 2
        
        fig, axes = plt.subplots(n_agents, n_steps, figsize=(4*n_steps, 8))
        if n_steps == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'Agent Comparison Over Time - {view_type.title()}\\nEpisode: {episode_id}', 
                    fontsize=16, fontweight='bold')
        
        agent_colors = ['#FF4444', '#4444FF']
        agent_names = ['Agent 0', 'Agent 1']
        
        for step_idx, step in enumerate(steps):
            agents_data = self.extract_step_data(episode_id, step)
            
            for agent_idx in range(n_agents):
                ax = axes[agent_idx, step_idx]
                
                if agent_idx in agents_data:
                    data = agents_data[agent_idx]
                    color = agent_colors[agent_idx]
                    
                    if (data['grayscale'] and 'error' not in data['grayscale'] and 
                        'images' in data['grayscale'] and view_type in data['grayscale']['images']):
                        
                        img_data = data['grayscale']['images'][view_type]
                        image = img_data['image']
                        
                        # Display image
                        im = ax.imshow(image, cmap='gray', aspect='auto', origin='upper')
                        
                        # Labels
                        if step_idx == 0:  # First column
                            ax.set_ylabel(agent_names[agent_idx], fontsize=12, 
                                        color=color, fontweight='bold')
                        if agent_idx == 0:  # First row
                            ax.set_title(f'Step {step}', fontsize=12)
                        
                        # Add mini info
                        ax.text(0.02, 0.98, f'({data["ego_x"]:.2f}, {data["ego_y"]:.2f})', 
                               transform=ax.transAxes, fontsize=8, verticalalignment='top',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
                               color='white')
                    else:
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                               transform=ax.transAxes, fontsize=10, color='red')
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        return fig
    
    def analyze_episode(self, episode_id: str, steps: List[int] = None, 
                       max_steps: int = 10, view_types: List[str] = None):
        """Analyze an episode with comprehensive visualizations."""
        if self.df is None:
            self.load_data()
        
        # Get episode data
        episode_data = self.df[self.df['episode_id'] == episode_id]
        if len(episode_data) == 0:
            print(f"❌ Episode {episode_id} not found")
            return
        
        # Determine steps to analyze
        if steps is None:
            available_steps = sorted(episode_data['step'].unique())
            steps = available_steps[:max_steps]
        
        print(f"\\n🎬 Analyzing episode: {episode_id}")
        print(f"📊 Steps to process: {steps}")
        
        # Default view types
        if view_types is None:
            view_types = ['best', 'rgb_grayscale']
        
        created_files = []
        
        # Create individual step comparisons
        for step in steps:
            print(f"\\n🔍 Processing step {step}...")
            
            agents_data = self.extract_step_data(episode_id, step)
            
            if len(agents_data) == 2:  # Both agents present
                for view_type in view_types:
                    try:
                        fig = self.create_dual_view_comparison(agents_data, episode_id, step, view_type)
                        
                        filename = f"dual_agent_{view_type}_ep_{episode_id}_step_{step:03d}.png"
                        filepath = self.output_dir / filename
                        
                        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                        plt.close(fig)
                        
                        created_files.append(filepath)
                        print(f"  💾 Saved: {filename}")
                        
                    except Exception as e:
                        print(f"  ❌ Error creating {view_type} view for step {step}: {e}")
            else:
                print(f"  ⚠️  Only {len(agents_data)} agents found for step {step}")
        
        # Create comparison grid if multiple steps
        if len(steps) > 1:
            try:
                fig = self.create_comparison_grid(episode_id, steps, 'best')
                filename = f"comparison_grid_{episode_id}_steps_{'_'.join(map(str, steps))}.png"
                filepath = self.output_dir / filename
                
                plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                
                created_files.append(filepath)
                print(f"\\n💾 Saved comparison grid: {filename}")
                
            except Exception as e:
                print(f"❌ Error creating comparison grid: {e}")
        
        print(f"\\n✅ Analysis complete! Created {len(created_files)} files in {self.output_dir}")
        return created_files


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Extract grayscale images for 2 agents from highway dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage:')[1] if 'Usage:' in __doc__ else ""
    )
    
    parser.add_argument(
        "--data", 
        default="/home/chettra/ITC/Research/AVs/data/highway_multimodal_dataset/dense_commuting/20250921_152749-1ea1c024_transitions.parquet",
        help="Path to parquet data file"
    )
    parser.add_argument("--episode", help="Specific episode ID to analyze")
    parser.add_argument("--steps", nargs="+", type=int, help="Specific steps to extract")
    parser.add_argument("--max-steps", type=int, default=5, help="Maximum steps to process")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--view-types", nargs="+", default=['best'], 
                       choices=['best', 'rgb_grayscale', 'channel_0', 'channel_1', 'channel_2'],
                       help="Types of views to create")
    parser.add_argument("--comparison-grid", action="store_true", help="Create comparison grid only")
    parser.add_argument("--all-steps", action="store_true", help="Process all available steps")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = DualAgentGrayscaleExtractor(args.data, args.output)
    extractor.load_data()
    
    # Determine episode
    if args.episode:
        episode_id = args.episode
    else:
        episode_id = extractor.df['episode_id'].iloc[0]
        print(f"\\n🎯 Using first episode: {episode_id}")
    
    # Determine steps
    steps = None
    max_steps = args.max_steps
    
    if args.steps:
        steps = args.steps
    elif args.all_steps:
        episode_data = extractor.df[extractor.df['episode_id'] == episode_id]
        steps = sorted(episode_data['step'].unique())
        max_steps = len(steps)
    
    # Run analysis
    if args.comparison_grid and steps:
        # Only create comparison grid
        print("\\n🖼️  Creating comparison grid only...")
        fig = extractor.create_comparison_grid(episode_id, steps[:10], args.view_types[0])  # Limit to 10 for readability
        
        filename = f"comparison_grid_{episode_id}.png"
        filepath = extractor.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved: {filepath}")
        
    else:
        # Full analysis
        extractor.analyze_episode(episode_id, steps, max_steps, args.view_types)


if __name__ == "__main__":
    main()