#!/usr/bin/env python3
"""
Enhanced grayscale image extractor for 2 agents with better format handling.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from highway_datacollection.storage.encoders import BinaryArrayEncoder
import matplotlib.patches as mpatches

class DualAgentGrayscaleExtractor:
    """Extract and visualize grayscale images for 2 agents."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.decoder = BinaryArrayEncoder()
        self.output_dir = self.data_path.parent / "extracted_grayscale"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🖼️  Dual-Agent Grayscale Extractor")
        print(f"📁 Data: {self.data_path.name}")
        print(f"💾 Output: {self.output_dir}")
    
    def load_data(self):
        """Load the dataset."""
        print("📊 Loading dataset...")
        self.df = pd.read_parquet(self.data_path)
        print(f"✅ Loaded {len(self.df)} rows")
        return self.df
    
    def decode_and_process_image(self, blob: bytes, shape: np.ndarray, dtype: str) -> dict:
        """Decode blob and return multiple image representations."""
        try:
            # Decode using proper decoder
            array = self.decoder.decode(blob, tuple(shape), dtype)
            
            result = {
                'raw_array': array,
                'raw_stats': {
                    'shape': array.shape,
                    'min': float(array.min()),
                    'max': float(array.max()),
                    'mean': float(array.mean()),
                    'std': float(array.std())
                }
            }
            
            # Try different interpretations of the data
            if len(array.shape) == 3:
                if array.shape[0] <= 4:  # Likely channels first: [C, H, W]
                    channels_first = array
                    channels_last = np.transpose(array, (1, 2, 0))
                else:  # Likely channels last: [H, W, C]
                    channels_last = array
                    channels_first = np.transpose(array, (2, 0, 1))
                
                # Process channels-first format
                result['channels_first'] = self._process_channels_first(channels_first)
                
                # Process channels-last format  
                result['channels_last'] = self._process_channels_last(channels_last)
                
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def _process_channels_first(self, array: np.ndarray) -> dict:
        """Process array in channels-first format [C, H, W]."""
        result = {}
        
        if array.shape[0] >= 3:  # RGB or RGBA
            # RGB to grayscale
            rgb_gray = (0.299 * array[0] + 0.587 * array[1] + 0.114 * array[2])
            result['rgb_grayscale'] = self._normalize_image(rgb_gray)
        
        # Individual channels
        for i in range(min(array.shape[0], 4)):
            channel_names = ['R', 'G', 'B', 'A']
            result[f'channel_{channel_names[i]}'] = self._normalize_image(array[i])
        
        # First channel as grayscale
        result['first_channel'] = self._normalize_image(array[0])
        
        return result
    
    def _process_channels_last(self, array: np.ndarray) -> dict:
        """Process array in channels-last format [H, W, C]."""
        result = {}
        
        if array.shape[-1] >= 3:  # RGB or RGBA
            # RGB to grayscale
            rgb_gray = (0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2])
            result['rgb_grayscale'] = self._normalize_image(rgb_gray)
        
        # Individual channels
        for i in range(min(array.shape[-1], 4)):
            channel_names = ['R', 'G', 'B', 'A']
            result[f'channel_{channel_names[i]}'] = self._normalize_image(array[:, :, i])
        
        return result
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image for display."""
        image = image.astype(np.float32)
        
        # Handle different value ranges
        if image.max() > 1.0:
            image = image / 255.0
        
        # Ensure [0, 1] range
        image = np.clip(image, 0, 1)
        
        return image
    
    def extract_step_images(self, episode_id: str, step: int) -> dict:
        """Extract and process images for both agents at a given step."""
        step_data = self.df[(self.df['episode_id'] == episode_id) & (self.df['step'] == step)]
        
        results = {}
        for _, row in step_data.iterrows():
            agent_id = row['agent_id']
            
            if pd.notna(row['grayscale_blob']):
                image_data = self.decode_and_process_image(
                    row['grayscale_blob'],
                    row['grayscale_shape'], 
                    row['grayscale_dtype']
                )
                
                results[agent_id] = {
                    'images': image_data,
                    'ego_x': row['ego_x'],
                    'ego_y': row['ego_y'],
                    'lane_position': row['lane_position'],
                    'action': row['action'],
                    'reward': row['reward']
                }
        
        return results
    
    def create_multi_view_visualization(self, agent_data: dict, episode_id: str, step: int):
        """Create comprehensive visualization showing multiple interpretations."""
        
        n_agents = len(agent_data)
        if n_agents == 0:
            print("❌ No agent data to visualize")
            return None
        
        # Determine what images we have
        sample_agent = list(agent_data.values())[0]
        if 'error' in sample_agent['images']:
            print(f"❌ Error in image data: {sample_agent['images']['error']}")
            return None
        
        available_views = []
        for view_type in ['rgb_grayscale', 'first_channel', 'channel_R', 'channel_G', 'channel_B']:
            if any(view_type in agent['images'].get('channels_first', {}) for agent in agent_data.values()):
                available_views.append(view_type)
        
        if not available_views:
            available_views = ['first_channel']  # Fallback
        
        n_views = len(available_views)
        
        # Create subplots
        fig, axes = plt.subplots(n_agents, n_views, figsize=(5*n_views, 6*n_agents))
        if n_agents == 1:
            axes = axes.reshape(1, -1)
        if n_views == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'Dual-Agent Grayscale Analysis\nEpisode: {episode_id}, Step: {step}', 
                    fontsize=16, fontweight='bold')
        
        agent_colors = ['red', 'blue']
        
        for agent_idx, (agent_id, data) in enumerate(agent_data.items()):
            color = agent_colors[agent_idx % len(agent_colors)]
            
            for view_idx, view_type in enumerate(available_views):
                ax = axes[agent_idx, view_idx]
                
                # Get the image for this view
                image = None
                if view_type in data['images'].get('channels_first', {}):
                    image = data['images']['channels_first'][view_type]
                elif view_type in data['images'].get('channels_last', {}):
                    image = data['images']['channels_last'][view_type]
                
                if image is not None:
                    # Display image
                    im = ax.imshow(image, cmap='gray', aspect='auto', origin='upper')
                    
                    # Title
                    if agent_idx == 0:  # Top row
                        ax.set_title(f'{view_type.replace("_", " ").title()}', fontsize=12)
                    
                    # Agent label on first column
                    if view_idx == 0:
                        ax.set_ylabel(f'Agent {agent_id}', fontsize=12, color=color, fontweight='bold')
                    
                    # Add statistics
                    stats = f'Range: [{image.min():.3f}, {image.max():.3f}]\\nMean: {image.mean():.3f}'
                    ax.text(0.02, 0.98, stats, transform=ax.transAxes, 
                           fontsize=8, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                    
                    # Add ego info
                    ego_info = f'Ego: ({data["ego_x"]:.2f}, {data["ego_y"]:.2f})\\nLane: {data["lane_position"]}'
                    ax.text(0.98, 0.02, ego_info, transform=ax.transAxes,
                           fontsize=8, horizontalalignment='right',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
                           color='white', fontweight='bold')
                    
                    # Colorbar for first view only
                    if view_idx == 0:
                        plt.colorbar(im, ax=ax, shrink=0.8)
                
                else:
                    ax.text(0.5, 0.5, f'No {view_type}\\ndata available', 
                           ha='center', va='center', transform=ax.transAxes)
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        
        # Save
        filename = f"dual_agent_multiview_ep_{episode_id}_step_{step:03d}.png"
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
        
        return fig
    
    def analyze_episode(self, episode_id: str = None, max_steps: int = 5):
        """Analyze an episode with multiple step visualizations."""
        if self.df is None:
            self.load_data()
        
        if episode_id is None:
            episode_id = self.df['episode_id'].iloc[0]
        
        print(f"\\n🎬 Analyzing episode: {episode_id}")
        
        # Get available steps
        episode_data = self.df[self.df['episode_id'] == episode_id]
        available_steps = sorted(episode_data['step'].unique())[:max_steps]
        
        print(f"📊 Processing {len(available_steps)} steps: {available_steps}")
        
        for step in available_steps:
            print(f"\\n🔍 Step {step}...")
            
            # Extract agent data
            agent_data = self.extract_step_images(episode_id, step)
            
            if agent_data:
                print(f"   Found data for agents: {list(agent_data.keys())}")
                
                # Create visualization
                fig = self.create_multi_view_visualization(agent_data, episode_id, step)
                
                if fig:
                    plt.close(fig)  # Close to save memory
            else:
                print(f"   ❌ No data found for step {step}")
        
        print(f"\\n✅ Analysis complete! Check output directory: {self.output_dir}")

def main():
    """Main function."""
    data_path = "/home/chettra/ITC/Research/AVs/data/highway_multimodal_dataset/dense_commuting/20250921_152749-1ea1c024_transitions.parquet"
    
    # Initialize extractor
    extractor = DualAgentGrayscaleExtractor(data_path)
    extractor.load_data()
    
    # Analyze the first episode
    extractor.analyze_episode(max_steps=3)
    
    print("\\n🎯 Quick single-step test...")
    
    # Quick test for first step
    episode_id = extractor.df['episode_id'].iloc[0]
    agent_data = extractor.extract_step_images(episode_id, 0)
    
    if agent_data:
        print("\\n📊 Raw data analysis:")
        for agent_id, data in agent_data.items():
            print(f"\\nAgent {agent_id}:")
            if 'raw_stats' in data['images']:
                stats = data['images']['raw_stats']
                print(f"  Raw shape: {stats['shape']}")
                print(f"  Value range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                print(f"  Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")

if __name__ == "__main__":
    main()