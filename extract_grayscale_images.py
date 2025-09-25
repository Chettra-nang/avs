#!/usr/bin/env python3
"""
Extract and visualize grayscale images for 2 agents from highway multimodal dataset.

This script loads the parquet data, decodes the binary grayscale blobs,
and creates visualizations showing both agents' perspectives side by side.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class GrayscaleImageExtractor:
    """Extract and visualize grayscale images from highway dataset."""
    
    def __init__(self, data_path: str):
        """
        Initialize the extractor.
        
        Args:
            data_path: Path to the parquet file containing the dataset
        """
        self.data_path = Path(data_path)
        self.df = None
        self.output_dir = self.data_path.parent / "extracted_images"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🖼️  Grayscale Image Extractor")
        print(f"📁 Data path: {self.data_path}")
        print(f"💾 Output directory: {self.output_dir}")
    
    def load_data(self) -> None:
        """Load the parquet dataset."""
        print(f"\n📊 Loading dataset...")
        self.df = pd.read_parquet(self.data_path)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Shape: {self.df.shape}")
        print(f"   Episodes: {self.df['episode_id'].nunique()}")
        print(f"   Agents: {sorted(self.df['agent_id'].unique())}")
        print(f"   Max steps: {self.df['step'].max()}")
    
    def decode_grayscale_blob(self, blob: bytes, shape: np.ndarray, dtype: str) -> np.ndarray:
        """
        Decode binary blob back to numpy array.
        
        Args:
            blob: Binary data
            shape: Array shape
            dtype: Data type string
            
        Returns:
            Decoded numpy array
        """
        # Convert blob back to numpy array
        array = np.frombuffer(blob, dtype=dtype)
        return array.reshape(shape)
    
    def process_grayscale_image(self, grayscale_array: np.ndarray) -> np.ndarray:
        """
        Process grayscale array for visualization.
        
        Args:
            grayscale_array: Raw grayscale array of shape [4, height, width] or [height, width, 4]
            
        Returns:
            Processed image array for display
        """
        # Handle different possible shapes
        if grayscale_array.shape[0] == 4:  # [4, height, width] - channels first
            # Take the first channel (grayscale) or convert RGB to grayscale
            if grayscale_array.shape[0] >= 3:  # RGB(A)
                # Convert RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
                image = (0.299 * grayscale_array[0] + 
                        0.587 * grayscale_array[1] + 
                        0.114 * grayscale_array[2])
            else:
                image = grayscale_array[0]  # Just take first channel
        elif grayscale_array.shape[-1] == 4:  # [height, width, 4] - channels last
            if grayscale_array.shape[-1] >= 3:  # RGB(A)
                image = (0.299 * grayscale_array[:, :, 0] + 
                        0.587 * grayscale_array[:, :, 1] + 
                        0.114 * grayscale_array[:, :, 2])
            else:
                image = grayscale_array[:, :, 0]
        else:
            # Already grayscale
            image = grayscale_array
        
        # Normalize to [0, 1] range for display
        if image.max() > 1.0:
            image = image / 255.0
        
        return image
    
    def extract_step_images(self, episode_id: str, step: int) -> Dict[int, np.ndarray]:
        """
        Extract grayscale images for both agents at a specific step.
        
        Args:
            episode_id: Episode identifier
            step: Time step
            
        Returns:
            Dictionary mapping agent_id to processed image array
        """
        step_data = self.df[(self.df['episode_id'] == episode_id) & (self.df['step'] == step)]
        
        agent_images = {}
        for _, row in step_data.iterrows():
            agent_id = row['agent_id']
            
            if pd.notna(row['grayscale_blob']):
                # Decode the binary blob
                grayscale_array = self.decode_grayscale_blob(
                    row['grayscale_blob'], 
                    row['grayscale_shape'], 
                    row['grayscale_dtype']
                )
                
                # Process for visualization
                image = self.process_grayscale_image(grayscale_array)
                agent_images[agent_id] = image
        
        return agent_images
    
    def create_dual_agent_visualization(self, agent_images: Dict[int, np.ndarray], 
                                      episode_id: str, step: int,
                                      ego_positions: Optional[Dict[int, Tuple[float, float]]] = None,
                                      save: bool = True) -> plt.Figure:
        """
        Create side-by-side visualization of both agents' grayscale images.
        
        Args:
            agent_images: Dictionary mapping agent_id to image array
            episode_id: Episode identifier
            step: Time step
            ego_positions: Optional ego positions for both agents
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Dual-Agent Grayscale Views\nEpisode: {episode_id}, Step: {step}', 
                    fontsize=14, fontweight='bold')
        
        agent_colors = ['red', 'blue']  # Colors for agent identification
        
        for i, (agent_id, ax) in enumerate(zip(sorted(agent_images.keys()), axes)):
            if agent_id in agent_images:
                image = agent_images[agent_id]
                
                # Display the image
                im = ax.imshow(image, cmap='gray', aspect='auto')
                ax.set_title(f'Agent {agent_id} (Ego View)', 
                           fontsize=12, color=agent_colors[i], fontweight='bold')
                
                # Add ego position annotation if available
                if ego_positions and agent_id in ego_positions:
                    ego_x, ego_y = ego_positions[agent_id]
                    ax.text(0.02, 0.98, f'Ego Position:\nX: {ego_x:.3f}\nY: {ego_y:.3f}', 
                           transform=ax.transAxes, fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=agent_colors[i], alpha=0.7))
                
                # Add colorbar
                plt.colorbar(im, ax=ax, shrink=0.8)
                
                # Add grid for better visualization
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Width (pixels)')
                ax.set_ylabel('Height (pixels)')
            else:
                ax.text(0.5, 0.5, f'No image data\nfor Agent {agent_id}', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'Agent {agent_id} - No Data', fontsize=12, color='gray')
        
        plt.tight_layout()
        
        if save:
            filename = f"dual_agent_grayscale_ep_{episode_id}_step_{step:03d}.png"
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved: {save_path}")
        
        return fig
    
    def extract_episode_sequence(self, episode_id: str, max_steps: int = None) -> None:
        """
        Extract grayscale images for an entire episode sequence.
        
        Args:
            episode_id: Episode identifier
            max_steps: Maximum number of steps to extract (None for all)
        """
        print(f"\n🎬 Extracting episode sequence: {episode_id}")
        
        episode_data = self.df[self.df['episode_id'] == episode_id]
        available_steps = sorted(episode_data['step'].unique())
        
        if max_steps:
            available_steps = available_steps[:max_steps]
        
        print(f"📊 Available steps: {len(available_steps)} (showing {min(len(available_steps), max_steps or len(available_steps))})")
        
        for step in available_steps:
            print(f"  Processing step {step}...")
            
            # Extract images
            agent_images = self.extract_step_images(episode_id, step)
            
            # Get ego positions for annotation
            step_data = episode_data[episode_data['step'] == step]
            ego_positions = {}
            for _, row in step_data.iterrows():
                ego_positions[row['agent_id']] = (row['ego_x'], row['ego_y'])
            
            # Create visualization
            fig = self.create_dual_agent_visualization(
                agent_images, episode_id, step, ego_positions, save=True
            )
            plt.close(fig)  # Close to save memory
    
    def create_comparison_grid(self, episode_id: str, steps: List[int]) -> plt.Figure:
        """
        Create a grid comparing both agents across multiple time steps.
        
        Args:
            episode_id: Episode identifier
            steps: List of steps to include
            
        Returns:
            Matplotlib figure
        """
        n_steps = len(steps)
        n_agents = 2
        
        fig, axes = plt.subplots(n_agents, n_steps, figsize=(4*n_steps, 8))
        if n_steps == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'Agent Comparison Across Time\nEpisode: {episode_id}', 
                    fontsize=16, fontweight='bold')
        
        agent_colors = ['red', 'blue']
        
        for step_idx, step in enumerate(steps):
            agent_images = self.extract_step_images(episode_id, step)
            
            # Get ego positions
            episode_data = self.df[self.df['episode_id'] == episode_id]
            step_data = episode_data[episode_data['step'] == step]
            ego_positions = {}
            for _, row in step_data.iterrows():
                ego_positions[row['agent_id']] = (row['ego_x'], row['ego_y'])
            
            for agent_idx, agent_id in enumerate([0, 1]):
                ax = axes[agent_idx, step_idx]
                
                if agent_id in agent_images:
                    image = agent_images[agent_id]
                    im = ax.imshow(image, cmap='gray', aspect='auto')
                    
                    if step_idx == 0:  # Label agents on first column
                        ax.set_ylabel(f'Agent {agent_id}', fontsize=12, 
                                    color=agent_colors[agent_idx], fontweight='bold')
                    
                    if agent_idx == 0:  # Label steps on first row
                        ax.set_title(f'Step {step}', fontsize=11)
                    
                    # Add ego position annotation
                    if agent_id in ego_positions:
                        ego_x, ego_y = ego_positions[agent_id]
                        ax.text(0.02, 0.98, f'X:{ego_x:.2f}\nY:{ego_y:.2f}', 
                               transform=ax.transAxes, fontsize=8, 
                               verticalalignment='top', color='white',
                               bbox=dict(boxstyle="round,pad=0.2", 
                                       facecolor=agent_colors[agent_idx], alpha=0.8))
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                           transform=ax.transAxes)
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        
        # Save the comparison grid
        filename = f"agent_comparison_grid_ep_{episode_id}.png"
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved comparison grid: {save_path}")
        
        return fig
    
    def get_episode_summary(self) -> pd.DataFrame:
        """Get summary of available episodes and their data."""
        if self.df is None:
            self.load_data()
        
        summary = []
        for episode_id in self.df['episode_id'].unique():
            episode_data = self.df[self.df['episode_id'] == episode_id]
            
            summary.append({
                'episode_id': episode_id,
                'total_steps': episode_data['step'].max() + 1,
                'agents': sorted(episode_data['agent_id'].unique()),
                'grayscale_coverage': f"{episode_data['grayscale_blob'].notna().sum()}/{len(episode_data)}"
            })
        
        return pd.DataFrame(summary)


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Extract grayscale images from highway multimodal dataset")
    parser.add_argument("data_path", help="Path to the parquet file")
    parser.add_argument("--episode", help="Specific episode ID to process")
    parser.add_argument("--steps", nargs="+", type=int, help="Specific steps to extract")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum steps to process")
    parser.add_argument("--comparison-grid", action="store_true", help="Create comparison grid")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = GrayscaleImageExtractor(args.data_path)
    extractor.load_data()
    
    # Show episode summary
    print("\n📋 Episode Summary:")
    summary = extractor.get_episode_summary()
    print(summary.to_string(index=False))
    
    if args.episode:
        episode_id = args.episode
    else:
        # Use the first available episode
        episode_id = extractor.df['episode_id'].iloc[0]
        print(f"\n🎯 Using first episode: {episode_id}")
    
    if args.steps:
        # Process specific steps
        print(f"\n🔍 Processing specific steps: {args.steps}")
        for step in args.steps:
            agent_images = extractor.extract_step_images(episode_id, step)
            if agent_images:
                episode_data = extractor.df[extractor.df['episode_id'] == episode_id]
                step_data = episode_data[episode_data['step'] == step]
                ego_positions = {}
                for _, row in step_data.iterrows():
                    ego_positions[row['agent_id']] = (row['ego_x'], row['ego_y'])
                
                fig = extractor.create_dual_agent_visualization(
                    agent_images, episode_id, step, ego_positions, save=True
                )
                plt.show()
    
    if args.comparison_grid:
        # Create comparison grid
        episode_data = extractor.df[extractor.df['episode_id'] == episode_id]
        available_steps = sorted(episode_data['step'].unique())[:5]  # First 5 steps
        
        fig = extractor.create_comparison_grid(episode_id, available_steps)
        plt.show()
    
    # Extract full sequence if not processing specific steps
    if not args.steps and not args.comparison_grid:
        extractor.extract_episode_sequence(episode_id, args.max_steps)
    
    print(f"\n✅ Processing complete! Check output directory: {extractor.output_dir}")


if __name__ == "__main__":
    main()