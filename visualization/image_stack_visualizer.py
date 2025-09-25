#!/usr/bin/env python3
"""
Image Stack Visualizer for Highway Multimodal Dataset
=====================================================

This script provides comprehensive visualization capabilities for the grayscale image stacks
and occupancy grids stored in the highway multimodal dataset Parquet files.

Features:
- Decode and visualize grayscale image stacks
- Plot occupancy grids with multiple channels
- Batch export frames to PNG files
- Interactive browsing of episodes and steps
- Statistical analysis of image data
- Support for different data formats and dtypes

Author: Highway Data Collection Team
Date: September 24, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ImageStackVisualizer:
    """
    A comprehensive class for visualizing image stacks from highway simulation data.
    """
    
    def __init__(self, data_root: str = "/home/chettra/ITC/Research/AVs/data"):
        """
        Initialize the visualizer.
        
        Args:
            data_root: Root directory containing the dataset folders
        """
        self.data_root = Path(data_root)
        self.current_df = None
        self.current_file = None
        
    def decode_blob(self, row: pd.Series, blob_col: str, shape_col: str, dtype_col: str) -> Optional[np.ndarray]:
        """
        Helper to decode a binary blob to numpy array using per-row shape/dtype.
        
        Args:
            row: Pandas Series containing the data row
            blob_col: Column name containing the binary blob
            shape_col: Column name containing the shape information
            dtype_col: Column name containing the dtype information
            
        Returns:
            Decoded numpy array or None if decoding fails
        """
        try:
            buf = row[blob_col]
            if pd.isna(buf) or buf is None:
                return None
            
            # Handle shape parsing
            shape = row[shape_col]
            if isinstance(shape, str):
                # Parse string representation like "[4, 128, 64]"
                shape = tuple(int(s) for s in shape.strip("[]()").split(",") if s.strip())
            elif isinstance(shape, np.ndarray):
                # Handle numpy array shape
                shape = tuple(shape.astype(int))
            elif isinstance(shape, (list, tuple)):
                # Handle list/tuple shape
                shape = tuple(int(s) for s in shape)
            else:
                shape = tuple(shape)
            
            # Debug information
            if len(shape) == 0:
                print(f"Warning: Empty shape parsed from {row[shape_col]}")
                return None
            
            dtype = np.dtype(row[dtype_col])
            arr = np.frombuffer(buf, dtype=dtype)
            
            # Verify the buffer size matches expected shape
            expected_size = np.prod(shape)
            if len(arr) != expected_size:
                # Try to reshape with available data (common with padding)
                if len(arr) > expected_size:
                    # Silently truncate buffer (this is common due to padding)
                    arr = arr[:expected_size]
                else:
                    print(f"Warning: Buffer too small ({len(arr)} < {expected_size}), cannot reshape")
                    return None
                
            return arr.reshape(shape)
            
        except Exception as e:
            print(f"Error decoding blob: {e}")
            return None
    
    def load_parquet_file(self, file_path: str) -> bool:
        """
        Load a Parquet file and store it for visualization.
        
        Args:
            file_path: Path to the Parquet file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.current_file = file_path
            self.current_df = pd.read_parquet(file_path)
            print(f"✓ Loaded {file_path}")
            print(f"  Dataset shape: {self.current_df.shape}")
            print(f"  Columns: {list(self.current_df.columns)}")
            return True
        except Exception as e:
            print(f"✗ Error loading {file_path}: {e}")
            return False
    
    def analyze_dataset_info(self) -> Dict[str, Any]:
        """
        Analyze the current dataset and return information about it.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if self.current_df is None:
            return {}
        
        df = self.current_df
        info = {
            'total_rows': len(df),
            'columns': list(df.columns),
            'episodes': df['episode_id'].nunique() if 'episode_id' in df.columns else 'N/A',
            'steps_range': (df['step'].min(), df['step'].max()) if 'step' in df.columns else 'N/A',
        }
        
        # Check for image data
        if 'grayscale_blob' in df.columns:
            non_null_images = df['grayscale_blob'].notna().sum()
            info['images_available'] = non_null_images
            info['image_coverage'] = f"{non_null_images/len(df)*100:.1f}%"
            
            # Sample image info
            if non_null_images > 0:
                sample_row = df[df['grayscale_blob'].notna()].iloc[0]
                sample_stack = self.decode_blob(sample_row, "grayscale_blob", "grayscale_shape", "grayscale_dtype")
                if sample_stack is not None:
                    info['image_shape'] = sample_stack.shape
                    info['image_dtype'] = str(sample_stack.dtype)
                else:
                    info['image_shape'] = "Decoding failed"
                    info['image_dtype'] = "Unknown"
        
        # Check for occupancy data
        if 'occupancy_blob' in df.columns:
            non_null_occ = df['occupancy_blob'].notna().sum()
            info['occupancy_available'] = non_null_occ
            info['occupancy_coverage'] = f"{non_null_occ/len(df)*100:.1f}%"
        
        return info
    
    def print_dataset_summary(self):
        """Print a summary of the current dataset."""
        info = self.analyze_dataset_info()
        if not info:
            print("No dataset loaded!")
            return
        
        print(f"\n{'='*60}")
        print(f"DATASET SUMMARY: {os.path.basename(self.current_file)}")
        print(f"{'='*60}")
        print(f"Total rows: {info['total_rows']:,}")
        print(f"Episodes: {info['episodes']}")
        print(f"Steps range: {info['steps_range']}")
        
        if 'images_available' in info:
            print(f"Images available: {info['images_available']:,} ({info['image_coverage']})")
            if 'image_shape' in info:
                print(f"Image shape: {info['image_shape']}")
            if 'image_dtype' in info:
                print(f"Image dtype: {info['image_dtype']}")
        
        if 'occupancy_available' in info:
            print(f"Occupancy grids: {info['occupancy_available']:,} ({info['occupancy_coverage']})")
        
        print(f"{'='*60}\n")
    
    def plot_sample_images(self, num_samples: int = 3, save_path: Optional[str] = None):
        """
        Plot sample grayscale image stacks from the current dataset.
        
        Args:
            num_samples: Number of samples to plot
            save_path: Optional path to save the plot
        """
        if self.current_df is None:
            print("No dataset loaded!")
            return
        
        df = self.current_df
        required_cols = ['grayscale_blob', 'grayscale_shape', 'grayscale_dtype']
        
        if not all(col in df.columns for col in required_cols):
            print(f"Missing required columns for images. Available: {df.columns.tolist()}")
            return
        
        # Get samples with valid image data
        valid_rows = df[df['grayscale_blob'].notna()]
        if len(valid_rows) == 0:
            print("No valid image data found!")
            return
        
        sample_rows = valid_rows.head(num_samples)
        
        # Create figure
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Grayscale Image Stacks - {os.path.basename(self.current_file)}', 
                     fontsize=16, fontweight='bold')
        
        for sample_idx, (_, row) in enumerate(sample_rows.iterrows()):
            # Decode grayscale stack
            stack = self.decode_blob(row, "grayscale_blob", "grayscale_shape", "grayscale_dtype")
            
            if stack is None:
                print(f"  Sample {sample_idx+1}: No grayscale data")
                continue
            
            print(f"  Sample {sample_idx+1}: Shape {stack.shape}, dtype {stack.dtype}")
            
            # Plot frames (assume first dimension is time/channels)
            T = min(stack.shape[0], 4)  # Show up to 4 frames
            
            for t in range(T):
                ax = axes[sample_idx, t]
                
                # Handle different value ranges
                if stack.dtype == np.uint8:
                    vmin, vmax = 0, 255
                else:
                    vmin, vmax = stack[t].min(), stack[t].max()
                
                im = ax.imshow(stack[t], cmap="gray", vmin=vmin, vmax=vmax)
                ax.set_title(f'Ep{row.get("episode_id", "?")} Step{row.get("step", "?")} F{t}')
                ax.axis('off')
                
                # Add colorbar for the first frame of each sample
                if t == 0:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Hide unused subplots
            for t in range(T, 4):
                axes[sample_idx, t].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        
        plt.show()
    
    def plot_occupancy_grids(self, num_samples: int = 2, save_path: Optional[str] = None):
        """
        Plot occupancy grids from the current dataset.
        
        Args:
            num_samples: Number of samples to plot
            save_path: Optional path to save the plot
        """
        if self.current_df is None:
            print("No dataset loaded!")
            return
        
        df = self.current_df
        occupancy_cols = ['occupancy_blob', 'occupancy_shape', 'occupancy_dtype']
        
        if not all(col in df.columns for col in occupancy_cols):
            print("No occupancy grid data found")
            return
        
        # Get samples with valid occupancy data
        valid_rows = df[df['occupancy_blob'].notna()]
        if len(valid_rows) == 0:
            print("No valid occupancy data found!")
            return
        
        sample_rows = valid_rows.head(num_samples)
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Occupancy Grids - {os.path.basename(self.current_file)}', 
                     fontsize=16, fontweight='bold')
        
        for sample_idx, (_, row) in enumerate(sample_rows.iterrows()):
            occ = self.decode_blob(row, "occupancy_blob", "occupancy_shape", "occupancy_dtype")
            
            if occ is None:
                continue
            
            print(f"  Sample {sample_idx+1}: Occupancy shape {occ.shape}, dtype {occ.dtype}")
            
            # Plot channels if 3D, otherwise plot single grid
            if len(occ.shape) == 3:
                C = min(occ.shape[0], 3)  # Show up to 3 channels
                for c in range(C):
                    ax = axes[sample_idx, c]
                    im = ax.imshow(occ[c], cmap="viridis")
                    ax.set_title(f'Ep{row.get("episode_id", "?")} Ch{c}')
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Hide unused subplots
                for c in range(C, 3):
                    axes[sample_idx, c].axis('off')
            else:
                # Single occupancy grid
                ax = axes[sample_idx, 0]
                im = ax.imshow(occ, cmap="viridis")
                ax.set_title(f'Ep{row.get("episode_id", "?")} Occupancy')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                # Hide unused subplots
                axes[sample_idx, 1].axis('off')
                axes[sample_idx, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        
        plt.show()
    
    def export_frames_to_png(self, output_dir: str, max_episodes: int = 5, max_steps_per_episode: int = 10):
        """
        Export individual frames to PNG files.
        
        Args:
            output_dir: Directory to save PNG files
            max_episodes: Maximum number of episodes to export
            max_steps_per_episode: Maximum steps per episode to export
        """
        if self.current_df is None:
            print("No dataset loaded!")
            return
        
        df = self.current_df
        if 'grayscale_blob' not in df.columns:
            print("No image data available for export!")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get valid rows with image data
        valid_rows = df[df['grayscale_blob'].notna()]
        
        exported_count = 0
        episode_count = 0
        current_episode = None
        steps_in_episode = 0
        
        print(f"Exporting frames to: {output_path}")
        
        for _, row in valid_rows.iterrows():
            episode_id = row.get('episode_id', 'unknown')
            step = row.get('step', 0)
            
            # Track episodes and steps
            if episode_id != current_episode:
                if episode_count >= max_episodes:
                    break
                current_episode = episode_id
                episode_count += 1
                steps_in_episode = 0
            
            if steps_in_episode >= max_steps_per_episode:
                continue
            
            steps_in_episode += 1
            
            # Decode and save frames
            stack = self.decode_blob(row, "grayscale_blob", "grayscale_shape", "grayscale_dtype")
            if stack is None:
                continue
            
            # Save each frame in the stack
            for frame_idx in range(stack.shape[0]):
                filename = f"ep_{episode_id}_step_{step:04d}_frame_{frame_idx}.png"
                filepath = output_path / filename
                
                # Normalize for saving
                frame = stack[frame_idx]
                if frame.dtype == np.uint8:
                    normalized_frame = frame
                else:
                    # Normalize to 0-255 range
                    normalized_frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
                
                plt.imsave(filepath, normalized_frame, cmap='gray')
                exported_count += 1
        
        print(f"✓ Exported {exported_count} frames from {episode_count} episodes")
    
    def create_animation_gif(self, episode_id: str, output_path: str, max_steps: int = 50):
        """
        Create an animated GIF from an episode's image sequence.
        
        Args:
            episode_id: ID of the episode to animate
            output_path: Path to save the GIF file
            max_steps: Maximum number of steps to include
        """
        try:
            from PIL import Image
        except ImportError:
            print("PIL/Pillow is required for GIF creation. Install with: pip install Pillow")
            return
        
        if self.current_df is None:
            print("No dataset loaded!")
            return
        
        df = self.current_df
        episode_data = df[df['episode_id'] == episode_id].head(max_steps)
        
        if len(episode_data) == 0:
            print(f"No data found for episode {episode_id}")
            return
        
        frames = []
        for _, row in episode_data.iterrows():
            stack = self.decode_blob(row, "grayscale_blob", "grayscale_shape", "grayscale_dtype")
            if stack is not None and len(stack.shape) >= 2:
                # Use the first frame of the stack
                frame = stack[0] if len(stack.shape) == 3 else stack
                
                # Normalize to 0-255
                if frame.dtype != np.uint8:
                    frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
                
                frames.append(Image.fromarray(frame))
        
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=200,  # 200ms per frame
                loop=0
            )
            print(f"✓ Created animation: {output_path} ({len(frames)} frames)")
        else:
            print("No valid frames found for animation")
    
    def compare_scenarios(self, scenario_paths: List[str], num_samples: int = 2):
        """
        Compare image data across different scenarios.
        
        Args:
            scenario_paths: List of paths to different scenario Parquet files
            num_samples: Number of samples per scenario
        """
        fig, axes = plt.subplots(len(scenario_paths), 4, figsize=(16, 4*len(scenario_paths)))
        if len(scenario_paths) == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Scenario Comparison', fontsize=16, fontweight='bold')
        
        for scenario_idx, path in enumerate(scenario_paths):
            scenario_name = Path(path).parent.name
            
            try:
                df = pd.read_parquet(path)
                valid_rows = df[df['grayscale_blob'].notna()]
                
                if len(valid_rows) > 0:
                    sample_row = valid_rows.iloc[0]
                    stack = self.decode_blob(sample_row, "grayscale_blob", "grayscale_shape", "grayscale_dtype")
                    
                    if stack is not None:
                        # Plot up to 4 frames
                        T = min(stack.shape[0], 4)
                        for t in range(T):
                            ax = axes[scenario_idx, t]
                            ax.imshow(stack[t], cmap="gray")
                            ax.set_title(f'{scenario_name} - F{t}')
                            ax.axis('off')
                        
                        # Hide unused subplots
                        for t in range(T, 4):
                            axes[scenario_idx, t].axis('off')
                            
            except Exception as e:
                print(f"Error processing {path}: {e}")
                for t in range(4):
                    axes[scenario_idx, t].text(0.5, 0.5, f'Error loading\n{scenario_name}', 
                                             ha='center', va='center', transform=axes[scenario_idx, t].transAxes)
                    axes[scenario_idx, t].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def find_parquet_files(self) -> List[str]:
        """
        Find all Parquet files in the data directory.
        
        Returns:
            List of Parquet file paths
        """
        parquet_files = []
        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, file))
        return sorted(parquet_files)
    
    def debug_data_format(self, num_rows: int = 3):
        """
        Debug method to inspect the raw data format.
        
        Args:
            num_rows: Number of rows to inspect
        """
        if self.current_df is None:
            print("No dataset loaded!")
            return
        
        df = self.current_df
        print(f"\n{'='*60}")
        print(f"DEBUG: Data Format Inspection")
        print(f"{'='*60}")
        
        # Check first few rows for image data
        for i, (_, row) in enumerate(df.head(num_rows).iterrows()):
            print(f"\nRow {i+1}:")
            if 'grayscale_shape' in row:
                print(f"  grayscale_shape: {row['grayscale_shape']} (type: {type(row['grayscale_shape'])})")
            if 'grayscale_dtype' in row:
                print(f"  grayscale_dtype: {row['grayscale_dtype']} (type: {type(row['grayscale_dtype'])})")
            if 'grayscale_blob' in row:
                blob_size = len(row['grayscale_blob']) if pd.notna(row['grayscale_blob']) else 0
                print(f"  grayscale_blob size: {blob_size} bytes")
            
            # Try to decode
            if all(col in row for col in ['grayscale_blob', 'grayscale_shape', 'grayscale_dtype']):
                stack = self.decode_blob(row, "grayscale_blob", "grayscale_shape", "grayscale_dtype")
                if stack is not None:
                    print(f"  ✓ Decoded successfully: shape={stack.shape}, dtype={stack.dtype}")
                else:
                    print(f"  ✗ Decoding failed")
        
        print(f"{'='*60}\n")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Visualize highway simulation image data')
    parser.add_argument('--file', '-f', type=str, help='Specific Parquet file to visualize')
    parser.add_argument('--data-root', '-d', type=str, 
                       default='/home/chettra/ITC/Research/AVs/data',
                       help='Root directory containing data files')
    parser.add_argument('--samples', '-s', type=int, default=3,
                       help='Number of samples to visualize')
    parser.add_argument('--export', '-e', type=str,
                       help='Directory to export individual frames as PNG')
    parser.add_argument('--gif', '-g', type=str,
                       help='Create animated GIF for specific episode ID')
    parser.add_argument('--gif-output', type=str, default='episode_animation.gif',
                       help='Output path for GIF file')
    parser.add_argument('--list-files', '-l', action='store_true',
                       help='List all available Parquet files')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    viz = ImageStackVisualizer(args.data_root)
    
    # List files if requested
    if args.list_files:
        files = viz.find_parquet_files()
        print(f"Found {len(files)} Parquet files:")
        for i, file in enumerate(files, 1):
            print(f"  {i:2d}. {file}")
        return
    
    # Determine which file to process
    if args.file:
        parquet_file = args.file
    else:
        # Try to find a sample file
        files = viz.find_parquet_files()
        if not files:
            print("No Parquet files found! Use --list-files to check available files.")
            return
        parquet_file = files[0]  # Use first found file
        print(f"No file specified, using: {parquet_file}")
    
    # Load and visualize
    if viz.load_parquet_file(parquet_file):
        viz.print_dataset_summary()
        
        print("Plotting sample images...")
        viz.plot_sample_images(num_samples=args.samples)
        
        print("Plotting occupancy grids...")
        viz.plot_occupancy_grids(num_samples=2)
        
        # Export frames if requested
        if args.export:
            viz.export_frames_to_png(args.export)
        
        # Create GIF if requested
        if args.gif:
            viz.create_animation_gif(args.gif, args.gif_output)


if __name__ == "__main__":
    main()