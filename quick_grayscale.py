#!/usr/bin/env python3
"""
Quick Grayscale Extractor - Simple script for extracting 2-agent grayscale images

This is a simplified version for quick grayscale image extraction from your dataset.

Usage:
    python quick_grayscale.py                    # Extract first 3 steps
    python quick_grayscale.py 5                  # Extract first 5 steps  
    python quick_grayscale.py 0 1 2 5 10         # Extract specific steps
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


def extract_and_display_grayscale(data_path: str, steps_to_extract=None):
    """
    Quick extraction and display of grayscale images for 2 agents.
    
    Args:
        data_path: Path to parquet file
        steps_to_extract: List of steps to extract, or int for first N steps
    """
    
    print("🚗🚗 Quick Grayscale Extractor for 2 Agents")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    df = pd.read_parquet(data_path)
    decoder = BinaryArrayEncoder()
    
    # Get first episode
    episode_id = df['episode_id'].iloc[0]
    print(f"🎬 Episode: {episode_id}")
    
    # Determine steps to extract
    if steps_to_extract is None:
        steps_to_extract = [0, 1, 2]  # Default: first 3 steps
    elif isinstance(steps_to_extract, int):
        episode_data = df[df['episode_id'] == episode_id]
        available_steps = sorted(episode_data['step'].unique())
        steps_to_extract = available_steps[:steps_to_extract]
    
    print(f"📈 Steps to extract: {steps_to_extract}")
    
    # Output directory
    output_dir = Path(data_path).parent / "quick_grayscale"
    output_dir.mkdir(exist_ok=True)
    print(f"💾 Output directory: {output_dir}")
    
    # Extract and save each step
    for step in steps_to_extract:
        print(f"\\n🔍 Processing step {step}...")
        
        # Get data for both agents at this step
        step_data = df[(df['episode_id'] == episode_id) & (df['step'] == step)]
        
        if len(step_data) != 2:
            print(f"  ⚠️  Expected 2 agents, found {len(step_data)}. Skipping.")
            continue
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Agent Grayscale Views - Step {step}\\nEpisode: {episode_id}', 
                    fontsize=14, fontweight='bold')
        
        agent_colors = ['red', 'blue']
        
        success_count = 0
        
        for i, (_, row) in enumerate(step_data.iterrows()):
            agent_id = row['agent_id']
            ax = axes[i]
            
            try:
                # Decode grayscale blob
                grayscale_array = decoder.decode(
                    row['grayscale_blob'],
                    tuple(row['grayscale_shape']),
                    row['grayscale_dtype']
                )
                
                # Convert to grayscale (assume shape [4, 128, 64] - RGBA)
                if grayscale_array.shape[0] >= 3:
                    # RGB to grayscale conversion
                    image = (0.299 * grayscale_array[0] + 
                            0.587 * grayscale_array[1] + 
                            0.114 * grayscale_array[2])
                else:
                    image = grayscale_array[0]
                
                # Normalize for display
                if image.max() > 1.0:
                    image = image / 255.0
                
                # Display
                im = ax.imshow(image, cmap='gray', aspect='auto', origin='upper')
                ax.set_title(f'Agent {agent_id}', fontsize=12, 
                           color=agent_colors[i], fontweight='bold')
                
                # Add info
                info_text = (f'Position: ({row["ego_x"]:.3f}, {row["ego_y"]:.3f})\\n'
                           f'Lane: {row["lane_position"]}\\n'
                           f'Speed: {np.sqrt(row["ego_vx"]**2 + row["ego_vy"]**2):.3f}\\n'
                           f'Reward: {row["reward"]:.3f}')
                
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor=agent_colors[i], alpha=0.8),
                       color='white', fontweight='bold')
                
                # Image stats
                stats_text = f'Range: [{image.min():.3f}, {image.max():.3f}]'
                ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                       fontsize=8, horizontalalignment='right',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                
                plt.colorbar(im, ax=ax, shrink=0.8)
                ax.set_xlabel('Width')
                ax.set_ylabel('Height')
                
                success_count += 1
                print(f"  ✅ Agent {agent_id}: decoded {grayscale_array.shape} -> {image.shape}")
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error decoding\\nAgent {agent_id}\\n{str(e)[:50]}...', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='red')
                ax.set_title(f'Agent {agent_id} - Error', fontsize=12, color='red')
                print(f"  ❌ Agent {agent_id}: Error - {e}")
        
        plt.tight_layout()
        
        # Save
        filename = f"agents_step_{step:03d}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        if success_count > 0:
            print(f"  💾 Saved: {filename} ({success_count}/2 agents successful)")
        else:
            print(f"  ❌ Failed to process any agents for step {step}")
    
    print(f"\\n✅ Quick extraction complete!")
    print(f"🖼️  Check output directory: {output_dir}")
    
    return output_dir


def main():
    """Main function with simple command line handling."""
    
    # Default data path
    default_data = "/home/chettra/ITC/Research/AVs/data/highway_multimodal_dataset/dense_commuting/20250921_152749-1ea1c024_transitions.parquet"
    
    # Parse simple command line arguments
    if len(sys.argv) == 1:
        # No arguments: extract first 3 steps
        steps = None
    elif len(sys.argv) == 2:
        # One argument: extract first N steps
        try:
            steps = int(sys.argv[1])
        except ValueError:
            print("❌ Invalid argument. Use: python quick_grayscale.py [N] or [step1 step2 ...]")
            return
    else:
        # Multiple arguments: specific steps
        try:
            steps = [int(arg) for arg in sys.argv[1:]]
        except ValueError:
            print("❌ Invalid arguments. All arguments must be integers (step numbers)")
            return
    
    # Run extraction
    try:
        output_dir = extract_and_display_grayscale(default_data, steps)
        print(f"\\n🎉 Success! Grayscale images saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()