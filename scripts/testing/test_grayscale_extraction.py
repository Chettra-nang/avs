#!/usr/bin/env python3
"""
Test script to quickly extract and display grayscale images for 2 agents.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def decode_grayscale_blob(blob: bytes, shape: np.ndarray, dtype: str) -> np.ndarray:
    """Decode binary blob back to numpy array."""
    array = np.frombuffer(blob, dtype=dtype)
    return array.reshape(shape)

def process_grayscale_image(grayscale_array: np.ndarray) -> np.ndarray:
    """Process grayscale array for visualization."""
    # Handle shape [4, height, width] - channels first
    if grayscale_array.shape[0] == 4:  
        # Convert RGBA to grayscale: 0.299*R + 0.587*G + 0.114*B
        image = (0.299 * grayscale_array[0] + 
                0.587 * grayscale_array[1] + 
                0.114 * grayscale_array[2])
    else:
        image = grayscale_array[0] if len(grayscale_array.shape) > 2 else grayscale_array
    
    # Normalize to [0, 1] range
    if image.max() > 1.0:
        image = image / 255.0
    
    return image

def main():
    """Quick test of grayscale extraction."""
    
    # Load data
    data_path = "/home/chettra/ITC/Research/AVs/data/highway_multimodal_dataset/dense_commuting/20250921_152749-1ea1c024_transitions.parquet"
    
    print("🖼️  Quick Grayscale Test")
    print("📊 Loading data...")
    
    df = pd.read_parquet(data_path)
    
    # Get first episode and first step
    episode_id = df['episode_id'].iloc[0]
    step = 0
    
    print(f"📋 Episode: {episode_id}")
    print(f"📋 Step: {step}")
    
    # Get data for both agents at this step
    step_data = df[(df['episode_id'] == episode_id) & (df['step'] == step)]
    
    print(f"📊 Found {len(step_data)} agent records")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Dual-Agent Grayscale Test\nEpisode: {episode_id}, Step: {step}', 
                fontsize=14, fontweight='bold')
    
    agent_colors = ['red', 'blue']
    
    for i, (_, row) in enumerate(step_data.iterrows()):
        agent_id = row['agent_id']
        ax = axes[i] if i < 2 else axes[-1]
        
        if pd.notna(row['grayscale_blob']):
            print(f"🎯 Processing Agent {agent_id}")
            print(f"   Blob size: {len(row['grayscale_blob'])} bytes")
            print(f"   Shape: {row['grayscale_shape']}")
            print(f"   Dtype: {row['grayscale_dtype']}")
            
            # Decode the blob
            grayscale_array = decode_grayscale_blob(
                row['grayscale_blob'], 
                row['grayscale_shape'], 
                row['grayscale_dtype']
            )
            
            print(f"   Decoded shape: {grayscale_array.shape}")
            print(f"   Value range: [{grayscale_array.min():.3f}, {grayscale_array.max():.3f}]")
            
            # Process for visualization
            image = process_grayscale_image(grayscale_array)
            
            print(f"   Processed shape: {image.shape}")
            print(f"   Processed range: [{image.min():.3f}, {image.max():.3f}]")
            
            # Display
            im = ax.imshow(image, cmap='gray', aspect='auto')
            ax.set_title(f'Agent {agent_id} (Ego View)', 
                       fontsize=12, color=agent_colors[i], fontweight='bold')
            
            # Add ego position
            ego_x, ego_y = row['ego_x'], row['ego_y']
            ax.text(0.02, 0.98, f'Ego Position:\nX: {ego_x:.3f}\nY: {ego_y:.3f}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=agent_colors[i], alpha=0.7))
            
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xlabel('Width (pixels)')
            ax.set_ylabel('Height (pixels)')
            
        else:
            ax.text(0.5, 0.5, f'No grayscale data\nfor Agent {agent_id}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Agent {agent_id} - No Data', fontsize=12, color='gray')
    
    plt.tight_layout()
    
    # Save the test image
    output_path = Path("/home/chettra/ITC/Research/AVs/test_grayscale_extraction.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved test image: {output_path}")
    
    # Show the plot
    plt.show()
    
    print("✅ Test completed!")

if __name__ == "__main__":
    main()