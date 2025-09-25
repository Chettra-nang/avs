#!/usr/bin/env python3
"""
Test script to extract grayscale images using the proper decoder.
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

def main():
    """Test grayscale extraction with proper decoder."""
    
    # Load data
    data_path = "/home/chettra/ITC/Research/AVs/data/highway_multimodal_dataset/dense_commuting/20250921_152749-1ea1c024_transitions.parquet"
    
    print("🖼️  Grayscale Extraction Test (Using Proper Decoder)")
    print("📊 Loading data...")
    
    df = pd.read_parquet(data_path)
    
    # Initialize decoder
    decoder = BinaryArrayEncoder()
    
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
    fig.suptitle(f'Dual-Agent Grayscale Images\nEpisode: {episode_id}, Step: {step}', 
                fontsize=14, fontweight='bold')
    
    agent_colors = ['red', 'blue']
    
    for i, (_, row) in enumerate(step_data.iterrows()):
        agent_id = row['agent_id']
        ax = axes[i] if i < 2 else axes[-1]
        
        if pd.notna(row['grayscale_blob']):
            print(f"\n🎯 Processing Agent {agent_id}")
            print(f"   Blob size: {len(row['grayscale_blob'])} bytes")
            print(f"   Shape: {row['grayscale_shape']}")
            print(f"   Dtype: {row['grayscale_dtype']}")
            
            try:
                # Use proper decoder
                grayscale_array = decoder.decode(
                    row['grayscale_blob'], 
                    tuple(row['grayscale_shape']), 
                    row['grayscale_dtype']
                )
                
                print(f"   ✅ Decoded shape: {grayscale_array.shape}")
                print(f"   Value range: [{grayscale_array.min():.3f}, {grayscale_array.max():.3f}]")
                
                # Process for visualization
                # Shape should be [4, 128, 64] - likely RGBA channels
                if grayscale_array.shape[0] == 4:  # Channels first
                    # Convert RGBA to grayscale
                    if grayscale_array.shape[0] >= 3:
                        # Use RGB channels for grayscale conversion
                        image = (0.299 * grayscale_array[0] + 
                                0.587 * grayscale_array[1] + 
                                0.114 * grayscale_array[2])
                    else:
                        image = grayscale_array[0]
                else:
                    # Handle other formats
                    if len(grayscale_array.shape) == 3 and grayscale_array.shape[-1] >= 3:
                        # Channels last format
                        image = (0.299 * grayscale_array[:, :, 0] + 
                                0.587 * grayscale_array[:, :, 1] + 
                                0.114 * grayscale_array[:, :, 2])
                    else:
                        image = grayscale_array
                
                # Normalize to [0, 1] for display
                if image.max() > 1.0:
                    image = image / 255.0
                
                print(f"   📐 Processed image shape: {image.shape}")
                print(f"   🎨 Display range: [{image.min():.3f}, {image.max():.3f}]")
                
                # Display the image
                im = ax.imshow(image, cmap='gray', aspect='auto', origin='upper')
                ax.set_title(f'Agent {agent_id} Perspective', 
                           fontsize=12, color=agent_colors[i], fontweight='bold')
                
                # Add ego position annotation
                ego_x, ego_y = row['ego_x'], row['ego_y']
                lane_pos = row['lane_position']
                ax.text(0.02, 0.98, 
                       f'Agent {agent_id}\nEgo X: {ego_x:.3f}\nEgo Y: {ego_y:.3f}\nLane: {lane_pos}', 
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=agent_colors[i], alpha=0.8),
                       color='white', fontweight='bold')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, shrink=0.8)
                
                # Set labels
                ax.set_xlabel('Width (pixels)', fontsize=10)
                ax.set_ylabel('Height (pixels)', fontsize=10)
                
                # Add some statistics
                stats_text = f'Min: {image.min():.3f}\nMax: {image.max():.3f}\nMean: {image.mean():.3f}'
                ax.text(0.98, 0.02, stats_text, 
                       transform=ax.transAxes, fontsize=8, 
                       verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                
            except Exception as e:
                print(f"   ❌ Error decoding Agent {agent_id}: {e}")
                ax.text(0.5, 0.5, f'Error decoding\nAgent {agent_id}\n{str(e)}', 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=10, color='red')
                ax.set_title(f'Agent {agent_id} - Decode Error', fontsize=12, color='red')
        
        else:
            print(f"   ❌ No grayscale data for Agent {agent_id}")
            ax.text(0.5, 0.5, f'No grayscale data\nfor Agent {agent_id}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Agent {agent_id} - No Data', fontsize=12, color='gray')
    
    plt.tight_layout()
    
    # Save the test image
    output_path = Path("/home/chettra/ITC/Research/AVs/dual_agent_grayscale_test.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved test image: {output_path}")
    
    # Show additional analysis
    print(f"\n📊 Dataset Analysis:")
    print(f"   Total rows: {len(df)}")
    print(f"   Episodes: {df['episode_id'].nunique()}")
    print(f"   Agents: {sorted(df['agent_id'].unique())}")
    print(f"   Steps range: {df['step'].min()} - {df['step'].max()}")
    
    # Check grayscale availability
    grayscale_available = df['grayscale_blob'].notna().sum()
    print(f"   Grayscale images: {grayscale_available}/{len(df)} ({100*grayscale_available/len(df):.1f}%)")
    
    plt.show()
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    main()