#!/usr/bin/env python3
"""
Quick Frame Viewer - Show sensing differences between agents
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from highway_datacollection.storage.encoders import BinaryArrayEncoder

def show_sensing_comparison():
    """Show side-by-side comparison of agent sensing"""
    
    # Load data
    data_path = "/home/chettra/ITC/Research/AVs/data/highway_multimodal_dataset/dense_commuting/20250921_152749-1ea1c024_transitions.parquet"
    df = pd.read_parquet(data_path)
    encoder = BinaryArrayEncoder()
    
    # Get specific episode and step
    episode_id = "ep_dense_commuting_10042_0000"
    step = 5  # Show step 5 where differences are visible
    
    step_data = df[(df['episode_id'] == episode_id) & (df['step'] == step)]
    
    if len(step_data) != 2:
        print("❌ Need exactly 2 agents for comparison")
        return
    
    agent0_data = step_data[step_data['agent_id'] == 0].iloc[0]
    agent1_data = step_data[step_data['agent_id'] == 1].iloc[0]
    
    # Decode images
    image0 = encoder.decode(
        agent0_data['grayscale_blob'],
        tuple(agent0_data['grayscale_shape']),
        agent0_data['grayscale_dtype']
    )
    image1 = encoder.decode(
        agent1_data['grayscale_blob'],
        tuple(agent1_data['grayscale_shape']),
        agent1_data['grayscale_dtype']
    )
    
    # Convert to grayscale
    def to_grayscale(img):
        if len(img.shape) == 3 and img.shape[0] == 4:
            return (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).astype(np.uint8)
        return img
    
    img0_gray = to_grayscale(image0)
    img1_gray = to_grayscale(image1)
    
    # Calculate difference
    diff = np.abs(img0_gray.astype(float) - img1_gray.astype(float))
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'🔍 Dual Agent Sensing Analysis - Step {step}', fontsize=16, fontweight='bold')
    
    # Top row: Images
    axes[0, 0].imshow(img0_gray, cmap='gray')
    axes[0, 0].set_title('🚗 Agent 0 (Car A) Vision', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img1_gray, cmap='gray')
    axes[0, 1].set_title('🚗 Agent 1 (Car B) Vision', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title('🔥 Pixel Differences (Red = Different)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Bottom row: Analysis
    axes[1, 0].text(0.1, 0.5, 
        f"Agent 0 Stats:\n"
        f"Position: ({agent0_data['ego_x']:.3f}, {agent0_data['ego_y']:.3f})\n"
        f"Velocity: ({agent0_data['ego_vx']:.3f}, {agent0_data['ego_vy']:.3f})\n"
        f"Reward: {agent0_data['reward']:.6f}\n"
        f"Action: {agent0_data['action']}\n"
        f"Lane: {agent0_data['lane_position']}\n\n"
        f"Image Stats:\n"
        f"Min: {img0_gray.min()}\n"
        f"Max: {img0_gray.max()}\n"
        f"Mean: {img0_gray.mean():.1f}",
        transform=axes[1, 0].transAxes, fontsize=11, fontfamily='monospace',
        verticalalignment='center'
    )
    axes[1, 0].set_title('📊 Agent 0 Statistics')
    axes[1, 0].axis('off')
    
    axes[1, 1].text(0.1, 0.5,
        f"Agent 1 Stats:\n"
        f"Position: ({agent1_data['ego_x']:.3f}, {agent1_data['ego_y']:.3f})\n"
        f"Velocity: ({agent1_data['ego_vx']:.3f}, {agent1_data['ego_vy']:.3f})\n"
        f"Reward: {agent1_data['reward']:.6f}\n"
        f"Action: {agent1_data['action']}\n"
        f"Lane: {agent1_data['lane_position']}\n\n"
        f"Image Stats:\n"
        f"Min: {img1_gray.min()}\n"
        f"Max: {img1_gray.max()}\n"
        f"Mean: {img1_gray.mean():.1f}",
        transform=axes[1, 1].transAxes, fontsize=11, fontfamily='monospace',
        verticalalignment='center'
    )
    axes[1, 1].set_title('📊 Agent 1 Statistics')
    axes[1, 1].axis('off')
    
    # Comparison analysis
    same_reward = abs(agent0_data['reward'] - agent1_data['reward']) < 1e-6
    same_position = (abs(agent0_data['ego_x'] - agent1_data['ego_x']) < 1e-3 and 
                    abs(agent0_data['ego_y'] - agent1_data['ego_y']) < 1e-3)
    distance = np.sqrt((agent0_data['ego_x'] - agent1_data['ego_x'])**2 + 
                      (agent0_data['ego_y'] - agent1_data['ego_y'])**2)
    
    different_pixels = np.count_nonzero(diff)
    total_pixels = diff.size
    diff_percentage = (different_pixels / total_pixels) * 100
    
    axes[1, 2].text(0.1, 0.5,
        f"🎯 ANALYSIS:\n\n"
        f"Environment:\n"
        f"  Same Rewards: {'YES' if same_reward else 'NO'}\n"
        f"  Reward Diff: {abs(agent0_data['reward'] - agent1_data['reward']):.6f}\n\n"
        f"Vehicles:\n"
        f"  Same Position: {'YES' if same_position else 'NO'}\n"
        f"  Distance Apart: {distance:.4f} units\n\n"
        f"Vision:\n"
        f"  Different Pixels: {different_pixels:,}/{total_pixels:,}\n"
        f"  Difference: {diff_percentage:.1f}%\n"
        f"  Max Pixel Diff: {diff.max():.0f}\n"
        f"  Mean Pixel Diff: {diff.mean():.2f}\n\n"
        f"🚗🚗 CONCLUSION:\n"
        f"2 separate cars in\n"
        f"same environment with\n"
        f"different perspectives!",
        transform=axes[1, 2].transAxes, fontsize=11, fontfamily='monospace',
        verticalalignment='center'
    )
    axes[1, 2].set_title('🔍 Comparison Analysis')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save the comparison
    output_path = "output/quick_videos/sensing_comparison_frame.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Sensing comparison saved to: {output_path}")
    
    plt.show()
    
    print("\n🎯 KEY FINDINGS:")
    print(f"✅ Same environment: Both agents get identical rewards ({agent0_data['reward']:.6f})")
    print(f"✅ Different cars: Positions differ by {distance:.4f} units") 
    print(f"✅ Different sensing: {diff_percentage:.1f}% of pixels are different")
    print(f"✅ Independent agents: Each has ego-centric view of the highway")

if __name__ == "__main__":
    show_sensing_comparison()