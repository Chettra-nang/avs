#!/usr/bin/env python3
"""
Quick Dual Agent Video Creator
Simple script to quickly create videos showing both agents
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from highway_datacollection.storage.encoders import BinaryArrayEncoder
import os
from pathlib import Path

def create_quick_dual_agent_video():
    """Create a quick video of both agents"""
    
    # Configuration
    data_path = "/home/chettra/ITC/Research/AVs/data/highway_multimodal_dataset/dense_commuting/20250921_152749-1ea1c024_transitions.parquet"
    output_dir = Path("output/quick_videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎬 QUICK DUAL AGENT VIDEO CREATOR")
    print("=" * 50)
    
    # Load data
    print("📊 Loading dataset...")
    df = pd.read_parquet(data_path)
    encoder = BinaryArrayEncoder()
    
    # Get first episode
    episode_id = df['episode_id'].iloc[0]
    episode_data = df[df['episode_id'] == episode_id]
    
    print(f"🎮 Creating video for episode: {episode_id}")
    
    # Get first 15 steps
    steps = sorted(episode_data['step'].unique())[:15]
    print(f"📈 Processing {len(steps)} steps...")
    
    # Collect data
    video_data = []
    for step in steps:
        step_data = episode_data[episode_data['step'] == step]
        
        if len(step_data) == 2:
            agent0_data = step_data[step_data['agent_id'] == 0].iloc[0]
            agent1_data = step_data[step_data['agent_id'] == 1].iloc[0]
            
            try:
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
                if len(image0.shape) == 3 and image0.shape[0] == 4:
                    img0_gray = 0.299 * image0[0] + 0.587 * image0[1] + 0.114 * image0[2]
                    img1_gray = 0.299 * image1[0] + 0.587 * image1[1] + 0.114 * image1[2]
                else:
                    img0_gray = image0
                    img1_gray = image1
                
                video_data.append({
                    'step': step,
                    'img0': img0_gray.astype(np.uint8),
                    'img1': img1_gray.astype(np.uint8),
                    'pos0': (agent0_data['ego_x'], agent0_data['ego_y']),
                    'pos1': (agent1_data['ego_x'], agent1_data['ego_y']),
                    'reward0': agent0_data['reward'],
                    'reward1': agent1_data['reward'],
                    'action0': agent0_data['action'],
                    'action1': agent1_data['action'],
                })
                
            except Exception as e:
                print(f"⚠️ Error processing step {step}: {e}")
                continue
    
    if not video_data:
        print("❌ No valid data to create video")
        return
    
    print(f"✅ Collected {len(video_data)} frames")
    
    # Create the video
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🚗🚗 Dual Agent Highway Visualization', fontsize=16, fontweight='bold')
    
    # Image displays
    ax_img0 = axes[0, 0]
    ax_img1 = axes[0, 1]
    ax_pos = axes[1, 0]
    ax_info = axes[1, 1]
    
    # Set up image displays
    im0 = ax_img0.imshow(video_data[0]['img0'], cmap='gray', animated=True)
    im1 = ax_img1.imshow(video_data[0]['img1'], cmap='gray', animated=True)
    
    ax_img0.set_title('🚗 Agent 0 (Car A)', fontsize=14, fontweight='bold')
    ax_img1.set_title('🚗 Agent 1 (Car B)', fontsize=14, fontweight='bold')
    ax_img0.axis('off')
    ax_img1.axis('off')
    
    # Position tracking
    all_pos0_x = [d['pos0'][0] for d in video_data]
    all_pos0_y = [d['pos0'][1] for d in video_data]
    all_pos1_x = [d['pos1'][0] for d in video_data]
    all_pos1_y = [d['pos1'][1] for d in video_data]
    
    line0, = ax_pos.plot([], [], 'bo-', label='Agent 0', markersize=8, linewidth=2)
    line1, = ax_pos.plot([], [], 'ro-', label='Agent 1', markersize=8, linewidth=2)
    
    ax_pos.set_xlim(min(min(all_pos0_x), min(all_pos1_x)) - 0.1, 
                    max(max(all_pos0_x), max(all_pos1_x)) + 0.1)
    ax_pos.set_ylim(min(min(all_pos0_y), min(all_pos1_y)) - 0.1, 
                    max(max(all_pos0_y), max(all_pos1_y)) + 0.1)
    ax_pos.set_xlabel('X Position')
    ax_pos.set_ylabel('Y Position')
    ax_pos.set_title('🛣️ Vehicle Positions')
    ax_pos.legend()
    ax_pos.grid(True, alpha=0.3)
    
    # Info display
    info_text = ax_info.text(0.1, 0.5, '', transform=ax_info.transAxes, 
                           fontsize=11, verticalalignment='center', 
                           fontfamily='monospace')
    ax_info.set_title('📊 Agent Statistics')
    ax_info.axis('off')
    
    def animate(frame):
        frame_data = video_data[frame]
        
        # Update images
        im0.set_array(frame_data['img0'])
        im1.set_array(frame_data['img1'])
        
        # Update positions
        current_frame = frame + 1
        line0.set_data(all_pos0_x[:current_frame], all_pos0_y[:current_frame])
        line1.set_data(all_pos1_x[:current_frame], all_pos1_y[:current_frame])
        
        # Update info
        same_reward = abs(frame_data['reward0'] - frame_data['reward1']) < 1e-6
        same_position = (abs(frame_data['pos0'][0] - frame_data['pos1'][0]) < 1e-3 and 
                        abs(frame_data['pos0'][1] - frame_data['pos1'][1]) < 1e-3)
        distance = np.sqrt((frame_data['pos0'][0] - frame_data['pos1'][0])**2 + 
                          (frame_data['pos0'][1] - frame_data['pos1'][1])**2)
        
        info_text.set_text(
            f"STEP {frame_data['step']:2d}\n"
            f"{'='*25}\n"
            f"Agent 0 (Car A):\n"
            f"  Position: ({frame_data['pos0'][0]:6.3f}, {frame_data['pos0'][1]:6.3f})\n"
            f"  Reward:   {frame_data['reward0']:6.4f}\n"
            f"  Action:   {frame_data['action0']}\n"
            f"\n"
            f"Agent 1 (Car B):\n"
            f"  Position: ({frame_data['pos1'][0]:6.3f}, {frame_data['pos1'][1]:6.3f})\n"
            f"  Reward:   {frame_data['reward1']:6.4f}\n"
            f"  Action:   {frame_data['action1']}\n"
            f"\n"
            f"Analysis:\n"
            f"  Same Rewards: {'YES' if same_reward else 'NO'}\n"
            f"  Same Position: {'YES' if same_position else 'NO'}\n"
            f"  Distance: {distance:.4f} units\n"
            f"\n"
            f"🎯 2 Cars, Same Environment!"
        )
        
        return [im0, im1, line0, line1, info_text]
    
    # Create animation
    print("🎬 Creating animation...")
    anim = animation.FuncAnimation(
        fig, animate, frames=len(video_data), 
        interval=800, blit=True, repeat=True
    )
    
    # Save as GIF (most compatible)
    output_file = output_dir / f"{episode_id}_dual_agents.gif"
    print(f"💾 Saving GIF to: {output_file}")
    
    try:
        anim.save(output_file, writer='pillow', fps=1.5)
        print(f"✅ GIF saved successfully!")
        
        # Also save individual frames as images
        frames_dir = output_dir / f"{episode_id}_frames"
        frames_dir.mkdir(exist_ok=True)
        
        print(f"📸 Saving individual frames to: {frames_dir}")
        
        for i, frame_data in enumerate(video_data):
            plt.figure(figsize=(16, 8))
            
            plt.subplot(1, 2, 1)
            plt.imshow(frame_data['img0'], cmap='gray')
            plt.title(f'Agent 0 - Step {frame_data["step"]}')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(frame_data['img1'], cmap='gray')
            plt.title(f'Agent 1 - Step {frame_data["step"]}')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(frames_dir / f"step_{frame_data['step']:03d}.png", dpi=100, bbox_inches='tight')
            plt.close()
        
        print(f"📸 Saved {len(video_data)} individual frames")
        
    except Exception as e:
        print(f"❌ Error saving: {e}")
    finally:
        plt.close(fig)
    
    print(f"\n🎉 COMPLETED!")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎬 Main video: {output_file}")
    print(f"📸 Individual frames: {frames_dir}")
    
    return output_file

if __name__ == "__main__":
    create_quick_dual_agent_video()