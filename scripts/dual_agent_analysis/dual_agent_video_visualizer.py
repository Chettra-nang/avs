#!/usr/bin/env python3
"""
Dual Agent Video Visualizer
Creates a video showing both agents' movements and sensing in the same environment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from highway_datacollection.storage.encoders import BinaryArrayEncoder
import argparse
import os
from pathlib import Path

class DualAgentVideoVisualizer:
    def __init__(self, data_path, output_dir="output/dual_agent_videos"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print(f"📊 Loading data from {data_path}")
        self.df = pd.read_parquet(data_path)
        self.encoder = BinaryArrayEncoder()
        
        # Get available episodes
        self.episodes = list(self.df['episode_id'].unique())
        print(f"Found {len(self.episodes)} episodes")
        
    def create_dual_agent_video(self, episode_id, max_steps=None, fps=2):
        """Create a video showing both agents simultaneously"""
        print(f"🎬 Creating video for episode: {episode_id}")
        
        # Get episode data
        episode_data = self.df[self.df['episode_id'] == episode_id].copy()
        
        if len(episode_data) == 0:
            print(f"❌ No data found for episode {episode_id}")
            return None
            
        # Get steps and limit if requested
        steps = sorted(episode_data['step'].unique())
        if max_steps:
            steps = steps[:max_steps]
            
        print(f"Processing {len(steps)} steps...")
        
        # Prepare data for animation
        animation_data = []
        
        for step in steps:
            step_data = episode_data[episode_data['step'] == step]
            
            if len(step_data) == 2:  # Both agents present
                agent0_data = step_data[step_data['agent_id'] == 0].iloc[0]
                agent1_data = step_data[step_data['agent_id'] == 1].iloc[0]
                
                # Decode images
                try:
                    image0 = self.encoder.decode(
                        agent0_data['grayscale_blob'],
                        tuple(agent0_data['grayscale_shape']),
                        agent0_data['grayscale_dtype']
                    )
                    image1 = self.encoder.decode(
                        agent1_data['grayscale_blob'],
                        tuple(agent1_data['grayscale_shape']),
                        agent1_data['grayscale_dtype']
                    )
                    
                    # Convert to displayable format
                    img0_display = self._prepare_image_for_display(image0)
                    img1_display = self._prepare_image_for_display(image1)
                    
                    animation_data.append({
                        'step': step,
                        'agent0': {
                            'image': img0_display,
                            'position': (agent0_data['ego_x'], agent0_data['ego_y']),
                            'velocity': (agent0_data['ego_vx'], agent0_data['ego_vy']),
                            'reward': agent0_data['reward'],
                            'action': agent0_data['action'],
                            'lane': agent0_data['lane_position'],
                            'speed': np.sqrt(agent0_data['ego_vx']**2 + agent0_data['ego_vy']**2)
                        },
                        'agent1': {
                            'image': img1_display,
                            'position': (agent1_data['ego_x'], agent1_data['ego_y']),
                            'velocity': (agent1_data['ego_vx'], agent1_data['ego_vy']),
                            'reward': agent1_data['reward'],
                            'action': agent1_data['action'],
                            'lane': agent1_data['lane_position'],
                            'speed': np.sqrt(agent1_data['ego_vx']**2 + agent1_data['ego_vy']**2)
                        }
                    })
                    
                except Exception as e:
                    print(f"⚠️ Error processing step {step}: {e}")
                    continue
        
        if not animation_data:
            print("❌ No valid data to animate")
            return None
            
        # Create the animation
        return self._create_animation(animation_data, episode_id, fps)
    
    def _prepare_image_for_display(self, image):
        """Convert 4-channel image to grayscale for display"""
        if len(image.shape) == 3 and image.shape[0] == 4:
            # Convert RGBA to grayscale using standard weights
            rgb_image = image[:3]  # Take RGB channels
            grayscale = (0.299 * rgb_image[0] + 0.587 * rgb_image[1] + 0.114 * rgb_image[2])
            return grayscale.astype(np.uint8)
        else:
            return image
    
    def _create_animation(self, data, episode_id, fps):
        """Create the matplotlib animation"""
        
        # Set up the figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Create subplots: 2x3 grid
        # Top row: Agent images
        ax_img0 = plt.subplot2grid((3, 4), (0, 0), colspan=2)
        ax_img1 = plt.subplot2grid((3, 4), (0, 2), colspan=2)
        
        # Middle row: Position tracking
        ax_pos = plt.subplot2grid((3, 4), (1, 0), colspan=4)
        
        # Bottom row: Stats
        ax_stats = plt.subplot2grid((3, 4), (2, 0), colspan=4)
        
        # Initialize plots
        im0 = ax_img0.imshow(data[0]['agent0']['image'], cmap='gray', animated=True)
        im1 = ax_img1.imshow(data[0]['agent1']['image'], cmap='gray', animated=True)
        
        ax_img0.set_title("🚗 Agent 0 (Car A) View", fontsize=14, fontweight='bold')
        ax_img1.set_title("🚗 Agent 1 (Car B) View", fontsize=14, fontweight='bold')
        ax_img0.axis('off')
        ax_img1.axis('off')
        
        # Position tracking setup
        steps = [d['step'] for d in data]
        pos0_x = [d['agent0']['position'][0] for d in data]
        pos0_y = [d['agent0']['position'][1] for d in data]
        pos1_x = [d['agent1']['position'][0] for d in data]
        pos1_y = [d['agent1']['position'][1] for d in data]
        
        line0, = ax_pos.plot([], [], 'bo-', label='Agent 0 (Car A)', markersize=8, linewidth=2)
        line1, = ax_pos.plot([], [], 'ro-', label='Agent 1 (Car B)', markersize=8, linewidth=2)
        
        ax_pos.set_xlim(min(min(pos0_x), min(pos1_x)) - 0.1, max(max(pos0_x), max(pos1_x)) + 0.1)
        ax_pos.set_ylim(min(min(pos0_y), min(pos1_y)) - 0.1, max(max(pos0_y), max(pos1_y)) + 0.1)
        ax_pos.set_xlabel('X Position', fontsize=12)
        ax_pos.set_ylabel('Y Position', fontsize=12)
        ax_pos.set_title('🛣️ Vehicle Positions in Highway Environment', fontsize=14, fontweight='bold')
        ax_pos.legend()
        ax_pos.grid(True, alpha=0.3)
        
        # Stats text
        stats_text = ax_stats.text(0.05, 0.5, '', transform=ax_stats.transAxes, 
                                  fontsize=11, verticalalignment='center', 
                                  fontfamily='monospace')
        ax_stats.axis('off')
        
        def animate(frame):
            """Animation function"""
            frame_data = data[frame]
            
            # Update images
            im0.set_array(frame_data['agent0']['image'])
            im1.set_array(frame_data['agent1']['image'])
            
            # Update position tracking
            current_step = frame + 1
            line0.set_data(pos0_x[:current_step], pos0_y[:current_step])
            line1.set_data(pos1_x[:current_step], pos1_y[:current_step])
            
            # Update stats
            agent0 = frame_data['agent0']
            agent1 = frame_data['agent1']
            
            stats_text.set_text(
                f"📊 STEP {frame_data['step']:2d} STATISTICS\n"
                f"{'='*80}\n"
                f"🚗 Agent 0 (Car A)     │  🚗 Agent 1 (Car B)\n"
                f"Position: ({agent0['position'][0]:6.3f}, {agent0['position'][1]:6.3f})  │  Position: ({agent1['position'][0]:6.3f}, {agent1['position'][1]:6.3f})\n"
                f"Velocity: ({agent0['velocity'][0]:6.3f}, {agent0['velocity'][1]:6.3f})  │  Velocity: ({agent1['velocity'][0]:6.3f}, {agent1['velocity'][1]:6.3f})\n"
                f"Speed:    {agent0['speed']:6.3f}           │  Speed:    {agent1['speed']:6.3f}\n"
                f"Lane:     {agent0['lane']:6.0f}             │  Lane:     {agent1['lane']:6.0f}\n"
                f"Reward:   {agent0['reward']:6.4f}          │  Reward:   {agent1['reward']:6.4f}\n"
                f"Action:   {str(agent0['action']):12s}     │  Action:   {str(agent1['action'])}\n"
                f"{'='*80}\n"
                f"🎯 Environment Analysis:\n"
                f"   • Same Rewards: {'YES' if abs(agent0['reward'] - agent1['reward']) < 1e-6 else 'NO'} "
                f"(Δ = {abs(agent0['reward'] - agent1['reward']):.6f})\n"
                f"   • Same Position: {'YES' if abs(agent0['position'][0] - agent1['position'][0]) < 1e-3 and abs(agent0['position'][1] - agent1['position'][1]) < 1e-3 else 'NO'}\n"
                f"   • Distance Apart: {np.sqrt((agent0['position'][0] - agent1['position'][0])**2 + (agent0['position'][1] - agent1['position'][1])**2):.4f} units"
            )
            
            return [im0, im1, line0, line1, stats_text]
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(data), 
            interval=1000//fps, blit=True, repeat=True
        )
        
        # Save animation
        output_file = self.output_dir / f"{episode_id}_dual_agent_video.mp4"
        print(f"💾 Saving video to: {output_file}")
        
        try:
            anim.save(output_file, writer='ffmpeg', fps=fps, bitrate=1800, 
                     extra_args=['-vcodec', 'libx264'])
            print(f"✅ Video saved successfully!")
            
            # Also save as GIF for easier viewing
            gif_file = self.output_dir / f"{episode_id}_dual_agent_video.gif"
            print(f"💾 Saving GIF to: {gif_file}")
            anim.save(gif_file, writer='pillow', fps=fps)
            print(f"✅ GIF saved successfully!")
            
        except Exception as e:
            print(f"⚠️ Error saving video: {e}")
            print("Trying to save as GIF only...")
            gif_file = self.output_dir / f"{episode_id}_dual_agent_video.gif"
            anim.save(gif_file, writer='pillow', fps=fps)
            print(f"✅ GIF saved to: {gif_file}")
        
        plt.close(fig)
        return output_file
    
    def create_sensing_comparison_video(self, episode_id, max_steps=None, fps=2):
        """Create a video focusing on sensing differences"""
        print(f"👁️ Creating sensing comparison video for: {episode_id}")
        
        episode_data = self.df[self.df['episode_id'] == episode_id].copy()
        steps = sorted(episode_data['step'].unique())
        if max_steps:
            steps = steps[:max_steps]
        
        # Set up figure for sensing comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'🔍 Dual Agent Sensing Comparison - {episode_id}', fontsize=16, fontweight='bold')
        
        # Image comparison plots
        ax_img0, ax_img1, ax_diff = axes[0]
        ax_img0.set_title('Agent 0 Vision', fontweight='bold')
        ax_img1.set_title('Agent 1 Vision', fontweight='bold') 
        ax_diff.set_title('Vision Difference', fontweight='bold')
        
        # Stats plots
        ax_pos, ax_reward, ax_action = axes[1]
        
        animation_frames = []
        
        for step in steps:
            step_data = episode_data[episode_data['step'] == step]
            if len(step_data) == 2:
                agent0_data = step_data[step_data['agent_id'] == 0].iloc[0]
                agent1_data = step_data[step_data['agent_id'] == 1].iloc[0]
                
                try:
                    image0 = self.encoder.decode(
                        agent0_data['grayscale_blob'],
                        tuple(agent0_data['grayscale_shape']),
                        agent0_data['grayscale_dtype']
                    )
                    image1 = self.encoder.decode(
                        agent1_data['grayscale_blob'],
                        tuple(agent1_data['grayscale_shape']),
                        agent1_data['grayscale_dtype']
                    )
                    
                    img0_gray = self._prepare_image_for_display(image0)
                    img1_gray = self._prepare_image_for_display(image1)
                    diff_img = np.abs(img0_gray.astype(float) - img1_gray.astype(float))
                    
                    animation_frames.append({
                        'step': step,
                        'img0': img0_gray,
                        'img1': img1_gray,
                        'diff': diff_img,
                        'agent0_data': agent0_data,
                        'agent1_data': agent1_data
                    })
                    
                except Exception as e:
                    print(f"Error processing step {step}: {e}")
        
        if not animation_frames:
            print("No valid frames to animate")
            return None
        
        # Initialize plots
        im0 = ax_img0.imshow(animation_frames[0]['img0'], cmap='gray', animated=True)
        im1 = ax_img1.imshow(animation_frames[0]['img1'], cmap='gray', animated=True)
        im_diff = ax_diff.imshow(animation_frames[0]['diff'], cmap='hot', animated=True)
        
        for ax in [ax_img0, ax_img1, ax_diff]:
            ax.axis('off')
        
        # Position plot
        positions0 = [(f['agent0_data']['ego_x'], f['agent0_data']['ego_y']) for f in animation_frames]
        positions1 = [(f['agent1_data']['ego_x'], f['agent1_data']['ego_y']) for f in animation_frames]
        
        pos_line0, = ax_pos.plot([], [], 'b-o', label='Agent 0', markersize=6)
        pos_line1, = ax_pos.plot([], [], 'r-o', label='Agent 1', markersize=6)
        ax_pos.set_title('Position Tracking')
        ax_pos.legend()
        ax_pos.grid(True, alpha=0.3)
        
        # Reward plot
        rewards0 = [f['agent0_data']['reward'] for f in animation_frames]
        rewards1 = [f['agent1_data']['reward'] for f in animation_frames]
        reward_steps = [f['step'] for f in animation_frames]
        
        reward_line0, = ax_reward.plot([], [], 'b-', label='Agent 0', linewidth=2)
        reward_line1, = ax_reward.plot([], [], 'r-', label='Agent 1', linewidth=2)
        ax_reward.set_xlim(min(reward_steps), max(reward_steps))
        ax_reward.set_ylim(min(min(rewards0), min(rewards1)) - 0.1, max(max(rewards0), max(rewards1)) + 0.1)
        ax_reward.set_title('Rewards Over Time')
        ax_reward.legend()
        ax_reward.grid(True, alpha=0.3)
        
        # Action display
        action_text = ax_action.text(0.5, 0.5, '', ha='center', va='center', 
                                   transform=ax_action.transAxes, fontsize=12, fontfamily='monospace')
        ax_action.set_title('Current Actions')
        ax_action.axis('off')
        
        def animate_sensing(frame):
            frame_data = animation_frames[frame]
            
            # Update images
            im0.set_array(frame_data['img0'])
            im1.set_array(frame_data['img1'])
            im_diff.set_array(frame_data['diff'])
            
            # Update position
            current_frame = frame + 1
            pos_x0 = [positions0[i][0] for i in range(current_frame)]
            pos_y0 = [positions0[i][1] for i in range(current_frame)]
            pos_x1 = [positions1[i][0] for i in range(current_frame)]
            pos_y1 = [positions1[i][1] for i in range(current_frame)]
            
            pos_line0.set_data(pos_x0, pos_y0)
            pos_line1.set_data(pos_x1, pos_y1)
            
            # Update rewards
            current_steps = reward_steps[:current_frame]
            current_rewards0 = rewards0[:current_frame]
            current_rewards1 = rewards1[:current_frame]
            
            reward_line0.set_data(current_steps, current_rewards0)
            reward_line1.set_data(current_steps, current_rewards1)
            
            # Update actions
            agent0_action = frame_data['agent0_data']['action']
            agent1_action = frame_data['agent1_data']['action']
            
            action_text.set_text(
                f"Step {frame_data['step']}\n\n"
                f"Agent 0: {agent0_action}\n"
                f"Agent 1: {agent1_action}\n\n"
                f"Same Action: {'YES' if np.array_equal(agent0_action, agent1_action) else 'NO'}"
            )
            
            return [im0, im1, im_diff, pos_line0, pos_line1, reward_line0, reward_line1, action_text]
        
        anim = animation.FuncAnimation(
            fig, animate_sensing, frames=len(animation_frames),
            interval=1000//fps, blit=True, repeat=True
        )
        
        # Save sensing comparison video
        output_file = self.output_dir / f"{episode_id}_sensing_comparison.gif"
        print(f"💾 Saving sensing comparison to: {output_file}")
        anim.save(output_file, writer='pillow', fps=fps)
        
        plt.close(fig)
        print(f"✅ Sensing comparison saved!")
        return output_file


def main():
    parser = argparse.ArgumentParser(description='Create dual agent visualization videos')
    parser.add_argument('--data', required=True, help='Path to parquet data file')
    parser.add_argument('--episode', help='Specific episode ID to visualize')
    parser.add_argument('--output', default='output/dual_agent_videos', help='Output directory')
    parser.add_argument('--max-steps', type=int, default=20, help='Maximum steps to include')
    parser.add_argument('--fps', type=int, default=2, help='Frames per second')
    parser.add_argument('--sensing', action='store_true', help='Create sensing comparison video')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = DualAgentVideoVisualizer(args.data, args.output)
    
    # Get episode to visualize
    if args.episode:
        if args.episode not in visualizer.episodes:
            print(f"❌ Episode {args.episode} not found!")
            print(f"Available episodes: {visualizer.episodes[:5]}...")
            return
        episodes_to_process = [args.episode]
    else:
        episodes_to_process = visualizer.episodes[:3]  # First 3 episodes
        print(f"🎬 Processing first 3 episodes: {episodes_to_process}")
    
    # Create videos
    for episode in episodes_to_process:
        print(f"\n🎥 Processing episode: {episode}")
        
        # Create main dual agent video
        visualizer.create_dual_agent_video(episode, args.max_steps, args.fps)
        
        # Create sensing comparison if requested
        if args.sensing:
            visualizer.create_sensing_comparison_video(episode, args.max_steps, args.fps)
    
    print(f"\n✅ All videos saved to: {args.output}")
    print("🎬 You can now watch the videos to see both agents in action!")


if __name__ == "__main__":
    main()