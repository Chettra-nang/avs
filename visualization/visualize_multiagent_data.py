#!/usr/bin/env python3
"""
Multi-Agent Data Visualization Script

This script creates comprehensive visualizations of the collected multi-agent highway data.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import glob
from datetime import datetime
import matplotlib.patches as patches

class MultiAgentDataVisualizer:
    """Visualize multi-agent highway simulation data."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.agent_colors = ['red', 'green', 'blue', 'orange']
        self.agent_names = ['Conservative', 'Aggressive', 'Speed Keeper', 'Adaptive']
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_data(self, filename=None):
        """Load multi-agent data from JSON file."""
        if filename is None:
            # Get the most recent file
            pattern = str(self.data_dir / "multiagent_demo_*.json")
            files = glob.glob(pattern)
            if not files:
                raise FileNotFoundError("No multi-agent data files found!")
            filename = max(files, key=lambda x: Path(x).stat().st_mtime)
        
        print(f"📊 Loading data from: {filename}")
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return data
    
    def create_comprehensive_dashboard(self, data, save_plots=True):
        """Create a comprehensive dashboard with multiple visualizations."""
        print("🎨 Creating comprehensive multi-agent dashboard...")
        
        # Create a large figure with subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'Multi-Agent Highway Simulation Dashboard\n'
                    f'Scenario: {data.get("scenario", "Unknown")} | '
                    f'Agents: {data.get("n_agents", 4)} | '
                    f'Episodes: {len(data.get("episodes", []))}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Rewards over time (top left)
        ax1 = plt.subplot(3, 3, 1)
        self.plot_rewards_over_time(data, ax1)
        
        # 2. Agent trajectories (top center)
        ax2 = plt.subplot(3, 3, 2)
        self.plot_agent_trajectories(data, ax2)
        
        # 3. Speed profiles (top right)
        ax3 = plt.subplot(3, 3, 3)
        self.plot_speed_profiles(data, ax3)
        
        # 4. Action distribution (middle left)
        ax4 = plt.subplot(3, 3, 4)
        self.plot_action_distribution(data, ax4)
        
        # 5. Cumulative rewards (middle center)
        ax5 = plt.subplot(3, 3, 5)
        self.plot_cumulative_rewards(data, ax5)
        
        # 6. Lane usage heatmap (middle right)
        ax6 = plt.subplot(3, 3, 6)
        self.plot_lane_usage_heatmap(data, ax6)
        
        # 7. Episode summary (bottom left)
        ax7 = plt.subplot(3, 3, 7)
        self.plot_episode_summary(data, ax7)
        
        # 8. Agent performance comparison (bottom center)
        ax8 = plt.subplot(3, 3, 8)
        self.plot_agent_performance_comparison(data, ax8)
        
        # 9. Reward distribution (bottom right)
        ax9 = plt.subplot(3, 3, 9)
        self.plot_reward_distribution(data, ax9)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multiagent_dashboard_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Dashboard saved as: {filename}")
        
        plt.show()
    
    def plot_rewards_over_time(self, data, ax):
        """Plot rewards over time for all agents."""
        ax.set_title("Rewards Over Time", fontweight='bold')
        
        for episode_idx, episode in enumerate(data['episodes']):
            steps = [step['step'] for step in episode['data']]
            
            for agent_idx in range(data['n_agents']):
                rewards = [step['rewards'][agent_idx] for step in episode['data']]
                
                # Offset steps for different episodes
                offset_steps = [s + episode_idx * 100 for s in steps]
                
                ax.plot(offset_steps, rewards, 
                       color=self.agent_colors[agent_idx], 
                       alpha=0.7,
                       label=f'Agent {agent_idx+1}' if episode_idx == 0 else "")
        
        ax.set_xlabel("Step (offset by episode)")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_agent_trajectories(self, data, ax):
        """Plot agent trajectories in 2D space."""
        ax.set_title("Agent Trajectories", fontweight='bold')
        
        for episode_idx, episode in enumerate(data['episodes']):
            for agent_idx in range(data['n_agents']):
                x_positions = []
                y_positions = []
                
                for step in episode['data']:
                    if f'agent_{agent_idx}_x' in step and f'agent_{agent_idx}_y' in step:
                        x_positions.append(step[f'agent_{agent_idx}_x'])
                        y_positions.append(step[f'agent_{agent_idx}_y'])
                
                if x_positions and y_positions:
                    ax.plot(x_positions, y_positions, 
                           color=self.agent_colors[agent_idx],
                           alpha=0.6,
                           linewidth=2,
                           label=f'{self.agent_names[agent_idx]}' if episode_idx == 0 else "")
                    
                    # Mark start and end points
                    ax.scatter(x_positions[0], y_positions[0], 
                             color=self.agent_colors[agent_idx], 
                             marker='o', s=50, alpha=0.8)
                    ax.scatter(x_positions[-1], y_positions[-1], 
                             color=self.agent_colors[agent_idx], 
                             marker='s', s=50, alpha=0.8)
        
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position (Lane)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_speed_profiles(self, data, ax):
        """Plot speed profiles for all agents."""
        ax.set_title("Speed Profiles", fontweight='bold')
        
        for episode_idx, episode in enumerate(data['episodes']):
            steps = [step['step'] for step in episode['data']]
            
            for agent_idx in range(data['n_agents']):
                speeds = []
                for step in episode['data']:
                    if f'agent_{agent_idx}_speed' in step:
                        speeds.append(step[f'agent_{agent_idx}_speed'])
                
                if speeds:
                    # Offset steps for different episodes
                    offset_steps = [s + episode_idx * 100 for s in steps]
                    
                    ax.plot(offset_steps, speeds, 
                           color=self.agent_colors[agent_idx],
                           alpha=0.7,
                           label=f'Agent {agent_idx+1}' if episode_idx == 0 else "")
        
        ax.set_xlabel("Step (offset by episode)")
        ax.set_ylabel("Speed")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_action_distribution(self, data, ax):
        """Plot action distribution for all agents."""
        ax.set_title("Action Distribution", fontweight='bold')
        
        action_names = ['SLOWER', 'IDLE', 'FASTER', 'LANE_LEFT', 'LANE_RIGHT']
        action_counts = {agent_idx: [0] * 5 for agent_idx in range(data['n_agents'])}
        
        for episode in data['episodes']:
            for step in episode['data']:
                for agent_idx, action in enumerate(step['actions']):
                    if action < 5:  # Valid action
                        action_counts[agent_idx][action] += 1
        
        # Create grouped bar chart
        x = np.arange(len(action_names))
        width = 0.2
        
        for agent_idx in range(data['n_agents']):
            offset = (agent_idx - data['n_agents']/2 + 0.5) * width
            ax.bar(x + offset, action_counts[agent_idx], width,
                  color=self.agent_colors[agent_idx],
                  alpha=0.8,
                  label=f'Agent {agent_idx+1}')
        
        ax.set_xlabel("Actions")
        ax.set_ylabel("Count")
        ax.set_xticks(x)
        ax.set_xticklabels(action_names, rotation=45)
        ax.legend()
    
    def plot_cumulative_rewards(self, data, ax):
        """Plot cumulative rewards for all agents."""
        ax.set_title("Cumulative Rewards", fontweight='bold')
        
        for episode_idx, episode in enumerate(data['episodes']):
            for agent_idx in range(data['n_agents']):
                rewards = [step['rewards'][agent_idx] for step in episode['data']]
                cumulative_rewards = np.cumsum(rewards)
                steps = range(len(rewards))
                
                # Offset steps for different episodes
                offset_steps = [s + episode_idx * 100 for s in steps]
                
                ax.plot(offset_steps, cumulative_rewards,
                       color=self.agent_colors[agent_idx],
                       alpha=0.7,
                       label=f'Agent {agent_idx+1}' if episode_idx == 0 else "")
        
        ax.set_xlabel("Step (offset by episode)")
        ax.set_ylabel("Cumulative Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_lane_usage_heatmap(self, data, ax):
        """Plot lane usage heatmap."""
        ax.set_title("Lane Usage Heatmap", fontweight='bold')
        
        # Create a grid for lane usage
        lane_usage = np.zeros((data['n_agents'], 4))  # 4 lanes
        
        for episode in data['episodes']:
            for step in episode['data']:
                for agent_idx in range(data['n_agents']):
                    if f'agent_{agent_idx}_y' in step:
                        y_pos = step[f'agent_{agent_idx}_y']
                        # Map y position to lane (assuming 4 lanes)
                        lane = int(np.clip(y_pos * 4, 0, 3))
                        lane_usage[agent_idx][lane] += 1
        
        # Normalize by total steps
        lane_usage = lane_usage / np.sum(lane_usage, axis=1, keepdims=True)
        
        im = ax.imshow(lane_usage, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(4))
        ax.set_xticklabels(['Lane 1', 'Lane 2', 'Lane 3', 'Lane 4'])
        ax.set_yticks(range(data['n_agents']))
        ax.set_yticklabels([f'Agent {i+1}' for i in range(data['n_agents'])])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add text annotations
        for i in range(data['n_agents']):
            for j in range(4):
                text = ax.text(j, i, f'{lane_usage[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
    
    def plot_episode_summary(self, data, ax):
        """Plot episode summary statistics."""
        ax.set_title("Episode Summary", fontweight='bold')
        
        episodes = []
        total_rewards = []
        steps = []
        
        for episode in data['episodes']:
            episodes.append(episode['episode'])
            total_rewards.append(sum(episode['total_rewards']))
            steps.append(episode['steps'])
        
        ax2 = ax.twinx()
        
        bars1 = ax.bar([e - 0.2 for e in episodes], total_rewards, 0.4, 
                      color='skyblue', alpha=0.7, label='Total Reward')
        bars2 = ax2.bar([e + 0.2 for e in episodes], steps, 0.4, 
                       color='lightcoral', alpha=0.7, label='Steps')
        
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward", color='skyblue')
        ax2.set_ylabel("Steps", color='lightcoral')
        
        # Add value labels on bars
        for bar, value in zip(bars1, total_rewards):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        for bar, value in zip(bars2, steps):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value}', ha='center', va='bottom', fontsize=8)
    
    def plot_agent_performance_comparison(self, data, ax):
        """Plot agent performance comparison."""
        ax.set_title("Agent Performance Comparison", fontweight='bold')
        
        agent_rewards = []
        for agent_idx in range(data['n_agents']):
            total_reward = sum(episode['total_rewards'][agent_idx] for episode in data['episodes'])
            agent_rewards.append(total_reward)
        
        bars = ax.bar(range(data['n_agents']), agent_rewards, 
                     color=self.agent_colors[:data['n_agents']], alpha=0.8)
        
        ax.set_xlabel("Agent")
        ax.set_ylabel("Total Reward")
        ax.set_xticks(range(data['n_agents']))
        ax.set_xticklabels([f'{self.agent_names[i]}\n(Agent {i+1})' 
                           for i in range(data['n_agents'])])
        
        # Add value labels on bars
        for bar, value in zip(bars, agent_rewards):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    def plot_reward_distribution(self, data, ax):
        """Plot reward distribution for all agents."""
        ax.set_title("Reward Distribution", fontweight='bold')
        
        all_rewards = {agent_idx: [] for agent_idx in range(data['n_agents'])}
        
        for episode in data['episodes']:
            for step in episode['data']:
                for agent_idx, reward in enumerate(step['rewards']):
                    all_rewards[agent_idx].append(reward)
        
        # Create violin plot
        reward_data = [all_rewards[i] for i in range(data['n_agents'])]
        parts = ax.violinplot(reward_data, positions=range(data['n_agents']), 
                             showmeans=True, showmedians=True)
        
        # Color the violin plots
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(self.agent_colors[i])
            pc.set_alpha(0.7)
        
        ax.set_xlabel("Agent")
        ax.set_ylabel("Reward")
        ax.set_xticks(range(data['n_agents']))
        ax.set_xticklabels([f'Agent {i+1}' for i in range(data['n_agents'])])
        ax.grid(True, alpha=0.3)
    
    def create_individual_plots(self, data, save_plots=True):
        """Create individual detailed plots."""
        print("🎨 Creating individual detailed plots...")
        
        # 1. Detailed trajectory plot
        self.plot_detailed_trajectories(data, save_plots)
        
        # 2. Detailed reward analysis
        self.plot_detailed_rewards(data, save_plots)
        
        # 3. Agent behavior analysis
        self.plot_agent_behavior_analysis(data, save_plots)
    
    def plot_detailed_trajectories(self, data, save_plots=True):
        """Create detailed trajectory visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detailed Agent Trajectories', fontsize=16, fontweight='bold')
        
        for episode_idx, episode in enumerate(data['episodes']):
            if episode_idx >= 4:  # Only plot first 4 episodes
                break
                
            ax = axes[episode_idx // 2, episode_idx % 2]
            ax.set_title(f'Episode {episode["episode"]} - {episode["steps"]} steps')
            
            # Draw lane boundaries
            for lane in range(4):
                ax.axhline(y=lane * 0.25, color='gray', linestyle='--', alpha=0.3)
            
            for agent_idx in range(data['n_agents']):
                x_positions = []
                y_positions = []
                
                for step in episode['data']:
                    if f'agent_{agent_idx}_x' in step and f'agent_{agent_idx}_y' in step:
                        x_positions.append(step[f'agent_{agent_idx}_x'])
                        y_positions.append(step[f'agent_{agent_idx}_y'])
                
                if x_positions and y_positions:
                    ax.plot(x_positions, y_positions, 
                           color=self.agent_colors[agent_idx],
                           linewidth=3, alpha=0.8,
                           label=f'{self.agent_names[agent_idx]}')
                    
                    # Mark start point
                    ax.scatter(x_positions[0], y_positions[0], 
                             color=self.agent_colors[agent_idx], 
                             marker='o', s=100, alpha=1.0, edgecolor='black')
                    
                    # Mark end point
                    ax.scatter(x_positions[-1], y_positions[-1], 
                             color=self.agent_colors[agent_idx], 
                             marker='s', s=100, alpha=1.0, edgecolor='black')
            
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position (Lane)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detailed_trajectories_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Detailed trajectories saved as: {filename}")
        
        plt.show()
    
    def plot_detailed_rewards(self, data, save_plots=True):
        """Create detailed reward analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detailed Reward Analysis', fontsize=16, fontweight='bold')
        
        # 1. Rewards over time with moving average
        ax1 = axes[0, 0]
        ax1.set_title("Rewards with Moving Average")
        
        for agent_idx in range(data['n_agents']):
            all_rewards = []
            all_steps = []
            step_offset = 0
            
            for episode in data['episodes']:
                episode_rewards = [step['rewards'][agent_idx] for step in episode['data']]
                episode_steps = [step['step'] + step_offset for step in episode['data']]
                
                all_rewards.extend(episode_rewards)
                all_steps.extend(episode_steps)
                step_offset += len(episode['data']) + 10  # Add gap between episodes
            
            # Plot raw rewards
            ax1.plot(all_steps, all_rewards, color=self.agent_colors[agent_idx], 
                    alpha=0.3, linewidth=1)
            
            # Plot moving average
            if len(all_rewards) > 5:
                window_size = min(5, len(all_rewards))
                moving_avg = pd.Series(all_rewards).rolling(window=window_size).mean()
                ax1.plot(all_steps, moving_avg, color=self.agent_colors[agent_idx], 
                        linewidth=2, label=f'{self.agent_names[agent_idx]}')
        
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Reward")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Reward statistics by agent
        ax2 = axes[0, 1]
        ax2.set_title("Reward Statistics by Agent")
        
        agent_stats = []
        for agent_idx in range(data['n_agents']):
            all_rewards = []
            for episode in data['episodes']:
                all_rewards.extend([step['rewards'][agent_idx] for step in episode['data']])
            
            stats = {
                'mean': np.mean(all_rewards),
                'std': np.std(all_rewards),
                'min': np.min(all_rewards),
                'max': np.max(all_rewards)
            }
            agent_stats.append(stats)
        
        x = range(data['n_agents'])
        means = [stats['mean'] for stats in agent_stats]
        stds = [stats['std'] for stats in agent_stats]
        
        bars = ax2.bar(x, means, yerr=stds, capsize=5, 
                      color=self.agent_colors[:data['n_agents']], alpha=0.8)
        ax2.set_xlabel("Agent")
        ax2.set_ylabel("Mean Reward ± Std")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Agent {i+1}' for i in range(data['n_agents'])])
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Episode-wise performance
        ax3 = axes[1, 0]
        ax3.set_title("Episode-wise Performance")
        
        episodes = [ep['episode'] for ep in data['episodes']]
        for agent_idx in range(data['n_agents']):
            episode_rewards = [ep['total_rewards'][agent_idx] for ep in data['episodes']]
            ax3.plot(episodes, episode_rewards, 'o-', 
                    color=self.agent_colors[agent_idx], linewidth=2, markersize=8,
                    label=f'{self.agent_names[agent_idx]}')
        
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Total Episode Reward")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Reward correlation matrix
        ax4 = axes[1, 1]
        ax4.set_title("Agent Reward Correlation")
        
        # Collect all rewards for correlation analysis
        reward_matrix = []
        for episode in data['episodes']:
            for step in episode['data']:
                reward_matrix.append(step['rewards'])
        
        reward_df = pd.DataFrame(reward_matrix, 
                               columns=[f'Agent {i+1}' for i in range(data['n_agents'])])
        correlation_matrix = reward_df.corr()
        
        im = ax4.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax4.set_xticks(range(data['n_agents']))
        ax4.set_yticks(range(data['n_agents']))
        ax4.set_xticklabels([f'Agent {i+1}' for i in range(data['n_agents'])])
        ax4.set_yticklabels([f'Agent {i+1}' for i in range(data['n_agents'])])
        
        # Add correlation values
        for i in range(data['n_agents']):
            for j in range(data['n_agents']):
                text = ax4.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detailed_rewards_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Detailed rewards saved as: {filename}")
        
        plt.show()
    
    def plot_agent_behavior_analysis(self, data, save_plots=True):
        """Create agent behavior analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Agent Behavior Analysis', fontsize=16, fontweight='bold')
        
        # 1. Action frequency by agent
        ax1 = axes[0, 0]
        ax1.set_title("Action Frequency by Agent")
        
        action_names = ['SLOWER', 'IDLE', 'FASTER', 'LANE_LEFT', 'LANE_RIGHT']
        action_data = []
        
        for agent_idx in range(data['n_agents']):
            agent_actions = [0] * 5
            total_actions = 0
            
            for episode in data['episodes']:
                for step in episode['data']:
                    action = step['actions'][agent_idx]
                    if action < 5:
                        agent_actions[action] += 1
                        total_actions += 1
            
            # Convert to percentages
            if total_actions > 0:
                agent_actions = [count / total_actions * 100 for count in agent_actions]
            action_data.append(agent_actions)
        
        # Create stacked bar chart
        bottom = np.zeros(data['n_agents'])
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        
        for action_idx, action_name in enumerate(action_names):
            values = [action_data[agent_idx][action_idx] for agent_idx in range(data['n_agents'])]
            ax1.bar(range(data['n_agents']), values, bottom=bottom, 
                   label=action_name, color=colors[action_idx], alpha=0.8)
            bottom += values
        
        ax1.set_xlabel("Agent")
        ax1.set_ylabel("Action Frequency (%)")
        ax1.set_xticks(range(data['n_agents']))
        ax1.set_xticklabels([f'{self.agent_names[i]}' for i in range(data['n_agents'])])
        ax1.legend()
        
        # 2. Speed distribution by agent
        ax2 = axes[0, 1]
        ax2.set_title("Speed Distribution by Agent")
        
        speed_data = []
        for agent_idx in range(data['n_agents']):
            agent_speeds = []
            for episode in data['episodes']:
                for step in episode['data']:
                    if f'agent_{agent_idx}_speed' in step:
                        agent_speeds.append(step[f'agent_{agent_idx}_speed'])
            speed_data.append(agent_speeds)
        
        # Create box plot
        box_plot = ax2.boxplot(speed_data, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], self.agent_colors[:data['n_agents']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel("Agent")
        ax2.set_ylabel("Speed")
        ax2.set_xticklabels([f'{self.agent_names[i]}' for i in range(data['n_agents'])])
        
        # 3. Lane change frequency
        ax3 = axes[1, 0]
        ax3.set_title("Lane Change Frequency")
        
        lane_changes = []
        for agent_idx in range(data['n_agents']):
            changes = 0
            total_steps = 0
            
            for episode in data['episodes']:
                for step in episode['data']:
                    action = step['actions'][agent_idx]
                    if action in [3, 4]:  # LANE_LEFT or LANE_RIGHT
                        changes += 1
                    total_steps += 1
            
            frequency = (changes / total_steps * 100) if total_steps > 0 else 0
            lane_changes.append(frequency)
        
        bars = ax3.bar(range(data['n_agents']), lane_changes, 
                      color=self.agent_colors[:data['n_agents']], alpha=0.8)
        ax3.set_xlabel("Agent")
        ax3.set_ylabel("Lane Change Frequency (%)")
        ax3.set_xticks(range(data['n_agents']))
        ax3.set_xticklabels([f'{self.agent_names[i]}' for i in range(data['n_agents'])])
        
        # Add value labels
        for bar, value in zip(bars, lane_changes):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Agent efficiency (reward per step)
        ax4 = axes[1, 1]
        ax4.set_title("Agent Efficiency (Reward per Step)")
        
        efficiency = []
        for agent_idx in range(data['n_agents']):
            total_reward = 0
            total_steps = 0
            
            for episode in data['episodes']:
                total_reward += episode['total_rewards'][agent_idx]
                total_steps += episode['steps']
            
            eff = total_reward / total_steps if total_steps > 0 else 0
            efficiency.append(eff)
        
        bars = ax4.bar(range(data['n_agents']), efficiency, 
                      color=self.agent_colors[:data['n_agents']], alpha=0.8)
        ax4.set_xlabel("Agent")
        ax4.set_ylabel("Reward per Step")
        ax4.set_xticks(range(data['n_agents']))
        ax4.set_xticklabels([f'{self.agent_names[i]}' for i in range(data['n_agents'])])
        
        # Add value labels
        for bar, value in zip(bars, efficiency):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"behavior_analysis_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Behavior analysis saved as: {filename}")
        
        plt.show()
    
    def generate_summary_report(self, data):
        """Generate a text summary report."""
        print("\n" + "="*60)
        print("📊 MULTI-AGENT SIMULATION SUMMARY REPORT")
        print("="*60)
        
        print(f"🎯 Configuration:")
        print(f"   Scenario: {data.get('scenario', 'Unknown')}")
        print(f"   Number of Agents: {data.get('n_agents', 4)}")
        print(f"   Total Episodes: {len(data.get('episodes', []))}")
        
        total_steps = sum(ep['steps'] for ep in data['episodes'])
        print(f"   Total Steps: {total_steps}")
        
        print(f"\n🏆 Agent Performance:")
        for agent_idx in range(data['n_agents']):
            total_reward = sum(ep['total_rewards'][agent_idx] for ep in data['episodes'])
            avg_reward = total_reward / len(data['episodes']) if data['episodes'] else 0
            
            print(f"   {self.agent_names[agent_idx]} (Agent {agent_idx+1}):")
            print(f"     Total Reward: {total_reward:.3f}")
            print(f"     Average Episode Reward: {avg_reward:.3f}")
        
        print(f"\n📈 Episode Statistics:")
        for episode in data['episodes']:
            print(f"   Episode {episode['episode']}: {episode['steps']} steps, "
                  f"Total Reward: {sum(episode['total_rewards']):.2f}")
        
        print("="*60)


def main():
    """Main function to run the visualization."""
    visualizer = MultiAgentDataVisualizer()
    
    try:
        # Load the most recent data
        data = visualizer.load_data()
        
        # Create comprehensive dashboard
        visualizer.create_comprehensive_dashboard(data, save_plots=True)
        
        # Create individual detailed plots
        visualizer.create_individual_plots(data, save_plots=True)
        
        # Generate summary report
        visualizer.generate_summary_report(data)
        
        print("\n✅ Visualization complete! Check the generated PNG files.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have multi-agent data files in the 'data' directory.")


if __name__ == "__main__":
    main()