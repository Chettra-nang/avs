#!/usr/bin/env python3
"""
Green and Blue Agent Analysis

This script focuses specifically on analyzing the green and blue agents
from the multi-agent highway simulation data.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import glob
from datetime import datetime

class GreenBlueAnalyzer:
    """Analyze specifically the green and blue agents."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        # Focus on agents 1 (Green) and 2 (Blue) - 0-indexed
        self.green_agent_idx = 1  # Agent 2 (Aggressive - Green)
        self.blue_agent_idx = 2   # Agent 3 (Speed Keeper - Blue)
        
        self.colors = {
            'green': '#00FF00',
            'blue': '#0000FF'
        }
        
        self.agent_names = {
            self.green_agent_idx: 'Green Agent (Aggressive)',
            self.blue_agent_idx: 'Blue Agent (Speed Keeper)'
        }
    
    def load_data(self, filename=None):
        """Load multi-agent data from JSON file."""
        if filename is None:
            pattern = str(self.data_dir / "multiagent_demo_*.json")
            files = glob.glob(pattern)
            if not files:
                raise FileNotFoundError("No multi-agent data files found!")
            filename = max(files, key=lambda x: Path(x).stat().st_mtime)
        
        print(f"📊 Loading data from: {filename}")
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return data
    
    def create_green_blue_dashboard(self, data, save_plots=True):
        """Create a dashboard focusing on green and blue agents."""
        print("🎨 Creating Green vs Blue Agent Dashboard...")
        
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Green vs Blue Agent Analysis Dashboard\n'
                    f'Green: Aggressive Driver | Blue: Speed Keeper', 
                    fontsize=16, fontweight='bold')
        
        # 1. Trajectory Comparison (top left)
        ax1 = plt.subplot(3, 3, 1)
        self.plot_trajectory_comparison(data, ax1)
        
        # 2. Reward Comparison (top center)
        ax2 = plt.subplot(3, 3, 2)
        self.plot_reward_comparison(data, ax2)
        
        # 3. Speed Comparison (top right)
        ax3 = plt.subplot(3, 3, 3)
        self.plot_speed_comparison(data, ax3)
        
        # 4. Action Comparison (middle left)
        ax4 = plt.subplot(3, 3, 4)
        self.plot_action_comparison(data, ax4)
        
        # 5. Performance Over Episodes (middle center)
        ax5 = plt.subplot(3, 3, 5)
        self.plot_episode_performance(data, ax5)
        
        # 6. Lane Usage Comparison (middle right)
        ax6 = plt.subplot(3, 3, 6)
        self.plot_lane_usage_comparison(data, ax6)
        
        # 7. Cumulative Rewards (bottom left)
        ax7 = plt.subplot(3, 3, 7)
        self.plot_cumulative_rewards(data, ax7)
        
        # 8. Speed vs Reward Scatter (bottom center)
        ax8 = plt.subplot(3, 3, 8)
        self.plot_speed_reward_scatter(data, ax8)
        
        # 9. Head-to-Head Stats (bottom right)
        ax9 = plt.subplot(3, 3, 9)
        self.plot_head_to_head_stats(data, ax9)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"green_blue_dashboard_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Green vs Blue dashboard saved as: {filename}")
        
        plt.show()
    
    def plot_trajectory_comparison(self, data, ax):
        """Plot trajectory comparison between green and blue agents."""
        ax.set_title("Trajectory Comparison", fontweight='bold')
        
        # Draw lane boundaries
        for lane in range(4):
            ax.axhline(y=lane * 0.25, color='gray', linestyle='--', alpha=0.3)
        
        for episode_idx, episode in enumerate(data['episodes']):
            for agent_idx in [self.green_agent_idx, self.blue_agent_idx]:
                x_positions = []
                y_positions = []
                
                for step in episode['data']:
                    if f'agent_{agent_idx}_x' in step and f'agent_{agent_idx}_y' in step:
                        x_positions.append(step[f'agent_{agent_idx}_x'])
                        y_positions.append(step[f'agent_{agent_idx}_y'])
                
                if x_positions and y_positions:
                    color = self.colors['green'] if agent_idx == self.green_agent_idx else self.colors['blue']
                    label = self.agent_names[agent_idx] if episode_idx == 0 else ""
                    
                    ax.plot(x_positions, y_positions, 
                           color=color, linewidth=2, alpha=0.7, label=label)
                    
                    # Mark start and end
                    ax.scatter(x_positions[0], y_positions[0], 
                             color=color, marker='o', s=60, alpha=0.9)
                    ax.scatter(x_positions[-1], y_positions[-1], 
                             color=color, marker='s', s=60, alpha=0.9)
        
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position (Lane)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
    
    def plot_reward_comparison(self, data, ax):
        """Plot reward comparison over time."""
        ax.set_title("Reward Comparison Over Time", fontweight='bold')
        
        for episode_idx, episode in enumerate(data['episodes']):
            steps = [step['step'] for step in episode['data']]
            
            for agent_idx in [self.green_agent_idx, self.blue_agent_idx]:
                rewards = [step['rewards'][agent_idx] for step in episode['data']]
                color = self.colors['green'] if agent_idx == self.green_agent_idx else self.colors['blue']
                label = self.agent_names[agent_idx] if episode_idx == 0 else ""
                
                # Offset steps for different episodes
                offset_steps = [s + episode_idx * 50 for s in steps]
                
                ax.plot(offset_steps, rewards, color=color, alpha=0.8, 
                       linewidth=2, marker='o', markersize=4, label=label)
        
        ax.set_xlabel("Step (offset by episode)")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_speed_comparison(self, data, ax):
        """Plot speed comparison over time."""
        ax.set_title("Speed Comparison Over Time", fontweight='bold')
        
        for episode_idx, episode in enumerate(data['episodes']):
            steps = [step['step'] for step in episode['data']]
            
            for agent_idx in [self.green_agent_idx, self.blue_agent_idx]:
                speeds = []
                for step in episode['data']:
                    if f'agent_{agent_idx}_speed' in step:
                        speeds.append(step[f'agent_{agent_idx}_speed'])
                
                if speeds:
                    color = self.colors['green'] if agent_idx == self.green_agent_idx else self.colors['blue']
                    label = self.agent_names[agent_idx] if episode_idx == 0 else ""
                    
                    # Offset steps for different episodes
                    offset_steps = [s + episode_idx * 50 for s in steps]
                    
                    ax.plot(offset_steps, speeds, color=color, alpha=0.8, 
                           linewidth=2, marker='s', markersize=4, label=label)
        
        ax.set_xlabel("Step (offset by episode)")
        ax.set_ylabel("Speed")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_action_comparison(self, data, ax):
        """Plot action distribution comparison."""
        ax.set_title("Action Distribution Comparison", fontweight='bold')
        
        action_names = ['SLOWER', 'IDLE', 'FASTER', 'LANE_LEFT', 'LANE_RIGHT']
        action_counts = {self.green_agent_idx: [0] * 5, self.blue_agent_idx: [0] * 5}
        
        for episode in data['episodes']:
            for step in episode['data']:
                for agent_idx in [self.green_agent_idx, self.blue_agent_idx]:
                    action = step['actions'][agent_idx]
                    if action < 5:
                        action_counts[agent_idx][action] += 1
        
        # Convert to percentages
        for agent_idx in [self.green_agent_idx, self.blue_agent_idx]:
            total = sum(action_counts[agent_idx])
            if total > 0:
                action_counts[agent_idx] = [count / total * 100 for count in action_counts[agent_idx]]
        
        x = np.arange(len(action_names))
        width = 0.35
        
        green_counts = action_counts[self.green_agent_idx]
        blue_counts = action_counts[self.blue_agent_idx]
        
        ax.bar(x - width/2, green_counts, width, color=self.colors['green'], 
               alpha=0.8, label='Green Agent')
        ax.bar(x + width/2, blue_counts, width, color=self.colors['blue'], 
               alpha=0.8, label='Blue Agent')
        
        ax.set_xlabel("Actions")
        ax.set_ylabel("Frequency (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(action_names, rotation=45)
        ax.legend()
    
    def plot_episode_performance(self, data, ax):
        """Plot performance over episodes."""
        ax.set_title("Performance Over Episodes", fontweight='bold')
        
        episodes = [ep['episode'] for ep in data['episodes']]
        green_rewards = [ep['total_rewards'][self.green_agent_idx] for ep in data['episodes']]
        blue_rewards = [ep['total_rewards'][self.blue_agent_idx] for ep in data['episodes']]
        
        ax.plot(episodes, green_rewards, 'o-', color=self.colors['green'], 
               linewidth=3, markersize=8, label='Green Agent')
        ax.plot(episodes, blue_rewards, 's-', color=self.colors['blue'], 
               linewidth=3, markersize=8, label='Blue Agent')
        
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Episode Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (ep, green_r, blue_r) in enumerate(zip(episodes, green_rewards, blue_rewards)):
            ax.text(ep, green_r + 0.1, f'{green_r:.2f}', ha='center', va='bottom', 
                   color=self.colors['green'], fontweight='bold')
            ax.text(ep, blue_r - 0.1, f'{blue_r:.2f}', ha='center', va='top', 
                   color=self.colors['blue'], fontweight='bold')
    
    def plot_lane_usage_comparison(self, data, ax):
        """Plot lane usage comparison."""
        ax.set_title("Lane Usage Comparison", fontweight='bold')
        
        lane_usage = {self.green_agent_idx: [0] * 4, self.blue_agent_idx: [0] * 4}
        
        for episode in data['episodes']:
            for step in episode['data']:
                for agent_idx in [self.green_agent_idx, self.blue_agent_idx]:
                    if f'agent_{agent_idx}_y' in step:
                        y_pos = step[f'agent_{agent_idx}_y']
                        lane = int(np.clip(y_pos * 4, 0, 3))
                        lane_usage[agent_idx][lane] += 1
        
        # Normalize
        for agent_idx in [self.green_agent_idx, self.blue_agent_idx]:
            total = sum(lane_usage[agent_idx])
            if total > 0:
                lane_usage[agent_idx] = [count / total * 100 for count in lane_usage[agent_idx]]
        
        x = np.arange(4)
        width = 0.35
        
        green_usage = lane_usage[self.green_agent_idx]
        blue_usage = lane_usage[self.blue_agent_idx]
        
        ax.bar(x - width/2, green_usage, width, color=self.colors['green'], 
               alpha=0.8, label='Green Agent')
        ax.bar(x + width/2, blue_usage, width, color=self.colors['blue'], 
               alpha=0.8, label='Blue Agent')
        
        ax.set_xlabel("Lane")
        ax.set_ylabel("Usage (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(['Lane 1', 'Lane 2', 'Lane 3', 'Lane 4'])
        ax.legend()
        
        # Add value labels
        for i, (green_val, blue_val) in enumerate(zip(green_usage, blue_usage)):
            ax.text(i - width/2, green_val + 1, f'{green_val:.1f}%', 
                   ha='center', va='bottom', fontsize=8)
            ax.text(i + width/2, blue_val + 1, f'{blue_val:.1f}%', 
                   ha='center', va='bottom', fontsize=8)
    
    def plot_cumulative_rewards(self, data, ax):
        """Plot cumulative rewards comparison."""
        ax.set_title("Cumulative Rewards", fontweight='bold')
        
        for episode_idx, episode in enumerate(data['episodes']):
            for agent_idx in [self.green_agent_idx, self.blue_agent_idx]:
                rewards = [step['rewards'][agent_idx] for step in episode['data']]
                cumulative_rewards = np.cumsum(rewards)
                steps = range(len(rewards))
                
                color = self.colors['green'] if agent_idx == self.green_agent_idx else self.colors['blue']
                label = self.agent_names[agent_idx] if episode_idx == 0 else ""
                
                # Offset steps for different episodes
                offset_steps = [s + episode_idx * 50 for s in steps]
                
                ax.plot(offset_steps, cumulative_rewards, color=color, 
                       linewidth=2, alpha=0.8, label=label)
        
        ax.set_xlabel("Step (offset by episode)")
        ax.set_ylabel("Cumulative Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_speed_reward_scatter(self, data, ax):
        """Plot speed vs reward scatter plot."""
        ax.set_title("Speed vs Reward Relationship", fontweight='bold')
        
        for agent_idx in [self.green_agent_idx, self.blue_agent_idx]:
            speeds = []
            rewards = []
            
            for episode in data['episodes']:
                for step in episode['data']:
                    if f'agent_{agent_idx}_speed' in step:
                        speeds.append(step[f'agent_{agent_idx}_speed'])
                        rewards.append(step['rewards'][agent_idx])
            
            color = self.colors['green'] if agent_idx == self.green_agent_idx else self.colors['blue']
            label = self.agent_names[agent_idx]
            
            ax.scatter(speeds, rewards, color=color, alpha=0.6, s=50, label=label)
        
        ax.set_xlabel("Speed")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_head_to_head_stats(self, data, ax):
        """Plot head-to-head statistics."""
        ax.set_title("Head-to-Head Statistics", fontweight='bold')
        
        # Calculate various statistics
        stats = {}
        
        for agent_idx in [self.green_agent_idx, self.blue_agent_idx]:
            agent_name = 'Green' if agent_idx == self.green_agent_idx else 'Blue'
            
            # Total reward
            total_reward = sum(ep['total_rewards'][agent_idx] for ep in data['episodes'])
            
            # Average speed
            all_speeds = []
            for episode in data['episodes']:
                for step in episode['data']:
                    if f'agent_{agent_idx}_speed' in step:
                        all_speeds.append(step[f'agent_{agent_idx}_speed'])
            avg_speed = np.mean(all_speeds) if all_speeds else 0
            
            # Lane changes
            lane_changes = 0
            total_actions = 0
            for episode in data['episodes']:
                for step in episode['data']:
                    action = step['actions'][agent_idx]
                    if action in [3, 4]:  # LANE_LEFT or LANE_RIGHT
                        lane_changes += 1
                    total_actions += 1
            
            lane_change_rate = (lane_changes / total_actions * 100) if total_actions > 0 else 0
            
            stats[agent_name] = {
                'Total Reward': total_reward,
                'Avg Speed': avg_speed,
                'Lane Change Rate (%)': lane_change_rate
            }
        
        # Create comparison bars
        categories = list(stats['Green'].keys())
        green_values = list(stats['Green'].values())
        blue_values = list(stats['Blue'].values())
        
        x = np.arange(len(categories))
        width = 0.35
        
        # Normalize values for better comparison
        normalized_green = []
        normalized_blue = []
        
        for i, category in enumerate(categories):
            max_val = max(green_values[i], blue_values[i])
            if max_val != 0:
                normalized_green.append(green_values[i] / max_val)
                normalized_blue.append(blue_values[i] / max_val)
            else:
                normalized_green.append(0)
                normalized_blue.append(0)
        
        ax.bar(x - width/2, normalized_green, width, color=self.colors['green'], 
               alpha=0.8, label='Green Agent')
        ax.bar(x + width/2, normalized_blue, width, color=self.colors['blue'], 
               alpha=0.8, label='Blue Agent')
        
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Normalized Values")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45)
        ax.legend()
        
        # Add actual values as text
        for i, (green_val, blue_val) in enumerate(zip(green_values, blue_values)):
            ax.text(i - width/2, normalized_green[i] + 0.05, f'{green_val:.2f}', 
                   ha='center', va='bottom', fontsize=8, rotation=90)
            ax.text(i + width/2, normalized_blue[i] + 0.05, f'{blue_val:.2f}', 
                   ha='center', va='bottom', fontsize=8, rotation=90)
    
    def generate_comparison_report(self, data):
        """Generate a comparison report between green and blue agents."""
        print("\n" + "="*60)
        print("🟢🔵 GREEN vs BLUE AGENT COMPARISON REPORT")
        print("="*60)
        
        green_total = sum(ep['total_rewards'][self.green_agent_idx] for ep in data['episodes'])
        blue_total = sum(ep['total_rewards'][self.blue_agent_idx] for ep in data['episodes'])
        
        print(f"🏆 Overall Performance:")
        print(f"   Green Agent (Aggressive): {green_total:.3f} total reward")
        print(f"   Blue Agent (Speed Keeper): {blue_total:.3f} total reward")
        
        if green_total > blue_total:
            print(f"   🟢 Green Agent wins by {green_total - blue_total:.3f} points!")
        elif blue_total > green_total:
            print(f"   🔵 Blue Agent wins by {blue_total - green_total:.3f} points!")
        else:
            print(f"   🤝 It's a tie!")
        
        print(f"\n📊 Episode-by-Episode:")
        green_wins = 0
        blue_wins = 0
        
        for episode in data['episodes']:
            green_reward = episode['total_rewards'][self.green_agent_idx]
            blue_reward = episode['total_rewards'][self.blue_agent_idx]
            
            if green_reward > blue_reward:
                winner = "🟢 Green"
                green_wins += 1
            elif blue_reward > green_reward:
                winner = "🔵 Blue"
                blue_wins += 1
            else:
                winner = "🤝 Tie"
            
            print(f"   Episode {episode['episode']}: Green={green_reward:.2f}, Blue={blue_reward:.2f} - {winner}")
        
        print(f"\n🏅 Episode Wins:")
        print(f"   Green Agent: {green_wins} episodes")
        print(f"   Blue Agent: {blue_wins} episodes")
        
        # Behavioral analysis
        print(f"\n🎭 Behavioral Analysis:")
        
        for agent_idx, agent_name in [(self.green_agent_idx, "Green"), (self.blue_agent_idx, "Blue")]:
            lane_changes = 0
            total_actions = 0
            all_speeds = []
            
            for episode in data['episodes']:
                for step in episode['data']:
                    action = step['actions'][agent_idx]
                    if action in [3, 4]:
                        lane_changes += 1
                    total_actions += 1
                    
                    if f'agent_{agent_idx}_speed' in step:
                        all_speeds.append(step[f'agent_{agent_idx}_speed'])
            
            lane_change_rate = (lane_changes / total_actions * 100) if total_actions > 0 else 0
            avg_speed = np.mean(all_speeds) if all_speeds else 0
            
            print(f"   {agent_name} Agent:")
            print(f"     Lane Change Rate: {lane_change_rate:.1f}%")
            print(f"     Average Speed: {avg_speed:.3f}")
        
        print("="*60)


def main():
    """Main function to run the green vs blue analysis."""
    analyzer = GreenBlueAnalyzer()
    
    try:
        # Load the most recent data
        data = analyzer.load_data()
        
        # Create green vs blue dashboard
        analyzer.create_green_blue_dashboard(data, save_plots=True)
        
        # Generate comparison report
        analyzer.generate_comparison_report(data)
        
        print("\n✅ Green vs Blue analysis complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have multi-agent data files in the 'data' directory.")


if __name__ == "__main__":
    main()