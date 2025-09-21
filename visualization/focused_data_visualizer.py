#!/usr/bin/env python3
"""
Focused Data Visualizer

This script creates focused, high-quality visualizations for all data types
in your data folder with proper handling of each data structure.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FocusedDataVisualizer:
    """Create focused visualizations for all data types."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        
        # Set up beautiful plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'agents': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        }
        
        # Set global font sizes
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9
        })
    
    def visualize_all_data(self):
        """Main function to visualize all data."""
        print("🎨 Starting focused data visualization...")
        
        # 1. Multi-agent comparison
        self.visualize_multiagent_data()
        
        # 2. Single simulation analysis
        self.visualize_simulation_data()
        
        # 3. Dataset overview
        self.visualize_dataset_overview()
        
        # 4. Create summary dashboard
        self.create_summary_dashboard()
        
        print("✅ All visualizations complete!")
    
    def visualize_multiagent_data(self):
        """Visualize multi-agent data with comprehensive analysis."""
        print("🤖 Analyzing multi-agent data...")
        
        # Load all valid multi-agent files
        pattern = str(self.data_dir / "multiagent_demo_*.json")
        files = glob.glob(pattern)
        
        valid_data = []
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['filename'] = Path(file_path).stem
                    data['timestamp'] = Path(file_path).stem.split('_')[-1]
                    valid_data.append(data)
            except Exception as e:
                print(f"   ⚠️ Skipping {Path(file_path).name}: {e}")
        
        if not valid_data:
            print("   ❌ No valid multi-agent data found")
            return
        
        print(f"   📊 Loaded {len(valid_data)} valid multi-agent runs")
        
        # Create comprehensive multi-agent analysis
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Multi-Agent Highway Simulation - Comprehensive Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Performance comparison across runs (top row)
        ax1 = plt.subplot(4, 4, 1)
        self.plot_multiagent_performance_comparison(valid_data, ax1)
        
        ax2 = plt.subplot(4, 4, 2)
        self.plot_multiagent_consistency(valid_data, ax2)
        
        ax3 = plt.subplot(4, 4, 3)
        self.plot_multiagent_success_rates(valid_data, ax3)
        
        ax4 = plt.subplot(4, 4, 4)
        self.plot_multiagent_episode_lengths(valid_data, ax4)
        
        # 2. Agent behavior analysis (second row)
        ax5 = plt.subplot(4, 4, 5)
        self.plot_agent_reward_distributions(valid_data, ax5)
        
        ax6 = plt.subplot(4, 4, 6)
        self.plot_agent_performance_trends(valid_data, ax6)
        
        ax7 = plt.subplot(4, 4, 7)
        self.plot_learning_curves(valid_data, ax7)
        
        ax8 = plt.subplot(4, 4, 8)
        self.plot_agent_correlation_matrix(valid_data, ax8)
        
        # 3. Detailed trajectory analysis (third row)
        ax9 = plt.subplot(4, 4, 9)
        self.plot_best_vs_worst_runs(valid_data, ax9)
        
        ax10 = plt.subplot(4, 4, 10)
        self.plot_episode_progression(valid_data, ax10)
        
        ax11 = plt.subplot(4, 4, 11)
        self.plot_reward_volatility(valid_data, ax11)
        
        ax12 = plt.subplot(4, 4, 12)
        self.plot_convergence_analysis(valid_data, ax12)
        
        # 4. Summary statistics (bottom row)
        ax13 = plt.subplot(4, 4, 13)
        self.plot_run_statistics_summary(valid_data, ax13)
        
        ax14 = plt.subplot(4, 4, 14)
        self.plot_agent_ranking(valid_data, ax14)
        
        ax15 = plt.subplot(4, 4, 15)
        self.plot_performance_heatmap(valid_data, ax15)
        
        ax16 = plt.subplot(4, 4, 16)
        self.plot_key_metrics_radar(valid_data, ax16)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multiagent_comprehensive_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   💾 Multi-agent analysis saved as: {filename}")
        plt.show()
    
    def plot_multiagent_performance_comparison(self, data_list, ax):
        """Plot performance comparison across all runs."""
        ax.set_title("Performance Across Runs", fontweight='bold')
        
        run_names = []
        total_rewards = []
        
        for i, data in enumerate(data_list):
            run_names.append(f"Run {i+1}")
            total_reward = sum(sum(ep['total_rewards']) for ep in data['episodes'])
            total_rewards.append(total_reward)
        
        bars = ax.bar(range(len(run_names)), total_rewards, 
                     color=self.colors['primary'], alpha=0.8)
        
        # Highlight best and worst
        if total_rewards:
            best_idx = np.argmax(total_rewards)
            worst_idx = np.argmin(total_rewards)
            bars[best_idx].set_color(self.colors['success'])
            bars[worst_idx].set_color(self.colors['secondary'])
        
        ax.set_xlabel("Run")
        ax.set_ylabel("Total Reward")
        ax.set_xticks(range(len(run_names)))
        ax.set_xticklabels(run_names, rotation=45)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, total_rewards)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_rewards)*0.01,
                   f'{value:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    def plot_multiagent_consistency(self, data_list, ax):
        """Plot consistency of agent performance."""
        ax.set_title("Agent Performance Consistency", fontweight='bold')
        
        agent_performances = [[] for _ in range(4)]
        
        for data in data_list:
            for agent_idx in range(4):
                total_reward = sum(ep['total_rewards'][agent_idx] for ep in data['episodes'])
                agent_performances[agent_idx].append(total_reward)
        
        # Create violin plot
        parts = ax.violinplot(agent_performances, positions=range(4), showmeans=True, showmedians=True)
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(self.colors['agents'][i])
            pc.set_alpha(0.7)
        
        ax.set_xlabel("Agent")
        ax.set_ylabel("Total Reward")
        ax.set_xticks(range(4))
        ax.set_xticklabels(['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4'])
    
    def plot_multiagent_success_rates(self, data_list, ax):
        """Plot success rates (positive reward episodes)."""
        ax.set_title("Success Rates by Run", fontweight='bold')
        
        success_rates = []
        run_names = []
        
        for i, data in enumerate(data_list):
            successful_episodes = sum(1 for ep in data['episodes'] if sum(ep['total_rewards']) > 0)
            total_episodes = len(data['episodes'])
            success_rate = (successful_episodes / total_episodes * 100) if total_episodes > 0 else 0
            
            success_rates.append(success_rate)
            run_names.append(f"Run {i+1}")
        
        bars = ax.bar(range(len(run_names)), success_rates, 
                     color=self.colors['accent'], alpha=0.8)
        
        ax.set_xlabel("Run")
        ax.set_ylabel("Success Rate (%)")
        ax.set_ylim(0, 100)
        ax.set_xticks(range(len(run_names)))
        ax.set_xticklabels(run_names, rotation=45)
        
        # Add percentage labels
        for bar, value in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{value:.0f}%', ha='center', va='bottom', fontsize=8)
    
    def plot_multiagent_episode_lengths(self, data_list, ax):
        """Plot episode length distribution."""
        ax.set_title("Episode Length Distribution", fontweight='bold')
        
        all_lengths = []
        for data in data_list:
            lengths = [ep['steps'] for ep in data['episodes']]
            all_lengths.extend(lengths)
        
        if all_lengths:
            ax.hist(all_lengths, bins=15, color=self.colors['primary'], alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(all_lengths), color=self.colors['secondary'], 
                      linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_lengths):.1f}')
            ax.legend()
        
        ax.set_xlabel("Episode Length (steps)")
        ax.set_ylabel("Frequency")
    
    def plot_agent_reward_distributions(self, data_list, ax):
        """Plot reward distributions for each agent."""
        ax.set_title("Agent Reward Distributions", fontweight='bold')
        
        agent_rewards = [[] for _ in range(4)]
        
        for data in data_list:
            for ep in data['episodes']:
                for agent_idx in range(4):
                    agent_rewards[agent_idx].append(ep['total_rewards'][agent_idx])
        
        # Create box plot
        box_plot = ax.boxplot(agent_rewards, patch_artist=True, labels=['A1', 'A2', 'A3', 'A4'])
        
        for patch, color in zip(box_plot['boxes'], self.colors['agents'][:4]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel("Agent")
        ax.set_ylabel("Episode Reward")
    
    def plot_agent_performance_trends(self, data_list, ax):
        """Plot performance trends over time."""
        ax.set_title("Performance Trends Over Time", fontweight='bold')
        
        # Combine all episodes chronologically
        all_episodes = []
        for data in data_list:
            for ep in data['episodes']:
                all_episodes.append(ep)
        
        if len(all_episodes) > 1:
            episode_nums = range(len(all_episodes))
            
            for agent_idx in range(4):
                rewards = [ep['total_rewards'][agent_idx] for ep in all_episodes]
                # Smooth the line with moving average
                if len(rewards) > 3:
                    smoothed = pd.Series(rewards).rolling(window=3, center=True).mean()
                    ax.plot(episode_nums, smoothed, color=self.colors['agents'][agent_idx], 
                           linewidth=2, label=f'Agent {agent_idx+1}', alpha=0.8)
        
        ax.set_xlabel("Episode (chronological)")
        ax.set_ylabel("Episode Reward")
        ax.legend()
    
    def plot_learning_curves(self, data_list, ax):
        """Plot learning curves showing improvement over time."""
        ax.set_title("Learning Curves", fontweight='bold')
        
        # Calculate cumulative average performance
        all_total_rewards = []
        for data in data_list:
            for ep in data['episodes']:
                all_total_rewards.append(sum(ep['total_rewards']))
        
        if all_total_rewards:
            cumulative_avg = np.cumsum(all_total_rewards) / np.arange(1, len(all_total_rewards) + 1)
            ax.plot(range(len(cumulative_avg)), cumulative_avg, 
                   color=self.colors['primary'], linewidth=2)
            
            # Add trend line
            if len(cumulative_avg) > 1:
                z = np.polyfit(range(len(cumulative_avg)), cumulative_avg, 1)
                p = np.poly1d(z)
                ax.plot(range(len(cumulative_avg)), p(range(len(cumulative_avg))), 
                       "--", color=self.colors['secondary'], alpha=0.8, 
                       label=f'Trend: {z[0]:.3f}x + {z[1]:.1f}')
                ax.legend()
        
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Average Reward")
    
    def plot_agent_correlation_matrix(self, data_list, ax):
        """Plot correlation matrix between agents."""
        ax.set_title("Agent Performance Correlation", fontweight='bold')
        
        # Collect all episode rewards
        agent_data = [[] for _ in range(4)]
        
        for data in data_list:
            for ep in data['episodes']:
                for agent_idx in range(4):
                    agent_data[agent_idx].append(ep['total_rewards'][agent_idx])
        
        if all(len(data) > 1 for data in agent_data):
            # Create correlation matrix
            df = pd.DataFrame({f'Agent {i+1}': agent_data[i] for i in range(4)})
            corr_matrix = df.corr()
            
            # Plot heatmap
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            
            # Add text annotations
            for i in range(4):
                for j in range(4):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontweight='bold')
            
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels([f'A{i+1}' for i in range(4)])
            ax.set_yticklabels([f'A{i+1}' for i in range(4)])
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def plot_best_vs_worst_runs(self, data_list, ax):
        """Compare best vs worst performing runs."""
        ax.set_title("Best vs Worst Run Comparison", fontweight='bold')
        
        # Find best and worst runs
        run_performances = []
        for data in data_list:
            total_reward = sum(sum(ep['total_rewards']) for ep in data['episodes'])
            run_performances.append(total_reward)
        
        if run_performances:
            best_idx = np.argmax(run_performances)
            worst_idx = np.argmin(run_performances)
            
            best_data = data_list[best_idx]
            worst_data = data_list[worst_idx]
            
            # Plot episode progression for both
            best_episodes = [sum(ep['total_rewards']) for ep in best_data['episodes']]
            worst_episodes = [sum(ep['total_rewards']) for ep in worst_data['episodes']]
            
            ax.plot(range(len(best_episodes)), best_episodes, 'o-', 
                   color=self.colors['success'], linewidth=2, label=f'Best Run ({run_performances[best_idx]:.1f})')
            ax.plot(range(len(worst_episodes)), worst_episodes, 's-', 
                   color=self.colors['secondary'], linewidth=2, label=f'Worst Run ({run_performances[worst_idx]:.1f})')
            
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total Team Reward")
            ax.legend()
    
    def plot_episode_progression(self, data_list, ax):
        """Plot how episodes progress within runs."""
        ax.set_title("Episode Progression Patterns", fontweight='bold')
        
        for i, data in enumerate(data_list):
            if len(data['episodes']) > 1:
                episode_rewards = [sum(ep['total_rewards']) for ep in data['episodes']]
                ax.plot(range(len(episode_rewards)), episode_rewards, 
                       alpha=0.6, linewidth=1, color=self.colors['primary'])
        
        # Add average progression
        all_progressions = []
        max_episodes = max(len(data['episodes']) for data in data_list)
        
        for ep_num in range(max_episodes):
            episode_rewards = []
            for data in data_list:
                if ep_num < len(data['episodes']):
                    episode_rewards.append(sum(data['episodes'][ep_num]['total_rewards']))
            if episode_rewards:
                all_progressions.append(np.mean(episode_rewards))
        
        if all_progressions:
            ax.plot(range(len(all_progressions)), all_progressions, 
                   color=self.colors['accent'], linewidth=3, label='Average')
            ax.legend()
        
        ax.set_xlabel("Episode Number")
        ax.set_ylabel("Total Team Reward")
    
    def plot_reward_volatility(self, data_list, ax):
        """Plot reward volatility across runs."""
        ax.set_title("Reward Volatility Analysis", fontweight='bold')
        
        volatilities = []
        run_names = []
        
        for i, data in enumerate(data_list):
            episode_rewards = [sum(ep['total_rewards']) for ep in data['episodes']]
            if len(episode_rewards) > 1:
                volatility = np.std(episode_rewards)
                volatilities.append(volatility)
                run_names.append(f"Run {i+1}")
        
        if volatilities:
            bars = ax.bar(range(len(run_names)), volatilities, 
                         color=self.colors['accent'], alpha=0.8)
            
            ax.set_xlabel("Run")
            ax.set_ylabel("Reward Standard Deviation")
            ax.set_xticks(range(len(run_names)))
            ax.set_xticklabels(run_names, rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, volatilities):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(volatilities)*0.01,
                       f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    def plot_convergence_analysis(self, data_list, ax):
        """Analyze convergence patterns."""
        ax.set_title("Convergence Analysis", fontweight='bold')
        
        # Calculate improvement rates
        improvement_rates = []
        
        for data in data_list:
            if len(data['episodes']) > 1:
                episode_rewards = [sum(ep['total_rewards']) for ep in data['episodes']]
                # Calculate linear trend
                x = np.arange(len(episode_rewards))
                slope, _ = np.polyfit(x, episode_rewards, 1)
                improvement_rates.append(slope)
        
        if improvement_rates:
            ax.hist(improvement_rates, bins=10, color=self.colors['primary'], alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', label='No improvement')
            ax.axvline(np.mean(improvement_rates), color=self.colors['accent'], 
                      linestyle='--', label=f'Mean: {np.mean(improvement_rates):.2f}')
            ax.legend()
        
        ax.set_xlabel("Improvement Rate (reward/episode)")
        ax.set_ylabel("Frequency")
    
    def plot_run_statistics_summary(self, data_list, ax):
        """Summary statistics for all runs."""
        ax.set_title("Run Statistics Summary", fontweight='bold')
        
        stats_text = f"📊 MULTI-AGENT ANALYSIS SUMMARY\n\n"
        stats_text += f"Total Runs: {len(data_list)}\n"
        
        total_episodes = sum(len(data['episodes']) for data in data_list)
        total_steps = sum(sum(ep['steps'] for ep in data['episodes']) for data in data_list)
        
        stats_text += f"Total Episodes: {total_episodes}\n"
        stats_text += f"Total Steps: {total_steps}\n\n"
        
        # Best performance
        all_rewards = [sum(sum(ep['total_rewards']) for ep in data['episodes']) for data in data_list]
        if all_rewards:
            stats_text += f"Best Run: {max(all_rewards):.1f}\n"
            stats_text += f"Worst Run: {min(all_rewards):.1f}\n"
            stats_text += f"Average: {np.mean(all_rewards):.1f}\n"
            stats_text += f"Std Dev: {np.std(all_rewards):.1f}\n\n"
        
        # Success rate
        successful_runs = sum(1 for reward in all_rewards if reward > 0)
        success_rate = (successful_runs / len(all_rewards) * 100) if all_rewards else 0
        stats_text += f"Success Rate: {success_rate:.1f}%"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        ax.axis('off')
    
    def plot_agent_ranking(self, data_list, ax):
        """Rank agents by overall performance."""
        ax.set_title("Agent Performance Ranking", fontweight='bold')
        
        agent_totals = [0] * 4
        
        for data in data_list:
            for ep in data['episodes']:
                for agent_idx in range(4):
                    agent_totals[agent_idx] += ep['total_rewards'][agent_idx]
        
        # Sort agents by performance
        agent_names = ['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4']
        sorted_data = sorted(zip(agent_names, agent_totals, self.colors['agents'][:4]), 
                           key=lambda x: x[1], reverse=True)
        
        names, totals, colors = zip(*sorted_data)
        
        bars = ax.barh(range(len(names)), totals, color=colors, alpha=0.8)
        
        ax.set_xlabel("Total Reward")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, totals)):
            ax.text(bar.get_width() + max(totals)*0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.1f}', ha='left', va='center', fontweight='bold')
    
    def plot_performance_heatmap(self, data_list, ax):
        """Create performance heatmap across runs and agents."""
        ax.set_title("Performance Heatmap", fontweight='bold')
        
        # Create matrix: runs x agents
        performance_matrix = []
        
        for data in data_list:
            run_performance = []
            for agent_idx in range(4):
                total_reward = sum(ep['total_rewards'][agent_idx] for ep in data['episodes'])
                run_performance.append(total_reward)
            performance_matrix.append(run_performance)
        
        if performance_matrix:
            performance_matrix = np.array(performance_matrix)
            
            im = ax.imshow(performance_matrix, cmap='RdYlBu_r', aspect='auto')
            
            # Add text annotations
            for i in range(len(performance_matrix)):
                for j in range(4):
                    text = ax.text(j, i, f'{performance_matrix[i, j]:.1f}',
                                 ha="center", va="center", color="black", fontsize=8)
            
            ax.set_xticks(range(4))
            ax.set_yticks(range(len(performance_matrix)))
            ax.set_xticklabels([f'Agent {i+1}' for i in range(4)])
            ax.set_yticklabels([f'Run {i+1}' for i in range(len(performance_matrix))])
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def plot_key_metrics_radar(self, data_list, ax):
        """Create radar chart of key metrics."""
        ax.set_title("Key Metrics Overview", fontweight='bold')
        
        # Calculate key metrics
        all_rewards = [sum(sum(ep['total_rewards']) for ep in data['episodes']) for data in data_list]
        all_episodes = [len(data['episodes']) for data in data_list]
        all_steps = [sum(ep['steps'] for ep in data['episodes']) for data in data_list]
        
        if all_rewards:
            metrics = {
                'Avg Reward': np.mean(all_rewards),
                'Consistency': 100 - (np.std(all_rewards) / np.mean(all_rewards) * 100) if np.mean(all_rewards) != 0 else 0,
                'Efficiency': np.mean(all_rewards) / np.mean(all_steps) * 100 if np.mean(all_steps) != 0 else 0,
                'Success Rate': sum(1 for r in all_rewards if r > 0) / len(all_rewards) * 100,
                'Avg Episodes': np.mean(all_episodes) * 10  # Scale for visibility
            }
            
            # Normalize metrics to 0-100 scale
            max_vals = {'Avg Reward': max(all_rewards), 'Consistency': 100, 'Efficiency': 10, 
                       'Success Rate': 100, 'Avg Episodes': max(all_episodes) * 10}
            
            normalized_metrics = {k: (v / max_vals[k] * 100) if max_vals[k] != 0 else 0 
                                for k, v in metrics.items()}
            
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(normalized_metrics), endpoint=False)
            values = list(normalized_metrics.values())
            values += values[:1]  # Complete the circle
            angles = np.concatenate((angles, [angles[0]]))
            
            ax.plot(angles, values, 'o-', linewidth=2, color=self.colors['primary'])
            ax.fill(angles, values, alpha=0.25, color=self.colors['primary'])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(list(normalized_metrics.keys()))
            ax.set_ylim(0, 100)
            ax.grid(True)
    
    def visualize_simulation_data(self):
        """Visualize single simulation data."""
        print("🎮 Analyzing simulation data...")
        
        sim_file = self.data_dir / "simple_simulations" / "simulation_dense_20250919_235300.json"
        
        if not sim_file.exists():
            print("   ❌ No simulation data found")
            return
        
        try:
            with open(sim_file, 'r') as f:
                data = json.load(f)
            
            print(f"   📊 Loaded simulation with {len(data['data'])} steps")
            
            # Create simulation analysis
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Single Agent Simulation Analysis - Dense Traffic', 
                        fontsize=14, fontweight='bold')
            
            # Extract time series data
            steps = [d['step'] for d in data['data']]
            rewards = [d['reward'] for d in data['data']]
            speeds = [d['speed'] for d in data['data']]
            actions = [d['action'] for d in data['data']]
            
            # 1. Reward over time
            ax1 = axes[0, 0]
            ax1.plot(steps, rewards, color=self.colors['primary'], linewidth=1.5)
            ax1.set_title("Reward Over Time", fontweight='bold')
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Reward")
            ax1.grid(True, alpha=0.3)
            
            # Add moving average
            if len(rewards) > 10:
                moving_avg = pd.Series(rewards).rolling(window=10).mean()
                ax1.plot(steps, moving_avg, color=self.colors['accent'], linewidth=2, 
                        label='Moving Average (10)')
                ax1.legend()
            
            # 2. Speed over time
            ax2 = axes[0, 1]
            ax2.plot(steps, speeds, color=self.colors['secondary'], linewidth=1.5)
            ax2.set_title("Speed Over Time", fontweight='bold')
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Speed")
            ax2.grid(True, alpha=0.3)
            
            # 3. Action distribution
            ax3 = axes[0, 2]
            action_counts = pd.Series(actions).value_counts().sort_index()
            action_names = {0: 'SLOWER', 1: 'IDLE', 2: 'FASTER', 3: 'LANE_LEFT', 4: 'LANE_RIGHT'}
            
            bars = ax3.bar([action_names.get(i, f'Action {i}') for i in action_counts.index], 
                          action_counts.values, color=self.colors['agents'][:len(action_counts)], alpha=0.8)
            ax3.set_title("Action Distribution", fontweight='bold')
            ax3.set_xlabel("Action")
            ax3.set_ylabel("Frequency")
            ax3.tick_params(axis='x', rotation=45)
            
            # Add percentage labels
            total_actions = sum(action_counts.values)
            for bar, count in zip(bars, action_counts.values):
                percentage = count / total_actions * 100
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(action_counts.values)*0.01,
                        f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
            
            # 4. Reward vs Speed scatter
            ax4 = axes[1, 0]
            scatter = ax4.scatter(speeds, rewards, c=steps, cmap='viridis', alpha=0.6, s=20)
            ax4.set_title("Reward vs Speed Relationship", fontweight='bold')
            ax4.set_xlabel("Speed")
            ax4.set_ylabel("Reward")
            plt.colorbar(scatter, ax=ax4, label='Step')
            
            # 5. Performance statistics
            ax5 = axes[1, 1]
            stats_text = f"📊 SIMULATION STATISTICS\n\n"
            stats_text += f"Scenario: {data['scenario']}\n"
            stats_text += f"Total Steps: {data['statistics']['steps']}\n"
            stats_text += f"Episodes: {data['statistics']['episodes']}\n"
            stats_text += f"Total Reward: {data['statistics']['total_reward']:.2f}\n"
            stats_text += f"Avg Reward/Step: {np.mean(rewards):.3f}\n"
            stats_text += f"Avg Speed: {np.mean(speeds):.3f}\n"
            stats_text += f"Speed Std: {np.std(speeds):.3f}\n"
            stats_text += f"Collisions: {data['statistics']['collisions']}\n"
            stats_text += f"Lane Changes: {data['statistics']['lane_changes']}\n\n"
            stats_text += f"Best Reward: {max(rewards):.3f}\n"
            stats_text += f"Worst Reward: {min(rewards):.3f}\n"
            stats_text += f"Reward Volatility: {np.std(rewards):.3f}"
            
            ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
            ax5.axis('off')
            
            # 6. Cumulative reward
            ax6 = axes[1, 2]
            cumulative_reward = np.cumsum(rewards)
            ax6.plot(steps, cumulative_reward, color=self.colors['success'], linewidth=2)
            ax6.set_title("Cumulative Reward", fontweight='bold')
            ax6.set_xlabel("Step")
            ax6.set_ylabel("Cumulative Reward")
            ax6.grid(True, alpha=0.3)
            
            # Add final value annotation
            ax6.text(steps[-1], cumulative_reward[-1], f'{cumulative_reward[-1]:.1f}',
                    ha='right', va='bottom', fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.tight_layout()
            
            # Save the plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_analysis_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"   💾 Simulation analysis saved as: {filename}")
            plt.show()
            
        except Exception as e:
            print(f"   ⚠️ Could not analyze simulation data: {e}")
    
    def visualize_dataset_overview(self):
        """Visualize dataset overview."""
        print("📈 Creating dataset overview...")
        
        index_file = self.data_dir / "highway_multimodal_dataset" / "index.json"
        
        if not index_file.exists():
            print("   ❌ No dataset index found")
            return
        
        try:
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            # Create dataset overview
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Highway Multimodal Dataset Overview', fontsize=14, fontweight='bold')
            
            # 1. Dataset info
            ax1 = axes[0, 0]
            info_text = f"📊 DATASET INFORMATION\n\n"
            info_text += f"Name: {index_data.get('dataset_name', 'Unknown')}\n"
            info_text += f"Created: {index_data.get('created', 'Unknown')[:10]}\n"
            info_text += f"Total Files: {index_data.get('total_files', 0)}\n"
            info_text += f"Scenarios: {len(index_data.get('scenarios', {}))}\n\n"
            
            for scenario, files in index_data.get('scenarios', {}).items():
                info_text += f"{scenario.replace('_', ' ').title()}: {len(files)} files\n"
            
            ax1.text(0.05, 0.95, info_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
            ax1.set_title("Dataset Information", fontweight='bold')
            ax1.axis('off')
            
            # 2. Scenario distribution
            ax2 = axes[0, 1]
            scenarios = index_data.get('scenarios', {})
            if scenarios:
                scenario_names = [s.replace('_', ' ').title() for s in scenarios.keys()]
                scenario_counts = [len(files) for files in scenarios.values()]
                
                colors = [self.colors['primary'], self.colors['secondary']][:len(scenario_names)]
                wedges, texts, autotexts = ax2.pie(scenario_counts, labels=scenario_names, 
                                                  autopct='%1.1f%%', startangle=90, colors=colors)
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            
            ax2.set_title("Scenario Distribution", fontweight='bold')
            
            # 3. File timeline (if timestamps available)
            ax3 = axes[1, 0]
            ax3.set_title("Dataset Timeline", fontweight='bold')
            
            # Extract creation dates
            creation_dates = []
            scenario_labels = []
            
            for scenario, files in scenarios.items():
                for file_info in files:
                    if 'created' in file_info:
                        creation_dates.append(file_info['created'][:10])
                        scenario_labels.append(scenario.replace('_', ' ').title())
            
            if creation_dates:
                # Simple timeline visualization
                unique_dates = sorted(set(creation_dates))
                date_counts = [creation_dates.count(date) for date in unique_dates]
                
                ax3.bar(range(len(unique_dates)), date_counts, 
                       color=self.colors['accent'], alpha=0.8)
                ax3.set_xlabel("Date")
                ax3.set_ylabel("Files Created")
                ax3.set_xticks(range(len(unique_dates)))
                ax3.set_xticklabels(unique_dates, rotation=45)
            else:
                ax3.text(0.5, 0.5, "No timeline data\navailable", 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            
            # 4. Data structure overview
            ax4 = axes[1, 1]
            ax4.set_title("Data Structure", fontweight='bold')
            
            structure_text = "📁 DATA STRUCTURE\n\n"
            structure_text += "Each scenario contains:\n"
            structure_text += "• Transition files (.parquet)\n"
            structure_text += "• Metadata files (.jsonl)\n\n"
            structure_text += "File types:\n"
            structure_text += "• Parquet: Structured data\n"
            structure_text += "• JSONL: Metadata & configs\n\n"
            structure_text += "Scenarios:\n"
            for scenario in scenarios.keys():
                structure_text += f"• {scenario.replace('_', ' ').title()}\n"
            
            ax4.text(0.05, 0.95, structure_text, transform=ax4.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
            ax4.axis('off')
            
            plt.tight_layout()
            
            # Save the plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dataset_overview_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"   💾 Dataset overview saved as: {filename}")
            plt.show()
            
        except Exception as e:
            print(f"   ⚠️ Could not create dataset overview: {e}")
    
    def create_summary_dashboard(self):
        """Create a summary dashboard of all data."""
        print("📊 Creating summary dashboard...")
        
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Highway Environment Data - Complete Summary Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Data inventory
        ax1 = plt.subplot(2, 3, 1)
        self.plot_data_inventory(ax1)
        
        # 2. Performance overview
        ax2 = plt.subplot(2, 3, 2)
        self.plot_performance_overview(ax2)
        
        # 3. Data timeline
        ax3 = plt.subplot(2, 3, 3)
        self.plot_data_timeline(ax3)
        
        # 4. Key insights
        ax4 = plt.subplot(2, 3, 4)
        self.plot_key_insights(ax4)
        
        # 5. Recommendations
        ax5 = plt.subplot(2, 3, 5)
        self.plot_recommendations(ax5)
        
        # 6. Next steps
        ax6 = plt.subplot(2, 3, 6)
        self.plot_next_steps(ax6)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_dashboard_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   💾 Summary dashboard saved as: {filename}")
        plt.show()
    
    def plot_data_inventory(self, ax):
        """Plot data inventory summary."""
        ax.set_title("Data Inventory", fontweight='bold')
        
        # Count different data types
        multiagent_files = len(glob.glob(str(self.data_dir / "multiagent_demo_*.json")))
        sim_files = len(glob.glob(str(self.data_dir / "simple_simulations" / "*.json")))
        car_watch_files = len(glob.glob(str(self.data_dir / "car_watching" / "*.json")))
        dataset_files = 1 if (self.data_dir / "highway_multimodal_dataset" / "index.json").exists() else 0
        
        categories = ['Multi-Agent\nDemos', 'Single Agent\nSimulations', 'Car Watching\nData', 'Multimodal\nDataset']
        counts = [multiagent_files, sim_files, car_watch_files, dataset_files]
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent'], self.colors['success']]
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.8)
        
        ax.set_ylabel("Number of Files")
        ax.tick_params(axis='x', rotation=45)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom', fontweight='bold')
    
    def plot_performance_overview(self, ax):
        """Plot overall performance summary."""
        ax.set_title("Performance Overview", fontweight='bold')
        
        # Load multi-agent data for performance summary
        pattern = str(self.data_dir / "multiagent_demo_*.json")
        files = glob.glob(pattern)
        
        valid_rewards = []
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    total_reward = sum(sum(ep['total_rewards']) for ep in data['episodes'])
                    valid_rewards.append(total_reward)
            except:
                continue
        
        if valid_rewards:
            # Performance distribution
            ax.hist(valid_rewards, bins=8, color=self.colors['primary'], alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(valid_rewards), color=self.colors['accent'], 
                      linestyle='--', linewidth=2, label=f'Mean: {np.mean(valid_rewards):.1f}')
            ax.axvline(np.median(valid_rewards), color=self.colors['secondary'], 
                      linestyle='--', linewidth=2, label=f'Median: {np.median(valid_rewards):.1f}')
            ax.legend()
            ax.set_xlabel("Total Run Reward")
            ax.set_ylabel("Frequency")
        else:
            ax.text(0.5, 0.5, "No performance\ndata available", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    def plot_data_timeline(self, ax):
        """Plot data creation timeline."""
        ax.set_title("Data Creation Timeline", fontweight='bold')
        
        # Extract timestamps from filenames
        pattern = str(self.data_dir / "multiagent_demo_*_*.json")
        files = glob.glob(pattern)
        
        timestamps = []
        for file_path in files:
            try:
                # Extract timestamp from filename
                filename = Path(file_path).stem
                timestamp_str = filename.split('_')[-1]  # Get the timestamp part
                # Convert to datetime for plotting
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                timestamps.append(timestamp)
            except:
                continue
        
        if timestamps:
            timestamps.sort()
            # Group by hour for visualization
            hours = [ts.strftime("%H:%M") for ts in timestamps]
            hour_counts = {}
            for hour in hours:
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
            
            if len(hour_counts) > 1:
                ax.plot(range(len(hour_counts)), list(hour_counts.values()), 
                       'o-', color=self.colors['primary'], linewidth=2, markersize=6)
                ax.set_xlabel("Time")
                ax.set_ylabel("Files Created")
                ax.set_xticks(range(len(hour_counts)))
                ax.set_xticklabels(list(hour_counts.keys()), rotation=45)
            else:
                ax.bar(['Today'], [len(timestamps)], color=self.colors['primary'], alpha=0.8)
                ax.set_ylabel("Total Files")
        else:
            ax.text(0.5, 0.5, "No timeline\ndata available", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    def plot_key_insights(self, ax):
        """Plot key insights from the data."""
        ax.set_title("Key Insights", fontweight='bold')
        
        insights_text = "🔍 KEY INSIGHTS\n\n"
        
        # Analyze multi-agent data
        pattern = str(self.data_dir / "multiagent_demo_*.json")
        files = glob.glob(pattern)
        
        valid_data = []
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    valid_data.append(data)
            except:
                continue
        
        if valid_data:
            # Calculate insights
            all_rewards = []
            all_episodes = []
            agent_performances = [[] for _ in range(4)]
            
            for data in valid_data:
                total_reward = sum(sum(ep['total_rewards']) for ep in data['episodes'])
                all_rewards.append(total_reward)
                all_episodes.append(len(data['episodes']))
                
                for ep in data['episodes']:
                    for agent_idx in range(4):
                        agent_performances[agent_idx].append(ep['total_rewards'][agent_idx])
            
            insights_text += f"• {len(valid_data)} successful runs analyzed\n"
            insights_text += f"• Best performance: {max(all_rewards):.1f}\n"
            insights_text += f"• Average performance: {np.mean(all_rewards):.1f}\n"
            insights_text += f"• Success rate: {sum(1 for r in all_rewards if r > 0)/len(all_rewards)*100:.0f}%\n\n"
            
            # Agent analysis
            agent_means = [np.mean(perf) for perf in agent_performances]
            best_agent = np.argmax(agent_means)
            insights_text += f"• Best agent: Agent {best_agent + 1}\n"
            insights_text += f"• Agent performance gap: {max(agent_means) - min(agent_means):.1f}\n\n"
            
            # Consistency analysis
            consistency = 100 - (np.std(all_rewards) / np.mean(all_rewards) * 100) if np.mean(all_rewards) != 0 else 0
            insights_text += f"• Performance consistency: {consistency:.0f}%\n"
            
            if consistency > 70:
                insights_text += "• High consistency indicates stable learning\n"
            elif consistency > 40:
                insights_text += "• Moderate consistency, room for improvement\n"
            else:
                insights_text += "• Low consistency, high variability\n"
        
        ax.text(0.05, 0.95, insights_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        ax.axis('off')
    
    def plot_recommendations(self, ax):
        """Plot recommendations based on data analysis."""
        ax.set_title("Recommendations", fontweight='bold')
        
        recommendations_text = "💡 RECOMMENDATIONS\n\n"
        
        # Analyze data to generate recommendations
        pattern = str(self.data_dir / "multiagent_demo_*.json")
        files = glob.glob(pattern)
        
        valid_rewards = []
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    total_reward = sum(sum(ep['total_rewards']) for ep in data['episodes'])
                    valid_rewards.append(total_reward)
            except:
                continue
        
        if valid_rewards:
            avg_performance = np.mean(valid_rewards)
            consistency = np.std(valid_rewards) / np.mean(valid_rewards) if np.mean(valid_rewards) != 0 else 1
            
            if avg_performance < 10:
                recommendations_text += "• Focus on reward engineering\n"
                recommendations_text += "• Consider adjusting learning rates\n"
            elif avg_performance < 30:
                recommendations_text += "• Good progress, optimize policies\n"
                recommendations_text += "• Experiment with different scenarios\n"
            else:
                recommendations_text += "• Excellent performance!\n"
                recommendations_text += "• Consider more challenging scenarios\n"
            
            if consistency > 0.5:
                recommendations_text += "• High variability detected\n"
                recommendations_text += "• Stabilize training parameters\n"
                recommendations_text += "• Increase training episodes\n"
            else:
                recommendations_text += "• Good consistency achieved\n"
                recommendations_text += "• Ready for deployment testing\n"
            
            recommendations_text += "\n🎯 NEXT EXPERIMENTS:\n"
            recommendations_text += "• Test different traffic densities\n"
            recommendations_text += "• Implement cooperative strategies\n"
            recommendations_text += "• Add more complex scenarios\n"
            recommendations_text += "• Analyze agent communication\n"
        
        ax.text(0.05, 0.95, recommendations_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        ax.axis('off')
    
    def plot_next_steps(self, ax):
        """Plot next steps for research."""
        ax.set_title("Next Steps", fontweight='bold')
        
        next_steps_text = "🚀 NEXT STEPS\n\n"
        next_steps_text += "📊 DATA COLLECTION:\n"
        next_steps_text += "• Collect more diverse scenarios\n"
        next_steps_text += "• Increase episode lengths\n"
        next_steps_text += "• Add weather conditions\n"
        next_steps_text += "• Test emergency scenarios\n\n"
        
        next_steps_text += "🤖 MODEL IMPROVEMENTS:\n"
        next_steps_text += "• Implement attention mechanisms\n"
        next_steps_text += "• Add memory components\n"
        next_steps_text += "• Test different architectures\n"
        next_steps_text += "• Optimize hyperparameters\n\n"
        
        next_steps_text += "🔬 ANALYSIS:\n"
        next_steps_text += "• Behavioral pattern analysis\n"
        next_steps_text += "• Safety metric evaluation\n"
        next_steps_text += "• Efficiency benchmarking\n"
        next_steps_text += "• Real-world validation\n\n"
        
        next_steps_text += "📈 VISUALIZATION:\n"
        next_steps_text += "• Interactive dashboards\n"
        next_steps_text += "• Real-time monitoring\n"
        next_steps_text += "• 3D trajectory plots\n"
        next_steps_text += "• Video generation\n"
        
        ax.text(0.05, 0.95, next_steps_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        ax.axis('off')


def main():
    """Main function to run focused data visualization."""
    visualizer = FocusedDataVisualizer()
    
    print("🎨 Starting focused data visualization...")
    print("This will create comprehensive, high-quality visualizations of all your data.")
    print()
    
    try:
        visualizer.visualize_all_data()
        
        print("\n🎉 Visualization complete!")
        print("📁 Check the generated PNG files for detailed analysis:")
        print("   • multiagent_comprehensive_*.png - Complete multi-agent analysis")
        print("   • simulation_analysis_*.png - Single agent simulation analysis") 
        print("   • dataset_overview_*.png - Dataset structure overview")
        print("   • summary_dashboard_*.png - Complete summary dashboard")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()