#!/usr/bin/env python3
"""
Comprehensive Data Plotter

This script automatically detects and visualizes all data files in the data folder,
including multi-agent demos, car watching data, and simulation data.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import glob
from datetime import datetime
import os

class ComprehensiveDataPlotter:
    """Plot all types of data from the data folder."""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Color schemes for different data types
        self.colors = {
            'multiagent': ['red', 'green', 'blue', 'orange', 'purple', 'brown'],
            'single': ['steelblue', 'darkgreen', 'crimson', 'darkorange'],
            'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        }
    
    def discover_data_files(self):
        """Discover all data files in the data directory."""
        print("🔍 Discovering data files...")
        
        data_files = {
            'multiagent': [],
            'car_watching': [],
            'simulations': [],
            'multimodal': [],
            'other': []
        }
        
        # Multi-agent demo files
        multiagent_pattern = str(self.data_dir / "multiagent_demo_*.json")
        data_files['multiagent'] = glob.glob(multiagent_pattern)
        
        # Car watching files
        car_watch_pattern = str(self.data_dir / "car_watching" / "*.json")
        data_files['car_watching'] = glob.glob(car_watch_pattern)
        
        # Simple simulation files
        sim_pattern = str(self.data_dir / "simple_simulations" / "*.json")
        data_files['simulations'] = glob.glob(sim_pattern)
        
        # Multimodal dataset
        multimodal_index = self.data_dir / "highway_multimodal_dataset" / "index.json"
        if multimodal_index.exists():
            data_files['multimodal'].append(str(multimodal_index))
        
        # Print discovery results
        for data_type, files in data_files.items():
            if files:
                print(f"   {data_type.title()}: {len(files)} files")
        
        return data_files
    
    def plot_all_data(self, save_plots=True):
        """Plot all discovered data files."""
        data_files = self.discover_data_files()
        
        # Plot multi-agent data
        if data_files['multiagent']:
            self.plot_multiagent_comparison(data_files['multiagent'], save_plots)
        
        # Plot car watching data
        if data_files['car_watching']:
            self.plot_car_watching_data(data_files['car_watching'], save_plots)
        
        # Plot simulation data
        if data_files['simulations']:
            self.plot_simulation_data(data_files['simulations'], save_plots)
        
        # Plot multimodal dataset overview
        if data_files['multimodal']:
            self.plot_multimodal_overview(data_files['multimodal'], save_plots)
    
    def plot_multiagent_comparison(self, files, save_plots=True):
        """Compare all multi-agent demo runs."""
        print("📊 Plotting multi-agent comparison...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Multi-Agent Demo Comparison Across All Runs', fontsize=16, fontweight='bold')
        
        all_data = []
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['filename'] = Path(file_path).stem
                    all_data.append(data)
            except Exception as e:
                print(f"   ⚠️ Could not load {file_path}: {e}")
        
        if not all_data:
            print("   ❌ No valid multi-agent data found")
            return
        
        # 1. Total rewards comparison across runs
        ax1 = axes[0, 0]
        self.plot_total_rewards_comparison(all_data, ax1)
        
        # 2. Episode performance trends
        ax2 = axes[0, 1]
        self.plot_episode_trends(all_data, ax2)
        
        # 3. Agent performance consistency
        ax3 = axes[0, 2]
        self.plot_agent_consistency(all_data, ax3)
        
        # 4. Average rewards per run
        ax4 = axes[1, 0]
        self.plot_run_averages(all_data, ax4)
        
        # 5. Success rate analysis
        ax5 = axes[1, 1]
        self.plot_success_rates(all_data, ax5)
        
        # 6. Run duration comparison
        ax6 = axes[1, 2]
        self.plot_run_durations(all_data, ax6)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multiagent_comparison_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   💾 Multi-agent comparison saved as: {filename}")
        
        plt.show()
    
    def plot_total_rewards_comparison(self, all_data, ax):
        """Plot total rewards comparison across runs."""
        ax.set_title("Total Rewards Across All Runs", fontweight='bold')
        
        run_names = []
        agent_rewards = [[] for _ in range(4)]  # Assuming 4 agents
        
        for i, data in enumerate(all_data):
            run_names.append(f"Run {i+1}")
            
            for agent_idx in range(4):
                total_reward = sum(ep['total_rewards'][agent_idx] for ep in data['episodes'])
                agent_rewards[agent_idx].append(total_reward)
        
        x = np.arange(len(run_names))
        width = 0.2
        
        for agent_idx in range(4):
            offset = (agent_idx - 1.5) * width
            ax.bar(x + offset, agent_rewards[agent_idx], width,
                  color=self.colors['multiagent'][agent_idx],
                  alpha=0.8, label=f'Agent {agent_idx+1}')
        
        ax.set_xlabel("Runs")
        ax.set_ylabel("Total Reward")
        ax.set_xticks(x)
        ax.set_xticklabels(run_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_episode_trends(self, all_data, ax):
        """Plot episode performance trends."""
        ax.set_title("Episode Performance Trends", fontweight='bold')
        
        for i, data in enumerate(all_data):
            episodes = [ep['episode'] for ep in data['episodes']]
            total_rewards = [sum(ep['total_rewards']) for ep in data['episodes']]
            
            ax.plot(episodes, total_rewards, 'o-', alpha=0.7, 
                   label=f'Run {i+1}', linewidth=2)
        
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Team Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_agent_consistency(self, all_data, ax):
        """Plot agent performance consistency across runs."""
        ax.set_title("Agent Performance Consistency", fontweight='bold')
        
        agent_performance = [[] for _ in range(4)]
        
        for data in all_data:
            for agent_idx in range(4):
                total_reward = sum(ep['total_rewards'][agent_idx] for ep in data['episodes'])
                agent_performance[agent_idx].append(total_reward)
        
        # Create box plot
        box_plot = ax.boxplot(agent_performance, patch_artist=True)
        
        for patch, color in zip(box_plot['boxes'], self.colors['multiagent'][:4]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel("Agent")
        ax.set_ylabel("Total Reward")
        ax.set_xticklabels(['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4'])
        ax.grid(True, alpha=0.3)
    
    def plot_run_averages(self, all_data, ax):
        """Plot average performance per run."""
        ax.set_title("Average Performance Per Run", fontweight='bold')
        
        run_averages = []
        run_names = []
        
        for i, data in enumerate(all_data):
            total_reward = 0
            total_episodes = len(data['episodes'])
            
            for episode in data['episodes']:
                total_reward += sum(episode['total_rewards'])
            
            avg_reward = total_reward / total_episodes if total_episodes > 0 else 0
            run_averages.append(avg_reward)
            run_names.append(f"Run {i+1}")
        
        bars = ax.bar(run_names, run_averages, color='skyblue', alpha=0.8)
        ax.set_xlabel("Run")
        ax.set_ylabel("Average Episode Reward")
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, run_averages):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    def plot_success_rates(self, all_data, ax):
        """Plot success rates (positive reward episodes)."""
        ax.set_title("Success Rates (Positive Reward Episodes)", fontweight='bold')
        
        success_rates = []
        run_names = []
        
        for i, data in enumerate(all_data):
            successful_episodes = 0
            total_episodes = len(data['episodes'])
            
            for episode in data['episodes']:
                if sum(episode['total_rewards']) > 0:
                    successful_episodes += 1
            
            success_rate = (successful_episodes / total_episodes * 100) if total_episodes > 0 else 0
            success_rates.append(success_rate)
            run_names.append(f"Run {i+1}")
        
        bars = ax.bar(run_names, success_rates, color='lightgreen', alpha=0.8)
        ax.set_xlabel("Run")
        ax.set_ylabel("Success Rate (%)")
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    def plot_run_durations(self, all_data, ax):
        """Plot run durations (total steps)."""
        ax.set_title("Run Durations (Total Steps)", fontweight='bold')
        
        durations = []
        run_names = []
        
        for i, data in enumerate(all_data):
            total_steps = sum(ep['steps'] for ep in data['episodes'])
            durations.append(total_steps)
            run_names.append(f"Run {i+1}")
        
        bars = ax.bar(run_names, durations, color='coral', alpha=0.8)
        ax.set_xlabel("Run")
        ax.set_ylabel("Total Steps")
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, durations):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{value}', ha='center', va='bottom', fontweight='bold')
    
    def plot_car_watching_data(self, files, save_plots=True):
        """Plot car watching data."""
        print("🚗 Plotting car watching data...")
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f'Car Watching Analysis - {Path(file_path).stem}', 
                           fontsize=14, fontweight='bold')
                
                # Extract time series data
                if 'episodes' in data:
                    self.plot_car_watching_episodes(data, axes)
                elif 'observations' in data:
                    self.plot_car_watching_observations(data, axes)
                
                plt.tight_layout()
                
                if save_plots:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"car_watching_{timestamp}.png"
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"   💾 Car watching plot saved as: {filename}")
                
                plt.show()
                
            except Exception as e:
                print(f"   ⚠️ Could not plot {file_path}: {e}")
    
    def plot_car_watching_episodes(self, data, axes):
        """Plot car watching episode data."""
        # This is a placeholder - adjust based on actual data structure
        ax1, ax2, ax3, ax4 = axes.flatten()
        
        ax1.set_title("Episode Rewards")
        ax2.set_title("Episode Durations")
        ax3.set_title("Action Distribution")
        ax4.set_title("Performance Over Time")
        
        # Add actual plotting logic based on your car watching data structure
        ax1.text(0.5, 0.5, "Car Watching\nData Visualization", 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
    
    def plot_simulation_data(self, files, save_plots=True):
        """Plot simple simulation data."""
        print("🎮 Plotting simulation data...")
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f'Simulation Analysis - {Path(file_path).stem}', 
                           fontsize=14, fontweight='bold')
                
                self.plot_simulation_metrics(data, axes)
                
                plt.tight_layout()
                
                if save_plots:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"simulation_{timestamp}.png"
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"   💾 Simulation plot saved as: {filename}")
                
                plt.show()
                
            except Exception as e:
                print(f"   ⚠️ Could not plot {file_path}: {e}")
    
    def plot_simulation_metrics(self, data, axes):
        """Plot simulation metrics."""
        ax1, ax2, ax3, ax4 = axes.flatten()
        
        # Check data structure and plot accordingly
        if 'episodes' in data:
            episodes = data['episodes']
            
            # Episode rewards
            ax1.set_title("Episode Rewards")
            episode_nums = [ep.get('episode', i) for i, ep in enumerate(episodes)]
            rewards = [ep.get('total_reward', 0) for ep in episodes]
            ax1.plot(episode_nums, rewards, 'o-', color='steelblue', linewidth=2)
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Total Reward")
            ax1.grid(True, alpha=0.3)
            
            # Episode durations
            ax2.set_title("Episode Durations")
            durations = [ep.get('steps', 0) for ep in episodes]
            ax2.bar(episode_nums, durations, color='lightcoral', alpha=0.8)
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Steps")
            
            # Performance trend
            ax3.set_title("Performance Trend")
            if rewards:
                cumulative_rewards = np.cumsum(rewards)
                ax3.plot(episode_nums, cumulative_rewards, color='darkgreen', linewidth=2)
                ax3.set_xlabel("Episode")
                ax3.set_ylabel("Cumulative Reward")
                ax3.grid(True, alpha=0.3)
            
            # Summary statistics
            ax4.set_title("Summary Statistics")
            if rewards:
                stats_text = f"Total Episodes: {len(episodes)}\n"
                stats_text += f"Average Reward: {np.mean(rewards):.2f}\n"
                stats_text += f"Best Episode: {max(rewards):.2f}\n"
                stats_text += f"Total Steps: {sum(durations)}"
                
                ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, 
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
                ax4.axis('off')
        else:
            # Generic data visualization
            for i, ax in enumerate(axes.flatten()):
                ax.text(0.5, 0.5, f"Simulation Data\nVisualization {i+1}", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    def plot_multimodal_overview(self, files, save_plots=True):
        """Plot multimodal dataset overview."""
        print("📈 Plotting multimodal dataset overview...")
        
        try:
            index_file = files[0]
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Highway Multimodal Dataset Overview', fontsize=14, fontweight='bold')
            
            # Plot dataset statistics
            self.plot_dataset_statistics(index_data, axes)
            
            plt.tight_layout()
            
            if save_plots:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"multimodal_overview_{timestamp}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"   💾 Multimodal overview saved as: {filename}")
            
            plt.show()
            
        except Exception as e:
            print(f"   ⚠️ Could not plot multimodal data: {e}")
    
    def plot_dataset_statistics(self, index_data, axes):
        """Plot dataset statistics from index."""
        ax1, ax2, ax3, ax4 = axes.flatten()
        
        # Dataset overview
        ax1.set_title("Dataset Overview")
        overview_text = "Highway Multimodal Dataset\n\n"
        
        if 'scenarios' in index_data:
            overview_text += f"Scenarios: {len(index_data['scenarios'])}\n"
        if 'total_episodes' in index_data:
            overview_text += f"Total Episodes: {index_data['total_episodes']}\n"
        if 'data_types' in index_data:
            overview_text += f"Data Types: {', '.join(index_data['data_types'])}\n"
        
        ax1.text(0.1, 0.5, overview_text, transform=ax1.transAxes, 
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        ax1.axis('off')
        
        # Scenario distribution (if available)
        ax2.set_title("Scenario Distribution")
        if 'scenarios' in index_data:
            scenarios = list(index_data['scenarios'].keys())
            counts = [len(index_data['scenarios'][s].get('episodes', [])) for s in scenarios]
            
            ax2.pie(counts, labels=scenarios, autopct='%1.1f%%', startangle=90)
        else:
            ax2.text(0.5, 0.5, "No scenario data\navailable", 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # Data type breakdown
        ax3.set_title("Data Types")
        if 'data_types' in index_data:
            data_types = index_data['data_types']
            ax3.bar(range(len(data_types)), [1]*len(data_types), 
                   tick_label=data_types, color='skyblue', alpha=0.8)
            ax3.set_ylabel("Available")
        else:
            ax3.text(0.5, 0.5, "No data type\ninformation", 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # Dataset metadata
        ax4.set_title("Dataset Metadata")
        metadata_text = ""
        
        for key, value in index_data.items():
            if key not in ['scenarios', 'data_types'] and isinstance(value, (str, int, float)):
                metadata_text += f"{key}: {value}\n"
        
        if metadata_text:
            ax4.text(0.1, 0.5, metadata_text, transform=ax4.transAxes, 
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        else:
            ax4.text(0.5, 0.5, "No metadata\navailable", 
                    ha='center', va='center', transform=ax4.transAxes)
        ax4.axis('off')
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive report of all data."""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE DATA ANALYSIS REPORT")
        print("="*80)
        
        data_files = self.discover_data_files()
        
        # Multi-agent analysis
        if data_files['multiagent']:
            print(f"\n🤖 MULTI-AGENT DEMOS ({len(data_files['multiagent'])} files):")
            
            total_episodes = 0
            total_steps = 0
            best_performance = float('-inf')
            
            for file_path in data_files['multiagent']:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    episodes = len(data['episodes'])
                    steps = sum(ep['steps'] for ep in data['episodes'])
                    performance = sum(sum(ep['total_rewards']) for ep in data['episodes'])
                    
                    total_episodes += episodes
                    total_steps += steps
                    best_performance = max(best_performance, performance)
                    
                    print(f"   {Path(file_path).stem}: {episodes} episodes, {steps} steps, {performance:.2f} total reward")
                    
                except Exception as e:
                    print(f"   ⚠️ Could not analyze {file_path}: {e}")
            
            print(f"   📈 Summary: {total_episodes} total episodes, {total_steps} total steps")
            print(f"   🏆 Best run performance: {best_performance:.2f}")
        
        # Other data types
        for data_type, files in data_files.items():
            if data_type != 'multiagent' and files:
                print(f"\n📁 {data_type.upper()} ({len(files)} files):")
                for file_path in files:
                    print(f"   {Path(file_path).name}")
        
        print("="*80)


def main():
    """Main function to run comprehensive data plotting."""
    plotter = ComprehensiveDataPlotter()
    
    print("🎨 Starting comprehensive data visualization...")
    
    try:
        # Plot all data
        plotter.plot_all_data(save_plots=True)
        
        # Generate comprehensive report
        plotter.generate_comprehensive_report()
        
        print("\n✅ Comprehensive data visualization complete!")
        print("📁 Check the generated PNG files for all visualizations.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()