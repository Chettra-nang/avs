#!/usr/bin/env python3
"""
🚑 Ambulance Yielding Behavior Analyzer
=========================================
Analyzes and visualizes whether other vehicles yield to the ambulance.

Key Metrics:
- Lane changes when ambulance approaches
- Speed reductions when ambulance is near
- Distance maintenance from ambulance
- Lateral movement away from ambulance path
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import json
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, '/home/chettra/ITC/Research/AVs')

try:
    from highway_datacollection.storage.encoders import BinaryArrayEncoder
except ImportError:
    print("⚠️  Warning: Could not import BinaryArrayEncoder, using basic decoding")
    BinaryArrayEncoder = None

@dataclass
class YieldingMetrics:
    """Metrics for analyzing yielding behavior"""
    distance_to_ambulance: float
    relative_speed: float
    lateral_distance: float
    lane_difference: int
    is_ahead: bool
    is_yielding: bool
    yield_type: str  # 'speed_reduction', 'lane_change', 'maintaining_distance', 'none'

class AmbulanceYieldingAnalyzer:
    """Analyze yielding behavior of vehicles around ambulance"""
    
    def __init__(self, data_path: str):
        """
        Initialize analyzer with ambulance dataset
        
        Args:
            data_path: Path to parquet file containing ambulance data
        """
        self.data_path = Path(data_path)
        self.df = pd.read_parquet(data_path)
        self.encoder = BinaryArrayEncoder()
        
        # Yielding thresholds
        self.YIELD_DISTANCE_THRESHOLD = 30.0  # meters
        self.SPEED_REDUCTION_THRESHOLD = 0.2  # 20% speed reduction
        self.LANE_CHANGE_THRESHOLD = 0.5  # significant lateral movement
        
        print(f"📊 Loaded ambulance dataset: {data_path}")
        print(f"   Episodes: {self.df['episode_id'].nunique()}")
        print(f"   Total steps: {len(self.df)}")
        print(f"   Agents: {self.df['agent_id'].nunique()}")
        
    def identify_ambulance_agent(self, episode_id: str) -> int:
        """
        Identify which agent is the ambulance
        
        Args:
            episode_id: Episode to analyze
            
        Returns:
            Agent ID of the ambulance (typically agent 0)
        """
        # Ambulance is typically the ego vehicle (agent 0)
        # We can also check for highest speed or special metadata
        episode_data = self.df[self.df['episode_id'] == episode_id]
        
        # Check metadata for ambulance tag
        if 'is_ambulance' in episode_data.columns:
            ambulance_agent = episode_data[episode_data['is_ambulance'] == True]['agent_id'].iloc[0]
            return ambulance_agent
        
        # Otherwise, assume agent 0 is ambulance
        return 0
    
    def calculate_yielding_metrics(self, episode_id: str, step: int) -> Dict[int, YieldingMetrics]:
        """
        Calculate yielding metrics for all vehicles at a given step
        
        Args:
            episode_id: Episode to analyze
            step: Time step
            
        Returns:
            Dictionary mapping agent_id to YieldingMetrics
        """
        step_data = self.df[(self.df['episode_id'] == episode_id) & (self.df['step'] == step)]
        
        if len(step_data) == 0:
            return {}
        
        ambulance_id = self.identify_ambulance_agent(episode_id)
        ambulance_data = step_data[step_data['agent_id'] == ambulance_id].iloc[0]
        
        metrics = {}
        
        for _, vehicle_data in step_data.iterrows():
            agent_id = vehicle_data['agent_id']
            
            if agent_id == ambulance_id:
                continue  # Skip ambulance itself
            
            # Calculate spatial metrics
            dx = vehicle_data['ego_x'] - ambulance_data['ego_x']
            dy = vehicle_data['ego_y'] - ambulance_data['ego_y']
            distance = np.sqrt(dx**2 + dy**2)
            
            # Calculate velocity metrics
            dvx = vehicle_data['ego_vx'] - ambulance_data['ego_vx']
            dvy = vehicle_data['ego_vy'] - ambulance_data['ego_vy']
            relative_speed = np.sqrt(dvx**2 + dvy**2)
            
            # Determine if vehicle is ahead
            is_ahead = dx > 0
            
            # Lane difference
            lane_diff = int(vehicle_data.get('lane_position', 0) - ambulance_data.get('lane_position', 0))
            
            # Lateral distance (perpendicular to road direction)
            lateral_distance = abs(dy)
            
            # Determine yielding behavior
            yield_type = 'none'
            is_yielding = False
            
            if distance < self.YIELD_DISTANCE_THRESHOLD:
                # Check for speed reduction
                if step > 0:
                    prev_step_data = self.df[(self.df['episode_id'] == episode_id) & 
                                            (self.df['step'] == step - 1) & 
                                            (self.df['agent_id'] == agent_id)]
                    if len(prev_step_data) > 0:
                        prev_speed = np.sqrt(prev_step_data.iloc[0]['ego_vx']**2 + 
                                           prev_step_data.iloc[0]['ego_vy']**2)
                        current_speed = np.sqrt(vehicle_data['ego_vx']**2 + vehicle_data['ego_vy']**2)
                        
                        if prev_speed > 0 and (prev_speed - current_speed) / prev_speed > self.SPEED_REDUCTION_THRESHOLD:
                            yield_type = 'speed_reduction'
                            is_yielding = True
                
                # Check for lane change (lateral movement)
                if step > 0:
                    prev_step_data = self.df[(self.df['episode_id'] == episode_id) & 
                                            (self.df['step'] == step - 1) & 
                                            (self.df['agent_id'] == agent_id)]
                    if len(prev_step_data) > 0:
                        prev_y = prev_step_data.iloc[0]['ego_y']
                        lateral_movement = abs(vehicle_data['ego_y'] - prev_y)
                        
                        if lateral_movement > self.LANE_CHANGE_THRESHOLD:
                            yield_type = 'lane_change'
                            is_yielding = True
                
                # Check for maintaining safe distance
                if distance > 10.0 and not is_yielding:
                    yield_type = 'maintaining_distance'
                    is_yielding = True
            
            metrics[agent_id] = YieldingMetrics(
                distance_to_ambulance=distance,
                relative_speed=relative_speed,
                lateral_distance=lateral_distance,
                lane_difference=lane_diff,
                is_ahead=is_ahead,
                is_yielding=is_yielding,
                yield_type=yield_type
            )
        
        return metrics
    
    def analyze_episode(self, episode_id: str) -> Dict[str, any]:
        """
        Comprehensive analysis of yielding behavior for an episode
        
        Args:
            episode_id: Episode to analyze
            
        Returns:
            Dictionary with analysis results
        """
        episode_data = self.df[self.df['episode_id'] == episode_id]
        steps = sorted(episode_data['step'].unique())
        
        ambulance_id = self.identify_ambulance_agent(episode_id)
        
        # Track yielding events
        total_interactions = 0
        yielding_events = 0
        yield_types = {'speed_reduction': 0, 'lane_change': 0, 'maintaining_distance': 0, 'none': 0}
        
        # Track per-step metrics
        step_metrics = []
        
        for step in steps:
            metrics = self.calculate_yielding_metrics(episode_id, step)
            
            for agent_id, metric in metrics.items():
                if metric.distance_to_ambulance < self.YIELD_DISTANCE_THRESHOLD:
                    total_interactions += 1
                    if metric.is_yielding:
                        yielding_events += 1
                    yield_types[metric.yield_type] += 1
            
            step_metrics.append({
                'step': step,
                'metrics': metrics
            })
        
        yielding_rate = (yielding_events / total_interactions * 100) if total_interactions > 0 else 0
        
        return {
            'episode_id': episode_id,
            'ambulance_agent_id': ambulance_id,
            'total_steps': len(steps),
            'total_interactions': total_interactions,
            'yielding_events': yielding_events,
            'yielding_rate': yielding_rate,
            'yield_type_distribution': yield_types,
            'step_metrics': step_metrics
        }
    
    def visualize_yielding_behavior(self, episode_id: str, output_dir: str = "output/ambulance_analysis"):
        """
        Create comprehensive visualization of yielding behavior
        
        Args:
            episode_id: Episode to visualize
            output_dir: Directory to save visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎨 Creating yielding behavior visualization for: {episode_id}")
        
        # Analyze episode
        analysis = self.analyze_episode(episode_id)
        
        # Create multi-panel visualization
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: Episode Overview
        ax_overview = fig.add_subplot(gs[0, :])
        self._plot_episode_overview(ax_overview, analysis)
        
        # Panel 2: Distance over time
        ax_distance = fig.add_subplot(gs[1, 0])
        self._plot_distance_over_time(ax_distance, episode_id, analysis)
        
        # Panel 3: Yielding events timeline
        ax_timeline = fig.add_subplot(gs[1, 1])
        self._plot_yielding_timeline(ax_timeline, analysis)
        
        # Panel 4: Yield type distribution
        ax_distribution = fig.add_subplot(gs[1, 2])
        self._plot_yield_distribution(ax_distribution, analysis)
        
        # Panel 5: Spatial visualization at key moment
        ax_spatial = fig.add_subplot(gs[2, :])
        self._plot_spatial_view(ax_spatial, episode_id, analysis)
        
        plt.suptitle(f'🚑 Ambulance Yielding Behavior Analysis - {episode_id}', 
                    fontsize=16, fontweight='bold')
        
        # Save figure
        output_file = output_path / f"{episode_id}_yielding_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"💾 Saved visualization: {output_file}")
        
        plt.close()
        
        # Print summary
        self._print_analysis_summary(analysis)
        
        return analysis
    
    def _plot_episode_overview(self, ax, analysis):
        """Plot episode overview with key statistics"""
        ax.axis('off')
        
        overview_text = (
            f"📊 Episode Overview\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Episode ID: {analysis['episode_id']}\n"
            f"Ambulance: Agent {analysis['ambulance_agent_id']}\n"
            f"Total Steps: {analysis['total_steps']}\n"
            f"\n"
            f"🚗 Vehicle Interactions:\n"
            f"  • Total Interactions: {analysis['total_interactions']}\n"
            f"  • Yielding Events: {analysis['yielding_events']}\n"
            f"  • Yielding Rate: {analysis['yielding_rate']:.1f}%\n"
            f"\n"
            f"📈 Yield Type Distribution:\n"
            f"  • Speed Reduction: {analysis['yield_type_distribution']['speed_reduction']}\n"
            f"  • Lane Change: {analysis['yield_type_distribution']['lane_change']}\n"
            f"  • Maintaining Distance: {analysis['yield_type_distribution']['maintaining_distance']}\n"
            f"  • No Yielding: {analysis['yield_type_distribution']['none']}\n"
        )
        
        ax.text(0.05, 0.5, overview_text, transform=ax.transAxes,
                fontsize=11, fontfamily='monospace', verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        ax.set_title('Episode Statistics', fontweight='bold', fontsize=12)
    
    def _plot_distance_over_time(self, ax, episode_id, analysis):
        """Plot distance to ambulance over time for each vehicle"""
        episode_data = self.df[self.df['episode_id'] == episode_id]
        ambulance_id = analysis['ambulance_agent_id']
        
        agents = episode_data['agent_id'].unique()
        
        for agent_id in agents:
            if agent_id == ambulance_id:
                continue
            
            agent_data = episode_data[episode_data['agent_id'] == agent_id]
            ambulance_data = episode_data[episode_data['agent_id'] == ambulance_id]
            
            distances = []
            steps = []
            
            for step in sorted(agent_data['step'].unique()):
                agent_step = agent_data[agent_data['step'] == step].iloc[0]
                amb_step = ambulance_data[ambulance_data['step'] == step].iloc[0]
                
                dx = agent_step['ego_x'] - amb_step['ego_x']
                dy = agent_step['ego_y'] - amb_step['ego_y']
                distance = np.sqrt(dx**2 + dy**2)
                
                distances.append(distance)
                steps.append(step)
            
            ax.plot(steps, distances, label=f'Vehicle {agent_id}', linewidth=2)
        
        ax.axhline(y=self.YIELD_DISTANCE_THRESHOLD, color='r', linestyle='--', 
                  label='Yield Threshold', linewidth=2)
        ax.set_xlabel('Step', fontweight='bold')
        ax.set_ylabel('Distance to Ambulance (m)', fontweight='bold')
        ax.set_title('Distance to Ambulance Over Time', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_yielding_timeline(self, ax, analysis):
        """Plot timeline of yielding events"""
        yielding_steps = []
        yield_types = []
        agents = []
        
        for step_data in analysis['step_metrics']:
            step = step_data['step']
            for agent_id, metric in step_data['metrics'].items():
                if metric.is_yielding:
                    yielding_steps.append(step)
                    yield_types.append(metric.yield_type)
                    agents.append(agent_id)
        
        if yielding_steps:
            colors = {'speed_reduction': 'red', 'lane_change': 'blue', 
                     'maintaining_distance': 'green'}
            
            for i, (step, ytype, agent) in enumerate(zip(yielding_steps, yield_types, agents)):
                ax.scatter(step, agent, c=colors.get(ytype, 'gray'), 
                          s=100, alpha=0.6, edgecolors='black', linewidth=1)
            
            # Create legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', label='Speed Reduction'),
                Patch(facecolor='blue', label='Lane Change'),
                Patch(facecolor='green', label='Maintaining Distance')
            ]
            ax.legend(handles=legend_elements, fontsize=8, loc='upper left')
        
        ax.set_xlabel('Step', fontweight='bold')
        ax.set_ylabel('Agent ID', fontweight='bold')
        ax.set_title('Yielding Events Timeline', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_yield_distribution(self, ax, analysis):
        """Plot distribution of yield types"""
        yield_dist = analysis['yield_type_distribution']
        
        labels = []
        values = []
        colors = []
        
        color_map = {
            'speed_reduction': '#FF6B6B',
            'lane_change': '#4ECDC4',
            'maintaining_distance': '#95E1D3',
            'none': '#C7C7C7'
        }
        
        for ytype, count in yield_dist.items():
            if count > 0:
                labels.append(ytype.replace('_', ' ').title())
                values.append(count)
                colors.append(color_map.get(ytype, '#C7C7C7'))
        
        if values:
            wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors,
                                              autopct='%1.1f%%', startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
        
        ax.set_title('Yield Type Distribution', fontweight='bold')
    
    def _plot_spatial_view(self, ax, episode_id, analysis):
        """Plot spatial view of vehicles at a key moment"""
        # Find step with most interactions
        max_interactions = 0
        best_step = 0
        
        for step_data in analysis['step_metrics']:
            interactions = sum(1 for m in step_data['metrics'].values() 
                             if m.distance_to_ambulance < self.YIELD_DISTANCE_THRESHOLD)
            if interactions > max_interactions:
                max_interactions = interactions
                best_step = step_data['step']
        
        # Plot vehicle positions at this step
        step_data = self.df[(self.df['episode_id'] == episode_id) & 
                           (self.df['step'] == best_step)]
        
        ambulance_id = analysis['ambulance_agent_id']
        
        for _, vehicle in step_data.iterrows():
            agent_id = vehicle['agent_id']
            x, y = vehicle['ego_x'], vehicle['ego_y']
            
            if agent_id == ambulance_id:
                # Ambulance - red with siren symbol
                ax.scatter(x, y, c='red', s=500, marker='s', 
                          edgecolors='darkred', linewidth=3, label='🚑 Ambulance', zorder=10)
                ax.annotate('🚑', (x, y), fontsize=20, ha='center', va='center', zorder=11)
            else:
                # Other vehicles - blue
                ax.scatter(x, y, c='blue', s=300, marker='o', 
                          edgecolors='darkblue', linewidth=2, alpha=0.7)
                ax.annotate(f'{agent_id}', (x, y), fontsize=10, ha='center', va='center',
                           color='white', fontweight='bold')
                
                # Draw arrow showing velocity
                vx, vy = vehicle['ego_vx'], vehicle['ego_vy']
                ax.arrow(x, y, vx*2, vy*2, head_width=1, head_length=0.5, 
                        fc='blue', ec='blue', alpha=0.5)
        
        # Draw yield zone around ambulance
        ambulance_pos = step_data[step_data['agent_id'] == ambulance_id].iloc[0]
        circle = plt.Circle((ambulance_pos['ego_x'], ambulance_pos['ego_y']), 
                           self.YIELD_DISTANCE_THRESHOLD, 
                           color='red', fill=False, linestyle='--', 
                           linewidth=2, label='Yield Zone')
        ax.add_patch(circle)
        
        ax.set_xlabel('X Position (m)', fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontweight='bold')
        ax.set_title(f'Spatial View at Step {best_step} (Peak Interaction)', fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _print_analysis_summary(self, analysis):
        """Print analysis summary to console"""
        print("\n" + "="*60)
        print("🚑 AMBULANCE YIELDING BEHAVIOR SUMMARY")
        print("="*60)
        print(f"\n📊 Episode: {analysis['episode_id']}")
        print(f"   Ambulance Agent: {analysis['ambulance_agent_id']}")
        print(f"   Duration: {analysis['total_steps']} steps")
        
        print(f"\n🚗 Vehicle Interactions:")
        print(f"   Total interactions within yield zone: {analysis['total_interactions']}")
        print(f"   Vehicles that yielded: {analysis['yielding_events']}")
        print(f"   Yielding rate: {analysis['yielding_rate']:.1f}%")
        
        print(f"\n📈 Yielding Behavior Breakdown:")
        for ytype, count in analysis['yield_type_distribution'].items():
            percentage = (count / analysis['total_interactions'] * 100) if analysis['total_interactions'] > 0 else 0
            print(f"   {ytype.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        print("\n" + "="*60)
        
        # Verdict
        if analysis['yielding_rate'] > 70:
            print("✅ VERDICT: Vehicles are ACTIVELY YIELDING to the ambulance!")
        elif analysis['yielding_rate'] > 40:
            print("⚠️  VERDICT: Vehicles show MODERATE yielding behavior")
        else:
            print("❌ VERDICT: Vehicles are NOT yielding effectively")
        print("="*60 + "\n")
    
    def create_yielding_animation(self, episode_id: str, output_dir: str = "output/ambulance_analysis"):
        """
        Create animation showing yielding behavior over time
        
        Args:
            episode_id: Episode to animate
            output_dir: Directory to save animation
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎬 Creating yielding animation for: {episode_id}")
        
        episode_data = self.df[self.df['episode_id'] == episode_id]
        steps = sorted(episode_data['step'].unique())
        ambulance_id = self.identify_ambulance_agent(episode_id)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        def animate(frame_idx):
            step = steps[frame_idx]
            ax1.clear()
            ax2.clear()
            
            step_data = episode_data[episode_data['step'] == step]
            
            # Left panel: Spatial view
            ambulance_pos = step_data[step_data['agent_id'] == ambulance_id].iloc[0]
            
            for _, vehicle in step_data.iterrows():
                agent_id = vehicle['agent_id']
                x, y = vehicle['ego_x'], vehicle['ego_y']
                
                if agent_id == ambulance_id:
                    ax1.scatter(x, y, c='red', s=500, marker='s', 
                              edgecolors='darkred', linewidth=3, zorder=10)
                    ax1.annotate('🚑', (x, y), fontsize=20, ha='center', va='center', zorder=11)
                else:
                    # Check if yielding
                    metrics = self.calculate_yielding_metrics(episode_id, step)
                    if agent_id in metrics and metrics[agent_id].is_yielding:
                        color = 'green'
                        alpha = 1.0
                    else:
                        color = 'blue'
                        alpha = 0.5
                    
                    ax1.scatter(x, y, c=color, s=300, marker='o', 
                              edgecolors='dark' + color, linewidth=2, alpha=alpha)
                    ax1.annotate(f'{agent_id}', (x, y), fontsize=10, ha='center', va='center',
                               color='white', fontweight='bold')
            
            # Draw yield zone
            circle = plt.Circle((ambulance_pos['ego_x'], ambulance_pos['ego_y']), 
                               self.YIELD_DISTANCE_THRESHOLD,
                               color='red', fill=False, linestyle='--', linewidth=2)
            ax1.add_patch(circle)
            
            ax1.set_xlabel('X Position (m)', fontweight='bold')
            ax1.set_ylabel('Y Position (m)', fontweight='bold')
            ax1.set_title(f'Step {step}: Vehicle Positions', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')
            
            # Right panel: Yielding status
            metrics = self.calculate_yielding_metrics(episode_id, step)
            
            yield_status = []
            agents_list = []
            for agent_id, metric in metrics.items():
                agents_list.append(agent_id)
                if metric.is_yielding:
                    yield_status.append(f"✅ YIELDING ({metric.yield_type})")
                else:
                    yield_status.append("❌ Not yielding")
            
            if agents_list:
                y_pos = np.arange(len(agents_list))
                colors = ['green' if '✅' in status else 'red' for status in yield_status]
                
                ax2.barh(y_pos, [1]*len(agents_list), color=colors, alpha=0.6)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels([f'Agent {a}' for a in agents_list])
                ax2.set_xlim([0, 1.5])
                ax2.set_xticks([])
                
                for i, status in enumerate(yield_status):
                    ax2.text(0.5, i, status, ha='center', va='center', 
                            fontweight='bold', fontsize=10, color='white')
            
            ax2.set_title('Yielding Status', fontweight='bold')
            
            fig.suptitle(f'🚑 Ambulance Yielding Behavior - {episode_id}', 
                        fontsize=14, fontweight='bold')
        
        anim = animation.FuncAnimation(fig, animate, frames=len(steps), 
                                      interval=200, repeat=True)
        
        output_file = output_path / f"{episode_id}_yielding_animation.gif"
        anim.save(output_file, writer='pillow', fps=5)
        print(f"💾 Saved animation: {output_file}")
        
        plt.close()


def main():
    """Main function to run yielding behavior analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze ambulance yielding behavior')
    parser.add_argument('--data', type=str, 
                       default='/home/chettra/ITC/Research/AVs/data/ambulance_dataset/batch_5802/highway_rush_hour/20250927_212622-fe8ada51_transitions.parquet',
                       help='Path to ambulance dataset parquet file')
    parser.add_argument('--episode', type=str, default=None,
                       help='Specific episode to analyze (default: first episode)')
    parser.add_argument('--output', type=str, default='output/ambulance_analysis',
                       help='Output directory for visualizations')
    parser.add_argument('--animate', action='store_true',
                       help='Create animation of yielding behavior')
    parser.add_argument('--all-episodes', action='store_true',
                       help='Analyze all episodes in dataset')
    
    args = parser.parse_args()
    
    print("🚑 Ambulance Yielding Behavior Analyzer")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = AmbulanceYieldingAnalyzer(args.data)
    
    # Get episodes to analyze
    if args.all_episodes:
        episodes = analyzer.df['episode_id'].unique()
        print(f"\n📊 Analyzing {len(episodes)} episodes...")
    elif args.episode:
        episodes = [args.episode]
    else:
        episodes = [analyzer.df['episode_id'].iloc[0]]
    
    # Analyze each episode
    all_results = []
    for episode_id in episodes:
        print(f"\n{'='*60}")
        print(f"Analyzing episode: {episode_id}")
        print(f"{'='*60}")
        
        # Create visualization
        result = analyzer.visualize_yielding_behavior(episode_id, args.output)
        all_results.append(result)
        
        # Create animation if requested
        if args.animate:
            analyzer.create_yielding_animation(episode_id, args.output)
    
    # Overall summary if multiple episodes
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("📊 OVERALL ANALYSIS ACROSS ALL EPISODES")
        print("="*60)
        
        total_interactions = sum(r['total_interactions'] for r in all_results)
        total_yielding = sum(r['yielding_events'] for r in all_results)
        overall_rate = (total_yielding / total_interactions * 100) if total_interactions > 0 else 0
        
        print(f"\nTotal interactions: {total_interactions}")
        print(f"Total yielding events: {total_yielding}")
        print(f"Overall yielding rate: {overall_rate:.1f}%")
        
        if overall_rate > 70:
            print("\n✅ Overall verdict: Vehicles ARE yielding to ambulance!")
        elif overall_rate > 40:
            print("\n⚠️  Overall verdict: Moderate yielding behavior")
        else:
            print("\n❌ Overall verdict: Poor yielding behavior")
        
        print("="*60)
    
    print(f"\n✅ Analysis complete! Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
