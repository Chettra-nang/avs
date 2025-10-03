#!/usr/bin/env python3
"""
🚗 Non-Agent Vehicle (NPC) Yielding Analysis
=============================================
Analyzes whether background traffic (non-agent vehicles) yield to ambulance.

This script examines the kinematics data to track ALL vehicles in the simulation,
not just the controlled agents, to see if NPCs are programmed to yield.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, '/home/chettra/ITC/Research/AVs')

try:
    from highway_datacollection.storage.encoders import BinaryArrayEncoder
except ImportError:
    print("⚠️  Warning: Could not import BinaryArrayEncoder")
    BinaryArrayEncoder = None


@dataclass
class NPCYieldingMetrics:
    """Metrics for NPC yielding behavior"""
    vehicle_index: int
    distance_to_ambulance: float
    relative_speed: float
    is_present: bool
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    is_yielding: bool
    yield_evidence: str


class NPCYieldingAnalyzer:
    """Analyze yielding behavior of non-agent vehicles (NPCs)"""
    
    def __init__(self, data_path: str):
        """Initialize analyzer with ambulance dataset"""
        self.data_path = Path(data_path)
        self.df = pd.read_parquet(data_path)
        self.encoder = BinaryArrayEncoder() if BinaryArrayEncoder else None
        
        # Yielding thresholds
        self.YIELD_DISTANCE_THRESHOLD = 40.0  # Slightly larger for NPCs
        self.SPEED_REDUCTION_THRESHOLD = 0.15  # 15% speed reduction
        
        print(f"📊 Loaded dataset: {data_path}")
        print(f"   Episodes: {self.df['episode_id'].nunique()}")
        print(f"   Agents: {self.df['agent_id'].nunique()}")
        
        # Check if we have kinematics data
        if 'kinematics_raw' in self.df.columns:
            print(f"   ✅ Kinematics data available (tracks ALL vehicles)")
        else:
            print(f"   ⚠️  No kinematics data found")
    
    def extract_npc_vehicles(self, episode_id: str, step: int, ambulance_agent_id: int = 0) -> List[Dict]:
        """
        Extract all NPC vehicles from kinematics observation
        
        Args:
            episode_id: Episode to analyze
            step: Time step
            ambulance_agent_id: Which agent is the ambulance
            
        Returns:
            List of NPC vehicle data
        """
        step_data = self.df[(self.df['episode_id'] == episode_id) & 
                           (self.df['step'] == step) & 
                           (self.df['agent_id'] == ambulance_agent_id)]
        
        if len(step_data) == 0:
            return []
        
        ambulance_data = step_data.iloc[0]
        kinematics_raw = ambulance_data.get('kinematics_raw', None)
        
        if kinematics_raw is None:
            return []
        
        # Convert to numpy array and reshape
        # Kinematics format: [presence, x, y, vx, vy, cos_h, sin_h] per vehicle (7 features)
        kinematics = np.array(kinematics_raw)
        if len(kinematics) == 0:
            return []
        
        # Reshape to (num_vehicles, 7)
        num_vehicles = len(kinematics) // 7
        if num_vehicles == 0:
            return []
        
        kinematics = kinematics.reshape(num_vehicles, 7)
        
        # Ambulance position
        amb_x = ambulance_data['ego_x']
        amb_y = ambulance_data['ego_y']
        amb_vx = ambulance_data['ego_vx']
        amb_vy = ambulance_data['ego_vy']
        
        npc_vehicles = []
        
        # Index 0 is typically the ego vehicle itself, so we start from 1
        for i, vehicle in enumerate(kinematics):
            if len(vehicle) < 5:
                continue
            
            presence = vehicle[0]
            if presence < 0.5:  # Vehicle not present
                continue
            
            # Positions are relative to ego vehicle
            rel_x = vehicle[1]
            rel_y = vehicle[2]
            rel_vx = vehicle[3]
            rel_vy = vehicle[4]
            
            # Convert to absolute positions
            abs_x = amb_x + rel_x
            abs_y = amb_y + rel_y
            abs_vx = amb_vx + rel_vx
            abs_vy = amb_vy + rel_vy
            
            # Calculate distance
            distance = np.sqrt(rel_x**2 + rel_y**2)
            
            # Skip if this is actually a controlled agent (very close to their ego position)
            # Controlled agents would appear in the observation with near-zero relative position
            if i == 0 or distance < 1.0:  # Index 0 or very close = likely a controlled agent
                continue
            
            npc_vehicles.append({
                'index': i,
                'presence': presence,
                'rel_x': rel_x,
                'rel_y': rel_y,
                'rel_vx': rel_vx,
                'rel_vy': rel_vy,
                'abs_x': abs_x,
                'abs_y': abs_y,
                'abs_vx': abs_vx,
                'abs_vy': abs_vy,
                'distance': distance
            })
        
        return npc_vehicles
    
    def analyze_npc_yielding(self, episode_id: str) -> Dict:
        """
        Analyze NPC yielding behavior throughout an episode
        
        Args:
            episode_id: Episode to analyze
            
        Returns:
            Analysis results
        """
        episode_data = self.df[self.df['episode_id'] == episode_id]
        steps = sorted(episode_data['step'].unique())
        ambulance_agent_id = 0  # Assume agent 0 is ambulance
        
        print(f"\n🔍 Analyzing NPC vehicles in episode: {episode_id}")
        print(f"   Steps: {len(steps)}")
        
        # Track NPC behavior over time
        npc_tracking = {}  # vehicle_index -> list of states
        total_npcs = 0
        yielding_events = 0
        close_encounters = 0
        
        for step in steps:
            npcs = self.extract_npc_vehicles(episode_id, step, ambulance_agent_id)
            
            for npc in npcs:
                idx = npc['index']
                
                # Initialize tracking for this NPC
                if idx not in npc_tracking:
                    npc_tracking[idx] = []
                
                # Check if this NPC should be yielding
                is_yielding = False
                yield_evidence = "none"
                
                if npc['distance'] < self.YIELD_DISTANCE_THRESHOLD:
                    close_encounters += 1
                    
                    # Check for speed reduction compared to previous step
                    if len(npc_tracking[idx]) > 0:
                        prev_state = npc_tracking[idx][-1]
                        prev_speed = np.sqrt(prev_state['abs_vx']**2 + prev_state['abs_vy']**2)
                        curr_speed = np.sqrt(npc['abs_vx']**2 + npc['abs_vy']**2)
                        
                        if prev_speed > 0.1:  # Avoid division by zero
                            speed_change = (prev_speed - curr_speed) / prev_speed
                            
                            if speed_change > self.SPEED_REDUCTION_THRESHOLD:
                                is_yielding = True
                                yield_evidence = f"speed_reduced_{speed_change*100:.1f}%"
                                yielding_events += 1
                            elif speed_change < -0.05:  # Speeding up
                                yield_evidence = "accelerating"
                            else:
                                yield_evidence = "maintaining_speed"
                        
                        # Check for lateral movement (lane change)
                        lateral_change = abs(npc['rel_y'] - prev_state['rel_y'])
                        if lateral_change > 0.3:
                            if not is_yielding:
                                is_yielding = True
                                yielding_events += 1
                            yield_evidence = f"lane_change_{lateral_change:.2f}m"
                
                npc_tracking[idx].append({
                    'step': step,
                    'distance': npc['distance'],
                    'abs_vx': npc['abs_vx'],
                    'abs_vy': npc['abs_vy'],
                    'rel_y': npc['rel_y'],
                    'is_yielding': is_yielding,
                    'yield_evidence': yield_evidence
                })
        
        total_npcs = len(npc_tracking)
        yielding_rate = (yielding_events / close_encounters * 100) if close_encounters > 0 else 0
        
        return {
            'episode_id': episode_id,
            'total_npcs_tracked': total_npcs,
            'close_encounters': close_encounters,
            'yielding_events': yielding_events,
            'yielding_rate': yielding_rate,
            'npc_tracking': npc_tracking,
            'total_steps': len(steps)
        }
    
    def visualize_npc_yielding(self, episode_id: str, output_dir: str = "output/ambulance_yielding_analysis"):
        """
        Create visualization of NPC yielding behavior
        
        Args:
            episode_id: Episode to visualize
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎨 Creating NPC yielding visualization...")
        
        # Analyze episode
        analysis = self.analyze_npc_yielding(episode_id)
        
        if analysis['total_npcs_tracked'] == 0:
            print("⚠️  No NPC vehicles found in this episode!")
            return analysis
        
        # Create visualization
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: Overview
        ax_overview = fig.add_subplot(gs[0, :])
        self._plot_npc_overview(ax_overview, analysis)
        
        # Panel 2: NPC distance over time
        ax_distance = fig.add_subplot(gs[1, 0])
        self._plot_npc_distances(ax_distance, analysis)
        
        # Panel 3: Yielding timeline
        ax_timeline = fig.add_subplot(gs[1, 1])
        self._plot_npc_yielding_timeline(ax_timeline, analysis)
        
        # Panel 4: Speed changes
        ax_speed = fig.add_subplot(gs[1, 2])
        self._plot_npc_speed_changes(ax_speed, analysis)
        
        # Panel 5: Spatial view at peak moment
        ax_spatial = fig.add_subplot(gs[2, :])
        self._plot_npc_spatial_view(ax_spatial, episode_id, analysis)
        
        plt.suptitle(f'🚗 NPC (Non-Agent) Vehicle Yielding Analysis - {episode_id}',
                    fontsize=16, fontweight='bold')
        
        # Save
        output_file = output_path / f"{episode_id}_npc_yielding_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: {output_file}")
        
        plt.close()
        
        # Print summary
        self._print_npc_summary(analysis)
        
        return analysis
    
    def _plot_npc_overview(self, ax, analysis):
        """Plot NPC overview statistics"""
        ax.axis('off')
        
        overview_text = (
            f"🚗 Non-Agent Vehicle (NPC) Analysis\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Episode: {analysis['episode_id']}\n"
            f"Duration: {analysis['total_steps']} steps\n"
            f"\n"
            f"📊 NPC Statistics:\n"
            f"  • Total NPCs Tracked: {analysis['total_npcs_tracked']}\n"
            f"  • Close Encounters (<40m): {analysis['close_encounters']}\n"
            f"  • Yielding Events: {analysis['yielding_events']}\n"
            f"  • NPC Yielding Rate: {analysis['yielding_rate']:.1f}%\n"
            f"\n"
            f"💡 What are NPCs?\n"
            f"  NPCs (Non-Player Characters) are background\n"
            f"  traffic vehicles controlled by the simulation\n"
            f"  (not AI agents). They use rule-based behavior\n"
            f"  like IDM (Intelligent Driver Model).\n"
            f"\n"
            f"🎯 Analysis Method:\n"
            f"  • Extracted from kinematics observations\n"
            f"  • Tracked via relative positions to ambulance\n"
            f"  • Monitored for speed/lane changes\n"
        )
        
        ax.text(0.05, 0.5, overview_text, transform=ax.transAxes,
                fontsize=11, fontfamily='monospace', verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        ax.set_title('NPC Vehicle Overview', fontweight='bold', fontsize=12)
    
    def _plot_npc_distances(self, ax, analysis):
        """Plot NPC distances to ambulance over time"""
        for npc_idx, states in analysis['npc_tracking'].items():
            steps = [s['step'] for s in states]
            distances = [s['distance'] for s in states]
            
            ax.plot(steps, distances, alpha=0.6, linewidth=1.5, label=f'NPC {npc_idx}')
        
        ax.axhline(y=self.YIELD_DISTANCE_THRESHOLD, color='r', linestyle='--',
                  label='Yield Zone', linewidth=2)
        ax.set_xlabel('Step', fontweight='bold')
        ax.set_ylabel('Distance to Ambulance (m)', fontweight='bold')
        ax.set_title('NPC Distances Over Time', fontweight='bold')
        ax.grid(True, alpha=0.3)
        if len(analysis['npc_tracking']) <= 10:
            ax.legend(fontsize=8, loc='best')
    
    def _plot_npc_yielding_timeline(self, ax, analysis):
        """Plot timeline of NPC yielding events"""
        yielding_steps = []
        npc_indices = []
        
        for npc_idx, states in analysis['npc_tracking'].items():
            for state in states:
                if state['is_yielding']:
                    yielding_steps.append(state['step'])
                    npc_indices.append(npc_idx)
        
        if yielding_steps:
            ax.scatter(yielding_steps, npc_indices, c='green', s=100,
                      alpha=0.6, edgecolors='darkgreen', linewidth=2)
        
        ax.set_xlabel('Step', fontweight='bold')
        ax.set_ylabel('NPC Vehicle Index', fontweight='bold')
        ax.set_title('NPC Yielding Events Timeline', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_npc_speed_changes(self, ax, analysis):
        """Plot speed change distribution"""
        speed_changes = []
        
        for npc_idx, states in analysis['npc_tracking'].items():
            for i in range(1, len(states)):
                prev_speed = np.sqrt(states[i-1]['abs_vx']**2 + states[i-1]['abs_vy']**2)
                curr_speed = np.sqrt(states[i]['abs_vx']**2 + states[i]['abs_vy']**2)
                
                if prev_speed > 0.1:
                    speed_change = (curr_speed - prev_speed) / prev_speed * 100
                    if states[i]['distance'] < self.YIELD_DISTANCE_THRESHOLD:
                        speed_changes.append(speed_change)
        
        if speed_changes:
            ax.hist(speed_changes, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='No change')
            ax.axvline(x=-self.SPEED_REDUCTION_THRESHOLD*100, color='g', linestyle='--',
                      linewidth=2, label='Yielding threshold')
        
        ax.set_xlabel('Speed Change (%)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('NPC Speed Changes Near Ambulance', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_npc_spatial_view(self, ax, episode_id, analysis):
        """Plot spatial distribution of NPCs"""
        # Find step with most NPCs in yield zone
        max_npcs = 0
        best_step = 0
        
        for step in range(analysis['total_steps']):
            count = sum(1 for states in analysis['npc_tracking'].values()
                       if any(s['step'] == step and s['distance'] < self.YIELD_DISTANCE_THRESHOLD 
                             for s in states))
            if count > max_npcs:
                max_npcs = count
                best_step = step
        
        # Get NPC positions at this step
        npcs = self.extract_npc_vehicles(episode_id, best_step)
        
        # Plot ambulance at origin (since NPCs are relative to it)
        ax.scatter(0, 0, c='red', s=500, marker='s', edgecolors='darkred',
                  linewidth=3, label='🚑 Ambulance', zorder=10)
        ax.annotate('🚑', (0, 0), fontsize=20, ha='center', va='center', zorder=11)
        
        # Plot NPCs
        for npc in npcs:
            color = 'green' if npc['distance'] < self.YIELD_DISTANCE_THRESHOLD else 'gray'
            alpha = 0.8 if npc['distance'] < self.YIELD_DISTANCE_THRESHOLD else 0.3
            
            ax.scatter(npc['rel_x'], npc['rel_y'], c=color, s=200, marker='o',
                      edgecolors='dark'+color, linewidth=2, alpha=alpha)
            ax.annotate(f"NPC{npc['index']}", (npc['rel_x'], npc['rel_y']),
                       fontsize=8, ha='center', va='center', color='white',
                       fontweight='bold')
            
            # Draw velocity arrow
            ax.arrow(npc['rel_x'], npc['rel_y'], npc['rel_vx']*2, npc['rel_vy']*2,
                    head_width=1, head_length=0.5, fc=color, ec=color, alpha=0.5)
        
        # Draw yield zone
        circle = plt.Circle((0, 0), self.YIELD_DISTANCE_THRESHOLD,
                          color='red', fill=False, linestyle='--',
                          linewidth=2, label='Yield Zone (40m)')
        ax.add_patch(circle)
        
        ax.set_xlabel('Relative X Position (m)', fontweight='bold')
        ax.set_ylabel('Relative Y Position (m)', fontweight='bold')
        ax.set_title(f'NPC Positions at Step {best_step} (Relative to Ambulance)',
                    fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _print_npc_summary(self, analysis):
        """Print NPC analysis summary"""
        print("\n" + "="*60)
        print("🚗 NPC VEHICLE YIELDING SUMMARY")
        print("="*60)
        print(f"\n📊 Episode: {analysis['episode_id']}")
        print(f"   Duration: {analysis['total_steps']} steps")
        
        print(f"\n🚗 NPC Vehicles:")
        print(f"   Total NPCs tracked: {analysis['total_npcs_tracked']}")
        print(f"   Close encounters: {analysis['close_encounters']}")
        print(f"   Yielding events: {analysis['yielding_events']}")
        print(f"   NPC yielding rate: {analysis['yielding_rate']:.1f}%")
        
        print("\n" + "="*60)
        
        # Verdict
        if analysis['yielding_rate'] > 60:
            print("✅ VERDICT: NPCs ARE yielding to ambulance!")
        elif analysis['yielding_rate'] > 30:
            print("⚠️  VERDICT: NPCs show MODERATE yielding")
        else:
            print("❌ VERDICT: NPCs are NOT yielding effectively")
        
        if analysis['total_npcs_tracked'] == 0:
            print("⚠️  Note: No NPC vehicles detected in this episode")
        
        print("="*60 + "\n")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze NPC yielding to ambulance')
    parser.add_argument('--data', type=str,
                       default='/home/chettra/ITC/Research/AVs/data/ambulance_dataset/batch_5802/highway_rush_hour/20250927_212622-fe8ada51_transitions.parquet',
                       help='Path to dataset')
    parser.add_argument('--episode', type=str, default=None,
                       help='Specific episode to analyze')
    parser.add_argument('--output', type=str,
                       default='output/ambulance_yielding_analysis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("🚗 NPC (Non-Agent) Vehicle Yielding Analyzer")
    print("=" * 60)
    
    analyzer = NPCYieldingAnalyzer(args.data)
    
    # Get episode
    if args.episode:
        episode_id = args.episode
    else:
        episode_id = analyzer.df['episode_id'].iloc[0]
    
    # Analyze
    result = analyzer.visualize_npc_yielding(episode_id, args.output)
    
    print(f"\n✅ Analysis complete! Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
