#!/usr/bin/env python3
"""
Live Visualization: Cars in Action - NPC Yielding to Ambulance

This script creates a REAL-TIME animated visualization showing:
1. Ambulance (red) navigating through traffic
2. NPCs (blue) yielding by slowing down and giving way
3. Speed indicators showing when NPCs reduce speed
4. Distance circles showing proximity detection zones
5. Real-time statistics

You'll SEE the NPCs actually yielding to the ambulance!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, FancyArrow, Wedge
from matplotlib.collections import PatchCollection
import time

# Import highway-env
import gymnasium as gym
import highway_env

from collecting_ambulance_data.scenarios.ambulance_scenarios import get_scenario_by_name


class LiveYieldingVisualizer:
    """Real-time visualization of NPC yielding behavior"""
    
    def __init__(self, scenario_name="roundabout_single_lane", duration=100):
        self.scenario_name = scenario_name
        self.duration = duration
        self.config = get_scenario_by_name(scenario_name)
        
        # Setup environment
        self.env = gym.make('highway-v0', render_mode='rgb_array')
        self.env.unwrapped.configure(self.config)
        
        # Data tracking
        self.step_count = 0
        self.yielding_count = 0
        self.close_encounters = 0
        self.history = {
            'ambulance_pos': [],
            'npc_positions': [],
            'npc_speeds': [],
            'yielding_vehicles': []
        }
        
        # Setup visualization
        self.setup_plot()
        
    def setup_plot(self):
        """Setup the matplotlib figure and axes"""
        self.fig, self.axes = plt.subplots(1, 2, figsize=(18, 8))
        self.fig.suptitle(f'🚑 Live NPC Yielding Visualization: {self.scenario_name}', 
                         fontsize=16, fontweight='bold')
        
        # Left plot: Bird's eye view of vehicles
        self.ax_map = self.axes[0]
        self.ax_map.set_title('Bird\'s Eye View - Real-time Vehicle Positions', fontsize=12, fontweight='bold')
        self.ax_map.set_xlabel('X Position (m)')
        self.ax_map.set_ylabel('Y Position (m)')
        self.ax_map.grid(True, alpha=0.3)
        self.ax_map.set_aspect('equal')
        
        # Right plot: Statistics and speed indicators
        self.ax_stats = self.axes[1]
        self.ax_stats.set_title('Real-time Statistics & Speed Analysis', fontsize=12, fontweight='bold')
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')
        
        plt.tight_layout()
        
    def reset_environment(self):
        """Reset the environment"""
        obs, info = self.env.reset()
        self.step_count = 0
        self.yielding_count = 0
        self.close_encounters = 0
        
    def get_vehicle_data(self):
        """Extract current vehicle data from environment"""
        vehicles = self.env.unwrapped.road.vehicles
        
        if not vehicles:
            return None
        
        # Ambulance (first vehicle)
        ambulance = vehicles[0]
        ambulance_data = {
            'pos': ambulance.position.copy(),
            'speed': ambulance.speed,
            'heading': ambulance.heading
        }
        
        # NPCs
        npc_data = []
        yielding_npcs = []
        
        for i, vehicle in enumerate(vehicles[1:], 1):
            distance = np.linalg.norm(vehicle.position - ambulance.position)
            
            veh = {
                'id': i,
                'pos': vehicle.position.copy(),
                'speed': vehicle.speed,
                'heading': vehicle.heading,
                'distance': distance
            }
            
            # Check if yielding (within 50m and significantly slower)
            if distance < 50:
                self.close_encounters += 1
                if vehicle.speed < ambulance.speed - 2:
                    veh['yielding'] = True
                    yielding_npcs.append(i)
                    self.yielding_count += 1
                else:
                    veh['yielding'] = False
            else:
                veh['yielding'] = False
            
            npc_data.append(veh)
        
        return {
            'ambulance': ambulance_data,
            'npcs': npc_data,
            'yielding_ids': yielding_npcs
        }
    
    def draw_vehicle(self, ax, pos, heading, speed, is_ambulance=False, is_yielding=False):
        """Draw a vehicle as a rectangle with direction arrow"""
        # Vehicle dimensions
        length = 5
        width = 2
        
        # Color coding
        if is_ambulance:
            color = 'red'
            alpha = 1.0
            label = '🚑'
        elif is_yielding:
            color = 'green'
            alpha = 0.8
            label = '✓'
        else:
            color = 'blue'
            alpha = 0.6
            label = ''
        
        # Draw vehicle body
        rect = Rectangle((pos[0] - length/2, pos[1] - width/2), 
                         length, width, 
                         angle=np.degrees(heading),
                         facecolor=color, edgecolor='black', 
                         alpha=alpha, linewidth=2)
        ax.add_patch(rect)
        
        # Draw direction arrow
        arrow_length = 3
        dx = arrow_length * np.cos(heading)
        dy = arrow_length * np.sin(heading)
        ax.arrow(pos[0], pos[1], dx, dy, 
                head_width=1.5, head_length=1, 
                fc=color, ec='black', alpha=alpha, linewidth=1.5)
        
        # Add label
        if label:
            ax.text(pos[0], pos[1] + width + 1, label, 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add speed indicator
        speed_text = f'{speed:.1f}'
        ax.text(pos[0], pos[1] - width - 1, speed_text, 
               ha='center', va='top', fontsize=8, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.5))
    
    def update_frame(self, frame):
        """Update function for animation"""
        # Clear axes
        self.ax_map.clear()
        self.ax_stats.clear()
        
        # Step environment
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        # Get vehicle data
        data = self.get_vehicle_data()
        
        if data is None:
            return
        
        ambulance = data['ambulance']
        npcs = data['npcs']
        
        # === LEFT PLOT: Bird's Eye View ===
        self.ax_map.set_title(f'Bird\'s Eye View - Step {self.step_count}/{self.duration}', 
                             fontsize=12, fontweight='bold')
        self.ax_map.set_xlabel('X Position (m)')
        self.ax_map.set_ylabel('Y Position (m)')
        self.ax_map.grid(True, alpha=0.3)
        
        # Draw detection zone around ambulance (50m radius)
        detection_circle = Circle(ambulance['pos'], 50, 
                                 fill=False, edgecolor='red', 
                                 linestyle='--', linewidth=2, alpha=0.5)
        self.ax_map.add_patch(detection_circle)
        
        # Draw warning zone (30m)
        warning_circle = Circle(ambulance['pos'], 30, 
                               fill=False, edgecolor='orange', 
                               linestyle='--', linewidth=1.5, alpha=0.3)
        self.ax_map.add_patch(warning_circle)
        
        # Draw ambulance
        self.draw_vehicle(self.ax_map, ambulance['pos'], ambulance['heading'], 
                         ambulance['speed'], is_ambulance=True)
        
        # Draw NPCs
        for npc in npcs:
            self.draw_vehicle(self.ax_map, npc['pos'], npc['heading'], 
                            npc['speed'], is_yielding=npc['yielding'])
            
            # Draw line to ambulance if close
            if npc['distance'] < 50:
                line_color = 'green' if npc['yielding'] else 'gray'
                line_alpha = 0.6 if npc['yielding'] else 0.2
                self.ax_map.plot([ambulance['pos'][0], npc['pos'][0]], 
                               [ambulance['pos'][1], npc['pos'][1]], 
                               color=line_color, linestyle=':', 
                               alpha=line_alpha, linewidth=1)
        
        # Set axis limits centered on ambulance
        margin = 80
        self.ax_map.set_xlim(ambulance['pos'][0] - margin, ambulance['pos'][0] + margin)
        self.ax_map.set_ylim(ambulance['pos'][1] - margin, ambulance['pos'][1] + margin)
        self.ax_map.set_aspect('equal')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='🚑 Ambulance'),
            Patch(facecolor='green', label='✓ Yielding NPC'),
            Patch(facecolor='blue', label='Normal NPC'),
            Patch(facecolor='none', edgecolor='red', linestyle='--', label='Detection Zone (50m)')
        ]
        self.ax_map.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # === RIGHT PLOT: Statistics ===
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')
        
        # Calculate statistics
        yielding_rate = (self.yielding_count / self.close_encounters * 100) if self.close_encounters > 0 else 0
        current_yielding = sum(1 for npc in npcs if npc['yielding'])
        current_close = sum(1 for npc in npcs if npc['distance'] < 50)
        
        # Build statistics text
        stats_text = f"""
╔════════════════════════════════════════╗
║     REAL-TIME YIELDING STATISTICS      ║
╚════════════════════════════════════════╝

⏱️  Simulation Step: {self.step_count} / {self.duration}

🚑 AMBULANCE STATUS
   Speed: {ambulance['speed']:.1f} m/s
   Position: ({ambulance['pos'][0]:.1f}, {ambulance['pos'][1]:.1f})

🚗 TRAFFIC STATUS
   Total NPCs: {len(npcs)}
   NPCs in Detection Zone (<50m): {current_close}
   Currently Yielding: {current_yielding}

📊 CUMULATIVE STATISTICS
   Total Close Encounters: {self.close_encounters}
   Total Yielding Instances: {self.yielding_count}
   
   YIELDING RATE: {yielding_rate:.1f}%

🎯 YIELDING DETECTION
   ✓ Distance < 50m from ambulance
   ✓ Speed < (Ambulance Speed - 2 m/s)

🎨 COLOR CODE
   🔴 Red = Ambulance (Emergency Vehicle)
   🟢 Green = Yielding NPC (Slowing down)
   🔵 Blue = Normal NPC (Regular traffic)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STATUS: {"✅ EXCELLENT!" if yielding_rate > 50 else "⚠️ MODERATE" if yielding_rate > 30 else "❌ POOR"}
"""
        
        # Draw statistics
        self.ax_stats.text(0.05, 0.95, stats_text, 
                          transform=self.ax_stats.transAxes,
                          fontsize=9, verticalalignment='top',
                          fontfamily='monospace',
                          bbox=dict(boxstyle='round,pad=1', 
                                  facecolor='wheat', alpha=0.8))
        
        # Add speed comparison bar chart
        if current_close > 0:
            # Mini speed chart
            ax_speed = self.fig.add_axes([0.55, 0.15, 0.35, 0.15])
            ax_speed.clear()
            
            speeds = [ambulance['speed']] + [npc['speed'] for npc in npcs if npc['distance'] < 50]
            labels = ['Ambulance'] + [f'NPC{i}' for i, npc in enumerate(npcs) if npc['distance'] < 50]
            colors = ['red'] + ['green' if npc['yielding'] else 'blue' 
                               for npc in npcs if npc['distance'] < 50]
            
            bars = ax_speed.barh(labels, speeds, color=colors, alpha=0.7)
            ax_speed.set_xlabel('Speed (m/s)', fontsize=8)
            ax_speed.set_title('Speed Comparison (Nearby Vehicles)', fontsize=9, fontweight='bold')
            ax_speed.tick_params(labelsize=7)
            ax_speed.grid(True, alpha=0.3, axis='x')
        
        # Check if done
        if terminated or truncated or self.step_count >= self.duration:
            print(f"\n✓ Simulation complete!")
            print(f"✓ Final Yielding Rate: {yielding_rate:.1f}%")
            self.ani.event_source.stop()
    
    def run_animation(self):
        """Run the live animation"""
        print(f"\n{'='*70}")
        print(f"🚑 Starting Live Visualization: {self.scenario_name}")
        print(f"{'='*70}\n")
        print("Creating real-time animation showing NPCs yielding to ambulance...")
        print(f"Duration: {self.duration} steps\n")
        print("Watch for:")
        print("  🔴 Red vehicle = Ambulance")
        print("  🟢 Green vehicles = NPCs yielding (slowing down)")
        print("  🔵 Blue vehicles = Normal NPCs")
        print("  ⭕ Red dashed circle = 50m detection zone")
        print("\n" + "="*70)
        
        # Reset environment
        self.reset_environment()
        
        # Create animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update_frame, 
            frames=self.duration,
            interval=100,  # 100ms between frames (10 fps)
            repeat=False
        )
        
        # Save animation
        output_dir = Path(__file__).parent.parent / 'output' / 'npc_yielding_test'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f'live_yielding_{self.scenario_name}.gif'
        print(f"\n💾 Saving animation to: {output_path}")
        print("   This may take a minute...")
        
        try:
            self.ani.save(str(output_path), writer='pillow', fps=10, dpi=100)
            print(f"✅ Animation saved successfully!")
        except Exception as e:
            print(f"⚠️  Could not save animation: {e}")
        
        plt.show()
        
        # Close environment
        self.env.close()
        
        # Final statistics
        yielding_rate = (self.yielding_count / self.close_encounters * 100) if self.close_encounters > 0 else 0
        print(f"\n{'='*70}")
        print(f"📊 FINAL STATISTICS")
        print(f"{'='*70}")
        print(f"Total Steps: {self.step_count}")
        print(f"Close Encounters: {self.close_encounters}")
        print(f"Yielding Instances: {self.yielding_count}")
        print(f"Yielding Rate: {yielding_rate:.1f}%")
        
        if yielding_rate > 50:
            print(f"\n✅ EXCELLENT! NPCs are yielding well!")
        elif yielding_rate > 30:
            print(f"\n⚠️  MODERATE: NPCs yielding somewhat")
        else:
            print(f"\n❌ POOR: NPCs not yielding enough")
        
        print(f"{'='*70}\n")


def main():
    """Main function"""
    print("\n" + "="*70)
    print("🎬 LIVE CARS IN ACTION - NPC YIELDING VISUALIZATION")
    print("="*70)
    print("\nThis will show you REAL-TIME animation of:")
    print("  • Ambulance (red) moving through traffic")
    print("  • NPCs (green) yielding by slowing down")
    print("  • Live statistics showing yielding rate")
    print("  • Bird's eye view of all vehicles")
    print("\nYou'll SEE the NPCs actually yielding!")
    print("="*70)
    
    # Choose scenario
    scenarios = [
        ("roundabout_single_lane", "Roundabout (97% yielding expected)"),
        ("highway_emergency_moderate", "Highway (53% yielding expected)"),
        ("merge_highway_entry", "Highway Merge (new scenario)"),
        ("intersection_t_junction", "T-Junction (new scenario)")
    ]
    
    print("\n📋 Available scenarios:")
    for i, (name, desc) in enumerate(scenarios, 1):
        print(f"   {i}. {desc}")
    
    # For now, use roundabout (best results)
    scenario_name = "roundabout_single_lane"
    duration = 80  # 80 steps animation
    
    print(f"\n🎯 Running: {scenario_name}")
    print(f"   Duration: {duration} steps (~8 seconds real-time)\n")
    
    input("Press ENTER to start the live visualization...")
    
    try:
        # Create visualizer
        viz = LiveYieldingVisualizer(scenario_name=scenario_name, duration=duration)
        
        # Run animation
        viz.run_animation()
        
        print("\n✅ Visualization complete!")
        print(f"📁 Animation saved in: collecting_ambulance_data/output/npc_yielding_test/")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Visualization interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
