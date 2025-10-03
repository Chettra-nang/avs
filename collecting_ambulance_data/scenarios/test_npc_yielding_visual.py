#!/usr/bin/env python3
"""
Visual Test: Verify NPC Yielding Behavior

This script:
1. Runs a short simulation with one of the new scenarios
2. Collects data with visualization
3. Analyzes if NPCs are yielding to the ambulance
4. Shows visual proof of yielding behavior

Run this BEFORE doing the big data collection to verify NPCs actually yield.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import time

# Import highway-env
import gymnasium as gym
import highway_env

from collecting_ambulance_data.scenarios.ambulance_scenarios import get_scenario_by_name


class NPCYieldingVisualTest:
    """Visual test to verify NPC yielding behavior"""
    
    def __init__(self, scenario_name="roundabout_single_lane"):
        self.scenario_name = scenario_name
        self.config = get_scenario_by_name(scenario_name)
        self.yielding_events = []
        self.vehicle_data = []
        
    def run_simulation(self, duration=30):
        """Run simulation and collect data"""
        print(f"\n{'='*70}")
        print(f"🚑 Testing NPC Yielding: {self.scenario_name}")
        print(f"{'='*70}\n")
        
        # Create environment
        print("📦 Creating environment...")
        env = gym.make('highway-v0', render_mode='rgb_array')
        env.unwrapped.configure(self.config)
        
        # Reset environment
        print("🔄 Resetting environment...")
        obs, info = env.reset()
        
        print(f"✓ Environment ready")
        print(f"✓ Vehicles: {len(env.unwrapped.road.vehicles)}")
        print(f"✓ Duration: {duration} steps\n")
        
        print("🎬 Running simulation with visualization...")
        
        # Run simulation
        for step in range(duration):
            # Get ambulance (first controlled vehicle)
            ambulance = env.unwrapped.road.vehicles[0]
            
            # Collect data about all vehicles
            frame_data = {
                'step': step,
                'ambulance_pos': ambulance.position.copy(),
                'ambulance_speed': ambulance.speed,
                'vehicles': []
            }
            
            # Check each vehicle
            for i, vehicle in enumerate(env.unwrapped.road.vehicles[1:], 1):  # Skip ambulance
                veh_data = {
                    'id': i,
                    'pos': vehicle.position.copy(),
                    'speed': vehicle.speed,
                    'lane': vehicle.lane_index[2] if hasattr(vehicle, 'lane_index') else 0
                }
                
                # Calculate distance to ambulance
                distance = np.linalg.norm(vehicle.position - ambulance.position)
                veh_data['distance_to_ambulance'] = distance
                
                # Detect yielding behavior
                if distance < 50:  # Within 50m of ambulance
                    # Check if vehicle is slowing down or moving away
                    if vehicle.speed < ambulance.speed - 2:  # Significantly slower
                        veh_data['yielding'] = True
                        self.yielding_events.append({
                            'step': step,
                            'vehicle_id': i,
                            'distance': distance,
                            'speed_diff': ambulance.speed - vehicle.speed
                        })
                    else:
                        veh_data['yielding'] = False
                else:
                    veh_data['yielding'] = False
                
                frame_data['vehicles'].append(veh_data)
            
            self.vehicle_data.append(frame_data)
            
            # Random action for ambulance (just for testing)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render frame
            if step % 5 == 0:
                env.render()
                print(f"  Step {step}/{duration} - Ambulance speed: {ambulance.speed:.1f} m/s", end='\r')
            
            if terminated or truncated:
                break
        
        print(f"\n\n✓ Simulation complete!")
        env.close()
        
    def analyze_yielding(self):
        """Analyze yielding behavior from collected data"""
        print(f"\n{'='*70}")
        print("📊 ANALYZING NPC YIELDING BEHAVIOR")
        print(f"{'='*70}\n")
        
        if not self.vehicle_data:
            print("❌ No data collected!")
            return
        
        # Count yielding events
        total_close_encounters = 0
        yielding_count = 0
        
        for frame in self.vehicle_data:
            for veh in frame['vehicles']:
                if veh['distance_to_ambulance'] < 50:  # Within 50m
                    total_close_encounters += 1
                    if veh['yielding']:
                        yielding_count += 1
        
        yielding_rate = (yielding_count / total_close_encounters * 100) if total_close_encounters > 0 else 0
        
        print(f"📈 Results:")
        print(f"   Total frames analyzed: {len(self.vehicle_data)}")
        print(f"   Close encounters (<50m): {total_close_encounters}")
        print(f"   Yielding instances: {yielding_count}")
        print(f"   Unique yielding events: {len(self.yielding_events)}")
        print(f"\n   🎯 YIELDING RATE: {yielding_rate:.1f}%\n")
        
        # Interpretation
        if yielding_rate > 50:
            print("   ✅ EXCELLENT! NPCs are yielding well (>50%)")
            print("   ✅ Ready for full data collection!")
        elif yielding_rate > 30:
            print("   ⚠️  MODERATE: NPCs are yielding somewhat (30-50%)")
            print("   ⚠️  Consider tuning IDM parameters further")
        else:
            print("   ❌ POOR: NPCs are NOT yielding enough (<30%)")
            print("   ❌ Need to implement custom vehicle behavior class")
        
        return yielding_rate
        
    def visualize_yielding(self):
        """Create visualization of yielding behavior"""
        print(f"\n{'='*70}")
        print("🎨 CREATING VISUALIZATION")
        print(f"{'='*70}\n")
        
        if not self.vehicle_data:
            print("❌ No data to visualize!")
            return
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'NPC Yielding Analysis: {self.scenario_name}', fontsize=16, fontweight='bold')
        
        # Subplot 1: Distance to ambulance over time
        ax1 = axes[0, 0]
        steps = [f['step'] for f in self.vehicle_data]
        
        # Track a few vehicles
        vehicle_ids = set()
        for frame in self.vehicle_data[:10]:  # First 10 frames
            for veh in frame['vehicles'][:5]:  # First 5 vehicles
                vehicle_ids.add(veh['id'])
        
        for vid in list(vehicle_ids)[:5]:  # Plot up to 5 vehicles
            distances = []
            for frame in self.vehicle_data:
                for veh in frame['vehicles']:
                    if veh['id'] == vid:
                        distances.append(veh['distance_to_ambulance'])
                        break
                else:
                    distances.append(np.nan)
            ax1.plot(steps, distances, label=f'Vehicle {vid}', alpha=0.7)
        
        ax1.axhline(y=50, color='r', linestyle='--', label='Detection Range (50m)')
        ax1.set_xlabel('Simulation Step')
        ax1.set_ylabel('Distance to Ambulance (m)')
        ax1.set_title('Vehicle Distances to Ambulance Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Speed differences
        ax2 = axes[0, 1]
        ambulance_speeds = [f['ambulance_speed'] for f in self.vehicle_data]
        avg_vehicle_speeds = []
        
        for frame in self.vehicle_data:
            if frame['vehicles']:
                avg_speed = np.mean([v['speed'] for v in frame['vehicles']])
                avg_vehicle_speeds.append(avg_speed)
            else:
                avg_vehicle_speeds.append(np.nan)
        
        ax2.plot(steps, ambulance_speeds, label='Ambulance Speed', linewidth=2, color='red')
        ax2.plot(steps, avg_vehicle_speeds, label='Avg NPC Speed', linewidth=2, color='blue', alpha=0.7)
        ax2.set_xlabel('Simulation Step')
        ax2.set_ylabel('Speed (m/s)')
        ax2.set_title('Speed Comparison: Ambulance vs NPCs')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Yielding events timeline
        ax3 = axes[1, 0]
        if self.yielding_events:
            event_steps = [e['step'] for e in self.yielding_events]
            event_distances = [e['distance'] for e in self.yielding_events]
            event_speeds = [e['speed_diff'] for e in self.yielding_events]
            
            scatter = ax3.scatter(event_steps, event_distances, 
                                c=event_speeds, s=100, 
                                cmap='RdYlGn', alpha=0.6,
                                edgecolors='black', linewidth=1)
            ax3.set_xlabel('Simulation Step')
            ax3.set_ylabel('Distance to Ambulance (m)')
            ax3.set_title(f'Yielding Events (n={len(self.yielding_events)})')
            ax3.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax3, label='Speed Difference (m/s)')
        else:
            ax3.text(0.5, 0.5, 'No yielding events detected', 
                    ha='center', va='center', fontsize=14, color='red')
            ax3.set_title('Yielding Events')
        
        # Subplot 4: Yielding statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate statistics
        total_close = sum(1 for f in self.vehicle_data for v in f['vehicles'] 
                         if v['distance_to_ambulance'] < 50)
        yielding_count = len(self.yielding_events)
        yielding_rate = (yielding_count / total_close * 100) if total_close > 0 else 0
        
        stats_text = f"""
        YIELDING STATISTICS
        {'='*40}
        
        Scenario: {self.scenario_name}
        Simulation Steps: {len(self.vehicle_data)}
        
        Close Encounters (<50m): {total_close}
        Yielding Instances: {yielding_count}
        
        YIELDING RATE: {yielding_rate:.1f}%
        
        IDM Parameters:
        • TIME_WANTED: 2.5s
        • DISTANCE_WANTED: 8.0m
        • COMFORT_ACC_MIN: -3.0 m/s²
        
        Status:
        """
        
        if yielding_rate > 50:
            stats_text += "✅ EXCELLENT - Ready for collection!"
        elif yielding_rate > 30:
            stats_text += "⚠️  MODERATE - Consider tuning"
        else:
            stats_text += "❌ POOR - Needs custom behavior"
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        output_dir = Path(__file__).parent.parent / 'output' / 'npc_yielding_test'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f'yielding_test_{self.scenario_name}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved: {output_path}")
        
        plt.show()
        print("✓ Visualization displayed")


def main():
    """Main test function"""
    print("\n" + "="*70)
    print("🚑 NPC YIELDING VISUAL TEST")
    print("="*70)
    print("\nThis test will:")
    print("  1. Run a short simulation with a new scenario")
    print("  2. Track NPC behavior around the ambulance")
    print("  3. Detect and analyze yielding events")
    print("  4. Create visualizations to verify yielding")
    print("\nThis helps verify NPCs are yielding BEFORE big data collection!")
    print("="*70)
    
    # Test scenarios
    test_scenarios = [
        "roundabout_single_lane",  # New roundabout scenario
        "highway_emergency_moderate",  # Original highway scenario
    ]
    
    print(f"\n📋 Will test {len(test_scenarios)} scenarios:")
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"   {i}. {scenario}")
    
    input("\nPress ENTER to start the test...")
    
    results = {}
    
    for scenario in test_scenarios:
        print(f"\n\n{'#'*70}")
        print(f"# Testing: {scenario}")
        print(f"{'#'*70}\n")
        
        try:
            # Create test
            test = NPCYieldingVisualTest(scenario_name=scenario)
            
            # Run simulation
            test.run_simulation(duration=50)
            
            # Analyze
            yielding_rate = test.analyze_yielding()
            
            # Visualize
            test.visualize_yielding()
            
            results[scenario] = {
                'success': True,
                'yielding_rate': yielding_rate
            }
            
            print(f"\n✅ Test completed for {scenario}")
            time.sleep(2)
            
        except Exception as e:
            print(f"\n❌ Error testing {scenario}: {e}")
            import traceback
            traceback.print_exc()
            results[scenario] = {
                'success': False,
                'error': str(e)
            }
    
    # Final summary
    print("\n\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70 + "\n")
    
    for scenario, result in results.items():
        if result['success']:
            rate = result['yielding_rate']
            status = "✅" if rate > 50 else "⚠️" if rate > 30 else "❌"
            print(f"{status} {scenario:40s} : {rate:.1f}% yielding")
        else:
            print(f"❌ {scenario:40s} : ERROR")
    
    print("\n" + "="*70)
    
    # Overall recommendation
    all_good = all(r['success'] and r.get('yielding_rate', 0) > 50 for r in results.values())
    some_good = any(r['success'] and r.get('yielding_rate', 0) > 50 for r in results.values())
    
    print("\n🎯 RECOMMENDATION:\n")
    if all_good:
        print("✅ ALL SCENARIOS SHOW GOOD YIELDING (>50%)")
        print("✅ You can proceed with full data collection!")
        print("\nNext step:")
        print("  cd ../examples && python parallel_ambulance_collection.py --scenarios all")
    elif some_good:
        print("⚠️  MIXED RESULTS - Some scenarios yield well, others don't")
        print("⚠️  Consider collecting only from scenarios with >50% yielding")
        print("⚠️  Or tune IDM parameters further for problematic scenarios")
    else:
        print("❌ NPCs ARE NOT YIELDING EFFECTIVELY (<50%)")
        print("❌ Current IDM parameters may not be sufficient")
        print("\nOptions:")
        print("  1. Increase IDM parameters further (TIME_WANTED=3.5, DISTANCE_WANTED=12.0)")
        print("  2. Implement custom IDMVehicle subclass with emergency awareness")
        print("  3. Add explicit lane-clearing logic in environment")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
