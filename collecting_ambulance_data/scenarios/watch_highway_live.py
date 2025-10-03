#!/usr/bin/env python3
"""
Real Highway-Env Visualization - Watch Cars in Action

This shows the ACTUAL highway-env rendering window like in Farama docs.
You'll see the real simulation with cars moving on the highway!

Press 'q' to quit early or wait for the simulation to finish.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import gymnasium as gym
import highway_env
import time

from collecting_ambulance_data.scenarios.ambulance_scenarios import get_scenario_by_name


def run_highway_simulation(scenario_name="highway_emergency_moderate", duration=200):
    """
    Run highway-env simulation with REAL rendering
    
    Args:
        scenario_name: Which scenario to visualize
        duration: Number of simulation steps
    """
    print("\n" + "="*70)
    print("🚗 HIGHWAY-ENV REAL SIMULATION - CARS IN ACTION")
    print("="*70)
    print(f"\nScenario: {scenario_name}")
    print(f"Duration: {duration} steps (~20 seconds)")
    print("\nWhat you'll see:")
    print("  🔴 Red vehicle at top = Ambulance (Agent 0)")
    print("  🔵 Blue vehicles = Other controlled agents")
    print("  ⚪ White vehicles = NPCs (should yield to ambulance)")
    print("  📊 Top-left: Observation views for each agent")
    print("\nControls:")
    print("  • Just watch the simulation")
    print("  • NPCs will yield by slowing down")
    print("  • Close window or Ctrl+C to stop")
    print("\n" + "="*70 + "\n")
    
    # Get scenario config
    config = get_scenario_by_name(scenario_name)
    
    # Modify for better visualization
    config["render_mode"] = "human"  # Real-time rendering
    config["real_time_rendering"] = True  # Match real-world speed
    config["show_trajectories"] = True  # Show paths
    config["duration"] = 1000  # Long duration so it doesn't end early
    
    # Make ambulance more visible
    config["screen_width"] = 1200
    config["screen_height"] = 400
    config["scaling"] = 7.0  # Zoom in more
    
    # Disable collision termination
    config["collision_reward"] = 0  # Don't end on collision
    
    print("🎬 Opening highway-env window...")
    print("   (If window doesn't appear, check your display settings)\n")
    
    try:
        # Create environment with human rendering
        env = gym.make('highway-v0', render_mode='human')
        env.unwrapped.configure(config)
        
        # Reset environment
        obs, info = env.reset()
        
        print("✅ Simulation started!")
        print("   Watch for NPCs slowing down near the ambulance\n")
        
        # Statistics tracking
        step_count = 0
        yielding_detected = 0
        close_encounters = 0
        
        # Run simulation
        for step in range(duration):
            # Get ambulance and NPCs
            ambulance = env.unwrapped.road.vehicles[0]
            
            # Check for yielding
            for vehicle in env.unwrapped.road.vehicles[1:]:
                distance = np.linalg.norm(vehicle.position - ambulance.position)
                if distance < 50:  # Within 50m
                    close_encounters += 1
                    if vehicle.speed < ambulance.speed - 2:  # Slower than ambulance
                        yielding_detected += 1
            
            # Random action (ambulance is learning)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render (this shows the visualization)
            env.render()
            
            step_count += 1
            
            # Print statistics every 50 steps
            if step % 50 == 0 and step > 0:
                yielding_rate = (yielding_detected / close_encounters * 100) if close_encounters > 0 else 0
                print(f"  Step {step:3d}/{duration} | "
                      f"Close encounters: {close_encounters:3d} | "
                      f"Yielding: {yielding_detected:3d} ({yielding_rate:.1f}%)")
            
            # Check if done
            if terminated or truncated:
                print(f"\n⚠️  Episode ended at step {step}")
                break
            
            # Small delay for visualization
            time.sleep(0.01)
        
        # Final statistics
        print("\n" + "="*70)
        print("📊 SIMULATION COMPLETE")
        print("="*70)
        print(f"Total steps: {step_count}")
        print(f"Close encounters: {close_encounters}")
        print(f"Yielding instances: {yielding_detected}")
        
        if close_encounters > 0:
            final_rate = yielding_detected / close_encounters * 100
            print(f"Yielding rate: {final_rate:.1f}%")
            
            if final_rate > 40:
                print("\n✅ EXCELLENT! NPCs are yielding to the ambulance!")
            elif final_rate > 25:
                print("\n⚠️  MODERATE: Some NPCs are yielding")
            else:
                print("\n❌ LOW: NPCs not yielding much")
        
        print("="*70 + "\n")
        
        # Close environment
        env.close()
        
        print("✅ Simulation window closed")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
        env.close()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Watch highway-env simulation with real rendering")
    parser.add_argument('--scenario', type=str, default='highway_emergency_moderate',
                       help='Scenario name (default: highway_emergency_moderate)')
    parser.add_argument('--duration', type=int, default=200,
                       help='Simulation duration in steps (default: 200)')
    parser.add_argument('--list', action='store_true',
                       help='List available scenarios')
    
    args = parser.parse_args()
    
    if args.list:
        print("\n📋 Available scenarios:")
        print("\nHighway scenarios:")
        print("  • highway_emergency_light")
        print("  • highway_emergency_moderate")
        print("  • highway_emergency_dense")
        print("  • highway_lane_closure")
        print("  • highway_rush_hour")
        print("\nNew scenarios:")
        print("  • roundabout_single_lane")
        print("  • intersection_t_junction")
        print("  • merge_highway_entry")
        print("  • urban_mixed_complex")
        print("\nUsage:")
        print("  python watch_highway_live.py --scenario highway_emergency_moderate")
        print("  python watch_highway_live.py --scenario roundabout_single_lane --duration 300")
        return
    
    # Run simulation
    run_highway_simulation(args.scenario, args.duration)


if __name__ == "__main__":
    main()
