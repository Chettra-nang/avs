#!/usr/bin/env python3
"""
Real-time Visualization of Corner/Intersection and Roundabout (Circle) Scenarios
Shows actual highway-env rendering to verify these scenarios work correctly
"""

import gymnasium as gym
import highway_env  # IMPORTANT: Import to register highway-env environments
import numpy as np
import argparse
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scenarios.ambulance_scenarios import get_all_ambulance_scenarios


def visualize_scenario(scenario_name: str, duration: int = 500, speed: float = 1.0):
    """
    Visualize a specific scenario using highway-env's native rendering
    
    Args:
        scenario_name: Name of the scenario to visualize
        duration: Number of simulation steps
        speed: Simulation speed multiplier (1.0 = normal, 0.5 = half speed)
    """
    print(f"\n{'='*80}")
    print(f"🚗 Visualizing Scenario: {scenario_name}")
    print(f"{'='*80}\n")
    
    # Get all scenarios (returns a dictionary)
    all_scenarios = get_all_ambulance_scenarios()
    
    # Find the requested scenario
    if scenario_name not in all_scenarios:
        print(f"❌ Scenario '{scenario_name}' not found!")
        print("\n📋 Available scenarios:")
        for i, name in enumerate(all_scenarios.keys(), 1):
            print(f"   {i}. {name}")
        return
    
    scenario_config = all_scenarios[scenario_name]
    
    # Print scenario info
    print(f"� Description: {scenario_config.get('description', 'Unknown')}")
    print(f"🚦 Vehicles: {scenario_config.get('vehicles_count', 'Unknown')}")
    print(f"� Traffic Density: {scenario_config.get('traffic_density', 'Unknown')}")
    print(f"🏃 Duration: {scenario_config.get('duration', 'Unknown')} steps")
    print(f"⏱️  Simulation Speed: {speed}x")
    print(f"\n🎯 Looking for NPC yielding behavior...")
    print(f"   - NPCs should slow down when ambulance approaches")
    print(f"   - Watch for speed differences and lane changes")
    print(f"   - Red vehicle at top = Ambulance (Agent 0)")
    print(f"   - White vehicles = NPCs (should yield)")
    print(f"\n▶️  Starting simulation... (Press Ctrl+C to stop)\n")
    
    # Modify config for better visualization
    scenario_config["render_mode"] = "human"
    scenario_config["real_time_rendering"] = True
    scenario_config["show_trajectories"] = True
    scenario_config["duration"] = 1000  # Long duration
    scenario_config["screen_width"] = 1200
    scenario_config["screen_height"] = 800
    scenario_config["scaling"] = 7.0
    scenario_config["collision_reward"] = 0  # Don't end on collision
    
    # Create environment with rendering
    try:
        # Select the appropriate environment based on scenario type
        if 'roundabout' in scenario_name.lower():
            env_id = 'roundabout-v0'
            print(f"🎯 Using roundabout-v0 environment for circular road geometry")
        elif 'intersection' in scenario_name.lower() or 'corner' in scenario_name.lower():
            env_id = 'intersection-v0'
            print(f"🎯 Using intersection-v0 environment for crossing roads")
        elif 'merge' in scenario_name.lower():
            env_id = 'merge-v0'
            print(f"🎯 Using merge-v0 environment for highway merge")
        else:
            env_id = 'highway-v0'
            print(f"🎯 Using highway-v0 environment")
        
        env = gym.make(env_id, render_mode='human')
        
        # Configure the environment with our scenario config
        env.unwrapped.configure(scenario_config)
        
        # Reset environment
        obs, info = env.reset()
        
        # Statistics
        step_count = 0
        ambulance_speeds = []
        npc_speeds = []
        yielding_events = 0
        
        # Simulation loop
        for step in range(duration):
            # Simple action: continue straight
            action = 1  # IDLE/continue
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render (pygame window updates automatically)
            env.render()
            
            # Analyze yielding behavior
            if hasattr(env.unwrapped, 'road') and hasattr(env.unwrapped.road, 'vehicles'):
                vehicles = env.unwrapped.road.vehicles
                
                if len(vehicles) > 0:
                    # Ambulance is typically the controlled vehicle (first)
                    ambulance = vehicles[0]
                    ambulance_speed = np.linalg.norm(ambulance.velocity) if hasattr(ambulance, 'velocity') else 0
                    ambulance_speeds.append(ambulance_speed)
                    
                    # Check NPCs
                    for npc in vehicles[1:]:
                        if hasattr(npc, 'velocity'):
                            npc_speed = np.linalg.norm(npc.velocity)
                            npc_speeds.append(npc_speed)
                            
                            # Check if NPC is yielding (significantly slower than ambulance)
                            if ambulance_speed > npc_speed + 5.0:  # 5 m/s difference
                                yielding_events += 1
            
            step_count += 1
            
            # Print progress every 50 steps
            if step % 50 == 0 and step > 0:
                avg_ambulance_speed = np.mean(ambulance_speeds[-50:]) if ambulance_speeds else 0
                avg_npc_speed = np.mean(npc_speeds[-50:]) if npc_speeds else 0
                print(f"Step {step}/{duration} | "
                      f"Ambulance: {avg_ambulance_speed:.1f} m/s | "
                      f"NPCs: {avg_npc_speed:.1f} m/s | "
                      f"Yielding Events: {yielding_events}")
            
            # Control simulation speed
            time.sleep(0.02 / speed)  # Adjust frame delay
            
            if terminated or truncated:
                print(f"\n⚠️  Episode ended at step {step}")
                break
        
        # Final statistics
        print(f"\n{'='*80}")
        print(f"📊 SIMULATION STATISTICS")
        print(f"{'='*80}")
        print(f"✅ Total Steps: {step_count}")
        print(f"🚑 Average Ambulance Speed: {np.mean(ambulance_speeds):.2f} m/s")
        print(f"🚗 Average NPC Speed: {np.mean(npc_speeds):.2f} m/s")
        print(f"👀 Yielding Events Detected: {yielding_events}")
        
        if ambulance_speeds and npc_speeds:
            speed_diff = np.mean(ambulance_speeds) - np.mean(npc_speeds)
            print(f"⚡ Speed Difference: {speed_diff:.2f} m/s")
            if speed_diff > 3.0:
                print(f"✅ NPCs ARE YIELDING! (Ambulance is {speed_diff:.1f} m/s faster)")
            else:
                print(f"⚠️  Limited yielding detected (only {speed_diff:.1f} m/s difference)")
        
        env.close()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Simulation stopped by user")
        env.close()
    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()


def show_all_corner_and_circle_scenarios():
    """Display all corner/intersection and circle/roundabout scenarios"""
    all_scenarios = get_all_ambulance_scenarios()
    
    print(f"\n{'='*80}")
    print(f"🌐 CORNER/INTERSECTION SCENARIOS (5 scenarios)")
    print(f"{'='*80}\n")
    
    corner_scenarios = {name: config for name, config in all_scenarios.items() 
                       if 'corner' in name.lower() or 'intersection' in name.lower()}
    for i, (name, config) in enumerate(corner_scenarios.items(), 1):
        print(f"{i}. {name}")
        print(f"   📍 Type: {config.get('id', 'Unknown')}")
        print(f"   🚦 Vehicles: {config.get('vehicles_count', 'Unknown')}")
        print(f"   ⏱️  Duration: {config.get('duration', 'Unknown')} steps")
        print(f"   📝 Description: {config.get('description', 'N/A')}")
        print()
    
    print(f"\n{'='*80}")
    print(f"⭕ ROUNDABOUT/CIRCLE SCENARIOS (3 scenarios)")
    print(f"{'='*80}\n")
    
    circle_scenarios = {name: config for name, config in all_scenarios.items() 
                       if 'roundabout' in name.lower() or 'circle' in name.lower()}
    for i, (name, config) in enumerate(circle_scenarios.items(), 1):
        print(f"{i}. {name}")
        print(f"   📍 Type: {config.get('id', 'Unknown')}")
        print(f"   🚦 Vehicles: {config.get('vehicles_count', 'Unknown')}")
        print(f"   ⏱️  Duration: {config.get('duration', 'Unknown')} steps")
        print(f"   📝 Description: {config.get('description', 'N/A')}")
        print()
    
    print(f"{'='*80}")
    print(f"📊 Total: {len(corner_scenarios)} corner/intersection + {len(circle_scenarios)} roundabout scenarios")
    print(f"{'='*80}\n")


def demo_all_scenarios(duration_per_scenario: int = 200, speed: float = 1.5):
    """
    Demo all corner and circle scenarios sequentially
    """
    all_scenarios = get_all_ambulance_scenarios()
    
    # Filter corner and circle scenarios
    corner_circle_scenarios = {
        name: config for name, config in all_scenarios.items()
        if 'corner' in name.lower() 
        or 'intersection' in name.lower() 
        or 'roundabout' in name.lower() 
        or 'circle' in name.lower()
    }
    
    print(f"\n{'='*80}")
    print(f"🎬 DEMO MODE: Running {len(corner_circle_scenarios)} scenarios")
    print(f"{'='*80}\n")
    print(f"⏱️  Duration per scenario: {duration_per_scenario} steps")
    print(f"🏃 Speed: {speed}x")
    print(f"\n▶️  Starting demo...\n")
    
    for i, scenario_name in enumerate(corner_circle_scenarios.keys(), 1):
        print(f"\n{'─'*80}")
        print(f"📍 Scenario {i}/{len(corner_circle_scenarios)}")
        print(f"{'─'*80}")
        
        visualize_scenario(scenario_name, duration=duration_per_scenario, speed=speed)
        
        if i < len(corner_circle_scenarios):
            print(f"\n⏸️  Next scenario in 3 seconds...")
            time.sleep(3)
    
    print(f"\n{'='*80}")
    print(f"✅ DEMO COMPLETE! All {len(corner_circle_scenarios)} scenarios visualized")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Corner/Intersection and Roundabout scenarios in real-time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all corner and circle scenarios
  python watch_corner_and_circle.py --list

  # Visualize a specific roundabout scenario
  python watch_corner_and_circle.py --scenario roundabout_single_lane --duration 500

  # Visualize a corner scenario at half speed
  python watch_corner_and_circle.py --scenario corner_moderate_crossing --speed 0.5

  # Demo all corner and circle scenarios (200 steps each)
  python watch_corner_and_circle.py --demo --duration 200 --speed 1.5

  # Quick demo mode (shorter duration)
  python watch_corner_and_circle.py --demo --duration 100 --speed 2.0

Available Corner/Intersection Scenarios:
  - corner_tight_turn
  - corner_moderate_crossing
  - corner_wide_intersection
  - corner_t_junction
  - corner_complex_intersection

Available Roundabout/Circle Scenarios:
  - roundabout_single_lane
  - roundabout_heavy_traffic
  - roundabout_multi_exit
        """
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        help='Name of the scenario to visualize (e.g., roundabout_single_lane, corner_tight_turn)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=500,
        help='Number of simulation steps (default: 500)'
    )
    
    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Simulation speed multiplier (default: 1.0, try 0.5 for slow-motion, 2.0 for fast)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available corner and circle scenarios'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Demo all corner and circle scenarios sequentially'
    )
    
    args = parser.parse_args()
    
    # Handle list mode
    if args.list:
        show_all_corner_and_circle_scenarios()
        return
    
    # Handle demo mode
    if args.demo:
        demo_all_scenarios(duration_per_scenario=args.duration, speed=args.speed)
        return
    
    # Handle single scenario mode
    if args.scenario:
        visualize_scenario(args.scenario, duration=args.duration, speed=args.speed)
    else:
        # No arguments: show help and list scenarios
        parser.print_help()
        print("\n")
        show_all_corner_and_circle_scenarios()


if __name__ == "__main__":
    main()
