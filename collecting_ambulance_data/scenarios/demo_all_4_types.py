#!/usr/bin/env python3
"""
Visual Demo: Show all 4 environment types in action
Runs one example from each: Highway, Roundabout, Intersection, and Merge
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gymnasium as gym
import highway_env
import numpy as np
import time

from collecting_ambulance_data.scenarios.ambulance_scenarios import get_all_ambulance_scenarios
from highway_datacollection.environments.factory import MultiAgentEnvFactory


def demo_environment(scenario_name: str, duration: int = 200):
    """
    Demonstrate a single scenario with highway-env rendering
    """
    print("\n" + "=" * 80)
    print(f"🎬 DEMONSTRATING: {scenario_name}")
    print("=" * 80 + "\n")
    
    # Get scenario config
    all_scenarios = get_all_ambulance_scenarios()
    scenario_config = all_scenarios[scenario_name]
    
    # Get environment type
    factory = MultiAgentEnvFactory()
    env_id = factory._get_env_id_for_scenario(scenario_name)
    
    # Print info
    print(f"📝 Description: {scenario_config.get('description', 'N/A')}")
    print(f"🏗️  Environment Type: {env_id}")
    print(f"🚦 Vehicles: {scenario_config.get('vehicles_count', 'N/A')}")
    print(f"🚥 Traffic Density: {scenario_config.get('traffic_density', 'N/A')}")
    print(f"⏱️  Duration: {duration} steps")
    
    # Print expected road geometry
    if 'highway' in env_id:
        print(f"🛣️  Expected: STRAIGHT MULTI-LANE HIGHWAY")
    elif 'roundabout' in env_id:
        print(f"⭕ Expected: CIRCULAR ROUNDABOUT")
    elif 'intersection' in env_id:
        print(f"🌐 Expected: CROSSING ROADS / INTERSECTION")
    elif 'merge' in env_id:
        print(f"🔀 Expected: HIGHWAY WITH MERGING LANES")
    
    print(f"\n▶️  Opening pygame window in 3 seconds...")
    print(f"   Watch for the road geometry!")
    print(f"   Press Ctrl+C to skip to next scenario\n")
    
    time.sleep(3)
    
    try:
        # Create environment using factory (this uses correct env_id!)
        env = factory.create_ambulance_env(
            scenario_name=scenario_name,
            obs_type='Kinematics',
            n_agents=4
        )
        
        # Modify for visualization
        env.unwrapped.config['render_mode'] = 'human'
        env.unwrapped.config['real_time_rendering'] = True
        env.unwrapped.config['show_trajectories'] = True
        env.unwrapped.config['scaling'] = 7.0
        
        # Create a new env with human rendering
        env.close()
        env = gym.make(env_id, render_mode='human')
        
        # Configure with scenario settings
        scenario_config_copy = scenario_config.copy()
        scenario_config_copy['duration'] = 1000  # Long duration
        scenario_config_copy['collision_reward'] = 0
        env.unwrapped.configure(scenario_config_copy)
        
        # Reset and run
        obs, info = env.reset()
        
        stats = {
            'steps': 0,
            'ambulance_speeds': [],
            'npc_speeds': []
        }
        
        print(f"✅ Simulation running...")
        print(f"   Look at the road shape!")
        
        for step in range(duration):
            # Random actions
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Collect stats
            if hasattr(env.unwrapped, 'road') and hasattr(env.unwrapped.road, 'vehicles'):
                vehicles = env.unwrapped.road.vehicles
                if len(vehicles) > 0:
                    ambulance = vehicles[0]
                    if hasattr(ambulance, 'velocity'):
                        stats['ambulance_speeds'].append(np.linalg.norm(ambulance.velocity))
                    
                    for npc in vehicles[1:]:
                        if hasattr(npc, 'velocity'):
                            stats['npc_speeds'].append(np.linalg.norm(npc.velocity))
            
            stats['steps'] += 1
            
            # Print progress
            if step % 50 == 0 and step > 0:
                avg_amb = np.mean(stats['ambulance_speeds'][-50:]) if stats['ambulance_speeds'] else 0
                avg_npc = np.mean(stats['npc_speeds'][-50:]) if stats['npc_speeds'] else 0
                print(f"   Step {step}/{duration} | Ambulance: {avg_amb:.1f} m/s | NPCs: {avg_npc:.1f} m/s")
            
            time.sleep(0.02)
            
            if terminated or truncated:
                break
        
        # Summary
        print(f"\n📊 Statistics:")
        print(f"   Steps: {stats['steps']}")
        if stats['ambulance_speeds'] and stats['npc_speeds']:
            print(f"   Avg Ambulance Speed: {np.mean(stats['ambulance_speeds']):.2f} m/s")
            print(f"   Avg NPC Speed: {np.mean(stats['npc_speeds']):.2f} m/s")
            print(f"   Speed Difference: {np.mean(stats['ambulance_speeds']) - np.mean(stats['npc_speeds']):.2f} m/s")
        
        env.close()
        
    except KeyboardInterrupt:
        print("\n⏭️  Skipping to next scenario...")
        try:
            env.close()
        except:
            pass
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Run demo of all 4 environment types
    """
    print("\n" + "=" * 80)
    print("🎬 VISUAL DEMO: ALL 4 ENVIRONMENT TYPES")
    print("=" * 80)
    print("\nThis will demonstrate:")
    print("  1. 🛣️  Highway (straight roads)")
    print("  2. ⭕ Roundabout (circular roads)")
    print("  3. 🌐 Intersection (crossing roads)")
    print("  4. 🔀 Merge (merging lanes)")
    print("\nEach scenario runs for ~20 seconds")
    print("Press Ctrl+C during a scenario to skip to the next")
    print("\n" + "=" * 80)
    
    input("\nPress Enter to start the demo...")
    
    # Demo scenarios (one from each type)
    demo_scenarios = [
        ('highway_rush_hour', 'Highway'),
        ('roundabout_single_lane', 'Roundabout'),
        ('intersection_four_way', 'Intersection'),
        ('merge_highway_entry', 'Merge')
    ]
    
    for i, (scenario_name, env_type) in enumerate(demo_scenarios, 1):
        print(f"\n{'╔' + '═' * 78 + '╗'}")
        print(f"║ DEMO {i}/4: {env_type.upper():<70} ║")
        print(f"{'╚' + '═' * 78 + '╝'}")
        
        demo_environment(scenario_name, duration=200)
        
        if i < len(demo_scenarios):
            print(f"\n⏸️  Pausing for 3 seconds before next demo...")
            time.sleep(3)
    
    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE!")
    print("=" * 80)
    print("\nYou just saw all 4 environment types:")
    print("  ✅ Highway: Straight multi-lane roads")
    print("  ✅ Roundabout: Circular roads")
    print("  ✅ Intersection: Crossing roads")
    print("  ✅ Merge: Merging lanes")
    print("\nAll 30 scenarios use these 4 environment types!")
    print("When you run data collection, you'll get all this diversity!")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
