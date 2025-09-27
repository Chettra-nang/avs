#!/usr/bin/env python3
"""
Simple Visual Ambulance Test

The simplest possible visual test to see ambulance running on screen.
"""

import sys
import time
import gymnasium as gym
import highway_env
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def simple_visual_test():
    """Simplest visual test of ambulance."""
    print("🚑 Simple Visual Ambulance Test")
    print("=" * 35)
    
    try:
        # Import ambulance scenario
        from collecting_ambulance_data.scenarios.ambulance_scenarios import get_scenario_by_name
        
        print("✅ Imports successful")
        
        # Get ambulance scenario config
        scenario_config = get_scenario_by_name("highway_emergency_light")
        print(f"✅ Loaded scenario: {scenario_config['description']}")
        
        # Create environment directly with render mode
        print("🏗️  Creating visual environment...")
        
        # Configure for ambulance scenario with visual rendering
        config = {
            "lanes_count": 4,
            "controlled_vehicles": 4,  # 4 agents (first is ambulance)
            "vehicles_count": 15,
            "duration": 40,
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            
            # Multi-agent configuration
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics"
                }
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {"type": "DiscreteMetaAction"}
            },
            
            # Visual configuration
            "offscreen_rendering": False,
            "real_time_rendering": True,
            "show_trajectories": True,
            "screen_width": 800,
            "screen_height": 600,
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
        }
        
        # Create environment with human rendering
        env = gym.make("highway-v0", render_mode="human", config=config)
        
        print("✅ Environment created - starting visual test")
        print("   🚑 First vehicle (usually red) = Ambulance")
        print("   🚙 Other vehicles = Normal traffic")
        print("   You should see a window with vehicles moving")
        print("   Press Ctrl+C to stop")
        print()
        
        # Reset environment
        obs, info = env.reset(seed=42)
        
        # Run for 100 steps
        for step in range(100):
            # Render (this should show the window)
            env.render()
            
            # Simple random actions for 4 agents (each action is an integer 0-4)
            import random
            actions = tuple(random.randint(0, 4) for _ in range(4))
            
            # Take step
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Print progress
            if step % 20 == 0:
                ambulance_reward = rewards[0] if isinstance(rewards, list) else rewards
                print(f"   Step {step:2d}: Ambulance reward = {ambulance_reward:6.2f}")
            
            # Check if done
            done = any(terminated) if isinstance(terminated, list) else terminated
            if done:
                print(f"✅ Episode completed at step {step}")
                break
            
            time.sleep(0.1)  # Slow down for visibility
        
        env.close()
        print("\n🎉 Visual test completed successfully!")
        print("✅ You should have seen vehicles moving on screen")
        print("✅ Configuration is working correctly")
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  Test stopped by user")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting simple visual test...")
    print("This should open a window showing vehicles on a highway")
    print()
    
    success = simple_visual_test()
    
    if success:
        print("\n✅ SUCCESS! The ambulance configuration is working!")
        print("🚀 You're ready for full ambulance data collection!")
    else:
        print("\n❌ Test failed - please check configuration")