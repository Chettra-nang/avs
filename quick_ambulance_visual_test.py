#!/usr/bin/env python3
"""
Quick Visual Ambulance Test

A minimal script to quickly see the ambulance running on screen.
This is the simplest possible test to verify configuration.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def quick_visual_test():
    """Quick visual test of ambulance configuration."""
    print("🚑 Quick Ambulance Visual Test")
    print("=" * 35)
    
    try:
        # Import required components
        from collecting_ambulance_data.scenarios.ambulance_scenarios import get_scenario_by_name
        from highway_datacollection.environments.factory import MultiAgentEnvFactory
        
        print("✅ Imports successful")
        
        # Get a simple scenario
        scenario_config = get_scenario_by_name("highway_emergency_light")
        print(f"✅ Loaded scenario: {scenario_config['description']}")
        
        # Create ambulance environment with visual rendering
        env_factory = MultiAgentEnvFactory()
        
        print("🏗️  Creating visual ambulance environment...")
        env = env_factory.create_ambulance_env(
            scenario_name="highway_emergency_light",
            obs_type="Kinematics",
            n_agents=4
        )
        
        # Configure for visual display after creation
        if hasattr(env, 'unwrapped'):
            env.unwrapped.config.update({
                'real_time_rendering': True,
                'offscreen_rendering': False,
                'show_trajectories': True,
                'manual_control': False
            })
            # Set render mode directly on the environment
            env.render_mode = 'human'
        
        print("✅ Environment created - starting visual test")
        print("   🚑 Red/first vehicle = Ambulance")
        print("   🚙 Other vehicles = Normal traffic")
        print("   Press Ctrl+C to stop")
        print()
        
        # Initialize action sampler
        from highway_datacollection.collection.action_samplers import RandomActionSampler
        action_sampler = RandomActionSampler(action_space_size=5, seed=42)
        
        # Reset and run
        obs, info = env.reset(seed=42)
        
        for step in range(50):  # Run for 50 steps
            # Render (shows the visual)
            env.render()
            
            # Sample actions using the proper action sampler
            actions = action_sampler.sample_actions(
                observations={"Kinematics": obs},
                n_agents=4,
                step=step
            )
            
            # Take step
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Print progress
            if step % 10 == 0:
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
    print("Starting quick visual test...")
    print("Make sure you've activated the environment: source avs_venv/bin/activate")
    print()
    
    success = quick_visual_test()
    
    if success:
        print("\n🚀 Ready for full ambulance data collection!")
    else:
        print("\n🔧 Please check your configuration and try again")