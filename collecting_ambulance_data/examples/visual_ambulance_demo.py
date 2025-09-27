#!/usr/bin/env python3
"""
Visual Ambulance Demo Script

This script creates a simple visual demonstration of the ambulance data collection
system where you can see the ambulance (first agent) running on screen.

This is a quick confidence check to verify the configuration is working correctly.
"""

import sys
import os
from pathlib import Path
import time
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import ambulance components
from collecting_ambulance_data.scenarios.ambulance_scenarios import get_scenario_by_name
from highway_datacollection.environments.factory import MultiAgentEnvFactory
from highway_datacollection.collection.action_samplers import RandomActionSampler


def create_visual_ambulance_demo(scenario_name="highway_emergency_light", max_steps=100):
    """
    Create a visual demo of ambulance running on screen.
    
    Args:
        scenario_name: Name of ambulance scenario to run
        max_steps: Maximum steps to run the demo
    """
    print(f"🚑 Visual Ambulance Demo: {scenario_name}")
    print("=" * 50)
    
    try:
        # Get scenario configuration
        scenario_config = get_scenario_by_name(scenario_name)
        print(f"✅ Loaded scenario: {scenario_config['description']}")
        print(f"   Traffic density: {scenario_config['traffic_density']}")
        print(f"   Vehicles: {scenario_config['vehicles_count']}")
        print(f"   Duration: {scenario_config['duration']}s")
        
        # Create environment factory
        env_factory = MultiAgentEnvFactory()
        
        # Create ambulance environment with visual rendering enabled
        print("\n🏗️  Creating visual ambulance environment...")
        
        env = env_factory.create_ambulance_env(
            scenario_name=scenario_name,
            obs_type="Kinematics",  # Simple observation type
            n_agents=4              # 4 agents (first is ambulance)
        )
        
        # Configure for visual rendering after creation
        if hasattr(env, 'unwrapped'):
            env.unwrapped.config.update({
                'real_time_rendering': True,
                'render_mode': 'human',
                'offscreen_rendering': False,
                'show_trajectories': True,
                'manual_control': False
            })
        
        print("✅ Environment created successfully")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        print(f"   Number of agents: {env.controlled_vehicles if hasattr(env, 'controlled_vehicles') else 'N/A'}")
        
        # Initialize action sampler
        action_sampler = RandomActionSampler(action_space_size=5, seed=42)
        
        # Reset environment
        print("\n🔄 Resetting environment...")
        obs, info = env.reset(seed=42)
        print("✅ Environment reset complete")
        
        # Show initial state
        if isinstance(obs, (list, np.ndarray)) and len(obs) > 0:
            print(f"   Initial observation shape: {np.array(obs).shape}")
            print(f"   Agents initialized: {len(obs) if isinstance(obs, list) else 'Single agent'}")
        
        print(f"\n🚗 Starting visual demo - {max_steps} steps")
        print("   🚑 Agent 0 (red/first) = Ambulance")
        print("   🚙 Agents 1-3 = Normal vehicles")
        print("   Press Ctrl+C to stop early")
        print()
        
        # Run the demo
        step = 0
        total_reward = 0
        
        try:
            while step < max_steps:
                # Render the environment (this shows the visual)
                env.render()
                
                # Sample actions using the proper action sampler
                actions = action_sampler.sample_actions(
                    observations={"Kinematics": obs},
                    n_agents=4,
                    step=step
                )
                
                # Take step in environment
                obs, rewards, terminated, truncated, info = env.step(actions)
                
                # Calculate total reward
                if isinstance(rewards, list):
                    step_reward = sum(rewards)
                    ambulance_reward = rewards[0]  # First agent is ambulance
                else:
                    step_reward = rewards
                    ambulance_reward = rewards
                
                total_reward += step_reward
                step += 1
                
                # Print progress every 20 steps
                if step % 20 == 0:
                    print(f"   Step {step:3d}: Ambulance reward = {ambulance_reward:6.2f}, Total = {total_reward:8.2f}")
                
                # Check if episode is done
                if isinstance(terminated, list):
                    if any(terminated) or any(truncated):
                        print(f"\n✅ Episode completed at step {step}")
                        break
                else:
                    if terminated or truncated:
                        print(f"\n✅ Episode completed at step {step}")
                        break
                
                # Small delay for visual effect
                time.sleep(0.05)  # 50ms delay for smooth visualization
        
        except KeyboardInterrupt:
            print(f"\n⏹️  Demo stopped by user at step {step}")
        
        # Final statistics
        print(f"\n📊 Demo Statistics:")
        print(f"   Steps completed: {step}")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Average reward per step: {total_reward/step:.3f}" if step > 0 else "   No steps completed")
        
        # Close environment
        env.close()
        print("✅ Environment closed")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def run_multiple_scenario_demo():
    """Run a quick demo of multiple ambulance scenarios."""
    print("\n🎯 MULTIPLE SCENARIO DEMO")
    print("=" * 30)
    
    # Select a few different scenarios to demonstrate
    demo_scenarios = [
        "highway_emergency_light",    # Light traffic
        "highway_emergency_moderate", # Moderate traffic  
        "highway_rush_hour"          # Heavy traffic
    ]
    
    for i, scenario in enumerate(demo_scenarios, 1):
        print(f"\n--- Demo {i}/{len(demo_scenarios)}: {scenario} ---")
        
        success = create_visual_ambulance_demo(
            scenario_name=scenario,
            max_steps=30  # Shorter for multiple demos
        )
        
        if not success:
            print(f"⚠️  Skipping remaining demos due to error")
            break
        
        if i < len(demo_scenarios):
            print("\n⏸️  Pausing 2 seconds before next demo...")
            time.sleep(2)
    
    print(f"\n🎉 Multiple scenario demo complete!")


def main():
    """Main function to run the visual ambulance demo."""
    print("🚑 VISUAL AMBULANCE DEMONSTRATION")
    print("=" * 40)
    print("This script shows the ambulance running visually on screen")
    print("to verify that the configuration is working correctly.")
    print()
    
    # Check if we should run multiple scenarios or just one
    if len(sys.argv) > 1:
        scenario_name = sys.argv[1]
        max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        
        print(f"Running single scenario demo: {scenario_name}")
        success = create_visual_ambulance_demo(scenario_name, max_steps)
        
    else:
        print("Running default scenario demo...")
        success = create_visual_ambulance_demo()
        
        if success:
            # Ask if user wants to see more scenarios
            try:
                response = input("\n🤔 Would you like to see multiple scenarios? (y/n): ").lower().strip()
                if response in ['y', 'yes']:
                    run_multiple_scenario_demo()
            except KeyboardInterrupt:
                print("\n👋 Demo ended by user")
    
    if success:
        print("\n✅ Visual demo completed successfully!")
        print("   Configuration appears to be working correctly.")
        print("\n🚀 Next steps:")
        print("   • Run full data collection with basic_ambulance_collection.py")
        print("   • Customize scenarios in ambulance_scenarios.py")
        print("   • Use collected data for training and analysis")
    else:
        print("\n❌ Demo failed - please check configuration")
        print("   • Verify highway-env is installed correctly")
        print("   • Check that all dependencies are available")
        print("   • Review error messages above")


if __name__ == "__main__":
    main()