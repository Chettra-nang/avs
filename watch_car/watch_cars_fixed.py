#!/usr/bin/env python3
"""
Fixed Watch Cars Demo - Simple visual highway simulation
"""

import gymnasium as gym
import highway_env
import time
import numpy as np

def run_visual_highway():
    """Run a visual highway demonstration with proper error handling."""
    print("Visual Highway Demo")
    print("=" * 30)
    print("Starting visual simulation...")
    print("Press Ctrl+C to stop")
    print("=" * 30)
    
    try:
        # Create environment with visual rendering
        env = gym.make('highway-v0', render_mode='human')
        
        # Configure for better visualization
        config = {
            'vehicles_count': 30,
            'lanes_count': 4,
            'duration': 40,
            'real_time_rendering': True,
            'show_trajectories': True,
            'render_agent': True,
            'simulation_frequency': 15,
            'policy_frequency': 1,
            'centering_position': [0.3, 0.5],
            'scaling': 5.5
        }
        
        if hasattr(env.unwrapped, 'configure'):
            env.unwrapped.configure(config)
        
        print("Environment created successfully!")
        
        # Reset environment
        obs, info = env.reset()
        step_count = 0
        episode_count = 0
        total_reward = 0
        
        while step_count < 2000:  # Run for 2000 steps
            # Simple intelligent action selection
            if len(obs) > 0:
                # Basic driving logic
                ego_speed = np.sqrt(obs[0][3]**2 + obs[0][4]**2) if obs[0][0] > 0 else 0
                
                if ego_speed < 20:
                    action = 2  # FASTER
                elif ego_speed > 30:
                    action = 0  # SLOWER
                elif np.random.random() < 0.1:  # 10% chance to change lanes
                    action = np.random.choice([3, 4])  # LANE_LEFT or LANE_RIGHT
                else:
                    action = 1  # IDLE
            else:
                action = 1  # IDLE
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Render the environment (this should show the visual window)
            env.render()
            
            # Small delay for smooth visualization
            time.sleep(0.05)  # 20 FPS
            
            step_count += 1
            
            # Print status every 100 steps
            if step_count % 100 == 0:
                avg_reward = total_reward / step_count
                print(f"Step {step_count}: Avg Reward = {avg_reward:.3f}")
            
            # Reset if episode ends
            if terminated or truncated:
                episode_count += 1
                print(f"Episode {episode_count} ended at step {step_count}")
                obs, info = env.reset()
        
        print(f"\nDemo completed!")
        print(f"Total steps: {step_count}")
        print(f"Total episodes: {episode_count}")
        print(f"Average reward: {total_reward/step_count:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("This might be due to display issues or missing dependencies.")
        print("Try running: sudo apt-get install python3-tk python3-dev")
        
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
        
    finally:
        try:
            env.close()
        except:
            pass

if __name__ == "__main__":
    run_visual_highway()