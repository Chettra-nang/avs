#!/usr/bin/env python3
"""
Simple Highway Environment Demo - Watch cars drive automatically
"""

import gymnasium as gym
import highway_env
import time
import numpy as np

def run_highway_demo():
    """Run a simple highway demonstration."""
    print("Highway Environment Demo")
    print("=" * 30)
    print("Watch the cars drive automatically!")
    print("Press Ctrl+C to stop")
    print("=" * 30)
    
    # Create environment
    env = gym.make('highway-v0', render_mode='human')
    
    try:
        obs, info = env.reset()
        
        for step in range(1000):  # Run for 1000 steps
            # Random action for demonstration
            action = env.action_space.sample()
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the environment
            env.render()
            
            # Small delay to make it watchable
            time.sleep(0.1)
            
            # Print status every 100 steps
            if step % 100 == 0:
                print(f"Step {step}: Action={action}, Reward={reward:.2f}")
            
            # Reset if episode ends
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                obs, info = env.reset()
        
        print("Demo completed!")
        
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    finally:
        env.close()

if __name__ == "__main__":
    run_highway_demo()