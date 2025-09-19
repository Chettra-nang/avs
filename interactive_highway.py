#!/usr/bin/env python3
"""
Interactive Highway Environment - Play and control cars manually
"""

import gymnasium as gym
import highway_env
import pygame
import numpy as np

def create_interactive_env():
    """Create an interactive highway environment."""
    # Create environment with configuration
    config = {
        "manual_control": True,
        "real_time_rendering": True,
        "action": {
            "type": "DiscreteMetaAction",
        },
        "observation": {
            "type": "Kinematics"
        },
        "duration": 60,  # 60 seconds
        "vehicles_count": 30,
        "show_trajectories": True,
        "render_agent": True
    }
    
    env = gym.make('highway-v0', render_mode='human')
    
    # Configure the environment
    if hasattr(env.unwrapped, 'configure'):
        env.unwrapped.configure(config)
    
    return env

def play_highway():
    """Play the highway environment interactively."""
    print("Starting Interactive Highway Environment")
    print("=" * 50)
    print("Controls:")
    print("  Arrow Keys or WASD:")
    print("    ↑/W: Accelerate")
    print("    ↓/S: Decelerate") 
    print("    ←/A: Change lane left")
    print("    →/D: Change lane right")
    print("    Space: Maintain speed")
    print("  ESC: Quit")
    print("=" * 50)
    
    env = create_interactive_env()
    
    try:
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            # Render the environment
            env.render()
            
            # Handle pygame events for manual control
            action = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
                    elif event.key in [pygame.K_UP, pygame.K_w]:
                        action = 2  # FASTER
                    elif event.key in [pygame.K_DOWN, pygame.K_s]:
                        action = 0  # SLOWER
                    elif event.key in [pygame.K_LEFT, pygame.K_a]:
                        action = 3  # LANE_LEFT
                    elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                        action = 4  # LANE_RIGHT
                    elif event.key == pygame.K_SPACE:
                        action = 1  # IDLE
            
            # Default action if no key pressed
            if action is None:
                action = 1  # IDLE
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Check if episode is done
            done = terminated or truncated
            
            # Print status every 50 steps
            if step_count % 50 == 0:
                print(f"Step {step_count}: Reward = {total_reward:.2f}")
        
        print(f"\nGame Over!")
        print(f"Total Steps: {step_count}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Average Reward: {total_reward/step_count:.3f}")
        
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    finally:
        env.close()

if __name__ == "__main__":
    play_highway()