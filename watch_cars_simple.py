#!/usr/bin/env python3
"""
Simple Car Watching Script - Works with highway-env limitations

This is a simplified version that works reliably with highway-env's
single-agent control while still providing intelligent behavior and statistics.
"""

import gymnasium as gym
import highway_env
import pygame
import numpy as np
import time
import argparse
import json
from pathlib import Path
from datetime import datetime

class SimpleCarWatcher:
    """Simple car watching with intelligent behavior."""
    
    SCENARIOS = {
        'free_flow': {'vehicles_count': 20, 'lanes_count': 4, 'duration': 40},
        'dense': {'vehicles_count': 80, 'lanes_count': 4, 'duration': 45},
        'stop_go': {'vehicles_count': 60, 'lanes_count': 3, 'duration': 50},
        'aggressive': {'vehicles_count': 45, 'lanes_count': 4, 'duration': 35},
        'lane_closure': {'vehicles_count': 55, 'lanes_count': 3, 'duration': 40}
    }
    
    def __init__(self, scenario='free_flow', duration=None, collect_data=False):
        self.scenario = scenario
        self.config = self.SCENARIOS.get(scenario, self.SCENARIOS['free_flow'])
        if duration:
            self.config['duration'] = duration
        self.collect_data = collect_data
        self.data = []
        self.stats = {
            'steps': 0, 'episodes': 0, 'total_reward': 0.0,
            'collisions': 0, 'lane_changes': 0
        }
    
    def intelligent_action(self, obs):
        """Generate intelligent action based on observation."""
        if obs.shape[0] == 0:
            return 1  # IDLE
        
        ego = obs[0]
        others = obs[1:] if obs.shape[0] > 1 else np.array([])
        
        if len(others) > 0:
            # Find vehicles ahead in same lane
            ego_lane, ego_x = ego[2], ego[1]
            ego_speed = np.sqrt(ego[3]**2 + ego[4]**2)
            
            ahead_vehicles = []
            for other in others:
                if (other[0] > 0.5 and abs(other[2] - ego_lane) < 2.0 and other[1] > ego_x):
                    ahead_vehicles.append(other)
            
            if ahead_vehicles:
                closest = min(ahead_vehicles, key=lambda v: v[1] - ego_x)
                distance = closest[1] - ego_x
                
                if distance < 15:  # Too close
                    if np.random.random() < 0.3:  # Try lane change
                        return np.random.choice([3, 4])  # LANE_LEFT or LANE_RIGHT
                    else:
                        return 0  # SLOWER
                elif ego_speed < 20:
                    return 2  # FASTER
                else:
                    return 1  # IDLE
            else:
                return 2 if ego_speed < 25 else 1  # Speed up or maintain
        else:
            return 1  # IDLE
    
    def run(self, max_steps=5000):
        """Run the simulation."""
        print(f"🚗 Watching Cars Drive Automatically")
        print(f"Scenario: {self.scenario}")
        print(f"Duration: {self.config['duration']}s")
        print(f"Vehicles: {self.config['vehicles_count']}")
        print(f"Data Collection: {'ON' if self.collect_data else 'OFF'}")
        print("Press Ctrl+C to stop")
        print("-" * 50)
        
        # Create environment
        env = gym.make('highway-v0', render_mode='human')
        
        # Configure
        config = {
            'vehicles_count': self.config['vehicles_count'],
            'lanes_count': self.config['lanes_count'],
            'duration': self.config['duration'],
            'controlled_vehicles': 1,
            'action': {'type': 'DiscreteMetaAction'},
            'observation': {'type': 'Kinematics'},
            'real_time_rendering': True,
            'show_trajectories': True
        }
        
        if hasattr(env.unwrapped, 'configure'):
            env.unwrapped.configure(config)
        
        try:
            obs, info = env.reset()
            start_time = time.time()
            
            for step in range(max_steps):
                # Generate intelligent action
                action = self.intelligent_action(obs)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                
                # Update statistics
                self.stats['steps'] += 1
                self.stats['total_reward'] += reward
                
                if action in [3, 4]:  # Lane changes
                    self.stats['lane_changes'] += 1
                
                # Collect data
                if self.collect_data:
                    self.data.append({
                        'step': step,
                        'action': int(action),
                        'reward': float(reward),
                        'speed': float(np.sqrt(obs[0][3]**2 + obs[0][4]**2)) if obs.shape[0] > 0 else 0
                    })
                
                # Print stats every 100 steps
                if step % 100 == 0:
                    avg_reward = self.stats['total_reward'] / max(1, self.stats['steps'])
                    runtime = time.time() - start_time
                    print(f"Step {step:4d} | Reward: {avg_reward:.3f} | "
                          f"Lane Changes: {self.stats['lane_changes']:3d} | "
                          f"Runtime: {runtime:.1f}s")
                
                # Reset if episode ends
                if terminated or truncated:
                    self.stats['episodes'] += 1
                    obs, info = env.reset()
                
                time.sleep(0.03)  # Control speed
            
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        
        finally:
            env.close()
            
            # Final stats
            runtime = time.time() - start_time
            print(f"\n📊 Final Statistics:")
            print(f"Runtime: {runtime:.1f}s")
            print(f"Steps: {self.stats['steps']}")
            print(f"Episodes: {self.stats['episodes']}")
            print(f"Total Reward: {self.stats['total_reward']:.2f}")
            print(f"Avg Reward: {self.stats['total_reward']/max(1, self.stats['steps']):.3f}")
            print(f"Lane Changes: {self.stats['lane_changes']}")
            
            # Save data
            if self.collect_data and self.data:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                data_dir = Path("data/simple_simulations")
                data_dir.mkdir(parents=True, exist_ok=True)
                
                filename = data_dir / f"simulation_{self.scenario}_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump({
                        'scenario': self.scenario,
                        'config': self.config,
                        'statistics': self.stats,
                        'data': self.data
                    }, f, indent=2)
                
                print(f"💾 Data saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Simple Car Watching")
    parser.add_argument('--scenario', choices=['free_flow', 'dense', 'stop_go', 'aggressive', 'lane_closure'],
                       default='free_flow', help='Traffic scenario')
    parser.add_argument('--duration', type=int, help='Duration in seconds')
    parser.add_argument('--collect-data', action='store_true', help='Collect simulation data')
    parser.add_argument('--max-steps', type=int, default=5000, help='Maximum steps')
    
    args = parser.parse_args()
    
    watcher = SimpleCarWatcher(
        scenario=args.scenario,
        duration=args.duration,
        collect_data=args.collect_data
    )
    
    watcher.run(max_steps=args.max_steps)

if __name__ == "__main__":
    main()