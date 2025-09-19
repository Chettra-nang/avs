#!/usr/bin/env python3
"""
Fixed Car Watching Script - Handles highway-env properly

This version correctly handles highway-env's action format and provides
intelligent car behavior with statistics and data collection.
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
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class SimulationStats:
    """Statistics for the simulation."""
    total_steps: int = 0
    total_episodes: int = 0
    total_rewards: float = 0.0
    collision_count: int = 0
    lane_changes: int = 0
    average_speed: float = 0.0
    max_speed: float = 0.0

class IntelligentCarWatcher:
    """Intelligent car watching with proper highway-env integration."""
    
    SCENARIOS = {
        'free_flow': {
            'name': 'Free Flow Traffic',
            'description': 'Light traffic with smooth flow',
            'vehicles_count': 20,
            'lanes_count': 4,
            'duration': 40,
            'initial_spacing': 3
        },
        'dense': {
            'name': 'Dense Commuting',
            'description': 'Heavy commuter traffic',
            'vehicles_count': 80,
            'lanes_count': 4,
            'duration': 45,
            'initial_spacing': 1.5
        },
        'stop_go': {
            'name': 'Stop and Go',
            'description': 'Congested traffic with frequent stops',
            'vehicles_count': 60,
            'lanes_count': 3,
            'duration': 50,
            'initial_spacing': 1.0
        },
        'aggressive': {
            'name': 'Aggressive Neighbors',
            'description': 'Traffic with aggressive lane-changing behavior',
            'vehicles_count': 45,
            'lanes_count': 4,
            'duration': 35,
            'initial_spacing': 2.0
        },
        'lane_closure': {
            'name': 'Lane Closure',
            'description': 'Traffic merging due to lane closure',
            'vehicles_count': 55,
            'lanes_count': 3,
            'duration': 40,
            'initial_spacing': 1.8
        }
    }
    
    ACTION_NAMES = {0: "SLOWER", 1: "IDLE", 2: "FASTER", 3: "LANE_LEFT", 4: "LANE_RIGHT"}
    
    def __init__(self, scenario: str = 'free_flow', duration: int = None, 
                 collect_data: bool = False, show_stats: bool = True):
        """Initialize the car watcher."""
        self.scenario_name = scenario
        self.scenario_config = self.SCENARIOS.get(scenario, self.SCENARIOS['free_flow'])
        self.collect_data = collect_data
        self.show_stats = show_stats
        
        if duration:
            self.scenario_config['duration'] = duration
            
        self.env = None
        self.stats = SimulationStats()
        self.data_collection = []
        self.start_time = None
        
    def create_environment(self) -> gym.Env:
        """Create and configure the highway environment."""
        print(f"Creating environment: {self.scenario_config['name']}")
        print(f"Description: {self.scenario_config['description']}")
        
        env = gym.make('highway-v0', render_mode='human')
        
        # Configure environment for single controlled vehicle
        config = {
            'vehicles_count': self.scenario_config['vehicles_count'],
            'lanes_count': self.scenario_config['lanes_count'],
            'duration': self.scenario_config['duration'],
            'initial_spacing': self.scenario_config['initial_spacing'],
            'controlled_vehicles': 1,  # Highway-env works best with 1 controlled vehicle
            'action': {'type': 'DiscreteMetaAction'},
            'observation': {
                'type': 'Kinematics',
                'vehicles_count': 15,
                'features': ['presence', 'x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                'absolute': False,
                'normalize': True
            },
            'simulation_frequency': 15,
            'policy_frequency': 1,
            'real_time_rendering': True,
            'show_trajectories': True,
            'render_agent': True,
            'centering_position': [0.3, 0.5],
            'scaling': 5.5,
            'high_speed_reward': 0.4,
            'right_lane_reward': 0.1,
            'lane_change_reward': 0,
            'reward_speed_range': [20, 30],
            'normalize_reward': True
        }
        
        if hasattr(env.unwrapped, 'configure'):
            env.unwrapped.configure(config)
        
        return env
    
    def generate_intelligent_action(self, observations: np.ndarray) -> int:
        """Generate intelligent action based on observations."""
        if observations.shape[0] == 0:
            return 1  # IDLE
        
        ego = observations[0]
        others = observations[1:] if observations.shape[0] > 1 else np.array([])
        
        # Extract ego vehicle info
        ego_x, ego_y = ego[1], ego[2]
        ego_vx, ego_vy = ego[3], ego[4]
        ego_speed = np.sqrt(ego_vx**2 + ego_vy**2)
        
        if len(others) > 0:
            # Find vehicles ahead in same lane
            same_lane_ahead = []
            for other in others:
                if (other[0] > 0.5 and  # vehicle present
                    abs(other[2] - ego_y) < 2.0 and  # same lane (within 2m laterally)
                    other[1] > ego_x):  # ahead of ego
                    distance = other[1] - ego_x
                    relative_speed = ego_vx - other[3]
                    same_lane_ahead.append((distance, relative_speed, other))
            
            if same_lane_ahead:
                # Find closest vehicle ahead
                closest_distance, closest_rel_speed, closest_vehicle = min(same_lane_ahead)
                
                # Decision making based on distance and relative speed
                if closest_distance < 10:  # Very close
                    if np.random.random() < 0.4:  # 40% chance to change lanes
                        # Check which lane change is safer
                        left_safe = self._check_lane_safety(ego, others, -4.0)  # Left lane
                        right_safe = self._check_lane_safety(ego, others, 4.0)  # Right lane
                        
                        if left_safe and right_safe:
                            return np.random.choice([3, 4])  # Random lane change
                        elif left_safe:
                            return 3  # LANE_LEFT
                        elif right_safe:
                            return 4  # LANE_RIGHT
                        else:
                            return 0  # SLOWER (can't change lanes)
                    else:
                        return 0  # SLOWER
                        
                elif closest_distance < 20 and closest_rel_speed > 5:  # Approaching fast
                    return 0  # SLOWER
                    
                elif closest_distance > 30 and ego_speed < 20:  # Safe distance, low speed
                    return 2  # FASTER
                    
                else:
                    return 1  # IDLE
            else:
                # No vehicle ahead, maintain or increase speed
                if ego_speed < 25:
                    return 2  # FASTER
                elif ego_speed > 35:
                    return 0  # SLOWER
                else:
                    return 1  # IDLE
        else:
            # No other vehicles visible, maintain reasonable speed
            if ego_speed < 25:
                return 2  # FASTER
            else:
                return 1  # IDLE
    
    def _check_lane_safety(self, ego: np.ndarray, others: np.ndarray, lane_offset: float) -> bool:
        """Check if lane change is safe."""
        target_lane = ego[2] + lane_offset
        ego_x = ego[1]
        
        # Check for vehicles in target lane
        for other in others:
            if (other[0] > 0.5 and  # vehicle present
                abs(other[2] - target_lane) < 2.0):  # in target lane
                distance = abs(other[1] - ego_x)
                if distance < 15:  # Too close for safe lane change
                    return False
        
        return True
    
    def update_statistics(self, action: int, reward: float, observations: np.ndarray):
        """Update simulation statistics."""
        self.stats.total_steps += 1
        self.stats.total_rewards += reward
        
        # Count lane changes
        if action in [3, 4]:
            self.stats.lane_changes += 1
        
        # Calculate speed
        if observations.shape[0] > 0:
            ego = observations[0]
            if ego[0] > 0:  # vehicle present
                speed = np.sqrt(ego[3]**2 + ego[4]**2)
                self.stats.average_speed = ((self.stats.average_speed * (self.stats.total_steps - 1) + speed) / 
                                          self.stats.total_steps)
                self.stats.max_speed = max(self.stats.max_speed, speed)
    
    def collect_step_data(self, action: int, reward: float, observations: np.ndarray, step: int):
        """Collect data for this step."""
        if not self.collect_data:
            return
        
        step_data = {
            'step': step,
            'timestamp': time.time(),
            'scenario': self.scenario_name,
            'action': int(action),
            'action_name': self.ACTION_NAMES[action],
            'reward': float(reward.item() if hasattr(reward, 'item') else reward),
            'total_reward': float(self.stats.total_rewards)
        }
        
        # Add observation data
        if observations.shape[0] > 0:
            ego = observations[0]
            if ego[0] > 0:
                step_data.update({
                    'ego_x': float(ego[1].item() if hasattr(ego[1], 'item') else ego[1]),
                    'ego_y': float(ego[2].item() if hasattr(ego[2], 'item') else ego[2]),
                    'ego_vx': float(ego[3].item() if hasattr(ego[3], 'item') else ego[3]),
                    'ego_vy': float(ego[4].item() if hasattr(ego[4], 'item') else ego[4]),
                    'ego_speed': float(np.sqrt(ego[3]**2 + ego[4]**2).item()),
                    'num_vehicles': int(observations.shape[0])
                })
        
        self.data_collection.append(step_data)
    
    def display_statistics(self):
        """Display real-time statistics."""
        if not self.show_stats:
            return
        
        runtime = time.time() - self.start_time if self.start_time else 0
        avg_reward = self.stats.total_rewards / max(1, self.stats.total_steps)
        
        stats_line = (f"Steps: {self.stats.total_steps:4d} | "
                     f"Episodes: {self.stats.total_episodes:2d} | "
                     f"Reward: {avg_reward:.3f} | "
                     f"Lane Changes: {self.stats.lane_changes:3d} | "
                     f"Speed: {self.stats.average_speed:.1f} m/s | "
                     f"Runtime: {runtime:.1f}s")
        
        print(f"\r{stats_line}", end='', flush=True)
    
    def save_data(self):
        """Save collected data."""
        if not self.collect_data or not self.data_collection:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = Path("data/car_watching")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed data
        data_file = data_dir / f"car_watch_{self.scenario_name}_{timestamp}.json"
        with open(data_file, 'w') as f:
            json.dump({
                'scenario': self.scenario_name,
                'config': self.scenario_config,
                'statistics': {
                    'total_steps': self.stats.total_steps,
                    'total_episodes': self.stats.total_episodes,
                    'total_rewards': self.stats.total_rewards,
                    'collision_count': self.stats.collision_count,
                    'lane_changes': self.stats.lane_changes,
                    'average_speed': self.stats.average_speed,
                    'max_speed': self.stats.max_speed
                },
                'data': self.data_collection
            }, f, indent=2)
        
        print(f"\n💾 Data saved to: {data_file}")
        
        # Save summary
        summary_file = data_dir / f"summary_{self.scenario_name}_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'scenario': self.scenario_name,
                'total_steps': self.stats.total_steps,
                'total_episodes': self.stats.total_episodes,
                'avg_reward_per_step': self.stats.total_rewards / max(1, self.stats.total_steps),
                'total_reward': self.stats.total_rewards,
                'lane_changes': self.stats.lane_changes,
                'lane_change_rate': self.stats.lane_changes / max(1, self.stats.total_steps),
                'average_speed': self.stats.average_speed,
                'max_speed': self.stats.max_speed
            }, f, indent=2)
        
        print(f"📊 Summary saved to: {summary_file}")
    
    def run_simulation(self, max_steps: int = 5000, target_episodes: int = None):
        """Run the car watching simulation."""
        print(f"\n🚗 Starting Intelligent Car Watching")
        print("=" * 50)
        print(f"Scenario: {self.scenario_config['name']}")
        print(f"Description: {self.scenario_config['description']}")
        print(f"Total Vehicles: {self.scenario_config['vehicles_count']}")
        print(f"Duration: {self.scenario_config['duration']}s")
        print(f"Data Collection: {'Enabled' if self.collect_data else 'Disabled'}")
        print("=" * 50)
        print("🎮 Controls: Press Ctrl+C to stop")
        print("🧠 The car will drive intelligently based on traffic conditions")
        print("=" * 50)
        
        self.env = self.create_environment()
        self.start_time = time.time()
        
        try:
            obs, info = self.env.reset()
            step_count = 0
            episode_step = 0
            
            while step_count < max_steps:
                # Generate intelligent action
                action = self.generate_intelligent_action(obs)
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Render
                self.env.render()
                
                # Update statistics
                self.update_statistics(action, reward, obs)
                
                # Collect data
                self.collect_step_data(action, reward, obs, episode_step)
                
                # Display statistics every 30 steps
                if step_count % 30 == 0:
                    self.display_statistics()
                
                step_count += 1
                episode_step += 1
                
                # Control simulation speed (30 FPS)
                time.sleep(0.033)
                
                # Check if episode ended
                if terminated or truncated:
                    self.stats.total_episodes += 1
                    print(f"\n🏁 Episode {self.stats.total_episodes} completed ({episode_step} steps)")
                    
                    if target_episodes and self.stats.total_episodes >= target_episodes:
                        print(f"🎯 Reached target of {target_episodes} episodes")
                        break
                    
                    obs, info = self.env.reset()
                    episode_step = 0
            
            print(f"\n✅ Simulation completed!")
            
        except KeyboardInterrupt:
            print(f"\n🛑 Simulation stopped by user")
        
        finally:
            if self.env:
                self.env.close()
            
            # Final statistics
            runtime = time.time() - self.start_time
            print(f"\n📊 Final Statistics:")
            print(f"Runtime: {runtime:.1f}s")
            print(f"Total Steps: {self.stats.total_steps}")
            print(f"Total Episodes: {self.stats.total_episodes}")
            print(f"Total Rewards: {self.stats.total_rewards:.2f}")
            print(f"Average Reward per Step: {self.stats.total_rewards/max(1, self.stats.total_steps):.3f}")
            print(f"Lane Changes: {self.stats.lane_changes}")
            print(f"Lane Change Rate: {self.stats.lane_changes/max(1, self.stats.total_steps):.3f}")
            print(f"Average Speed: {self.stats.average_speed:.1f} m/s")
            print(f"Max Speed: {self.stats.max_speed:.1f} m/s")
            
            # Save data
            self.save_data()

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Watch Intelligent Cars Drive Automatically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python watch_cars_fixed.py                           # Default free flow
  python watch_cars_fixed.py --scenario dense          # Dense traffic
  python watch_cars_fixed.py --duration 120 --collect-data  # 2 min with data
  python watch_cars_fixed.py --episodes 10             # Stop after 10 episodes
        """
    )
    
    parser.add_argument('--scenario', '-s', 
                       choices=list(IntelligentCarWatcher.SCENARIOS.keys()),
                       default='free_flow',
                       help='Traffic scenario to simulate')
    
    parser.add_argument('--duration', '-d', type=int,
                       help='Simulation duration in seconds')
    
    parser.add_argument('--max-steps', type=int, default=5000,
                       help='Maximum number of simulation steps')
    
    parser.add_argument('--episodes', '-e', type=int,
                       help='Target number of episodes to complete')
    
    parser.add_argument('--collect-data', action='store_true',
                       help='Collect and save simulation data')
    
    parser.add_argument('--no-stats', action='store_true',
                       help='Disable real-time statistics display')
    
    args = parser.parse_args()
    
    # Create and run watcher
    watcher = IntelligentCarWatcher(
        scenario=args.scenario,
        duration=args.duration,
        collect_data=args.collect_data,
        show_stats=not args.no_stats
    )
    
    watcher.run_simulation(
        max_steps=args.max_steps,
        target_episodes=args.episodes
    )

if __name__ == "__main__":
    main()