#!/usr/bin/env python3
"""
Watch Cars Drive Automatically - Comprehensive Highway Simulation Viewer

This script provides a complete autonomous driving simulation viewer with:
- Multiple traffic scenarios
- Real-time statistics
- Configurable parameters
- Different driving behaviors
- Performance monitoring
- Data collection capabilities

Requirements:
- gymnasium
- highway-env
- pygame
- numpy
- matplotlib (optional, for statistics)
- pandas (optional, for data analysis)

Usage:
    python watch_cars_auto.py                    # Default simulation
    python watch_cars_auto.py --scenario dense   # Dense traffic
    python watch_cars_auto.py --agents 5         # 5 controlled agents
    python watch_cars_auto.py --duration 120     # 2 minutes simulation
    python watch_cars_auto.py --collect-data     # Save simulation data
"""

import gymnasium as gym
import highway_env
import pygame
import numpy as np
import time
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Optional imports for enhanced features
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available. Statistics plotting disabled.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Note: pandas not available. Data analysis features disabled.")


@dataclass
class SimulationStats:
    """Statistics tracking for the simulation."""
    total_steps: int = 0
    total_episodes: int = 0
    total_rewards: float = 0.0
    collision_count: int = 0
    lane_changes: int = 0
    average_speed: float = 0.0
    max_speed: float = 0.0
    episode_lengths: List[int] = None
    rewards_history: List[float] = None
    
    def __post_init__(self):
        if self.episode_lengths is None:
            self.episode_lengths = []
        if self.rewards_history is None:
            self.rewards_history = []


class HighwaySimulationViewer:
    """Comprehensive highway simulation viewer with multiple features."""
    
    # Predefined scenario configurations
    SCENARIOS = {
        'free_flow': {
            'name': 'Free Flow Traffic',
            'description': 'Light traffic with smooth flow',
            'vehicles_count': 20,
            'lanes_count': 4,
            'duration': 40,
            'initial_spacing': 3,
            'controlled_vehicles': 1
        },
        'dense': {
            'name': 'Dense Commuting',
            'description': 'Heavy commuter traffic',
            'vehicles_count': 80,
            'lanes_count': 4,
            'duration': 45,
            'initial_spacing': 1.5,
            'controlled_vehicles': 2
        },
        'stop_go': {
            'name': 'Stop and Go',
            'description': 'Congested traffic with frequent stops',
            'vehicles_count': 60,
            'lanes_count': 3,
            'duration': 50,
            'initial_spacing': 1.0,
            'controlled_vehicles': 2
        },
        'aggressive': {
            'name': 'Aggressive Neighbors',
            'description': 'Traffic with aggressive lane-changing behavior',
            'vehicles_count': 45,
            'lanes_count': 4,
            'duration': 35,
            'initial_spacing': 2.0,
            'controlled_vehicles': 3
        },
        'lane_closure': {
            'name': 'Lane Closure',
            'description': 'Traffic merging due to lane closure',
            'vehicles_count': 55,
            'lanes_count': 3,
            'duration': 40,
            'initial_spacing': 1.8,
            'controlled_vehicles': 2
        }
    }
    
    # Action names for display
    ACTION_NAMES = {
        0: "SLOWER",
        1: "IDLE", 
        2: "FASTER",
        3: "LANE_LEFT",
        4: "LANE_RIGHT"
    }
    
    def __init__(self, scenario: str = 'free_flow', controlled_agents: int = None, 
                 duration: int = None, collect_data: bool = False, 
                 show_stats: bool = True, save_video: bool = False):
        """
        Initialize the highway simulation viewer.
        
        Args:
            scenario: Scenario name from SCENARIOS
            controlled_agents: Number of controlled agents (overrides scenario default)
            duration: Simulation duration in seconds (overrides scenario default)
            collect_data: Whether to collect and save simulation data
            show_stats: Whether to display real-time statistics
            save_video: Whether to save video recording (requires additional setup)
        """
        self.scenario_name = scenario
        self.scenario_config = self.SCENARIOS.get(scenario, self.SCENARIOS['free_flow'])
        self.collect_data = collect_data
        self.show_stats = show_stats
        self.save_video = save_video
        
        # Override configuration if specified
        if controlled_agents is not None:
            self.scenario_config['controlled_vehicles'] = controlled_agents
        if duration is not None:
            self.scenario_config['duration'] = duration
            
        # Initialize components
        self.env = None
        self.stats = SimulationStats()
        self.data_collection = []
        self.current_episode_data = []
        self.start_time = None
        self.last_stats_update = 0
        
        # Display settings
        self.font = None
        self.clock = None
        
    def create_environment(self) -> gym.Env:
        """Create and configure the highway environment."""
        print(f"Creating environment: {self.scenario_config['name']}")
        print(f"Description: {self.scenario_config['description']}")
        
        # Create environment
        env = gym.make('highway-v0', render_mode='human')
        
        # Configure environment
        # Note: highway-env supports multiple controlled vehicles, but we need to handle it properly
        actual_controlled = self.scenario_config['controlled_vehicles']
        
        config = {
            'vehicles_count': self.scenario_config['vehicles_count'],
            'lanes_count': self.scenario_config['lanes_count'],
            'duration': self.scenario_config['duration'],
            'initial_spacing': self.scenario_config['initial_spacing'],
            'controlled_vehicles': actual_controlled,  # Highway-env limitation
            'action': {
                'type': 'DiscreteMetaAction'
            },
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
            'offscreen_rendering': False,
            'centering_position': [0.3, 0.5],
            'scaling': 5.5,
            'high_speed_reward': 0.4,
            'right_lane_reward': 0.1,
            'lane_change_reward': 0,
            'reward_speed_range': [20, 30],
            'normalize_reward': True
        }
        
        # Store the requested number for display purposes
        self.requested_agents = self.scenario_config['controlled_vehicles']
        self.actual_controlled = actual_controlled
        
        # Apply configuration
        if hasattr(env.unwrapped, 'configure'):
            env.unwrapped.configure(config)
        
        return env
    
    def initialize_display(self):
        """Initialize pygame display components."""
        pygame.init()
        if pygame.font.get_init():
            self.font = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()
    
    def generate_intelligent_actions(self, observations: np.ndarray, agent_idx: int) -> int:
        """
        Generate intelligent actions based on observations.
        
        Args:
            observations: Current observations
            agent_idx: Index of the agent
            
        Returns:
            Action to take
        """
        if observations.shape[0] == 0:
            return 1  # IDLE
        
        # Extract ego vehicle and other vehicles
        ego = observations[0]
        others = observations[1:] if observations.shape[0] > 1 else np.array([])
        
        # Simple intelligent behavior
        if len(others) > 0:
            # Check for vehicles ahead in same lane
            ego_lane = ego[2]  # y position (lane)
            ego_x = ego[1]     # x position
            ego_speed = np.sqrt(ego[3]**2 + ego[4]**2)  # speed
            
            # Find vehicles ahead in same lane
            same_lane_ahead = []
            for other in others:
                if (other[0] > 0.5 and  # vehicle present
                    abs(other[2] - ego_lane) < 2.0 and  # same lane
                    other[1] > ego_x):  # ahead
                    same_lane_ahead.append(other)
            
            if same_lane_ahead:
                # Find closest vehicle ahead
                closest = min(same_lane_ahead, key=lambda v: v[1] - ego_x)
                distance = closest[1] - ego_x
                relative_speed = ego[3] - closest[3]
                
                # Decision making
                if distance < 15 and relative_speed > 0:  # Too close and approaching
                    # Try to change lanes or slow down
                    if np.random.random() < 0.3:  # 30% chance to change lanes
                        return np.random.choice([3, 4])  # LANE_LEFT or LANE_RIGHT
                    else:
                        return 0  # SLOWER
                elif distance < 25 and relative_speed > 5:  # Moderate distance, high relative speed
                    return 0  # SLOWER
                elif ego_speed < 20:  # Too slow
                    return 2  # FASTER
                else:
                    return 1  # IDLE
            else:
                # No vehicle ahead, maintain or increase speed
                if ego_speed < 25:
                    return 2  # FASTER
                else:
                    return 1  # IDLE
        else:
            # No other vehicles, maintain speed
            return 1  # IDLE
    
    def update_statistics(self, observations: np.ndarray, actions: List[int], 
                         rewards: List[float], info: Dict):
        """Update simulation statistics."""
        self.stats.total_steps += 1
        
        if isinstance(rewards, (list, tuple)):
            step_reward = sum(rewards)
        else:
            step_reward = rewards
            
        self.stats.total_rewards += step_reward
        
        # Calculate speeds
        if observations.shape[0] > 0:
            speeds = []
            for i in range(min(len(actions), observations.shape[0])):
                if i < observations.shape[0]:
                    ego = observations[i] if len(observations.shape) > 2 else observations
                    speed = np.sqrt(ego[3]**2 + ego[4]**2) if ego[0] > 0 else 0
                    speeds.append(speed)
            
            if speeds:
                current_avg_speed = np.mean(speeds)
                self.stats.average_speed = (self.stats.average_speed * (self.stats.total_steps - 1) + 
                                          current_avg_speed) / self.stats.total_steps
                self.stats.max_speed = max(self.stats.max_speed, max(speeds))
        
        # Count lane changes
        for action in actions:
            if action in [3, 4]:  # LANE_LEFT or LANE_RIGHT
                self.stats.lane_changes += 1
        
        # Check for collisions (simplified)
        if isinstance(info, dict) and info.get('crashed', False):
            self.stats.collision_count += 1
        elif isinstance(info, list):
            for agent_info in info:
                if isinstance(agent_info, dict) and agent_info.get('crashed', False):
                    self.stats.collision_count += 1
    
    def collect_step_data(self, observations: np.ndarray, actions: List[int], 
                         rewards: List[float], info: Dict, step: int):
        """Collect data for this simulation step."""
        if not self.collect_data:
            return
        
        step_data = {
            'step': step,
            'timestamp': time.time(),
            'scenario': self.scenario_name,
            'actions': actions.copy() if isinstance(actions, list) else [actions],
            'rewards': rewards.copy() if isinstance(rewards, list) else [rewards],
            'total_reward': sum(rewards) if isinstance(rewards, list) else rewards,
            'info': info
        }
        
        # Add observation features
        if observations.shape[0] > 0:
            step_data['num_vehicles'] = observations.shape[0]
            
            # Extract features for each controlled vehicle
            for i in range(min(len(actions), observations.shape[0])):
                ego = observations[i] if len(observations.shape) > 2 else observations
                if ego[0] > 0:  # Vehicle present
                    step_data[f'agent_{i}_x'] = float(ego[1])
                    step_data[f'agent_{i}_y'] = float(ego[2])
                    step_data[f'agent_{i}_vx'] = float(ego[3])
                    step_data[f'agent_{i}_vy'] = float(ego[4])
                    step_data[f'agent_{i}_speed'] = float(np.sqrt(ego[3]**2 + ego[4]**2))
        
        self.current_episode_data.append(step_data)
    
    def display_statistics(self, screen_surface=None):
        """Display real-time statistics on screen."""
        if not self.show_stats or not self.font:
            return
        
        # Only update display every 30 steps to avoid flickering
        if self.stats.total_steps - self.last_stats_update < 30:
            return
        self.last_stats_update = self.stats.total_steps
        
        # Calculate runtime
        runtime = time.time() - self.start_time if self.start_time else 0
        
        # Prepare statistics text
        stats_text = [
            f"Scenario: {self.scenario_config['name']}",
            f"Runtime: {runtime:.1f}s",
            f"Steps: {self.stats.total_steps}",
            f"Episodes: {self.stats.total_episodes}",
            f"Total Reward: {self.stats.total_rewards:.2f}",
            f"Avg Reward: {self.stats.total_rewards/max(1, self.stats.total_steps):.3f}",
            f"Collisions: {self.stats.collision_count}",
            f"Lane Changes: {self.stats.lane_changes}",
            f"Avg Speed: {self.stats.average_speed:.1f} m/s",
            f"Max Speed: {self.stats.max_speed:.1f} m/s",
            f"Controlled Agents: {self.actual_controlled}/{self.requested_agents}"
        ]
        
        # Print to console
        print(f"\r{' | '.join(stats_text[:6])}", end='', flush=True)
    
    def save_collected_data(self):
        """Save collected simulation data to files."""
        if not self.collect_data or not self.data_collection:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = Path("data/simulations")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw data as JSON
        json_file = data_dir / f"simulation_{self.scenario_name}_{timestamp}.json"
        with open(json_file, 'w') as f:
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
                'episodes': self.data_collection
            }, f, indent=2)
        
        print(f"\n✓ Simulation data saved to: {json_file}")
        
        # Save as CSV if pandas is available
        if HAS_PANDAS and self.data_collection:
            try:
                # Flatten episode data
                all_steps = []
                for episode_idx, episode in enumerate(self.data_collection):
                    for step_data in episode:
                        step_data['episode'] = episode_idx
                        all_steps.append(step_data)
                
                df = pd.DataFrame(all_steps)
                csv_file = data_dir / f"simulation_{self.scenario_name}_{timestamp}.csv"
                df.to_csv(csv_file, index=False)
                print(f"✓ CSV data saved to: {csv_file}")
                
                # Generate summary statistics
                summary = {
                    'total_episodes': len(self.data_collection),
                    'total_steps': len(all_steps),
                    'avg_episode_length': len(all_steps) / len(self.data_collection),
                    'avg_reward_per_step': df['total_reward'].mean(),
                    'total_reward': df['total_reward'].sum(),
                    'collision_rate': self.stats.collision_count / len(all_steps),
                    'lane_change_rate': self.stats.lane_changes / len(all_steps)
                }
                
                summary_file = data_dir / f"summary_{self.scenario_name}_{timestamp}.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"✓ Summary statistics saved to: {summary_file}")
                
            except Exception as e:
                print(f"Warning: Could not save CSV data: {e}")
    
    def plot_statistics(self):
        """Plot simulation statistics if matplotlib is available."""
        if not HAS_MATPLOTLIB or not self.stats.rewards_history:
            return
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Rewards over time
            ax1.plot(self.stats.rewards_history)
            ax1.set_title('Rewards Over Time')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Reward')
            ax1.grid(True)
            
            # Episode lengths
            if self.stats.episode_lengths:
                ax2.hist(self.stats.episode_lengths, bins=20, alpha=0.7)
                ax2.set_title('Episode Length Distribution')
                ax2.set_xlabel('Episode Length (steps)')
                ax2.set_ylabel('Frequency')
                ax2.grid(True)
            
            # Cumulative rewards
            cumulative_rewards = np.cumsum(self.stats.rewards_history)
            ax3.plot(cumulative_rewards)
            ax3.set_title('Cumulative Rewards')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Cumulative Reward')
            ax3.grid(True)
            
            # Statistics summary
            ax4.axis('off')
            stats_text = f"""
            Scenario: {self.scenario_config['name']}
            Total Steps: {self.stats.total_steps}
            Total Episodes: {self.stats.total_episodes}
            Total Rewards: {self.stats.total_rewards:.2f}
            Avg Reward/Step: {self.stats.total_rewards/max(1, self.stats.total_steps):.3f}
            Collisions: {self.stats.collision_count}
            Lane Changes: {self.stats.lane_changes}
            Avg Speed: {self.stats.average_speed:.1f} m/s
            Max Speed: {self.stats.max_speed:.1f} m/s
            """
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                    verticalalignment='top', fontsize=10, fontfamily='monospace')
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_dir = Path("data/plots")
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_file = plot_dir / f"simulation_stats_{self.scenario_name}_{timestamp}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"✓ Statistics plot saved to: {plot_file}")
            
            plt.show()
            
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
    
    def run_simulation(self, max_steps: int = 5000, target_episodes: int = None):
        """
        Run the highway simulation.
        
        Args:
            max_steps: Maximum number of simulation steps
            target_episodes: Target number of episodes to complete (None for unlimited)
        """
        print(f"\nStarting Highway Simulation")
        print("=" * 50)
        print(f"Scenario: {self.scenario_config['name']}")
        print(f"Description: {self.scenario_config['description']}")
        print(f"Requested Agents: {self.requested_agents}")
        print(f"Actually Controlled: {self.actual_controlled} (highway-env limitation)")
        print(f"Total Vehicles: {self.scenario_config['vehicles_count']}")
        print(f"Duration: {self.scenario_config['duration']}s")
        print(f"Data Collection: {'Enabled' if self.collect_data else 'Disabled'}")
        print("=" * 50)
        print("Note: Highway-env controls 1 vehicle, others are intelligent traffic")
        print("Controls: Press Ctrl+C to stop simulation")
        print("=" * 50)
        
        # Initialize components
        self.env = self.create_environment()
        self.initialize_display()
        self.start_time = time.time()
        
        try:
            # Reset environment
            obs, info = self.env.reset()
            step_count = 0
            episode_step = 0
            
            while step_count < max_steps:
                # Generate actions based on number of controlled vehicles
                n_controlled = self.actual_controlled
                
                if n_controlled == 1:
                    # Single agent case
                    action = self.generate_intelligent_actions(obs, 0)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    actions = [action]
                    rewards = [reward] if not isinstance(reward, (list, tuple)) else reward
                else:
                    # Multi-agent case - generate actions for each controlled vehicle
                    actions = []
                    for agent_idx in range(n_controlled):
                        if len(obs.shape) > 2 and agent_idx < obs.shape[0]:
                            agent_obs = obs[agent_idx]
                        else:
                            agent_obs = obs
                        action = self.generate_intelligent_actions(agent_obs, agent_idx)
                        actions.append(action)
                    
                    # Step with list of actions
                    obs, rewards, terminated, truncated, info = self.env.step(actions)
                    if not isinstance(rewards, (list, tuple)):
                        rewards = [rewards]
                
                # Render environment
                self.env.render()
                
                # Update statistics
                self.update_statistics(obs, actions, rewards, info)
                
                # Collect data
                self.collect_step_data(obs, actions, rewards, info, episode_step)
                
                # Display statistics
                self.display_statistics()
                
                # Track rewards for plotting
                if isinstance(rewards, (list, tuple)):
                    self.stats.rewards_history.append(sum(rewards))
                else:
                    self.stats.rewards_history.append(rewards)
                
                step_count += 1
                episode_step += 1
                
                # Control simulation speed
                if self.clock:
                    self.clock.tick(30)  # 30 FPS
                else:
                    time.sleep(0.033)  # ~30 FPS fallback
                
                # Check if episode ended
                if terminated or truncated:
                    self.stats.total_episodes += 1
                    self.stats.episode_lengths.append(episode_step)
                    
                    # Save episode data
                    if self.collect_data and self.current_episode_data:
                        self.data_collection.append(self.current_episode_data.copy())
                        self.current_episode_data.clear()
                    
                    print(f"\nEpisode {self.stats.total_episodes} completed ({episode_step} steps)")
                    
                    # Check if we've reached target episodes
                    if target_episodes and self.stats.total_episodes >= target_episodes:
                        print(f"Reached target of {target_episodes} episodes")
                        break
                    
                    # Reset for next episode
                    obs, info = self.env.reset()
                    episode_step = 0
            
            print(f"\nSimulation completed!")
            
        except KeyboardInterrupt:
            print(f"\nSimulation stopped by user")
        
        finally:
            # Cleanup and save data
            if self.env:
                self.env.close()
            
            # Final statistics
            runtime = time.time() - self.start_time
            print(f"\nFinal Statistics:")
            print(f"Runtime: {runtime:.1f}s")
            print(f"Total Steps: {self.stats.total_steps}")
            print(f"Total Episodes: {self.stats.total_episodes}")
            print(f"Total Rewards: {self.stats.total_rewards:.2f}")
            print(f"Average Reward per Step: {self.stats.total_rewards/max(1, self.stats.total_steps):.3f}")
            print(f"Collisions: {self.stats.collision_count}")
            print(f"Lane Changes: {self.stats.lane_changes}")
            print(f"Average Speed: {self.stats.average_speed:.1f} m/s")
            print(f"Max Speed: {self.stats.max_speed:.1f} m/s")
            
            # Save data and generate plots
            if self.collect_data:
                self.save_collected_data()
            
            if self.show_stats:
                self.plot_statistics()


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Watch Cars Drive Automatically - Highway Simulation Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python watch_cars_auto.py                           # Default free flow scenario
  python watch_cars_auto.py --scenario dense          # Dense traffic scenario
  python watch_cars_auto.py --agents 5 --duration 120 # 5 agents, 2 minutes
  python watch_cars_auto.py --collect-data --no-stats # Collect data, no stats display
  python watch_cars_auto.py --list-scenarios          # List available scenarios
        """
    )
    
    parser.add_argument('--scenario', '-s', 
                       choices=list(HighwaySimulationViewer.SCENARIOS.keys()),
                       default='free_flow',
                       help='Traffic scenario to simulate')
    
    parser.add_argument('--agents', '-a', type=int,
                       help='Number of controlled agents (overrides scenario default)')
    
    parser.add_argument('--duration', '-d', type=int,
                       help='Simulation duration in seconds (overrides scenario default)')
    
    parser.add_argument('--max-steps', type=int, default=5000,
                       help='Maximum number of simulation steps (default: 5000)')
    
    parser.add_argument('--episodes', '-e', type=int,
                       help='Target number of episodes to complete')
    
    parser.add_argument('--collect-data', action='store_true',
                       help='Collect and save simulation data')
    
    parser.add_argument('--no-stats', action='store_true',
                       help='Disable real-time statistics display')
    
    parser.add_argument('--save-video', action='store_true',
                       help='Save video recording (requires additional setup)')
    
    parser.add_argument('--list-scenarios', action='store_true',
                       help='List available scenarios and exit')
    
    args = parser.parse_args()
    
    # List scenarios if requested
    if args.list_scenarios:
        print("Available Scenarios:")
        print("=" * 50)
        for key, config in HighwaySimulationViewer.SCENARIOS.items():
            print(f"{key:15} - {config['name']}")
            print(f"{'':15}   {config['description']}")
            print(f"{'':15}   Vehicles: {config['vehicles_count']}, "
                  f"Lanes: {config['lanes_count']}, "
                  f"Duration: {config['duration']}s")
            print()
        return
    
    # Create and run simulation
    viewer = HighwaySimulationViewer(
        scenario=args.scenario,
        controlled_agents=args.agents,
        duration=args.duration,
        collect_data=args.collect_data,
        show_stats=not args.no_stats,
        save_video=args.save_video
    )
    
    viewer.run_simulation(
        max_steps=args.max_steps,
        target_episodes=args.episodes
    )


if __name__ == "__main__":
    main()