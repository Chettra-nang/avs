#!/usr/bin/env python3
"""
Visual Multi-Agent Highway Demo - Watch 4 cars drive together!

This demo shows 4 controlled agents driving on the highway with visual feedback.
You can watch them interact, make decisions, and collect data in real-time.
"""

import gymnasium as gym
import highway_env
import numpy as np
import time
import pygame
from typing import Dict, List, Tuple, Any
from collections import deque

class MultiAgentVisualDemo:
    """Visual demo for multi-agent highway driving."""
    
    def __init__(self, n_agents=4, scenario="dense_commuting"):
        self.n_agents = n_agents
        self.scenario = scenario
        self.env = None
        
        # Agent colors for visualization - using more distinct colors
        self.agent_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
        ]
        
        # Statistics tracking
        self.episode_stats = {
            'rewards': [[] for _ in range(n_agents)],
            'actions': [[] for _ in range(n_agents)],
            'collisions': [0 for _ in range(n_agents)],
            'lane_changes': [0 for _ in range(n_agents)]
        }
        
        # Action names for display
        self.action_names = {
            0: "SLOWER", 1: "IDLE", 2: "FASTER", 
            3: "LANE_LEFT", 4: "LANE_RIGHT"
        }
        
        # Data collection
        self.collected_data = []
        self.current_episode_data = []
        
    def create_environment(self):
        """Create multi-agent highway environment."""
        print(f"🚗 Creating Multi-Agent Environment")
        print(f"   Agents: {self.n_agents}")
        print(f"   Scenario: {self.scenario}")
        
        # Create environment
        env = gym.make('highway-v0', render_mode='human')
        
        # Configure for proper multi-agent simulation following official docs
        config = {
            'observation': {
                'type': 'MultiAgentObservation',
                'observation_config': {
                    'type': 'Kinematics',
                    'vehicles_count': 15,
                    'features': ['presence', 'x', 'y', 'vx', 'vy'],
                    'absolute': False,
                    'normalize': True
                }
            },
            'action': {
                'type': 'MultiAgentAction',
                'action_config': {
                    'type': 'DiscreteMetaAction'
                }
            },
            'controlled_vehicles': self.n_agents,
            'lanes_count': 4,
            'vehicles_count': 50,  # Lots of traffic
            'duration': 40,
            'reward_speed_range': [20, 30],
            'simulation_frequency': 15,
            'policy_frequency': 1,
            # Try to enable per-agent rewards
            'normalize_reward': False,
            'real_time_rendering': True,
            'show_trajectories': True,
            'render_agent': True,
            'centering_position': [0.3, 0.5],
            'scaling': 5.5,
            'offscreen_rendering': False,
            
            # Try to enable different rendering options
            'render_mode': 'human',
            'screen_width': 1200,
            'screen_height': 300,
            
            # Scenario-specific settings
            'initial_spacing': 2.0 if self.scenario == "dense_commuting" else 3.0,
            'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle'
        }
        
        # Apply configuration
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'configure'):
            env.unwrapped.configure(config)
        
        self.env = env
        
        # Check environment capabilities and configure
        self._check_environment_capabilities(env)
        
        # Print observation structure for debugging
        obs, info = env.reset()
        print(f"🔍 Multi-Agent Observation Structure:")
        print(f"   Type: {type(obs)}")
        if isinstance(obs, tuple):
            print(f"   Number of agent observations: {len(obs)}")
            for i, agent_obs in enumerate(obs):
                print(f"   Agent {i} obs shape: {agent_obs.shape if hasattr(agent_obs, 'shape') else 'N/A'}")
        else:
            print(f"   Single observation shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
        
        # Create custom colored vehicle classes
        self._create_colored_vehicle_classes()
        
        # Try to set vehicle colors for controlled vehicles
        self._configure_vehicle_colors(env)
        
        return env
    
    def _create_colored_vehicle_classes(self):
        """Create custom vehicle classes with different colors."""
        try:
            from highway_env.vehicle.controller import MDPVehicle
            
            # Create colored vehicle classes with distinct colors
            class Agent1Vehicle(MDPVehicle):
                COLOR = (255, 0, 0)      # Red
                DEFAULT_COLOR = (255, 0, 0)
            
            class Agent2Vehicle(MDPVehicle):
                COLOR = (0, 255, 0)      # Green
                DEFAULT_COLOR = (0, 255, 0)
            
            class Agent3Vehicle(MDPVehicle):
                COLOR = (0, 0, 255)      # Blue
                DEFAULT_COLOR = (0, 0, 255)
            
            class Agent4Vehicle(MDPVehicle):
                COLOR = (255, 255, 0)    # Yellow
                DEFAULT_COLOR = (255, 255, 0)
            
            # Store the classes for later use
            self.colored_vehicle_classes = [
                Agent1Vehicle, Agent2Vehicle, Agent3Vehicle, Agent4Vehicle
            ]
            
            print("✅ Created 4 distinct colored vehicle classes")
            
        except Exception as e:
            print(f"⚠️  Could not create colored vehicle classes: {e}")
            self.colored_vehicle_classes = []
    
    def _configure_vehicle_colors(self, env):
        """Configure colors for controlled vehicles."""
        try:
            # Reset to get initial state
            obs, info = env.reset()
            
            # Access the road and controlled vehicles
            if hasattr(env.unwrapped, 'road') and hasattr(env.unwrapped.road, 'vehicles'):
                # Find MDPVehicles (controlled vehicles)
                mdp_vehicles = [v for v in env.unwrapped.road.vehicles 
                               if type(v).__name__ == 'MDPVehicle']
                
                print(f"   Found {len(mdp_vehicles)} MDPVehicles to color")
                
                # Set colors for controlled vehicles - simplified approach
                colored_count = 0
                for i, vehicle in enumerate(mdp_vehicles[:self.n_agents]):
                    color = self.agent_colors[i % len(self.agent_colors)]
                    
                    # Try the most reliable method first - class-level COLOR
                    try:
                        vehicle.__class__.COLOR = color
                        colored_count += 1
                        print(f"   ✅ Set color for Agent {i+1}: {color}")
                    except Exception as e:
                        print(f"   ❌ Color setting failed for Agent {i+1}: {e}")
                
                print(f"✅ Successfully configured colors for {colored_count}/{len(mdp_vehicles[:self.n_agents])} vehicles")
                        
        except Exception as e:
            print(f"⚠️  Could not configure vehicle colors: {e}")
            print("   Vehicles will use default colors")
    
    def _replace_with_colored_vehicles(self, env, mdp_vehicles):
        """Replace MDPVehicles with colored versions."""
        try:
            road = env.unwrapped.road
            
            for i, vehicle in enumerate(mdp_vehicles[:self.n_agents]):
                if i < len(self.colored_vehicle_classes):
                    # Get vehicle properties
                    position = vehicle.position.copy()
                    heading = vehicle.heading
                    speed = vehicle.speed
                    
                    # Create new colored vehicle
                    ColoredVehicleClass = self.colored_vehicle_classes[i]
                    new_vehicle = ColoredVehicleClass(
                        road=road,
                        position=position,
                        heading=heading,
                        speed=speed
                    )
                    
                    # Copy important attributes
                    new_vehicle.controlled = getattr(vehicle, 'controlled', True)
                    
                    # Replace in road vehicles list
                    vehicle_index = road.vehicles.index(vehicle)
                    road.vehicles[vehicle_index] = new_vehicle
                    
                    print(f"   ✅ Replaced vehicle {i} with {ColoredVehicleClass.__name__}")
            
            print(f"✅ Replaced {min(len(mdp_vehicles), len(self.colored_vehicle_classes))} vehicles with colored versions")
            
        except Exception as e:
            print(f"⚠️  Could not replace vehicles with colored versions: {e}")
    
    def _check_environment_capabilities(self, env):
        """Check what multi-agent capabilities the environment has."""
        try:
            obs, info = env.reset()
            
            print(f"🔍 Environment Analysis:")
            print(f"   Observation type: {type(obs)}")
            if hasattr(obs, 'shape'):
                print(f"   Observation shape: {obs.shape}")
            
            if hasattr(env.unwrapped, 'controlled_vehicles'):
                controlled_vehicles = env.unwrapped.controlled_vehicles
                print(f"   Controlled vehicles count: {len(controlled_vehicles) if controlled_vehicles else 0}")
                if controlled_vehicles:
                    print(f"   Controlled vehicles: {controlled_vehicles[:2]}{'...' if len(controlled_vehicles) > 2 else ''}")
            
            if hasattr(env.unwrapped, 'road') and hasattr(env.unwrapped.road, 'vehicles'):
                total_vehicles = len(env.unwrapped.road.vehicles)
                controlled = sum(1 for v in env.unwrapped.road.vehicles if getattr(v, 'controlled', False))
                print(f"   Total vehicles: {total_vehicles}")
                print(f"   Actually controlled: {controlled}")
                
                # Check vehicle types
                vehicle_types = [type(v).__name__ for v in env.unwrapped.road.vehicles[:5]]
                print(f"   Vehicle types (first 5): {vehicle_types}")
                
        except Exception as e:
            print(f"⚠️  Could not analyze environment: {e}")
    
    def _multi_agent_step(self, env, actions):
        """Multi-agent step function using proper tuple of actions."""
        try:
            # Convert actions list to tuple as required by MultiAgentAction
            action_tuple = tuple(actions)
            
            # Execute the multi-agent step
            next_obs, reward, terminated, truncated, info = env.step(action_tuple)
            
            return next_obs, reward, terminated, truncated, info
                
        except Exception as e:
            print(f"⚠️  Multi-agent step failed: {e}")
            # If anything fails, fallback to single agent
            return env.step(actions[0])
    
    def generate_intelligent_actions(self, observations) -> List[int]:
        """Generate intelligent actions for all agents."""
        actions = []
        
        # Handle MultiAgentObservation - should be a tuple of observations
        if isinstance(observations, tuple):
            # Each element in the tuple is an observation for one agent
            for agent_idx in range(self.n_agents):
                if agent_idx < len(observations):
                    agent_obs = observations[agent_idx]
                    action = self._agent_policy(agent_obs, agent_idx)
                    actions.append(action)
                else:
                    # If we don't have enough observations, use default action
                    actions.append(1)  # IDLE
        else:
            # Fallback for single observation - use same observation for all agents
            if not isinstance(observations, np.ndarray):
                observations = np.array(observations) if observations is not None else np.array([])
            
            for agent_idx in range(self.n_agents):
                action = self._agent_policy(observations, agent_idx)
                actions.append(action)
        
        return actions
    
    def _agent_policy(self, observation: np.ndarray, agent_idx: int) -> int:
        """Individual agent policy based on personality."""
        if observation.shape[0] == 0:
            return 1  # IDLE
        
        ego = observation[0]
        others = observation[1:] if observation.shape[0] > 1 else np.array([])
        
        # Agent personalities
        if agent_idx == 0:  # Conservative driver
            return self._conservative_policy(ego, others)
        elif agent_idx == 1:  # Aggressive driver
            return self._aggressive_policy(ego, others)
        elif agent_idx == 2:  # Speed keeper
            return self._speed_keeper_policy(ego, others)
        else:  # Adaptive driver
            return self._adaptive_policy(ego, others)
    
    def _conservative_policy(self, ego: np.ndarray, others: np.ndarray) -> int:
        """Conservative driving policy."""
        if ego[0] < 0.5:  # No vehicle
            return 1
        
        speed = np.sqrt(ego[3]**2 + ego[4]**2)
        
        # Check for vehicles ahead
        if len(others) > 0:
            ego_lane = ego[2]
            ahead_vehicles = others[(others[:, 0] > 0.5) & 
                                  (abs(others[:, 2] - ego_lane) < 2.0) & 
                                  (others[:, 1] > ego[1])]
            
            if len(ahead_vehicles) > 0:
                closest_distance = np.min(ahead_vehicles[:, 1] - ego[1])
                if closest_distance < 20:  # Too close
                    return 0  # SLOWER
        
        # Maintain moderate speed
        if speed < 18:
            return 2  # FASTER
        elif speed > 25:
            return 0  # SLOWER
        else:
            return 1  # IDLE
    
    def _aggressive_policy(self, ego: np.ndarray, others: np.ndarray) -> int:
        """Aggressive driving policy."""
        if ego[0] < 0.5:
            return 1
        
        speed = np.sqrt(ego[3]**2 + ego[4]**2)
        
        # Aggressive drivers change lanes more often
        if np.random.random() < 0.15:  # 15% chance to change lanes
            return np.random.choice([3, 4])  # LANE_LEFT or LANE_RIGHT
        
        # Prefer higher speeds
        if speed < 28:
            return 2  # FASTER
        elif speed > 35:
            return 0  # SLOWER
        else:
            return 1  # IDLE
    
    def _speed_keeper_policy(self, ego: np.ndarray, others: np.ndarray) -> int:
        """Speed-focused policy."""
        if ego[0] < 0.5:
            return 1
        
        speed = np.sqrt(ego[3]**2 + ego[4]**2)
        target_speed = 25.0
        
        if speed < target_speed - 2:
            return 2  # FASTER
        elif speed > target_speed + 2:
            return 0  # SLOWER
        else:
            return 1  # IDLE
    
    def _adaptive_policy(self, ego: np.ndarray, others: np.ndarray) -> int:
        """Adaptive policy based on traffic conditions."""
        if ego[0] < 0.5:
            return 1
        
        # Count nearby vehicles
        nearby_count = 0
        if len(others) > 0:
            distances = np.sqrt((others[:, 1] - ego[1])**2 + (others[:, 2] - ego[2])**2)
            nearby_count = np.sum((others[:, 0] > 0.5) & (distances < 30))
        
        speed = np.sqrt(ego[3]**2 + ego[4]**2)
        
        if nearby_count > 3:  # Dense traffic - be conservative
            if speed > 20:
                return 0  # SLOWER
            else:
                return 1  # IDLE
        else:  # Light traffic - be more aggressive
            if speed < 25:
                return 2  # FASTER
            elif np.random.random() < 0.1:
                return np.random.choice([3, 4])  # Lane change
            else:
                return 1  # IDLE
    
    def update_statistics(self, actions: List[int], rewards: List[float], info: Dict):
        """Update agent statistics."""
        for i, (action, reward) in enumerate(zip(actions, rewards)):
            self.episode_stats['actions'][i].append(action)
            self.episode_stats['rewards'][i].append(reward)
            
            # Count lane changes
            if action in [3, 4]:  # LANE_LEFT or LANE_RIGHT
                self.episode_stats['lane_changes'][i] += 1
            
            # Check for collisions (simplified)
            if isinstance(info, list) and i < len(info):
                if info[i].get('crashed', False):
                    self.episode_stats['collisions'][i] += 1
            elif isinstance(info, dict) and info.get('crashed', False):
                self.episode_stats['collisions'][0] += 1  # Single agent info
    
    def collect_step_data(self, observations, actions: List[int], 
                         rewards: List[float], step: int):
        """Collect data for this step."""
        step_data = {
            'step': step,
            'timestamp': time.time(),
            'n_agents': self.n_agents,
            'actions': actions.copy(),
            'rewards': rewards.copy(),
            'total_reward': sum(rewards)
        }
        
        # Add observation features for each agent
        if isinstance(observations, tuple):
            # Multi-agent observations - tuple of observations
            for i in range(min(self.n_agents, len(observations))):
                agent_obs = observations[i]
                if len(agent_obs) > 0 and agent_obs[0][0] > 0.5:  # Vehicle present
                    ego = agent_obs[0]
                    step_data[f'agent_{i}_x'] = float(ego[1])
                    step_data[f'agent_{i}_y'] = float(ego[2])
                    step_data[f'agent_{i}_speed'] = float(np.sqrt(ego[3]**2 + ego[4]**2))
        else:
            # Single observation - use for all agents
            for i in range(self.n_agents):
                if isinstance(observations, np.ndarray) and len(observations) > 0 and observations[0][0] > 0.5:
                    ego = observations[0]
                    step_data[f'agent_{i}_x'] = float(ego[1])
                    step_data[f'agent_{i}_y'] = float(ego[2])
                    step_data[f'agent_{i}_speed'] = float(np.sqrt(ego[3]**2 + ego[4]**2))
        
        self.current_episode_data.append(step_data)
    
    def display_statistics(self):
        """Display real-time statistics."""
        print("\r", end="")  # Clear line
        
        # Calculate current stats
        stats_text = []
        for i in range(self.n_agents):
            if self.episode_stats['rewards'][i]:
                avg_reward = np.mean(self.episode_stats['rewards'][i][-10:])  # Last 10 steps
                total_reward = sum(self.episode_stats['rewards'][i])
                lane_changes = self.episode_stats['lane_changes'][i]
                
                stats_text.append(f"Agent{i+1}: R={avg_reward:.2f} LC={lane_changes}")
        
        print(" | ".join(stats_text), end="", flush=True)
    
    def print_agent_summary(self):
        """Print summary of agent behaviors."""
        print(f"\n🤖 Agent Behavior Summary:")
        
        agent_types = ["Conservative", "Aggressive", "Speed Keeper", "Adaptive"]
        
        for i in range(self.n_agents):
            agent_type = agent_types[i] if i < len(agent_types) else f"Agent {i+1}"
            
            if self.episode_stats['rewards'][i]:
                total_reward = sum(self.episode_stats['rewards'][i])
                avg_reward = np.mean(self.episode_stats['rewards'][i])
                lane_changes = self.episode_stats['lane_changes'][i]
                collisions = self.episode_stats['collisions'][i]
                
                # Most common action
                if self.episode_stats['actions'][i]:
                    actions = self.episode_stats['actions'][i]
                    most_common_action = max(set(actions), key=actions.count)
                    action_name = self.action_names.get(most_common_action, "Unknown")
                else:
                    action_name = "None"
                
                print(f"  {agent_type} (Agent {i+1}):")
                print(f"    Total Reward: {total_reward:.2f}")
                print(f"    Avg Reward: {avg_reward:.3f}")
                print(f"    Lane Changes: {lane_changes}")
                print(f"    Collisions: {collisions}")
                print(f"    Favorite Action: {action_name}")
    
    def run_visual_demo(self, max_episodes=5, max_steps_per_episode=300):
        """Run the visual multi-agent demo."""
        print("🚗 Multi-Agent Visual Highway Demo")
        print("=" * 50)
        print(f"👥 Agents: {self.n_agents}")
        print(f"🎭 Agent Types:")
        print("   Agent 1 (Red): Conservative Driver")
        print("   Agent 2 (Green): Aggressive Driver") 
        print("   Agent 3 (Blue): Speed Keeper")
        print("   Agent 4 (Yellow): Adaptive Driver")
        print()
        print("🎮 Controls: Press Ctrl+C to stop")
        print("=" * 50)
        
        # Create environment
        env = self.create_environment()
        
        try:
            for episode in range(max_episodes):
                print(f"\n🎬 Episode {episode + 1}/{max_episodes}")
                
                # Reset environment
                obs, info = env.reset()
                episode_step = 0
                episode_rewards = [0.0] * self.n_agents
                
                # Reset episode statistics
                for i in range(self.n_agents):
                    self.episode_stats['rewards'][i].clear()
                    self.episode_stats['actions'][i].clear()
                    self.episode_stats['lane_changes'][i] = 0
                    self.episode_stats['collisions'][i] = 0
                
                self.current_episode_data.clear()
                
                while episode_step < max_steps_per_episode:
                    # Generate actions for all agents
                    actions = self.generate_intelligent_actions(obs)
                    
                    # Custom multi-agent step
                    next_obs, reward, terminated, truncated, info = self._multi_agent_step(env, actions)
                    
                    # Handle multi-agent rewards
                    if isinstance(reward, (list, tuple)):
                        reward_list = list(reward)
                    else:
                        # Single reward - distribute to all agents with personality variations
                        base_reward = reward if isinstance(reward, (int, float)) else 0.0
                        reward_list = []
                        for i in range(self.n_agents):
                            agent_reward = base_reward
                            if i == 1:  # Aggressive agent gets penalty for risky behavior
                                agent_reward *= 0.9
                            elif i == 0:  # Conservative agent gets bonus for safety
                                agent_reward *= 1.1
                            reward_list.append(agent_reward)
                    
                    # Ensure we have the right number of rewards
                    while len(reward_list) < self.n_agents:
                        reward_list.append(0.0)
                    
                    # Update statistics
                    self.update_statistics(actions, reward_list, info)
                    
                    # Collect data
                    self.collect_step_data(obs, actions, reward_list, episode_step)
                    
                    # Update episode rewards
                    for i in range(self.n_agents):
                        episode_rewards[i] += reward_list[i]
                    
                    # Render environment
                    env.render()
                    
                    # Display real-time stats every 10 steps
                    if episode_step % 10 == 0:
                        self.display_statistics()
                    
                    # Control simulation speed
                    time.sleep(0.05)  # 20 FPS
                    
                    # Check if episode should end
                    if isinstance(terminated, (list, tuple)):
                        done = any(terminated) or any(truncated) if isinstance(truncated, (list, tuple)) else any(terminated)
                    else:
                        done = terminated or truncated
                    
                    if done:
                        break
                    
                    obs = next_obs
                    episode_step += 1
                
                # Episode summary
                print(f"\n✅ Episode {episode + 1} completed ({episode_step} steps)")
                print(f"   Episode Rewards: {[f'{r:.2f}' for r in episode_rewards]}")
                
                # Save episode data
                if self.current_episode_data:
                    episode_summary = {
                        'episode': episode + 1,
                        'steps': episode_step,
                        'total_rewards': episode_rewards,
                        'data': self.current_episode_data.copy()
                    }
                    self.collected_data.append(episode_summary)
                
                # Show agent behavior summary
                self.print_agent_summary()
                
                # Brief pause between episodes
                time.sleep(2)
        
        except KeyboardInterrupt:
            print(f"\n⏹️  Demo stopped by user")
        
        finally:
            env.close()
            
            # Final summary
            print(f"\n🏁 Multi-Agent Demo Complete!")
            print(f"   Episodes completed: {len(self.collected_data)}")
            
            if self.collected_data:
                total_steps = sum(ep['steps'] for ep in self.collected_data)
                total_rewards = [sum(ep['total_rewards'][i] for ep in self.collected_data) 
                               for i in range(self.n_agents)]
                
                print(f"   Total steps: {total_steps}")
                print(f"   Final agent rewards: {[f'{r:.2f}' for r in total_rewards]}")
                
                # Save collected data
                self.save_collected_data()
    
    def save_collected_data(self):
        """Save collected multi-agent data."""
        if not self.collected_data:
            return
        
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/multiagent_demo_{self.n_agents}agents_{timestamp}.json"
        
        # Create data directory if it doesn't exist
        import os
        os.makedirs("data", exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # Save data
        data_to_save = convert_numpy_types({
            'n_agents': self.n_agents,
            'scenario': self.scenario,
            'timestamp': timestamp,
            'episodes': self.collected_data
        })
        
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        print(f"💾 Multi-agent data saved to: {filename}")

def main():
    """Main function to run the multi-agent demo."""
    print("🚗 Multi-Agent Highway Visualization")
    print("Choose your demo configuration:")
    print()
    
    # Get user preferences
    try:
        n_agents = int(input("Number of agents (2-6, default 4): ") or "4")
        n_agents = max(2, min(6, n_agents))  # Clamp between 2-6
    except ValueError:
        n_agents = 4
    
    print("\nAvailable scenarios:")
    print("1. free_flow - Light traffic")
    print("2. dense_commuting - Heavy traffic (recommended)")
    print("3. stop_and_go - Congested traffic")
    
    try:
        scenario_choice = input("Choose scenario (1-3, default 2): ") or "2"
        scenarios = {"1": "free_flow", "2": "dense_commuting", "3": "stop_and_go"}
        scenario = scenarios.get(scenario_choice, "dense_commuting")
    except:
        scenario = "dense_commuting"
    
    try:
        episodes = int(input("Number of episodes (1-10, default 3): ") or "3")
        episodes = max(1, min(10, episodes))
    except ValueError:
        episodes = 3
    
    print(f"\n🎯 Configuration:")
    print(f"   Agents: {n_agents}")
    print(f"   Scenario: {scenario}")
    print(f"   Episodes: {episodes}")
    print()
    
    # Create and run demo
    demo = MultiAgentVisualDemo(n_agents=n_agents, scenario=scenario)
    demo.run_visual_demo(max_episodes=episodes)

if __name__ == "__main__":
    main()