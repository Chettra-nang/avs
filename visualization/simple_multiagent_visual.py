#!/usr/bin/env python3
"""
Simple Multi-Agent Visual Demo - Watch multiple car behaviors!

This demo simulates multiple agents with different driving personalities
in a single highway-env environment with visual feedback.
"""

import gymnasium as gym
import highway_env
import numpy as np
import time
from typing import List, Dict

class SimpleMultiAgentDemo:
    """Simple multi-agent demo that works with highway-env limitations."""
    
    def __init__(self, n_agents=4):
        self.n_agents = n_agents
        self.env = None
        
        # Agent personalities
        self.agent_personalities = [
            "Conservative",  # Agent 0
            "Aggressive",    # Agent 1  
            "Speed Keeper",  # Agent 2
            "Adaptive"       # Agent 3
        ]
        
        # Statistics
        self.agent_stats = {
            'rewards': [[] for _ in range(n_agents)],
            'actions': [[] for _ in range(n_agents)],
            'decisions': [[] for _ in range(n_agents)]
        }
        
        self.action_names = {
            0: "SLOWER", 1: "IDLE", 2: "FASTER", 
            3: "LANE_LEFT", 4: "LANE_RIGHT"
        }
        
        # Current active agent (rotates each episode)
        self.active_agent = 0
    
    def create_environment(self):
        """Create highway environment with visual rendering."""
        print("🚗 Creating Highway Environment")
        print(f"   Simulating {self.n_agents} different agent personalities")
        print("   (Due to highway-env limitations, agents take turns controlling)")
        
        env = gym.make('highway-v0', render_mode='human')
        
        config = {
            'observation': {
                'type': 'Kinematics',
                'vehicles_count': 15,
                'features': ['presence', 'x', 'y', 'vx', 'vy'],
                'absolute': False,
                'normalize': True
            },
            'action': {
                'type': 'DiscreteMetaAction'
            },
            'lanes_count': 4,
            'vehicles_count': 40,  # Busy traffic
            'duration': 30,
            'reward_speed_range': [20, 30],
            'simulation_frequency': 15,
            'policy_frequency': 1,
            'real_time_rendering': True,
            'show_trajectories': True,
            'render_agent': True,
            'centering_position': [0.3, 0.5],
            'scaling': 5.5
        }
        
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'configure'):
            env.unwrapped.configure(config)
        
        self.env = env
        return env
    
    def get_agent_action(self, observation: np.ndarray, agent_id: int, step: int) -> int:
        """Get action based on agent personality."""
        if len(observation) == 0:
            return 1  # IDLE
        
        ego = observation[0]
        others = observation[1:] if len(observation) > 1 else np.array([])
        
        personality = self.agent_personalities[agent_id % len(self.agent_personalities)]
        
        if personality == "Conservative":
            return self._conservative_policy(ego, others, step)
        elif personality == "Aggressive":
            return self._aggressive_policy(ego, others, step)
        elif personality == "Speed Keeper":
            return self._speed_keeper_policy(ego, others, step)
        else:  # Adaptive
            return self._adaptive_policy(ego, others, step)
    
    def _conservative_policy(self, ego: np.ndarray, others: np.ndarray, step: int) -> int:
        """Conservative driving: safety first."""
        if ego[0] < 0.5:
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
                if closest_distance < 25:  # Conservative following distance
                    return 0  # SLOWER
        
        # Maintain moderate speed
        if speed < 20:
            return 2  # FASTER
        elif speed > 26:
            return 0  # SLOWER
        else:
            return 1  # IDLE
    
    def _aggressive_policy(self, ego: np.ndarray, others: np.ndarray, step: int) -> int:
        """Aggressive driving: speed and lane changes."""
        if ego[0] < 0.5:
            return 1
        
        speed = np.sqrt(ego[3]**2 + ego[4]**2)
        
        # Frequent lane changes
        if step % 20 == 0 and np.random.random() < 0.3:  # 30% chance every 20 steps
            return np.random.choice([3, 4])  # LANE_LEFT or LANE_RIGHT
        
        # Prefer higher speeds
        if speed < 28:
            return 2  # FASTER
        elif speed > 35:
            return 0  # SLOWER
        else:
            return 1  # IDLE
    
    def _speed_keeper_policy(self, ego: np.ndarray, others: np.ndarray, step: int) -> int:
        """Speed keeper: maintain target speed."""
        if ego[0] < 0.5:
            return 1
        
        speed = np.sqrt(ego[3]**2 + ego[4]**2)
        target_speed = 25.0
        
        if speed < target_speed - 3:
            return 2  # FASTER
        elif speed > target_speed + 3:
            return 0  # SLOWER
        else:
            return 1  # IDLE
    
    def _adaptive_policy(self, ego: np.ndarray, others: np.ndarray, step: int) -> int:
        """Adaptive: changes behavior based on traffic."""
        if ego[0] < 0.5:
            return 1
        
        # Count nearby vehicles
        nearby_count = 0
        if len(others) > 0:
            distances = np.sqrt((others[:, 1] - ego[1])**2 + (others[:, 2] - ego[2])**2)
            nearby_count = np.sum((others[:, 0] > 0.5) & (distances < 30))
        
        speed = np.sqrt(ego[3]**2 + ego[4]**2)
        
        if nearby_count > 4:  # Dense traffic - be conservative
            if speed > 22:
                return 0  # SLOWER
            else:
                return 1  # IDLE
        else:  # Light traffic - be more aggressive
            if speed < 27:
                return 2  # FASTER
            elif step % 30 == 0 and np.random.random() < 0.2:
                return np.random.choice([3, 4])  # Lane change
            else:
                return 1  # IDLE
    
    def run_demo(self, episodes_per_agent=2):
        """Run the multi-agent personality demo."""
        print("🎭 Multi-Agent Personality Demo")
        print("=" * 50)
        print("Each episode showcases a different driving personality:")
        for i, personality in enumerate(self.agent_personalities):
            print(f"  Agent {i+1}: {personality}")
        print()
        print("🎮 Press Ctrl+C to stop early")
        print("=" * 50)
        
        env = self.create_environment()
        
        try:
            total_episodes = 0
            
            # Run episodes for each agent personality
            for agent_id in range(self.n_agents):
                personality = self.agent_personalities[agent_id]
                print(f"\n🎭 Now showcasing: {personality} Driver (Agent {agent_id + 1})")
                print("-" * 40)
                
                for episode in range(episodes_per_agent):
                    print(f"\n🎬 Episode {episode + 1}/{episodes_per_agent} - {personality}")
                    
                    # Reset environment
                    obs, info = env.reset()
                    episode_reward = 0
                    episode_steps = 0
                    episode_actions = []
                    
                    while episode_steps < 200:  # Max steps per episode
                        # Get action from current agent personality
                        action = self.get_agent_action(obs, agent_id, episode_steps)
                        episode_actions.append(action)
                        
                        # Step environment
                        next_obs, reward, terminated, truncated, info = env.step(action)
                        episode_reward += reward
                        
                        # Render
                        env.render()
                        
                        # Display current action every 20 steps
                        if episode_steps % 20 == 0:
                            action_name = self.action_names.get(action, "UNKNOWN")
                            print(f"  Step {episode_steps}: {personality} chose {action_name} (Reward: {reward:.3f})")
                        
                        # Control speed
                        time.sleep(0.05)  # 20 FPS
                        
                        # Check if done
                        if terminated or truncated:
                            break
                        
                        obs = next_obs
                        episode_steps += 1
                    
                    # Episode summary
                    total_episodes += 1
                    self.agent_stats['rewards'][agent_id].append(episode_reward)
                    self.agent_stats['actions'][agent_id].extend(episode_actions)
                    
                    # Action distribution for this episode
                    action_counts = {}
                    for a in episode_actions:
                        action_counts[a] = action_counts.get(a, 0) + 1
                    
                    most_common_action = max(action_counts, key=action_counts.get)
                    most_common_name = self.action_names.get(most_common_action, "UNKNOWN")
                    
                    print(f"  ✅ {personality} completed {episode_steps} steps")
                    print(f"     Total Reward: {episode_reward:.2f}")
                    print(f"     Favorite Action: {most_common_name} ({action_counts[most_common_action]} times)")
                    
                    # Brief pause between episodes
                    time.sleep(1)
                
                # Agent summary
                if self.agent_stats['rewards'][agent_id]:
                    avg_reward = np.mean(self.agent_stats['rewards'][agent_id])
                    total_reward = sum(self.agent_stats['rewards'][agent_id])
                    
                    print(f"\n📊 {personality} Summary:")
                    print(f"   Average Reward per Episode: {avg_reward:.3f}")
                    print(f"   Total Reward: {total_reward:.2f}")
                    
                    # Overall action preferences
                    all_actions = self.agent_stats['actions'][agent_id]
                    if all_actions:
                        action_dist = {}
                        for a in all_actions:
                            action_dist[a] = action_dist.get(a, 0) + 1
                        
                        print(f"   Action Preferences:")
                        for action, count in sorted(action_dist.items()):
                            name = self.action_names.get(action, f"Action_{action}")
                            percentage = (count / len(all_actions)) * 100
                            print(f"     {name}: {percentage:.1f}%")
        
        except KeyboardInterrupt:
            print(f"\n⏹️  Demo stopped by user")
        
        finally:
            env.close()
            
            # Final comparison
            print(f"\n🏆 Final Agent Comparison:")
            print("-" * 50)
            
            for i, personality in enumerate(self.agent_personalities):
                if self.agent_stats['rewards'][i]:
                    avg_reward = np.mean(self.agent_stats['rewards'][i])
                    episodes = len(self.agent_stats['rewards'][i])
                    print(f"{personality:12}: {avg_reward:6.3f} avg reward ({episodes} episodes)")
            
            print(f"\nTotal episodes completed: {total_episodes}")
            print("Thanks for watching the multi-agent personality demo! 🚗")

def main():
    """Main function."""
    print("🚗 Multi-Agent Personality Demo")
    print("Watch different driving personalities in action!")
    print()
    
    try:
        n_agents = int(input("Number of agent personalities to demo (2-4, default 4): ") or "4")
        n_agents = max(2, min(4, n_agents))
    except ValueError:
        n_agents = 4
    
    try:
        episodes = int(input("Episodes per personality (1-3, default 2): ") or "2")
        episodes = max(1, min(3, episodes))
    except ValueError:
        episodes = 2
    
    print(f"\n🎯 Configuration:")
    print(f"   Agent personalities: {n_agents}")
    print(f"   Episodes per personality: {episodes}")
    print(f"   Total episodes: {n_agents * episodes}")
    print()
    
    demo = SimpleMultiAgentDemo(n_agents=n_agents)
    demo.run_demo(episodes_per_agent=episodes)

if __name__ == "__main__":
    main()