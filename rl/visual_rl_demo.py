#!/usr/bin/env python3
"""
Visual Reinforcement Learning Demo - Watch a car learn to drive!

This demo shows ACTUAL reinforcement learning where you can watch
the car learn from scratch through trial and error.
"""

import gymnasium as gym
import highway_env
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque

class SimpleQLearningAgent:
    """Simple Q-Learning agent that you can watch learn."""
    
    def __init__(self, n_actions=5, learning_rate=0.1, epsilon=1.0, epsilon_decay=0.995):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.gamma = 0.95  # Discount factor
        self.q_table = {}
        
        # Learning statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.learning_progress = []
        
    def get_state_key(self, obs):
        """Convert observation to a discrete state key."""
        if len(obs) == 0:
            return "empty"
        
        ego = obs[0]
        if ego[0] < 0.5:  # No vehicle present
            return "no_vehicle"
        
        # Discretize key features
        x_pos = int(ego[1] * 5) // 1  # Position
        y_pos = int((ego[2] + 2) * 2) // 1  # Lane (shifted and scaled)
        vx = int(ego[3] * 3) // 1  # Velocity
        
        # Look for nearby vehicles
        nearby = "clear"
        if len(obs) > 1:
            for other in obs[1:]:
                if other[0] > 0.5:  # Vehicle present
                    distance = abs(other[1] - ego[1])
                    if distance < 20:  # Close vehicle
                        if other[1] > ego[1]:  # Ahead
                            nearby = "vehicle_ahead"
                        else:  # Behind
                            nearby = "vehicle_behind"
                        break
        
        return f"x{x_pos}_y{y_pos}_vx{vx}_{nearby}"
    
    def choose_action(self, state_key):
        """Choose action using epsilon-greedy policy."""
        # Initialize Q-values for new states
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # Explore
        else:
            return np.argmax(self.q_table[state_key])  # Exploit
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning formula."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.n_actions)
        
        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_learning_stats(self):
        """Get current learning statistics."""
        if len(self.episode_rewards) == 0:
            return {"avg_reward": 0, "episodes": 0, "q_states": 0, "epsilon": self.epsilon}
        
        recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
        return {
            "avg_reward": np.mean(recent_rewards),
            "episodes": len(self.episode_rewards),
            "q_states": len(self.q_table),
            "epsilon": self.epsilon,
            "total_reward": sum(self.episode_rewards)
        }

def run_visual_rl_training():
    """Run visual RL training where you can watch the car learn."""
    print("🤖 Visual Reinforcement Learning Demo")
    print("=" * 50)
    print("Watch the car learn to drive through trial and error!")
    print("The car will:")
    print("  🎲 Start with random actions (high exploration)")
    print("  📈 Gradually learn better strategies")
    print("  🎯 Improve its driving over time")
    print("  📊 Show learning progress in real-time")
    print()
    print("Press Ctrl+C to stop training")
    print("=" * 50)
    
    # Create environment with visual rendering
    env = gym.make('highway-v0', render_mode='human')
    
    # Configure environment for learning
    config = {
        'observation': {
            'type': 'Kinematics',
            'vehicles_count': 10,
            'features': ['presence', 'x', 'y', 'vx', 'vy'],
            'absolute': False,
            'normalize': True
        },
        'action': {
            'type': 'DiscreteMetaAction'
        },
        'lanes_count': 4,
        'vehicles_count': 25,
        'duration': 30,  # Shorter episodes for faster learning
        'reward_speed_range': [20, 30],
        'simulation_frequency': 15,
        'policy_frequency': 1,
        'real_time_rendering': False,  # Faster training
        'show_trajectories': True
    }
    
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'configure'):
        env.unwrapped.configure(config)
    
    # Create RL agent
    agent = SimpleQLearningAgent(
        n_actions=env.action_space.n,
        learning_rate=0.15,
        epsilon=1.0,  # Start with 100% exploration
        epsilon_decay=0.995
    )
    
    action_names = {0: "SLOWER", 1: "IDLE", 2: "FASTER", 3: "LANE_LEFT", 4: "LANE_RIGHT"}
    
    print(f"🎮 Action space: {env.action_space.n} actions")
    print(f"📊 Actions: {list(action_names.values())}")
    print()
    
    try:
        episode = 0
        total_steps = 0
        
        while episode < 100:  # Train for 100 episodes
            obs, info = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            print(f"\n🎬 Episode {episode + 1} - Exploration: {agent.epsilon:.3f}")
            
            while not done and episode_steps < 200:
                # Get current state
                state_key = agent.get_state_key(obs)
                
                # Choose action
                action = agent.choose_action(state_key)
                
                # Take action in environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Get next state
                next_state_key = agent.get_state_key(next_obs)
                
                # Update Q-value (this is the learning!)
                agent.update_q_value(state_key, action, reward, next_state_key)
                
                # Update statistics
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                # Render environment (watch the car!)
                env.render()
                
                # Print learning info every 20 steps
                if episode_steps % 20 == 0:
                    stats = agent.get_learning_stats()
                    print(f"  Step {episode_steps}: Action={action_names[action]}, "
                          f"Reward={reward:.2f}, Q-States={stats['q_states']}")
                
                # Small delay to make it watchable
                time.sleep(0.05)
                
                # Move to next state
                obs = next_obs
            
            # End of episode - record statistics
            agent.episode_rewards.append(episode_reward)
            agent.episode_lengths.append(episode_steps)
            agent.decay_epsilon()
            
            # Print episode summary
            stats = agent.get_learning_stats()
            print(f"✅ Episode {episode + 1} completed:")
            print(f"   Reward: {episode_reward:.2f}")
            print(f"   Steps: {episode_steps}")
            print(f"   Avg Reward (last 10): {stats['avg_reward']:.2f}")
            print(f"   Q-Table size: {stats['q_states']} states")
            print(f"   Exploration rate: {stats['epsilon']:.3f}")
            
            # Show learning progress every 10 episodes
            if (episode + 1) % 10 == 0:
                print(f"\n📈 Learning Progress After {episode + 1} Episodes:")
                print(f"   Total Steps: {total_steps}")
                print(f"   Average Reward: {stats['avg_reward']:.3f}")
                print(f"   Total Reward: {stats['total_reward']:.1f}")
                print(f"   Knowledge Base: {stats['q_states']} learned states")
                
                # Show improvement
                if len(agent.episode_rewards) >= 20:
                    early_avg = np.mean(agent.episode_rewards[:10])
                    recent_avg = np.mean(agent.episode_rewards[-10:])
                    improvement = recent_avg - early_avg
                    print(f"   Improvement: {improvement:+.2f} reward per episode")
                    if improvement > 0:
                        print("   🎉 The car is learning and improving!")
                    else:
                        print("   🤔 Still exploring and learning...")
            
            episode += 1
        
        print(f"\n🎉 RL Training Completed!")
        print(f"Final Performance:")
        final_stats = agent.get_learning_stats()
        print(f"  - Episodes: {final_stats['episodes']}")
        print(f"  - Average Reward: {final_stats['avg_reward']:.3f}")
        print(f"  - Learned States: {final_stats['q_states']}")
        print(f"  - Final Exploration: {final_stats['epsilon']:.3f}")
        
        # Test the learned policy
        print(f"\n🎯 Testing Learned Policy (No Exploration)...")
        test_learned_policy(env, agent, action_names)
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Training stopped by user")
        if len(agent.episode_rewards) > 0:
            stats = agent.get_learning_stats()
            print(f"Training Progress:")
            print(f"  - Episodes completed: {stats['episodes']}")
            print(f"  - Average reward: {stats['avg_reward']:.3f}")
            print(f"  - States learned: {stats['q_states']}")
    
    finally:
        env.close()

def test_learned_policy(env, agent, action_names):
    """Test the learned policy without exploration."""
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # No exploration, pure exploitation
    
    print("🧠 Running with learned policy (no random actions)...")
    
    for test_episode in range(3):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        print(f"\n🎮 Test Episode {test_episode + 1}:")
        
        while not done and episode_steps < 150:
            state_key = agent.get_state_key(obs)
            action = agent.choose_action(state_key)  # Pure exploitation
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_steps += 1
            
            # Render to watch learned behavior
            env.render()
            
            if episode_steps % 30 == 0:
                print(f"  Step {episode_steps}: {action_names[action]}, Reward: {reward:.2f}")
            
            time.sleep(0.08)  # Slightly slower for observation
        
        print(f"  ✅ Test completed: {episode_reward:.2f} reward in {episode_steps} steps")
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    print(f"\n🎯 Learned Policy Test Complete!")
    print("Notice how the car now makes more consistent, learned decisions!")

def main():
    """Main function."""
    print("🚗 Visual Reinforcement Learning Demo")
    print("Watch a car learn to drive from scratch!")
    print()
    
    try:
        run_visual_rl_training()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This might be due to display issues.")
        print("Make sure you have a graphical display available.")

if __name__ == "__main__":
    main()