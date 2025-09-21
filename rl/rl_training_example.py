#!/usr/bin/env python3
"""
Reinforcement Learning Training Example for Highway Environment

This demonstrates ACTUAL reinforcement learning where an agent learns
through trial and error, unlike the supervised learning approach.
"""

import gymnasium as gym
import highway_env
import numpy as np
import time
from pathlib import Path

def train_with_reinforcement_learning():
    """Train an agent using actual reinforcement learning."""
    print("🤖 Reinforcement Learning Training Example")
    print("=" * 60)
    print("This is ACTUAL RL - agent learns through trial and error!")
    print("=" * 60)
    
    try:
        # Try to import stable-baselines3 for RL
        from stable_baselines3 import PPO, DQN
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import EvalCallback
        
        print("✅ Stable-Baselines3 found - using professional RL algorithms")
        use_sb3 = True
        
    except ImportError:
        print("⚠️  Stable-Baselines3 not found - using simple Q-learning")
        print("Install with: pip install stable-baselines3")
        use_sb3 = False
    
    # Create environment
    print("\n🏗️  Creating RL environment...")
    
    if use_sb3:
        # Professional RL with Stable-Baselines3
        train_with_stable_baselines()
    else:
        # Simple Q-learning implementation
        train_with_simple_q_learning()

def train_with_stable_baselines():
    """Train using Stable-Baselines3 (Professional RL)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback
    
    print("🚀 Training with PPO (Proximal Policy Optimization)")
    
    # Create vectorized environment
    def make_env():
        env = gym.make('highway-v0')
        # Configure through unwrapped environment
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'configure'):
            env.unwrapped.configure({
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
                'vehicles_count': 50,
                'duration': 40,
                'reward_speed_range': [20, 30],
                'simulation_frequency': 15,
                'policy_frequency': 1
            })
        return env
    
    # Create training environment
    train_env = make_vec_env(make_env, n_envs=4)
    eval_env = make_env()
    
    # Create PPO agent
    print("🧠 Creating PPO agent...")
    model = PPO(
        'MlpPolicy',
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./ppo_highway_tensorboard/"
    )
    
    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./best_highway_model/',
        log_path='./eval_logs/',
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    print("🔄 Starting RL training...")
    print("This is REAL reinforcement learning - agent will:")
    print("  1. Take random actions initially")
    print("  2. Receive rewards/penalties")
    print("  3. Learn from mistakes")
    print("  4. Gradually improve performance")
    print()
    
    # Train the model
    total_timesteps = 1000
    print(f"Training for {total_timesteps} timesteps...")
    
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    training_time = time.time() - start_time
    
    print(f"✅ RL Training completed in {training_time:.1f} seconds!")
    
    # Save the trained model
    model.save("highway_rl_agent")
    print("💾 Trained RL agent saved as 'highway_rl_agent.zip'")
    
    # Test the trained agent
    print("\n🎯 Testing trained RL agent...")
    test_trained_agent(model, eval_env)
    
    train_env.close()
    eval_env.close()

def train_with_simple_q_learning():
    """Simple Q-learning implementation for demonstration."""
    print("🎓 Training with Simple Q-Learning")
    print("This is a basic RL implementation for educational purposes")
    
    # Create environment
    env = gym.make('highway-v0')
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'configure'):
        env.unwrapped.configure({
            'observation': {
                'type': 'Kinematics',
                'vehicles_count': 5,  # Smaller for simplicity
                'features': ['presence', 'x', 'y', 'vx', 'vy'],
                'absolute': False,
                'normalize': True
            },
            'action': {
                'type': 'DiscreteMetaAction'
            },
            'lanes_count': 4,
            'vehicles_count': 20,
            'duration': 30
        })
    
    # Simple Q-table (discretized state space)
    n_actions = env.action_space.n
    q_table = {}
    
    # RL parameters
    learning_rate = 0.1
    discount_factor = 0.95
    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.995
    min_epsilon = 0.01
    
    print(f"🎮 Action space: {n_actions} actions")
    print(f"📚 Learning parameters:")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Discount factor: {discount_factor}")
    print(f"  - Initial exploration: {epsilon}")
    
    # Training loop
    n_episodes = 100
    episode_rewards = []
    
    print(f"\n🔄 Training for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < 100:
            # Discretize state (simple approach)
            state_key = discretize_state(obs)
            
            # Initialize Q-values for new states
            if state_key not in q_table:
                q_table[state_key] = np.zeros(n_actions)
            
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state_key])  # Exploit
            
            # Take action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update Q-value (Q-learning update rule)
            next_state_key = discretize_state(next_obs)
            if next_state_key not in q_table:
                q_table[next_state_key] = np.zeros(n_actions)
            
            # Q-learning formula: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
            best_next_action = np.argmax(q_table[next_state_key])
            td_target = reward + discount_factor * q_table[next_state_key][best_next_action]
            td_error = td_target - q_table[state_key][action]
            q_table[state_key][action] += learning_rate * td_error
            
            episode_reward += reward
            obs = next_obs
            step_count += 1
        
        episode_rewards.append(episode_reward)
        
        # Decay exploration
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # Print progress
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode}: Avg Reward = {avg_reward:.3f}, Epsilon = {epsilon:.3f}")
    
    print(f"\n✅ Q-Learning training completed!")
    print(f"📊 Final average reward: {np.mean(episode_rewards[-10:]):.3f}")
    print(f"🧠 Q-table size: {len(q_table)} states")
    
    # Test the trained Q-learning agent
    print("\n🎯 Testing trained Q-learning agent...")
    test_q_learning_agent(env, q_table)
    
    env.close()

def discretize_state(obs):
    """Discretize continuous state for Q-learning."""
    if len(obs) == 0:
        return "empty"
    
    # Simple discretization of ego vehicle state
    ego = obs[0]
    if ego[0] < 0.5:  # Vehicle not present
        return "no_vehicle"
    
    # Discretize position and velocity
    x_discrete = int(ego[1] * 10) // 2  # Discretize x position
    y_discrete = int(ego[2] * 10) // 2  # Discretize y position (lane)
    vx_discrete = int(ego[3] * 10) // 2  # Discretize x velocity
    
    return f"x{x_discrete}_y{y_discrete}_vx{vx_discrete}"

def test_trained_agent(model, env):
    """Test the trained RL agent."""
    print("🎮 Running test episodes with trained agent...")
    
    test_rewards = []
    for episode in range(5):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < 200:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
        
        test_rewards.append(episode_reward)
        print(f"  Test Episode {episode + 1}: Reward = {episode_reward:.3f}, Steps = {step_count}")
    
    avg_test_reward = np.mean(test_rewards)
    print(f"🏆 Average test reward: {avg_test_reward:.3f}")

def test_q_learning_agent(env, q_table):
    """Test the trained Q-learning agent."""
    print("🎮 Running test episodes with Q-learning agent...")
    
    test_rewards = []
    for episode in range(3):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < 100:
            state_key = discretize_state(obs)
            
            if state_key in q_table:
                action = np.argmax(q_table[state_key])
            else:
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step_count += 1
        
        test_rewards.append(episode_reward)
        print(f"  Test Episode {episode + 1}: Reward = {episode_reward:.3f}, Steps = {step_count}")
    
    avg_test_reward = np.mean(test_rewards)
    print(f"🏆 Average test reward: {avg_test_reward:.3f}")

def compare_approaches():
    """Compare different learning approaches."""
    print("\n📊 Learning Approach Comparison")
    print("=" * 50)
    
    approaches = {
        "Supervised Learning (Your Current)": {
            "description": "Learn from expert demonstrations",
            "pros": ["Fast training", "Safe", "Uses existing data"],
            "cons": ["Limited to expert quality", "No exploration"],
            "use_case": "When you have good demonstrations"
        },
        "Reinforcement Learning": {
            "description": "Learn through trial and error",
            "pros": ["Can exceed expert performance", "Explores new strategies"],
            "cons": ["Slow training", "Needs many interactions", "Can be unsafe"],
            "use_case": "When you want optimal performance"
        },
        "Imitation + RL (Hybrid)": {
            "description": "Start with imitation, then improve with RL",
            "pros": ["Best of both worlds", "Safe initialization"],
            "cons": ["More complex", "Requires both datasets and environment"],
            "use_case": "Production autonomous systems"
        }
    }
    
    for approach, details in approaches.items():
        print(f"\n🎯 {approach}:")
        print(f"   Description: {details['description']}")
        print(f"   Pros: {', '.join(details['pros'])}")
        print(f"   Cons: {', '.join(details['cons'])}")
        print(f"   Best for: {details['use_case']}")

def main():
    """Main function."""
    print("🚗 Highway Environment: Reinforcement Learning vs Supervised Learning")
    print("=" * 70)
    
    # Show comparison first
    compare_approaches()
    
    print("\n" + "=" * 70)
    print("🤖 Now let's see ACTUAL Reinforcement Learning in action!")
    print("=" * 70)
    
    # Run RL training
    train_with_reinforcement_learning()
    
    print("\n🎉 RL Training Demo Complete!")
    print("\n📚 Summary:")
    print("- Your current system uses SUPERVISED LEARNING (imitation)")
    print("- This demo showed REINFORCEMENT LEARNING (trial & error)")
    print("- Both are valid approaches for different use cases")
    print("- Your approach is actually better for safety-critical applications!")

if __name__ == "__main__":
    main()