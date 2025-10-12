#!/usr/bin/env python3
"""
Evaluate Trained Offline RL Models in Highway Environment

This script loads your trained BC/DQN/PPO models and evaluates them
in the actual Highway environment to see real performance.

Usage:
    # Evaluate BC model
    python3 scripts/evaluate_trained_models.py --model bc --checkpoint /home/chettra/ITC/Research/checkpoints/bc_pretrain/best_model.pt
    
    # Evaluate DQN model
    python3 scripts/evaluate_trained_models.py --model dqn --checkpoint /home/chettra/ITC/Research/checkpoints/offline_dqn/best_model.pt
    
    # Evaluate PPO model
    python3 scripts/evaluate_trained_models.py --model ppo --checkpoint /home/chettra/ITC/Research/checkpoints/offline_ppo/best_model.pt
    
    # Evaluate all three and compare
    python3 scripts/evaluate_trained_models.py --compare-all
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import gymnasium as gym

# Import CLIP
try:
    import clip
except ImportError:
    print("Installing CLIP...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'git+https://github.com/openai/CLIP.git'])
    import clip


# ============================================================================
# Model Architectures (must match training!)
# ============================================================================

class BCPolicy(nn.Module):
    """BC Policy - matches train_bc_ultrafast.py"""
    
    def __init__(self, feature_dim: int = 512, n_actions: int = 5):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_actions)
        )
    
    def forward(self, clip_features):
        return self.policy(clip_features)


class DQNNetwork(nn.Module):
    """DQN Network - matches train_dqn_ultrafast.py"""
    
    def __init__(self, feature_dim: int = 512, n_actions: int = 4):
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_actions)
        )
    
    def forward(self, clip_features):
        return self.q_network(clip_features)


class ActorCritic(nn.Module):
    """PPO Actor-Critic - matches train_ppo_ultrafast.py"""
    
    def __init__(self, feature_dim: int = 512, n_actions: int = 4):
        super().__init__()
        
        # Shared feature layer
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_actions)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, clip_features):
        shared_features = self.shared(clip_features)
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_logits, value.squeeze(-1)


# ============================================================================
# Agent Classes
# ============================================================================

class BCAgent:
    """BC Agent for evaluation"""
    
    def __init__(self, checkpoint_path: Path, device: str = 'cuda'):
        self.device = device
        self.policy = BCPolicy(feature_dim=512, n_actions=5).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'policy_state_dict' in checkpoint:
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
        else:
            # Try loading directly
            self.policy.load_state_dict(checkpoint)
        
        self.policy.eval()
        
        # Load CLIP
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        
        print(f"✅ Loaded BC model from {checkpoint_path}")
    
    def select_action(self, observation: np.ndarray) -> int:
        """Select action from observation"""
        with torch.no_grad():
            # Handle GrayscaleObservation (stacked frames)
            # Shape: (stack_size, height, width) - take last frame and convert to RGB
            if len(observation.shape) == 3:
                # Take the last frame from stack
                gray_frame = observation[-1]  # Shape: (height, width)
                # Convert to RGB by repeating channels
                rgb_frame = np.stack([gray_frame] * 3, axis=-1)  # Shape: (height, width, 3)
            else:
                rgb_frame = observation
            
            # Convert to PIL Image
            if rgb_frame.dtype != np.uint8:
                rgb_frame = (rgb_frame * 255).astype(np.uint8)
            
            img = Image.fromarray(rgb_frame)
            
            # Preprocess for CLIP
            img_tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)
            
            # Encode with CLIP
            clip_features = self.clip_model.encode_image(img_tensor).float()
            
            # Get action from policy
            action_logits = self.policy(clip_features)
            action = action_logits.argmax(dim=-1).item()
            
        return action


class DQNAgent:
    """DQN Agent for evaluation"""
    
    def __init__(self, checkpoint_path: Path, device: str = 'cuda'):
        self.device = device
        self.q_network = DQNNetwork(feature_dim=512, n_actions=4).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'q_net_state_dict' in checkpoint:
            self.q_network.load_state_dict(checkpoint['q_net_state_dict'])
        else:
            self.q_network.load_state_dict(checkpoint)
        
        self.q_network.eval()
        
        # Load CLIP
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        
        print(f"✅ Loaded DQN model from {checkpoint_path}")
    
    def select_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action using epsilon-greedy"""
        if np.random.random() < epsilon:
            return np.random.randint(0, 5)
        
        with torch.no_grad():
            # Handle GrayscaleObservation (stacked frames)
            # Shape: (stack_size, height, width) - take last frame and convert to RGB
            if len(observation.shape) == 3:
                # Take the last frame from stack
                gray_frame = observation[-1]  # Shape: (height, width)
                # Convert to RGB by repeating channels
                rgb_frame = np.stack([gray_frame] * 3, axis=-1)  # Shape: (height, width, 3)
            else:
                rgb_frame = observation
            
            # Convert to PIL Image
            if rgb_frame.dtype != np.uint8:
                rgb_frame = (rgb_frame * 255).astype(np.uint8)
            
            img = Image.fromarray(rgb_frame)
            
            # Preprocess for CLIP
            img_tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)
            
            # Encode with CLIP
            clip_features = self.clip_model.encode_image(img_tensor).float()
            
            # Get Q-values and select best action
            q_values = self.q_network(clip_features)
            action = q_values.argmax(dim=-1).item()
            
        return action


class PPOAgent:
    """PPO Agent for evaluation"""
    
    def __init__(self, checkpoint_path: Path, device: str = 'cuda'):
        self.device = device
        self.ac_net = ActorCritic(feature_dim=512, n_actions=4).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'ac_net_state_dict' in checkpoint:
            self.ac_net.load_state_dict(checkpoint['ac_net_state_dict'])
        else:
            self.ac_net.load_state_dict(checkpoint)
        
        self.ac_net.eval()
        
        # Load CLIP
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        
        print(f"✅ Loaded PPO model from {checkpoint_path}")
    
    def select_action(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Select action from policy"""
        with torch.no_grad():
            # Handle GrayscaleObservation (stacked frames)
            # Shape: (stack_size, height, width) - take last frame and convert to RGB
            if len(observation.shape) == 3:
                # Take the last frame from stack
                gray_frame = observation[-1]  # Shape: (height, width)
                # Convert to RGB by repeating channels
                rgb_frame = np.stack([gray_frame] * 3, axis=-1)  # Shape: (height, width, 3)
            else:
                rgb_frame = observation
            
            # Convert to PIL Image
            if rgb_frame.dtype != np.uint8:
                rgb_frame = (rgb_frame * 255).astype(np.uint8)
            
            img = Image.fromarray(rgb_frame)
            
            # Preprocess for CLIP
            img_tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)
            
            # Encode with CLIP
            clip_features = self.clip_model.encode_image(img_tensor).float()
            
            # Get action from actor
            action_logits, _ = self.ac_net(clip_features)
            
            if deterministic:
                action = action_logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(action_logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
            
        return action


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_agent(agent, env, n_episodes: int = 10, render: bool = False) -> Dict:
    """Evaluate agent for n episodes"""
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    action_names = ['LANE_LEFT', 'IDLE', 'LANE_RIGHT', 'FASTER']
    
    print(f"\nEvaluating for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action
            action = agent.select_action(obs)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Check success (didn't crash)
        if not info.get('crashed', False):
            success_count += 1
        
        print(f"  Episode {episode+1}/{n_episodes}: Reward={episode_reward:.2f}, Length={episode_length}, Success={not info.get('crashed', False)}")
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': success_count / n_episodes,
        'episode_rewards': episode_rewards,
    }
    
    return results


def create_highway_env():
    """Create Highway environment matching training config"""
    
    # Import highway_env
    try:
        import highway_env
    except ImportError:
        print("Installing highway_env...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'highway-env'])
        import highway_env
    
    env = gym.make('highway-v0', render_mode='rgb_array')
    
    # Configure to match training (GrayscaleObservation)
    env.unwrapped.config.update({
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 256),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],
            "scaling": 1.75,
        },
        "action": {
            "type": "DiscreteMetaAction",  # 5 actions: SLOWER, IDLE, FASTER, LANE_LEFT, LANE_RIGHT
        },
        "policy_frequency": 2,
        "duration": 40,
        "simulation_frequency": 15,
        "screen_width": 600,
        "screen_height": 150,
        "lanes_count": 4,
        "vehicles_count": 50,
        "vehicles_density": 1,
        "reward_speed_range": [20, 30],
        "normalize_reward": True,
    })
    
    return env


def compare_all_models(checkpoint_dir: Path, n_episodes: int = 10):
    """Compare all three trained models"""
    
    print("="*60)
    print("🏆 COMPARING ALL TRAINED MODELS")
    print("="*60)
    
    # Create environment
    env = create_highway_env()
    
    results = {}
    
    # BC
    bc_checkpoint = checkpoint_dir / 'bc_pretrain' / 'best_model.pt'
    if bc_checkpoint.exists():
        print("\n[1/3] Evaluating BC (Behavior Cloning)...")
        print("-"*60)
        bc_agent = BCAgent(bc_checkpoint)
        results['BC'] = evaluate_agent(bc_agent, env, n_episodes)
    
    # DQN
    dqn_checkpoint = checkpoint_dir / 'offline_dqn' / 'best_model.pt'
    if dqn_checkpoint.exists():
        print("\n[2/3] Evaluating DQN (Deep Q-Network)...")
        print("-"*60)
        dqn_agent = DQNAgent(dqn_checkpoint)
        results['DQN'] = evaluate_agent(dqn_agent, env, n_episodes)
    
    # PPO
    ppo_checkpoint = checkpoint_dir / 'offline_ppo' / 'best_model.pt'
    if ppo_checkpoint.exists():
        print("\n[3/3] Evaluating PPO (Proximal Policy Optimization)...")
        print("-"*60)
        ppo_agent = PPOAgent(ppo_checkpoint)
        results['PPO'] = evaluate_agent(ppo_agent, env, n_episodes)
    
    env.close()
    
    # Print comparison
    print("\n")
    print("="*60)
    print("📊 RESULTS COMPARISON")
    print("="*60)
    print(f"{'Method':<10} {'Mean Reward':<15} {'Success Rate':<15} {'Mean Length':<15}")
    print("-"*60)
    
    for method, res in results.items():
        print(f"{method:<10} {res['mean_reward']:>6.2f} ± {res['std_reward']:<5.2f} "
              f"{res['success_rate']:>6.1%}           {res['mean_length']:>6.1f}")
    
    print("="*60)
    
    # Save results
    output_file = checkpoint_dir / 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained offline RL models")
    parser.add_argument('--model', type=str, choices=['bc', 'dqn', 'ppo'],
                       help='Model type to evaluate')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all three models')
    parser.add_argument('--checkpoint-dir', type=str,
                       default='checkpoints',
                       help='Directory containing all checkpoints (relative to AVs/avs/)')
    parser.add_argument('--n-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes (slows down evaluation)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    args = parser.parse_args()
    
    if args.compare_all:
        # Compare all models
        compare_all_models(Path(args.checkpoint_dir), args.n_episodes)
    elif args.model and args.checkpoint:
        # Evaluate single model
        checkpoint_path = Path(args.checkpoint)
        
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return
        
        # Create environment
        env = create_highway_env()
        
        # Create agent
        print(f"\nEvaluating {args.model.upper()} model...")
        print("-"*60)
        
        if args.model == 'bc':
            agent = BCAgent(checkpoint_path, args.device)
        elif args.model == 'dqn':
            agent = DQNAgent(checkpoint_path, args.device)
        elif args.model == 'ppo':
            agent = PPOAgent(checkpoint_path, args.device)
        
        # Evaluate
        results = evaluate_agent(agent, env, args.n_episodes, args.render)
        
        env.close()
        
        # Print results
        print("\n")
        print("="*60)
        print(f"📊 {args.model.upper()} EVALUATION RESULTS")
        print("="*60)
        print(f"Mean Reward:   {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Mean Length:   {results['mean_length']:.1f} ± {results['std_length']:.1f}")
        print(f"Success Rate:  {results['success_rate']:.1%}")
        print("="*60)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
