#!/usr/bin/env python3
"""
Evaluate Trained Models on AMBULANCE SCENARIOS (Matching Training Data)

This evaluates your models on the same 30 diverse scenarios they were trained on,
not just standard highway-v0!

Usage:
    python3 scripts/evaluate_on_ambulance_scenarios.py --compare-all --n-episodes-per-scenario 5
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import gymnasium as gym

# Import agents from evaluate_trained_models
import sys
sys.path.append(str(Path(__file__).parent))
from evaluate_trained_models import BCAgent, DQNAgent, PPOAgent

# Import ambulance scenarios
sys.path.append(str(Path(__file__).parent.parent / 'collecting_ambulance_data'))
try:
    from scenarios.ambulance_scenarios import AMBULANCE_SCENARIOS
except ImportError:
    print("Warning: Could not import AMBULANCE_SCENARIOS, using default list")
    AMBULANCE_SCENARIOS = {}


def create_ambulance_env(scenario_name: str, scenario_config: Dict):
    """Create environment matching the scenario used in training."""
    import highway_env
    
    # Determine environment type based on scenario
    if 'roundabout' in scenario_name.lower():
        env_id = 'roundabout-v0'
    elif 'intersection' in scenario_name.lower() or 'corner' in scenario_name.lower():
        env_id = 'intersection-v1'
    elif 'merge' in scenario_name.lower():
        env_id = 'merge-v0'
    else:
        env_id = 'highway-v0'
    
    env = gym.make(env_id, render_mode='rgb_array')
    
    # Configure to match training
    config = {
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 256),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],
            "scaling": 1.75,
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "policy_frequency": 2,
        "duration": scenario_config.get('duration', 40),
        "simulation_frequency": 15,
        "lanes_count": scenario_config.get('lanes_count', 4),
        "vehicles_count": scenario_config.get('vehicles_count', 50),
        "reward_speed_range": [20, 30],
        "normalize_reward": True,
    }
    
    env.unwrapped.config.update(config)
    
    return env


def evaluate_on_scenario(agent, scenario_name: str, scenario_config: Dict, n_episodes: int = 5):
    """Evaluate agent on a specific scenario."""
    
    results = []
    
    for episode in range(n_episodes):
        try:
            # Create fresh environment for each episode
            env = create_ambulance_env(scenario_name, scenario_config)
            
            obs, info = env.reset(seed=42 + episode)
            episode_reward = 0
            episode_length = 0
            done = False
            truncated = False
            
            max_steps = 200  # Safety limit
            
            while not (done or truncated) and episode_length < max_steps:
                # Select action
                action = agent.select_action(obs)
                
                # Step environment
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
            
            success = not info.get('crashed', False)
            
            results.append({
                'episode': episode,
                'reward': episode_reward,
                'length': episode_length,
                'success': success
            })
            
            env.close()
            
        except Exception as e:
            print(f"    Episode {episode+1} failed: {str(e)}")
            results.append({
                'episode': episode,
                'reward': 0.0,
                'length': 0,
                'success': False,
                'error': str(e)
            })
    
    # Compute statistics
    rewards = [r['reward'] for r in results if 'error' not in r]
    lengths = [r['length'] for r in results if 'error' not in r]
    successes = [r['success'] for r in results if 'error' not in r]
    
    return {
        'scenario': scenario_name,
        'episodes': results,
        'mean_reward': np.mean(rewards) if rewards else 0.0,
        'std_reward': np.std(rewards) if rewards else 0.0,
        'mean_length': np.mean(lengths) if lengths else 0.0,
        'success_rate': np.mean(successes) if successes else 0.0,
        'successful_episodes': sum(successes),
        'total_episodes': len(results)
    }


def evaluate_on_all_scenarios(agent, agent_name: str, scenarios: Dict, n_episodes_per_scenario: int = 5):
    """Evaluate agent on all ambulance scenarios."""
    
    print(f"\n{'='*60}")
    print(f"Evaluating {agent_name} on {len(scenarios)} scenarios")
    print(f"{'='*60}")
    
    scenario_results = []
    
    for idx, (scenario_name, scenario_config) in enumerate(scenarios.items(), 1):
        print(f"\n[{idx}/{len(scenarios)}] {scenario_name}")
        print("-" * 60)
        
        result = evaluate_on_scenario(agent, scenario_name, scenario_config, n_episodes_per_scenario)
        scenario_results.append(result)
        
        print(f"  Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"  Success Rate: {result['success_rate']:.1%} ({result['successful_episodes']}/{result['total_episodes']})")
        print(f"  Mean Length: {result['mean_length']:.1f}")
    
    # Overall statistics
    overall_mean_reward = np.mean([r['mean_reward'] for r in scenario_results])
    overall_success_rate = np.mean([r['success_rate'] for r in scenario_results])
    overall_mean_length = np.mean([r['mean_length'] for r in scenario_results])
    
    return {
        'agent': agent_name,
        'scenario_results': scenario_results,
        'overall_mean_reward': overall_mean_reward,
        'overall_success_rate': overall_success_rate,
        'overall_mean_length': overall_mean_length,
        'total_episodes': len(scenarios) * n_episodes_per_scenario
    }


def compare_all_models(checkpoint_dir: Path, scenarios: Dict, n_episodes_per_scenario: int = 5):
    """Compare all three models on ambulance scenarios."""
    
    print("="*60)
    print("🏆 EVALUATING ON AMBULANCE SCENARIOS")
    print("   (Matching Training Distribution)")
    print("="*60)
    print(f"Total scenarios: {len(scenarios)}")
    print(f"Episodes per scenario: {n_episodes_per_scenario}")
    print(f"Total episodes: {len(scenarios) * n_episodes_per_scenario}")
    print("="*60)
    
    results = {}
    
    # BC
    bc_checkpoint = checkpoint_dir / 'bc_pretrain' / 'best_model.pt'
    if bc_checkpoint.exists():
        bc_agent = BCAgent(bc_checkpoint)
        results['BC'] = evaluate_on_all_scenarios(bc_agent, 'BC', scenarios, n_episodes_per_scenario)
    
    # DQN
    dqn_checkpoint = checkpoint_dir / 'offline_dqn' / 'best_model.pt'
    if dqn_checkpoint.exists():
        dqn_agent = DQNAgent(dqn_checkpoint)
        results['DQN'] = evaluate_on_all_scenarios(dqn_agent, 'DQN', scenarios, n_episodes_per_scenario)
    
    # PPO
    ppo_checkpoint = checkpoint_dir / 'offline_ppo' / 'best_model.pt'
    if ppo_checkpoint.exists():
        ppo_agent = PPOAgent(ppo_checkpoint)
        results['PPO'] = evaluate_on_all_scenarios(ppo_agent, 'PPO', scenarios, n_episodes_per_scenario)
    
    # Print comparison
    print("\n")
    print("="*60)
    print("📊 OVERALL RESULTS (Across All Scenarios)")
    print("="*60)
    print(f"{'Method':<10} {'Mean Reward':<15} {'Success Rate':<15} {'Mean Length':<15}")
    print("-"*60)
    
    for method, res in results.items():
        print(f"{method:<10} {res['overall_mean_reward']:>6.2f}           "
              f"{res['overall_success_rate']:>6.1%}           "
              f"{res['overall_mean_length']:>6.1f}")
    
    print("="*60)
    
    # Save results
    output_file = checkpoint_dir / 'evaluation_ambulance_scenarios.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate on ambulance scenarios")
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory containing checkpoints')
    parser.add_argument('--n-episodes-per-scenario', type=int, default=5,
                       help='Episodes per scenario')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all three models')
    parser.add_argument('--model', type=str, choices=['bc', 'dqn', 'ppo'],
                       help='Evaluate single model')
    parser.add_argument('--scenario', type=str,
                       help='Evaluate single scenario only')
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    
    # Load ambulance scenarios
    if AMBULANCE_SCENARIOS:
        scenarios = AMBULANCE_SCENARIOS
    else:
        # Fallback: use a subset if scenarios not available
        scenarios = {
            'highway_free_flow': {'vehicles_count': 30, 'lanes_count': 4, 'duration': 40},
            'highway_dense': {'vehicles_count': 60, 'lanes_count': 4, 'duration': 45},
            'highway_aggressive': {'vehicles_count': 45, 'lanes_count': 4, 'duration': 40},
        }
        print(f"⚠️  Using fallback scenarios ({len(scenarios)} scenarios)")
    
    if args.scenario:
        # Single scenario
        if args.scenario in scenarios:
            scenarios = {args.scenario: scenarios[args.scenario]}
        else:
            print(f"❌ Scenario '{args.scenario}' not found!")
            print(f"Available: {list(scenarios.keys())}")
            return
    
    if args.compare_all:
        compare_all_models(checkpoint_dir, scenarios, args.n_episodes_per_scenario)
    elif args.model:
        # Single model evaluation
        checkpoint_path = checkpoint_dir / f'{args.model}_pretrain' if args.model == 'bc' else checkpoint_dir / f'offline_{args.model}'
        checkpoint_path = checkpoint_path / 'best_model.pt'
        
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return
        
        if args.model == 'bc':
            agent = BCAgent(checkpoint_path)
        elif args.model == 'dqn':
            agent = DQNAgent(checkpoint_path)
        elif args.model == 'ppo':
            agent = PPOAgent(checkpoint_path)
        
        results = evaluate_on_all_scenarios(agent, args.model.upper(), scenarios, args.n_episodes_per_scenario)
        
        print("\n" + "="*60)
        print(f"📊 {args.model.upper()} RESULTS")
        print("="*60)
        print(f"Overall Mean Reward: {results['overall_mean_reward']:.2f}")
        print(f"Overall Success Rate: {results['overall_success_rate']:.1%}")
        print(f"Overall Mean Length: {results['overall_mean_length']:.1f}")
        print("="*60)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
