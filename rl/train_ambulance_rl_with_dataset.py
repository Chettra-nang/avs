# train_ambulance_rl_with_dataset.py - Complete RL training pipeline with your dataset
#!/usr/bin/env python3
"""
Complete RL training pipeline for ambulance scenarios that:
1. Uses your collected dataset for behavior cloning warm-start
2. Continues with RL training (PPO/DQN) for improvement
3. Optimized for RTX 5090 training

Usage:
    # Train BC first, then RL
    python train_ambulance_rl_with_dataset.py --data_dir /path/to/dataset --algorithm ppo
    
    # Skip BC and train RL only
    python train_ambulance_rl_with_dataset.py --algorithm ppo --skip_bc
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import highway_env

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

# Local imports
from clip_embedder import CLIPImageEncoder
from language_embedder import FrozenTextEmbedder
from ambulance_highway_wrapper import AmbulanceHighwayCLIPWrapper
from features_extractor_clip import CLIPLangExtractor
from train_bc_from_dataset import train_behavior_cloning


class ETACallback(BaseCallback):
    """Enhanced ETA callback with performance monitoring."""
    
    def __init__(self, every_steps: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.every_steps = int(every_steps)
        self.t0 = None
        self.total = None
        self.best_reward = -np.inf

    def _on_training_start(self) -> None:
        self.t0 = time.time()
        self.total = int(self.model._total_timesteps)
        return True

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every_steps == 0 and self.num_timesteps > 0:
            done = self.num_timesteps
            elapsed = max(time.time() - self.t0, 1e-6)
            fps = done / elapsed
            remain = max(self.total - done, 0)
            eta_s = remain / max(fps, 1e-6)
            
            # Get recent reward if available
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                recent_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer[-10:]])
                if recent_reward > self.best_reward:
                    self.best_reward = recent_reward
                    status = f"🔥 NEW BEST: {recent_reward:.2f}"
                else:
                    status = f"reward: {recent_reward:.2f}"
            else:
                status = "training..."
            
            print(f"[ETA] {done:,}/{self.total:,} steps | "
                  f"fps≈{fps:.1f} | eta≈{int(eta_s//60)}m{int(eta_s%60)}s | {status}")
        return True


def make_env_thunk(config: dict, seed: int):
    """Create environment factory for multiprocessing."""
    def _init():
        # Set seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create base environment
        env_cfg = config.get("env", {})
        env_cfg = {
            **env_cfg,
            "offscreen_rendering": True,
            "render_agent": False,
            "show_trajectories": False
        }
        env = gym.make("highway-v0", render_mode="rgb_array", config=env_cfg)
        
        try:
            env.reset(seed=seed)
        except TypeError:
            pass
        
        # Create encoders
        vision_cfg = config.get("vision", {})
        clip_name = vision_cfg.get("clip_model", "ViT-B-32")
        clip_device = vision_cfg.get("device", "auto")
        if clip_device == "auto":
            clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        clip_encoder = CLIPImageEncoder(clip_name, device=clip_device)
        text_embedder = FrozenTextEmbedder()
        
        # Wrap environment
        clip_stride = int(config.get("wrapper", {}).get("clip_stride", 4))
        env = AmbulanceHighwayCLIPWrapper(
            env,
            clip_encoder,
            text_embedder=text_embedder,
            clip_stride=clip_stride
        )
        
        return env
    
    return _init


def make_vec_envs(config: dict, n_envs: int, base_seed: int):
    """Create vectorized environments."""
    thunks = [
        make_env_thunk(config, seed=base_seed + i) 
        for i in range(max(1, n_envs))
    ]
    
    # Use SubprocVecEnv for multiple environments on Linux
    if n_envs > 1 and os.name == "posix":
        env = SubprocVecEnv(thunks)
    else:
        env = DummyVecEnv(thunks)
    
    env = VecMonitor(env)
    env = VecNormalize(
        env, norm_obs=True, norm_reward=True, clip_obs=10.0, training=True
    )
    
    return env


def train_rl_with_bc_warmstart(
    config: dict,
    algorithm: str = "ppo",
    bc_policy_path: str = None,
    total_timesteps: int = 500000,
    n_seeds: int = 3,
    output_dir: str = "rl_models"
):
    """Train RL with optional BC warm-start."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for seed in range(n_seeds):
        print(f"\n=== Training {algorithm.upper()} - Seed {seed} ===")
        
        # Setup directories
        tb_dir = f"runs/{algorithm}_ambulance_dataset/seed_{seed}"
        best_dir = f"{tb_dir}/best"
        ckpt_dir = f"{tb_dir}/ckpt"
        os.makedirs(best_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Create environments
        n_envs = config.get("algo", {}).get("n_envs", 4 if algorithm == "ppo" else 1)
        env = make_vec_envs(config, n_envs, base_seed=seed)
        eval_env = make_vec_envs(config, 1, base_seed=seed + 10000)
        eval_env.training = False
        
        # Policy configuration
        policy_kwargs = dict(
            features_extractor_class=CLIPLangExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        )
        
        # Create model
        if algorithm.lower() == "ppo":
            ppo_cfg = config.get("algo", {}).get("ppo", {})
            model = PPO(
                "MultiInputPolicy",
                env,
                seed=seed,
                policy_kwargs=policy_kwargs,
                n_steps=ppo_cfg.get("n_steps", 1024),
                batch_size=ppo_cfg.get("batch_size", 256),
                learning_rate=ppo_cfg.get("lr", 2.5e-4),
                gamma=ppo_cfg.get("gamma", 0.99),
                clip_range=ppo_cfg.get("clip_range", 0.2),
                gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
                ent_coef=ppo_cfg.get("ent_coef", 0.01),
                verbose=1,
                tensorboard_log=tb_dir,
            )
        else:  # DQN
            dqn_cfg = config.get("algo", {}).get("dqn", {})
            policy_kwargs["net_arch"] = [256, 256]  # DQN uses different format
            model = DQN(
                "MultiInputPolicy",
                env,
                seed=seed,
                policy_kwargs=policy_kwargs,
                learning_rate=dqn_cfg.get("lr", 1e-4),
                buffer_size=dqn_cfg.get("buffer_size", 200000),
                learning_starts=dqn_cfg.get("learning_starts", 10000),
                batch_size=dqn_cfg.get("batch_size", 256),
                target_update_interval=dqn_cfg.get("target_update_interval", 2000),
                train_freq=dqn_cfg.get("train_freq", 4),
                exploration_fraction=dqn_cfg.get("exploration_fraction", 0.2),
                exploration_final_eps=dqn_cfg.get("exploration_final_eps", 0.05),
                gamma=dqn_cfg.get("gamma", 0.99),
                verbose=1,
                tensorboard_log=tb_dir,
            )
        
        # Load BC weights if available
        if bc_policy_path and os.path.exists(bc_policy_path):
            try:
                print(f"🔥 Loading BC weights from: {bc_policy_path}")
                bc_weights = torch.load(bc_policy_path, map_location="cpu")
                model.policy.load_state_dict(bc_weights, strict=False)
                print("✅ BC weights loaded successfully!")
            except Exception as e:
                print(f"⚠️  BC weight loading failed: {e}")
        
        # Callbacks
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=best_dir,
            log_path=f"{tb_dir}/eval",
            eval_freq=max(1, (10000 // max(1, n_envs))),
            n_eval_episodes=5,
            deterministic=False,
        )
        
        ckpt_cb = CheckpointCallback(
            save_freq=max(1, (25000 // max(1, n_envs))),
            save_path=ckpt_dir,
            name_prefix=algorithm,
        )
        
        eta_cb = ETACallback(5000)
        
        # Train
        print(f"🚀 Starting {algorithm.upper()} training for {total_timesteps:,} steps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_cb, ckpt_cb, eta_cb]
        )
        
        # Save final model
        model_path = f"{output_dir}/{algorithm}_ambulance_dataset_seed{seed}.zip"
        model.save(model_path)
        env.save(f"{tb_dir}/vecnorm.pkl")
        print(f"💾 Saved model: {model_path}")
        
        # Cleanup
        try:
            env.close()
            eval_env.close()
        except Exception:
            pass


def get_default_config(algorithm: str = "ppo") -> dict:
    """Get optimized config for RTX 5090."""
    config = {
        "env": {
            "lanes_count": 4,
            "vehicles_count": 40,
            "duration": 50,
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "high_speed_reward": 0.4,
            "right_lane_reward": 0.1,
            "lane_change_reward": 0.0,
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "scenario": "highway_emergency_dense",
        },
        "vision": {
            "clip_model": "ViT-B-32",
            "device": "auto"
        },
        "text": {
            "local_model": "sentence-transformers/all-MiniLM-L6-v2"
        },
        "wrapper": {
            "clip_stride": 4
        },
        "algo": {
            "n_envs": 4 if algorithm == "ppo" else 1,  # RTX 5090 can handle multiple envs
            "ppo": {
                "n_steps": 2048,  # Larger for RTX 5090
                "batch_size": 512,  # Larger batch for better GPU utilization
                "lr": 2.5e-4,
                "gamma": 0.99,
                "clip_range": 0.2,
                "gae_lambda": 0.95,
                "ent_coef": 0.01
            },
            "dqn": {
                "lr": 1e-4,
                "buffer_size": 500000,  # Larger buffer for RTX 5090
                "learning_starts": 10000,
                "batch_size": 512,  # Larger batch
                "target_update_interval": 2000,
                "train_freq": 4,
                "exploration_fraction": 0.2,
                "exploration_final_eps": 0.05,
                "gamma": 0.99
            }
        }
    }
    
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RL with ambulance dataset")
    parser.add_argument("--data_dir", type=str, 
                       help="Path to ambulance dataset directory")
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], default="ppo",
                       help="RL algorithm to use")
    parser.add_argument("--steps", type=int, default=500000,
                       help="Total RL training steps")
    parser.add_argument("--seeds", type=int, default=3,
                       help="Number of seeds to train")
    parser.add_argument("--skip_bc", action="store_true",
                       help="Skip behavior cloning and train RL only")
    parser.add_argument("--bc_epochs", type=int, default=50,
                       help="BC training epochs")
    parser.add_argument("--output_dir", type=str, default="rl_models",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_default_config(args.algorithm)
    if args.device != "auto":
        config["vision"]["device"] = args.device
    
    # Print setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🚀 Ambulance RL Training with Dataset")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Device: {device}")
    print(f"Total steps: {args.steps:,}")
    print(f"Seeds: {args.seeds}")
    
    bc_policy_path = None
    
    # Step 1: Behavior Cloning (if not skipped)
    if not args.skip_bc and args.data_dir:
        if not os.path.exists(args.data_dir):
            print(f"❌ Dataset directory not found: {args.data_dir}")
            print("Either provide --data_dir or use --skip_bc")
            sys.exit(1)
        
        print(f"\n📚 Step 1: Behavior Cloning from dataset...")
        bc_policy_path = train_behavior_cloning(
            data_dir=args.data_dir,
            output_dir="bc_models",
            epochs=args.bc_epochs,
            device=args.device
        )
    elif args.skip_bc:
        print("⏭️  Skipping Behavior Cloning")
    
    # Step 2: RL Training
    print(f"\n🧠 Step 2: {args.algorithm.upper()} Training...")
    train_rl_with_bc_warmstart(
        config=config,
        algorithm=args.algorithm,
        bc_policy_path=bc_policy_path,
        total_timesteps=args.steps,
        n_seeds=args.seeds,
        output_dir=args.output_dir
    )
    
    print(f"\n🎉 Training Complete!")
    print(f"Models saved in: {args.output_dir}")
    print("Monitor with: tensorboard --logdir runs")


if __name__ == "__main__":
    main()