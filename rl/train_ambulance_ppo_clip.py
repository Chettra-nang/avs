#!/usr/bin/env python3
"""
Standalone PPO training script for ambulance highway scenarios using CLIP vision.
This script can be run independently and is designed for training on RTX 5090.

Usage:
    python train_ambulance_ppo_clip.py [--steps STEPS] [--seeds SEEDS] [--profile PROFILE]

Arguments:
    --steps: Total training steps (default: 300000)
    --seeds: Number of seeds to train (default: 5) 
    --profile: Training profile 'smoke' (10k steps) or 'full' (300k steps)
"""

import os
import sys
import time
import random
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import highway_env

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
)
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)

# Import our local modules
from clip_embedder import CLIPImageEncoder
from language_embedder import FrozenTextEmbedder
from ambulance_highway_wrapper import AmbulanceHighwayCLIPWrapper, CachedLLMEmbedder
from features_extractor_clip import CLIPLangExtractor


def pick_device() -> str:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_global_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ETACallback(BaseCallback):
    """Callback to print ETA during training."""
    
    def __init__(self, every_steps: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.every_steps = int(every_steps)
        self.t0 = None
        self.total = None

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
            print(f"[ETA] {done}/{self.total} steps | "
                  f"fps≈{fps:.1f} | remaining≈{int(eta_s)}s")
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
        clip_name = vision_cfg.get("clip_model", "openai/clip-vit-base-patch32")
        clip_device = vision_cfg.get("device", "auto")
        if clip_device == "auto":
            clip_device = pick_device()
        
        clip_encoder = CLIPImageEncoder(clip_name, device=clip_device)
        
        # Text encoder setup
        text_cfg = config.get("text", {})
        cached_dir = text_cfg.get("cached_llm_dir")
        if cached_dir and os.path.exists(cached_dir):
            cached_llm = CachedLLMEmbedder(
                cached_dir, dim=text_cfg.get("cached_dim", 1536)
            )
            text_embedder = None
        else:
            text_embedder = FrozenTextEmbedder(
                text_cfg.get("local_model", "sentence-transformers/all-MiniLM-L6-v2")
            )
            cached_llm = None
        
        # Wrap environment
        clip_stride = int(config.get("wrapper", {}).get("clip_stride", 8))
        env = AmbulanceHighwayCLIPWrapper(
            env,
            clip_encoder,
            text_embedder=text_embedder,
            cached_llm=cached_llm,
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
    
    # Use SubprocVecEnv for multiple environments on Linux, DummyVecEnv on Windows
    if n_envs > 1 and os.name == "posix":
        env = SubprocVecEnv(thunks)
    else:
        env = DummyVecEnv(thunks)
    
    env = VecMonitor(env)
    env = VecNormalize(
        env, norm_obs=True, norm_reward=True, clip_obs=10.0, training=True
    )
    
    return env


def train_one_seed(config: dict, seed: int, total_timesteps: int, model_prefix: str):
    """Train a single seed."""
    print(f"\n=== Training seed {seed} ===")
    set_global_seeds(seed)
    
    n_envs = int(config.get("algo", {}).get("n_envs", 1))
    
    # Create directories
    tb_dir = f"runs/ppo_ambulance/seed_{seed}"
    best_dir = f"{tb_dir}/best"
    ckpt_dir = f"{tb_dir}/ckpt"
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Create environments
    env = make_vec_envs(config, n_envs, base_seed=seed)
    eval_env = make_vec_envs(config, 1, base_seed=seed + 10_000)
    eval_env.training = False
    
    # Model configuration
    policy_kwargs = dict(
        features_extractor_class=CLIPLangExtractor,
        features_extractor_kwargs=dict(
            features_dim=config.get("policy", {}).get("feat_dim", 512)
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )
    
    # PPO hyperparameters
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
        ent_coef=ppo_cfg.get("ent_coef", 0.0),
        target_kl=ppo_cfg.get("target_kl", 0.03),
        verbose=1,
        tensorboard_log=tb_dir,
    )
    
    # Optional: Load pre-trained policy for warm start
    bc_path = "bc_clip_vlm_policy.pt"
    if os.path.exists(bc_path):
        try:
            print(f"[seed {seed}] Warm-starting from BC: {bc_path}")
            state_dict = torch.load(bc_path, map_location="cpu")
            model.policy.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"[seed {seed}] BC load failed: {e}")
    
    # Callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=f"{tb_dir}/eval",
        eval_freq=max(1, (5000 // max(1, n_envs))),
        n_eval_episodes=5,
        deterministic=False,
        render=False,
    )
    
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, (20000 // max(1, n_envs))),
        save_path=ckpt_dir,
        name_prefix="ppo",
    )
    
    eta_cb = ETACallback(5000)
    
    # Train
    print(f"[seed {seed}] Starting training for {total_timesteps:,} steps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_cb, ckpt_cb, eta_cb]
    )
    
    # Save final model
    model_path = f"{model_prefix}_seed{seed}.zip"
    model.save(model_path)
    env.save(f"{tb_dir}/vecnorm.pkl")
    print(f"[seed {seed}] Saved model: {model_path}")
    
    # Cleanup
    try:
        env.close()
        eval_env.close()
    except Exception:
        pass


def train_multiple_seeds(config: dict, seeds: list, total_timesteps: int, model_prefix: str):
    """Train multiple seeds sequentially."""
    for seed in seeds:
        train_one_seed(config, seed, total_timesteps, model_prefix)


def get_default_config(profile: str = "full") -> dict:
    """Get default training configuration."""
    config = {
        "env": {
            "lanes_count": 4,
            "vehicles_count": 40,
            "duration": 50,
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "offscreen_rendering": True,
            "render_agent": False,
            "show_trajectories": False,
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
            "clip_model": "openai/clip-vit-base-patch32",
            "device": "auto"
        },
        "text": {
            "cached_llm_dir": None,
            "cached_dim": 1536,
            "local_model": "sentence-transformers/all-MiniLM-L6-v2"
        },
        "policy": {
            "feat_dim": 512
        },
        "wrapper": {
            "clip_stride": 8 if profile == "smoke" else 4
        },
        "algo": {
            "n_envs": 1,  # Set to higher value on Linux for better performance
            "ppo": {
                "n_steps": 1024,
                "batch_size": 256,
                "lr": 2.5e-4,
                "gamma": 0.99,
                "clip_range": 0.2,
                "gae_lambda": 0.95,
                "ent_coef": 0.0,
                "target_kl": 0.03
            }
        }
    }
    
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PPO with CLIP for ambulance scenarios")
    parser.add_argument("--steps", type=int, default=300000, help="Total training steps")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to train")
    parser.add_argument("--profile", choices=["smoke", "full"], default="full", 
                       help="Training profile")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to use (auto, cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    # Set training steps based on profile if not explicitly set
    if args.profile == "smoke" and args.steps == 300000:
        args.steps = 10000
    
    # Load or create config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config(args.profile)
    
    # Override device if specified
    if args.device != "auto":
        config["vision"]["device"] = args.device
    
    # Print configuration
    print("=== Training Configuration ===")
    print(f"Profile: {args.profile}")
    print(f"Total steps: {args.steps:,}")
    print(f"Seeds: {list(range(args.seeds))}")
    print(f"Device: {config['vision']['device']}")
    print(f"Environment: {config['env']['scenario']}")
    print(f"CLIP model: {config['vision']['clip_model']}")
    
    # Save configuration
    config_path = f"config_ppo_{args.profile}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")
    
    # Start training
    print(f"\n=== Starting Training ===")
    model_prefix = f"ppo_clip_ambulance_{args.profile}"
    seeds = list(range(args.seeds))
    
    train_multiple_seeds(config, seeds, args.steps, model_prefix)
    
    print("\n=== Training Complete! ===")
    print("To monitor training progress, run:")
    print("tensorboard --logdir runs/ppo_ambulance")


if __name__ == "__main__":
    main()