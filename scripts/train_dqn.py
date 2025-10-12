#!/usr/bin/env python3
"""
Train a DQN agent with Stable-Baselines3 on a highway-env variant.
Saves the model to checkpoints/dqn_{env_id}.zip
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--env-id', type=str, default='highway-fast-v0', help='Gym env id to use')
parser.add_argument('--timesteps', type=int, default=200_000)
parser.add_argument('--buffer-size', type=int, default=100000)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--train-freq', type=int, default=4, help='Update the model every N env steps')
parser.add_argument('--gradient-steps', type=int, default=1, help='Gradient steps after each rollout (use -1 for as many as steps)')
parser.add_argument('--learning-rate', type=float, default=1e-4)
parser.add_argument('--device', type=str, default='cuda', help='Torch device')
parser.add_argument('--policy', type=str, default='MlpPolicy', help='DQN policy (MlpPolicy|CnnPolicy|MultiInputPolicy)')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--verbose', type=int, default=1)
args = parser.parse_args()

# Try to import SB3
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
except Exception as e:
    print('Could not import stable_baselines3. Install with: pip install stable-baselines3[extra]')
    print('Error:', e)
    sys.exit(1)

# ensure highway_env is importable
try:
    import highway_env
except Exception:
    print('Could not import highway_env. Ensure it is installed in your venv: pip install highway-env')
    # we won't exit; allow gym to raise detailed error later

# env factory

def make_env(env_id: str, render_mode=None):
    import gym
    if render_mode is not None:
        return gym.make(env_id, render_mode=render_mode)
    return gym.make(env_id)

# create vectorized env (use subprocesses if available)
num_envs = 8
try:
    env_fns = [lambda eid=args.env_id: make_env(eid) for _ in range(num_envs)]
    env = SubprocVecEnv(env_fns)
except Exception:
    env = DummyVecEnv([lambda: make_env(args.env_id)])

print('Training DQN on env:', args.env_id)
model = DQN(
    args.policy,
    env,
    learning_rate=args.learning_rate,
    buffer_size=args.buffer_size,
    batch_size=args.batch_size,
    train_freq=args.train_freq,
    gradient_steps=args.gradient_steps,
    device=args.device,
    seed=args.seed,
    verbose=args.verbose,
)

out = Path('checkpoints')
out.mkdir(parents=True, exist_ok=True)
model_path = out / f'dqn_{args.env_id}.zip'

print('Starting training, saving to', model_path)
model.learn(total_timesteps=args.timesteps)
model.save(str(model_path))
print('Saved model to', model_path)
