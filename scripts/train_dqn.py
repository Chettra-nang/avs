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
parser.add_argument('--num-envs', type=int, default=8, help='Number of parallel envs for SubprocVecEnv')
parser.add_argument('--no-subproc', action='store_true', help='Force single-process DummyVecEnv (avoid subprocesses)')
args = parser.parse_args()

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
except Exception as e:
    print('Could not import stable_baselines3. Install with: pip install stable-baselines3[extra]')
    print('Error:', e)
    sys.exit(1)

# ensure highway_env is importable and use gymnasium
try:
    import highway_env  # registers highway envs
except Exception:
    print('Could not import highway_env. Ensure it is installed in your venv: pip install highway-env')
    # continue; env.make will raise a clearer error

try:
    import gymnasium as gym
except Exception:
    # fall back to gym for older compatibility
    import gym


def make_env(env_id: str, render_mode=None):
    if render_mode is not None:
        return gym.make(env_id, render_mode=render_mode)
    return gym.make(env_id)


def main():
    # create vectorized env (use subprocesses if available)
    num_envs = int(args.num_envs)
    # prefer SubprocVecEnv but guard against multiprocessing import issues by
    # creating envs inside __main__ guarded function
    env = None
    try:
        env_fns = []
        for _ in range(num_envs):
            # avoid late-binding lambda by creating a closure
            def make_fn(eid=args.env_id):
                return lambda: make_env(eid)
            env_fns.append(make_fn())
        if not args.no_subproc:
            env = SubprocVecEnv(env_fns)
        else:
            raise RuntimeError('Subproc disabled by --no-subproc')
    except Exception:
        # fallback to single-process DummyVecEnv
        env = DummyVecEnv([lambda: make_env(args.env_id)])

    # Print system / device diagnostics to help debug GPU usage
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count() if cuda_available else 0
        cuda_version = torch.version.cuda if hasattr(torch, 'version') else 'unknown'
        try:
            device_name = torch.cuda.get_device_name(0) if cuda_available and cuda_count > 0 else 'n/a'
        except Exception:
            device_name = 'n/a'
        print(f'Training DQN on env: {args.env_id}')
        print('PyTorch version:', getattr(torch, '__version__', 'unknown'))
        print('CUDA available:', cuda_available, 'count:', cuda_count, 'cuda_version:', cuda_version)
        print('CUDA device name:', device_name)
    except Exception:
        # If torch isn't importable, still print the env notice
        print('Training DQN on env:', args.env_id)

    # ensure selected device is actually available; stable-baselines3 will accept a str but
    # we guard to provide a clear message when falling back to CPU.
    selected_device = args.device
    try:
        if selected_device == 'cuda':
            import torch
            if not torch.cuda.is_available():
                print('WARNING: args.device=\'cuda\' but CUDA is not available in this Python environment. Falling back to cpu.')
                selected_device = 'cpu'
    except Exception:
        # ignore issues importing torch here
        pass
    model = DQN(
        args.policy,
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        device=selected_device,
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


if __name__ == '__main__':
    main()
