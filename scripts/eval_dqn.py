#!/usr/bin/env python3
"""
Evaluate a Stable-Baselines3 DQN model saved by `scripts/train_dqn.py`.
Runs deterministic episodes and prints aggregated metrics. Optionally saves videos.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

try:
    from stable_baselines3 import DQN
except Exception:
    DQN = None

try:
    import gymnasium as gym
except Exception:
    import gym

try:
    import imageio
except Exception:
    imageio = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='checkpoints/dqn_highway-fast-v0.zip', help='Path to SB3 DQN zip')
    parser.add_argument('--env-id', type=str, default='highway-fast-v0')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--save-videos', action='store_true')
    parser.add_argument('--video-dir', type=str, default='videos/eval_dqn')
    args = parser.parse_args()

    if DQN is None:
        raise SystemExit('stable_baselines3 not available in this environment. Install with: pip install stable-baselines3[extra]')

    model_path = Path(args.model)
    if not model_path.exists():
        # also try .zip pattern
        alt = model_path.with_suffix('.zip')
        if alt.exists():
            model_path = alt
        else:
            raise SystemExit(f'Model file not found: {model_path}')

    # Load model using SB3 loader (device mapping handled by SB3)
    model = None
    try:
        model = DQN.load(str(model_path), device=args.device)
    except AssertionError as e:
        # SB3 can raise AssertionError: 'No data found in the saved file'
        msg = str(e)
        print('Initial DQN.load failed:', msg)
        # If the checkpoint is a .zip containing a folder, extract and retry
        if model_path.suffix == '.zip':
            import tempfile, zipfile
            tmp = tempfile.mkdtemp(prefix='dqn_eval_')
            try:
                with zipfile.ZipFile(str(model_path), 'r') as z:
                    z.extractall(tmp)
                # SB3 expects a folder with same base name inside the zip
                # try loading from extracted path
                base = model_path.stem
                extracted_dir = Path(tmp) / base
                if extracted_dir.exists():
                    try:
                        model = DQN.load(str(extracted_dir), device=args.device)
                    except Exception as e2:
                        print('Retry load from extracted folder failed:', e2)
                else:
                    # try loading from tmp root
                    try:
                        model = DQN.load(str(tmp), device=args.device)
                    except Exception as e3:
                        print('Retry load from tmp root failed:', e3)
            except Exception as ex:
                print('Failed to extract zip for retry:', ex)
    except Exception as e:
        print('DQN.load failed with exception:', e)

    if model is None:
        raise SystemExit('Could not load DQN model from provided path (tried zip fallback)')

    # Create eval env
    try:
        env = make_env = None
        # prefer imported highway_env registration
        import highway_env  # may raise
        try:
            env = gym.make(args.env_id)
        except Exception:
            # fallback to highway-v0
            env = gym.make('highway-v0')
    except Exception:
        # if highway_env not installed, try making env directly (may raise)
        env = gym.make(args.env_id)

    rewards = []
    lengths = []
    collisions = []

    video_dir = Path(args.video_dir)
    if args.save_videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    for epi in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        length = 0
        frames = []
        col = 0
        while not done:
            # SB3 models expect numpy observations
            action, _ = model.predict(obs, deterministic=args.deterministic)
            out = env.step(int(action))
            if len(out) == 5:
                obs, reward, term, trunc, info = out
                done = bool(term or trunc)
            else:
                obs, reward, done, info = out
            total_r += float(reward)
            length += 1
            if isinstance(info, dict):
                if info.get('collision', False) or info.get('crash', False):
                    col += 1
            if args.save_videos and imageio is not None:
                try:
                    frm = env.render()
                    if frm is not None:
                        frames.append(frm)
                except Exception:
                    pass

        rewards.append(total_r)
        lengths.append(length)
        collisions.append(col)
        print(f'Ep {epi+1}/{args.episodes}: reward={total_r:.2f} len={length} collisions={col}')

        if args.save_videos and imageio is not None and len(frames) > 0:
            fname = video_dir / f'dqn_eval_ep{epi+1}.mp4'
            try:
                imageio.mimwrite(str(fname), frames, fps=30)
                print('Wrote video', fname)
            except Exception as e:
                print('Failed to write video:', e)

    rewards = np.array(rewards, dtype=np.float32)
    lengths = np.array(lengths, dtype=np.int32)
    collisions = np.array(collisions, dtype=np.int32)
    print('=== DQN Eval summary ===')
    print(f'episodes: {len(rewards)}')
    print(f'reward mean={rewards.mean():.3f} std={rewards.std():.3f} median={np.median(rewards):.3f} min={rewards.min():.3f} max={rewards.max():.3f}')
    print(f'length mean={lengths.mean():.1f} std={lengths.std():.1f}')
    print(f'collisions total={collisions.sum()} mean_per_episode={collisions.mean():.3f}')


if __name__ == '__main__':
    main()
