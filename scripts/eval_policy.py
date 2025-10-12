#!/usr/bin/env python3
"""
Evaluate a saved PPO policy from `online_finetune_ppo.py`.
Runs deterministic episodes, prints mean/std reward and episode length, and can save short videos.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
import torch

# bring in compatible helpers from online_finetune_ppo.py
try:
    from online_finetune_ppo import make_env, preprocess_obs, ActorCritic, CLIPImageEncoder, HAS_RLLANG
except Exception:
    # fallback: import via package path if script executed from repo root
    from scripts.online_finetune_ppo import make_env, preprocess_obs, ActorCritic, CLIPImageEncoder, HAS_RLLANG

try:
    import imageio
except Exception:
    imageio = None


def load_policy(ckpt_path: str, device: torch.device):
    data = torch.load(str(ckpt_path), map_location=device)
    sd = None
    if isinstance(data, dict):
        if 'policy_state_dict' in data:
            sd = data['policy_state_dict']
        elif 'state_dict' in data:
            sd = data['state_dict']
        else:
            sd = data
    else:
        sd = data
    return sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/ppo_online_finetuned_final.pt')
    parser.add_argument('--env-id', type=str, default='highway-fast-v0')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-videos', action='store_true')
    parser.add_argument('--video-dir', type=str, default='videos/eval')
    parser.add_argument('--deterministic', action='store_true', help='Use greedy/deterministic action selection')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # create a single eval env (no vectorization)
    eval_env = None
    try:
        eval_env = make_env(render_mode='rgb_array' if args.save_videos else None)
    except Exception as e:
        print('Could not create eval env via make_env:', e)
        # try gym directly
        import gymnasium as gym
        eval_env = gym.make(args.env_id)

    # prepare policy
    obs, _ = eval_env.reset()
    # detect clip encoder if online_finetune used it
    clip_encoder = None
    if CLIPImageEncoder is not None and HAS_RLLANG:
        try:
            clip_encoder = CLIPImageEncoder(device=str(device))
        except Exception:
            clip_encoder = None

    sample_feat = preprocess_obs(obs, clip_encoder=clip_encoder, device=device)
    obs_dim = sample_feat.numel()
    n_actions = eval_env.action_space.n if hasattr(eval_env.action_space, 'n') else eval_env.action_space.shape[0]

    policy = ActorCritic(obs_dim, n_actions).to(device)
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f'Checkpoint not found: {ckpt}')
    sd = load_policy(ckpt, device)
    try:
        policy.load_state_dict(sd, strict=False)
        print(f'Loaded policy from {ckpt}')
    except Exception as e:
        print('Failed to strictly load state dict, attempting partial load:', e)
        # attempt to adapt keys
        new = {}
        for k, v in sd.items():
            nk = k.replace('policy.', '').replace('network.', '')
            new[nk] = v
        policy.load_state_dict(new, strict=False)
        print('Loaded adapted state dict (best-effort)')

    rewards = []
    lengths = []
    collision_counts = []

    video_dir = Path(args.video_dir)
    if args.save_videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    for epi in range(args.episodes):
        obs, _ = eval_env.reset()
        done = False
        total_r = 0.0
        length = 0
        frames = []
        collisions = 0
        while not done:
            feat = preprocess_obs(obs, clip_encoder=clip_encoder, device=device)
            with torch.no_grad():
                logits, _ = policy(feat.unsqueeze(0))
                probs = torch.softmax(logits, dim=-1)
                if args.deterministic:
                    action = int(torch.argmax(probs, dim=-1).item())
                else:
                    dist = torch.distributions.Categorical(probs)
                    action = int(dist.sample().item())
            out = eval_env.step(action)
            if len(out) == 5:
                obs, reward, terminated, truncated, info = out
                done = terminated or truncated
            else:
                obs, reward, done, info = out
            total_r += float(reward)
            length += 1
            # try to detect collision info in info dicts
            try:
                if isinstance(info, dict):
                    # typical highway-env uses 'collision' or 'crash' in info
                    if info.get('collision', False) or info.get('crash', False):
                        collisions += 1
            except Exception:
                pass
            if args.save_videos and imageio is not None:
                try:
                    frm = None
                    try:
                        frm = eval_env.render()
                    except Exception:
                        pass
                    if frm is not None:
                        frames.append(frm)
                except Exception:
                    pass

        rewards.append(total_r)
        lengths.append(length)
        collision_counts.append(collisions)
        print(f'Ep {epi + 1}/{args.episodes}: reward={total_r:.2f} len={length} collisions={collisions}')

        if args.save_videos and imageio is not None and len(frames) > 0:
            fname = video_dir / f'eval_ep{epi + 1}.mp4'
            try:
                imageio.mimwrite(str(fname), frames, fps=30)
                print('Wrote video', fname)
            except Exception as e:
                print('Failed to write video:', e)

    # Summary
    rewards = np.array(rewards, dtype=np.float32)
    lengths = np.array(lengths, dtype=np.int32)
    collisions = np.array(collision_counts, dtype=np.int32)
    print('=== Eval summary ===')
    print(f'episodes: {len(rewards)}')
    print(f'reward mean={rewards.mean():.3f} std={rewards.std():.3f} median={np.median(rewards):.3f} min={rewards.min():.3f} max={rewards.max():.3f}')
    print(f'length mean={lengths.mean():.1f} std={lengths.std():.1f}')
    print(f'collisions total={collisions.sum()} mean_per_episode={collisions.mean():.3f}')


if __name__ == '__main__':
    main()
