#!/usr/bin/env python3
"""
Evaluate a DQN policy by directly loading the PyTorch state dict saved under
`checkpoints/dqn_<env>/policy.pth` and running deterministic episodes.

This bypasses SB3's zip loading and works when the SB3 archive layout is nonstandard.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
try:
    import gymnasium as gym
except Exception:
    import gym
try:
    import imageio
except Exception:
    imageio = None


class QNetModule(nn.Module):
    def __init__(self, in_dim: int, hidden: int, mid_hidden: int, out_dim: int):
        super().__init__()
        # this module intentionally has attribute name 'q_net' matching SB3 keys
        self.q_net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, mid_hidden),
            nn.ReLU(),
            nn.Linear(mid_hidden, out_dim),
        )


def load_policy_state(policy_dir: Path):
    p = policy_dir / 'policy.pth'
    if not p.exists():
        raise SystemExit(f'policy.pth not found in {policy_dir}')
    sd = torch.load(str(p), map_location='cpu')
    return sd


def flatten_obs(o):
    # handle dict-like or numpy array
    if isinstance(o, dict):
        # try to find a numpy array in values
        for v in o.values():
            if isinstance(v, np.ndarray):
                return v.ravel()
        # fallback: convert to flat array
        return np.asarray(o).ravel()
    else:
        return np.asarray(o).ravel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy-dir', type=str, default='checkpoints/dqn_highway-fast-v0')
    parser.add_argument('--env-id', type=str, default='highway-fast-v0')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--save-videos', action='store_true', help='Save per-episode MP4s')
    parser.add_argument('--video-dir', type=str, default='videos/dqn_manual_eval')
    parser.add_argument('--save-csv', action='store_true', help='Write per-episode CSV summary to video-dir/eval.csv')
    parser.add_argument('--overlay', action='store_true', help='Overlay reward/collision info on saved frames')
    args = parser.parse_args()

    policy_dir = Path(args.policy_dir)
    if not policy_dir.exists():
        raise SystemExit(f'Policy dir not found: {policy_dir}')

    sd = load_policy_state(policy_dir)
    # find q_net shapes from state dict keys
    # expect keys like 'q_net.q_net.0.weight' etc
    # infer dims from shapes
    weight_keys = [k for k in sd.keys() if k.endswith('.weight')]
    # find final layer weight with smallest first dim (out_dim)
    final_w = None
    for k in weight_keys:
        w = sd[k]
        if w.ndim == 2:
            if final_w is None or w.shape[0] < final_w.shape[0]:
                final_w = w
                final_key = k
    if final_w is None:
        raise SystemExit('Could not infer final layer weight from state dict')
    out_dim = final_w.shape[0]
    # infer input dim from first linear weight
    first_w = None
    for k in weight_keys:
        w = sd[k]
        if w.ndim == 2:
            if first_w is None or w.shape[1] > first_w.shape[1]:
                first_w = w
                first_key = k
    in_dim = first_w.shape[1]
    # hidden dims inference (best-effort)
    hidden = sd[[k for k in weight_keys if '0.weight' in k][0]].shape[0] if any('0.weight' in k for k in weight_keys) else 64
    mid_hidden = sd[[k for k in weight_keys if '2.weight' in k][0]].shape[0] if any('2.weight' in k for k in weight_keys) else 64

    print(f'Inferred dims: in={in_dim} hidden={hidden} mid={mid_hidden} out={out_dim}')

    model = QNetModule(in_dim, hidden, mid_hidden, out_dim)
    # create target net attribute to match keys if present
    model.q_net_target = QNetModule(in_dim, hidden, mid_hidden, out_dim).q_net

    # load state dict into model (non-strict)
    try:
        model.load_state_dict(sd, strict=False)
        print('Loaded state dict into model (strict=False)')
    except Exception as e:
        print('Partial load failed:', e)

    # create env
    try:
        import highway_env
    except Exception:
        pass
    # If saving videos, request an env that can render frames
    try:
        if args.save_videos:
            try:
                env = gym.make(args.env_id, render_mode='rgb_array')
            except Exception:
                env = gym.make(args.env_id)
        else:
            env = gym.make(args.env_id)
    except Exception as e:
        raise SystemExit(f'Could not create env {args.env_id}: {e}')

    rewards = []
    lengths = []
    collisions = []

    video_dir = None
    if args.save_videos or args.save_csv:
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)

    # CSV file setup
    csv_path = None
    if args.save_csv and video_dir is not None:
        import csv
        csv_path = video_dir / 'eval.csv'
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['episode', 'reward', 'length', 'collisions'])
    else:
        csv_writer = None

    # prepare PIL for overlays when requested
    if args.overlay:
        try:
            from PIL import Image, ImageDraw, ImageFont
            pil_available = True
            # choose a default font (may fallback)
            try:
                font = ImageFont.truetype('DejaVuSans.ttf', size=16)
            except Exception:
                font = ImageFont.load_default()
        except Exception:
            pil_available = False
            font = None
            print('PIL not available; --overlay will be ignored')

    for epi in range(args.episodes):
        # Reset; for gymnasium, env.reset() can return (obs, info)
        try:
            obs, _ = env.reset()
        except Exception:
            obs = env.reset()
        done = False
        total_r = 0.0
        length = 0
        col = 0
        frames = []
        while not done:
            arr = flatten_obs(obs)
            if arr.size != in_dim:
                # try reshape if shapes match 2D->flatten
                arr = np.resize(arr, (in_dim,))
            x = torch.as_tensor(arr.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                q = model.q_net(x).squeeze(0).numpy()
            if args.deterministic:
                action = int(np.argmax(q))
            else:
                # sample proportional to absolute Q-values as a heuristic
                probs = np.abs(q)
                probs = probs / (probs.sum() + 1e-8)
                action = int(np.random.choice(len(q), p=probs))
            out = env.step(action)
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
            # capture frame if requested
            if args.save_videos:
                try:
                    frm = None
                    try:
                        frm = env.render()
                    except Exception:
                        frm = None
                    if frm is not None:
                        # overlay text/indicator if requested and PIL available
                        if args.overlay and pil_available:
                            try:
                                im = Image.fromarray(frm)
                                draw = ImageDraw.Draw(im)
                                txt = f'rew={total_r:.2f} len={length} col={col}'
                                # draw semi-opaque rectangle
                                draw.rectangle([(0,0),(200,24)], fill=(0,0,0,120))
                                draw.text((4,2), txt, fill=(255,255,255), font=font)
                                if col:
                                    # red border to indicate collision
                                    w,h = im.size
                                    draw.rectangle([(0,0),(w-1,h-1)], outline=(255,0,0), width=4)
                                frm = np.asarray(im)
                            except Exception:
                                pass
                        frames.append(frm)
                except Exception:
                    pass

        rewards.append(total_r)
        lengths.append(length)
        collisions.append(col)
        print(f'Ep {epi+1}/{args.episodes}: reward={total_r:.2f} len={length} collisions={col}')

        if args.save_videos and imageio is not None and len(frames) > 0:
            fname = video_dir / f'dqn_manual_eval_ep{epi+1}.mp4'
            try:
                imageio.mimwrite(str(fname), frames, fps=30)
                print('Wrote video', fname)
            except Exception as e:
                print('Failed to write video:', e)

        if csv_writer is not None:
            csv_writer.writerow([epi+1, f'{total_r:.6f}', length, col])

    rewards = np.array(rewards)
    lengths = np.array(lengths)
    collisions = np.array(collisions)
    print('=== Manual DQN Eval summary ===')
    print(f'episodes: {len(rewards)}')
    print(f'reward mean={rewards.mean():.3f} std={rewards.std():.3f} median={np.median(rewards):.3f} min={rewards.min():.3f} max={rewards.max():.3f}')
    print(f'length mean={lengths.mean():.1f} std={lengths.std():.1f}')
    print(f'collisions total={collisions.sum()} mean_per_episode={collisions.mean():.3f}')


if __name__ == '__main__':
    main()
