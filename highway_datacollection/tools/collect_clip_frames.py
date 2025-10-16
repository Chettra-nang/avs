#!/usr/bin/env python3
"""
Collect frames from highway-env scenarios for CLIP fine-tuning.

Saves PNG frames and a CSV with columns: frame_file,instruction,action_id,scenario,episode,step

This script uses the project's MultiAgentEnvFactory to create environments and
samples actions randomly to gather diverse visual examples. It defaults to
selecting a grayscale observation modality when available.
"""
import argparse
import csv
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image

from ..environments.factory import MultiAgentEnvFactory


DEFAULT_PROMPTS = {
    0: "Drive slower approaching cross traffic",
    1: "Maintain speed / hold position",
    2: "Drive faster / accelerate to clear intersection",
}


def choose_obs_type(factory: MultiAgentEnvFactory) -> str:
    types = factory.get_supported_observation_types()
    # prefer 'grayscale' if available, otherwise pick the first
    for t in types:
        if 'gray' in t.lower() or 'grayscale' in t.lower():
            return t
    return types[0]


def to_rgb_and_save(img_arr: np.ndarray, path: Path) -> None:
    """Convert a numpy image to RGB and save as PNG.

    Accepts HxWx3 or HxW (grayscale) arrays.
    """
    if img_arr is None:
        raise ValueError("No image data to save")

    if img_arr.ndim == 2:
        img = Image.fromarray(img_arr.astype('uint8'), mode='L').convert('RGB')
    elif img_arr.ndim == 3 and img_arr.shape[2] == 3:
        img = Image.fromarray(img_arr.astype('uint8'))
    else:
        # try to squeeze channel-first
        # handle channel-first (C,H,W)
        if img_arr.ndim == 3 and img_arr.shape[0] in (1, 3):
            arr = np.transpose(img_arr, (1, 2, 0))
            if arr.shape[2] == 1:
                img = Image.fromarray(arr.squeeze().astype('uint8'), mode='L').convert('RGB')
            else:
                img = Image.fromarray(arr.astype('uint8'))
        # handle stacked grayscale frames (T,H,W) -> take last frame
        elif img_arr.ndim == 3 and img_arr.shape[0] > 3 and img_arr.shape[1] > 10:
            last = img_arr[-1]
            img = Image.fromarray(last.astype('uint8'), mode='L').convert('RGB')
        else:
            raise ValueError(f"Unrecognized image shape: {img_arr.shape}")

    img.save(path)


def main():
    parser = argparse.ArgumentParser(description='Collect frames for CLIP fine-tuning')
    parser.add_argument('--scenario', type=str, default='intersection_four_way', help='Scenario name')
    parser.add_argument('--n-episodes', type=int, default=50, help='Episodes to run')
    parser.add_argument('--max-steps', type=int, default=200, help='Max steps per episode')
    parser.add_argument('--target-frames', type=int, default=500, help='Approx number of frames to collect')
    parser.add_argument('--output-dir', type=str, default='data/highway_multimodal_dataset', help='Output dir')
    parser.add_argument('--n-agents', type=int, default=1, help='Controlled agents')
    parser.add_argument('--force-render', action='store_true', help='Force offscreen rgb_array rendering on the env (recommended)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--prompts-file', type=str, default=None, help='Optional prompts CSV (action_id,prompt)')
    parser.add_argument('--every-n-steps', type=int, default=3, help='Save one frame every N steps')
    args = parser.parse_args()

    random.seed(args.seed)

    out_root = Path(args.output_dir)
    frames_dir = out_root / 'frames' / args.scenario
    texts_dir = out_root / 'texts'
    frames_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = texts_dir / f'{args.scenario}.csv'

    factory = MultiAgentEnvFactory()
    obs_type = choose_obs_type(factory)
    env = factory.create_env(args.scenario, obs_type, args.n_agents)

    if args.force_render:
        # Try to enable offscreen rendering and rgb_array mode on the env
        try:
            if hasattr(env, 'unwrapped'):
                uw = env.unwrapped
                # set offscreen_rendering in config if present
                try:
                    if hasattr(uw, 'config') and isinstance(uw.config, dict):
                        uw.config['offscreen_rendering'] = True
                        uw.config['render_mode'] = 'rgb_array'
                except Exception:
                    pass
                # set attributes directly
                try:
                    setattr(uw, 'render_mode', 'rgb_array')
                except Exception:
                    pass
        except Exception:
            pass

    # action prompts
    action_prompts = DEFAULT_PROMPTS.copy()
    if args.prompts_file:
        # expect CSV with action_id,prompt
        with open(args.prompts_file, 'r') as pf:
            for line in pf:
                parts = line.strip().split(',', 1)
                if len(parts) == 2:
                    aid = int(parts[0])
                    action_prompts[aid] = parts[1]

    total_saved = 0
    start_time = time.time()

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame_file', 'instruction', 'action_id', 'scenario', 'episode', 'step'])

        for ep in range(args.n_episodes):
            obs, info = env.reset()
            # Some envs return (obs, info) tuple; support both
            if isinstance(obs, tuple) or isinstance(obs, list):
                obs = obs[0]

            for step in range(args.max_steps):
                # sample random action
                try:
                    action = env.action_space.sample()
                except Exception:
                    action = 0

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Save every N steps to reduce correlation
                if step % args.every_n_steps == 0:
                    # try to get rendered rgb image
                    img = None
                    try:
                        # highway-env supports render(mode='rgb_array')
                        arr = env.render(mode='rgb_array')
                        img = arr
                    except Exception:
                        # try reading from observation
                        if isinstance(obs, dict):
                            # try common keys
                            for key in ('rgb', 'image', 'screen'):
                                if key in obs:
                                    img = obs[key]
                                    break
                        else:
                            # if it's grayscale stack, use last frame
                            try:
                                a = np.array(obs)
                                if a.ndim >= 3:
                                    # stack,channels,H,W or frames,H,W
                                    if a.shape[0] in (1, 3):
                                        img = np.transpose(a, (1, 2, 0))
                                    else:
                                        # take last frame
                                        img = a[-1]
                            except Exception:
                                img = None

                    if img is not None:
                        frame_name = f"{args.scenario}_ep{ep:04d}_s{step:04d}.png"
                        frame_path = frames_dir / frame_name
                        try:
                            to_rgb_and_save(np.asarray(img), frame_path)
                        except Exception as e:
                            print(f"Failed to save frame: {e}")
                        # choose instruction mapping based on action if possible
                        aid = int(action) if isinstance(action, (int, float)) or hasattr(action, '__int__') else 0
                        if aid not in action_prompts:
                            # map to 0/1/2 by clipping
                            aid = int(aid) % 3
                        instruction = action_prompts.get(aid, action_prompts[1])
                        writer.writerow([str(frame_path), instruction, aid, args.scenario, ep, step])
                        total_saved += 1

                if done:
                    break

                # Stop early if reached target frames
                if total_saved >= args.target_frames:
                    break

            if total_saved >= args.target_frames:
                break

    elapsed = time.time() - start_time
    print(f"Saved {total_saved} frames to {frames_dir} and labels to {csv_path} in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
