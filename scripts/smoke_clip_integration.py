#!/usr/bin/env python3
"""Smoke test: run a small episode batch through ClipRewardWrapper (or a local equivalent)
and log base reward vs CLIP reward statistics.

This is intentionally lightweight: it uses a dummy Gym-like env that emits random
224x224 RGB frames so we can validate the wrapper integration and scoring
pipeline without depending on the full simulator.
"""
import sys
import os
import argparse
import random
import json
from collections import defaultdict

import numpy as np
import torch

try:
    from PIL import Image
except Exception:
    Image = None

def add_repo_to_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

add_repo_to_path()

def safe_load_text_embeddings(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        # data: {label: [floats]}
        labels = list(data.keys())
        embs = np.stack([np.array(data[k], dtype=np.float32) for k in labels])
        return labels, embs
    return None, None

class DummyEnv:
    """A tiny gym-like env returning random 224x224 RGB frames and random base rewards."""
    def __init__(self, seed=0, max_steps=50, n_actions=5):
        self.observation_space = None
        self.action_space = type('A', (), {'n': n_actions})
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return self._obs()

    def _obs(self):
        # uint8 RGB 224x224
        return (np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))

    def step(self, action):
        self.step_count += 1
        obs = self._obs()
        # base reward: small random float
        base_r = float(self.rng.uniform(-1.0, 1.0))
        done = self.step_count >= self.max_steps
        info = {'step': self.step_count}
        return obs, base_r, done, info


def compute_clip_scores(model_dir, images, device='cpu', text_emb_path=None):
    """Load CLIP (or use finetuned model from model_dir) and compute cosine similarities
    between images and provided text embeddings. Returns max similarity per image.
    """
    from transformers import CLIPProcessor, CLIPModel

    # Try to load saved text embeddings first
    labels, text_embs = None, None
    if text_emb_path:
        labels, text_embs = safe_load_text_embeddings(text_emb_path)

    # fallback prompts
    if labels is None or text_embs is None:
        labels = ["action_0", "action_1", "action_2", "action_3", "action_4"]
        # we'll compute text embeddings on the fly
        compute_text_on_the_fly = True
    else:
        compute_text_on_the_fly = False

    # Load CLIP model (prefer local model_dir if available)
    try:
        model = CLIPModel.from_pretrained(model_dir)
        processor = CLIPProcessor.from_pretrained(model_dir)
    except Exception:
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    model = model.to(device)
    model.eval()

    if compute_text_on_the_fly:
        # create text embeddings
        texts = [f"ambulance behavior {lab}" for lab in labels]
        with torch.no_grad():
            inputs = processor(text=texts, images=None, return_tensors='pt', padding=True)
            text_outputs = model.get_text_features(**{k: v.to(device) for k, v in inputs.items()})
            text_embs = text_outputs.cpu().numpy()

    # normalize text embeddings
    text_embs = text_embs.astype(np.float32)
    text_embs /= np.linalg.norm(text_embs, axis=1, keepdims=True) + 1e-10

    # process images in a batch
    # convert to PIL if needed
    imgs = []
    for im in images:
        if isinstance(im, np.ndarray):
            if Image is None:
                imgs.append(im)
            else:
                imgs.append(Image.fromarray(im))
        else:
            imgs.append(im)

    with torch.no_grad():
        proc = processor(images=imgs, return_tensors='pt')
        pixel_values = proc['pixel_values'].to(device)
        img_feats = model.get_image_features(pixel_values)
        img_feats = img_feats.cpu().numpy()
        img_feats /= np.linalg.norm(img_feats, axis=1, keepdims=True) + 1e-10

    # cosine similarities
    sims = img_feats.dot(text_embs.T)
    # for each image return max similarity (this is a design choice for smoke test)
    max_sims = sims.max(axis=1)
    return max_sims, sims


def run_smoke(args):
    out_csv = args.out_csv
    env = DummyEnv(seed=args.seed, max_steps=args.max_steps, n_actions=5)
    device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'

    rows = []
    per_episode = []
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_base = 0.0
        ep_clip = 0.0
        step = 0
        while not done:
            # choose random action
            action = random.randrange(env.action_space.n)
            next_obs, base_r, done, info = env.step(action)
            # compute clip score for this single image
            max_sim, sims = compute_clip_scores(args.model_dir, [next_obs], device=device, text_emb_path=args.text_embeddings)
            r_clip = float(max_sim[0])
            total = base_r + args.clip_weight * r_clip
            rows.append({'episode': ep, 'step': step, 'base_reward': base_r, 'r_clip': r_clip, 'total_reward': total})
            ep_base += base_r
            ep_clip += r_clip
            step += 1
        per_episode.append({'episode': ep, 'base_sum': ep_base, 'clip_sum': ep_clip, 'mean_clip': ep_clip / float(args.max_steps)})

    # write CSV
    import csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'step', 'base_reward', 'r_clip', 'total_reward'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # print summary
    print('Episodes:', args.episodes)
    total_base = sum(p['base_sum'] for p in per_episode)
    total_clip = sum(p['clip_sum'] for p in per_episode)
    print(f'Total base reward sum: {total_base:.3f}')
    print(f'Total clip similarity sum: {total_clip:.3f}')
    print('Per-episode sample (first 5):')
    for p in per_episode[:5]:
        print(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='AVs/models/clip_finetuned', help='Path to finetuned CLIP (or model id)')
    parser.add_argument('--text-embeddings', default='AVs/models/clip_finetuned/text_embeddings.json', help='Optional text embeddings json')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=20)
    parser.add_argument('--out-csv', default='AVs/data/clip_smoke_eval.csv')
    parser.add_argument('--clip-weight', type=float, default=1.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force-cpu', dest='force_cpu', action='store_true')
    args = parser.parse_args()
    run_smoke(args)


if __name__ == '__main__':
    main()
