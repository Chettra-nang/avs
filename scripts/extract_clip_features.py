#!/usr/bin/env python3
"""
Compute CLIP image embeddings for images listed in frames.csv using a fine-tuned CLIP model.

Outputs a .npz with arrays: image_embeddings, action, reward, episode_id, step, image_path
so offline trainers can be adapted to load them. This script expects a model dir produced by `finetune_clip.py`.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames-csv', required=True)
    parser.add_argument('--images-root', default='AVs/data/clip_frames')
    parser.add_argument('--model-dir', required=True, help='Directory with fine-tuned CLIP (from finetune_clip.py)')
    parser.add_argument('--out-npz', default='AVs/data/clip_features.npz')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    df = pd.read_csv(args.frames_csv)
    processor = CLIPProcessor.from_pretrained(args.model_dir)
    model = CLIPModel.from_pretrained(args.model_dir).to(args.device)
    model.eval()

    embeddings = []
    image_paths = []
    actions = []
    rewards = []
    episode_ids = []
    steps = []

    # Resolve image paths robustly (similar to finetune script)
    resolved_rows = []
    for _, row in df.iterrows():
        rel = Path(row['image_path'])
        candidates = [
            Path(args.images_root) / rel,
            Path(args.images_root) / 'images' / rel.name,
            Path(args.images_root) / rel.name,
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break
        if found is not None:
            resolved_rows.append((found, row))
        else:
            print(f"[extract_clip_features] warning: missing image, skipping: {rel}")

    if not resolved_rows:
        raise RuntimeError('No resolvable images found for feature extraction')

    from PIL import Image

    # iterate over resolved rows in batches
    for i in tqdm(range(0, len(resolved_rows), args.batch_size)):
        batch = resolved_rows[i:i+args.batch_size]
        imgs = [Image.open(x[0]).convert('RGB') for x in batch]
        batch_rows = [x[1] for x in batch]
        inputs = processor(images=imgs, return_tensors='pt', padding=True).to(args.device)
        with torch.no_grad():
            img_features = model.get_image_features(**inputs)
            img_features = torch.nn.functional.normalize(img_features, dim=-1).cpu().numpy()
        embeddings.append(img_features)
        image_paths.extend([r['image_path'] for r in batch_rows])
        actions.extend([r['action'] for r in batch_rows])
        rewards.extend([r['reward'] for r in batch_rows])
        episode_ids.extend([r['episode_id'] for r in batch_rows])
        steps.extend([r['step'] for r in batch_rows])

    embeddings = np.vstack(embeddings)
    np.savez_compressed(args.out_npz,
                        image_embeddings=embeddings,
                        image_path=np.array(image_paths),
                        action=np.array(actions),
                        reward=np.array(rewards),
                        episode_id=np.array(episode_ids),
                        step=np.array(steps))
    print('Wrote', args.out_npz, 'with', embeddings.shape[0], 'embeddings')


if __name__ == '__main__':
    main()
