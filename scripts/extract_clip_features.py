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

    from PIL import Image

    for i in tqdm(range(0, len(df), args.batch_size)):
        batch = df.iloc[i:i+args.batch_size]
        imgs = [Image.open(Path(args.images_root) / p).convert('RGB') for p in batch['image_path']]
        inputs = processor(images=imgs, return_tensors='pt', padding=True).to(args.device)
        with torch.no_grad():
            img_features = model.get_image_features(**inputs)
            img_features = torch.nn.functional.normalize(img_features, dim=-1).cpu().numpy()
        embeddings.append(img_features)
        image_paths.extend(batch['image_path'].tolist())
        actions.extend(batch['action'].tolist())
        rewards.extend(batch['reward'].tolist())
        episode_ids.extend(batch['episode_id'].tolist())
        steps.extend(batch['step'].tolist())

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
