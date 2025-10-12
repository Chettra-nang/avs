#!/usr/bin/env python3
"""
Compute CLIP-based reward for images using a fine-tuned CLIP model and saved text embeddings.

This provides a small helper and a CLI to test R_clip for a single image against all action texts.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


class ClipReward:
    def __init__(self, model_dir: str, text_embeddings_path: str = None, device: str = None):
        self.model_dir = Path(model_dir)
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = CLIPProcessor.from_pretrained(self.model_dir)
        self.model = CLIPModel.from_pretrained(self.model_dir).to(self.device)
        # load text embeddings if provided (JSON {text: list})
        self.text_embeddings = None
        if text_embeddings_path:
            p = Path(text_embeddings_path)
            if p.exists():
                with open(p, 'r') as f:
                    data = json.load(f)
                # convert to numpy normalized vectors
                self.texts = list(data.keys())
                self.text_embeddings = np.stack([np.asarray(data[t], dtype=np.float32) for t in self.texts])
            else:
                raise FileNotFoundError(text_embeddings_path)

    def encode_image(self, image_path: str):
        img = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=img, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
            feats = torch.nn.functional.normalize(feats, dim=-1).cpu().numpy()
        return feats[0]

    def compute_similarities(self, image_feat: np.ndarray):
        if self.text_embeddings is None:
            raise RuntimeError('text_embeddings not loaded')
        # both normalized -> cosine similarity is dot product
        sims = (self.text_embeddings @ image_feat).astype(float)
        return sims


def _find_sample_image(frames_csv: Path):
    import pandas as pd
    df = pd.read_csv(frames_csv)
    # pick first agent0 row
    r = df[df['agent_id'] == 0].iloc[0]
    return r['image_path'], r['action']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='AVs/models/clip_finetuned')
    parser.add_argument('--text-emb', default='AVs/models/clip_finetuned/text_embeddings.json')
    parser.add_argument('--frames-csv', default='AVs/data/clip_frames/frames.csv')
    parser.add_argument('--images-root', default='AVs/data/clip_frames')
    parser.add_argument('--image', type=str, default='')
    args = parser.parse_args()

    cr = ClipReward(args.model_dir, args.text_emb)
    if args.image:
        img_path = Path(args.image)
    else:
        img_rel, _ = _find_sample_image(Path(args.frames_csv))
        img_path = Path(args.images_root) / img_rel

    print('Scoring image:', img_path)
    feat = cr.encode_image(str(img_path))
    sims = cr.compute_similarities(feat)
    # print texts and scores
    for t, s in zip(cr.texts, sims.tolist()):
        print(f'{s:0.4f}  {t}')


if __name__ == '__main__':
    main()
