#!/usr/bin/env python3
"""
Fine-tune CLIP (ViT-B/32) last layers on a small (image, text) dataset.

Saves a checkpoint and an `action_embeds.npz` file containing L2-normalized
text embeddings for each action label.

Usage (quick):
  python clip_finetune/finetune_clip.py --data-dir data/highway_multimodal_dataset/frames \
      --captions data/highway_multimodal_dataset/texts/intersection_four_way.csv \
      --output clip_finetune/outputs --model roberta-ViT-B-32 --epochs 3

"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    import open_clip
except Exception:
    open_clip = None


class ImageTextDataset(Dataset):
    def __init__(self, captions_csv: str, root_dir: str, preprocess):
        import csv
        self.root = Path(root_dir)
        self.rows = []
        with open(captions_csv, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append(r)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        # If the CSV stored a full path, use it; otherwise join with root and keep subdir if present
        frame_file = r['frame_file']
        p = Path(frame_file)
        if p.parent and str(p.parent) != '.':
            # CSV contains subdirectory or absolute path -> use as-is (but make relative to repo if needed)
            img_path = Path(frame_file)
            if not img_path.is_absolute():
                # Keep relative path as provided
                img_path = Path(frame_file)
        else:
            img_path = self.root / Path(r['frame_file']).name
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        img_t = self.preprocess(img)
        text = r.get('instruction', '')
        action_id = int(r.get('action_id', 1))
        return img_t, text, action_id


def freeze_except_projection(model):
    # Freeze all parameters first
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze projection heads and last visual block if present
    try:
        # Visual proj
        if hasattr(model.visual, 'proj'):
            for p in model.visual.proj.parameters():
                p.requires_grad = True
    except Exception:
        pass
    try:
        # Text proj
        if hasattr(model.transformer, 'proj'):
            for p in model.transformer.proj.parameters():
                p.requires_grad = True
    except Exception:
        pass
    # Try to unfreeze last transformer block (vision)
    try:
        blocks = getattr(model.visual, 'transformer', None) or getattr(model.visual, 'blocks', None)
        if blocks is not None:
            last = blocks[-1]
            for p in last.parameters():
                p.requires_grad = True
    except Exception:
        pass


def train(args):
    if open_clip is None:
        raise SystemExit('open_clip not available. Install with: pip install open_clip_torch')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = args.model
    print(f'Loading model {model_name}...')
    # Use a pretrained tag available in this open_clip installation
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s12b_b32k')
    model.to(device)

    # Freeze strategy
    freeze_except_projection(model)

    # Ensure we have parameters to optimize; if freeze_except_projection left
    # no trainable parameters (various open_clip variants differ), fall back
    # to unfreezing the whole model for the smoke run.
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        print('Warning: no trainable parameters found after freezing; unfreezing entire model for smoke run')
        for p in model.parameters():
            p.requires_grad = True
        params = [p for p in model.parameters() if p.requires_grad]

    # Prepare dataset
    dataset = ImageTextDataset(args.captions, args.data_dir, preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Use simple optimizer on parameters that require grad
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for imgs, texts, aids in loader:
            imgs = imgs.to(device)
            # tokenize texts
            tokenized = open_clip.tokenize(texts).to(device)
            optimizer.zero_grad()
            image_embeds = model.encode_image(imgs)
            text_embeds = model.encode_text(tokenized)
            # normalize
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            # contrastive logits
            logits_per_image = 100.0 * image_embeds @ text_embeds.t()
            labels = torch.arange(len(imgs), device=device)
            loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f'Epoch {epoch+1}/{args.epochs} loss={running:.4f}')

    # Save checkpoint
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / 'clip_finetuned.pt'
    torch.save(model.state_dict(), ckpt_path)
    print('Saved checkpoint to', ckpt_path)

    # Compute and save normalized text embeddings for each action label
    prompts = args.prompts
    if not prompts:
        prompts = {
            0: "Drive slower approaching cross traffic",
            1: "Maintain speed / hold position",
            2: "Drive faster / accelerate to clear intersection",
        }

    model.eval()
    with torch.no_grad():
        all_embeds = {}
        for aid, prompt in prompts.items():
            tok = open_clip.tokenize([prompt]).to(device)
            emb = model.encode_text(tok)
            emb = emb.cpu().numpy().astype(np.float32)
            # normalize
            emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
            all_embeds[str(aid)] = emb.squeeze()

    npz_path = out / 'action_embeds.npz'
    np.savez(npz_path, **all_embeds)
    meta = {'model': model_name, 'prompts': prompts}
    with open(out / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print('Saved action embeddings to', npz_path)


def parse_prompts_file(path: str) -> dict:
    if not path:
        return {}
    d = {}
    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                d[int(parts[0])] = parts[1]
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Root dir where frames are stored')
    parser.add_argument('--captions', type=str, required=True, help='CSV with columns frame_file,instruction,action_id')
    parser.add_argument('--output', type=str, default='clip_finetune/outputs', help='Output folder')
    parser.add_argument('--model', type=str, default='roberta-ViT-B-32', help='open_clip model name')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--prompts-file', type=str, default=None, help='Optional action prompts CSV')
    args = parser.parse_args()

    prompts = parse_prompts_file(args.prompts_file) if args.prompts_file else None
    args.prompts = prompts

    train(args)


if __name__ == '__main__':
    main()
