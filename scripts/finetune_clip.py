#!/usr/bin/env python3
"""
Fine-tune CLIP ViT-B/32 on a small image->text dataset.

This script expects a CSV with columns `image_path,text` (relative image_path under --images-root).
It loads a pretrained CLIP model from Hugging Face (`openai/clip-vit-base-patch32`), freezes most parameters,
and fine-tunes the projection heads with a contrastive loss or cross-entropy depending on setup.

This is a minimal, opinionated script intended for quick experiments. It assumes you have `transformers`,
`torch`, and `datasets` installed. Adjust hyperparameters as needed.

Outputs: saved model to --out-dir and a small JSON listing text embeddings for label texts.
"""

import argparse
import json
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


class ImageTextDataset(Dataset):
    def __init__(self, csv_path, images_root, transform=None):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.images_root = Path(images_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.images_root / row['image_path']).convert('RGB')
        text = str(row['text'])
        return img, text


def collate_fn(batch, processor):
    imgs, texts = zip(*batch)
    inputs = processor(text=list(texts), images=list(imgs), return_tensors='pt', padding=True)
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--images-root', default='AVs/data/clip_frames')
    parser.add_argument('--out-dir', default='AVs/models/clip_finetuned')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = args.device
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    model.to(device)

    # Freeze most parameters: keep projection layers trainable
    for name, p in model.named_parameters():
        p.requires_grad = False
    # make final projection layers trainable
    for name, p in model.visual_projection.named_parameters():
        p.requires_grad = True
    for name, p in model.text_projection.named_parameters():
        p.requires_grad = True

    ds = ImageTextDataset(args.csv, args.images_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, processor))

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f'epoch {epoch+1}/{args.epochs}')
        for batch in pbar:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            # use cosine similarity + softmax cross-entropy on logits
            logits_per_image = outputs.logits_per_image
            labels = torch.arange(logits_per_image.size(0), device=device)
            loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.cpu().detach().numpy()))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)

    # save text embeddings for unique labels
    import pandas as pd
    df = pd.read_csv(args.csv)
    texts = df['text'].drop_duplicates().tolist()
    model.eval()
    with torch.no_grad():
        inputs = processor(text=texts, images=None, return_tensors='pt', padding=True).to(device)
        text_embeds = model.get_text_features(**inputs)
        text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1).cpu().numpy()
    emb_map = {t: text_embeds[i].tolist() for i, t in enumerate(texts)}
    with open(out_dir / 'text_embeddings.json', 'w') as f:
        json.dump(emb_map, f)
    print('Saved fine-tuned model to', out_dir)


if __name__ == '__main__':
    main()
