#!/usr/bin/env python3
"""Small smoke trainer that loads AVs/data/clip_features.npz and runs a quick
1-epoch classifier on the embeddings to validate data loading and trainer wiring.
"""
import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class EmbeddingDataset(Dataset):
    def __init__(self, npz_path, max_samples=None):
        data = np.load(npz_path, allow_pickle=True)
        self.emb = data['image_embeddings']
        # map actions to labels
        self.y = data['action'].astype(int)
        if max_samples:
            idx = np.random.choice(len(self.emb), min(max_samples, len(self.emb)), replace=False)
            self.emb = self.emb[idx]
            self.y = self.y[idx]

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, idx):
        return torch.from_numpy(self.emb[idx]).float(), int(self.y[idx])


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, n_classes=5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.fc(x)


def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * x.size(0)
        total += x.size(0)
    return total_loss / (total + 1e-12)


def eval_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return correct / (total + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', default='AVs/data/clip_features.npz')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max-samples', type=int, default=2000)
    args = parser.parse_args()

    if not os.path.exists(args.npz):
        print('ERROR: npz not found:', args.npz)
        return

    ds = EmbeddingDataset(args.npz, max_samples=args.max_samples)
    # split
    n = len(ds)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    in_dim = ds.emb.shape[1]
    n_classes = int(ds.y.max()) + 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SimpleClassifier(in_dim, n_classes=n_classes).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    for ep in range(args.epochs):
        loss = train_one_epoch(model, train_loader, opt, device)
        acc = eval_model(model, val_loader, device)
        print(f'epoch {ep+1}/{args.epochs} loss={loss:.4f} val_acc={acc:.4f}')

    # save a tiny checkpoint
    os.makedirs('AVs/models/smoke_offline', exist_ok=True)
    torch.save({'model_state': model.state_dict()}, 'AVs/models/smoke_offline/checkpoint.pt')
    print('Saved smoke checkpoint to AVs/models/smoke_offline/checkpoint.pt')


if __name__ == '__main__':
    main()
