#!/usr/bin/env python3
"""
Behavior Cloning (Imitation Learning) from collected parquet transitions.

Usage:
    python tools/train_bc.py \
        --dataset ../../../AVs/data/offline_dataset/offline_dataset.npz \
        --output checkpoints/bc_pretrain \
        --epochs 50 \
        --batch-size 512

Trains a policy network to imitate collected actions via supervised learning.
The pretrained policy can then be fine-tuned with online PPO.
"""
import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# Import CLIP encoder
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from rl_langvision.clip_embedder import CLIPImageEncoder


class OfflineRLDataset(Dataset):
    """PyTorch dataset for offline RL transitions."""
    
    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.obs = torch.from_numpy(data['obs']).float() / 255.0
        self.actions = torch.from_numpy(data['action']).long()
        
        print(f"Loaded BC dataset: {len(self)} transitions")
        print(f"  Obs shape: {self.obs.shape}")
        print(f"  Action distribution: {np.bincount(self.actions.numpy())}")
    
    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


class CLIPPolicyNetwork(nn.Module):
    """
    Policy network with CLIP vision encoder.
    
    Takes grayscale (C,H,W) observations, converts to RGB, encodes with CLIP,
    and outputs action logits.
    """
    
    def __init__(self, n_actions: int = 5, hidden_dim: int = 256, freeze_clip: bool = True):
        super().__init__()
        self.n_actions = n_actions
        
        # CLIP encoder
        self.clip_encoder = CLIPImageEncoder(
            model_name="ViT-B-32",
            pretrained="openai",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        if freeze_clip:
            for param in self.clip_encoder.model.parameters():
                param.requires_grad = False
        
        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_actions),
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, C, H, W) grayscale
        Returns:
            logits: (B, n_actions)
        """
        # Convert to RGB (take last frame if temporal stack)
        if obs.shape[1] == 4:
            obs_rgb = obs[:, -1:].repeat(1, 3, 1, 1)
        else:
            obs_rgb = obs.repeat(1, 3, 1, 1)
        
        # Encode
        B = obs_rgb.shape[0]
        clip_feats = []
        for i in range(B):
            img = (obs_rgb[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            feat = self.clip_encoder.encode_np_rgb(img)
            clip_feats.append(torch.from_numpy(feat))
        
        clip_feats = torch.stack(clip_feats).to(obs.device)
        
        # Policy
        logits = self.policy(clip_feats)
        return logits


class BCTrainer:
    """Behavior cloning trainer."""
    
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        n_actions: int = 5,
        lr: float = 1e-4,
        batch_size: int = 512,
        device: str = 'cuda',
    ):
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.device = device
        
        # Model
        self.policy = CLIPPolicyNetwork(n_actions=n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        
        self.step = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch."""
        self.policy.train()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        
        for obs, actions in tqdm(self.train_loader, desc="Training", leave=False):
            obs = obs.to(self.device)
            actions = actions.to(self.device)
            
            # Forward pass
            logits = self.policy(obs)
            loss = F.cross_entropy(logits, actions)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            acc = (logits.argmax(dim=1) == actions).float().mean()
            total_loss += loss.item()
            total_acc += acc.item()
            n_batches += 1
            self.step += 1
        
        return {
            'loss': total_loss / n_batches,
            'accuracy': total_acc / n_batches,
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.policy.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for obs, actions in tqdm(self.val_loader, desc="Validating", leave=False):
                obs = obs.to(self.device)
                actions = actions.to(self.device)
                
                logits = self.policy(obs)
                loss = F.cross_entropy(logits, actions)
                acc = (logits.argmax(dim=1) == actions).float().mean()
                
                total_loss += loss.item()
                total_acc += acc.item()
                n_batches += 1
        
        return {
            'val_loss': total_loss / n_batches,
            'val_accuracy': total_acc / n_batches,
        }
    
    def save_checkpoint(self, path: Path):
        """Save checkpoint."""
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': self.step,
        }, path)


def main():
    parser = argparse.ArgumentParser(description="Train behavior cloning policy")
    parser.add_argument('--dataset', type=str, required=True, help='Path to offline_dataset.npz')
    parser.add_argument('--output', type=str, default='checkpoints/bc_pretrain', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and split dataset
    full_dataset = OfflineRLDataset(Path(args.dataset))
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create trainer
    trainer = BCTrainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_actions=5,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # Training loop
    metrics_history = []
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = trainer.train_epoch()
        
        # Validate
        val_metrics = trainer.validate()
        
        # Scheduler step
        trainer.scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.3f} | "
              f"Val Loss: {val_metrics['val_loss']:.4f} | "
              f"Val Acc: {val_metrics['val_accuracy']:.3f}")
        
        metrics_history.append({
            'epoch': epoch + 1,
            **train_metrics,
            **val_metrics,
        })
        
        # Save best model
        if val_metrics['val_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['val_accuracy']
            trainer.save_checkpoint(output_dir / 'best_model.pt')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(output_dir / f'checkpoint_epoch{epoch+1}.pt')
    
    # Save final model
    trainer.save_checkpoint(output_dir / 'final_model.pt')
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    print(f"✅ BC training complete! Best val accuracy: {best_val_acc:.3f}")
    print(f"Models saved to {output_dir}")


if __name__ == '__main__':
    main()
