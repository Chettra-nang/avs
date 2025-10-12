#!/usr/bin/env python3
"""
ULTRA-FAST Offline DQN with Pre-computed CLIP Features

Optimized for:
- Python 3.13.3
- PyTorch 2.5.1
- RTX 5090 (33GB VRAM)
- CUDA 12.8

Expected speed: 30-60 seconds for 100 epochs (20-30x faster than on-the-fly encoding)

Usage:
    # Train ultra-fast
    python3 offline_rl/trainers/train_dqn_ultrafast.py \
        --dataset data/offline_dataset/clip_features.npz \
        --output checkpoints/offline_dqn \
        --epochs 100 \
        --batch-size 2048 \
        --device cuda
"""

import argparse
import json
from pathlib import Path
from typing import Dict
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


class CLIPFeatureDataset(Dataset):
    """Dataset with pre-computed CLIP features for DQN."""
    
    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.clip_features = torch.from_numpy(data['clip_features']).float()
        self.actions = torch.from_numpy(data['action']).long()
        self.rewards = torch.from_numpy(data['reward']).float()
        self.next_clip_features = torch.from_numpy(data['next_clip_features']).float()
        self.dones = torch.from_numpy(data['done']).float()
        
        # Pin memory for faster GPU transfer
        self.clip_features = self.clip_features.pin_memory()
        self.actions = self.actions.pin_memory()
        self.rewards = self.rewards.pin_memory()
        self.next_clip_features = self.next_clip_features.pin_memory()
        self.dones = self.dones.pin_memory()
        
        print(f"✅ Loaded DQN dataset with pre-computed CLIP features")
        print(f"   Transitions: {len(self)}")
        print(f"   Feature shape: {self.clip_features.shape}")
        print(f"   Rewards: mean={self.rewards.mean():.3f}, std={self.rewards.std():.3f}")
    
    def __len__(self):
        return len(self.clip_features)
    
    def __getitem__(self, idx):
        return (
            self.clip_features[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_clip_features[idx],
            self.dones[idx]
        )


class FastDQNNetwork(nn.Module):
    """Lightweight DQN Q-network - no CLIP encoding!"""
    
    def __init__(self, feature_dim: int = 512, n_actions: int = 4):
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_actions)
        )
    
    def forward(self, clip_features):
        """Compute Q-values from pre-computed CLIP features."""
        return self.q_network(clip_features)


class UltraFastDQNTrainer:
    """Ultra-fast DQN trainer with pre-computed features."""
    
    def __init__(
        self,
        dataset: Dataset,
        n_actions: int = 4,
        lr: float = 3e-4,
        batch_size: int = 2048,
        gamma: float = 0.99,
        tau: float = 0.005,  # Soft update
        device: str = 'cuda',
        mixed_precision: bool = True,
    ):
        # Fast dataloader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
            drop_last=True
        )
        
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        # Q-networks (main and target)
        self.q_net = FastDQNNetwork(feature_dim=512, n_actions=n_actions).to(device)
        self.target_q_net = FastDQNNetwork(feature_dim=512, n_actions=n_actions).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer (fused for speed)
        if device == 'cuda':
            self.optimizer = torch.optim.AdamW(
                self.q_net.parameters(),
                lr=lr,
                weight_decay=1e-4,
                fused=True
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.q_net.parameters(),
                lr=lr,
                weight_decay=1e-4
            )
        
        # Mixed precision (PyTorch 2.5.1 compatible)
        self.use_amp = mixed_precision and device == 'cuda'
        self.scaler = torch.amp.GradScaler() if self.use_amp else None
        
        self.step = 0
        
        print(f"✅ Ultra-fast DQN trainer initialized")
        if self.use_amp:
            print(f"   Mixed precision: ENABLED (FP16)")
        print(f"   Batch size: {batch_size}")
        print(f"   Target network update: Soft (tau={tau})")
        print(f"   Device: {device}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch - ULTRA FAST!"""
        self.q_net.train()
        total_loss = 0.0
        total_q_mean = 0.0
        n_batches = 0
        
        for state_feat, actions, rewards, next_state_feat, dones in self.dataloader:
            # Move to device
            state_feat = state_feat.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)
            rewards = rewards.to(self.device, non_blocking=True)
            next_state_feat = next_state_feat.to(self.device, non_blocking=True)
            dones = dones.to(self.device, non_blocking=True)
            
            if self.use_amp:
                # Mixed precision training
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # Current Q-values
                    q_values = self.q_net(state_feat)
                    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                    
                    # Target Q-values (no gradient)
                    with torch.no_grad():
                        next_q_values = self.target_q_net(next_state_feat).max(1)[0]
                        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
                    
                    # Loss
                    loss = F.mse_loss(q_values, target_q_values)
                
                # Backward with scaling
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                q_values = self.q_net(state_feat)
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    next_q_values = self.target_q_net(next_state_feat).max(1)[0]
                    target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
                
                loss = F.mse_loss(q_values, target_q_values)
                
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
                self.optimizer.step()
            
            # Soft update target network
            with torch.no_grad():
                for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
                    target_param.data.mul_(1 - self.tau)
                    target_param.data.add_(self.tau * param.data)
            
            # Metrics
            with torch.no_grad():
                total_loss += loss.item()
                total_q_mean += q_values.mean().item()
            
            n_batches += 1
            self.step += 1
        
        return {
            'loss': total_loss / n_batches,
            'q_mean': total_q_mean / n_batches,
        }
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_q_net_state_dict': self.target_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
        }, path)


def main():
    parser = argparse.ArgumentParser(description="Ultra-fast DQN with pre-computed features")
    parser.add_argument('--dataset', type=str, required=True,
                       help='Pre-computed CLIP features .npz')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=2048,
                       help='Batch size (RTX 5090: 2048)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Enable RTX 5090 optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"✅ RTX 5090 optimizations: TF32 + cuDNN benchmark")
    
    print("")
    print("="*60)
    print("ULTRA-FAST OFFLINE DQN")
    print("="*60)
    
    # Load dataset
    dataset = CLIPFeatureDataset(Path(args.dataset))
    
    print(f"   Batch size: {args.batch_size}")
    print(f"   Batches/epoch: ~{len(dataset) // args.batch_size}")
    print("="*60)
    print("")
    
    # Create trainer
    trainer = UltraFastDQNTrainer(
        dataset=dataset,
        n_actions=4,
        lr=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        device=args.device,
        mixed_precision=not args.no_amp,
    )
    
    # Training loop
    metrics_history = []
    best_loss = float('inf')
    total_start = time.time()
    
    print("Starting training...")
    print("")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = trainer.train_epoch()
        
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - total_start
        eta = (elapsed / (epoch + 1)) * (args.epochs - epoch - 1)
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"Q: {train_metrics['q_mean']:.3f} | "
              f"Time: {epoch_time:.1f}s | "
              f"ETA: {eta/60:.1f}m")
        
        metrics_history.append({
            'epoch': epoch + 1,
            'epoch_time': epoch_time,
            **train_metrics,
        })
        
        # Save best model
        if train_metrics['loss'] < best_loss:
            best_loss = train_metrics['loss']
            trainer.save_checkpoint(output_dir / 'best_model.pt')
        
        # Periodic checkpoints
        if (epoch + 1) % 20 == 0:
            trainer.save_checkpoint(output_dir / f'checkpoint_epoch{epoch+1}.pt')
    
    # Save final
    trainer.save_checkpoint(output_dir / 'final_model.pt')
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    total_time = time.time() - total_start
    
    print("")
    print("="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f}s)")
    print(f"Avg time/epoch: {total_time/args.epochs:.1f}s")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
