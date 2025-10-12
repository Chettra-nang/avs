#!/usr/bin/env python3
"""
ULTRA-FAST Behavior Cloning with Pre-computed CLIP Features

This version uses pre-computed CLIP embeddings, eliminating encoding overhead.
Expected speed: 2-3 minutes for 50 epochs (10-20x faster than on-the-fly encoding)

Usage:
    # First, pre-compute features (run once):
    python3 scripts/precompute_clip_features.py \
        --dataset data/offline_dataset/offline_dataset.npz \
        --output data/offline_dataset/clip_features.npz

    # Then train (ULTRA FAST):
    python3 offline_rl/trainers/train_bc_ultrafast.py \
        --dataset data/offline_dataset/clip_features.npz \
        --output checkpoints/bc_pretrain \
        --epochs 50 \
        --batch-size 4096 \
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
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


class CLIPFeatureDataset(Dataset):
    """Dataset with pre-computed CLIP features - ULTRA FAST!"""
    
    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.clip_features = torch.from_numpy(data['clip_features']).float()
        self.actions = torch.from_numpy(data['action']).long()
        
        # Pin memory for faster GPU transfer
        self.clip_features = self.clip_features.pin_memory()
        self.actions = self.actions.pin_memory()
        
        print(f"✅ Loaded pre-computed CLIP features dataset")
        print(f"   Transitions: {len(self)}")
        print(f"   Feature shape: {self.clip_features.shape}")
        print(f"   Action distribution: {np.bincount(self.actions.numpy())}")
    
    def __len__(self):
        return len(self.clip_features)
    
    def __getitem__(self, idx):
        return self.clip_features[idx], self.actions[idx]


class FastPolicyNetwork(nn.Module):
    """Simple MLP policy - no CLIP encoding needed!"""
    
    def __init__(self, feature_dim: int = 512, n_actions: int = 5):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_actions)
        )
    
    def forward(self, clip_features):
        """Forward pass - features already encoded!"""
        return self.policy(clip_features)


class UltraFastBCTrainer:
    """Ultra-fast BC trainer with pre-computed features."""
    
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        n_actions: int = 5,
        lr: float = 1e-4,
        batch_size: int = 4096,
        device: str = 'cuda',
        mixed_precision: bool = True,
    ):
        # Fast dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
            drop_last=True  # For consistent batch sizes
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        self.device = device
        
        # Lightweight model (no CLIP!)
        self.policy = FastPolicyNetwork(feature_dim=512, n_actions=n_actions).to(device)
        
        # Optimized for speed
        if device == 'cuda':
            self.optimizer = torch.optim.AdamW(
                self.policy.parameters(),
                lr=lr,
                weight_decay=1e-4,
                fused=True  # Faster on CUDA
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.policy.parameters(),
                lr=lr,
                weight_decay=1e-4
            )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50
        )
        
        # Mixed precision
        self.use_amp = mixed_precision and device == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        self.step = 0
        
        print(f"✅ Ultra-fast trainer initialized")
        if self.use_amp:
            print(f"   Mixed precision: ENABLED (FP16)")
        print(f"   Batch size: {batch_size}")
        print(f"   Device: {device}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch - ULTRA FAST!"""
        self.policy.train()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        
        for clip_features, actions in self.train_loader:
            clip_features = clip_features.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)
            
            if self.use_amp:
                # Mixed precision training
                with autocast('cuda'):
                    logits = self.policy(clip_features)
                    loss = F.cross_entropy(logits, actions)
                
                self.optimizer.zero_grad(set_to_none=True)  # Faster
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                logits = self.policy(clip_features)
                loss = F.cross_entropy(logits, actions)
                
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                acc = (logits.argmax(dim=1) == actions).float().mean()
            
            total_loss += loss.item()
            total_acc += acc.item()
            n_batches += 1
            self.step += 1
        
        return {
            'loss': total_loss / n_batches,
            'accuracy': total_acc / n_batches,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate - ULTRA FAST!"""
        self.policy.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        
        for clip_features, actions in self.val_loader:
            clip_features = clip_features.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)
            
            if self.use_amp:
                with autocast('cuda'):
                    logits = self.policy(clip_features)
                    loss = F.cross_entropy(logits, actions)
            else:
                logits = self.policy(clip_features)
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
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
        }, path)


def main():
    parser = argparse.ArgumentParser(description="Ultra-fast BC with pre-computed features")
    parser.add_argument('--dataset', type=str, required=True, 
                       help='Pre-computed CLIP features .npz')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=4096,
                       help='Batch size (RTX 5090: 4096)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split')
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
    print("ULTRA-FAST BEHAVIOR CLONING")
    print("="*60)
    
    # Load pre-computed features
    full_dataset = CLIPFeatureDataset(Path(args.dataset))
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Batches/epoch: ~{len(train_dataset) // args.batch_size}")
    print("="*60)
    print("")
    
    # Create trainer
    trainer = UltraFastBCTrainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_actions=5,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        mixed_precision=not args.no_amp,
    )
    
    # Training loop with timing
    metrics_history = []
    best_val_acc = 0.0
    total_start = time.time()
    
    print("Starting training...")
    print("")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = trainer.train_epoch()
        
        # Validate
        val_metrics = trainer.validate()
        
        # Scheduler
        trainer.scheduler.step()
        
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - total_start
        eta = (elapsed / (epoch + 1)) * (args.epochs - epoch - 1)
        
        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.3f} | "
              f"Val: {val_metrics['val_accuracy']:.3f} | "
              f"Time: {epoch_time:.1f}s | "
              f"ETA: {eta/60:.1f}m")
        
        metrics_history.append({
            'epoch': epoch + 1,
            'epoch_time': epoch_time,
            **train_metrics,
            **val_metrics,
        })
        
        # Save best model
        if val_metrics['val_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['val_accuracy']
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
    print(f"Best val accuracy: {best_val_acc:.3f}")
    print(f"Final val accuracy: {val_metrics['val_accuracy']:.3f}")
    print(f"Checkpoints: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
