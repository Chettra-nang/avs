#!/usr/bin/env python3
"""
ULTRA-FAST Offline PPO with Pre-computed CLIP Features

Optimized for:
- Python 3.13.3
- PyTorch 2.5.1
- RTX 5090 (33GB VRAM)
- CUDA 12.8

Expected speed: 40-80 seconds for 100 epochs (20-30x faster than on-the-fly encoding)

Usage:
    # Train ultra-fast
    python3 offline_rl/trainers/train_ppo_ultrafast.py \
        --dataset data/offline_dataset/clip_features.npz \
        --output checkpoints/offline_ppo \
        --epochs 100 \
        --batch-size 1536 \
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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class CLIPFeatureDataset(Dataset):
    """Dataset with pre-computed CLIP features for PPO."""
    
    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.clip_features = torch.from_numpy(data['clip_features']).float()
        self.actions = torch.from_numpy(data['action']).long()
        self.rewards = torch.from_numpy(data['reward']).float()
        self.next_clip_features = torch.from_numpy(data['next_clip_features']).float()
        self.dones = torch.from_numpy(data['done']).float()
        
        # Compute advantages (GAE)
        self.advantages, self.returns = self._compute_gae()
        
        # Pin memory
        self.clip_features = self.clip_features.pin_memory()
        self.actions = self.actions.pin_memory()
        self.advantages = self.advantages.pin_memory()
        self.returns = self.returns.pin_memory()
        
        print(f"✅ Loaded PPO dataset with pre-computed CLIP features")
        print(f"   Transitions: {len(self)}")
        print(f"   Feature shape: {self.clip_features.shape}")
        print(f"   Advantages: mean={self.advantages.mean():.3f}, std={self.advantages.std():.3f}")
    
    def _compute_gae(self, gamma=0.99, lambda_=0.95):
        """Compute GAE advantages."""
        # Simple advantage estimation (can be improved)
        advantages = []
        returns = []
        
        # Compute returns
        G = 0
        for i in reversed(range(len(self.rewards))):
            G = self.rewards[i] + gamma * G * (1 - self.dones[i])
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize advantages (use returns as proxy)
        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return advantages, returns
    
    def __len__(self):
        return len(self.clip_features)
    
    def __getitem__(self, idx):
        return (
            self.clip_features[idx],
            self.actions[idx],
            self.advantages[idx],
            self.returns[idx]
        )


class FastActorCritic(nn.Module):
    """Lightweight Actor-Critic - no CLIP encoding!"""
    
    def __init__(self, feature_dim: int = 512, n_actions: int = 4):
        super().__init__()
        
        # Shared feature layer
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_actions)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, clip_features):
        """Compute action logits and value from pre-computed CLIP features."""
        shared_features = self.shared(clip_features)
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_logits, value.squeeze(-1)


class UltraFastPPOTrainer:
    """Ultra-fast PPO trainer with pre-computed features."""
    
    def __init__(
        self,
        dataset: Dataset,
        n_actions: int = 4,
        lr: float = 3e-4,
        batch_size: int = 1536,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
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
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        
        # Actor-Critic network
        self.ac_net = FastActorCritic(feature_dim=512, n_actions=n_actions).to(device)
        
        # Optimizer (fused for speed)
        if device == 'cuda':
            self.optimizer = torch.optim.AdamW(
                self.ac_net.parameters(),
                lr=lr,
                weight_decay=1e-4,
                fused=True
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.ac_net.parameters(),
                lr=lr,
                weight_decay=1e-4
            )
        
        # Mixed precision (PyTorch 2.5.1 compatible)
        self.use_amp = mixed_precision and device == 'cuda'
        self.scaler = torch.amp.GradScaler() if self.use_amp else None
        
        self.step = 0
        
        print(f"✅ Ultra-fast PPO trainer initialized")
        if self.use_amp:
            print(f"   Mixed precision: ENABLED (FP16)")
        print(f"   Batch size: {batch_size}")
        print(f"   Clip coefficient: {clip_coef}")
        print(f"   Device: {device}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch - ULTRA FAST!"""
        self.ac_net.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_batches = 0
        
        for state_feat, actions, advantages, returns in self.dataloader:
            # Move to device
            state_feat = state_feat.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)
            advantages = advantages.to(self.device, non_blocking=True)
            returns = returns.to(self.device, non_blocking=True)
            
            if self.use_amp:
                # Mixed precision training
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # Forward pass
                    action_logits, values = self.ac_net(state_feat)
                    
                    # Policy loss (PPO clipped objective)
                    log_probs = F.log_softmax(action_logits, dim=-1)
                    action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                    
                    ratio = torch.exp(action_log_probs)  # Simplified (no old log probs)
                    clipped_ratio = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
                    
                    # Value loss
                    value_loss = F.mse_loss(values, returns)
                    
                    # Entropy bonus
                    probs = F.softmax(action_logits, dim=-1)
                    entropy = -(probs * log_probs).sum(dim=-1).mean()
                    
                    # Total loss
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
                # Backward with scaling
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                action_logits, values = self.ac_net(state_feat)
                
                log_probs = F.log_softmax(action_logits, dim=-1)
                action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                
                ratio = torch.exp(action_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
                
                value_loss = F.mse_loss(values, returns)
                
                probs = F.softmax(action_logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 1.0)
                self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
            
            n_batches += 1
            self.step += 1
        
        return {
            'policy_loss': total_policy_loss / n_batches,
            'value_loss': total_value_loss / n_batches,
            'entropy': total_entropy / n_batches,
        }
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            'ac_net_state_dict': self.ac_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
        }, path)


def main():
    parser = argparse.ArgumentParser(description="Ultra-fast PPO with pre-computed features")
    parser.add_argument('--dataset', type=str, required=True,
                       help='Pre-computed CLIP features .npz')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=1536,
                       help='Batch size (RTX 5090: 1536)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--clip-coef', type=float, default=0.2,
                       help='PPO clip coefficient')
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
    print("ULTRA-FAST OFFLINE PPO")
    print("="*60)
    
    # Load dataset
    dataset = CLIPFeatureDataset(Path(args.dataset))
    
    print(f"   Batch size: {args.batch_size}")
    print(f"   Batches/epoch: ~{len(dataset) // args.batch_size}")
    print("="*60)
    print("")
    
    # Create trainer
    trainer = UltraFastPPOTrainer(
        dataset=dataset,
        n_actions=4,
        lr=args.lr,
        batch_size=args.batch_size,
        clip_coef=args.clip_coef,
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
        
        total_loss = (train_metrics['policy_loss'] + 
                     train_metrics['value_loss'])
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Policy: {train_metrics['policy_loss']:.4f} | "
              f"Value: {train_metrics['value_loss']:.4f} | "
              f"Ent: {train_metrics['entropy']:.3f} | "
              f"Time: {epoch_time:.1f}s | "
              f"ETA: {eta/60:.1f}m")
        
        metrics_history.append({
            'epoch': epoch + 1,
            'epoch_time': epoch_time,
            **train_metrics,
        })
        
        # Save best model
        if total_loss < best_loss:
            best_loss = total_loss
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
    print(f"Best total loss: {best_loss:.4f}")
    print(f"Checkpoints: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
