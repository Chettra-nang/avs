#!/usr/bin/env python3
"""
Offline DQN training from collected parquet transitions.

Usage:
    python tools/train_offline_dqn.py \
        --dataset ../../../AVs/data/offline_dataset/offline_dataset.npz \
        --output checkpoints/offline_dqn \
        --epochs 100 \
        --batch-size 256

Architecture:
    - CLIP ViT-B/32 visual encoder (512-d embeddings)
    - Optional text encoder (384-d)
    - MLP Q-network for 5 discrete actions
    - Uses grayscale temporal stack (C,H,W) observations from collected data
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import CLIP encoder from rl_langvision
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from rl_langvision.clip_embedder import CLIPImageEncoder


class OfflineRLDataset(Dataset):
    """PyTorch dataset for offline RL transitions."""
    
    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.obs = torch.from_numpy(data['obs']).float() / 255.0  # Normalize to [0,1]
        self.actions = torch.from_numpy(data['action']).long()
        self.rewards = torch.from_numpy(data['reward']).float()
        self.next_obs = torch.from_numpy(data['next_obs']).float() / 255.0
        self.dones = torch.from_numpy(data['done']).float()
        
        print(f"Loaded dataset: {len(self)} transitions")
        print(f"  Obs shape: {self.obs.shape}")
        print(f"  Action range: {self.actions.min().item()}-{self.actions.max().item()}")
        print(f"  Reward: mean={self.rewards.mean():.3f}, std={self.rewards.std():.3f}")
    
    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        return {
            'obs': self.obs[idx],
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'next_obs': self.next_obs[idx],
            'done': self.dones[idx],
        }


class CLIPQNetwork(nn.Module):
    """
    Q-network with CLIP vision encoder.
    
    Takes grayscale (C,H,W) observations, converts to RGB, encodes with CLIP,
    and outputs Q-values for discrete actions.
    """
    
    def __init__(self, n_actions: int = 5, hidden_dim: int = 256, freeze_clip: bool = True):
        super().__init__()
        self.n_actions = n_actions
        
        # CLIP encoder (512-d embeddings)
        self.clip_encoder = CLIPImageEncoder(
            model_name="ViT-B-32",
            pretrained="openai",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        if freeze_clip:
            # Freeze CLIP weights for faster training
            for param in self.clip_encoder.model.parameters():
                param.requires_grad = False
        
        # Q-network head
        self.q_net = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, C, H, W) grayscale temporal stack
        Returns:
            q_values: (B, n_actions)
        """
        # Convert grayscale to RGB (repeat channels)
        if obs.shape[1] == 4:  # Temporal stack -> take last frame
            obs_rgb = obs[:, -1:].repeat(1, 3, 1, 1)  # (B, 1, H, W) -> (B, 3, H, W)
        else:
            obs_rgb = obs.repeat(1, 3, 1, 1)
        
        # Encode with CLIP
        B = obs_rgb.shape[0]
        clip_feats = []
        for i in range(B):
            # CLIPImageEncoder expects numpy HWC uint8
            img = (obs_rgb[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            feat = self.clip_encoder.encode_np_rgb(img)  # (512,)
            clip_feats.append(torch.from_numpy(feat))
        
        clip_feats = torch.stack(clip_feats).to(obs.device)  # (B, 512)
        
        # Compute Q-values
        q_values = self.q_net(clip_feats)
        return q_values


class OfflineDQNTrainer:
    """Offline DQN trainer using batch-constrained Q-learning."""
    
    def __init__(
        self,
        dataset: OfflineRLDataset,
        n_actions: int = 5,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        device: str = 'cuda',
    ):
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        # Q-networks
        self.q_net = CLIPQNetwork(n_actions=n_actions).to(device)
        self.q_target = CLIPQNetwork(n_actions=n_actions).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        
        self.step = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch."""
        self.q_net.train()
        total_loss = 0.0
        total_q_mean = 0.0
        n_batches = 0
        
        for batch in tqdm(self.dataloader, desc="Training", leave=False):
            obs = batch['obs'].to(self.device)
            actions = batch['action'].to(self.device)
            rewards = batch['reward'].to(self.device)
            next_obs = batch['next_obs'].to(self.device)
            dones = batch['done'].to(self.device)
            
            # Compute current Q-values
            q_values = self.q_net(obs)
            q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute target Q-values
            with torch.no_grad():
                next_q_values = self.q_target(next_obs)
                next_q_max = next_q_values.max(dim=1)[0]
                q_target = rewards + self.gamma * next_q_max * (1 - dones)
            
            # Compute loss
            loss = F.mse_loss(q_taken, q_target)
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
            self.optimizer.step()
            
            # Soft update target network
            for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            total_loss += loss.item()
            total_q_mean += q_taken.mean().item()
            n_batches += 1
            self.step += 1
        
        return {
            'loss': total_loss / n_batches,
            'q_mean': total_q_mean / n_batches,
        }
    
    def evaluate(self, n_samples: int = 1000) -> Dict[str, float]:
        """Evaluate policy on random subset of dataset."""
        self.q_net.eval()
        
        indices = np.random.choice(len(self.dataset), min(n_samples, len(self.dataset)), replace=False)
        
        total_return = 0.0
        total_q = 0.0
        n_episodes = 0
        
        with torch.no_grad():
            for idx in indices:
                sample = self.dataset[idx]
                obs = sample['obs'].unsqueeze(0).to(self.device)
                reward = sample['reward'].item()
                
                q_values = self.q_net(obs)
                total_q += q_values.max().item()
                total_return += reward
                n_episodes += 1
        
        return {
            'mean_return': total_return / n_episodes,
            'mean_q': total_q / n_episodes,
        }
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            'q_net': self.q_net.state_dict(),
            'q_target': self.q_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
        }, path)


def main():
    parser = argparse.ArgumentParser(description="Train offline DQN on collected data")
    parser.add_argument('--dataset', type=str, required=True, help='Path to offline_dataset.npz')
    parser.add_argument('--output', type=str, default='checkpoints/offline_dqn', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = OfflineRLDataset(Path(args.dataset))
    
    # Create trainer
    trainer = OfflineDQNTrainer(
        dataset=dataset,
        n_actions=5,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # Training loop
    metrics_history = []
    best_q = -float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = trainer.train_epoch()
        
        # Evaluate
        if (epoch + 1) % 10 == 0:
            eval_metrics = trainer.evaluate()
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Loss: {train_metrics['loss']:.4f} | "
                  f"Q: {train_metrics['q_mean']:.3f} | "
                  f"Eval Q: {eval_metrics['mean_q']:.3f} | "
                  f"Eval Return: {eval_metrics['mean_return']:.3f}")
            
            metrics_history.append({
                'epoch': epoch + 1,
                **train_metrics,
                **eval_metrics,
            })
            
            # Save best model
            if eval_metrics['mean_q'] > best_q:
                best_q = eval_metrics['mean_q']
                trainer.save_checkpoint(output_dir / 'best_model.pt')
        else:
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Loss: {train_metrics['loss']:.4f} | "
                  f"Q: {train_metrics['q_mean']:.3f}")
        
        # Save checkpoint
        if (epoch + 1) % 20 == 0:
            trainer.save_checkpoint(output_dir / f'checkpoint_epoch{epoch+1}.pt')
    
    # Save final model
    trainer.save_checkpoint(output_dir / 'final_model.pt')
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    print(f"✅ Training complete! Models saved to {output_dir}")


if __name__ == '__main__':
    main()
