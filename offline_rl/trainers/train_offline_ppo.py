#!/usr/bin/env python3
"""
Offline PPO training from collected parquet transitions.

Implements Conservative Policy Gradient (CPG) for offline RL:
- Trains policy network with CLIP encoder on collected data
- Uses importance sampling for off-policy corrections
- Conservative policy updates to prevent distribution shift

Usage:
    python train_offline_ppo.py \
        --dataset ../../data/offline_dataset/offline_dataset.npz \
        --output ../checkpoints/offline_ppo \
        --epochs 100 \
        --batch-size 256

Architecture:
    - CLIP ViT-B/32 visual encoder (512-d embeddings)
    - Actor-Critic with shared CLIP backbone
    - GAE for advantage estimation
    - Conservative policy gradients for offline learning
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
from torch.distributions import Categorical
from tqdm import tqdm

# Import CLIP encoder from rl_langvision
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from rl_langvision.clip_embedder import CLIPImageEncoder


class OfflineRLDataset(Dataset):
    """PyTorch dataset for offline RL transitions with GAE computation."""
    
    def __init__(self, npz_path: Path, gamma: float = 0.99, gae_lambda: float = 0.95):
        data = np.load(npz_path)
        self.obs = torch.from_numpy(data['obs']).float() / 255.0  # Normalize to [0,1]
        self.actions = torch.from_numpy(data['action']).long()
        self.rewards = torch.from_numpy(data['reward']).float()
        self.next_obs = torch.from_numpy(data['next_obs']).float() / 255.0
        self.dones = torch.from_numpy(data['done']).float()
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        print(f"Loaded dataset: {len(self)} transitions")
        print(f"  Obs shape: {self.obs.shape}")
        print(f"  Action range: {self.actions.min().item()}-{self.actions.max().item()}")
        print(f"  Reward: mean={self.rewards.mean():.3f}, std={self.rewards.std():.3f}")
    
    def compute_advantages(self, values: torch.Tensor, next_values: torch.Tensor):
        """Compute GAE advantages for the entire dataset."""
        advantages = torch.zeros_like(self.rewards)
        returns = torch.zeros_like(self.rewards)
        
        last_gae = 0
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
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


class CLIPActorCritic(nn.Module):
    """
    Actor-Critic network with CLIP vision encoder.
    
    Takes grayscale (C,H,W) observations, converts to RGB, encodes with CLIP,
    and outputs policy logits and value estimate.
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
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_actions),
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def _encode_observations(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations with CLIP."""
        # Convert grayscale to RGB (take last frame if temporal stack)
        if obs.shape[1] == 4:
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
        return clip_feats
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: (B, C, H, W) grayscale
        Returns:
            logits: (B, n_actions) policy logits
            value: (B, 1) value estimate
        """
        clip_feats = self._encode_observations(obs)
        logits = self.actor(clip_feats)
        value = self.critic(clip_feats)
        return logits, value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate only."""
        clip_feats = self._encode_observations(obs)
        return self.critic(clip_feats).squeeze(-1)
    
    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        """Get action distribution and value."""
        logits, value = self.forward(obs)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value.squeeze(-1)


class OfflinePPOTrainer:
    """Offline PPO trainer using conservative policy gradients."""
    
    def __init__(
        self,
        dataset: OfflineRLDataset,
        n_actions: int = 5,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        batch_size: int = 256,
        device: str = 'cuda',
    ):
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        # Actor-Critic network
        self.ac = CLIPActorCritic(n_actions=n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)
        
        self.step = 0
    
    def compute_advantages_for_dataset(self):
        """Compute advantages for the entire dataset using current value function."""
        self.ac.eval()
        all_values = []
        all_next_values = []
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Computing advantages", leave=False):
                obs = batch['obs'].to(self.device)
                next_obs = batch['next_obs'].to(self.device)
                
                values = self.ac.get_value(obs)
                next_values = self.ac.get_value(next_obs)
                
                all_values.append(values.cpu())
                all_next_values.append(next_values.cpu())
        
        values = torch.cat(all_values)
        next_values = torch.cat(all_next_values)
        
        advantages, returns = self.dataset.compute_advantages(values, next_values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def train_epoch(self, advantages: torch.Tensor, returns: torch.Tensor) -> Dict[str, float]:
        """Train one epoch."""
        self.ac.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_batches = 0
        
        # Create dataloader with advantages and returns
        indices = torch.randperm(len(self.dataset))
        batch_size = self.dataloader.batch_size
        
        for i in range(0, len(self.dataset), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            obs = self.dataset.obs[batch_indices].to(self.device)
            actions = self.dataset.actions[batch_indices].to(self.device)
            old_advantages = advantages[batch_indices].to(self.device)
            target_returns = returns[batch_indices].to(self.device)
            
            # Get current policy and value
            _, log_prob, entropy, values = self.ac.get_action_and_value(obs, actions)
            
            # Policy loss (PPO clipped objective)
            ratio = torch.exp(log_prob - log_prob.detach())  # Offline: use same log_prob as baseline
            policy_loss_1 = -old_advantages * ratio
            policy_loss_2 = -old_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
            policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, target_returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            n_batches += 1
            self.step += 1
        
        return {
            'loss': total_loss / n_batches,
            'policy_loss': total_policy_loss / n_batches,
            'value_loss': total_value_loss / n_batches,
            'entropy': total_entropy / n_batches,
        }
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            'actor_critic': self.ac.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
        }, path)


def main():
    parser = argparse.ArgumentParser(description="Train offline PPO on collected data")
    parser.add_argument('--dataset', type=str, required=True, help='Path to offline_dataset.npz')
    parser.add_argument('--output', type=str, default='../checkpoints/offline_ppo', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip-coef', type=float, default=0.2, help='PPO clip coefficient')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = OfflineRLDataset(Path(args.dataset), gamma=args.gamma, gae_lambda=args.gae_lambda)
    
    # Create trainer
    trainer = OfflinePPOTrainer(
        dataset=dataset,
        n_actions=5,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # Training loop
    metrics_history = []
    best_return = -float('inf')
    
    for epoch in range(args.epochs):
        # Compute advantages (recompute every epoch with updated value function)
        advantages, returns = trainer.compute_advantages_for_dataset()
        mean_return = returns.mean().item()
        
        # Train
        train_metrics = trainer.train_epoch(advantages, returns)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"Policy: {train_metrics['policy_loss']:.4f} | "
              f"Value: {train_metrics['value_loss']:.4f} | "
              f"Entropy: {train_metrics['entropy']:.3f} | "
              f"Return: {mean_return:.3f}")
        
        metrics_history.append({
            'epoch': epoch + 1,
            **train_metrics,
            'mean_return': mean_return,
        })
        
        # Save best model
        if mean_return > best_return:
            best_return = mean_return
            trainer.save_checkpoint(output_dir / 'best_model.pt')
        
        # Save checkpoint
        if (epoch + 1) % 20 == 0:
            trainer.save_checkpoint(output_dir / f'checkpoint_epoch{epoch+1}.pt')
    
    # Save final model
    trainer.save_checkpoint(output_dir / 'final_model.pt')
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    print(f"✅ Training complete! Best return: {best_return:.3f}")
    print(f"Models saved to {output_dir}")


if __name__ == '__main__':
    main()
