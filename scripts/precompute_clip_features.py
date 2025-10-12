#!/usr/bin/env python3
"""
Pre-compute CLIP features for offline dataset - MASSIVE SPEEDUP!

This script:
1. Loads the offline dataset (41K transitions)
2. Encodes all images with CLIP ViT-B/32 in batches
3. Saves pre-computed features to disk
4. Run once, use for all training experiments

Usage:
    python3 scripts/precompute_clip_features.py \
        --dataset data/offline_dataset/offline_dataset.npz \
        --output data/offline_dataset/clip_features.npz \
        --batch-size 256 \
        --device cuda

Expected time: 5-10 minutes for 41K images
After this, training becomes 10-50x faster!
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from offline_rl.rl_langvision.clip_embedder import CLIPImageEncoder


class ImageDataset(Dataset):
    """Simple dataset for images only."""
    
    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.obs = torch.from_numpy(data['obs']).float() / 255.0
        print(f"Loaded {len(self)} images from {npz_path}")
        print(f"  Shape: {self.obs.shape}")
    
    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        obs = self.obs[idx]
        # Convert to RGB (take last frame if temporal stack)
        if obs.shape[0] == 4:
            obs_rgb = obs[-1:].repeat(3, 1, 1)
        else:
            obs_rgb = obs.repeat(3, 1, 1)
        return obs_rgb


def precompute_clip_features(
    dataset_path: Path,
    output_path: Path,
    batch_size: int = 256,
    device: str = 'cuda',
    clip_model: str = 'ViT-B-32',
):
    """Pre-compute CLIP features for entire dataset."""
    
    print("="*60)
    print("CLIP Feature Pre-computation")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_path}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"CLIP model: {clip_model}")
    print("")
    
    # Load dataset
    dataset = ImageDataset(dataset_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Initialize CLIP encoder
    print("Initializing CLIP encoder...")
    clip_encoder = CLIPImageEncoder(clip_model, device=device)
    print(f"✅ CLIP encoder loaded on {device}")
    print("")
    
    # Pre-compute features
    print("Encoding images in batches...")
    all_features = []
    start_time = time.time()
    
    with torch.no_grad():
        for batch_obs in tqdm(dataloader, desc="Encoding batches"):
            batch_obs = batch_obs.to(device)
            
            # Batch encode with CLIP
            # Convert to numpy for clip_encoder (expected format)
            B = batch_obs.shape[0]
            batch_features = []
            
            for i in range(B):
                img = (batch_obs[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                feat = clip_encoder.encode_np_rgb(img)
                batch_features.append(feat)
            
            batch_features = np.stack(batch_features, axis=0)
            all_features.append(batch_features)
    
    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)
    elapsed = time.time() - start_time
    
    print("")
    print(f"✅ Encoded {len(all_features)} images in {elapsed:.1f}s ({len(all_features)/elapsed:.1f} img/s)")
    print(f"   Feature shape: {all_features.shape}")
    print(f"   Feature dtype: {all_features.dtype}")
    print("")
    
    # Load original dataset to include other data
    print("Saving pre-computed features...")
    original_data = np.load(dataset_path)
    
    # Save with original data + CLIP features
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        clip_features=all_features,
        action=original_data['action'],
        reward=original_data['reward'],
        next_clip_features=all_features,  # For DQN/PPO if needed
        done=original_data['done']
    )
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved to {output_path}")
    print(f"   File size: {file_size_mb:.1f} MB")
    print("")
    print("="*60)
    print("✅ PRE-COMPUTATION COMPLETE!")
    print("="*60)
    print("")
    print("Now train with pre-computed features:")
    print(f"  bash run_offline_training_ULTRAFAST.sh")
    print("")
    print("Expected training times (with pre-computed features):")
    print("  BC:  2-3 minutes   (50 epochs)")
    print("  DQN: 8-10 minutes  (100 epochs)")
    print("  PPO: 10-12 minutes (100 epochs)")
    print("  ALL: 20-25 minutes total")
    print("")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute CLIP features")
    parser.add_argument('--dataset', type=str, required=True, help='Input .npz dataset')
    parser.add_argument('--output', type=str, required=True, help='Output .npz with features')
    parser.add_argument('--batch-size', type=int, default=256, help='Encoding batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--clip-model', type=str, default='ViT-B-32', help='CLIP model')
    args = parser.parse_args()
    
    precompute_clip_features(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        batch_size=args.batch_size,
        device=args.device,
        clip_model=args.clip_model,
    )


if __name__ == '__main__':
    main()
