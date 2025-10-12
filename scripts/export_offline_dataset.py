#!/usr/bin/env python3
"""
Export collected parquet transitions to clean offline RL dataset.

Usage:
    python scripts/export_offline_dataset.py \
        --input data/ambulance_dataset_diagnose \
        --output data/offline_dataset \
        --format npz

Reads parquet files, decodes grayscale_blob, deduplicates rows,
reconstructs next_obs, and exports as (obs, action, reward, next_obs, done) tuples.
"""
from __future__ import annotations
import argparse
import json
import io
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


def decode_blob(blob, shape=None, dtype=None) -> np.ndarray:
    """Decode grayscale_blob to numpy array."""
    try:
        return np.load(io.BytesIO(blob), allow_pickle=False)
    except Exception:
        if shape is None or dtype is None:
            raise
        return np.frombuffer(blob, dtype=np.dtype(dtype)).reshape(tuple(shape))


def normalize_to_chw(arr: np.ndarray, meta_shape=None) -> np.ndarray:
    """Normalize array to canonical (C,H,W) uint8 format."""
    arr = np.squeeze(arr)
    
    if arr.ndim == 2:
        return arr.reshape((1, arr.shape[0], arr.shape[1])).astype('uint8')
    
    if arr.ndim == 3:
        d0, d1, d2 = arr.shape
        # If first dim is small (<=4) and last two differ -> (C,H,W)
        if d0 <= 4 and d1 != d2:
            return arr.astype('uint8')
        # If last dim is small -> (H,W,C) -> transpose
        if d2 <= 4:
            return arr.transpose(2, 0, 1).astype('uint8')
        # Default: assume (C,H,W)
        return arr.astype('uint8')
    
    if arr.ndim == 4 and arr.shape[0] == 1:
        return normalize_to_chw(arr.squeeze(0), meta_shape)
    
    # Fallback
    return arr.astype('uint8')


def process_parquet(parquet_path: Path) -> List[Dict]:
    """
    Process one parquet file and return list of clean transitions.
    Each transition: {obs, action, reward, next_obs, done, episode_id, step}
    """
    tbl = pq.read_table(parquet_path)
    df = tbl.to_pandas()
    
    # Deduplicate: keep only agent_id=0 rows (or first occurrence if no agent_id)
    if 'agent_id' in df.columns:
        df = df[df['agent_id'] == 0].copy()
    else:
        # If duplicate steps exist, keep first
        df = df.drop_duplicates(subset=['episode_id', 'step'], keep='first')
    
    df = df.sort_values(['episode_id', 'step']).reset_index(drop=True)
    
    transitions = []
    
    # Group by episode to reconstruct next_obs
    for ep_id, ep_df in df.groupby('episode_id', sort=False):
        ep_df = ep_df.sort_values('step').reset_index(drop=True)
        
        for i in range(len(ep_df) - 1):  # Skip last row (no next_obs)
            row = ep_df.iloc[i]
            next_row = ep_df.iloc[i + 1]
            
            # Decode current observation
            try:
                blob = row['grayscale_blob']
                shape = tuple(row['grayscale_shape']) if 'grayscale_shape' in row else None
                dtype = row['grayscale_dtype'] if 'grayscale_dtype' in row else None
                obs_arr = decode_blob(blob, shape, dtype)
                obs = normalize_to_chw(obs_arr, shape)
                
                # Decode next observation
                next_blob = next_row['grayscale_blob']
                next_shape = tuple(next_row['grayscale_shape']) if 'grayscale_shape' in next_row else None
                next_dtype = next_row['grayscale_dtype'] if 'grayscale_dtype' in next_row else None
                next_obs_arr = decode_blob(next_blob, next_shape, next_dtype)
                next_obs = normalize_to_chw(next_obs_arr, next_shape)
                
            except Exception as e:
                print(f"Decode error in {parquet_path.name} step {row['step']}: {e}")
                continue
            
            # Parse action (stored as array or list)
            action_raw = row['action']
            if isinstance(action_raw, (list, np.ndarray)):
                action = int(np.argmax(action_raw))  # Convert one-hot or multi-discrete to discrete
            else:
                action = int(action_raw)
            
            # Get reward and done
            reward = float(row['reward'])
            done = bool(row.get('done', False))
            
            transitions.append({
                'obs': obs,
                'action': action,
                'reward': reward,
                'next_obs': next_obs,
                'done': done,
                'episode_id': str(ep_id),
                'step': int(row['step']),
            })
    
    return transitions


def export_dataset(input_dir: Path, output_dir: Path, format: str = 'npz'):
    """
    Scan input directory for parquet files and export to output directory.
    
    Args:
        input_dir: Root directory containing batch_*/scenario/*_transitions.parquet
        output_dir: Where to write exported dataset
        format: 'npz' (numpy) or 'hdf5'
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all parquet files
    parquet_files = sorted(input_dir.rglob('*_transitions.parquet'))
    print(f"Found {len(parquet_files)} parquet files")
    
    all_transitions = []
    stats = {
        'total_files': len(parquet_files),
        'total_transitions': 0,
        'total_episodes': 0,
        'scenarios': {},
    }
    
    for pq_path in tqdm(parquet_files, desc="Processing parquets"):
        scenario = pq_path.parent.name
        transitions = process_parquet(pq_path)
        all_transitions.extend(transitions)
        
        ep_ids = set(t['episode_id'] for t in transitions)
        stats['scenarios'][scenario] = stats['scenarios'].get(scenario, 0) + len(ep_ids)
    
    stats['total_transitions'] = len(all_transitions)
    stats['total_episodes'] = len(set(t['episode_id'] for t in all_transitions))
    
    print(f"\nExported {stats['total_transitions']:,} transitions from {stats['total_episodes']} episodes")
    
    # Export based on format
    if format == 'npz':
        export_npz(all_transitions, output_dir, stats)
    elif format == 'hdf5':
        export_hdf5(all_transitions, output_dir, stats)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    # Save stats
    with open(output_dir / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Dataset exported to {output_dir}")


def export_npz(transitions: List[Dict], output_dir: Path, stats: Dict):
    """Export as single .npz file (good for small-medium datasets)."""
    arrays = {
        'obs': np.stack([t['obs'] for t in transitions]),
        'action': np.array([t['action'] for t in transitions], dtype=np.int64),
        'reward': np.array([t['reward'] for t in transitions], dtype=np.float32),
        'next_obs': np.stack([t['next_obs'] for t in transitions]),
        'done': np.array([t['done'] for t in transitions], dtype=bool),
    }
    
    # Check for shape consistency
    obs_shape = arrays['obs'][0].shape
    print(f"Observation shape: {obs_shape}")
    print(f"Actions: {arrays['action'].min()} to {arrays['action'].max()}")
    print(f"Rewards: mean={arrays['reward'].mean():.3f}, std={arrays['reward'].std():.3f}")
    print(f"Done rate: {arrays['done'].mean()*100:.1f}%")
    
    np.savez_compressed(
        output_dir / 'offline_dataset.npz',
        **arrays
    )
    print(f"Saved {output_dir / 'offline_dataset.npz'}")


def export_hdf5(transitions: List[Dict], output_dir: Path, stats: Dict):
    """Export as HDF5 (good for large datasets)."""
    import h5py
    
    with h5py.File(output_dir / 'offline_dataset.h5', 'w') as f:
        f.create_dataset('obs', data=np.stack([t['obs'] for t in transitions]), compression='gzip')
        f.create_dataset('action', data=np.array([t['action'] for t in transitions], dtype=np.int64))
        f.create_dataset('reward', data=np.array([t['reward'] for t in transitions], dtype=np.float32))
        f.create_dataset('next_obs', data=np.stack([t['next_obs'] for t in transitions]), compression='gzip')
        f.create_dataset('done', data=np.array([t['done'] for t in transitions], dtype=bool))
    
    print(f"Saved {output_dir / 'offline_dataset.h5'}")


def main():
    parser = argparse.ArgumentParser(description="Export offline RL dataset from parquet files")
    parser.add_argument('--input', type=str, required=True, help='Input directory with parquet files')
    parser.add_argument('--output', type=str, required=True, help='Output directory for dataset')
    parser.add_argument('--format', choices=['npz', 'hdf5'], default='npz', help='Export format')
    args = parser.parse_args()
    
    export_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        format=args.format
    )


if __name__ == '__main__':
    main()
