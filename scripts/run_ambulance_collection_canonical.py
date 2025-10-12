#!/usr/bin/env python3
"""Run ambulance data collection and enforce canonical grayscale format.

This runner uses `AmbulanceDataCollector` to collect episodes, then post-processes
collected observations to ensure `grayscale_blob` is stored in canonical form:
- shape: (C, H, W)
- dtype: float32
- values normalized to [0.0, 1.0]

Usage examples:
  # Dry-run (no heavy collection) to validate arguments
  python scripts/run_ambulance_collection_canonical.py --dry-run --output-dir AVs/data/out

  # Real run
  python scripts/run_ambulance_collection_canonical.py --output-dir AVs/data/out --episodes-per-scenario 50

The script is conservative and non-destructive: it writes to the provided output dir
(using `AmbulanceDataCollector.store_ambulance_data`).
"""

from __future__ import annotations
import argparse
import io
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def decode_blob(blob: bytes, meta_shape=None, meta_dtype=None) -> np.ndarray:
    """Decode blob stored as np.save bytes or raw buffer. Return numpy array."""
    try:
        return np.load(io.BytesIO(blob))
    except Exception:
        if meta_shape is None or meta_dtype is None:
            raise
        dt = np.dtype(meta_dtype)
        arr = np.frombuffer(blob, dtype=dt).reshape(tuple(meta_shape))
        return arr


def to_c_hw_float(arr: np.ndarray, meta_shape=None) -> np.ndarray:
    """Normalize array to (C,H,W) float32 with values in [0,1]."""
    a = np.asarray(arr)
    # Squeeze leading singleton dims (handle (1,C,H,W))
    if a.ndim == 4 and any(d == 1 for d in a.shape):
        a = np.squeeze(a)

    if a.ndim == 2:
        a = a[None, ...]
    elif a.ndim == 3:
        # if last dim small (<=4) assume H,W,C
        if a.shape[2] <= 4 and a.shape[0] > 4:
            a = a.transpose(2, 0, 1)
        # else assume C,H,W
    else:
        raise ValueError(f"Unexpected array ndim={a.ndim}; shape={a.shape}")

    # Convert to float32 0..1
    if np.issubdtype(a.dtype, np.floating):
        if a.max() > 1.01:
            a = np.clip(a, 0.0, 255.0) / 255.0
        else:
            a = np.clip(a, 0.0, 1.0)
    else:
        # integer types
        a = np.clip(a, 0, 255).astype(np.float32) / 255.0

    return a.astype(np.float32)


def encode_blob_npy_bytes(arr: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.save(bio, arr, allow_pickle=False)
    return bio.getvalue()


def canonicalize_collection_results(collection_results, force_float32=True):
    """Mutate collection_results in-place: canonicalize grayscale blobs.

    collection_results: dict mapping scenario_name -> CollectionResult
    Each CollectionResult contains EpisodeData objects; episodes have "observations"
    where each step is a list of observation dicts.
    """
    num_fixed = 0
    num_skipped = 0

    for scenario, result in collection_results.items():
        for episode in result.episodes:
            # episode.observations is a list of steps; each step is a list of agent obs
            for step_idx, step_obs in enumerate(episode.observations):
                # step_obs is list of obs dicts
                for obs_idx, obs in enumerate(step_obs):
                    if not isinstance(obs, dict):
                        continue
                    if 'grayscale_blob' not in obs or obs.get('grayscale_blob') is None:
                        num_skipped += 1
                        continue
                    try:
                        blob = obs['grayscale_blob']
                        meta_shape = obs.get('grayscale_shape')
                        meta_dtype = obs.get('grayscale_dtype')
                        arr = decode_blob(blob, meta_shape, meta_dtype)
                        arr_c = to_c_hw_float(arr, meta_shape)
                        if force_float32:
                            new_blob = encode_blob_npy_bytes(arr_c.astype(np.float32))
                            obs['grayscale_blob'] = new_blob
                            obs['grayscale_shape'] = list(arr_c.shape)
                            obs['grayscale_dtype'] = 'float32'
                            obs['grayscale_canonicalized'] = True
                            num_fixed += 1
                        else:
                            # alternative: store uint8
                            arr_u8 = (np.clip(arr_c, 0, 1) * 255.0).astype(np.uint8)
                            new_blob = encode_blob_npy_bytes(arr_u8)
                            obs['grayscale_blob'] = new_blob
                            obs['grayscale_shape'] = list(arr_u8.shape)
                            obs['grayscale_dtype'] = 'uint8'
                            obs['grayscale_canonicalized'] = True
                            num_fixed += 1
                    except Exception as e:
                        obs['grayscale_canonicalize_error'] = str(e)
                        num_skipped += 1

    return {'fixed': num_fixed, 'skipped': num_skipped}


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Run ambulance collection and canonicalize grayscale storage")
    parser.add_argument('--output-dir', required=True, help='Directory to store collected dataset')
    parser.add_argument('--scenarios', help='Comma separated list of scenario names (default: all)')
    parser.add_argument('--episodes-per-scenario', type=int, default=10)
    parser.add_argument('--max-steps-per-episode', type=int, default=100)
    parser.add_argument('--base-seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--dry-run', action='store_true', help='Do not perform heavy collection; just validate args')
    parser.add_argument('--force-float32', action='store_true', default=True, help='Store canonical grayscale as float32 in 0..1')
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure repository root is on sys.path so local packages can be imported when
    # this script is executed directly from scripts/ or elsewhere.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Import the AmbulanceDataCollector lazily (may be heavy)
    try:
        from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector
    except Exception:
        # Provide a clearer error message if import still fails
        print('Failed to import collecting_ambulance_data. Make sure you run this from the repository or have the project on PYTHONPATH.')
        raise

    collector = AmbulanceDataCollector()

    scenarios = None
    if args.scenarios:
        scenarios = [s.strip() for s in args.scenarios.split(',') if s.strip()]

    print('Planned collection:')
    print('  output_dir:', out_dir)
    print('  scenarios:', 'ALL' if scenarios is None else scenarios)
    print('  episodes_per_scenario:', args.episodes_per_scenario)
    print('  max_steps_per_episode:', args.max_steps_per_episode)
    print('  batch_size:', args.batch_size)
    print('  base_seed:', args.base_seed)
    print('  dry_run:', args.dry_run)

    if args.dry_run:
        print('\nDry-run mode: no collection executed. Exiting.')
        return 0

    # Run collection
    print('\nStarting collection...')
    collection_results = collector.collect_ambulance_data(
        scenarios=scenarios,
        episodes_per_scenario=args.episodes_per_scenario,
        max_steps_per_episode=args.max_steps_per_episode,
        base_seed=args.base_seed,
        batch_size=args.batch_size
    )

    print('Collection completed. Canonicalizing grayscale blobs...')
    summary = canonicalize_collection_results(collection_results, force_float32=args.force_float32)
    print('Canonicalization summary:', summary)

    # Store data
    print('Storing canonicalized data to output dir...')
    storage_info = collector.store_ambulance_data(collection_results, out_dir)
    print('Storage completed:')
    print(json.dumps(storage_info, indent=2))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
