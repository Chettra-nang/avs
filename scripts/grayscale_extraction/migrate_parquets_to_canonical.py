#!/usr/bin/env python3
"""Migrate transitions.parquet files to canonical grayscale storage.

This script performs a dry-run or in-place sample rewrite of selected parquet files.
It reads parquet rows, decodes the `grayscale_blob` (np.save bytes or raw buffer),
converts arrays to canonical shape (C,H,W) and dtype uint8 (0-255) and then
writes a sampled parquet to an output folder.

Usage:
  migrate_parquets_to_canonical.py --out-dir /path/to/out --dry-run file1.parquet file2.parquet ...

Outputs (dry-run): writes per-input sample parquet with _migrated_sample suffix under --out-dir
and a JSON summary report `migration_summary.json` in the out dir describing counts and sample stats.

This is intentionally conservative: it doesn't modify original files and only writes small sampled
parquets (max_samples_per_file) so you can inspect results before full migration.
"""

import argparse
import json
import os
import io
import sys
from pathlib import Path
from typing import Tuple, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


MAX_SAMPLES_PER_FILE = 10


def decode_blob(blob: bytes, shape=None, dtype=None):
    """Decode a grayscale_blob stored as numpy .npy bytes or raw buffer.

    Returns a numpy array.
    """
    # Try np.load from BytesIO (handles .npy)
    try:
        arr = np.load(io.BytesIO(blob))
        return arr
    except Exception:
        pass

    # Fallback: interpret as raw buffer
    if dtype is None or shape is None:
        raise ValueError("Raw buffer decoding requires shape and dtype metadata")
    dt = np.dtype(dtype)
    arr = np.frombuffer(blob, dtype=dt)
    arr = arr.reshape(shape)
    return arr


def to_c_hw(arr: np.ndarray, meta_shape=None) -> np.ndarray:
    """Normalize array to (C, H, W) uint8 in [0,255].

    Accepts arr in shapes: (H,W), (H,W,C), (C,H,W), (T,H,W) etc.
    Uses meta_shape to disambiguate if provided.
    """
    a = np.asarray(arr)
    # If meta_shape given, try to reshape accordingly
    if meta_shape is not None:
        try:
            if a.size == np.prod(meta_shape):
                a = a.reshape(meta_shape)
        except Exception:
            pass

    # Handle 4D cases like (1, C, H, W) by removing singleton dims first.
    if a.ndim == 4:
        # If there are singleton dimensions (e.g. leading 1), squeeze them.
        if any(d == 1 for d in a.shape):
            a = np.squeeze(a)
        else:
            # Unknown 4D layout (e.g. multiple timesteps). We can't safely
            # canonicalize without more rules, so raise for now.
            raise ValueError(f"Unexpected array ndim=4 with no singleton dims; shape={a.shape}")

    # Cases
    if a.ndim == 2:
        # single channel, H,W -> 1,H,W
        a = a[None, ...]
    elif a.ndim == 3:
        # heuristic: if first dim is small (<=4) treat as C,H,W else H,W,C
        if a.shape[0] <= 4:
            # assume C,H,W
            pass
        elif a.shape[2] <= 4:
            # H,W,C -> transpose
            a = a.transpose(2, 0, 1)
        else:
            # ambiguous: assume C,H,W
            pass
    else:
        raise ValueError(f"Unexpected array ndim={a.ndim}")

    # Convert to float in 0..1 then to uint8 0..255
    if np.issubdtype(a.dtype, np.floating):
        amin = a.min()
        amax = a.max()
        if amax <= 1.0 + 1e-6:
            scaled = (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            # assume in 0-255
            scaled = np.clip(a, 0, 255).astype(np.uint8)
    else:
        # integer types
        if a.dtype == np.uint8:
            scaled = a
        else:
            # cast and clip
            scaled = np.clip(a, 0, 255).astype(np.uint8)

    return scaled


def encode_blob_as_npy_bytes(arr: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.save(bio, arr, allow_pickle=False)
    return bio.getvalue()


def process_parquet(src_parquet: str, out_dir: str, dry_run=True, max_samples=MAX_SAMPLES_PER_FILE) -> dict:
    """Read src_parquet, sample up to max_samples rows, normalize grayscale_blob and write a sample parquet.

    Returns a summary dict with counts and sample paths.
    """
    src_path = Path(src_parquet)
    table = pq.read_table(str(src_path))
    df = table.to_pandas()
    nrows = len(df)

    indices = list(range(0, nrows, max(1, nrows // max_samples)))[:max_samples]
    migrated_rows = []
    problems = []
    sample_out_path = None

    for i in indices:
        row = df.iloc[i]
        blob = row.get('grayscale_blob')
        meta_shape = None
        meta_dtype = None
        if 'grayscale_shape' in row and row['grayscale_shape'] is not None:
            meta_shape = tuple(row['grayscale_shape'])
        if 'grayscale_dtype' in row and row['grayscale_dtype'] is not None:
            meta_dtype = row['grayscale_dtype']
        try:
            arr = decode_blob(blob, shape=meta_shape, dtype=meta_dtype)
            arr_c = to_c_hw(arr, meta_shape)
            # store metadata
            new_blob = encode_blob_as_npy_bytes(arr_c)
            new_row = row.copy()
            new_row['grayscale_blob'] = new_blob
            new_row['grayscale_shape'] = list(arr_c.shape)
            new_row['grayscale_dtype'] = str(arr_c.dtype)
            migrated_rows.append(new_row)
        except Exception as e:
            problems.append({'index': i, 'error': str(e)})

    # Write sampled parquet
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    sample_stem = src_path.stem + '_migrated_sample.parquet'
    sample_out_path = out_dir_p / sample_stem

    if len(migrated_rows) > 0:
        sample_df = pd.DataFrame(migrated_rows)
        # Ensure grayscale_blob is bytes
        sample_df['grayscale_blob'] = sample_df['grayscale_blob'].apply(lambda b: b if isinstance(b, (bytes, bytearray)) else bytes(b))
        pq.write_table(pa.Table.from_pandas(sample_df), str(sample_out_path))

    summary = {
        'src': str(src_path),
        'nrows': int(nrows),
        'sampled': len(indices),
        'migrated': len(migrated_rows),
        'problems': problems,
        'sample_out': str(sample_out_path) if sample_out_path.exists() else None,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('parquets', nargs='+')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--dry-run', action='store_true', default=False)
    parser.add_argument('--max-samples-per-file', type=int, default=MAX_SAMPLES_PER_FILE)
    args = parser.parse_args()

    summaries = []
    for pqf in args.parquets:
        print('Processing', pqf)
        try:
            s = process_parquet(pqf, args.out_dir, dry_run=args.dry_run, max_samples=args.max_samples_per_file)
            summaries.append(s)
            print(' -> done, migrated', s['migrated'], 'samples')
        except Exception as e:
            summaries.append({'src': pqf, 'error': str(e)})
            print(' -> error', e)

    summary_path = Path(args.out_dir) / 'migration_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summaries, f, indent=2)

    print('Wrote summary to', summary_path)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
