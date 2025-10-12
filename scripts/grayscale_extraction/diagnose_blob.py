#!/usr/bin/env python3
"""Diagnostic: decode a grayscale_blob from a parquet and save per-channel PNGs

Usage: python3 scripts/grayscale_extraction/diagnose_blob.py <parquet_path> <episode_id> <step> <agent_id>
"""
import sys
from pathlib import Path
import io
import numpy as np
import pandas as pd
from PIL import Image


def decode_blob(blob, shape, dtype):
    buf = io.BytesIO(blob)
    try:
        arr = np.load(buf, allow_pickle=False)
        arr = np.asarray(arr)
    except Exception:
        arr = np.frombuffer(blob, dtype=dtype)
    return arr.reshape(shape)


def save_channel_images(arr, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Normalize each channel to 0-255 and save
    arr = np.asarray(arr)
    if arr.ndim == 3:
        C, H, W = arr.shape
        for c in range(C):
            img = arr[c]
            # Normalize
            imin, imax = img.min(), img.max()
            if imax > imin:
                norm = (img - imin) / (imax - imin)
            else:
                norm = img - imin
            img8 = (norm * 255).astype('uint8')
            Image.fromarray(img8).save(out_dir / f"{prefix}_ch{c}.png")
    elif arr.ndim == 2:
        imin, imax = arr.min(), arr.max()
        if imax > imin:
            norm = (arr - imin) / (imax - imin)
        else:
            norm = arr - imin
        img8 = (norm * 255).astype('uint8')
        Image.fromarray(img8).save(out_dir / f"{prefix}_gray.png")
    else:
        raise ValueError('Unsupported array shape')


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: diagnose_blob.py <parquet_path> <episode_id> <step> <agent_id>')
        sys.exit(2)
    parquet = Path(sys.argv[1])
    episode_id = sys.argv[2]
    step = int(sys.argv[3])
    agent_id = int(sys.argv[4])

    df = pd.read_parquet(parquet)
    row = df[(df['episode_id'] == episode_id) & (df['step'] == step) & (df['agent_id'] == agent_id)]
    if row.empty:
        print('No matching row found')
        sys.exit(1)
    row = row.iloc[0]
    blob = row['grayscale_blob']
    shape = tuple(row['grayscale_shape'])
    dtype = row['grayscale_dtype']

    arr = decode_blob(blob, shape, dtype)
    print('Decoded array shape:', arr.shape)
    print('dtype:', arr.dtype)
    print('min, max, mean:', float(arr.min()), float(arr.max()), float(arr.mean()))

    out_dir = parquet.parent / 'diagnose_output'
    save_channel_images(arr, out_dir, f"{episode_id}_s{step}_a{agent_id}")
    print('Saved channel images to', out_dir)
