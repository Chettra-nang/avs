#!/usr/bin/env python3
"""
Extract grayscale frames and build a CSV index for CLIP fine-tuning.

Reads transitions Parquet files produced by DatasetStorageManager, decodes
`grayscale_blob` fields using the project's BinaryArrayEncoder, writes PNG
images to an output directory, and creates a `frames.csv` with one row per
image: image_path, episode_id, step, agent_id, action, reward, scenario.

Example:
  python3 scripts/extract_frames_for_clip.py --input data/ambulance_dataset_diagnose \
      --output data/clip_frames --max-per-scenario 500
"""

import argparse
import sys
import os
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
import io
import numpy as np
from tqdm import tqdm
from PIL import Image

# Ensure project root is on sys.path so local packages (highway_datacollection) import correctly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from highway_datacollection.storage.encoders import BinaryArrayEncoder


def decode_grayscale_from_record(record: dict):
    """Decode grayscale array from a transitions record dict.

    Expects keys: 'grayscale_blob', 'grayscale_shape', 'grayscale_dtype'
    Returns a numpy array with shape (C,H,W) or None if not present.
    """
    blob = record.get('grayscale_blob', b'')
    shape = record.get('grayscale_shape', [])
    dtype = record.get('grayscale_dtype', 'uint8')
    # Guard against various blob representations returned by pyarrow/pandas:
    # - None
    # - empty bytes / bytearray
    # - memoryview
    # - numpy.ndarray of dtype uint8
    if blob is None:
        return None
    # Convert memoryview to bytes
    if isinstance(blob, memoryview):
        try:
            blob = blob.tobytes()
        except Exception:
            return None
    # Empty bytes/bytearray -> missing
    if isinstance(blob, (bytes, bytearray)) and len(blob) == 0:
        return None
    # If pyarrow returned a numpy array of bytes (uint8), convert to bytes
    if isinstance(blob, np.ndarray):
        try:
            if blob.size == 0:
                return None
            # prefer converting uint8 arrays to raw bytes
            blob = blob.tobytes()
        except Exception:
            return None

    encoder = BinaryArrayEncoder()
    # Try the canonical decoder first
    try:
        arr = encoder.decode(blob, tuple(shape), dtype)
    except Exception:
        # Fallback: try numpy load and reshape according to provided shape
        try:
            import io as _io
            buf = _io.BytesIO(blob)
            arr = np.load(buf, allow_pickle=False)
            # If arr is 1D but shape is available, reshape
            if arr.ndim == 1 and shape:
                try:
                    arr = arr.reshape(tuple(shape))
                except Exception:
                    pass
        except Exception:
            return None

    # Normalize dtype strings like '|u1' to numpy dtype
    try:
        if isinstance(dtype, str) and ('|' in dtype or '<' in dtype or '>' in dtype):
            # treat as uint8 if unsigned byte
            if 'u1' in dtype or 'uint8' in dtype:
                arr = arr.astype(np.uint8)
            else:
                # try to cast to a safe type
                arr = arr.astype(np.float32)
    except Exception:
        pass

    # If shape suggests axes swapped (collector sometimes uses (C,W,H)), try to detect
    try:
        if isinstance(shape, (list, tuple)) and len(shape) == 3:
            c, d1, d2 = shape
            # If middle and last dims differ and look like width/height swapped
            if d1 != d2 and arr.ndim == 3 and arr.shape[1] != arr.shape[2]:
                # Try transposing from (C,W,H) -> (C,H,W)
                try:
                    arr = np.transpose(arr, (0, 2, 1))
                except Exception:
                    pass

    except Exception:
        pass

    return arr


def process_parquet_file(parquet_path: Path, out_dir: Path, csv_rows: list, max_images: int = 0):
    table = pq.read_table(str(parquet_path))

    # Convert to plain Python dict of columns (avoid pandas conversion which can choke on binary blobs)
    try:
        data_dict = table.to_pydict()
    except Exception:
        # Fallback to pandas if pyarrow to_pydict fails
        df = table.to_pandas()
        data_dict = df.to_dict(orient='list')

    # Number of rows
    n_rows = 0
    for v in data_dict.values():
        n_rows = max(n_rows, len(v))

    saved = 0
    for idx in range(n_rows):
        # Build record for this row
        rec = {}
        for k, col in data_dict.items():
            try:
                rec[k] = col[idx]
            except Exception:
                rec[k] = None

        # Try to decode grayscale
        gray = decode_grayscale_from_record(rec)
        if gray is None:
            continue

        # Normalize to HxW (take last channel if C>1). Handle singleton axes.
        try:
            arr = np.array(gray)
            # Remove leading singleton axes
            if arr.ndim == 3 and 1 in arr.shape:
                arr = arr.squeeze()

            if arr.ndim == 3:
                # If shape is (C,H,W) take channel 0 or last channel
                c = arr.shape[0]
                if c in (1, 3):
                    img = arr[-1]
                else:
                    # fallback: collapse to 2D via mean
                    img = arr.mean(axis=0).astype(arr.dtype)
            elif arr.ndim == 2:
                img = arr
            else:
                # Flattened or unexpected shape -> try to reshape using width 128 guess
                if arr.size == 128:
                    img = arr.reshape((128, 1))
                else:
                    img = arr.squeeze()
        except Exception:
            img = np.array(gray)

        # Ensure numeric and finite
        try:
            if np.isnan(img).any():
                # normalize replacing nans
                img = np.nan_to_num(img)
        except Exception:
            pass

        # Ensure uint8
        if img.dtype != np.uint8:
            try:
                img = (255 * (img.astype(np.float32) - float(img.min())) / max(1e-6, float(img.max() - img.min()))).astype(np.uint8)
            except Exception:
                img = img.astype(np.uint8, copy=False)

        # Resize to 224x224 and convert to 3-channel for CLIP
        try:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((224, 224)).convert('RGB')
        except Exception:
            # fallback: coerce array shape then convert
            a = np.array(img)
            if a.ndim == 1:
                # attempt square-ish reshape
                s = int(np.ceil(np.sqrt(a.size)))
                a = np.pad(a, (0, s*s - a.size), mode='edge').reshape((s, s))
            pil_img = Image.fromarray(a.astype('uint8')).resize((224, 224)).convert('RGB')

        # Build filename
        episode_id = rec.get('episode_id', 'unknown_ep')
        step = rec.get('step', idx)
        agent_id = rec.get('agent_id', 0)
        scenario = rec.get('scenario', 'unknown')

        img_dir = out_dir / scenario
        img_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{episode_id}_agent{agent_id}_step{step:04d}.png"
        out_path = img_dir / fname
        pil_img.save(out_path)

        # action and reward may be present in row
        action = rec.get('action', None)
        reward = rec.get('reward', None)

        # Safely extract scalar action/reward values when fields may be arrays or lists
        def _safe_scalar(val):
            if val is None:
                return None
            # numpy arrays or lists
            if isinstance(val, (list, tuple, np.ndarray)):
                try:
                    arr = np.asarray(val)
                    if arr.size == 0:
                        return None
                    v = arr.flat[0]
                    # convert numpy scalar to Python native
                    if hasattr(v, 'item'):
                        v = v.item()
                    return int(v) if float(v).is_integer() else float(v)
                except Exception:
                    return None
            # pandas NA handling or plain scalars
            try:
                if pd.isna(val):
                    return None
            except Exception:
                pass
            try:
                return int(val) if float(val).is_integer() else float(val)
            except Exception:
                try:
                    return float(val)
                except Exception:
                    return None

        csv_rows.append({
            'image_path': str(out_path.relative_to(out_dir)),
            'episode_id': episode_id,
            'step': int(step),
            'agent_id': int(agent_id),
            'action': _safe_scalar(action),
            'reward': _safe_scalar(reward),
            'scenario': scenario
        })

        saved += 1
        if max_images and saved >= max_images:
            break


def find_parquet_files(input_dir: Path):
    files = list(input_dir.rglob("*_transitions.parquet"))
    return files


def main():
    parser = argparse.ArgumentParser(description='Extract grayscale frames for CLIP fine-tuning')
    parser.add_argument('--input', type=str, required=True, help='Input dataset base dir (where scenario dirs live)')
    parser.add_argument('--output', type=str, required=True, help='Output frames base dir')
    parser.add_argument('--max-per-scenario', type=int, default=0, help='Max images to extract per parquet file (0=all)')
    parser.add_argument('--csv-name', type=str, default='frames.csv', help='CSV filename to write in output dir')
    args = parser.parse_args()

    input_dir = Path(args.input)
    # if provided path doesn't exist, try relative to project root
    project_root = Path(__file__).resolve().parent.parent
    if not input_dir.exists():
        alt = project_root / args.input
        if alt.exists():
            input_dir = alt

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = find_parquet_files(input_dir)
    if not parquet_files:
        print('No parquet transition files found under', input_dir)
        return 1

    csv_rows = []
    for p in tqdm(parquet_files, desc='Parquet files'):
        try:
            process_parquet_file(p, out_dir, csv_rows, max_images=args.max_per_scenario)
        except Exception as e:
            print(f'Failed to process {p}: {e}')

    # Write CSV
    df = pd.DataFrame(csv_rows)
    csv_path = out_dir / args.csv_name
    df.to_csv(csv_path, index=False)
    print(f'Wrote {len(df)} image records to {csv_path}')


if __name__ == '__main__':
    raise SystemExit(main())
