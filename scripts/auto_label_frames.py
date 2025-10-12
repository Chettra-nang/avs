#!/usr/bin/env python3
"""
Auto-label extracted frames for CLIP fine-tuning.

Reads the `frames.csv` produced by `extract_frames_for_clip.py`, samples up to N
images balanced across action classes for `agent_id==0`, and writes a CSV of
image_path,text. Optionally copies the selected images to an output folder.

Default action->text mapping is inferred from project docs (modify with
--mapping to override). The mapping here is a conservative default and should
be validated by the user.

Example:
  python3 AVs/scripts/auto_label_frames.py --frames AVs/data/clip_frames/frames.csv --out AVs/data/clip_frames/labels_autolabel.csv --n 500 --copy-to AVs/data/clip_frames/labeled_images
"""

import argparse
import json
import random
from pathlib import Path
import shutil
import pandas as pd


DEFAULT_MAPPING = {
    # inferred order from project docs: SLOWER, IDLE, FASTER, LANE_LEFT, LANE_RIGHT
    0: "Slow down",
    1: "Hold position",
    2: "Speed up",
    3: "Change lane left",
    4: "Change lane right",
}


def parse_mapping(s: str):
    # accept JSON like '{"0":"Slow down","1":"Hold position"}' or
    # comma separated '0=Slow down,1=Hold position'
    if not s:
        return DEFAULT_MAPPING
    try:
        m = json.loads(s)
        return {int(k): str(v) for k, v in m.items()}
    except Exception:
        out = {}
        for part in s.split(','):
            if '=' in part:
                k, v = part.split('=', 1)
                try:
                    out[int(k.strip())] = v.strip()
                except Exception:
                    continue
        # fill missing keys from default
        for k, v in DEFAULT_MAPPING.items():
            out.setdefault(k, v)
        return out


def main():
    parser = argparse.ArgumentParser(description='Auto-label frames for CLIP')
    parser.add_argument('--frames', type=str, default='AVs/data/clip_frames/frames.csv')
    parser.add_argument('--out', type=str, default='AVs/data/clip_frames/labels_autolabel.csv')
    parser.add_argument('--n', type=int, default=500, help='Total number of labeled samples desired')
    parser.add_argument('--mapping', type=str, default='', help='Action->text mapping as JSON or comma list e.g. "0=Slow down,1=Hold"')
    parser.add_argument('--copy-to', type=str, default='', help='Optional: copy selected images to this directory under images/')
    parser.add_argument('--images-root', type=str, default='AVs/data/clip_frames', help='Root where image_path in frames.csv is relative to')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    frames_p = Path(args.frames)
    if not frames_p.exists():
        raise SystemExit(f'frames CSV not found: {frames_p}')

    df = pd.read_csv(frames_p)
    # focus on ambulance agent (agent_id==0)
    df0 = df[df['agent_id'] == 0].copy()
    if df0.empty:
        raise SystemExit('No agent_id==0 rows found in frames CSV')

    mapping = parse_mapping(args.mapping)

    # Determine available actions and sample per-class
    action_values = sorted(df0['action'].dropna().unique().tolist())
    n_classes = len(action_values)
    per_class = max(1, args.n // n_classes)

    selected = []
    for a in action_values:
        rows = df0[df0['action'] == a]
        if rows.empty:
            continue
        take = min(len(rows), per_class)
        chosen = rows.sample(take, random_state=args.seed)
        selected.append(chosen)

    # If under target, fill from remaining rows
    sel_df = pd.concat(selected) if selected else pd.DataFrame()
    remaining_needed = args.n - len(sel_df)
    if remaining_needed > 0:
        pool = df0.drop(sel_df.index, errors='ignore')
        if not pool.empty:
            extra = pool.sample(min(remaining_needed, len(pool)), random_state=args.seed)
            sel_df = pd.concat([sel_df, extra])

    # Final shuffle
    sel_df = sel_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out_rows = []
    images_root = Path(args.images_root)
    copy_dir = Path(args.copy_to) if args.copy_to else None
    if copy_dir:
        (copy_dir / 'images').mkdir(parents=True, exist_ok=True)

    for _, row in sel_df.iterrows():
        img_rel = row['image_path']
        action = int(row['action']) if not pd.isna(row['action']) else None
        text = mapping.get(action, f'Action {action}')
        out_rows.append({'image_path': img_rel, 'text': text, 'action': action, 'scenario': row.get('scenario')})

        if copy_dir:
            src = images_root / img_rel
            dst = copy_dir / 'images' / Path(img_rel).name
            try:
                shutil.copy(src, dst)
            except Exception:
                # skip copy failures but keep CSV
                pass

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.out, index=False)
    print(f'Wrote {len(out_df)} labeled rows to {args.out}')
    print('Action counts in labeled set:')
    print(out_df['action'].value_counts())


if __name__ == '__main__':
    main()
