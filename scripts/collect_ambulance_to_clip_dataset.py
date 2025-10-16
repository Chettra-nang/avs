#!/usr/bin/env python3
"""
Consolidate ambulance collector outputs into CLIP dataset layout:
- frames/<scenario>/*.png (symlinks or copies)
- texts/<scenario>.csv with columns: image_path,caption,action_id,scenario,episode,step

The collector output layout can vary; this script walks the output directory and pulls image files.
It attempts to infer action labels from nearby metadata files if present; otherwise it assigns a default caption mapping.

Usage:
    python scripts/collect_ambulance_to_clip_dataset.py --collector-root data/ambulance_parallel/three_types --out-dir data/highway_multimodal_dataset --force-copy

"""
import argparse
from pathlib import Path
import csv
import json
import shutil
import sys
import os
from typing import Dict

DEFAULT_ACTION_CAPTIONS = {
    0: "Drive slower",
    1: "IDLE",
    2: "Drive faster",
}

IMAGE_EXTS = {'.png', '.jpg', '.jpeg'}


def find_images(root: Path):
    for p in root.rglob('*'):
        if p.suffix.lower() in IMAGE_EXTS:
            yield p


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collector-root', type=str, required=True,
                        help='Root directory where parallel collector saved batches (or consolidated_index.json location)')
    parser.add_argument('--out-dir', type=str, default='data/highway_multimodal_dataset',
                        help='Output CLIP dataset dir')
    parser.add_argument('--force-copy', action='store_true', help='Copy images instead of symlink')
    parser.add_argument('--action-map', type=str, default=None,
                        help='Optional JSON file mapping found labels to action_id and caption')
    args = parser.parse_args()

    collector_root = Path(args.collector_root)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / 'frames'
    texts_dir = out_dir / 'texts'
    frames_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)

    # Load optional action map
    action_map = None
    if args.action_map:
        try:
            with open(args.action_map, 'r') as f:
                action_map = json.load(f)
        except Exception as e:
            print(f"Failed to load action_map: {e}")
            action_map = None

    # If a consolidated_index.json exists, read it to find storage paths
    possible_index = collector_root / 'consolidated_index.json'
    storage_paths = []
    if possible_index.exists():
        try:
            with open(possible_index, 'r') as f:
                idx = json.load(f)
            for b in idx.get('batches', []):
                sp = b.get('storage_path')
                if sp:
                    storage_paths.append(Path(sp))
        except Exception:
            # fallback to scanning root
            storage_paths.append(collector_root)
    else:
        storage_paths.append(collector_root)

    # Walk storage paths and collect images grouped by scenario name (best-effort)
    images_by_scenario = {}
    for sp in storage_paths:
        if not sp.exists():
            continue
        for img in find_images(sp):
            # try to extract scenario from path parts: look for known scenario names (heuristic)
            parts = [p for p in img.parts]
            scenario = None
            for part in parts[::-1]:
                if part.startswith('highway_') or part.startswith('merge_') or part.startswith('intersection') or part.startswith('roundabout') or part.startswith('corner'):
                    scenario = part
                    break
            if scenario is None:
                # fallback: use parent directory name
                scenario = img.parent.name
            images_by_scenario.setdefault(scenario, []).append(img)

    # For each scenario, create frames dir and texts CSV
    for scenario, imgs in images_by_scenario.items():
        dest_frames = frames_dir / scenario
        dest_frames.mkdir(parents=True, exist_ok=True)
        csv_path = texts_dir / f"{scenario}.csv"
        print(f"Processing scenario {scenario}: {len(imgs)} images -> {dest_frames}")

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # write header expected by finetune ImageTextDataset
            writer.writerow(['frame_file', 'instruction', 'action_id', 'scenario', 'episode', 'step'])
            for img in imgs:
                # determine destination path
                dest = dest_frames / img.name
                img_abspath = str(img.resolve())
                try:
                    if args.force_copy:
                        shutil.copy2(img, dest)
                    else:
                        # create symlink if not exists
                        if dest.exists() or dest.is_symlink():
                            try:
                                dest.unlink()
                            except Exception:
                                pass
                        os.symlink(img_abspath, str(dest))
                except Exception:
                    # fallback to copying
                    shutil.copy2(img, dest)

                # attempt to find metadata near image (same folder .json or .meta)
                caption = None
                action_id = None
                meta_json = img.with_suffix('.json')
                if meta_json.exists():
                    try:
                        with open(meta_json, 'r') as f:
                            meta = json.load(f)
                        # heuristics: prefer fields 'action', 'label', 'action_id', 'caption'
                        if 'action_id' in meta:
                            action_id = int(meta['action_id'])
                        elif 'action' in meta:
                            # try map
                            action_str = str(meta['action'])
                            if action_map and action_str in action_map:
                                action_map_item = action_map[action_str]
                                action_id = int(action_map_item.get('action_id', 1))
                                caption = action_map_item.get('caption')
                            else:
                                # fallback: assign ID 1
                                action_id = 1
                        if 'caption' in meta:
                            caption = meta['caption']
                    except Exception:
                        pass

                # fallback to defaults
                if action_id is None:
                    action_id = 1
                if caption is None:
                    caption = DEFAULT_ACTION_CAPTIONS.get(action_id, 'IDLE')

                # episode/step not known reliably here; leave empty placeholders
                # Write absolute frame_file path so finetune loader can pick it up
                writer.writerow([str(dest.resolve()), caption, action_id, scenario, '', ''])

        print(f"Wrote CSV {csv_path}")

    print("Consolidation complete.")


if __name__ == '__main__':
    main()
