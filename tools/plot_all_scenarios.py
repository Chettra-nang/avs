"""Driver: extract grayscale preview frames for all scenarios listed in consolidated_index.json

This script uses the helper functions from tools/plot_episode_frames.py to extract
frames for each scenario in `data/ambulance_dataset_diagnose/consolidated_index.json`.
It writes up to N preview frames per scenario into `data/ambulance_plots/<scenario>/`.
It also creates a simple HTML gallery at `data/ambulance_plots_html/index.html`.

Run from project root:

python tools/plot_all_scenarios.py --nframes 5

"""
import os
import json
import argparse
import sys
from pathlib import Path

# Make sure the repository root is on sys.path so `tools` can be imported when running
# this script directly (not as an installed package).
ROOT = Path('.')
sys.path.insert(0, str(ROOT.resolve()))

# Import helper functions from existing script
from tools.plot_episode_frames import find_parquet_in_scenario, extract_grayscale_from_parquet

ROOT = Path('.')
# Default index path (kept for backward-compatibility)
INDEX_PATH = ROOT / 'data' / 'ambulance_dataset_diagnose' / 'consolidated_index.json'
OUT_ROOT = ROOT / 'data' / 'ambulance_plots'
GALLERY_DIR = ROOT / 'data' / 'ambulance_plots_html'


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def run_all(nframes=5):
    with open(INDEX_PATH, 'r') as f:
        idx = json.load(f)

    total_saved = 0
    scenarios_processed = 0
    ensure_dir(OUT_ROOT)

    for batch in idx.get('batches', []):
        storage_path = batch.get('storage_path')
        # storage_path is relative like 'data/ambulance_dataset_diagnose/batch_17220'
        batch_dir = ROOT / storage_path
        for scenario in batch.get('scenarios', []):
            scenarios_processed += 1
            out_dir = OUT_ROOT / scenario
            ensure_dir(out_dir)
            print('Processing', scenario, 'from', batch_dir)
            try:
                parquet_files = find_parquet_in_scenario(str(batch_dir), scenario)
                if not parquet_files:
                    print('  no parquet files found for', scenario)
                    continue
                # process first parquet file for preview
                saved = 0
                for p in parquet_files:
                    saved += extract_grayscale_from_parquet(p, str(out_dir), nframes=nframes - saved)
                    if saved >= nframes:
                        break
                print(f'  saved {saved} frames to {out_dir}')
                total_saved += saved
            except Exception as e:
                print('  error processing', scenario, ':', e)

    # Build gallery
    ensure_dir(GALLERY_DIR)
    html_path = GALLERY_DIR / 'index.html'
    with open(html_path, 'w') as h:
        h.write('<html><meta charset="utf-8"><body><h1>Ambulance dataset previews</h1>')
        for folder in sorted(OUT_ROOT.iterdir()):
            if not folder.is_dir():
                continue
            h.write(f'<h2>{folder.name}</h2>')
            imgs = sorted(folder.glob('*.png'))[:10]
            for img in imgs:
                # use absolute file:// URL so browsers can open local files reliably
                img_path = img.resolve()
                h.write(f'<img src="file://{img_path.as_posix()}" style="margin:6px;height:160px;">')
        h.write('</body></html>')

    print('Done. scenarios_processed=', scenarios_processed, 'total_saved=', total_saved)
    print('Gallery:', html_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nframes', type=int, default=5)
    parser.add_argument('--index', type=str, default=None, help='Path to consolidated_index.json (overrides default)')
    args = parser.parse_args()
    # Allow custom index path
    if args.index:
        INDEX_PATH = Path(args.index)
    run_all(nframes=args.nframes)
