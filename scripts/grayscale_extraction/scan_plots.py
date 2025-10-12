#!/usr/bin/env python3
"""Scan generated preview PNGs to find vertical or black images.

Usage: python3 scripts/grayscale_extraction/scan_plots.py --plots-dir data/ambulance_plots

Outputs:
 - CSV report at <plots-dir>_scan_report.csv
 - Copies of problematic images to <plots-dir>_diagnostics/
"""
import argparse
from pathlib import Path
from PIL import Image
import csv
import shutil


def analyze_image(path: Path):
    try:
        im = Image.open(path).convert('L')  # grayscale
        w, h = im.size
        arr = im
        # compute mean brightness [0..255]
        pixels = list(arr.getdata())
        mean = sum(pixels) / len(pixels)
        # percent near zero
        near_zero = sum(1 for p in pixels if p <= 5) / len(pixels)
        return {
            'path': str(path),
            'width': w,
            'height': h,
            'mean': mean,
            'near_zero_frac': near_zero,
            'vertical': int(h > w),
            'black': int(mean < 8 or near_zero > 0.5)
        }
    except Exception as e:
        return {
            'path': str(path),
            'width': None,
            'height': None,
            'mean': None,
            'near_zero_frac': None,
            'vertical': 0,
            'black': 1,
            'error': str(e)
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plots-dir', required=True)
    p.add_argument('--out-csv', default=None)
    p.add_argument('--copy-problems', default=None)
    args = p.parse_args()

    plots_dir = Path(args.plots_dir)
    if not plots_dir.exists():
        print('plots-dir does not exist:', plots_dir)
        return

    rows = []
    pngs = list(plots_dir.glob('**/*.png'))
    print('Found', len(pngs), 'png files under', plots_dir)
    for img in pngs:
        info = analyze_image(img)
        rows.append(info)

    out_csv = Path(args.out_csv) if args.out_csv else plots_dir.with_name(plots_dir.name + '_scan_report.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path','width','height','mean','near_zero_frac','vertical','black','error'])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in writer.fieldnames})

    print('Wrote report to', out_csv)

    if args.copy_problems:
        dest = Path(args.copy_problems)
        dest.mkdir(parents=True, exist_ok=True)
        copied = 0
        for r in rows:
            if r.get('black') or r.get('vertical'):
                src = Path(r['path'])
                # preserve subdirs to avoid collisions
                rel = src.relative_to(plots_dir)
                out_path = dest / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, out_path)
                copied += 1
        print('Copied', copied, 'problem images to', dest)

if __name__ == '__main__':
    main()
