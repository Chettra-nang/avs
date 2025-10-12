"""Post-process generated PNG previews by applying histogram equalization.

Usage: python3 tools/equalize_gallery.py

This script walks data/ambulance_plots/ and applies PIL.ImageOps.equalize to each PNG.
It overwrites files in-place and prints progress.
"""
import sys
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "data" / "ambulance_plots"

if not PLOTS_DIR.exists():
    print("No plots directory found:", PLOTS_DIR)
    sys.exit(1)

count = 0
for png in PLOTS_DIR.rglob("*.png"):
    try:
        im = Image.open(png)
        # Ensure grayscale
        if im.mode not in ("L", "RGB"):
            im = im.convert("L")
        # If RGB, convert to L before equalize
        if im.mode == "RGB":
            im = im.convert("L")
        eq = ImageOps.equalize(im)
        eq.save(png)
        count += 1
    except Exception as e:
        print("ERROR processing", png, e)

print(f"Processed {count} PNG files under {PLOTS_DIR}")
