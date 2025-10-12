"""Stronger thumbnail enhancement: autocontrast -> equalize -> brightness & contrast boost.

Usage: python3 tools/enhance_gallery.py

This overwrites PNGs in data/ambulance_plots/ in-place. Tune BRIGHTNESS_FACTOR and CONTRAST_FACTOR.
"""
import sys
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "data" / "ambulance_plots"

BRIGHTNESS_FACTOR = 1.8
CONTRAST_FACTOR = 1.2

if not PLOTS_DIR.exists():
    print("No plots directory found:", PLOTS_DIR)
    sys.exit(1)

count = 0
for png in PLOTS_DIR.rglob("*.png"):
    try:
        im = Image.open(png)
        # Convert to L (grayscale)
        if im.mode != "L":
            im = im.convert("L")
        # Autocontrast to stretch histogram
        im = ImageOps.autocontrast(im, cutoff=0)
        # Equalize histogram
        im = ImageOps.equalize(im)
        # Boost brightness and contrast
        im = ImageEnhance.Brightness(im).enhance(BRIGHTNESS_FACTOR)
        im = ImageEnhance.Contrast(im).enhance(CONTRAST_FACTOR)
        im.save(png)
        count += 1
    except Exception as e:
        print("ERROR processing", png, e)

print(f"Enhanced {count} PNG files under {PLOTS_DIR}")
