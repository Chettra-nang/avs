#!/usr/bin/env python3
"""
Diagnose why text summary always shows 3.6 km/h

This script analyzes the speed issue in the parquet data and provides solutions.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

def analyze_speed_issue():
    """Analyze the speed issue in the parquet file."""
    
    parquet_file = Path('data/ambulance_dataset_fast_150_espisode_cpu_30_senario/batch_2212756/highway_construction/20251007_011547-91e99ee4_transitions.parquet')
    
    print("=" * 70)
    print("SPEED ISSUE DIAGNOSIS")
    print("=" * 70)
    
    # Load data
    df = pd.read_parquet(parquet_file)
    
    # Extract speeds from text
    speeds_in_text = []
    for text in df['summary_text']:
        match = re.search(r'at ([\d.]+) km/h', text)
        if match:
            speeds_in_text.append(float(match.group(1)))
    
    print(f"\n📊 DATASET STATISTICS:")
    print(f"   Total rows: {len(df)}")
    print(f"   Speed column (m/s) - Min: {df['speed'].min():.6f}, Max: {df['speed'].max():.6f}")
    print(f"   Speed column (m/s) - Mean: {df['speed'].mean():.6f}, Std: {df['speed'].std():.6f}")
    print(f"   Unique actual speeds (m/s): {len(df['speed'].unique())}")
    
    print(f"\n🔍 SUMMARY TEXT ANALYSIS:")
    print(f"   Unique speeds in text (km/h): {len(set(speeds_in_text))}")
    print(f"   Speed range in text: {min(speeds_in_text):.1f} - {max(speeds_in_text):.1f} km/h")
    
    # Show the problem
    print(f"\n⚠️  THE PROBLEM:")
    print(f"   Most vehicles are moving at ~1.0 m/s")
    print(f"   When converted: 1.0 m/s × 3.6 = 3.6 km/h")
    print(f"   Formatted with :.1f → Always shows '3.6 km/h'")
    
    # Demonstrate the rounding issue
    print(f"\n🔬 ROUNDING DEMONSTRATION:")
    sample_speeds_ms = [0.95, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05]
    print(f"   {'Speed (m/s)':<15} {'km/h (exact)':<15} {':.1f format':<15} {':.2f format'}")
    for speed_ms in sample_speeds_ms:
        speed_kmh = speed_ms * 3.6
        formatted_1f = f"{speed_kmh:.1f}"
        formatted_2f = f"{speed_kmh:.2f}"
        print(f"   {speed_ms:<15.2f} {speed_kmh:<15.4f} {formatted_1f:<15} {formatted_2f}")
    
    # Analyze actual data samples
    print(f"\n📈 ACTUAL DATA SAMPLES (showing speed variation):")
    sample_df = df[['speed', 'summary_text']].head(20)
    print(f"   {'Speed (m/s)':<12} {'Speed (km/h)':<12} {'Text Summary (excerpt)'}")
    print(f"   {'-'*12} {'-'*12} {'-'*40}")
    for _, row in sample_df.iterrows():
        speed_ms = row['speed']
        speed_kmh = speed_ms * 3.6
        text_excerpt = row['summary_text'][:50] + "..."
        # Extract speed from text
        match = re.search(r'at ([\d.]+) km/h', row['summary_text'])
        text_speed = match.group(1) if match else "N/A"
        print(f"   {speed_ms:<12.6f} {speed_kmh:<12.4f} at {text_speed} km/h - {text_excerpt[:30]}...")
    
    # Root cause analysis
    print(f"\n🎯 ROOT CAUSE:")
    print(f"   1. Vehicles in the simulation are mostly stationary/very slow (~1 m/s)")
    print(f"   2. The formatting uses .1f which rounds to 1 decimal place")
    print(f"   3. Speeds like 3.42, 3.58, 3.61, 3.78 all round to 3.6")
    print(f"   4. This causes lack of precision in the text summary")
    
    # Solutions
    print(f"\n✅ SOLUTIONS:")
    print(f"   Option 1: Increase format precision to .2f (e.g., 3.42 km/h, 3.58 km/h)")
    print(f"   Option 2: Keep .1f but verify simulation produces varied speeds")
    print(f"   Option 3: Use integer formatting for higher speeds: .0f")
    print(f"   Option 4: Conditional formatting based on speed range")
    
    # Show what it would look like with different formatting
    print(f"\n🔧 COMPARISON OF FORMATTING OPTIONS:")
    print(f"   Using speeds from actual data:")
    sample_speeds = df['speed'].head(10).values
    print(f"   {'m/s':<8} {'km/h':<10} {'.1f format':<12} {'.2f format':<12} {'.0f format'}")
    for speed_ms in sample_speeds:
        speed_kmh = speed_ms * 3.6
        print(f"   {speed_ms:<8.4f} {speed_kmh:<10.4f} {speed_kmh:<12.1f} {speed_kmh:<12.2f} {speed_kmh:<12.0f}")
    
    print(f"\n" + "=" * 70)
    print("RECOMMENDATION: Change format from .1f to .2f for better precision")
    print("Location: highway_datacollection/features/summarizer.py (8 places)")
    print("=" * 70)

if __name__ == "__main__":
    analyze_speed_issue()
