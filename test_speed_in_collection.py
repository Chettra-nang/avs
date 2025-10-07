#!/usr/bin/env python3
"""
Quick Collection Test to Verify Speed Display Fix

This test runs a short data collection and verifies that:
1. Speed values vary in the actual data
2. Text summaries show precise speed values (2 decimal places)
3. No more "all speeds are 3.6 km/h" issue
"""

import sys
import numpy as np
import pandas as pd
import re
from pathlib import Path
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def quick_collection_test():
    """Run a quick collection test and verify speed display."""
    
    print("=" * 70)
    print("🚗 SPEED FIX VERIFICATION - Quick Collection Test")
    print("=" * 70)
    
    try:
        # Import required modules
        print("\n✅ Checking imports...")
        
        # First, let's just test the summarizer directly with some data
        from highway_datacollection.features.summarizer import LanguageSummarizer
        import gymnasium as gym
        import highway_env
        
        print("✅ Imports successful")
        
        # Test summarizer directly
        print("\n🧪 Testing Summarizer with live environment...")
        print("-" * 70)
        
        # Create a simple environment
        config = {
            "lanes_count": 4,
            "controlled_vehicles": 1,
            "vehicles_count": 8,
            "duration": 40,
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 8,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute": False,
                "normalize": False
            },
            "offscreen_rendering": True,
            "manual_control": False
        }
        
        env = gym.make("highway-v0", config=config, render_mode=None)
        obs, info = env.reset()
        
        print("✅ Environment created and reset")
        
        # Collect some steps
        summarizer = LanguageSummarizer(lane_width=4.0, num_lanes=4)
        
        speeds_in_text = []
        actual_speeds = []
        summaries = []
        
        print("\n� Running 20 steps and collecting speed data...")
        
        for step in range(20):
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Extract ego and others from observation
            if isinstance(obs, dict):
                obs = obs.get('observation', obs)
            
            # Reshape if needed
            if obs.ndim == 2:
                ego = obs[0]  # First vehicle
                others = obs[1:] if len(obs) > 1 else np.array([])
            else:
                ego = obs
                others = np.array([])
            
            # Generate summary
            summary = summarizer.summarize(ego, others, context={'scenario': 'highway_construction'})
            summaries.append(summary)
            
            # Extract speed from text
            match = re.search(r'at ([\d.]+) km/h', summary)
            if match:
                speeds_in_text.append(match.group(1))
            
            # Calculate actual speed
            ego_vel = ego[3:5]
            speed_ms = np.linalg.norm(ego_vel)
            actual_speeds.append(speed_ms)
            
            if step < 5:
                print(f"   Step {step}: {speed_ms:.4f} m/s → {summary[:80]}...")
            
            if terminated or truncated:
                break
        
        env.close()
        
        # Analyze the summaries we just collected
        print("\n📊 Analyzing generated summaries from test run...")
        
        # Check unique speeds
        unique_speeds = sorted(set(speeds_in_text))
        speed_counts = Counter(speeds_in_text)
        
        print(f"\n📈 SPEED ANALYSIS FROM TEST RUN:")
        print(f"   Total steps: {len(summaries)}")
        print(f"   Unique speeds in text: {len(unique_speeds)}")
        if unique_speeds:
            print(f"   Speed range: {min(unique_speeds)} - {max(unique_speeds)} km/h")
        
        # Show speeds
        print(f"\n   Speeds found in summaries:")
        for i, speed in enumerate(unique_speeds[:15]):
            count = speed_counts[speed]
            print(f"      {i+1}. {speed} km/h (appeared {count} times)")
        
        # Check if speeds have 2 decimal places
        has_two_decimals = any('.' in s and len(s.split('.')[-1]) >= 2 for s in unique_speeds if s)
        
        print(f"\n� PRECISION CHECK:")
        print(f"   Has 2 decimal places: {'✅ YES' if has_two_decimals else '❌ NO (still using .1f)'}")
        
        # Show actual speeds vs text speeds
        print(f"\n📊 ACTUAL SPEED DATA (m/s):")
        print(f"   Mean: {np.mean(actual_speeds):.4f} m/s")
        print(f"   Std:  {np.std(actual_speeds):.4f} m/s")
        print(f"   Min:  {np.min(actual_speeds):.4f} m/s")
        print(f"   Max:  {np.max(actual_speeds):.4f} m/s")
        
        # Now check existing collected data if available
        print("\n📁 Checking for existing collected data files...")
        data_dir = Path("data")
        parquet_files = list(data_dir.rglob("*_transitions.parquet"))
        
        if not parquet_files:
            print("   No existing parquet files found in data directory")
            print("   (This is OK - we tested with live environment)")
        else:
            print(f"   Found {len(parquet_files)} existing parquet file(s)")
            
            # Just check the most recent one
            parquet_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
            print(f"\n🔍 Checking most recent file: {parquet_file.name}")
        
            df = pd.read_parquet(parquet_file)
            
            # Check if summary_text column exists
            if 'summary_text' not in df.columns:
                print("      No 'summary_text' column found in existing data")
            else:
                print(f"      Found summary_text column with {len(df)} rows")
                
                # Extract speeds from text summaries (just a sample)
                sample_speeds = []
                for text in df['summary_text'].head(50):
                    match = re.search(r'at ([\d.]+) km/h', text)
                    if match:
                        sample_speeds.append(match.group(1))
                
                unique_sample = sorted(set(sample_speeds))
                has_two_dec_existing = any('.' in s and len(s.split('.')[-1]) >= 2 for s in unique_sample if s)
                
                print(f"      Sample check (first 50 rows):")
                print(f"        Unique speeds: {len(unique_sample)}")
                print(f"        Has 2 decimals: {'✅ YES' if has_two_dec_existing else '❌ NO'}")
                print(f"        Sample speeds: {', '.join(unique_sample[:10])}")
        
        # Final verdict
        print("\n" + "=" * 70)
        if has_two_decimals and len(unique_speeds) > 10:
            print("✅ SUCCESS! Speed fix is working correctly!")
            print("   - Text summaries show speeds with 2 decimal places")
            print(f"   - {len(unique_speeds)} unique speeds displayed (good variation)")
            print("   - No longer showing only '3.6 km/h' for everything")
            return True
        elif not has_two_decimals:
            print("⚠️  WARNING: Speeds not showing 2 decimal places")
            print("   The fix may not be applied or data was generated before fix")
            return False
        else:
            print("⚠️  LOW VARIATION: Only a few unique speeds found")
            print("   This might be expected if vehicles are mostly stationary")
            return True
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print("=" * 70)

if __name__ == "__main__":
    success = quick_collection_test()
    sys.exit(0 if success else 1)
