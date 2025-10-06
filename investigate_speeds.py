#!/usr/bin/env python3
"""
Diagnostic script to investigate why ambulance speeds are unexpectedly low
"""

import pandas as pd
import numpy as np
from pathlib import Path

def investigate_speed_data():
    """Investigate the speed data to understand why speeds are low."""
    
    # Load a sample file for detailed analysis
    sample_file = Path("d:/Research_ITC/avs_folder/avs/data/ambulance_dataset_150_espisode_cpu_30_senario/ambulance_dataset_150_espisode_cpu_30_senario/batch_653831/highway_merge_heavy/20251004_121200-32e17d2d_transitions.parquet")
    
    df = pd.read_parquet(sample_file)
    
    print("🔍 SPEED DATA INVESTIGATION")
    print("=" * 50)
    
    print(f"\n📁 Sample file: {sample_file.name}")
    print(f"📊 Total rows: {len(df)}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Check speed column directly
    print(f"\n🏃 SPEED COLUMN ANALYSIS:")
    speed_col = df['speed']
    print(f"Data type: {speed_col.dtype}")
    print(f"Sample values (first 10):")
    for i in range(min(10, len(speed_col))):
        print(f"  [{i}] {speed_col.iloc[i]} (type: {type(speed_col.iloc[i])})")
    
    # Check if speeds are numeric
    numeric_speeds = pd.to_numeric(speed_col, errors='coerce')
    valid_speeds = numeric_speeds.dropna()
    
    if len(valid_speeds) > 0:
        print(f"\n📈 NUMERIC SPEED STATS:")
        print(f"Valid numeric values: {len(valid_speeds)}/{len(speed_col)}")
        print(f"Min: {valid_speeds.min():.6f}")
        print(f"Max: {valid_speeds.max():.6f}")
        print(f"Mean: {valid_speeds.mean():.6f}")
        print(f"Median: {valid_speeds.median():.6f}")
        print(f"Std: {valid_speeds.std():.6f}")
    
    # Check other speed-related columns
    speed_related_cols = [col for col in df.columns if 'speed' in col.lower() or 'velocity' in col.lower() or col in ['ego_vx', 'ego_vy']]
    print(f"\n🔍 OTHER SPEED-RELATED COLUMNS:")
    for col in speed_related_cols:
        if col != 'speed':
            print(f"\n{col}:")
            sample_vals = df[col].head(5)
            for i, val in enumerate(sample_vals):
                print(f"  [{i}] {val} (type: {type(val)})")
    
    # Check velocity calculation from ego_vx, ego_vy
    if 'ego_vx' in df.columns and 'ego_vy' in df.columns:
        print(f"\n🧮 VELOCITY MAGNITUDE CALCULATION:")
        vx = pd.to_numeric(df['ego_vx'], errors='coerce')
        vy = pd.to_numeric(df['ego_vy'], errors='coerce')
        
        calculated_speed = np.sqrt(vx**2 + vy**2)
        valid_calc_speeds = calculated_speed.dropna()
        
        if len(valid_calc_speeds) > 0:
            print(f"Calculated from ego_vx, ego_vy:")
            print(f"  Min: {valid_calc_speeds.min():.6f} m/s")
            print(f"  Max: {valid_calc_speeds.max():.6f} m/s") 
            print(f"  Mean: {valid_calc_speeds.mean():.6f} m/s")
            
            # Compare with recorded speeds
            if len(valid_speeds) > 0:
                print(f"\n🔄 COMPARISON:")
                print(f"  Recorded speed mean: {valid_speeds.mean():.6f} m/s")
                print(f"  Calculated speed mean: {valid_calc_speeds.mean():.6f} m/s")
                print(f"  Ratio (recorded/calculated): {valid_speeds.mean()/valid_calc_speeds.mean():.3f}")
    
    # Check ambulance vs NPC data
    print(f"\n🚑 AGENT ANALYSIS:")
    ambulance_data = df[df['agent_id'] == 0]
    npc_data = df[df['agent_id'] != 0]
    
    print(f"Ambulance rows: {len(ambulance_data)}")
    print(f"NPC rows: {len(npc_data)}")
    
    if len(ambulance_data) > 0:
        amb_speeds = pd.to_numeric(ambulance_data['speed'], errors='coerce').dropna()
        if len(amb_speeds) > 0:
            print(f"Ambulance speed stats:")
            print(f"  Count: {len(amb_speeds)}")
            print(f"  Mean: {amb_speeds.mean():.6f} m/s ({amb_speeds.mean()*3.6:.2f} km/h)")
            print(f"  Max: {amb_speeds.max():.6f} m/s ({amb_speeds.max()*3.6:.2f} km/h)")
    
    if len(npc_data) > 0:
        npc_speeds = pd.to_numeric(npc_data['speed'], errors='coerce').dropna()
        if len(npc_speeds) > 0:
            print(f"NPC speed stats:")
            print(f"  Count: {len(npc_speeds)}")
            print(f"  Mean: {npc_speeds.mean():.6f} m/s ({npc_speeds.mean()*3.6:.2f} km/h)")
            print(f"  Max: {npc_speeds.max():.6f} m/s ({npc_speeds.max()*3.6:.2f} km/h)")
    
    # Check summary_text for context
    if 'summary_text' in df.columns:
        print(f"\n📝 SAMPLE SUMMARY TEXTS (for context):")
        sample_summaries = df['summary_text'].dropna().head(3)
        for i, summary in enumerate(sample_summaries):
            print(f"  [{i}] {str(summary)[:100]}...")
    
    # Check scenario and ambulance_scenario columns
    if 'scenario' in df.columns:
        scenarios = df['scenario'].value_counts()
        print(f"\n🎭 SCENARIOS IN FILE:")
        print(scenarios)
    
    if 'ambulance_scenario' in df.columns:
        amb_scenarios = df['ambulance_scenario'].value_counts()
        print(f"\n🚑 AMBULANCE SCENARIOS:")
        print(amb_scenarios)
    
    # Check if vehicles are actually moving or stationary
    if 'ego_x' in df.columns and 'ego_y' in df.columns:
        print(f"\n📍 POSITION ANALYSIS:")
        x_positions = pd.to_numeric(df['ego_x'], errors='coerce')
        y_positions = pd.to_numeric(df['ego_y'], errors='coerce')
        
        if len(x_positions.dropna()) > 1:
            x_range = x_positions.max() - x_positions.min()
            y_range = y_positions.max() - y_positions.min()
            print(f"X position range: {x_range:.2f} meters")
            print(f"Y position range: {y_range:.2f} meters")
            
            if x_range < 1 and y_range < 1:
                print("⚠️  WARNING: Very small position changes - vehicles may be mostly stationary!")
    
    print(f"\n🔍 POTENTIAL CAUSES OF LOW SPEEDS:")
    print("1. Units issue - speeds might be in different units than expected")
    print("2. Simulation timestep - speeds might be per-timestep rather than per-second")
    print("3. Traffic conditions - heavy congestion causing low speeds")
    print("4. Data collection during stationary periods")
    print("5. Speed calculation method differences")

if __name__ == "__main__":
    investigate_speed_data()