#!/usr/bin/env python3
"""
Corrected Fast Ambulance Data Analysis Script

Analyzes the actual collected data format and checks if fast scenarios are working.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_collected_speeds():
    """
    Analyze the actual speeds from collected ambulance data.
    """
    logger.info("Analyzing collected ambulance data speeds...")
    
    data_path = Path("data/test_fast_ambulance")
    
    # Load index
    index_file = data_path / "index.json"
    with open(index_file, 'r') as f:
        index = json.load(f)
    
    all_speeds = []
    all_ego_speeds = []
    scenario_results = {}
    
    for scenario_name in index['scenarios']:
        logger.info(f"Processing scenario: {scenario_name}")
        scenario_dir = data_path / scenario_name
        
        # Load parquet file
        parquet_files = list(scenario_dir.glob("*_transitions.parquet"))
        if not parquet_files:
            continue
            
        df = pd.read_parquet(parquet_files[0])
        
        # Get ambulance data (agent 0)
        ambulance_data = df[df['agent_id'] == 0].copy()
        
        if len(ambulance_data) > 0:
            # Method 1: Use the 'speed' column (km/h)
            speeds_from_column = ambulance_data['speed'].tolist()
            
            # Method 2: Calculate from ego_vx, ego_vy (convert m/s to km/h)
            ambulance_data['calculated_speed'] = np.sqrt(ambulance_data['ego_vx']**2 + ambulance_data['ego_vy']**2) * 3.6
            calculated_speeds = ambulance_data['calculated_speed'].tolist()
            
            all_speeds.extend(speeds_from_column)
            all_ego_speeds.extend(calculated_speeds)
            
            scenario_results[scenario_name] = {
                'speed_column': speeds_from_column,
                'calculated_speed': calculated_speeds,
                'mean_speed_column': np.mean(speeds_from_column),
                'mean_calculated_speed': np.mean(calculated_speeds),
                'max_speed_column': max(speeds_from_column),
                'max_calculated_speed': max(calculated_speeds),
                'data_points': len(ambulance_data)
            }
            
            logger.info(f"  {scenario_name}: {len(ambulance_data)} data points")
            logger.info(f"    Speed column: mean={np.mean(speeds_from_column):.1f} km/h, max={max(speeds_from_column):.1f} km/h")
            logger.info(f"    Calculated: mean={np.mean(calculated_speeds):.1f} km/h, max={max(calculated_speeds):.1f} km/h")
    
    # Load metadata to check reward_speed_range
    logger.info("\n=== CHECKING SCENARIO CONFIGURATIONS ===")
    for scenario_name in index['scenarios']:
        scenario_dir = data_path / scenario_name
        metadata_files = list(scenario_dir.glob("*_meta.jsonl"))
        if metadata_files:
            with open(metadata_files[0], 'r') as f:
                first_line = f.readline()
                metadata = json.loads(first_line)
                config = metadata.get('config', {})
                reward_speed_range = config.get('reward_speed_range', [0, 0])
                vehicles_count = config.get('vehicles_count', 0)
                
                logger.info(f"  {scenario_name}:")
                logger.info(f"    Reward speed range: {reward_speed_range} km/h")
                logger.info(f"    Vehicle count: {vehicles_count}")
    
    # Overall analysis
    if all_speeds:
        logger.info("\n=== SPEED ANALYSIS RESULTS ===")
        logger.info(f"Total data points: {len(all_speeds)}")
        logger.info(f"Speed column - Mean: {np.mean(all_speeds):.1f} km/h, Max: {max(all_speeds):.1f} km/h")
        logger.info(f"Calculated speed - Mean: {np.mean(all_ego_speeds):.1f} km/h, Max: {max(all_ego_speeds):.1f} km/h")
        
        # Check if fast speeds are present
        fast_speeds = [s for s in all_speeds if s >= 40]
        highway_speeds = [s for s in all_speeds if s >= 60]
        
        logger.info(f"\nSpeed Categories:")
        logger.info(f"  Fast speeds (40+ km/h): {len(fast_speeds)} / {len(all_speeds)} ({len(fast_speeds)/len(all_speeds)*100:.1f}%)")
        logger.info(f"  Highway speeds (60+ km/h): {len(highway_speeds)} / {len(all_speeds)} ({len(highway_speeds)/len(all_speeds)*100:.1f}%)")
        
        # Create visualization
        create_speed_plots(scenario_results)
        
        return scenario_results
    else:
        logger.error("No speed data found!")
        return None

def create_speed_plots(scenario_results):
    """Create speed visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Speed comparison by scenario
    scenarios = list(scenario_results.keys())
    mean_speeds = [scenario_results[s]['mean_speed_column'] for s in scenarios]
    max_speeds = [scenario_results[s]['max_speed_column'] for s in scenarios]
    
    x_pos = np.arange(len(scenarios))
    width = 0.35
    
    axes[0, 0].bar(x_pos - width/2, mean_speeds, width, label='Mean Speed', alpha=0.8, color='blue')
    axes[0, 0].bar(x_pos + width/2, max_speeds, width, label='Max Speed', alpha=0.8, color='red')
    axes[0, 0].set_xlabel('Scenario')
    axes[0, 0].set_ylabel('Speed (km/h)')
    axes[0, 0].set_title('Ambulance Speeds by Scenario')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(scenarios, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add expected fast speed lines
    axes[0, 0].axhline(y=40, color='orange', linestyle='--', alpha=0.7, label='Arterial Speed (40 km/h)')
    axes[0, 0].axhline(y=60, color='green', linestyle='--', alpha=0.7, label='Highway Speed (60 km/h)')
    
    # 2. Speed distribution histogram
    all_speeds = []
    for scenario_data in scenario_results.values():
        all_speeds.extend(scenario_data['speed_column'])
    
    axes[0, 1].hist(all_speeds, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(np.mean(all_speeds), color='red', linestyle='--', label=f'Mean: {np.mean(all_speeds):.1f} km/h')
    axes[0, 1].axvline(40, color='orange', linestyle='--', label='Target: 40+ km/h')
    axes[0, 1].set_xlabel('Speed (km/h)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Speed Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Speed over time for first scenario
    if scenarios:
        first_scenario = scenarios[0]
        speeds = scenario_results[first_scenario]['speed_column']
        axes[1, 0].plot(speeds, marker='o', markersize=3, linewidth=2, color='blue')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Speed (km/h)')
        axes[1, 0].set_title(f'Speed Profile: {first_scenario}')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(40, color='orange', linestyle='--', alpha=0.7, label='Target: 40 km/h')
        axes[1, 0].legend()
    
    # 4. Speed comparison: column vs calculated
    scenario_names_short = [s.replace('highway_', '').replace('_', '\n') for s in scenarios]
    column_means = [scenario_results[s]['mean_speed_column'] for s in scenarios]
    calc_means = [scenario_results[s]['mean_calculated_speed'] for s in scenarios]
    
    x_pos = np.arange(len(scenarios))
    axes[1, 1].scatter(x_pos, column_means, color='blue', label='Speed Column', s=100, alpha=0.7)
    axes[1, 1].scatter(x_pos, calc_means, color='red', label='Calculated Speed', s=100, alpha=0.7)
    axes[1, 1].set_xlabel('Scenario')
    axes[1, 1].set_ylabel('Mean Speed (km/h)')
    axes[1, 1].set_title('Speed Measurement Comparison')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(scenario_names_short, fontsize=8)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("output/fast_ambulance_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_file = output_dir / "ambulance_speed_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Saved speed analysis plot: {plot_file}")

def check_scenario_configs():
    """
    Check if the ambulance scenarios are using the updated fast configurations.
    """
    logger.info("\n=== CHECKING SCENARIO CONFIGURATION FILES ===")
    
    # Check if ambulance_scenarios.py has the fast configurations
    scenarios_file = Path("collecting_ambulance_data/scenarios/ambulance_scenarios.py")
    if scenarios_file.exists():
        with open(scenarios_file, 'r') as f:
            content = f.read()
        
        # Look for speed limit configurations
        if "speed_limit_kmh" in content:
            logger.info("✅ Found speed_limit_kmh configurations in ambulance_scenarios.py")
            
            # Extract some sample configurations
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'speed_limit_kmh' in line and i < len(lines) - 1:
                    logger.info(f"  Line {i+1}: {line.strip()}")
                    # Show context
                    if i > 0:
                        logger.info(f"  Context: {lines[i-1].strip()}")
        else:
            logger.warning("⚠️ No speed_limit_kmh found in ambulance_scenarios.py")
            logger.info("This might explain why we're seeing slow speeds!")
    else:
        logger.error("❌ ambulance_scenarios.py not found!")
    
    # Also check the main ambulance scenario registry
    registry_files = [
        "collecting_ambulance_data/scenarios/__init__.py",
        "collecting_ambulance_data/__init__.py"
    ]
    
    for registry_file in registry_files:
        if Path(registry_file).exists():
            logger.info(f"✅ Found {registry_file}")
        else:
            logger.warning(f"⚠️ Missing {registry_file}")

def main():
    """Main analysis function."""
    logger.info("=== FAST AMBULANCE DATA COLLECTION ANALYSIS ===")
    
    # Analyze collected data
    results = analyze_collected_speeds()
    
    if results:
        # Check configuration files
        check_scenario_configs()
        
        # Summary
        all_speeds = []
        for scenario_data in results.values():
            all_speeds.extend(scenario_data['speed_column'])
        
        mean_speed = np.mean(all_speeds)
        max_speed = max(all_speeds)
        fast_count = len([s for s in all_speeds if s >= 40])
        
        logger.info("\n=== FINAL ASSESSMENT ===")
        if mean_speed >= 40:
            logger.info("🎉 SUCCESS: Ambulance scenarios are using fast speeds!")
        elif mean_speed >= 25:
            logger.info("⚠️ PARTIAL: Some fast speeds detected, but not consistently high")
        else:
            logger.info("❌ ISSUE: Ambulance speeds are still slow (city-level)")
            logger.info("   This suggests the fast scenario transformations are not being applied")
        
        logger.info(f"📊 Results: Mean={mean_speed:.1f} km/h, Max={max_speed:.1f} km/h")
        logger.info(f"🚀 Fast episodes: {fast_count}/{len(all_speeds)} ({fast_count/len(all_speeds)*100:.1f}%)")
        
        return results
    else:
        logger.error("Analysis failed!")
        return None

if __name__ == "__main__":
    main()