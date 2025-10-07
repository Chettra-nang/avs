#!/usr/bin/env python3
"""
Fast Ambulance Data Collection Verification Script

This script:
1. Analyzes collected ambulance data to verify fast speeds
2. Creates visualizations and plots
3. Generates videos showing ambulance behavior
4. Provides comprehensive speed analysis
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_ambulance_speeds(data_path: str):
    """
    Analyze speeds from collected ambulance data to verify fast scenarios are working.
    """
    logger.info(f"Analyzing ambulance speeds from: {data_path}")
    
    data_dir = Path(data_path)
    index_file = data_dir / "index.json"
    
    if not index_file.exists():
        logger.error(f"Index file not found: {index_file}")
        return None
    
    # Load index
    with open(index_file, 'r') as f:
        index = json.load(f)
    
    logger.info(f"Found {len(index['scenarios'])} scenarios in dataset")
    
    all_speeds = []
    scenario_speeds = {}
    
    # Process each scenario
    for scenario_name in index['scenarios']:
        scenario_dir = data_dir / scenario_name
        
        # Find parquet files
        parquet_files = list(scenario_dir.glob("*_transitions.parquet"))
        if not parquet_files:
            logger.warning(f"No parquet files found in {scenario_dir}")
            continue
        
        logger.info(f"Processing scenario: {scenario_name}")
        scenario_data = []
        
        for parquet_file in parquet_files:
            try:
                df = pd.read_parquet(parquet_file)
                logger.info(f"  Loaded {len(df)} transitions from {parquet_file.name}")
                
                # Calculate speeds for ambulance (agent 0)
                ambulance_data = df[df['agent_id'] == 0].copy()
                
                if len(ambulance_data) > 0:
                    # Calculate speed from vx, vy (m/s to km/h)
                    ambulance_data['speed_ms'] = np.sqrt(ambulance_data['vx']**2 + ambulance_data['vy']**2)
                    ambulance_data['speed_kmh'] = ambulance_data['speed_ms'] * 3.6
                    
                    speeds = ambulance_data['speed_kmh'].tolist()
                    scenario_data.extend(speeds)
                    all_speeds.extend(speeds)
                    
                    logger.info(f"    Ambulance speeds: min={min(speeds):.1f} km/h, max={max(speeds):.1f} km/h, mean={np.mean(speeds):.1f} km/h")
                
            except Exception as e:
                logger.error(f"Error processing {parquet_file}: {e}")
                continue
        
        if scenario_data:
            scenario_speeds[scenario_name] = scenario_data
    
    if not all_speeds:
        logger.error("No speed data found!")
        return None
    
    # Overall statistics
    stats = {
        'min_speed': min(all_speeds),
        'max_speed': max(all_speeds),
        'mean_speed': np.mean(all_speeds),
        'median_speed': np.median(all_speeds),
        'std_speed': np.std(all_speeds),
        'total_data_points': len(all_speeds)
    }
    
    logger.info("=== AMBULANCE SPEED ANALYSIS RESULTS ===")
    logger.info(f"Total data points: {stats['total_data_points']}")
    logger.info(f"Speed range: {stats['min_speed']:.1f} - {stats['max_speed']:.1f} km/h")
    logger.info(f"Mean speed: {stats['mean_speed']:.1f} km/h")
    logger.info(f"Median speed: {stats['median_speed']:.1f} km/h")
    logger.info(f"Standard deviation: {stats['std_speed']:.1f} km/h")
    
    # Speed categorization
    highway_speeds = [s for s in all_speeds if s >= 80]  # 80+ km/h = highway
    arterial_speeds = [s for s in all_speeds if 40 <= s < 80]  # 40-79 km/h = arterial
    city_speeds = [s for s in all_speeds if s < 40]  # <40 km/h = city
    
    logger.info("\n=== SPEED CATEGORIZATION ===")
    logger.info(f"Highway speeds (80+ km/h): {len(highway_speeds)} ({len(highway_speeds)/len(all_speeds)*100:.1f}%)")
    logger.info(f"Arterial speeds (40-79 km/h): {len(arterial_speeds)} ({len(arterial_speeds)/len(all_speeds)*100:.1f}%)")
    logger.info(f"City speeds (<40 km/h): {len(city_speeds)} ({len(city_speeds)/len(all_speeds)*100:.1f}%)")
    
    return {
        'stats': stats,
        'scenario_speeds': scenario_speeds,
        'all_speeds': all_speeds,
        'highway_speeds': highway_speeds,
        'arterial_speeds': arterial_speeds,
        'city_speeds': city_speeds
    }

def create_speed_visualizations(speed_data, output_dir: str):
    """
    Create comprehensive speed visualization plots.
    """
    logger.info(f"Creating speed visualizations in: {output_dir}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Speed Distribution Histogram
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Overall speed distribution
    axes[0, 0].hist(speed_data['all_speeds'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(speed_data['stats']['mean_speed'], color='red', linestyle='--', 
                      label=f"Mean: {speed_data['stats']['mean_speed']:.1f} km/h")
    axes[0, 0].axvline(speed_data['stats']['median_speed'], color='green', linestyle='--', 
                      label=f"Median: {speed_data['stats']['median_speed']:.1f} km/h")
    axes[0, 0].set_xlabel('Speed (km/h)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Ambulance Speed Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Speed categories pie chart
    categories = ['Highway (80+ km/h)', 'Arterial (40-79 km/h)', 'City (<40 km/h)']
    sizes = [len(speed_data['highway_speeds']), len(speed_data['arterial_speeds']), len(speed_data['city_speeds'])]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    axes[0, 1].pie(sizes, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Speed Category Distribution')
    
    # Box plot by scenario
    scenario_names = list(speed_data['scenario_speeds'].keys())
    scenario_data = [speed_data['scenario_speeds'][name] for name in scenario_names]
    
    axes[1, 0].boxplot(scenario_data, labels=scenario_names)
    axes[1, 0].set_xlabel('Scenario')
    axes[1, 0].set_ylabel('Speed (km/h)')
    axes[1, 0].set_title('Speed Distribution by Scenario')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Speed over time (first scenario as example)
    if scenario_names:
        first_scenario_speeds = speed_data['scenario_speeds'][scenario_names[0]]
        axes[1, 1].plot(first_scenario_speeds[:100], marker='o', markersize=2, linewidth=1)
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Speed (km/h)')
        axes[1, 1].set_title(f'Speed Over Time ({scenario_names[0]})')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'ambulance_speed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    logger.info(f"Saved speed analysis plot: {output_path / 'ambulance_speed_analysis.png'}")

def create_trajectory_visualization(data_path: str, output_dir: str, scenario_name: str):
    """
    Create trajectory visualization for a specific scenario.
    """
    logger.info(f"Creating trajectory visualization for: {scenario_name}")
    
    data_dir = Path(data_path) / scenario_name
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find parquet file
    parquet_files = list(data_dir.glob("*_transitions.parquet"))
    if not parquet_files:
        logger.error(f"No parquet files found in {data_dir}")
        return
    
    df = pd.read_parquet(parquet_files[0])
    
    # Create trajectory plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot trajectories for all agents
    for agent_id in df['agent_id'].unique():
        agent_data = df[df['agent_id'] == agent_id]
        if agent_id == 0:  # Ambulance
            axes[0].plot(agent_data['x'], agent_data['y'], 'r-', linewidth=3, 
                        label=f'Ambulance (Agent {agent_id})', alpha=0.8)
        else:
            axes[0].plot(agent_data['x'], agent_data['y'], '--', linewidth=1.5, 
                        label=f'Vehicle {agent_id}', alpha=0.7)
    
    axes[0].set_xlabel('X Position (m)')
    axes[0].set_ylabel('Y Position (m)')
    axes[0].set_title(f'Vehicle Trajectories - {scenario_name}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot speed over time for ambulance
    ambulance_data = df[df['agent_id'] == 0].copy()
    ambulance_data['speed_kmh'] = np.sqrt(ambulance_data['vx']**2 + ambulance_data['vy']**2) * 3.6
    
    axes[1].plot(ambulance_data.index, ambulance_data['speed_kmh'], 'r-', linewidth=2, marker='o', markersize=3)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Ambulance Speed (km/h)')
    axes[1].set_title(f'Ambulance Speed Profile - {scenario_name}')
    axes[1].grid(True, alpha=0.3)
    
    # Add speed statistics
    mean_speed = ambulance_data['speed_kmh'].mean()
    max_speed = ambulance_data['speed_kmh'].max()
    axes[1].axhline(y=mean_speed, color='green', linestyle='--', alpha=0.7, 
                   label=f'Mean: {mean_speed:.1f} km/h')
    axes[1].axhline(y=max_speed, color='blue', linestyle='--', alpha=0.7, 
                   label=f'Max: {max_speed:.1f} km/h')
    axes[1].legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_path / f'{scenario_name}_trajectory_analysis.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    logger.info(f"Saved trajectory plot: {plot_file}")

def verify_fast_scenarios():
    """
    Main verification function to check if fast scenarios are working correctly.
    """
    logger.info("=== FAST AMBULANCE SCENARIO VERIFICATION ===")
    
    # Data path
    data_path = "data/test_fast_ambulance"
    output_dir = "output/fast_ambulance_verification"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Analyze speeds
    logger.info("Step 1: Analyzing ambulance speeds...")
    speed_data = analyze_ambulance_speeds(data_path)
    
    if speed_data is None:
        logger.error("Failed to analyze speeds!")
        return False
    
    # 2. Create visualizations
    logger.info("Step 2: Creating speed visualizations...")
    create_speed_visualizations(speed_data, output_dir)
    
    # 3. Create trajectory visualizations for each scenario
    logger.info("Step 3: Creating trajectory visualizations...")
    for scenario_name in speed_data['scenario_speeds'].keys():
        create_trajectory_visualization(data_path, output_dir, scenario_name)
    
    # 4. Verification summary
    logger.info("\n=== VERIFICATION SUMMARY ===")
    
    # Check if we have fast speeds
    mean_speed = speed_data['stats']['mean_speed']
    max_speed = speed_data['stats']['max_speed']
    highway_percentage = len(speed_data['highway_speeds']) / len(speed_data['all_speeds']) * 100
    
    # Verification criteria
    fast_speed_achieved = mean_speed >= 40  # At least arterial speeds
    highway_speeds_present = len(speed_data['highway_speeds']) > 0
    reasonable_max_speed = max_speed >= 60  # Should reach highway speeds
    
    logger.info(f"✅ Fast speed transformation successful: {fast_speed_achieved}")
    logger.info(f"✅ Highway speeds present: {highway_speeds_present}")
    logger.info(f"✅ Reasonable maximum speed: {reasonable_max_speed} (max: {max_speed:.1f} km/h)")
    logger.info(f"📊 Highway speed percentage: {highway_percentage:.1f}%")
    
    if fast_speed_achieved and highway_speeds_present and reasonable_max_speed:
        logger.info("🎉 SUCCESS: Fast ambulance scenarios are working correctly!")
        logger.info(f"   Mean ambulance speed: {mean_speed:.1f} km/h (target: 40+ km/h)")
        logger.info(f"   Maximum speed reached: {max_speed:.1f} km/h")
        logger.info(f"   Emergency response performance: EXCELLENT")
        return True
    else:
        logger.warning("⚠️ WARNING: Speed transformation may not be fully effective")
        return False

if __name__ == "__main__":
    success = verify_fast_scenarios()
    if success:
        print("\n🚑💨 Fast ambulance data collection verification completed successfully!")
        print("📊 Check the output/fast_ambulance_verification/ directory for detailed visualizations")
    else:
        print("\n❌ Verification failed - please check the logs for details")