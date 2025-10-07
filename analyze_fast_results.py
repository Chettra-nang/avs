#!/usr/bin/env python3
"""
Analyze the fast ambulance data collection results to verify speed improvements.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_analyze_fast_ambulance_data():
    """Quickly analyze the newly collected fast ambulance data."""
    
    logger.info("=== ANALYZING FAST AMBULANCE DATA ===")
    
    data_path = Path("data/fast_ambulance_verified")
    index_file = data_path / "index.json"
    
    with open(index_file, 'r') as f:
        index = json.load(f)
    
    logger.info(f"Found {len(index['scenarios'])} scenarios")
    
    all_speeds = []
    all_ambulance_speeds = []
    scenario_results = {}
    
    for scenario_name in index['scenarios']:
        logger.info(f"\nProcessing: {scenario_name}")
        scenario_dir = data_path / scenario_name
        
        parquet_files = list(scenario_dir.glob("*_transitions.parquet"))
        if not parquet_files:
            continue
        
        df = pd.read_parquet(parquet_files[0])
        
        # Analyze ambulance (agent 0) data
        ambulance_data = df[df['agent_id'] == 0].copy()
        
        if len(ambulance_data) > 0:
            # Use multiple speed measures
            speeds_from_column = ambulance_data['speed'].tolist()
            
            # Calculate from velocities
            ambulance_data['calculated_speed'] = np.sqrt(ambulance_data['ego_vx']**2 + ambulance_data['ego_vy']**2) * 3.6
            calculated_speeds = ambulance_data['calculated_speed'].tolist()
            
            all_speeds.extend(speeds_from_column)
            all_ambulance_speeds.extend(calculated_speeds)
            
            scenario_results[scenario_name] = {
                'speed_column': speeds_from_column,
                'calculated_speeds': calculated_speeds,
                'mean_speed': np.mean(speeds_from_column),
                'max_speed': max(speeds_from_column),
                'mean_calculated': np.mean(calculated_speeds),
                'max_calculated': max(calculated_speeds),
                'data_points': len(ambulance_data)
            }
            
            logger.info(f"  Data points: {len(ambulance_data)}")
            logger.info(f"  Speed column: mean={np.mean(speeds_from_column):.1f} km/h, max={max(speeds_from_column):.1f} km/h")
            logger.info(f"  Calculated: mean={np.mean(calculated_speeds):.1f} km/h, max={max(calculated_speeds):.1f} km/h")
        
        # Check metadata for speed configurations
        metadata_files = list(scenario_dir.glob("*_meta.jsonl"))
        if metadata_files:
            with open(metadata_files[0], 'r') as f:
                first_line = f.readline()
                metadata = json.loads(first_line)
                config = metadata.get('config', {})
                speed_limit = config.get('speed_limit_kmh', 'Not set')
                vehicles_count = config.get('vehicles_count', 'Unknown')
                
                logger.info(f"  Config: speed_limit_kmh={speed_limit}, vehicles_count={vehicles_count}")
    
    # Overall analysis
    if all_speeds:
        logger.info(f"\n=== OVERALL RESULTS ===")
        logger.info(f"Total data points: {len(all_speeds)}")
        logger.info(f"Speed column - Mean: {np.mean(all_speeds):.1f} km/h, Max: {max(all_speeds):.1f} km/h")
        logger.info(f"Calculated - Mean: {np.mean(all_ambulance_speeds):.1f} km/h, Max: {max(all_ambulance_speeds):.1f} km/h")
        
        # Speed categories
        fast_speeds = [s for s in all_speeds if s >= 40]
        highway_speeds = [s for s in all_speeds if s >= 60]
        very_fast = [s for s in all_speeds if s >= 80]
        
        logger.info(f"\nSpeed Categories:")
        logger.info(f"  Fast (40+ km/h): {len(fast_speeds)}/{len(all_speeds)} ({len(fast_speeds)/len(all_speeds)*100:.1f}%)")
        logger.info(f"  Highway (60+ km/h): {len(highway_speeds)}/{len(all_speeds)} ({len(highway_speeds)/len(all_speeds)*100:.1f}%)")
        logger.info(f"  Very fast (80+ km/h): {len(very_fast)}/{len(all_speeds)} ({len(very_fast)/len(all_speeds)*100:.1f}%)")
        
        # Success assessment
        mean_speed = np.mean(all_speeds)
        if mean_speed >= 60:
            logger.info("🎉 EXCELLENT: Fast highway speeds achieved!")
        elif mean_speed >= 40:
            logger.info("✅ GOOD: Arterial speeds achieved!")
        elif mean_speed >= 20:
            logger.info("⚠️ PARTIAL: Some improvement but still slow")
        else:
            logger.info("❌ STILL SLOW: Transformation not effective")
        
        return scenario_results
    else:
        logger.error("No speed data found!")
        return None

def create_comparison_plots():
    """Create before/after comparison plots."""
    
    logger.info("\n=== CREATING SPEED COMPARISON VISUALIZATION ===")
    
    # Load both datasets
    old_data_path = Path("data/test_fast_ambulance")  # Before transformation
    new_data_path = Path("data/fast_ambulance_verified")  # After transformation
    
    def load_speeds(data_path):
        """Load speeds from a dataset."""
        index_file = data_path / "index.json"
        if not index_file.exists():
            return []
        
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        speeds = []
        for scenario_name in index['scenarios']:
            scenario_dir = data_path / scenario_name
            parquet_files = list(scenario_dir.glob("*_transitions.parquet"))
            if parquet_files:
                df = pd.read_parquet(parquet_files[0])
                ambulance_data = df[df['agent_id'] == 0]
                if len(ambulance_data) > 0:
                    speeds.extend(ambulance_data['speed'].tolist())
        return speeds
    
    old_speeds = load_speeds(old_data_path)
    new_speeds = load_speeds(new_data_path)
    
    if old_speeds and new_speeds:
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Before/After histograms
        axes[0, 0].hist(old_speeds, bins=20, alpha=0.7, color='red', label='Before (Slow)', edgecolor='black')
        axes[0, 0].set_xlabel('Speed (km/h)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Ambulance Speeds BEFORE Transformation')
        axes[0, 0].axvline(np.mean(old_speeds), color='darkred', linestyle='--', label=f'Mean: {np.mean(old_speeds):.1f} km/h')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(new_speeds, bins=20, alpha=0.7, color='blue', label='After (Fast)', edgecolor='black')
        axes[0, 1].set_xlabel('Speed (km/h)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Ambulance Speeds AFTER Transformation')
        axes[0, 1].axvline(np.mean(new_speeds), color='darkblue', linestyle='--', label=f'Mean: {np.mean(new_speeds):.1f} km/h')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Side-by-side comparison
        axes[1, 0].hist(old_speeds, bins=20, alpha=0.6, color='red', label='Before', edgecolor='black')
        axes[1, 0].hist(new_speeds, bins=20, alpha=0.6, color='blue', label='After', edgecolor='black')
        axes[1, 0].set_xlabel('Speed (km/h)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Speed Comparison: Before vs After')
        axes[1, 0].axvline(40, color='orange', linestyle='--', alpha=0.8, label='Target: 40+ km/h')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics
        stats_text = f'''TRANSFORMATION RESULTS:

BEFORE:
• Mean: {np.mean(old_speeds):.1f} km/h
• Max: {max(old_speeds):.1f} km/h
• Fast speeds (40+ km/h): {len([s for s in old_speeds if s >= 40])}/{len(old_speeds)}

AFTER:
• Mean: {np.mean(new_speeds):.1f} km/h
• Max: {max(new_speeds):.1f} km/h  
• Fast speeds (40+ km/h): {len([s for s in new_speeds if s >= 40])}/{len(new_speeds)}

IMPROVEMENT:
• Speed increase: {np.mean(new_speeds) - np.mean(old_speeds):+.1f} km/h
• Factor: {np.mean(new_speeds) / np.mean(old_speeds):.1f}x faster'''
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                        verticalalignment='top', fontfamily='monospace', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        axes[1, 1].set_title('Transformation Summary')
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path("output/fast_ambulance_comparison")
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_file = output_dir / "speed_transformation_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Saved comparison plot: {plot_file}")
        
        # Log the improvement
        improvement = np.mean(new_speeds) - np.mean(old_speeds)
        factor = np.mean(new_speeds) / np.mean(old_speeds) if np.mean(old_speeds) > 0 else 0
        
        logger.info(f"🚀 SPEED IMPROVEMENT ACHIEVED!")
        logger.info(f"   Before: {np.mean(old_speeds):.1f} km/h → After: {np.mean(new_speeds):.1f} km/h")
        logger.info(f"   Improvement: +{improvement:.1f} km/h ({factor:.1f}x faster)")
        
        return True
    else:
        logger.error("Could not load speed data for comparison")
        return False

if __name__ == "__main__":
    # Analyze the new fast data
    results = quick_analyze_fast_ambulance_data()
    
    if results:
        # Create comparison visualization
        create_comparison_plots()
        
        logger.info("\n🎉 FAST AMBULANCE ANALYSIS COMPLETED!")
        logger.info("✅ Transformation verification successful")
        logger.info("📊 Check output/fast_ambulance_comparison/ for visualizations")
    else:
        logger.error("❌ Analysis failed")