#!/usr/bin/env python3
"""
Ambulance Dataset Video Generator and Speed Analysis

This script creates comprehensive analysis of the ambulance dataset including:
1. Speed statistics analysis across all scenarios
2. Scenario visualization videos
3. Real-time scenario playback with speed information
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime
import glob
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AmbulanceDatasetProcessor:
    def __init__(self, dataset_path: str):
        """Initialize the processor with the dataset path."""
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path("d:/Research_ITC/avs_folder/avs/output/dataset_videos")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.parquet_files = []
        self.scenario_data = {}
        self.speed_statistics = {}
        
    def find_parquet_files(self) -> List[Path]:
        """Find all parquet files in the dataset."""
        pattern = "**/*.parquet"
        self.parquet_files = list(self.dataset_path.glob(pattern))
        logger.info(f"Found {len(self.parquet_files)} parquet files")
        return self.parquet_files
    
    def extract_scenario_name(self, file_path: Path) -> str:
        """Extract scenario name from file path."""
        parts = file_path.parts
        for i, part in enumerate(parts):
            if part.startswith('batch_'):
                if i + 1 < len(parts):
                    return parts[i + 1]
        return "unknown_scenario"
    
    def analyze_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single parquet file for speed and movement data."""
        try:
            df = pd.read_parquet(file_path)
            scenario_name = self.extract_scenario_name(file_path)
            
            logger.info(f"Analyzing {scenario_name}: {len(df)} timesteps")
            
            analysis = {
                'scenario_name': scenario_name,
                'file_path': str(file_path),
                'timesteps': len(df),
                'columns': list(df.columns),
                'speed_data': [],
                'ambulance_speeds': [],
                'npc_speeds': [],
                'positions': [],
                'actions': []
            }
            
            # Extract speed information
            if 'speed' in df.columns:
                speeds = df['speed'].dropna()
                analysis['speed_data'] = speeds.tolist()
                analysis['avg_speed'] = float(speeds.mean()) if len(speeds) > 0 else 0
                analysis['max_speed'] = float(speeds.max()) if len(speeds) > 0 else 0
                analysis['min_speed'] = float(speeds.min()) if len(speeds) > 0 else 0
            
            # Separate ambulance vs regular vehicle speeds
            if 'ambulance_agent_index' in df.columns and 'agent_id' in df.columns and 'speed' in df.columns:
                ambulance_data = df[df['agent_id'] == 0]  # Assuming ambulance is agent 0
                npc_data = df[df['agent_id'] != 0]
                
                if len(ambulance_data) > 0:
                    amb_speeds = ambulance_data['speed'].dropna()
                    analysis['ambulance_speeds'] = amb_speeds.tolist()
                    analysis['ambulance_avg_speed'] = float(amb_speeds.mean()) if len(amb_speeds) > 0 else 0
                    analysis['ambulance_max_speed'] = float(amb_speeds.max()) if len(amb_speeds) > 0 else 0
                
                if len(npc_data) > 0:
                    npc_speeds = npc_data['speed'].dropna()
                    analysis['npc_speeds'] = npc_speeds.tolist()
                    analysis['npc_avg_speed'] = float(npc_speeds.mean()) if len(npc_speeds) > 0 else 0
                    analysis['npc_max_speed'] = float(npc_speeds.max()) if len(npc_speeds) > 0 else 0
            
            # Extract position data if available
            pos_columns = [col for col in df.columns if any(x in col.lower() for x in ['x', 'y', 'position'])]
            if pos_columns:
                analysis['position_columns'] = pos_columns
                # Sample some position data for visualization
                sample_size = min(100, len(df))
                sample_df = df.head(sample_size)
                analysis['sample_positions'] = []
                for _, row in sample_df.iterrows():
                    pos_data = {}
                    for col in pos_columns:
                        if not pd.isna(row[col]):
                            pos_data[col] = float(row[col])
                    if pos_data:
                        analysis['sample_positions'].append(pos_data)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {'error': str(e), 'file_path': str(file_path)}
    
    def create_scenario_video(self, analysis_data: Dict[str, Any]) -> str:
        """Create a video visualization for a scenario."""
        try:
            scenario_name = analysis_data['scenario_name']
            logger.info(f"Creating video for {scenario_name}")
            
            # Set up the figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Upper plot: Speed over time
            ax1.set_title(f'Speed Analysis: {scenario_name}', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Timestep')
            ax1.set_ylabel('Speed (m/s)')
            ax1.grid(True, alpha=0.3)
            
            # Lower plot: Vehicle positions (if available)
            ax2.set_title('Vehicle Movement Pattern', fontsize=12)
            ax2.set_xlabel('X Position (m)')
            ax2.set_ylabel('Y Position (m)')
            ax2.grid(True, alpha=0.3)
            
            # Plot speed data
            if 'speed_data' in analysis_data and analysis_data['speed_data']:
                speed_data = analysis_data['speed_data']
                timesteps = range(len(speed_data))
                ax1.plot(timesteps, speed_data, 'b-', alpha=0.7, label='All Vehicles')
            
            if 'ambulance_speeds' in analysis_data and analysis_data['ambulance_speeds']:
                amb_speeds = analysis_data['ambulance_speeds']
                timesteps = range(len(amb_speeds))
                ax1.plot(timesteps, amb_speeds, 'r-', linewidth=2, label='Ambulance')
            
            if 'npc_speeds' in analysis_data and analysis_data['npc_speeds']:
                npc_speeds = analysis_data['npc_speeds']
                timesteps = range(len(npc_speeds))
                ax1.plot(timesteps, npc_speeds, 'g-', alpha=0.5, label='NPCs')
            
            ax1.legend()
            
            # Add speed statistics as text
            speed_stats_text = []
            if 'avg_speed' in analysis_data:
                speed_stats_text.append(f"Avg Speed: {analysis_data['avg_speed']:.2f} m/s")
            if 'max_speed' in analysis_data:
                speed_stats_text.append(f"Max Speed: {analysis_data['max_speed']:.2f} m/s")
            if 'ambulance_avg_speed' in analysis_data:
                speed_stats_text.append(f"Ambulance Avg: {analysis_data['ambulance_avg_speed']:.2f} m/s")
            if 'ambulance_max_speed' in analysis_data:
                speed_stats_text.append(f"Ambulance Max: {analysis_data['ambulance_max_speed']:.2f} m/s")
            
            if speed_stats_text:
                ax1.text(0.02, 0.98, '\n'.join(speed_stats_text), 
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Plot position data if available
            if 'sample_positions' in analysis_data and analysis_data['sample_positions']:
                positions = analysis_data['sample_positions']
                # Try to extract x, y coordinates
                x_coords = []
                y_coords = []
                
                for pos in positions:
                    # Look for x, y coordinates
                    x_val = None
                    y_val = None
                    for key, val in pos.items():
                        if 'x' in key.lower() and x_val is None:
                            x_val = val
                        elif 'y' in key.lower() and y_val is None:
                            y_val = val
                    
                    if x_val is not None and y_val is not None:
                        x_coords.append(x_val)
                        y_coords.append(y_val)
                
                if x_coords and y_coords:
                    ax2.plot(x_coords, y_coords, 'b-', alpha=0.7, label='Vehicle Path')
                    ax2.scatter(x_coords[0], y_coords[0], c='green', s=100, label='Start', marker='o')
                    ax2.scatter(x_coords[-1], y_coords[-1], c='red', s=100, label='End', marker='s')
                    ax2.legend()
                else:
                    ax2.text(0.5, 0.5, 'No position data available', 
                            transform=ax2.transAxes, ha='center', va='center')
            else:
                ax2.text(0.5, 0.5, 'No position data available', 
                        transform=ax2.transAxes, ha='center', va='center')
            
            plt.tight_layout()
            
            # Save the plot
            video_path = self.output_dir / f"{scenario_name}_analysis.png"
            plt.savefig(video_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Analysis plot saved: {video_path}")
            return str(video_path)
            
        except Exception as e:
            logger.error(f"Error creating video for {analysis_data.get('scenario_name', 'unknown')}: {e}")
            return ""
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report."""
        logger.info("=== Starting Comprehensive Ambulance Dataset Analysis ===")
        
        # Find all parquet files
        parquet_files = self.find_parquet_files()
        if not parquet_files:
            logger.error("No parquet files found!")
            return
        
        all_analyses = []
        scenario_stats = {}
        
        # Process each file
        for i, file_path in enumerate(parquet_files):
            logger.info(f"\nProcessing {i+1}/{len(parquet_files)}: {file_path.name}")
            
            analysis = self.analyze_single_file(file_path)
            if 'error' not in analysis:
                all_analyses.append(analysis)
                
                scenario_name = analysis['scenario_name']
                if scenario_name not in scenario_stats:
                    scenario_stats[scenario_name] = []
                scenario_stats[scenario_name].append(analysis)
                
                # Create visualization
                self.create_scenario_video(analysis)
        
        # Generate summary statistics
        self.generate_summary_report(all_analyses, scenario_stats)
        
        logger.info(f"\n✅ Analysis complete! Results saved to: {self.output_dir}")
        
    def generate_summary_report(self, all_analyses: List[Dict], scenario_stats: Dict):
        """Generate summary report with speed statistics."""
        
        report_path = self.output_dir / "ambulance_dataset_speed_analysis.md"
        
        with open(report_path, 'w') as f:
            f.write("# Ambulance Dataset Comprehensive Speed Analysis\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Files Processed:** {len(all_analyses)}\n")
            f.write(f"**Unique Scenarios:** {len(scenario_stats)}\n\n")
            
            # Overall speed statistics
            f.write("## Overall Speed Statistics\n\n")
            
            all_speeds = []
            all_amb_speeds = []
            all_npc_speeds = []
            
            for analysis in all_analyses:
                if 'speed_data' in analysis:
                    all_speeds.extend(analysis['speed_data'])
                if 'ambulance_speeds' in analysis:
                    all_amb_speeds.extend(analysis['ambulance_speeds'])
                if 'npc_speeds' in analysis:
                    all_npc_speeds.extend(analysis['npc_speeds'])
            
            if all_speeds:
                f.write(f"- **Total Speed Measurements:** {len(all_speeds)}\n")
                f.write(f"- **Average Speed (All Vehicles):** {np.mean(all_speeds):.2f} m/s ({np.mean(all_speeds)*3.6:.2f} km/h)\n")
                f.write(f"- **Maximum Speed (All Vehicles):** {np.max(all_speeds):.2f} m/s ({np.max(all_speeds)*3.6:.2f} km/h)\n")
                f.write(f"- **Minimum Speed (All Vehicles):** {np.min(all_speeds):.2f} m/s ({np.min(all_speeds)*3.6:.2f} km/h)\n")
            
            if all_amb_speeds:
                f.write(f"\n### Ambulance Speed Statistics\n")
                f.write(f"- **Total Ambulance Measurements:** {len(all_amb_speeds)}\n")
                f.write(f"- **Average Ambulance Speed:** {np.mean(all_amb_speeds):.2f} m/s ({np.mean(all_amb_speeds)*3.6:.2f} km/h)\n")
                f.write(f"- **Maximum Ambulance Speed:** {np.max(all_amb_speeds):.2f} m/s ({np.max(all_amb_speeds)*3.6:.2f} km/h)\n")
                f.write(f"- **Minimum Ambulance Speed:** {np.min(all_amb_speeds):.2f} m/s ({np.min(all_amb_speeds)*3.6:.2f} km/h)\n")
            
            if all_npc_speeds:
                f.write(f"\n### NPC Vehicle Speed Statistics\n")
                f.write(f"- **Total NPC Measurements:** {len(all_npc_speeds)}\n")
                f.write(f"- **Average NPC Speed:** {np.mean(all_npc_speeds):.2f} m/s ({np.mean(all_npc_speeds)*3.6:.2f} km/h)\n")
                f.write(f"- **Maximum NPC Speed:** {np.max(all_npc_speeds):.2f} m/s ({np.max(all_npc_speeds)*3.6:.2f} km/h)\n")
                f.write(f"- **Minimum NPC Speed:** {np.min(all_npc_speeds):.2f} m/s ({np.min(all_npc_speeds)*3.6:.2f} km/h)\n")
            
            # Scenario breakdown
            f.write(f"\n## Speed Analysis by Scenario\n\n")
            f.write("| Scenario | Episodes | Avg Speed (m/s) | Max Speed (m/s) | Ambulance Avg (m/s) | Ambulance Max (m/s) |\n")
            f.write("|----------|----------|-----------------|-----------------|---------------------|---------------------|\n")
            
            for scenario_name, analyses in scenario_stats.items():
                episode_count = len(analyses)
                
                # Calculate scenario averages
                scenario_speeds = []
                scenario_amb_speeds = []
                max_speeds = []
                max_amb_speeds = []
                
                for analysis in analyses:
                    if 'avg_speed' in analysis and analysis['avg_speed'] > 0:
                        scenario_speeds.append(analysis['avg_speed'])
                    if 'max_speed' in analysis and analysis['max_speed'] > 0:
                        max_speeds.append(analysis['max_speed'])
                    if 'ambulance_avg_speed' in analysis and analysis['ambulance_avg_speed'] > 0:
                        scenario_amb_speeds.append(analysis['ambulance_avg_speed'])
                    if 'ambulance_max_speed' in analysis and analysis['ambulance_max_speed'] > 0:
                        max_amb_speeds.append(analysis['ambulance_max_speed'])
                
                avg_speed = np.mean(scenario_speeds) if scenario_speeds else 0
                max_speed = np.max(max_speeds) if max_speeds else 0
                avg_amb_speed = np.mean(scenario_amb_speeds) if scenario_amb_speeds else 0
                max_amb_speed = np.max(max_amb_speeds) if max_amb_speeds else 0
                
                f.write(f"| {scenario_name} | {episode_count} | {avg_speed:.2f} | {max_speed:.2f} | {avg_amb_speed:.2f} | {max_amb_speed:.2f} |\n")
            
            f.write("\n## Key Findings\n\n")
            
            if all_amb_speeds and all_npc_speeds:
                amb_avg = np.mean(all_amb_speeds)
                npc_avg = np.mean(all_npc_speeds)
                speed_diff = amb_avg - npc_avg
                
                f.write(f"- **Ambulance vs NPC Speed Difference:** {speed_diff:.2f} m/s ({speed_diff*3.6:.2f} km/h)\n")
                f.write(f"- **Ambulance Speed Advantage:** {((amb_avg/npc_avg-1)*100):.1f}% faster than NPCs\n" if npc_avg > 0 else "")
            
            if scenario_stats:
                fastest_scenario = max(scenario_stats.keys(), 
                                     key=lambda x: np.mean([a.get('ambulance_avg_speed', 0) for a in scenario_stats[x]]))
                slowest_scenario = min(scenario_stats.keys(), 
                                     key=lambda x: np.mean([a.get('ambulance_avg_speed', 0) for a in scenario_stats[x]]))
                
                f.write(f"- **Fastest Scenario:** {fastest_scenario}\n")
                f.write(f"- **Slowest Scenario:** {slowest_scenario}\n")
            
            f.write("\n## Generated Visualizations\n\n")
            f.write("Individual scenario analysis plots have been generated for each scenario.\n")
            f.write("Check the output directory for detailed visualizations.\n")
        
        logger.info(f"Summary report saved: {report_path}")


def main():
    """Main execution function."""
    dataset_path = r"d:\Research_ITC\avs_folder\avs\data\ambulance_dataset_150_espisode_cpu_30_senario\ambulance_dataset_150_espisode_cpu_30_senario"
    
    processor = AmbulanceDatasetProcessor(dataset_path)
    processor.generate_comprehensive_report()


if __name__ == "__main__":
    main()