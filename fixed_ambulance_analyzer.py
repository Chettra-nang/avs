#!/usr/bin/env python3
"""
Fixed Ambulance Dataset Analyzer

This version properly handles the complex data structure of the ambulance dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Any
from datetime import datetime
import ast
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FixedAmbulanceAnalyzer:
    def __init__(self, dataset_path: str):
        """Initialize the analyzer with the dataset path."""
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path("d:/Research_ITC/avs_folder/avs/output/dataset_videos")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_speed_from_kinematics(self, kinematics_data) -> float:
        """Extract speed from kinematics data structure."""
        try:
            if isinstance(kinematics_data, str):
                # If it's a string representation, try to parse it
                if 'km/h' in kinematics_data:
                    # Extract speed from text like "Vehicle is stationary at 3.6 km/h"
                    match = re.search(r'(\d+\.?\d*)\s*km/h', kinematics_data)
                    if match:
                        speed_kmh = float(match.group(1))
                        return speed_kmh / 3.6  # Convert to m/s
                return 0.0
            
            elif isinstance(kinematics_data, (list, tuple)) and len(kinematics_data) >= 2:
                # If it's a list/tuple, assume velocity is at index 1
                velocity_data = kinematics_data[1]
                if isinstance(velocity_data, (list, tuple)) and len(velocity_data) >= 2:
                    vx, vy = velocity_data[0], velocity_data[1]
                    return np.sqrt(vx**2 + vy**2)  # Speed magnitude
            
            elif hasattr(kinematics_data, '__len__') and len(kinematics_data) > 0:
                # Try to extract numeric values
                if isinstance(kinematics_data[0], (int, float)):
                    return float(kinematics_data[0])
            
            return 0.0
        except Exception as e:
            logger.debug(f"Error extracting speed: {e}")
            return 0.0
    
    def extract_position_from_kinematics(self, kinematics_data) -> tuple:
        """Extract position from kinematics data structure."""
        try:
            if isinstance(kinematics_data, (list, tuple)) and len(kinematics_data) >= 1:
                position_data = kinematics_data[0]
                if isinstance(position_data, (list, tuple)) and len(position_data) >= 2:
                    return float(position_data[0]), float(position_data[1])
            return 0.0, 0.0
        except Exception:
            return 0.0, 0.0
    
    def analyze_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single parquet file."""
        try:
            df = pd.read_parquet(file_path)
            scenario_name = self.extract_scenario_name(file_path)
            
            logger.info(f"Analyzing {scenario_name}: {len(df)} rows, columns: {list(df.columns)}")
            
            analysis = {
                'scenario_name': scenario_name,
                'file_path': str(file_path),
                'timesteps': len(df),
                'columns': list(df.columns),
                'speeds': [],
                'positions': [],
                'ambulance_data': [],
                'npc_data': []
            }
            
            # Process each row
            speeds_all = []
            ambulance_speeds = []
            npc_speeds = []
            positions = []
            
            for idx, row in df.iterrows():
                if idx % 1000 == 0:
                    logger.info(f"Processing row {idx}/{len(df)}")
                
                # Extract speed
                speed = 0.0
                if 'speed' in row and pd.notna(row['speed']):
                    if isinstance(row['speed'], str):
                        speed = self.extract_speed_from_kinematics(row['speed'])
                    else:
                        try:
                            speed = float(row['speed'])
                        except (ValueError, TypeError):
                            speed = 0.0
                
                # Try to extract from kinematics_raw if speed column didn't work
                if speed == 0.0 and 'kinematics_raw' in row and pd.notna(row['kinematics_raw']):
                    kinematics = row['kinematics_raw']
                    speed = self.extract_speed_from_kinematics(kinematics)
                
                speeds_all.append(speed)
                
                # Extract position
                pos_x, pos_y = 0.0, 0.0
                if 'kinematics_raw' in row and pd.notna(row['kinematics_raw']):
                    pos_x, pos_y = self.extract_position_from_kinematics(row['kinematics_raw'])
                
                positions.append((pos_x, pos_y))
                
                # Separate ambulance vs NPC data
                agent_id = row.get('agent_id', 0)
                if agent_id == 0:  # Ambulance
                    ambulance_speeds.append(speed)
                    analysis['ambulance_data'].append({
                        'step': idx,
                        'speed': speed,
                        'position': (pos_x, pos_y),
                        'agent_id': agent_id
                    })
                else:  # NPC vehicles
                    npc_speeds.append(speed)
                    analysis['npc_data'].append({
                        'step': idx,
                        'speed': speed,
                        'position': (pos_x, pos_y),
                        'agent_id': agent_id
                    })
            
            # Calculate statistics
            analysis['speeds'] = speeds_all
            analysis['positions'] = positions
            
            if speeds_all:
                valid_speeds = [s for s in speeds_all if s > 0]
                if valid_speeds:
                    analysis['avg_speed'] = np.mean(valid_speeds)
                    analysis['max_speed'] = np.max(valid_speeds)
                    analysis['min_speed'] = np.min(valid_speeds)
                else:
                    analysis['avg_speed'] = analysis['max_speed'] = analysis['min_speed'] = 0.0
            
            if ambulance_speeds:
                valid_amb_speeds = [s for s in ambulance_speeds if s > 0]
                if valid_amb_speeds:
                    analysis['ambulance_avg_speed'] = np.mean(valid_amb_speeds)
                    analysis['ambulance_max_speed'] = np.max(valid_amb_speeds)
                    analysis['ambulance_count'] = len(valid_amb_speeds)
                else:
                    analysis['ambulance_avg_speed'] = analysis['ambulance_max_speed'] = 0.0
                    analysis['ambulance_count'] = 0
            
            if npc_speeds:
                valid_npc_speeds = [s for s in npc_speeds if s > 0]
                if valid_npc_speeds:
                    analysis['npc_avg_speed'] = np.mean(valid_npc_speeds)
                    analysis['npc_max_speed'] = np.max(valid_npc_speeds)
                    analysis['npc_count'] = len(valid_npc_speeds)
                else:
                    analysis['npc_avg_speed'] = analysis['npc_max_speed'] = 0.0
                    analysis['npc_count'] = 0
            
            logger.info(f"✅ {scenario_name}: Avg={analysis.get('avg_speed', 0):.2f}m/s, "
                       f"Ambulance Avg={analysis.get('ambulance_avg_speed', 0):.2f}m/s")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return {'error': str(e), 'file_path': str(file_path), 'scenario_name': self.extract_scenario_name(file_path)}
    
    def extract_scenario_name(self, file_path: Path) -> str:
        """Extract scenario name from file path."""
        parts = file_path.parts
        for i, part in enumerate(parts):
            if part.startswith('batch_'):
                if i + 1 < len(parts):
                    return parts[i + 1]
        return "unknown_scenario"
    
    def create_scenario_visualization(self, analysis_data: Dict[str, Any]) -> str:
        """Create visualization for a scenario."""
        try:
            scenario_name = analysis_data['scenario_name']
            
            if 'error' in analysis_data:
                logger.warning(f"Skipping visualization for {scenario_name} due to error")
                return ""
            
            logger.info(f"Creating visualization for {scenario_name}")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Speed over time
            ax1.set_title(f'Speed Analysis: {scenario_name}', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Timestep')
            ax1.set_ylabel('Speed (m/s)')
            ax1.grid(True, alpha=0.3)
            
            speeds = analysis_data.get('speeds', [])
            if speeds:
                valid_speeds = [(i, s) for i, s in enumerate(speeds) if s > 0]
                if valid_speeds:
                    steps, speed_vals = zip(*valid_speeds)
                    ax1.plot(steps, speed_vals, 'b-', alpha=0.7, label='All Vehicles')
            
            # Ambulance vs NPC comparison
            ambulance_data = analysis_data.get('ambulance_data', [])
            npc_data = analysis_data.get('npc_data', [])
            
            if ambulance_data:
                amb_steps = [d['step'] for d in ambulance_data if d['speed'] > 0]
                amb_speeds = [d['speed'] for d in ambulance_data if d['speed'] > 0]
                if amb_steps and amb_speeds:
                    ax1.plot(amb_steps, amb_speeds, 'r-', linewidth=2, label='Ambulance', alpha=0.8)
            
            if npc_data:
                # Sample NPC data to avoid overcrowding
                npc_sample = npc_data[::max(1, len(npc_data)//100)]
                npc_steps = [d['step'] for d in npc_sample if d['speed'] > 0]
                npc_speeds = [d['speed'] for d in npc_sample if d['speed'] > 0]
                if npc_steps and npc_speeds:
                    ax1.plot(npc_steps, npc_speeds, 'g.', alpha=0.5, label='NPCs (sampled)')
            
            ax1.legend()
            
            # Speed distribution
            ax2.set_title('Speed Distribution')
            ax2.set_xlabel('Speed (m/s)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            if speeds:
                valid_speeds = [s for s in speeds if s > 0]
                if valid_speeds:
                    ax2.hist(valid_speeds, bins=30, alpha=0.7, edgecolor='black')
            
            # Vehicle trajectories
            ax3.set_title('Vehicle Movement Patterns')
            ax3.set_xlabel('X Position (m)')
            ax3.set_ylabel('Y Position (m)')
            ax3.grid(True, alpha=0.3)
            
            # Plot ambulance trajectory
            if ambulance_data:
                amb_x = [d['position'][0] for d in ambulance_data if d['position'][0] != 0 or d['position'][1] != 0]
                amb_y = [d['position'][1] for d in ambulance_data if d['position'][0] != 0 or d['position'][1] != 0]
                if amb_x and amb_y:
                    ax3.plot(amb_x, amb_y, 'r-', linewidth=3, label='Ambulance Path', alpha=0.8)
                    ax3.scatter(amb_x[0], amb_y[0], c='green', s=100, marker='o', label='Start')
                    ax3.scatter(amb_x[-1], amb_y[-1], c='red', s=100, marker='s', label='End')
            
            # Sample NPC trajectories
            if npc_data:
                unique_agents = set(d['agent_id'] for d in npc_data)
                colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_agents), 10)))
                
                for i, agent_id in enumerate(list(unique_agents)[:5]):  # Show max 5 NPC trajectories
                    agent_data = [d for d in npc_data if d['agent_id'] == agent_id]
                    npc_x = [d['position'][0] for d in agent_data if d['position'][0] != 0 or d['position'][1] != 0]
                    npc_y = [d['position'][1] for d in agent_data if d['position'][0] != 0 or d['position'][1] != 0]
                    if npc_x and npc_y:
                        ax3.plot(npc_x, npc_y, '--', color=colors[i], alpha=0.5, 
                                label=f'NPC {agent_id}' if i < 3 else '')
            
            ax3.legend()
            
            # Statistics summary
            ax4.axis('off')
            stats_text = []
            
            avg_speed = analysis_data.get('avg_speed', 0)
            max_speed = analysis_data.get('max_speed', 0)
            amb_avg = analysis_data.get('ambulance_avg_speed', 0)
            amb_max = analysis_data.get('ambulance_max_speed', 0)
            npc_avg = analysis_data.get('npc_avg_speed', 0)
            npc_max = analysis_data.get('npc_max_speed', 0)
            
            stats_text.append(f"📊 SPEED STATISTICS")
            stats_text.append(f"")
            stats_text.append(f"Overall:")
            stats_text.append(f"  • Average Speed: {avg_speed:.2f} m/s ({avg_speed*3.6:.1f} km/h)")
            stats_text.append(f"  • Maximum Speed: {max_speed:.2f} m/s ({max_speed*3.6:.1f} km/h)")
            stats_text.append(f"")
            stats_text.append(f"🚑 Ambulance:")
            stats_text.append(f"  • Average Speed: {amb_avg:.2f} m/s ({amb_avg*3.6:.1f} km/h)")
            stats_text.append(f"  • Maximum Speed: {amb_max:.2f} m/s ({amb_max*3.6:.1f} km/h)")
            stats_text.append(f"  • Data Points: {analysis_data.get('ambulance_count', 0)}")
            stats_text.append(f"")
            stats_text.append(f"🚗 NPCs:")
            stats_text.append(f"  • Average Speed: {npc_avg:.2f} m/s ({npc_avg*3.6:.1f} km/h)")
            stats_text.append(f"  • Maximum Speed: {npc_max:.2f} m/s ({npc_max*3.6:.1f} km/h)")
            stats_text.append(f"  • Data Points: {analysis_data.get('npc_count', 0)}")
            
            if amb_avg > 0 and npc_avg > 0:
                speed_advantage = ((amb_avg / npc_avg - 1) * 100)
                stats_text.append(f"")
                stats_text.append(f"📈 Performance:")
                stats_text.append(f"  • Ambulance Speed Advantage: {speed_advantage:.1f}%")
            
            ax4.text(0.1, 0.9, '\n'.join(stats_text), transform=ax4.transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            # Save the visualization
            viz_path = self.output_dir / f"{scenario_name}_detailed_analysis.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Visualization saved: {viz_path}")
            return str(viz_path)
            
        except Exception as e:
            logger.error(f"Error creating visualization for {analysis_data.get('scenario_name', 'unknown')}: {e}")
            return ""
    
    def run_analysis(self):
        """Run the complete analysis."""
        logger.info("🚑 === AMBULANCE DATASET COMPREHENSIVE ANALYSIS ===")
        
        # Find all parquet files
        parquet_files = list(self.dataset_path.glob("**/*.parquet"))
        if not parquet_files:
            logger.error("❌ No parquet files found!")
            return
        
        logger.info(f"📁 Found {len(parquet_files)} parquet files")
        
        all_analyses = []
        scenario_stats = {}
        
        # Process each file
        for i, file_path in enumerate(parquet_files):
            logger.info(f"\n🔄 Processing {i+1}/{len(parquet_files)}: {file_path.name}")
            
            analysis = self.analyze_single_file(file_path)
            all_analyses.append(analysis)
            
            if 'error' not in analysis:
                scenario_name = analysis['scenario_name']
                if scenario_name not in scenario_stats:
                    scenario_stats[scenario_name] = []
                scenario_stats[scenario_name].append(analysis)
                
                # Create visualization
                self.create_scenario_visualization(analysis)
        
        # Generate comprehensive report
        self.generate_comprehensive_report(all_analyses, scenario_stats)
        
        logger.info(f"\n✅ ANALYSIS COMPLETE! Results saved to: {self.output_dir}")
    
    def generate_comprehensive_report(self, all_analyses: List[Dict], scenario_stats: Dict):
        """Generate comprehensive markdown report."""
        
        report_path = self.output_dir / "AMBULANCE_COMPREHENSIVE_ANALYSIS_REPORT.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 🚑 Ambulance Dataset Comprehensive Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Dataset Path:** `{self.dataset_path}`\n")
            f.write(f"**Total Files Processed:** {len(all_analyses)}\n")
            f.write(f"**Unique Scenarios:** {len(scenario_stats)}\n\n")
            
            # Calculate overall statistics
            all_speeds = []
            all_amb_speeds = []
            all_npc_speeds = []
            total_timesteps = 0
            
            successful_analyses = [a for a in all_analyses if 'error' not in a]
            
            for analysis in successful_analyses:
                speeds = analysis.get('speeds', [])
                valid_speeds = [s for s in speeds if s > 0]
                all_speeds.extend(valid_speeds)
                
                total_timesteps += analysis.get('timesteps', 0)
                
                # Ambulance data
                amb_data = analysis.get('ambulance_data', [])
                amb_speeds = [d['speed'] for d in amb_data if d['speed'] > 0]
                all_amb_speeds.extend(amb_speeds)
                
                # NPC data
                npc_data = analysis.get('npc_data', [])
                npc_speeds = [d['speed'] for d in npc_data if d['speed'] > 0]
                all_npc_speeds.extend(npc_speeds)
            
            # Overall statistics section
            f.write("## 📊 Overall Speed Statistics\n\n")
            
            if all_speeds:
                f.write(f"- **Total Speed Measurements:** {len(all_speeds):,}\n")
                f.write(f"- **Total Simulation Timesteps:** {total_timesteps:,}\n")
                f.write(f"- **Average Speed (All Vehicles):** {np.mean(all_speeds):.2f} m/s ({np.mean(all_speeds)*3.6:.1f} km/h)\n")
                f.write(f"- **Maximum Speed (All Vehicles):** {np.max(all_speeds):.2f} m/s ({np.max(all_speeds)*3.6:.1f} km/h)\n")
                f.write(f"- **Minimum Speed (All Vehicles):** {np.min(all_speeds):.2f} m/s ({np.min(all_speeds)*3.6:.1f} km/h)\n")
                f.write(f"- **Speed Standard Deviation:** {np.std(all_speeds):.2f} m/s\n\n")
            
            if all_amb_speeds:
                f.write("### 🚑 Ambulance Performance\n\n")
                f.write(f"- **Total Ambulance Measurements:** {len(all_amb_speeds):,}\n")
                f.write(f"- **Average Ambulance Speed:** {np.mean(all_amb_speeds):.2f} m/s ({np.mean(all_amb_speeds)*3.6:.1f} km/h)\n")
                f.write(f"- **Maximum Ambulance Speed:** {np.max(all_amb_speeds):.2f} m/s ({np.max(all_amb_speeds)*3.6:.1f} km/h)\n")
                f.write(f"- **Minimum Ambulance Speed:** {np.min(all_amb_speeds):.2f} m/s ({np.min(all_amb_speeds)*3.6:.1f} km/h)\n")
                f.write(f"- **Ambulance Speed Std Dev:** {np.std(all_amb_speeds):.2f} m/s\n\n")
            
            if all_npc_speeds:
                f.write("### 🚗 NPC Vehicle Performance\n\n")
                f.write(f"- **Total NPC Measurements:** {len(all_npc_speeds):,}\n")
                f.write(f"- **Average NPC Speed:** {np.mean(all_npc_speeds):.2f} m/s ({np.mean(all_npc_speeds)*3.6:.1f} km/h)\n")
                f.write(f"- **Maximum NPC Speed:** {np.max(all_npc_speeds):.2f} m/s ({np.max(all_npc_speeds)*3.6:.1f} km/h)\n")
                f.write(f"- **Minimum NPC Speed:** {np.min(all_npc_speeds):.2f} m/s ({np.min(all_npc_speeds)*3.6:.1f} km/h)\n")
                f.write(f"- **NPC Speed Std Dev:** {np.std(all_npc_speeds):.2f} m/s\n\n")
            
            # Comparative analysis
            if all_amb_speeds and all_npc_speeds:
                amb_avg = np.mean(all_amb_speeds)
                npc_avg = np.mean(all_npc_speeds)
                speed_diff = amb_avg - npc_avg
                speed_advantage = ((amb_avg / npc_avg - 1) * 100) if npc_avg > 0 else 0
                
                f.write("## 📈 Performance Comparison\n\n")
                f.write(f"- **Ambulance vs NPC Speed Difference:** {speed_diff:.2f} m/s ({speed_diff*3.6:.1f} km/h)\n")
                f.write(f"- **Ambulance Speed Advantage:** {speed_advantage:.1f}% faster than NPCs\n")
                f.write(f"- **Speed Ratio (Ambulance/NPC):** {amb_avg/npc_avg:.2f}\n\n")
            
            # Scenario breakdown
            f.write("## 🎯 Detailed Analysis by Scenario\n\n")
            f.write("| Scenario | Episodes | Timesteps | Avg Speed | Max Speed | Amb Avg | Amb Max | NPC Avg | Performance |\n")
            f.write("|----------|----------|-----------|-----------|-----------|---------|---------|---------|-------------|\n")
            
            scenario_performance = []
            
            for scenario_name, analyses in scenario_stats.items():
                episode_count = len(analyses)
                
                # Aggregate statistics
                total_steps = sum(a.get('timesteps', 0) for a in analyses)
                all_scenario_speeds = []
                all_scenario_amb = []
                all_scenario_npc = []
                max_speeds = []
                max_amb_speeds = []
                
                for analysis in analyses:
                    if analysis.get('avg_speed', 0) > 0:
                        all_scenario_speeds.append(analysis['avg_speed'])
                    if analysis.get('max_speed', 0) > 0:
                        max_speeds.append(analysis['max_speed'])
                    if analysis.get('ambulance_avg_speed', 0) > 0:
                        all_scenario_amb.append(analysis['ambulance_avg_speed'])
                    if analysis.get('ambulance_max_speed', 0) > 0:
                        max_amb_speeds.append(analysis['ambulance_max_speed'])
                    if analysis.get('npc_avg_speed', 0) > 0:
                        all_scenario_npc.append(analysis['npc_avg_speed'])
                
                avg_speed = np.mean(all_scenario_speeds) if all_scenario_speeds else 0
                max_speed = np.max(max_speeds) if max_speeds else 0
                avg_amb_speed = np.mean(all_scenario_amb) if all_scenario_amb else 0
                max_amb_speed = np.max(max_amb_speeds) if max_amb_speeds else 0
                avg_npc_speed = np.mean(all_scenario_npc) if all_scenario_npc else 0
                
                performance = "N/A"
                if avg_amb_speed > 0 and avg_npc_speed > 0:
                    perf_pct = ((avg_amb_speed / avg_npc_speed - 1) * 100)
                    performance = f"{perf_pct:+.1f}%"
                
                scenario_performance.append((scenario_name, avg_amb_speed))
                
                f.write(f"| {scenario_name} | {episode_count} | {total_steps:,} | {avg_speed:.2f} | {max_speed:.2f} | {avg_amb_speed:.2f} | {max_amb_speed:.2f} | {avg_npc_speed:.2f} | {performance} |\n")
            
            # Top performing scenarios
            f.write("\n## 🏆 Top Performing Scenarios\n\n")
            
            if scenario_performance:
                top_scenarios = sorted(scenario_performance, key=lambda x: x[1], reverse=True)[:5]
                f.write("**Fastest Ambulance Scenarios:**\n")
                for i, (scenario, speed) in enumerate(top_scenarios, 1):
                    f.write(f"{i}. **{scenario}**: {speed:.2f} m/s ({speed*3.6:.1f} km/h)\n")
                
                f.write("\n**Slowest Ambulance Scenarios:**\n")
                bottom_scenarios = sorted(scenario_performance, key=lambda x: x[1])[:5]
                for i, (scenario, speed) in enumerate(bottom_scenarios, 1):
                    if speed > 0:
                        f.write(f"{i}. **{scenario}**: {speed:.2f} m/s ({speed*3.6:.1f} km/h)\n")
            
            # Key findings
            f.write("\n## 🔍 Key Findings\n\n")
            
            total_successful = len(successful_analyses)
            total_failed = len(all_analyses) - total_successful
            
            f.write(f"1. **Data Quality**: {total_successful}/{len(all_analyses)} files processed successfully ({total_successful/len(all_analyses)*100:.1f}%)\n")
            
            if all_amb_speeds and all_npc_speeds:
                f.write(f"2. **Emergency Vehicle Advantage**: Ambulances maintain {((np.mean(all_amb_speeds)/np.mean(all_npc_speeds)-1)*100):.1f}% speed advantage\n")
            
            if scenario_performance:
                best_scenario = max(scenario_performance, key=lambda x: x[1])
                worst_scenario = min([s for s in scenario_performance if s[1] > 0], key=lambda x: x[1])
                f.write(f"3. **Best Performance**: {best_scenario[0]} ({best_scenario[1]*3.6:.1f} km/h)\n")
                f.write(f"4. **Challenging Scenario**: {worst_scenario[0]} ({worst_scenario[1]*3.6:.1f} km/h)\n")
            
            f.write(f"5. **Dataset Scale**: {total_timesteps:,} total simulation timesteps across {len(scenario_stats)} unique scenarios\n")
            
            # Visualizations section
            f.write("\n## 📊 Generated Visualizations\n\n")
            f.write("Individual detailed analysis plots have been generated for each scenario in the output directory.\n\n")
            f.write("Each visualization includes:\n")
            f.write("- Speed over time analysis\n")
            f.write("- Speed distribution histogram\n")
            f.write("- Vehicle movement trajectories\n")
            f.write("- Comprehensive statistics summary\n\n")
            
            if total_failed > 0:
                f.write("## ⚠️ Processing Issues\n\n")
                f.write(f"{total_failed} files encountered processing errors. These may be due to:\n")
                f.write("- Incomplete data recording\n")
                f.write("- Data format variations\n")
                f.write("- Simulation interruptions\n\n")
        
        logger.info(f"📋 Comprehensive report saved: {report_path}")


def main():
    """Main execution function."""
    dataset_path = r"d:\Research_ITC\avs_folder\avs\data\ambulance_dataset_150_espisode_cpu_30_senario\ambulance_dataset_150_espisode_cpu_30_senario"
    
    analyzer = FixedAmbulanceAnalyzer(dataset_path)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()