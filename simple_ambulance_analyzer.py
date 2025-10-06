#!/usr/bin/env python3
"""
Simple Ambulance Dataset Speed Analyzer

This version uses the existing numeric columns directly for speed analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleAmbulanceAnalyzer:
    def __init__(self, dataset_path: str):
        """Initialize the analyzer."""
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path("d:/Research_ITC/avs_folder/avs/output/dataset_videos")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_scenario_name(self, file_path: Path) -> str:
        """Extract scenario name from file path."""
        parts = file_path.parts
        for i, part in enumerate(parts):
            if part.startswith('batch_'):
                if i + 1 < len(parts):
                    return parts[i + 1]
        return "unknown_scenario"
    
    def analyze_single_file(self, file_path: Path) -> dict:
        """Analyze a single parquet file using direct column access."""
        try:
            df = pd.read_parquet(file_path)
            scenario_name = self.extract_scenario_name(file_path)
            
            logger.info(f"✅ {scenario_name}: {len(df)} rows")
            
            analysis = {
                'scenario_name': scenario_name,
                'file_path': str(file_path),
                'total_rows': len(df),
                'unique_agents': df['agent_id'].nunique() if 'agent_id' in df else 0,
            }
            
            # Use existing speed columns directly
            speed_cols = ['speed', 'average_speed', 'ego_vx', 'ego_vy']
            
            # Primary speed analysis using 'speed' column
            if 'speed' in df.columns:
                speeds = pd.to_numeric(df['speed'], errors='coerce').dropna()
                valid_speeds = speeds[speeds > 0]
                
                if len(valid_speeds) > 0:
                    analysis.update({
                        'speed_measurements': len(valid_speeds),
                        'avg_speed': float(valid_speeds.mean()),
                        'max_speed': float(valid_speeds.max()),
                        'min_speed': float(valid_speeds.min()),
                        'speed_std': float(valid_speeds.std())
                    })
            
            # Average speed analysis
            if 'average_speed' in df.columns:
                avg_speeds = pd.to_numeric(df['average_speed'], errors='coerce').dropna()
                valid_avg_speeds = avg_speeds[avg_speeds > 0]
                
                if len(valid_avg_speeds) > 0:
                    analysis.update({
                        'average_speed_mean': float(valid_avg_speeds.mean()),
                        'average_speed_max': float(valid_avg_speeds.max()),
                        'average_speed_min': float(valid_avg_speeds.min())
                    })
            
            # Velocity-based speed analysis
            if 'ego_vx' in df.columns and 'ego_vy' in df.columns:
                vx = pd.to_numeric(df['ego_vx'], errors='coerce').fillna(0)
                vy = pd.to_numeric(df['ego_vy'], errors='coerce').fillna(0)
                
                calculated_speeds = np.sqrt(vx**2 + vy**2)
                valid_calc_speeds = calculated_speeds[calculated_speeds > 0]
                
                if len(valid_calc_speeds) > 0:
                    analysis.update({
                        'calculated_speed_mean': float(valid_calc_speeds.mean()),
                        'calculated_speed_max': float(valid_calc_speeds.max()),
                        'calculated_speed_std': float(valid_calc_speeds.std())
                    })
            
            # Agent-specific analysis
            if 'agent_id' in df.columns:
                # Ambulance (agent_id == 0) analysis
                ambulance_data = df[df['agent_id'] == 0]
                if len(ambulance_data) > 0 and 'speed' in df.columns:
                    amb_speeds = pd.to_numeric(ambulance_data['speed'], errors='coerce').dropna()
                    amb_valid_speeds = amb_speeds[amb_speeds > 0]
                    
                    if len(amb_valid_speeds) > 0:
                        analysis.update({
                            'ambulance_speed_measurements': len(amb_valid_speeds),
                            'ambulance_avg_speed': float(amb_valid_speeds.mean()),
                            'ambulance_max_speed': float(amb_valid_speeds.max()),
                            'ambulance_min_speed': float(amb_valid_speeds.min())
                        })
                
                # NPC (agent_id != 0) analysis
                npc_data = df[df['agent_id'] != 0]
                if len(npc_data) > 0 and 'speed' in df.columns:
                    npc_speeds = pd.to_numeric(npc_data['speed'], errors='coerce').dropna()
                    npc_valid_speeds = npc_speeds[npc_speeds > 0]
                    
                    if len(npc_valid_speeds) > 0:
                        analysis.update({
                            'npc_speed_measurements': len(npc_valid_speeds),
                            'npc_avg_speed': float(npc_valid_speeds.mean()),
                            'npc_max_speed': float(npc_valid_speeds.max()),
                            'npc_min_speed': float(npc_valid_speeds.min())
                        })
            
            # Additional metrics
            if 'ttc' in df.columns:
                ttc_data = pd.to_numeric(df['ttc'], errors='coerce').dropna()
                if len(ttc_data) > 0:
                    analysis['avg_ttc'] = float(ttc_data.mean())
                    analysis['min_ttc'] = float(ttc_data.min())
            
            if 'traffic_density' in df.columns:
                density_data = pd.to_numeric(df['traffic_density'], errors='coerce').dropna()
                if len(density_data) > 0:
                    analysis['avg_traffic_density'] = float(density_data.mean())
                    analysis['max_traffic_density'] = float(density_data.max())
            
            if 'reward' in df.columns:
                reward_data = pd.to_numeric(df['reward'], errors='coerce').dropna()
                if len(reward_data) > 0:
                    analysis['total_reward'] = float(reward_data.sum())
                    analysis['avg_reward'] = float(reward_data.mean())
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {file_path.name}: {e}")
            return {
                'error': str(e), 
                'file_path': str(file_path), 
                'scenario_name': self.extract_scenario_name(file_path)
            }
    
    def create_scenario_visualization(self, analysis_data: dict) -> str:
        """Create visualization for scenario."""
        if 'error' in analysis_data:
            return ""
        
        try:
            scenario_name = analysis_data['scenario_name']
            logger.info(f"📊 Creating plot for {scenario_name}")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
            
            # Speed comparison chart
            ax1.set_title(f'Speed Analysis: {scenario_name}', fontweight='bold')
            
            speed_types = []
            speed_values = []
            colors = []
            
            if 'avg_speed' in analysis_data:
                speed_types.append('Primary\nSpeed')
                speed_values.append(analysis_data['avg_speed'])
                colors.append('blue')
            
            if 'average_speed_mean' in analysis_data:
                speed_types.append('Average\nSpeed')
                speed_values.append(analysis_data['average_speed_mean'])
                colors.append('green')
            
            if 'calculated_speed_mean' in analysis_data:
                speed_types.append('Calculated\nSpeed')
                speed_values.append(analysis_data['calculated_speed_mean'])
                colors.append('orange')
            
            if speed_types:
                bars = ax1.bar(speed_types, speed_values, color=colors, alpha=0.7)
                ax1.set_ylabel('Speed (m/s)')
                ax1.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, speed_values):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{value:.2f}', ha='center', va='bottom')
            
            # Vehicle type comparison
            ax2.set_title('Speed by Vehicle Type')
            
            vehicle_types = []
            vehicle_speeds = []
            vehicle_colors = []
            
            if 'ambulance_avg_speed' in analysis_data:
                vehicle_types.append('Ambulance')
                vehicle_speeds.append(analysis_data['ambulance_avg_speed'])
                vehicle_colors.append('red')
            
            if 'npc_avg_speed' in analysis_data:
                vehicle_types.append('NPCs')
                vehicle_speeds.append(analysis_data['npc_avg_speed'])
                vehicle_colors.append('gray')
            
            if vehicle_types:
                bars = ax2.bar(vehicle_types, vehicle_speeds, color=vehicle_colors, alpha=0.7)
                ax2.set_ylabel('Speed (m/s)')
                ax2.grid(True, alpha=0.3)
                
                for bar, value in zip(bars, vehicle_speeds):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{value:.2f}', ha='center', va='bottom')
            
            # Performance metrics
            ax3.set_title('Performance Metrics')
            
            metrics = []
            metric_values = []
            metric_colors = []
            
            if 'total_reward' in analysis_data:
                metrics.append('Total\nReward')
                metric_values.append(analysis_data['total_reward'])
                metric_colors.append('green' if analysis_data['total_reward'] > 0 else 'red')
            
            if 'avg_ttc' in analysis_data:
                metrics.append('Avg TTC\n(seconds)')
                metric_values.append(analysis_data['avg_ttc'])
                metric_colors.append('blue')
            
            if 'avg_traffic_density' in analysis_data:
                metrics.append('Traffic\nDensity')
                metric_values.append(analysis_data['avg_traffic_density'])
                metric_colors.append('orange')
            
            if metrics:
                bars = ax3.bar(metrics, metric_values, color=metric_colors, alpha=0.7)
                ax3.set_ylabel('Value')
                ax3.grid(True, alpha=0.3)
                
                for bar, value in zip(bars, metric_values):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:.1f}', ha='center', va='bottom')
            
            # Statistics summary
            ax4.axis('off')
            
            stats_text = []
            stats_text.append(f"🚑 {scenario_name.upper()}")
            stats_text.append(f"")
            stats_text.append(f"📊 Dataset Overview:")
            stats_text.append(f"  • Total Rows: {analysis_data.get('total_rows', 0):,}")
            stats_text.append(f"  • Unique Agents: {analysis_data.get('unique_agents', 0)}")
            
            if 'speed_measurements' in analysis_data:
                stats_text.append(f"  • Speed Measurements: {analysis_data['speed_measurements']:,}")
            
            stats_text.append(f"")
            stats_text.append(f"🏃 Speed Analysis:")
            
            if 'avg_speed' in analysis_data:
                avg_speed = analysis_data['avg_speed']
                max_speed = analysis_data.get('max_speed', 0)
                stats_text.append(f"  • Average: {avg_speed:.2f} m/s ({avg_speed*3.6:.1f} km/h)")
                stats_text.append(f"  • Maximum: {max_speed:.2f} m/s ({max_speed*3.6:.1f} km/h)")
            
            if 'ambulance_avg_speed' in analysis_data and 'npc_avg_speed' in analysis_data:
                amb_speed = analysis_data['ambulance_avg_speed']
                npc_speed = analysis_data['npc_avg_speed']
                advantage = ((amb_speed / npc_speed - 1) * 100) if npc_speed > 0 else 0
                
                stats_text.append(f"")
                stats_text.append(f"🚑 vs 🚗 Comparison:")
                stats_text.append(f"  • Ambulance: {amb_speed:.2f} m/s ({amb_speed*3.6:.1f} km/h)")
                stats_text.append(f"  • NPCs: {npc_speed:.2f} m/s ({npc_speed*3.6:.1f} km/h)")
                stats_text.append(f"  • Speed Advantage: {advantage:+.1f}%")
            
            if 'total_reward' in analysis_data:
                stats_text.append(f"")
                stats_text.append(f"🎯 Performance:")
                stats_text.append(f"  • Total Reward: {analysis_data['total_reward']:.1f}")
                if 'avg_reward' in analysis_data:
                    stats_text.append(f"  • Avg Reward: {analysis_data['avg_reward']:.3f}")
            
            ax4.text(0.05, 0.95, '\n'.join(stats_text), transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = self.output_dir / f"{scenario_name}_speed_analysis.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(viz_path)
            
        except Exception as e:
            logger.error(f"❌ Error creating visualization: {e}")
            return ""
    
    def run_complete_analysis(self):
        """Run the complete dataset analysis."""
        logger.info("🚑 === AMBULANCE DATASET SPEED ANALYSIS STARTING ===")
        
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
            logger.info(f"🔄 Processing {i+1}/{len(parquet_files)}: {file_path.name}")
            
            analysis = self.analyze_single_file(file_path)
            all_analyses.append(analysis)
            
            if 'error' not in analysis:
                scenario_name = analysis['scenario_name']
                if scenario_name not in scenario_stats:
                    scenario_stats[scenario_name] = []
                scenario_stats[scenario_name].append(analysis)
                
                # Create visualization
                viz_path = self.create_scenario_visualization(analysis)
                if viz_path:
                    logger.info(f"📊 Saved: {Path(viz_path).name}")
        
        # Generate comprehensive report
        self.generate_report(all_analyses, scenario_stats)
        
        logger.info(f"\n✅ ANALYSIS COMPLETE!")
        logger.info(f"📂 Results saved to: {self.output_dir}")
    
    def generate_report(self, all_analyses, scenario_stats):
        """Generate comprehensive markdown report."""
        
        report_path = self.output_dir / "AMBULANCE_SPEED_ANALYSIS_SUMMARY.md"
        
        successful_analyses = [a for a in all_analyses if 'error' not in a]
        failed_analyses = [a for a in all_analyses if 'error' in a]
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 🚑 Ambulance Dataset Speed Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Dataset Location:** `{self.dataset_path}`\n")
            f.write(f"**Files Processed:** {len(successful_analyses)}/{len(all_analyses)} successful\n")
            f.write(f"**Unique Scenarios:** {len(scenario_stats)}\n\n")
            
            # Overall statistics
            if successful_analyses:
                total_rows = sum(a.get('total_rows', 0) for a in successful_analyses)
                total_speed_measurements = sum(a.get('speed_measurements', 0) for a in successful_analyses)
                
                all_avg_speeds = [a['avg_speed'] for a in successful_analyses if 'avg_speed' in a]
                all_max_speeds = [a['max_speed'] for a in successful_analyses if 'max_speed' in a]
                all_ambulance_speeds = [a['ambulance_avg_speed'] for a in successful_analyses if 'ambulance_avg_speed' in a]
                all_npc_speeds = [a['npc_avg_speed'] for a in successful_analyses if 'npc_avg_speed' in a]
                
                f.write("## 📊 Overall Statistics\n\n")
                f.write(f"- **Total Data Rows:** {total_rows:,}\n")
                f.write(f"- **Total Speed Measurements:** {total_speed_measurements:,}\n")
                
                if all_avg_speeds:
                    f.write(f"- **Overall Average Speed:** {np.mean(all_avg_speeds):.2f} m/s ({np.mean(all_avg_speeds)*3.6:.1f} km/h)\n")
                    f.write(f"- **Highest Recorded Speed:** {np.max(all_max_speeds):.2f} m/s ({np.max(all_max_speeds)*3.6:.1f} km/h)\n")
                
                if all_ambulance_speeds:
                    f.write(f"\n### 🚑 Ambulance Performance\n")
                    f.write(f"- **Average Ambulance Speed:** {np.mean(all_ambulance_speeds):.2f} m/s ({np.mean(all_ambulance_speeds)*3.6:.1f} km/h)\n")
                    f.write(f"- **Fastest Ambulance Speed:** {np.max([a.get('ambulance_max_speed', 0) for a in successful_analyses]):.2f} m/s\n")
                
                if all_npc_speeds:
                    f.write(f"\n### 🚗 NPC Vehicle Performance\n")
                    f.write(f"- **Average NPC Speed:** {np.mean(all_npc_speeds):.2f} m/s ({np.mean(all_npc_speeds)*3.6:.1f} km/h)\n")
                    f.write(f"- **Fastest NPC Speed:** {np.max([a.get('npc_max_speed', 0) for a in successful_analyses]):.2f} m/s\n")
                
                if all_ambulance_speeds and all_npc_speeds:
                    amb_avg = np.mean(all_ambulance_speeds)
                    npc_avg = np.mean(all_npc_speeds)
                    advantage = ((amb_avg / npc_avg - 1) * 100) if npc_avg > 0 else 0
                    f.write(f"\n### 📈 Performance Comparison\n")
                    f.write(f"- **Ambulance Speed Advantage:** {advantage:+.1f}%\n")
                    f.write(f"- **Speed Difference:** {amb_avg - npc_avg:+.2f} m/s ({(amb_avg - npc_avg)*3.6:+.1f} km/h)\n")
            
            # Scenario breakdown
            f.write(f"\n## 🎯 Scenario Analysis\n\n")
            f.write("| Scenario | Episodes | Rows | Avg Speed | Max Speed | Amb Speed | NPC Speed | Advantage |\n")
            f.write("|----------|----------|------|-----------|-----------|-----------|-----------|----------|\n")
            
            scenario_performance = []
            
            for scenario_name, analyses in scenario_stats.items():
                episode_count = len(analyses)
                total_rows = sum(a.get('total_rows', 0) for a in analyses)
                
                scenario_avg_speeds = [a.get('avg_speed', 0) for a in analyses if a.get('avg_speed', 0) > 0]
                scenario_max_speeds = [a.get('max_speed', 0) for a in analyses if a.get('max_speed', 0) > 0]
                scenario_amb_speeds = [a.get('ambulance_avg_speed', 0) for a in analyses if a.get('ambulance_avg_speed', 0) > 0]
                scenario_npc_speeds = [a.get('npc_avg_speed', 0) for a in analyses if a.get('npc_avg_speed', 0) > 0]
                
                avg_speed = np.mean(scenario_avg_speeds) if scenario_avg_speeds else 0
                max_speed = np.max(scenario_max_speeds) if scenario_max_speeds else 0
                avg_amb_speed = np.mean(scenario_amb_speeds) if scenario_amb_speeds else 0
                avg_npc_speed = np.mean(scenario_npc_speeds) if scenario_npc_speeds else 0
                
                advantage = "N/A"
                if avg_amb_speed > 0 and avg_npc_speed > 0:
                    advantage_pct = ((avg_amb_speed / avg_npc_speed - 1) * 100)
                    advantage = f"{advantage_pct:+.1f}%"
                
                scenario_performance.append((scenario_name, avg_amb_speed))
                
                f.write(f"| {scenario_name} | {episode_count} | {total_rows:,} | {avg_speed:.2f} | {max_speed:.2f} | {avg_amb_speed:.2f} | {avg_npc_speed:.2f} | {advantage} |\n")
            
            # Top/bottom scenarios
            if scenario_performance:
                f.write(f"\n## 🏆 Performance Rankings\n\n")
                
                sorted_scenarios = sorted(scenario_performance, key=lambda x: x[1], reverse=True)
                
                f.write("**🥇 Fastest Ambulance Scenarios:**\n")
                for i, (scenario, speed) in enumerate(sorted_scenarios[:5], 1):
                    if speed > 0:
                        f.write(f"{i}. **{scenario}**: {speed:.2f} m/s ({speed*3.6:.1f} km/h)\n")
                
                f.write("\n**🐌 Slowest Ambulance Scenarios:**\n")
                bottom_scenarios = [s for s in sorted_scenarios if s[1] > 0][-5:]
                for i, (scenario, speed) in enumerate(reversed(bottom_scenarios), 1):
                    f.write(f"{i}. **{scenario}**: {speed:.2f} m/s ({speed*3.6:.1f} km/h)\n")
            
            # Summary findings
            f.write(f"\n## 🔍 Key Findings\n\n")
            f.write(f"1. **Data Coverage**: Analyzed {len(successful_analyses)} scenarios with {sum(a.get('total_rows', 0) for a in successful_analyses):,} total data points\n")
            
            if all_ambulance_speeds and all_npc_speeds:
                avg_advantage = np.mean([(a.get('ambulance_avg_speed', 0) / a.get('npc_avg_speed', 1) - 1) * 100 
                                       for a in successful_analyses 
                                       if a.get('ambulance_avg_speed', 0) > 0 and a.get('npc_avg_speed', 0) > 0])
                f.write(f"2. **Emergency Response Advantage**: Ambulances maintain average {avg_advantage:.1f}% speed advantage over regular traffic\n")
            
            if scenario_performance:
                best_scenario = max(scenario_performance, key=lambda x: x[1])
                worst_scenario = min([s for s in scenario_performance if s[1] > 0], key=lambda x: x[1], default=("N/A", 0))
                f.write(f"3. **Optimal Scenario**: {best_scenario[0]} allows fastest ambulance response ({best_scenario[1]*3.6:.1f} km/h)\n")
                if worst_scenario[1] > 0:
                    f.write(f"4. **Challenging Scenario**: {worst_scenario[0]} presents most difficult conditions ({worst_scenario[1]*3.6:.1f} km/h)\n")
            
            f.write(f"5. **Dataset Quality**: {len(successful_analyses)}/{len(all_analyses)} files processed successfully ({len(successful_analyses)/len(all_analyses)*100:.1f}% success rate)\n")
            
            if failed_analyses:
                f.write(f"\n## ⚠️ Processing Issues\n\n")
                f.write(f"The following {len(failed_analyses)} files encountered processing errors:\n\n")
                for failed in failed_analyses:
                    f.write(f"- `{Path(failed['file_path']).name}`: {failed.get('error', 'Unknown error')}\n")
            
            f.write(f"\n## 📊 Generated Visualizations\n\n")
            f.write("Individual speed analysis charts have been generated for each successfully processed scenario.\n")
            f.write("Each chart includes:\n")
            f.write("- Speed comparison across different measurement methods\n")
            f.write("- Ambulance vs NPC performance comparison\n")
            f.write("- Performance metrics (rewards, time-to-collision, traffic density)\n")
            f.write("- Comprehensive statistics summary\n")
        
        logger.info(f"📋 Report saved: {report_path}")


def main():
    """Main function."""
    dataset_path = r"d:\Research_ITC\avs_folder\avs\data\ambulance_dataset_150_espisode_cpu_30_senario\ambulance_dataset_150_espisode_cpu_30_senario"
    
    analyzer = SimpleAmbulanceAnalyzer(dataset_path)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()