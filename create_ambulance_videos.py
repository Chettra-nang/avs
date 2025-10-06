#!/usr/bin/env python3
"""
Ambulance Video Creator

Creates videos showing the ambulance behavior in different scenarios
to visualize how the vehicles are actually moving.
"""

import sys
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import json

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_ambulance_trajectory_video(data_path: str, scenario_name: str, output_dir: str):
    """
    Create a video showing ambulance trajectory and speed over time.
    """
    logger.info(f"Creating video for scenario: {scenario_name}")
    
    data_dir = Path(data_path) / scenario_name
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    parquet_files = list(data_dir.glob("*_transitions.parquet"))
    if not parquet_files:
        logger.error(f"No data files found in {data_dir}")
        return False
    
    df = pd.read_parquet(parquet_files[0])
    logger.info(f"Loaded {len(df)} transitions")
    
    # Get data for all agents
    agents = {}
    for agent_id in sorted(df['agent_id'].unique()):
        agent_data = df[df['agent_id'] == agent_id].sort_values('step').copy()
        
        # Calculate speeds
        agent_data['speed_kmh'] = agent_data['speed']  # Use the speed column
        agent_data['calculated_speed'] = np.sqrt(agent_data['ego_vx']**2 + agent_data['ego_vy']**2) * 3.6
        
        agents[agent_id] = agent_data
        
        if agent_id == 0:  # Ambulance
            logger.info(f"Ambulance: {len(agent_data)} steps, speed range: {agent_data['speed_kmh'].min():.1f}-{agent_data['speed_kmh'].max():.1f} km/h")
    
    # Create figure and animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Determine plot limits
    all_x = df['ego_x']
    all_y = df['ego_y']
    x_range = (all_x.min() - 10, all_x.max() + 10)
    y_range = (all_y.min() - 5, all_y.max() + 5)
    
    max_steps = max(len(data) for data in agents.values())
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        # Plot 1: Trajectory view
        ax1.set_xlim(x_range)
        ax1.set_ylim(y_range)
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title(f'{scenario_name} - Step {frame}')
        ax1.grid(True, alpha=0.3)
        
        # Draw lane markers (approximate)
        for y in range(int(y_range[0]), int(y_range[1]), 4):
            ax1.axhline(y, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        
        # Plot each agent at current frame
        colors = ['red', 'blue', 'green', 'orange']
        labels = ['Ambulance', 'Vehicle 1', 'Vehicle 2', 'Vehicle 3']
        
        current_speeds = []
        
        for i, (agent_id, agent_data) in enumerate(agents.items()):
            if frame < len(agent_data):
                row = agent_data.iloc[frame]
                x, y = row['ego_x'], row['ego_y']
                speed = row['speed_kmh']
                
                # Plot vehicle position
                color = colors[i % len(colors)]
                label = labels[i % len(labels)]
                
                if agent_id == 0:  # Ambulance - larger marker
                    ax1.scatter(x, y, color=color, s=150, marker='s', 
                              label=f'{label}: {speed:.1f} km/h', edgecolors='black', linewidths=2)
                else:
                    ax1.scatter(x, y, color=color, s=80, marker='o', 
                              label=f'{label}: {speed:.1f} km/h', alpha=0.8)
                
                # Draw trajectory trail
                if frame > 0:
                    trail_x = agent_data['ego_x'].iloc[:frame+1]
                    trail_y = agent_data['ego_y'].iloc[:frame+1]
                    ax1.plot(trail_x, trail_y, color=color, alpha=0.4, linewidth=1)
                
                current_speeds.append(speed)
        
        ax1.legend(loc='upper right')
        
        # Plot 2: Speed over time
        ax2.set_xlim(0, max_steps)
        ax2.set_ylim(0, max(100, max(max(data['speed_kmh']) for data in agents.values()) + 10))
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Speed (km/h)')
        ax2.set_title('Speed Profiles')
        ax2.grid(True, alpha=0.3)
        
        # Plot speed curves up to current frame
        for i, (agent_id, agent_data) in enumerate(agents.items()):
            color = colors[i % len(colors)]
            label = labels[i % len(labels)]
            
            if frame < len(agent_data):
                steps = agent_data['step'].iloc[:frame+1]
                speeds = agent_data['speed_kmh'].iloc[:frame+1]
                
                ax2.plot(steps, speeds, color=color, linewidth=2, 
                        label=label if frame == 0 else "", alpha=0.8)
                
                # Mark current point
                if len(steps) > 0:
                    ax2.scatter(steps.iloc[-1], speeds.iloc[-1], 
                              color=color, s=100, zorder=5)
        
        # Add speed reference lines
        ax2.axhline(40, color='orange', linestyle='--', alpha=0.7, label='Arterial (40 km/h)')
        ax2.axhline(60, color='green', linestyle='--', alpha=0.7, label='Highway (60 km/h)')
        
        if frame == 0:
            ax2.legend(loc='upper left')
        
        # Add scenario info
        fig.suptitle(f'Ambulance Emergency Response - {scenario_name}', fontsize=14, fontweight='bold')
        
        return ax1, ax2
    
    # Create animation
    frames = max_steps
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=200, blit=False, repeat=True)
    
    # Save as MP4 video
    video_file = output_path / f'{scenario_name}_ambulance_behavior.mp4'
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='Highway Ambulance System'), bitrate=1800)
        anim.save(str(video_file), writer=writer)
        logger.info(f"✅ Saved video: {video_file}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save video: {e}")
        
        # Save as GIF instead
        gif_file = output_path / f'{scenario_name}_ambulance_behavior.gif'
        try:
            anim.save(str(gif_file), writer='pillow', fps=2)
            logger.info(f"✅ Saved GIF: {gif_file}")
            return True
        except Exception as e2:
            logger.error(f"❌ Failed to save GIF: {e2}")
            return False

def create_ambulance_demo_with_real_env():
    """
    Create a live demonstration showing ambulance behavior using the actual environment.
    """
    logger.info("=== CREATING LIVE AMBULANCE DEMONSTRATION ===")
    
    try:
        # Import the ambulance collection system
        from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector
        
        # Create a demo with visual rendering
        with AmbulanceDataCollector(n_agents=4) as collector:
            
            # Get the first scenario
            scenarios = collector.get_available_scenarios()
            demo_scenario = scenarios[0] if scenarios else "highway_emergency_light"
            
            logger.info(f"Creating live demo for scenario: {demo_scenario}")
            
            # Collect a single episode with visualization
            demo_results = collector.collect_ambulance_data(
                scenarios=[demo_scenario],
                episodes_per_scenario=1,
                max_steps_per_episode=30,
                base_seed=42,
                batch_size=1  # Single environment for demo
            )
            
            logger.info("✅ Live demo completed successfully!")
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed to create live demo: {e}")
        return False

def main():
    """Create comprehensive ambulance visualizations."""
    logger.info("=== AMBULANCE BEHAVIOR VISUALIZATION SYSTEM ===")
    
    # Check if we have collected data
    data_path = Path("data/fast_ambulance_verified")
    if not data_path.exists():
        logger.error("No collected ambulance data found! Run data collection first.")
        return False
    
    output_dir = "output/ambulance_videos"
    
    # Load index to get scenarios
    index_file = data_path / "index.json"
    with open(index_file, 'r') as f:
        index = json.load(f)
    
    scenarios = list(index['scenarios'].keys())
    logger.info(f"Found {len(scenarios)} scenarios: {scenarios}")
    
    # Create videos for each scenario
    success_count = 0
    for scenario_name in scenarios:
        logger.info(f"\nProcessing scenario: {scenario_name}")
        
        if create_ambulance_trajectory_video(str(data_path), scenario_name, output_dir):
            success_count += 1
        else:
            logger.warning(f"Failed to create video for {scenario_name}")
    
    # Summary
    logger.info(f"\n=== VIDEO CREATION SUMMARY ===")
    logger.info(f"✅ Successfully created {success_count}/{len(scenarios)} videos")
    logger.info(f"📁 Output directory: {output_dir}")
    
    if success_count > 0:
        logger.info("🎬 Videos show ambulance trajectory and speed over time")
        logger.info("🚑 Red squares represent ambulance, circles are other vehicles")
        logger.info("📈 Right panel shows speed profiles for all agents")
        
        # Also create live demo
        logger.info("\n=== CREATING LIVE DEMONSTRATION ===")
        create_ambulance_demo_with_real_env()
        
        return True
    else:
        logger.error("❌ No videos were created successfully")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎥 Ambulance video visualization completed!")
        print("📺 Check output/ambulance_videos/ for MP4 files or GIFs")
        print("🚑 Videos show real-time ambulance behavior and speed analysis")
    else:
        print("❌ Video creation failed - check logs for details")