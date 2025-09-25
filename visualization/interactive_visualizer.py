#!/usr/bin/env python3
"""
Interactive Image Visualizer
============================

An interactive script to easily explore your highway simulation image data.
"""

import os
from image_stack_visualizer import ImageStackVisualizer
from pathlib import Path

def main():
    print("🚗 Interactive Highway Image Visualizer")
    print("=" * 50)
    
    # Initialize visualizer
    viz = ImageStackVisualizer()
    
    # Find available files
    files = viz.find_parquet_files()
    
    if not files:
        print("❌ No Parquet files found in the data directory!")
        return
    
    print(f"\nFound {len(files)} Parquet files:")
    
    # Group files by scenario type
    scenarios = {}
    for file in files:
        path_parts = Path(file).parts
        if 'dense_commuting' in path_parts:
            scenario = 'dense_commuting'
        elif 'free_flow' in path_parts:
            scenario = 'free_flow'
        elif 'stop_and_go' in path_parts:
            scenario = 'stop_and_go'
        else:
            scenario = 'other'
        
        if scenario not in scenarios:
            scenarios[scenario] = []
        scenarios[scenario].append(file)
    
    # Display organized by scenario
    file_index = 1
    file_map = {}
    
    for scenario, scenario_files in scenarios.items():
        print(f"\n📁 {scenario.upper().replace('_', ' ')} ({len(scenario_files)} files):")
        for file in scenario_files:
            filename = Path(file).name
            print(f"  {file_index:2d}. {filename}")
            file_map[file_index] = file
            file_index += 1
    
    # Interactive selection
    while True:
        print(f"\n{'='*50}")
        print("What would you like to do?")
        print("1. Visualize a specific file")
        print("2. Compare scenarios")
        print("3. Export frames from a file")
        print("4. Create animation GIF")
        print("5. Quick demo with first file")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == '0':
            print("👋 Goodbye!")
            break
        
        elif choice == '1':
            # Visualize specific file
            file_num = input(f"Enter file number (1-{len(files)}): ").strip()
            try:
                file_num = int(file_num)
                if 1 <= file_num <= len(files):
                    selected_file = file_map[file_num]
                    num_samples = input("Number of samples to show (default: 3): ").strip()
                    num_samples = int(num_samples) if num_samples else 3
                    
                    print(f"\n🔍 Loading {Path(selected_file).name}...")
                    if viz.load_parquet_file(selected_file):
                        viz.print_dataset_summary()
                        print("📊 Plotting images...")
                        viz.plot_sample_images(num_samples=num_samples)
                        viz.plot_occupancy_grids(num_samples=min(2, num_samples))
                else:
                    print("❌ Invalid file number!")
            except ValueError:
                print("❌ Please enter a valid number!")
        
        elif choice == '2':
            # Compare scenarios
            print("\n🔍 Comparing different scenarios...")
            scenario_files = []
            for scenario, scenario_files_list in scenarios.items():
                if scenario_files_list:
                    scenario_files.append(scenario_files_list[0])  # Take first file from each scenario
            
            if len(scenario_files) >= 2:
                viz.compare_scenarios(scenario_files[:3])  # Compare up to 3 scenarios
            else:
                print("❌ Need at least 2 different scenarios to compare!")
        
        elif choice == '3':
            # Export frames
            file_num = input(f"Enter file number (1-{len(files)}): ").strip()
            try:
                file_num = int(file_num)
                if 1 <= file_num <= len(files):
                    selected_file = file_map[file_num]
                    output_dir = input("Output directory (default: ./exported_frames): ").strip()
                    output_dir = output_dir if output_dir else "./exported_frames"
                    
                    max_episodes = input("Max episodes to export (default: 2): ").strip()
                    max_episodes = int(max_episodes) if max_episodes else 2
                    
                    max_steps = input("Max steps per episode (default: 10): ").strip()
                    max_steps = int(max_steps) if max_steps else 10
                    
                    print(f"\n📁 Loading {Path(selected_file).name}...")
                    if viz.load_parquet_file(selected_file):
                        viz.export_frames_to_png(output_dir, max_episodes, max_steps)
                else:
                    print("❌ Invalid file number!")
            except ValueError:
                print("❌ Please enter valid numbers!")
        
        elif choice == '4':
            # Create animation
            file_num = input(f"Enter file number (1-{len(files)}): ").strip()
            try:
                file_num = int(file_num)
                if 1 <= file_num <= len(files):
                    selected_file = file_map[file_num]
                    
                    print(f"\n📁 Loading {Path(selected_file).name}...")
                    if viz.load_parquet_file(selected_file):
                        # Show available episodes
                        df = viz.current_df
                        episodes = df['episode_id'].unique() if 'episode_id' in df.columns else []
                        
                        if len(episodes) > 0:
                            print(f"Available episodes: {list(episodes)}")
                            episode_id = input(f"Enter episode ID (default: {episodes[0]}): ").strip()
                            episode_id = episode_id if episode_id else str(episodes[0])
                            
                            output_path = input("Output GIF path (default: animation.gif): ").strip()
                            output_path = output_path if output_path else "animation.gif"
                            
                            max_steps = input("Max steps for animation (default: 30): ").strip()
                            max_steps = int(max_steps) if max_steps else 30
                            
                            viz.create_animation_gif(episode_id, output_path, max_steps)
                        else:
                            print("❌ No episode data found!")
                else:
                    print("❌ Invalid file number!")
            except ValueError:
                print("❌ Please enter valid numbers!")
        
        elif choice == '5':
            # Quick demo
            print(f"\n🚀 Quick demo with first file...")
            if viz.load_parquet_file(files[0]):
                viz.print_dataset_summary()
                viz.plot_sample_images(num_samples=2)
                viz.plot_occupancy_grids(num_samples=1)
        
        else:
            print("❌ Invalid choice! Please enter 0-5.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()