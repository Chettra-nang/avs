#!/usr/bin/env python3
"""
Example Usage of Image Stack Visualizer
=======================================

This script demonstrates how to use the ImageStackVisualizer class
to explore and visualize highway simulation data.
"""

from image_stack_visualizer import ImageStackVisualizer
import os
from pathlib import Path

def main():
    """Demonstrate various visualization capabilities."""
    
    # Initialize the visualizer
    print("🚗 Highway Image Stack Visualizer Demo")
    print("=" * 50)
    
    viz = ImageStackVisualizer()
    
    # Find available files
    print("\n1. Finding available Parquet files...")
    files = viz.find_parquet_files()
    print(f"Found {len(files)} Parquet files")
    
    if not files:
        print("❌ No Parquet files found! Please check your data directory.")
        return
    
    # Display first few files
    print("\nAvailable files:")
    for i, file in enumerate(files[:5], 1):
        print(f"  {i}. {Path(file).name}")
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more files")
    
    # Load and analyze the first file
    print(f"\n2. Loading first file: {files[0]}")
    if viz.load_parquet_file(files[0]):
        viz.print_dataset_summary()
        
        # Plot sample images
        print("\n3. Plotting sample grayscale images...")
        viz.plot_sample_images(num_samples=2)
        
        # Plot occupancy grids
        print("\n4. Plotting occupancy grids...")
        viz.plot_occupancy_grids(num_samples=2)
        
        # Export some frames (optional)
        export_dir = "/home/chettra/ITC/Research/AVs/visualization/exported_frames"
        print(f"\n5. Exporting sample frames to: {export_dir}")
        viz.export_frames_to_png(export_dir, max_episodes=2, max_steps_per_episode=5)
        
        # Try scenario comparison if multiple files exist
        if len(files) >= 2:
            print("\n6. Comparing different scenarios...")
            scenario_files = files[:min(3, len(files))]  # Compare up to 3 scenarios
            viz.compare_scenarios(scenario_files)
        
        print("\n✅ Visualization demo completed!")
        print("\nTips:")
        print("- Use command line arguments for more control:")
        print("  python image_stack_visualizer.py --file /path/to/file.parquet")
        print("  python image_stack_visualizer.py --list-files")
        print("  python image_stack_visualizer.py --export /path/to/output/")
        print("- Modify the script parameters to customize visualizations")
        
    else:
        print("❌ Failed to load the file!")

if __name__ == "__main__":
    main()