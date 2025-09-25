# Image Stack Visualizer

A comprehensive tool for visualizing grayscale image stacks and occupancy grids from highway simulation Parquet data files.

## Features

- 🖼️ **Decode and visualize grayscale image stacks** with proper dtype handling
- 🗺️ **Plot occupancy grids** with multiple channels
- 📁 **Batch export frames** to individual PNG files
- 🎬 **Create animated GIFs** from episode sequences
- 📊 **Statistical analysis** of image data
- 🔍 **Interactive dataset exploration**
- 📈 **Scenario comparison** across different traffic conditions

## Installation

1. Install required dependencies:
```bash
pip install -r requirements_viz.txt
```

Or install individually:
```bash
pip install pandas numpy matplotlib seaborn pyarrow pillow
```

## Quick Start

### Basic Usage

```python
from image_stack_visualizer import ImageStackVisualizer

# Initialize visualizer
viz = ImageStackVisualizer()

# Load a Parquet file
viz.load_parquet_file("/path/to/your/data.parquet")

# Print dataset information
viz.print_dataset_summary()

# Plot sample images
viz.plot_sample_images(num_samples=3)

# Plot occupancy grids
viz.plot_occupancy_grids(num_samples=2)
```

### Command Line Usage

```bash
# List all available Parquet files
python image_stack_visualizer.py --list-files

# Visualize a specific file
python image_stack_visualizer.py --file /path/to/data.parquet --samples 5

# Export frames to PNG files
python image_stack_visualizer.py --file /path/to/data.parquet --export ./exported_frames/

# Create animated GIF for a specific episode
python image_stack_visualizer.py --file /path/to/data.parquet --gif "ep_20250921_001" --gif-output animation.gif
```

### Demo Script

Run the demonstration script to see all features:
```bash
python demo_visualizer.py
```

### Interactive Interface

Use the interactive menu-driven interface:
```bash
python interactive_visualizer.py
```
This provides a user-friendly menu to:
- Browse and select files
- Visualize specific datasets
- Compare scenarios
- Export frames
- Create animations

## Data Format Requirements

The visualizer expects Parquet files with the following columns:

### Required for Images:
- `grayscale_blob`: Binary blob containing image data
- `grayscale_shape`: Shape information (e.g., "[4, 128, 64]")
- `grayscale_dtype`: Data type (e.g., "|u1" for uint8)

### Optional for Occupancy Grids:
- `occupancy_blob`: Binary blob containing occupancy grid data
- `occupancy_shape`: Shape information for occupancy grids
- `occupancy_dtype`: Data type for occupancy grids

### Metadata Columns:
- `episode_id`: Episode identifier
- `step`: Step number within episode

## Examples

### 1. Basic Visualization

```python
# Load and visualize data
viz = ImageStackVisualizer()
viz.load_parquet_file("data/highway_multimodal_dataset/dense_commuting/transitions.parquet")

# Show dataset info
viz.print_dataset_summary()

# Plot 3 sample image stacks
viz.plot_sample_images(num_samples=3)
```

### 2. Export Frames for External Analysis

```python
# Export individual frames as PNG files
viz.export_frames_to_png(
    output_dir="./exported_frames",
    max_episodes=5,
    max_steps_per_episode=10
)
```

### 3. Compare Different Traffic Scenarios

```python
# Compare images from different scenarios
scenario_files = [
    "data/dense_commuting/transitions.parquet",
    "data/free_flow/transitions.parquet",
    "data/stop_and_go/transitions.parquet"
]
viz.compare_scenarios(scenario_files)
```

### 4. Create Episode Animation

```python
# Create animated GIF from episode sequence
viz.create_animation_gif(
    episode_id="ep_20250921_001",
    output_path="episode_animation.gif",
    max_steps=50
)
```

## Class Methods Reference

### `ImageStackVisualizer`

#### Core Methods:
- `load_parquet_file(file_path)`: Load a Parquet file for analysis
- `analyze_dataset_info()`: Get dataset statistics and information
- `print_dataset_summary()`: Print formatted dataset summary

#### Visualization Methods:
- `plot_sample_images(num_samples, save_path)`: Plot grayscale image stacks
- `plot_occupancy_grids(num_samples, save_path)`: Plot occupancy grids
- `compare_scenarios(scenario_paths)`: Compare multiple scenarios

#### Export Methods:
- `export_frames_to_png(output_dir, max_episodes, max_steps_per_episode)`: Export frames as PNG
- `create_animation_gif(episode_id, output_path, max_steps)`: Create animated GIF

#### Utility Methods:
- `find_parquet_files()`: Find all Parquet files in data directory
- `decode_blob(row, blob_col, shape_col, dtype_col)`: Decode binary blob to numpy array

## Command Line Arguments

```
--file, -f          Specific Parquet file to visualize
--data-root, -d     Root directory containing data files (default: /home/chettra/ITC/Research/AVs/data)
--samples, -s       Number of samples to visualize (default: 3)
--export, -e        Directory to export individual frames as PNG
--gif, -g           Create animated GIF for specific episode ID
--gif-output        Output path for GIF file (default: episode_animation.gif)
--list-files, -l    List all available Parquet files
```

## Troubleshooting

### Common Issues:

1. **"No Parquet files found"**
   - Check that the data directory path is correct
   - Ensure Parquet files exist in the specified location

2. **"Missing required columns"**
   - Verify that the Parquet file contains the expected blob columns
   - Check column names match expected format

3. **"Error decoding blob"**
   - The binary data may be corrupted or in unexpected format
   - Verify that shape and dtype information is correct

4. **Memory issues with large files**
   - Reduce the number of samples being processed
   - Process files in smaller batches

### Performance Tips:

- Use `num_samples` parameter to limit data loading
- Export frames in batches rather than all at once
- Close matplotlib figures after viewing to free memory

## File Structure

```
visualization/
├── image_stack_visualizer.py    # Main visualizer class
├── demo_visualizer.py           # Demonstration script
├── interactive_visualizer.py    # Interactive menu-driven interface
├── debug_data.py               # Debug script for troubleshooting
├── requirements_viz.txt        # Dependencies
└── README.md                   # This file
```

## Contributing

Feel free to extend the visualizer with additional features:
- Support for other data formats (CSV, HDF5)
- Interactive plotting with plotly
- Statistical analysis tools
- Custom colormap support
- Batch processing utilities

## License

This visualization tool is part of the Highway Multimodal Dataset project.