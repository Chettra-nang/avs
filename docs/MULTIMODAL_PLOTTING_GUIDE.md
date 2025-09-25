# Multimodal Dataset Plotting Guide

This guide shows you how to visualize your hierarchical multimodal highway dataset with images, occupancy grids, kinematics data, and natural language summaries.

## Quick Start

### 1. Basic Usage

```python
from visualization.multimodal_parquet_plotter import MultimodalParquetPlotter

# Initialize plotter with your dataset path
plotter = MultimodalParquetPlotter("data/highway_multimodal_dataset")

# Analyze your dataset
plotter.analyze_dataset()

# Compare all scenarios
plotter.plot_scenario_comparison(save_plots=True)
```

### 2. Plot Individual Episodes

```python
# Discover available data
parquet_files = plotter.discover_parquet_files()

# Load specific episode
df = plotter.load_episode_data(parquet_files['free_flow']['transitions'][0])
episode_df = df[df['episode_id'] == 'ep_20250921_001']

# Plot comprehensive episode overview
plotter.plot_episode_overview(episode_df, 'ep_20250921_001', save_plots=True)
```

### 3. Run Example

```bash
# Run the complete example
python examples/plot_multimodal_data_example.py
```

## What You'll See

### Episode Overview Plot
The main episode visualization includes:

1. **Vehicle Trajectory** - 2D path colored by speed
2. **Speed Profile** - Speed over time with statistics
3. **Safety Metrics** - Time-to-collision and traffic density
4. **Sample Occupancy Grid** - Decompressed occupancy data
5. **Sample Camera Image** - Decompressed grayscale image
6. **Action Distribution** - Frequency of different actions
7. **Natural Language Summaries** - Sample text descriptions

### Scenario Comparison Plot
Compares multiple scenarios across:

1. **Speed Profiles** - Speed patterns across scenarios
2. **Safety Metrics** - Average time-to-collision comparison
3. **Trajectory Patterns** - Different driving patterns
4. **Action Distributions** - Action usage across scenarios
5. **Episode Lengths** - Duration comparison
6. **Reward Distributions** - Performance comparison

## Data Format Requirements

Your Parquet files should have this schema:

```python
# Core episode information
episode_id: string
step: int32
agent_id: int32
action: int32
reward: float64

# Kinematics features
ego_x: float64
ego_y: float64
ego_vx: float64
ego_vy: float64
speed: float64
lane_position: float64

# Safety and traffic metrics
ttc: float64
traffic_density: float64
lead_vehicle_gap: float64
vehicle_count: int32

# Binary array references
occ_blob: binary               # Compressed occupancy grid
occ_shape: string              # Array shape info
occ_dtype: string              # Data type info
gray_blob: binary              # Compressed grayscale image
gray_shape: string
gray_dtype: string

# Natural language summaries
summary_text: string           # Human-readable descriptions
```

## Customizing for Your Data

### 1. Modify Binary Data Decompression

If your binary data uses different compression:

```python
def decompress_binary_data(self, blob, shape_str, dtype_str):
    """Customize this method for your compression format."""
    try:
        # Your decompression logic here
        # Example for different formats:
        
        # For numpy compressed arrays:
        # data = np.load(io.BytesIO(blob))
        
        # For custom compression:
        # data = your_decompression_function(blob)
        
        shape = eval(shape_str)
        dtype = np.dtype(dtype_str)
        return data.reshape(shape).astype(dtype)
    except Exception as e:
        print(f"Error decompressing: {e}")
        return None
```

### 2. Add Custom Visualizations

```python
def plot_custom_metric(self, df, ax):
    """Add your own custom visualization."""
    ax.set_title("Custom Metric")
    
    # Your plotting logic here
    ax.plot(df['step'], df['your_metric'])
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Your Metric")
```

### 3. Modify Natural Language Analysis

```python
def analyze_text_summaries(self, df):
    """Custom text analysis."""
    summaries = df['summary_text'].dropna()
    
    # Your text analysis here
    # - Sentiment analysis
    # - Topic modeling
    # - Keyword extraction
    # - etc.
```

## Advanced Usage

### 1. Batch Processing Multiple Episodes

```python
def plot_all_episodes_in_scenario(self, scenario_name):
    """Plot all episodes in a scenario."""
    parquet_files = self.discover_parquet_files()
    
    for parquet_file in parquet_files[scenario_name]['transitions']:
        df = self.load_episode_data(parquet_file)
        
        for episode_id in df['episode_id'].unique():
            episode_df = df[df['episode_id'] == episode_id]
            self.plot_episode_overview(episode_df, episode_id, save_plots=True)
```

### 2. Statistical Analysis

```python
def compute_scenario_statistics(self):
    """Compute comprehensive statistics across scenarios."""
    parquet_files = self.discover_parquet_files()
    stats = {}
    
    for scenario_name, files in parquet_files.items():
        scenario_stats = {
            'total_episodes': 0,
            'total_steps': 0,
            'avg_speed': [],
            'avg_ttc': [],
            'action_distribution': {}
        }
        
        for parquet_file in files['transitions']:
            df = self.load_episode_data(parquet_file)
            if df is not None:
                scenario_stats['total_episodes'] += df['episode_id'].nunique()
                scenario_stats['total_steps'] += len(df)
                scenario_stats['avg_speed'].extend(df['speed'].tolist())
                scenario_stats['avg_ttc'].extend(df['ttc'].tolist())
        
        stats[scenario_name] = scenario_stats
    
    return stats
```

### 3. Export Data for External Analysis

```python
def export_for_analysis(self, output_format='csv'):
    """Export processed data for external analysis."""
    parquet_files = self.discover_parquet_files()
    
    for scenario_name, files in parquet_files.items():
        combined_df = pd.DataFrame()
        
        for parquet_file in files['transitions']:
            df = self.load_episode_data(parquet_file)
            if df is not None:
                # Remove binary columns for export
                export_df = df.drop(columns=['occ_blob', 'gray_blob'], errors='ignore')
                combined_df = pd.concat([combined_df, export_df], ignore_index=True)
        
        if not combined_df.empty:
            if output_format == 'csv':
                combined_df.to_csv(f"{scenario_name}_exported.csv", index=False)
            elif output_format == 'json':
                combined_df.to_json(f"{scenario_name}_exported.json", orient='records')
```

## Troubleshooting

### Common Issues

1. **Binary Data Decompression Fails**
   - Check your compression format (pickle, numpy, custom)
   - Verify shape and dtype strings are correct
   - Test decompression with a single sample first

2. **Missing Data Columns**
   - Verify your Parquet schema matches expected format
   - Check for typos in column names
   - Handle missing columns gracefully in plotting code

3. **Memory Issues with Large Datasets**
   - Process episodes one at a time
   - Use data sampling for overview plots
   - Consider using Dask for large datasets

4. **Natural Language Text Issues**
   - Handle NaN/null values in text columns
   - Check text encoding (UTF-8)
   - Truncate very long summaries for display

### Performance Tips

1. **Use data sampling** for quick overviews
2. **Cache processed data** to avoid recomputation
3. **Process in batches** for large datasets
4. **Use appropriate figure sizes** to avoid memory issues

## Next Steps

1. **Adapt the code** to your specific data format
2. **Add domain-specific visualizations** for your use case
3. **Integrate with your ML pipeline** for model analysis
4. **Create automated reporting** for regular dataset monitoring
5. **Add interactive visualizations** using Plotly or Bokeh

## Example Output Files

Running the plotter will generate:
- `episode_[ID]_[timestamp].png` - Individual episode overviews
- `scenario_comparison_[timestamp].png` - Cross-scenario analysis
- `multimodal_overview_[timestamp].png` - Dataset summary

All plots are saved as high-resolution PNG files suitable for reports and presentations.