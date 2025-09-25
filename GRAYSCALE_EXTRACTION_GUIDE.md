# Dual-Agent Grayscale Image Extraction Guide

This guide explains how to extract and visualize grayscale images for your 2-agent highway simulation dataset.

## 📋 Overview

Your dataset contains grayscale images from both agents' perspectives stored as binary blobs in the parquet file. The images have the following characteristics:

- **Shape**: [4, 128, 64] (4 channels, 128 height, 64 width)  
- **Format**: RGBA channels stored as uint8
- **Storage**: Compressed binary blobs with shape and dtype metadata
- **Coverage**: 100% - all 84 rows have grayscale data

## 🚗 Agent Identification

- **Agent 0**: First controlled vehicle (red in visualizations)
- **Agent 1**: Second controlled vehicle (blue in visualizations)
- Each agent has its own ego-centric view of the environment
- Coordinates are relative to each agent's perspective

## 🛠️ Available Scripts

### 1. Quick Grayscale Extractor (`quick_grayscale.py`)

**Purpose**: Simple, fast extraction for basic visualization

**Usage**:
```bash
# Extract first 3 steps (default)
python3 quick_grayscale.py

# Extract first N steps  
python3 quick_grayscale.py 5

# Extract specific steps
python3 quick_grayscale.py 0 1 2 5 10
```

**Output**: 
- Simple side-by-side comparison images
- Basic agent information overlays
- Saved to `quick_grayscale/` directory

### 2. Full Dual-Agent Extractor (`dual_agent_grayscale.py`)

**Purpose**: Comprehensive extraction with multiple visualization options

**Usage**:
```bash
# Basic extraction (first 5 steps)
python3 dual_agent_grayscale.py

# Specific episode and steps
python3 dual_agent_grayscale.py --episode ep_dense_commuting_10043_0001 --steps 0 1 2 3

# Multiple view types
python3 dual_agent_grayscale.py --view-types best rgb_grayscale channel_0

# Comparison grid
python3 dual_agent_grayscale.py --comparison-grid --steps 0 2 4 6 8

# All available steps
python3 dual_agent_grayscale.py --all-steps
```

**View Types Available**:
- `best`: Automatically chosen best grayscale representation
- `rgb_grayscale`: RGB-to-grayscale conversion  
- `channel_0` through `channel_2`: Individual RGBA channels

**Output**:
- Individual step comparisons for each view type
- Comparison grids showing temporal evolution
- Detailed agent information and image statistics
- Saved to `grayscale_images/` directory

## 📊 Understanding the Output

### Image Information Displayed

Each visualization includes:

**Agent Information**:
- Position: (ego_x, ego_y) coordinates
- Velocity: (ego_vx, ego_vy) components  
- Speed: Calculated from velocity components
- Lane: Estimated lane position
- Action: Agent's action at this step
- Reward: Reward received

**Image Statistics**:
- Value range: [min, max] pixel intensities
- Mean: Average pixel intensity
- Non-zero pixels: Count of non-black pixels

### File Naming Convention

**Individual Steps**: `dual_agent_{view_type}_ep_{episode_id}_step_{step:03d}.png`

**Comparison Grids**: `comparison_grid_{episode_id}_steps_{step1}_{step2}_{stepN}.png`

**Quick Extractions**: `agents_step_{step:03d}.png`

## 🎯 Usage Examples

### Example 1: Quick Check of First Few Steps
```bash
cd /home/chettra/ITC/Research/AVs
python3 quick_grayscale.py
```

### Example 2: Detailed Analysis of Specific Steps  
```bash
python3 dual_agent_grayscale.py --steps 0 1 2 5 10 --view-types best rgb_grayscale
```

### Example 3: Compare Agents Over Time
```bash
python3 dual_agent_grayscale.py --comparison-grid --steps 0 2 4 6 8
```

### Example 4: Analyze Different Episode
```bash
python3 dual_agent_grayscale.py --episode ep_dense_commuting_10043_0001 --max-steps 10
```

## 📁 Output Locations

The scripts create output directories relative to your data file:

```
/home/chettra/ITC/Research/AVs/data/highway_multimodal_dataset/dense_commuting/
├── quick_grayscale/           # Quick extractor output
│   ├── agents_step_000.png
│   ├── agents_step_001.png
│   └── agents_step_002.png
├── grayscale_images/          # Full extractor output
│   ├── dual_agent_best_ep_ep_dense_commuting_10042_0000_step_000.png
│   ├── dual_agent_rgb_grayscale_ep_ep_dense_commuting_10042_0000_step_000.png
│   └── comparison_grid_ep_dense_commuting_10042_0000_steps_0_1_2.png
└── extracted_grayscale/       # Enhanced extractor output
    ├── dual_agent_multiview_ep_ep_dense_commuting_10042_0000_step_000.png
    ├── dual_agent_multiview_ep_ep_dense_commuting_10042_0000_step_001.png
    └── dual_agent_multiview_ep_ep_dense_commuting_10042_0000_step_002.png
```

## 🔧 Technical Details

### Image Processing Pipeline

1. **Decode**: Binary blob → numpy array using `BinaryArrayEncoder`
2. **Shape**: [4, 128, 64] → Channels-first RGBA format
3. **Convert**: RGB channels → Grayscale using standard weights (0.299R + 0.587G + 0.114B)
4. **Normalize**: uint8 [0,255] → float32 [0,1] for display
5. **Visualize**: matplotlib with gray colormap

### Data Structure
```python
# Each row in the dataset contains:
{
    'episode_id': 'ep_dense_commuting_10042_0000',
    'step': 0,
    'agent_id': 0,  # or 1
    'grayscale_blob': b'...',      # Compressed image data  
    'grayscale_shape': [4, 128, 64], # RGBA, height, width
    'grayscale_dtype': 'uint8',     # Data type
    'ego_x': 1.0,                   # Agent position
    'ego_y': 0.08,                  # Agent lane position
    # ... other fields
}
```

## 🚨 Troubleshooting

### Common Issues

1. **"No data found"**: Check episode ID and step numbers exist in dataset
2. **"Cannot reshape array"**: Verify blob decoding - use proper `BinaryArrayEncoder`
3. **Black images**: RGB channels might be empty - try individual channel views
4. **Missing fonts**: Unicode car emoji warnings - images still save correctly

### Checking Available Data
```python
import pandas as pd
df = pd.read_parquet("your_data.parquet")
print("Episodes:", df['episode_id'].unique())
print("Steps range:", df['step'].min(), "-", df['step'].max())
print("Agents:", sorted(df['agent_id'].unique()))
```

## 🎨 Customization

You can modify the scripts to:
- Change colormap (e.g., `cmap='viridis'` instead of `'gray'`)
- Adjust figure sizes in `figsize` parameters
- Modify color schemes for agents
- Add additional information overlays
- Change output formats (PNG, PDF, SVG)

## 📝 Notes

- Images are ego-centric: each agent sees from its own perspective
- Grayscale conversion uses standard RGB weights for perceptual accuracy
- All coordinates are normalized relative to each agent's reference frame
- The 4th channel (Alpha) is typically ignored in grayscale conversion
- Processing is optimized for memory efficiency with automatic cleanup

## 🎉 Success Examples

After running the scripts successfully, you should see:
- Side-by-side agent comparisons showing different perspectives
- Clear visualization of how each agent sees the highway environment  
- Agent position and state information overlaid on images
- Temporal progression showing how the scene evolves over time

The extracted images will help you understand:
- How each agent perceives the same environment differently
- Agent positioning and movement patterns
- Environmental features visible to each agent
- Coordination patterns between the two agents