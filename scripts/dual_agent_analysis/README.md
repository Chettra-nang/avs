# 🚗🚗 Dual Agent Analysis Scripts

Scripts for analyzing and visualizing dual-agent autonomous vehicle behavior.

## 📄 Scripts

### **quick_dual_video.py**
- **Purpose**: Quick dual-agent video creation
- **Usage**: `python3 quick_dual_video.py`
- **Features**: 
  - Fast video generation
  - Side-by-side agent views
  - Position tracking
  - Basic statistics
  - Automatic episode selection

### **dual_agent_video_visualizer.py**
- **Purpose**: Advanced dual-agent video visualization
- **Usage**: `python3 dual_agent_video_visualizer.py --data path/to/data.parquet [options]`
- **Features**:
  - Comprehensive dashboard
  - Multiple visualization types
  - Sensing comparison videos
  - Custom episode selection
  - Advanced statistics

### **show_sensing_comparison.py**
- **Purpose**: Static sensing comparison analysis
- **Usage**: `python3 show_sensing_comparison.py`
- **Features**:
  - Pixel-level comparison
  - Statistical analysis
  - Heat map visualization
  - Detailed metrics

### **dual_agent_summary.py**
- **Purpose**: Summary of dual-agent analysis results
- **Usage**: `python3 dual_agent_summary.py`
- **Features**:
  - Comprehensive analysis summary
  - Key findings presentation
  - Evidence compilation

## 🎬 Video Types Created

1. **Main Dual Agent Video**: Shows both agents with position tracking and statistics
2. **Sensing Comparison Video**: Focuses on visual differences between agents
3. **Static Analysis**: Single-frame detailed comparison

## 🚀 Quick Start

```bash
# Quick video
python3 quick_dual_video.py

# Advanced video with specific episode
python3 dual_agent_video_visualizer.py \
  --data "../../data/highway_multimodal_dataset/dense_commuting/20250921_152749-1ea1c024_transitions.parquet" \
  --episode "ep_dense_commuting_10042_0000" \
  --max-steps 15 \
  --sensing

# Static comparison
python3 show_sensing_comparison.py

# View summary
python3 dual_agent_summary.py
```

## 📁 Output Structure

Videos and images are saved to `../../output/` with these subdirectories:
- `dual_agent_videos/` - Advanced visualizations
- `quick_videos/` - Quick analysis results

## 🎯 Key Findings

The scripts prove that the dataset contains:
✅ 2 separate autonomous vehicles  
✅ 2 independent AI brains  
✅ Same highway environment  
✅ Different ego-centric sensing  

## 📚 Documentation

See `../../docs/DUAL_AGENT_VIDEO_GUIDE.md` for complete usage instructions and interpretation guide.