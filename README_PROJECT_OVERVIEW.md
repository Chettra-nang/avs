# 🚗 AVS Research Project - Dual Agent Highway Analysis

**Multi-Agent Autonomous Vehicle Systems Research Platform**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This research project focuses on analyzing **dual-agent autonomous vehicle behavior** in highway environments. We've proven that our dataset contains **2 separate cars with independent AI brains** operating in the same traffic simulation, each with different ego-centric sensing capabilities.

### ✅ Key Discoveries
- **2 Independent Autonomous Vehicles**: Agent 0 (Car A) and Agent 1 (Car B)
- **Same Highway Environment**: Shared traffic simulation with identical rewards
- **Different Ego-Centric Sensing**: Each car sees the highway from its own perspective  
- **Independent Decision Making**: Each AI brain makes autonomous driving decisions

## 🚀 Quick Start

### Prerequisites
```bash
# Clone the repository
git clone https://github.com/Chettra-nang/avs.git
cd avs

# Install dependencies
pip install -r requirements.txt
```

### Quick Analysis
```bash
# Generate dual-agent videos
python3 scripts/dual_agent_analysis/quick_dual_video.py

# Create sensing comparison
python3 scripts/dual_agent_analysis/show_sensing_comparison.py

# View analysis summary
python3 scripts/dual_agent_analysis/dual_agent_summary.py
```

## 📁 Project Structure

```
├── README.md                      # This overview file
├── README_ORIGINAL.md             # 📚 Comprehensive research documentation
├── main.py                        # Main entry point
├── scripts/                       # 🔧 Analysis tools
│   ├── dual_agent_analysis/       # 🚗🚗 Dual agent videos & analysis
│   ├── grayscale_extraction/       # 🎨 Image extraction tools
│   └── testing/                   # 🧪 Test scripts
├── docs/                         # 📖 Documentation
├── output/                       # 📊 Generated visualizations
└── data/                         # 📈 Highway multimodal dataset
```

## 🎬 Generated Visualizations

The project creates several types of visualizations:

### **Videos Created:**
- **Quick Dual Agent Video** (`output/quick_videos/`) - Side-by-side agent comparison
- **Advanced Dual Agent Video** (`output/dual_agent_videos/`) - Comprehensive dashboard
- **Sensing Comparison Video** - Visual differences analysis

### **Key Findings Visualized:**
- ✅ Same environment (identical rewards: 0.733769)
- ✅ Different cars (positions 0.0242 units apart)
- ✅ Different sensing (23.7% of pixels differ)
- ✅ Independent brains (autonomous decision making)

## 🔬 Research Applications

This platform supports research in:
- **Multi-Agent Reinforcement Learning**
- **Autonomous Vehicle Coordination**
- **Computer Vision and Perception**
- **Safety-Critical AI Systems**
- **Multi-Modal Learning**

## 📊 Dataset Information

**Format**: Apache Parquet (.parquet) for efficient data analysis
**Content**: Synchronized multi-modal observations from 2 agents
**Scenarios**: Highway driving with various traffic conditions
**Features**: Kinematics, visual observations, safety metrics

## 🛠️ Available Tools

### **Dual Agent Analysis**
- `quick_dual_video.py` - Fast video generation
- `dual_agent_video_visualizer.py` - Advanced visualizations
- `show_sensing_comparison.py` - Static analysis
- `dual_agent_summary.py` - Results summary

### **Grayscale Extraction**
- `quick_grayscale.py` - Simple image extraction
- `dual_agent_grayscale.py` - Comprehensive analysis
- `enhanced_grayscale_extractor.py` - Advanced features

## 📚 Documentation

- **[README_ORIGINAL.md](README_ORIGINAL.md)** - Complete research documentation with technical details
- **[DUAL_AGENT_VIDEO_GUIDE.md](docs/DUAL_AGENT_VIDEO_GUIDE.md)** - Video visualization guide
- **[GRAYSCALE_EXTRACTION_GUIDE.md](docs/GRAYSCALE_EXTRACTION_GUIDE.md)** - Image extraction guide

## 🎯 Quick Examples

### Create Dual Agent Video
```python
from scripts.dual_agent_analysis.quick_dual_video import create_quick_dual_agent_video
create_quick_dual_agent_video()
```

### Extract Grayscale Images
```python
from scripts.grayscale_extraction.quick_grayscale import extract_and_display_grayscale
extract_and_display_grayscale()
```

### Analyze Data
```python
import pandas as pd
df = pd.read_parquet("data/highway_multimodal_dataset/dense_commuting/transitions.parquet")
print(f"Dataset contains {len(df)} transitions from 2 agents")
```

## 🚗🤖 What Makes This Special

This project definitively proves through data analysis and visualization that:

1. **Multi-Agent Setup**: Two independent AI-controlled vehicles
2. **Shared Environment**: Both cars experience the same traffic simulation  
3. **Different Perspectives**: Each car has ego-centric observations
4. **Independent Intelligence**: Separate decision-making systems
5. **Rich Dataset**: Comprehensive multi-modal data for research

## 💡 Use Cases

- **Academic Research**: Multi-agent autonomous vehicle studies
- **Algorithm Development**: Testing coordination strategies
- **Safety Analysis**: Understanding vehicle interactions
- **Computer Vision**: Multi-perspective image analysis
- **Reinforcement Learning**: Multi-agent training environments

## 📈 Getting Started with Research

1. **Explore the data**: Load and analyze the Parquet datasets
2. **Generate videos**: See both agents in action
3. **Run comparisons**: Understand sensing differences
4. **Read the docs**: Dive deep with README_ORIGINAL.md
5. **Extend the tools**: Add your own analysis scripts

## 🤝 Contributing

This is a research platform designed for autonomous vehicle studies. Feel free to:
- Add new analysis tools
- Enhance visualizations  
- Contribute documentation
- Share research findings

## 📄 License

MIT License - see LICENSE file for details.

## 📧 Contact

**Research Project**: AVS (Autonomous Vehicle Systems)
**Focus**: Dual-agent highway behavior analysis
**Platform**: Multi-modal data collection and analysis

---

**🚗🤖 Ready to explore multi-agent autonomous vehicle research!**

> For detailed technical documentation, architecture details, and comprehensive research context, see [README_ORIGINAL.md](README_ORIGINAL.md)