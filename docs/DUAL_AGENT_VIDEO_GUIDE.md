# 🎬 Dual Agent Video Visualization Guide

This guide explains how to create and interpret videos showing **2 independent autonomous vehicles** operating in the same highway environment.

## 🚗🚗 What You're Seeing

### **The Setup**
- **Agent 0** = Autonomous Car A with Brain A  
- **Agent 1** = Autonomous Car B with Brain B
- Both cars drive in the same highway simulation
- They can see and interact with each other
- Each makes independent decisions

### **Key Evidence This Is 2 Separate Cars**
✅ **Different Positions**: Cars move to different locations after initial steps  
✅ **Independent Actions**: Each car makes its own driving decisions  
✅ **Same Environment**: Both get identical rewards (shared traffic conditions)  
✅ **Ego-Centric Vision**: Each car sees the highway from its own perspective  

## 📁 Generated Videos

### 1. **Quick Overview Video**
```
output/quick_videos/ep_dense_commuting_10042_0000_dual_agents.gif
```
- **What it shows**: Side-by-side view of both cars' vision
- **Key elements**: Position tracking, statistics, visual differences
- **Best for**: Quick understanding of the dual-agent setup

### 2. **Comprehensive Dual Agent Video**
```
output/dual_agent_videos/ep_dense_commuting_10042_0000_dual_agent_video.gif  
```
- **What it shows**: Full dashboard with images, position tracking, and detailed stats
- **Key elements**: 
  - Top: Both agents' camera views
  - Middle: Real-time position tracking on coordinate system
  - Bottom: Detailed statistics showing rewards, actions, distances
- **Best for**: Detailed analysis of agent behavior

### 3. **Sensing Comparison Video**
```
output/dual_agent_videos/ep_dense_commuting_10042_0000_sensing_comparison.gif
```
- **What it shows**: Focus on sensory differences between agents
- **Key elements**:
  - Agent 0 vision vs Agent 1 vision
  - Pixel-level differences (heatmap)
  - Position, reward, and action tracking
- **Best for**: Understanding how different perspectives affect decision-making

### 4. **Individual Frame Images**
```
output/quick_videos/ep_dense_commuting_10042_0000_frames/step_XXX.png
```
- **What it shows**: Static frame-by-frame comparison
- **Best for**: Detailed inspection of specific moments

## 🔍 How to Interpret the Videos

### **Position Tracking Plot**
- **Blue line/dots**: Agent 0 (Car A) movement path
- **Red line/dots**: Agent 1 (Car B) movement path
- **X-axis**: Longitudinal position on highway
- **Y-axis**: Lateral position (lane changes)

### **Statistics Panel**
```
Agent 0 (Car A)     │  Agent 1 (Car B)
Position: (x, y)    │  Position: (x, y)     <- Different = separate cars
Velocity: (vx, vy)  │  Velocity: (vx, vy)   <- Independent movement  
Speed: X.XXX        │  Speed: X.XXX         <- Different speeds
Reward: X.XXXX      │  Reward: X.XXXX       <- SAME = shared environment
Action: [X X]       │  Action: [X X]        <- Independent decisions
```

### **Key Indicators**
- **Same Rewards**: ✅ Confirms shared environment
- **Different Positions**: ✅ Confirms separate vehicles
- **Distance Apart**: Shows how far the cars are from each other
- **Different Images**: Shows ego-centric perspectives

## 🛠️ Creating Your Own Videos

### **Quick Video (Simple)**
```bash
python3 quick_dual_video.py
```
- Creates basic dual-agent visualization
- Automatic episode selection
- Fast generation

### **Advanced Video (Full Control)**
```bash
python3 dual_agent_video_visualizer.py \
  --data "path/to/your/data.parquet" \
  --episode "ep_dense_commuting_10042_0000" \
  --max-steps 15 \
  --sensing \
  --fps 1
```

**Parameters:**
- `--data`: Path to your parquet dataset
- `--episode`: Specific episode ID (optional)
- `--max-steps`: Number of steps to include
- `--sensing`: Create sensing comparison video
- `--fps`: Animation speed (frames per second)

## 🎯 What the Videos Prove

### **Question**: "Are these 2 agents in the same environment and sense?"

### **Answer**: 
- **✅ Same Environment**: Both cars drive in identical highway simulation
- **❌ Different Sensing**: Each car has ego-centric observations

### **Practical Meaning**:
```
🛣️ SHARED HIGHWAY ENVIRONMENT
     ┌─────────────────────────┐
     │  🚗 Agent 0 (Car A)     │ ← Sees highway from position A
     │      ↕️ 0.05 units       │
     │  🚗 Agent 1 (Car B)     │ ← Sees highway from position B  
     │  🚛 Other traffic...    │
     └─────────────────────────┘
```

## 🔬 Technical Details

### **Image Processing**
- Original images: 4-channel (RGBA), 128x64 pixels
- Conversion: RGB to grayscale using standard weights (0.299*R + 0.587*G + 0.114*B)
- Display: Normalized to [0, 255] range

### **Coordinate System**
- **X-axis**: Forward/backward movement on highway
- **Y-axis**: Left/right movement (lane changes)
- **Units**: Normalized simulation coordinates

### **Rewards**
- Identical rewards confirm shared environment
- Based on overall traffic flow, safety, efficiency
- Both cars affected by same external conditions

## 🎥 Video Formats

### **GIF Files** (Recommended)
- ✅ Universal compatibility  
- ✅ No additional software needed
- ✅ Can be viewed in browsers, markdown files
- ✅ Loop automatically

### **MP4 Files** (If available)
- Higher quality
- Smaller file sizes
- Requires video player

## 🚀 Next Steps

1. **Watch the videos** to see both agents in action
2. **Compare different episodes** to see various traffic scenarios
3. **Analyze specific moments** using individual frames
4. **Create custom visualizations** for your research needs

## 📊 Example Analysis Questions

Use the videos to answer:
- How do the cars coordinate when close together?
- What happens when one car changes lanes?
- How do different visual perspectives affect decisions?
- When do the cars make similar vs different choices?

---

**🎬 Ready to explore multi-agent autonomous vehicle behavior!**  
Your videos are saved and ready to watch! 🚗🤖🚗