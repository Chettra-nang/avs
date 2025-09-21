# 🚗 Highway Data Collection System - Step-by-Step Tutorial

## 📋 **Table of Contents**
1. [Quick Start (5 minutes)](#quick-start)
2. [Understanding Your System](#understanding-your-system)
3. [Basic Data Collection](#basic-data-collection)
4. [Advanced Configuration](#advanced-configuration)
5. [Loading and Using Data](#loading-and-using-data)
6. [Troubleshooting](#troubleshooting)

---

## 🚀 **Quick Start (5 minutes)**

### **Step 1: Run Your First Collection**
```bash
# Activate your environment
source avs_venv/bin/activate

# Run basic demo (collects small dataset)
python main.py --demo collection
```

**What this does:**
- ✅ Creates a small dataset with 3 episodes per scenario
- ✅ Collects from 2 scenarios: free_flow, dense_commuting
- ✅ Saves data to `data/highway_multimodal_dataset/`
- ✅ Takes about 2-3 minutes

### **Step 2: Check Your Data**
```bash
# Load and explore the data
python main.py --demo loading
```

**What you'll see:**
- 📊 Dataset statistics (episodes, agents, steps)
- 📁 File structure and locations
- 🔍 Sample data preview
- 📈 Feature analysis

---

## 🎯 **Understanding Your System**

### **What Your System Collects:**

| Data Type | Description | Use Case |
|-----------|-------------|----------|
| **Kinematics** | Vehicle positions, velocities | RL training, trajectory analysis |
| **Occupancy Grids** | Spatial traffic representation | Path planning, collision detection |
| **Grayscale Images** | Visual observations | Computer vision, perception |
| **Features** | TTC, lane position, traffic density | Safety analysis, behavior modeling |
| **Natural Language** | Human-readable descriptions | Explainable AI, documentation |

### **System Architecture:**
```
Highway Environment → Multi-Modal Collector → Data Storage
     ↓                        ↓                    ↓
Traffic Simulation    Observations + Actions    Parquet Files
Multiple Scenarios    Feature Extraction       JSON Metadata
```

---

## 📊 **Basic Data Collection**

### **Step 1: Choose Your Configuration**

#### **Option A: Quick Demo (Recommended for Learning)**
```bash
python main.py --demo collection
```
- 3 episodes per scenario
- 2 scenarios (free_flow, dense_commuting)
- ~2-3 minutes runtime

#### **Option B: Small Research Dataset**
```python
# Create custom_collection.py
from highway_datacollection.collection.orchestrator import run_full_collection
from pathlib import Path

result = run_full_collection(
    base_storage_path=Path("data/my_dataset"),
    episodes_per_scenario=10,        # More episodes
    n_agents=2,                      # Number of controlled agents
    max_steps_per_episode=100,       # Longer episodes
    scenarios=["free_flow", "dense_commuting", "stop_and_go"],
    base_seed=42,                    # Reproducible results
    batch_size=5                     # Process 5 episodes at once
)

print(f"Collected {result.successful_episodes} episodes!")
```

#### **Option C: Full Production Dataset**
```python
# For serious research - takes 30+ minutes
result = run_full_collection(
    base_storage_path=Path("data/full_dataset"),
    episodes_per_scenario=100,       # Lots of data
    n_agents=3,                      # More agents
    max_steps_per_episode=200,       # Long episodes
    scenarios=["free_flow", "dense_commuting", "stop_and_go", 
              "aggressive_neighbors", "lane_closure"],
    base_seed=42,
    batch_size=10
)
```

### **Step 2: Monitor Collection Progress**

When running collection, you'll see:
```
[free_flow] Episode 1/10 (Success: 0, Failed: 0) - Scenario 1/3
[free_flow] Episode 2/10 (Success: 1, Failed: 0) - Scenario 1/3
...
✓ Collection completed successfully!
Total episodes: 30
Successful episodes: 28
Collection time: 156.2s
```

### **Step 3: Verify Your Data**

Check the generated files:
```bash
ls -la data/highway_multimodal_dataset/
# You should see:
# - index.json (dataset catalog)
# - free_flow/ (scenario folder)
# - dense_commuting/ (scenario folder)
# - stop_and_go/ (if included)
```

---

## ⚙️ **Advanced Configuration**

### **Scenario Selection**

Available scenarios:
```python
from highway_datacollection import ScenarioRegistry

registry = ScenarioRegistry()
scenarios = registry.list_scenarios()
print(scenarios)
# ['free_flow', 'dense_commuting', 'stop_and_go', 'aggressive_neighbors', 'lane_closure']
```

**Scenario Descriptions:**
- **free_flow**: Light traffic, smooth driving
- **dense_commuting**: Heavy traffic, frequent lane changes
- **stop_and_go**: Congested traffic with stops
- **aggressive_neighbors**: Aggressive driving behaviors
- **lane_closure**: Merging scenarios

### **Modality Selection**

Choose what data to collect:
```python
# In your collection script
from highway_datacollection.collection.collector import SynchronizedCollector

# Collect only kinematics (fastest)
collector = SynchronizedCollector(
    modalities=["kinematics"]
)

# Collect kinematics + vision (computer vision research)
collector = SynchronizedCollector(
    modalities=["kinematics", "grayscale"]
)

# Collect everything (complete dataset)
collector = SynchronizedCollector(
    modalities=["kinematics", "occupancy_grid", "grayscale"]
)
```

### **Custom Episode Configuration**

```python
# Longer episodes for behavior analysis
result = run_full_collection(
    max_steps_per_episode=300,    # 5 minutes at 15 Hz
    duration=20,                  # 20 seconds simulation time
)

# More agents for multi-agent research
result = run_full_collection(
    n_agents=5,                   # 5 controlled vehicles
    controlled_vehicles=5
)

# Specific random seeds for reproducibility
result = run_full_collection(
    base_seed=12345,              # Starting seed
    episodes_per_scenario=50      # Seeds: 12345, 12346, ..., 12394
)
```

---

## 📁 **Loading and Using Data**

### **Step 1: Basic Data Loading**

```python
import pandas as pd
import json
from pathlib import Path

# Load dataset index
dataset_path = Path("data/highway_multimodal_dataset")
with open(dataset_path / "index.json", 'r') as f:
    index = json.load(f)

print(f"Dataset has {len(index['scenarios'])} scenarios")

# Load specific scenario
scenario_name = "free_flow"
parquet_files = list((dataset_path / scenario_name).glob("*_transitions.parquet"))

dfs = []
for file in parquet_files:
    df = pd.read_parquet(file)
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(combined_df)} samples")
```

### **Step 2: Extract Features for ML**

```python
# Get input features (observations)
feature_columns = [
    'ego_x', 'ego_y', 'ego_vx', 'ego_vy',  # Position and velocity
    'speed', 'ttc', 'lane_position',        # Derived features
    'traffic_density', 'vehicle_count'       # Context features
]

X = combined_df[feature_columns].values

# Get labels (actions)
y = combined_df['action'].apply(
    lambda x: x[0] if isinstance(x, list) else x
).values

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
```

### **Step 3: Reconstruct Visual Data**

```python
import numpy as np

def reconstruct_occupancy_grid(row):
    """Reconstruct occupancy grid from binary data"""
    if pd.isna(row['occ_blob']):
        return None
    
    blob_data = row['occ_blob']
    shape = eval(row['occ_shape'])
    dtype = row['occ_dtype']
    
    return np.frombuffer(blob_data, dtype=dtype).reshape(shape)

# Example usage
sample_row = combined_df.iloc[0]
occupancy_grid = reconstruct_occupancy_grid(sample_row)
print(f"Occupancy grid shape: {occupancy_grid.shape}")
```

### **Step 4: Analyze Your Data**

```python
# Episode statistics
print("Dataset Overview:")
print(f"- Episodes: {combined_df['episode_id'].nunique()}")
print(f"- Agents: {combined_df['agent_id'].nunique()}")
print(f"- Total steps: {len(combined_df)}")
print(f"- Average episode length: {len(combined_df) / combined_df['episode_id'].nunique():.1f}")

# Action distribution
action_counts = combined_df['action'].apply(
    lambda x: x[0] if isinstance(x, list) else x
).value_counts()

action_names = {0: "SLOWER", 1: "IDLE", 2: "FASTER", 3: "LANE_LEFT", 4: "LANE_RIGHT"}
print("\nAction Distribution:")
for action, count in action_counts.items():
    name = action_names.get(action, f"Action_{action}")
    percentage = (count / len(combined_df)) * 100
    print(f"- {name}: {count} ({percentage:.1f}%)")

# Safety analysis
if 'ttc' in combined_df.columns:
    finite_ttc = combined_df[combined_df['ttc'] != np.inf]['ttc']
    print(f"\nSafety Metrics:")
    print(f"- Average TTC: {finite_ttc.mean():.2f}s")
    print(f"- Dangerous situations (TTC < 2s): {(finite_ttc < 2).sum()}")
```

---

## 🔧 **Troubleshooting**

### **Common Issues and Solutions**

#### **Issue 1: "No module named 'highway_env'"**
```bash
# Solution: Install highway-env
pip install highway-env
```

#### **Issue 2: "Permission denied" or file errors**
```bash
# Solution: Check permissions and create directories
mkdir -p data logs
chmod 755 data logs
```

#### **Issue 3: Collection runs but no data saved**
```python
# Check if collection actually succeeded
result = run_full_collection(...)
print(f"Success: {result.successful_episodes}/{result.total_episodes}")
print(f"Errors: {result.errors}")
```

#### **Issue 4: Out of memory during collection**
```python
# Solution: Reduce batch size and episode length
result = run_full_collection(
    batch_size=2,                    # Smaller batches
    max_steps_per_episode=50,        # Shorter episodes
    episodes_per_scenario=5          # Fewer episodes
)
```

#### **Issue 5: Data loading fails**
```python
# Check if files exist
dataset_path = Path("data/highway_multimodal_dataset")
if not dataset_path.exists():
    print("Dataset directory not found!")
    
index_path = dataset_path / "index.json"
if not index_path.exists():
    print("Dataset index not found - run collection first!")
```

### **Performance Tips**

1. **Start Small**: Begin with 3-5 episodes per scenario
2. **Monitor Resources**: Watch CPU and memory usage
3. **Use Batch Processing**: Set appropriate batch_size (2-10)
4. **Choose Modalities Wisely**: Only collect what you need
5. **Parallel Processing**: The system automatically uses multiple cores

### **Data Quality Checks**

```python
# Check for missing data
print("Missing data check:")
for col in combined_df.columns:
    missing = combined_df[col].isna().sum()
    if missing > 0:
        print(f"- {col}: {missing} missing values")

# Check episode completeness
episode_lengths = combined_df.groupby('episode_id')['step'].max()
print(f"Episode lengths: min={episode_lengths.min()}, max={episode_lengths.max()}, avg={episode_lengths.mean():.1f}")

# Check for data consistency
print(f"Unique scenarios: {combined_df['scenario'].unique() if 'scenario' in combined_df.columns else 'Not available'}")
```

---

## 🎯 **Next Steps**

After collecting data, you can:

1. **Train ML Models**: Use `python train_model_example.py`
2. **Visualize Data**: Create plots and analysis
3. **Export Data**: Convert to other formats (CSV, HDF5)
4. **Integrate with Research**: Use in your autonomous vehicle research
5. **Scale Up**: Collect larger datasets for production use

## 📚 **Additional Resources**

- **Main Script**: `python main.py --help` for all options
- **Examples**: Check the `examples/` directory
- **Documentation**: See `docs/` for detailed guides
- **Training**: Use `train_model_example.py` for ML workflows

---

**Happy Data Collecting! 🚗📊**