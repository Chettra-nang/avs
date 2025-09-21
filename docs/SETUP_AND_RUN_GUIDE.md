# HighwayEnv Multi-Modal Data Collection System
## Complete Setup and Simulation Guide

This comprehensive guide will walk you through setting up and running the HighwayEnv Multi-Modal Data Collection System from start to finish.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Running Simulations](#running-simulations)
5. [Interactive Demos](#interactive-demos)
6. [Data Collection](#data-collection)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)
9. [Examples and Tutorials](#examples-and-tutorials)

---

## System Requirements

### Operating System
- **Linux** (Ubuntu 18.04+, CentOS 7+)
- **macOS** (10.14+)
- **Windows** (10/11 with WSL2 recommended)

### Hardware
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB minimum, 16GB+ recommended for large datasets
- **GPU**: Optional, but recommended for faster processing
- **Storage**: 10GB+ free space for datasets

### Software Dependencies
- **Python**: 3.8 - 3.12
- **pip**: Latest version
- **Git**: For cloning repositories

---

## Installation

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone <repository-url>
cd highway-multimodal-datacollection

# Or if you already have the files, navigate to the directory
cd /path/to/your/project
```

### Step 2: Create Virtual Environment

```bash
# Create a virtual environment
python -m venv avs_venv

# Activate the virtual environment
# On Linux/macOS:
source avs_venv/bin/activate

# On Windows:
avs_venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install gymnasium
pip install highway-env
pip install pygame
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn

# Install additional dependencies for data processing
pip install pyarrow  # For Parquet files
pip install scikit-learn
pip install tqdm

# Install optional ML dependencies (if you want to use trained models)
pip install stable-baselines3  # Optional: for RL models
pip install torch torchvision  # Optional: for PyTorch models
pip install tensorflow        # Optional: for TensorFlow models
```

### Step 4: Verify Installation

```bash
# Test basic imports
python -c "
import gymnasium as gym
import highway_env
import pygame
import numpy as np
import pandas as pd
print('✓ All core dependencies installed successfully!')
"
```

### Step 5: Set Up Project Structure

```bash
# Create necessary directories
mkdir -p data logs/collection

# Verify project structure
ls -la
# You should see: highway_datacollection/, examples/, docs/, main.py, etc.
```

---

## Quick Start

### Test the System

```bash
# Run the basic system test
python main.py --demo basic
```

This will show you:
- Available scenarios
- System configuration
- Modality options

### Watch Cars Drive

```bash
# Run a simple visual demo
python simple_highway_demo.py
```

This opens a window showing cars driving on the highway automatically.

---

## Running Simulations

### 1. Basic Highway Simulation

Create and run a basic highway simulation:

```python
# Create file: basic_simulation.py
import gymnasium as gym
import highway_env
import time

# Create environment
env = gym.make('highway-v0', render_mode='human')

# Run simulation
obs, info = env.reset()
for step in range(500):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.05)  # Slow down for viewing
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

```bash
# Run the simulation
python basic_simulation.py
```

### 2. Interactive Control

Control a car manually using keyboard:

```bash
# Run interactive simulation
python interactive_highway.py
```

**Controls:**
- `↑/W`: Accelerate
- `↓/S`: Decelerate  
- `←/A`: Change lane left
- `→/D`: Change lane right
- `Space`: Maintain speed
- `ESC`: Quit

### 3. Multi-Agent Simulation

Run simulation with multiple controlled agents:

```python
# Create file: multi_agent_sim.py
import gymnasium as gym
import highway_env
import numpy as np

# Configure multi-agent environment
config = {
    'controlled_vehicles': 3,  # Control 3 vehicles
    'vehicles_count': 50,      # Total vehicles on road
    'duration': 40,            # Simulation duration
    'lanes_count': 4,          # Number of lanes
}

env = gym.make('highway-v0', render_mode='human')
if hasattr(env.unwrapped, 'configure'):
    env.unwrapped.configure(config)

obs, info = env.reset()
for step in range(1000):
    # Actions for each controlled vehicle
    actions = [env.action_space.sample() for _ in range(3)]
    obs, reward, terminated, truncated, info = env.step(actions)
    env.render()
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

---

## Interactive Demos

### 1. System Overview Demo

```bash
# Interactive menu-driven demo
python main.py
```

Choose from:
1. Basic functionality and scenario configurations
2. Complete dataset collection workflow  
3. Data loading and access examples
4. Policy integration for custom action sampling
5. Custom feature extraction capabilities
6. Run all demonstrations

### 2. Specific Feature Demos

```bash
# Basic functionality
python main.py --demo basic

# Policy integration (AI agents)
python main.py --demo policy

# Custom feature extraction
python main.py --demo features

# Data loading examples
python main.py --demo loading

# Complete workflow
python main.py --demo all
```

### 3. Visual Data Collection

```bash
# Watch data collection in action
python visual_data_collection.py
```

---

## Data Collection

### 1. Simple Data Collection

Collect data from highway simulations:

```python
# Create file: collect_data.py
from highway_datacollection.collection.orchestrator import run_full_collection
from pathlib import Path

# Configure collection
result = run_full_collection(
    base_storage_path=Path("data/my_dataset"),
    episodes_per_scenario=10,      # Number of episodes per scenario
    n_agents=2,                    # Number of controlled agents
    max_steps_per_episode=200,     # Maximum steps per episode
    scenarios=["free_flow", "dense_commuting"],  # Scenarios to collect
    base_seed=42                   # Random seed for reproducibility
)

print(f"✓ Collected {result.successful_episodes} episodes")
print(f"✓ Dataset saved to: {result.dataset_index_path}")
```

```bash
# Run data collection
python collect_data.py
```

### 2. Large-Scale Collection

For collecting large datasets:

```python
# Create file: large_collection.py
from highway_datacollection.collection.orchestrator import run_full_collection
from pathlib import Path

def progress_callback(progress):
    """Show collection progress"""
    print(f"Scenario {progress.current_scenario}: "
          f"Episode {progress.current_episode}/{progress.total_episodes}")

# Large-scale collection
result = run_full_collection(
    base_storage_path=Path("data/large_dataset"),
    episodes_per_scenario=100,     # More episodes
    n_agents=3,                    # More agents
    max_steps_per_episode=500,     # Longer episodes
    scenarios=["free_flow", "dense_commuting", "stop_and_go", 
               "aggressive_neighbors", "lane_closure", "time_budget"],
    batch_size=20,                 # Process in batches
    progress_callback=progress_callback
)
```

### 3. Custom Policy Collection

Collect data using your own AI policies:

```python
# Create file: policy_collection.py
from highway_datacollection.collection.action_samplers import ActionSampler
from highway_datacollection.collection.collector import SynchronizedCollector
from highway_datacollection.environments.factory import MultiAgentEnvFactory
import numpy as np

class MyPolicy(ActionSampler):
    """Custom policy for data collection"""
    
    def sample_actions(self, observations, n_agents, step=0, episode_id=""):
        """Sample actions based on observations"""
        actions = []
        kin_obs = observations.get('kinematics', np.zeros((n_agents, 5, 5)))
        
        for agent_idx in range(n_agents):
            # Simple policy: maintain speed, avoid collisions
            agent_obs = kin_obs[agent_idx] if len(kin_obs.shape) > 2 else kin_obs
            
            # Check if there's a vehicle ahead
            if agent_obs.shape[0] > 1:
                ego = agent_obs[0]
                others = agent_obs[1:]
                
                # Simple collision avoidance
                close_vehicles = others[others[:, 1] - ego[1] < 20]  # Within 20m
                if len(close_vehicles) > 0:
                    action = 0  # SLOWER
                else:
                    action = 1  # IDLE (maintain speed)
            else:
                action = 1  # IDLE
            
            actions.append(action)
        
        return tuple(actions)
    
    def reset(self, seed=None):
        """Reset policy state"""
        if seed is not None:
            np.random.seed(seed)

# Use custom policy for collection
env_factory = MultiAgentEnvFactory()
policy = MyPolicy()
collector = SynchronizedCollector(env_factory, action_sampler=policy)

result = collector.collect_episode_batch(
    scenario_name="free_flow",
    episodes=50,
    seed=42
)

print(f"✓ Collected {len(result.transitions)} transitions using custom policy")
```

---

## Advanced Usage

### 1. Loading and Analyzing Data

```python
# Create file: analyze_data.py
from examples.data_loading_examples import DatasetLoader
import pandas as pd
import matplotlib.pyplot as plt

# Load collected dataset
loader = DatasetLoader("data/my_dataset")

# Get dataset statistics
stats = loader.get_dataset_statistics()
print(f"Total scenarios: {stats['total_scenarios']}")
print(f"Total files: {stats['total_files']}")

# Load specific scenario data
df = loader.load_scenario_data("free_flow")
print(f"Loaded {len(df)} transitions")

# Analyze rewards
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(df['reward'], bins=50, alpha=0.7)
plt.title('Reward Distribution')
plt.xlabel('Reward')
plt.ylabel('Frequency')

# Analyze actions
plt.subplot(1, 2, 2)
action_counts = df['action'].value_counts()
plt.bar(action_counts.index, action_counts.values)
plt.title('Action Distribution')
plt.xlabel('Action')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('data_analysis.png')
plt.show()
```

### 2. Custom Feature Extraction

```python
# Create file: custom_features.py
from main import CustomFeatureExtractor
import numpy as np

# Create custom feature extractor
extractor = CustomFeatureExtractor()

# Example observation (ego vehicle + other vehicles)
observation = np.array([
    [1.0, 100.0, 8.0, 25.0, 0.5],   # Ego: [presence, x, y, vx, vy]
    [1.0, 120.0, 8.0, 20.0, 0.0],   # Vehicle ahead
    [1.0, 80.0, 12.0, 30.0, 0.0],   # Vehicle in right lane
    [1.0, 110.0, 4.0, 22.0, 0.0],   # Vehicle in left lane
])

# Extract features
features = extractor.extract_features(observation, {"scenario": "free_flow"})

print("Extracted Features:")
for name, value in features.items():
    if isinstance(value, float):
        if value == float('inf'):
            print(f"  {name}: ∞")
        else:
            print(f"  {name}: {value:.3f}")
    else:
        print(f"  {name}: {value}")
```

### 3. Performance Monitoring

```python
# Create file: monitor_performance.py
from highway_datacollection.performance.monitor import PerformanceMonitor
from highway_datacollection.performance.profiler import MemoryProfiler
import time

# Monitor performance during operations
monitor = PerformanceMonitor()
profiler = MemoryProfiler()

with monitor.time_context("data_processing"):
    with profiler.profile_context():
        # Your data processing code here
        time.sleep(2)  # Simulate work
        
        # Simulate memory usage
        data = [i for i in range(1000000)]

print(f"Processing time: {monitor.get_timing('data_processing'):.2f}s")
print(f"Peak memory usage: {profiler.get_peak_memory():.2f} MB")
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Environment Creation Fails

**Problem**: `AttributeError: 'OrderEnforcing' object has no attribute 'configure'`

**Solution**:
```python
# Use unwrapped environment
env = gym.make('highway-v0', render_mode='human')
if hasattr(env.unwrapped, 'configure'):
    env.unwrapped.configure(config)
```

#### 2. Display Issues

**Problem**: No window appears or graphics issues

**Solutions**:
```bash
# On Linux, install display dependencies
sudo apt-get install python3-tk
sudo apt-get install xvfb  # For headless systems

# Set display variable if using SSH
export DISPLAY=:0

# For WSL2 on Windows, install VcXsrv or similar X server
```

#### 3. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'highway_env'`

**Solution**:
```bash
# Ensure virtual environment is activated
source avs_venv/bin/activate  # Linux/macOS
# or
avs_venv\Scripts\activate     # Windows

# Reinstall highway-env
pip install highway-env
```

#### 4. Performance Issues

**Problem**: Simulation runs slowly

**Solutions**:
```python
# Reduce rendering frequency
env.configure({"simulation_frequency": 5})  # Lower frequency

# Disable real-time rendering for data collection
env.configure({"real_time_rendering": False})

# Use smaller batch sizes
result = run_full_collection(batch_size=10)  # Smaller batches
```

#### 5. Memory Issues

**Problem**: Out of memory during large collections

**Solutions**:
```python
# Process scenarios sequentially
for scenario in scenarios:
    result = run_full_collection(
        scenarios=[scenario],  # One at a time
        episodes_per_scenario=50
    )

# Use memory profiling
from highway_datacollection.performance.profiler import MemoryProfiler
profiler = MemoryProfiler()
# Monitor memory usage
```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run your code with debug output
```

---

## Examples and Tutorials

### Tutorial 1: Basic Simulation

```bash
# 1. Start with simple demo
python simple_highway_demo.py

# 2. Try interactive control
python interactive_highway.py

# 3. Run system demos
python main.py --demo basic
```

### Tutorial 2: Data Collection Workflow

```bash
# 1. Collect small dataset
python -c "
from highway_datacollection.collection.orchestrator import run_full_collection
from pathlib import Path

result = run_full_collection(
    base_storage_path=Path('data/tutorial'),
    episodes_per_scenario=5,
    scenarios=['free_flow']
)
print(f'Collected {result.successful_episodes} episodes')
"

# 2. Load and examine data
python examples/data_loading_examples.py

# 3. Analyze results
python -c "
from examples.data_loading_examples import DatasetLoader
loader = DatasetLoader('data/tutorial')
stats = loader.get_dataset_statistics()
print('Dataset Statistics:', stats)
"
```

### Tutorial 3: Custom Policy Development

```bash
# 1. Study policy examples
python examples/policy_integration_examples.py

# 2. Test policy integration
python main.py --demo policy

# 3. Create your own policy (see custom_features.py example above)
```

### Tutorial 4: Advanced Analysis

```bash
# 1. Run feature extraction demo
python main.py --demo features

# 2. Analyze collected data
python analyze_data.py

# 3. Monitor performance
python monitor_performance.py
```

---

## File Structure Reference

```
highway-multimodal-datacollection/
├── highway_datacollection/          # Core system modules
│   ├── collection/                  # Data collection components
│   ├── environments/               # Environment factory
│   ├── features/                   # Feature extractors
│   ├── storage/                    # Data storage management
│   └── performance/                # Performance monitoring
├── examples/                       # Example scripts
│   ├── data_loading_examples.py    # Data loading utilities
│   ├── policy_integration_examples.py  # Policy examples
│   └── complete_workflow_demo.py   # End-to-end workflow
├── docs/                          # Documentation
├── tests/                         # Test suite
├── data/                          # Generated datasets (created during use)
├── logs/                          # Log files (created during use)
├── main.py                        # Main demonstration script
├── simple_highway_demo.py         # Simple visual demo
├── interactive_highway.py         # Interactive control demo
├── visual_data_collection.py      # Visual collection demo
└── requirements.txt               # Python dependencies
```

---

## Next Steps

1. **Start Simple**: Begin with `python simple_highway_demo.py`
2. **Explore Demos**: Run `python main.py` to see all capabilities
3. **Collect Data**: Use the data collection system for your research
4. **Customize**: Develop your own policies and feature extractors
5. **Analyze**: Use the data loading and analysis tools
6. **Scale Up**: Run large-scale data collection for your projects

## Support and Resources

- **Documentation**: See `docs/` directory for detailed guides
- **Examples**: Check `examples/` directory for code samples
- **Issues**: Check troubleshooting section above
- **Performance**: Use monitoring tools for optimization

---

**Happy Simulating! 🚗🛣️**

The HighwayEnv Multi-Modal Data Collection System provides a comprehensive platform for autonomous vehicle research, reinforcement learning, and traffic simulation studies.