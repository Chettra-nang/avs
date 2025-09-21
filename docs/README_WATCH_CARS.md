# Watch Cars Drive Automatically 🚗🛣️

A comprehensive highway simulation viewer that lets you watch intelligent cars drive automatically with real-time statistics, data collection, and multiple traffic scenarios.

## 🎯 Features

- **🚗 Intelligent Car Behavior**: Cars make smart decisions based on traffic conditions
- **📊 Real-time Statistics**: Live display of performance metrics
- **🎬 Multiple Scenarios**: 5 different traffic scenarios to choose from
- **💾 Data Collection**: Save simulation data for analysis
- **📈 Visualization**: Generate plots and statistics
- **⚙️ Configurable**: Customize agents, duration, and parameters
- **🎮 Easy to Use**: Simple command-line interface

## 🚀 Quick Start

### Option 1: Automatic Setup (Recommended)

```bash
# Run the setup script
python setup_watch_cars.py

# Start watching cars (Windows)
run_watch_cars.bat

# Start watching cars (Linux/macOS)
./run_watch_cars.sh
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv_watch_cars

# Activate virtual environment
# Windows:
venv_watch_cars\Scripts\activate
# Linux/macOS:
source venv_watch_cars/bin/activate

# Install dependencies
pip install -r requirements_watch_cars.txt

# Run the simulation
python watch_cars_auto.py
```

## 🎮 Usage Examples

### Basic Usage

```bash
# Default simulation (free flow traffic)
python watch_cars_auto.py

# Dense traffic scenario
python watch_cars_auto.py --scenario dense

# Control 5 agents for 2 minutes
python watch_cars_auto.py --agents 5 --duration 120

# Collect data while watching
python watch_cars_auto.py --collect-data

# Run specific number of episodes
python watch_cars_auto.py --episodes 10
```

### Advanced Usage

```bash
# List all available scenarios
python watch_cars_auto.py --list-scenarios

# Dense traffic with data collection and no stats overlay
python watch_cars_auto.py --scenario dense --collect-data --no-stats

# Long simulation with multiple agents
python watch_cars_auto.py --scenario aggressive --agents 3 --max-steps 10000

# Stop-and-go traffic for research
python watch_cars_auto.py --scenario stop_go --episodes 5 --collect-data
```

## 🛣️ Available Scenarios

| Scenario | Description | Vehicles | Lanes | Agents |
|----------|-------------|----------|-------|---------|
| `free_flow` | Light traffic with smooth flow | 20 | 4 | 1 |
| `dense` | Heavy commuter traffic | 80 | 4 | 2 |
| `stop_go` | Congested traffic with frequent stops | 60 | 3 | 2 |
| `aggressive` | Traffic with aggressive lane-changing | 45 | 4 | 3 |
| `lane_closure` | Traffic merging due to lane closure | 55 | 3 | 2 |

## 🧠 Intelligent Car Behavior

The cars use intelligent decision-making based on:

- **🚦 Traffic Awareness**: Detect vehicles ahead and adjust speed
- **🛣️ Lane Management**: Smart lane changing when safe
- **⚡ Speed Control**: Maintain optimal speed for conditions  
- **🚨 Collision Avoidance**: Slow down or change lanes to avoid crashes
- **📏 Distance Keeping**: Maintain safe following distances

### Behavior Logic

```python
# Example of intelligent behavior
if distance_to_lead_car < 15m and approaching_fast:
    if safe_to_change_lanes:
        change_lanes()
    else:
        slow_down()
elif current_speed < optimal_speed:
    speed_up()
else:
    maintain_speed()
```

## 📊 Real-time Statistics

The simulation displays live statistics:

- **🏁 Runtime**: Total simulation time
- **📈 Steps**: Number of simulation steps
- **🎯 Episodes**: Completed episodes
- **🏆 Rewards**: Total and average rewards
- **💥 Collisions**: Number of crashes
- **🔄 Lane Changes**: Lane change count
- **⚡ Speed**: Average and maximum speeds
- **🤖 Agents**: Number of controlled vehicles

## 💾 Data Collection

When enabled with `--collect-data`, the system saves:

### JSON Format
```json
{
  "scenario": "free_flow",
  "statistics": {
    "total_steps": 1500,
    "total_episodes": 3,
    "collision_count": 0,
    "average_speed": 24.5
  },
  "episodes": [
    {
      "step": 0,
      "actions": [1, 2],
      "rewards": [0.8, 0.7],
      "agent_0_speed": 25.3,
      "agent_1_speed": 23.1
    }
  ]
}
```

### CSV Format (with pandas)
| episode | step | action | reward | agent_0_speed | agent_1_speed |
|---------|------|--------|--------|---------------|---------------|
| 0 | 0 | 1 | 0.8 | 25.3 | 23.1 |
| 0 | 1 | 2 | 0.7 | 26.1 | 24.0 |

## 📈 Visualization

Automatic generation of:

- **📊 Reward Plots**: Rewards over time
- **📏 Episode Length Distribution**: Episode duration histogram  
- **📈 Cumulative Rewards**: Total reward accumulation
- **📋 Statistics Summary**: Complete performance overview

## ⚙️ Configuration Options

### Command Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--scenario` | Traffic scenario | `--scenario dense` |
| `--agents` | Number of controlled agents | `--agents 5` |
| `--duration` | Simulation duration (seconds) | `--duration 120` |
| `--max-steps` | Maximum simulation steps | `--max-steps 10000` |
| `--episodes` | Target number of episodes | `--episodes 10` |
| `--collect-data` | Enable data collection | `--collect-data` |
| `--no-stats` | Disable statistics display | `--no-stats` |
| `--list-scenarios` | List available scenarios | `--list-scenarios` |

### Environment Variables

```bash
# Set display for headless systems
export DISPLAY=:0

# Disable pygame welcome message
export PYGAME_HIDE_SUPPORT_PROMPT=1
```

## 🔧 Requirements

### System Requirements
- **OS**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Display**: Graphics display (or X11 forwarding for remote)

### Core Dependencies
```
gymnasium>=0.28.0    # Reinforcement learning environments
highway-env>=1.8.0   # Highway driving environment
pygame>=2.1.0        # Graphics and display
numpy>=1.21.0        # Numerical computing
```

### Optional Dependencies
```
pandas>=1.3.0        # Data analysis
matplotlib>=3.5.0    # Plotting and visualization
pyarrow>=5.0.0       # Parquet file support
stable-baselines3    # RL algorithms (for advanced features)
```

## 🐛 Troubleshooting

### Common Issues

#### No Display Window
```bash
# Linux: Install display dependencies
sudo apt-get install python3-tk xvfb

# Set display variable
export DISPLAY=:0

# For WSL2: Install VcXsrv or similar X server
```

#### Import Errors
```bash
# Ensure virtual environment is activated
source venv_watch_cars/bin/activate  # Linux/macOS
venv_watch_cars\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements_watch_cars.txt
```

#### Performance Issues
```bash
# Reduce simulation frequency
python watch_cars_auto.py --max-steps 1000

# Disable statistics for better performance
python watch_cars_auto.py --no-stats
```

#### Memory Issues
```bash
# Use shorter episodes
python watch_cars_auto.py --episodes 5

# Disable data collection
python watch_cars_auto.py  # (without --collect-data)
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug output
python watch_cars_auto.py --scenario dense
```

## 📁 File Structure

```
watch_cars_auto.py              # Main simulation script
setup_watch_cars.py             # Automatic setup script
requirements_watch_cars.txt     # Python dependencies
README_WATCH_CARS.md           # This documentation

# Generated during use:
data/
├── simulations/               # Collected simulation data
│   ├── simulation_dense_20240101_120000.json
│   ├── simulation_dense_20240101_120000.csv
│   └── summary_dense_20240101_120000.json
└── plots/                     # Generated visualizations
    └── simulation_stats_dense_20240101_120000.png

venv_watch_cars/               # Virtual environment (after setup)
run_watch_cars.sh             # Convenience run script (Linux/macOS)
run_watch_cars.bat            # Convenience run script (Windows)
```

## 🎓 Learning Examples

### Example 1: Basic Observation
```bash
# Watch free-flowing traffic
python watch_cars_auto.py --scenario free_flow
```
**What to observe**: Smooth traffic flow, minimal lane changes, consistent speeds

### Example 2: Traffic Congestion
```bash
# Watch dense traffic
python watch_cars_auto.py --scenario dense --collect-data
```
**What to observe**: Frequent lane changes, speed variations, collision avoidance

### Example 3: Research Data Collection
```bash
# Collect research data
python watch_cars_auto.py --scenario stop_go --episodes 20 --collect-data
```
**What to observe**: Stop-and-go patterns, acceleration/deceleration cycles

### Example 4: Multi-Agent Coordination
```bash
# Watch multiple agents coordinate
python watch_cars_auto.py --scenario aggressive --agents 3
```
**What to observe**: How multiple controlled agents interact and coordinate

## 🔬 Research Applications

- **🚗 Autonomous Vehicle Research**: Study multi-agent coordination
- **🛣️ Traffic Flow Analysis**: Analyze congestion patterns
- **🧠 Reinforcement Learning**: Generate training data
- **📊 Behavioral Studies**: Study driving behavior patterns
- **🚦 Traffic Engineering**: Test traffic management strategies

## 🤝 Contributing

To extend the simulation:

1. **Add New Scenarios**: Modify `SCENARIOS` dictionary
2. **Enhance Behavior**: Update `generate_intelligent_actions()`
3. **Add Statistics**: Extend `SimulationStats` class
4. **Improve Visualization**: Enhance `plot_statistics()`

## 📄 License

This project is part of the HighwayEnv Multi-Modal Data Collection System.

---

**🎉 Happy Car Watching! 🚗💨**

Watch intelligent cars navigate complex traffic scenarios while collecting valuable research data!