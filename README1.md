# HighwayEnv Multi-Modal Data Collection System

A comprehensive framework for capturing synchronized observations across multiple modalities in multi-agent highway driving scenarios. This system supports curriculum-based scenario generation for training both reinforcement learning agents and large language model planners in autonomous driving contexts.

## Project Structure

```
highway_datacollection/
├── __init__.py                 # Main package initialization
├── config.py                   # Global configuration constants
├── scenarios/                  # Scenario management
│   ├── __init__.py
│   ├── config.py              # Scenario configurations and constants
│   └── registry.py            # ScenarioRegistry class
├── environments/               # Environment factory
│   └── __init__.py
├── collection/                 # Data collection orchestration
│   └── __init__.py
├── features/                   # Feature derivation and processing
│   └── __init__.py
└── storage/                    # Storage management
    └── __init__.py
```

## Features

- **Multi-Modal Observation Capture**: Synchronized collection of Kinematics, OccupancyGrid, and Grayscale observations
- **Curriculum-Based Scenarios**: Six predefined driving scenarios (free_flow, dense_commuting, stop_and_go, aggressive_neighbors, lane_closure, time_budget)
- **Feature Derivation**: Automatic calculation of Time-to-Collision, traffic density, and natural language summaries
- **Efficient Storage**: Parquet-based storage with binary blob encoding for large arrays
- **Extensible Architecture**: Plugin-based design for custom policies and observation types

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

```python
from highway_datacollection import ScenarioRegistry

# Initialize scenario registry
registry = ScenarioRegistry()

# List available scenarios
scenarios = registry.list_scenarios()
print(f"Available scenarios: {scenarios}")

# Get configuration for a specific scenario
config = registry.get_scenario_config("free_flow")
print(f"Free flow config: {config}")
```

## Requirements

- Python 3.8+
- highway-env >= 1.8.0
- gymnasium >= 0.29.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- pyarrow >= 12.0.0
- stable-baselines3 >= 2.0.0

## Development

This project follows the specification-driven development methodology. See the `.kiro/specs/highway-multimodal-datacollection/` directory for detailed requirements, design, and implementation tasks.

## License

MIT License - see LICENSE file for details.