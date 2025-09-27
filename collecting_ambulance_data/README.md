# Ambulance Data Collection System

This system extends the existing highway data collection infrastructure to support ambulance ego vehicles with specialized emergency scenarios. It maintains full compatibility with current multi-modal data collection while introducing ambulance-specific vehicle configurations and 15 diverse scenarios that capture emergency vehicle interactions with regular traffic.

## Overview

The ambulance data collection system provides:
- **Ambulance Ego Vehicle**: First controlled agent configured as ambulance type
- **Multi-Agent Support**: 3-4 controlled agents with ambulance + normal vehicles
- **15 Emergency Scenarios**: Diverse highway situations for comprehensive data collection
- **Multi-Modal Data**: Full support for Kinematics, OccupancyGrid, and GrayscaleObservation
- **Seamless Integration**: Works with existing visualization and analysis tools

## Features

### Emergency Vehicle Scenarios
- Light, moderate, and heavy traffic conditions
- Rush hour and accident response situations
- Construction zones and lane closures
- Weather conditions and aggressive driver interactions
- Hospital runs and time-critical scenarios

### Data Collection Capabilities
- **Multi-modal observation collection**: All 3 observation types supported
  - **Kinematics**: Vehicle state data (position, velocity, heading)
  - **OccupancyGrid**: Spatial grid representation of the environment
  - **GrayscaleObservation**: Visual/image observations of the driving scene
- Horizontal image orientation for all visual observations (800x600)
- Dynamic observation type selection (no hardcoded observation configs)
- Integration with existing storage and analysis pipeline
- Support for all current action sampling methods

## Directory Structure

```
collecting_ambulance_data/
├── scenarios/           # Ambulance scenario configurations
├── collection/          # Specialized data collection classes
├── examples/           # Demonstration and usage scripts
└── README.md          # This documentation file
```

## Quick Start

### Basic Usage

```python
from collecting_ambulance_data import AmbulanceDataCollector, AmbulanceScenarioConfig

# Initialize the ambulance data collector
collector = AmbulanceDataCollector()

# Get available ambulance scenarios
scenarios = AmbulanceScenarioConfig.get_ambulance_scenarios()

# Collect data for specific scenarios
collector.collect_ambulance_data(
    scenarios=["highway_emergency_light", "highway_rush_hour"],
    output_dir="data/ambulance_collection"
)
```

### Running Example Scripts

```bash
# Basic ambulance data collection
python collecting_ambulance_data/examples/basic_ambulance_collection.py

# Ambulance demonstration with visualization
python collecting_ambulance_data/examples/ambulance_demo.py
```

## Available Scenarios

The system includes 15 distinct ambulance scenarios designed to capture diverse emergency response situations:

### Traffic Density-Based Scenarios

#### Light Traffic Emergency Response
1. **highway_emergency_light**
   - **Description**: Ambulance on highway with light traffic flow
   - **Traffic Density**: Light (15 vehicles)
   - **Duration**: 40 seconds
   - **Speed Limit**: 30 km/h
   - **Conditions**: Normal highway conditions
   - **Use Case**: Basic emergency response training, optimal path learning

8. **highway_weather_conditions**
   - **Description**: Ambulance on highway with weather challenges
   - **Traffic Density**: Light (18 vehicles)
   - **Duration**: 45 seconds
   - **Speed Limit**: 20 km/h (reduced due to weather)
   - **Conditions**: Adverse weather conditions
   - **Use Case**: Emergency response under challenging environmental conditions

#### Moderate Traffic Emergency Response
2. **highway_emergency_moderate**
   - **Description**: Ambulance navigating moderate highway traffic
   - **Traffic Density**: Moderate (25 vehicles)
   - **Duration**: 45 seconds
   - **Speed Limit**: 30 km/h
   - **Conditions**: Standard highway traffic
   - **Use Case**: Typical emergency response scenarios, lane changing strategies

4. **highway_lane_closure**
   - **Description**: Ambulance navigating highway with lane closure
   - **Traffic Density**: Moderate (30 vehicles)
   - **Duration**: 45 seconds
   - **Speed Limit**: 20 km/h
   - **Conditions**: Construction zone with reduced lanes (3 lanes)
   - **Use Case**: Emergency navigation through infrastructure limitations

6. **highway_accident_scene**
   - **Description**: Ambulance approaching highway accident location
   - **Traffic Density**: Moderate (25 vehicles)
   - **Duration**: 40 seconds
   - **Speed Limit**: 15 km/h
   - **Conditions**: Accident scene with increased collision penalty (-2)
   - **Use Case**: Emergency response to accident sites, careful navigation

7. **highway_construction**
   - **Description**: Ambulance through highway construction zone
   - **Traffic Density**: Moderate (20 vehicles)
   - **Duration**: 50 seconds
   - **Speed Limit**: 15 km/h
   - **Conditions**: Active construction zone
   - **Use Case**: Emergency response through work zones, reduced speed navigation

10. **highway_aggressive_drivers**
    - **Description**: Ambulance with aggressive highway drivers
    - **Traffic Density**: Moderate (28 vehicles)
    - **Duration**: 45 seconds
    - **Speed Limit**: 30 km/h
    - **Conditions**: Aggressive vehicle behaviors
    - **Use Case**: Emergency response with non-cooperative traffic

12. **highway_speed_variation**
    - **Description**: Ambulance with varying highway speed zones
    - **Traffic Density**: Moderate (25 vehicles)
    - **Duration**: 45 seconds
    - **Speed Limit**: 35 km/h (variable)
    - **Conditions**: Multiple speed zones
    - **Use Case**: Adaptive speed control during emergency response

14. **highway_truck_heavy**
    - **Description**: Ambulance on highway with heavy truck traffic
    - **Traffic Density**: Moderate (22 vehicles)
    - **Duration**: 50 seconds
    - **Speed Limit**: 25 km/h
    - **Conditions**: Heavy truck presence
    - **Use Case**: Emergency navigation around large vehicles

15. **highway_time_pressure**
    - **Description**: Ambulance with high time pressure scenario
    - **Traffic Density**: Moderate (30 vehicles)
    - **Duration**: 35 seconds (shortened for urgency)
    - **Speed Limit**: 35 km/h
    - **Conditions**: Time-critical emergency with increased collision penalty (-2)
    - **Use Case**: High-urgency emergency response, time-optimal routing

#### Heavy Traffic Emergency Response
3. **highway_emergency_dense**
   - **Description**: Ambulance in heavy highway congestion
   - **Traffic Density**: Heavy (40 vehicles)
   - **Duration**: 50 seconds
   - **Speed Limit**: 25 km/h
   - **Conditions**: Congested highway conditions
   - **Use Case**: Emergency response in heavy traffic, gap utilization

5. **highway_rush_hour**
   - **Description**: Ambulance during peak highway rush hour
   - **Traffic Density**: Heavy (45 vehicles)
   - **Duration**: 55 seconds
   - **Speed Limit**: 25 km/h
   - **Conditions**: Peak traffic conditions
   - **Use Case**: Emergency response during commuter hours

9. **highway_stop_and_go**
   - **Description**: Ambulance in stop-and-go highway traffic
   - **Traffic Density**: Heavy (35 vehicles)
   - **Duration**: 60 seconds
   - **Speed Limit**: 10 km/h
   - **Conditions**: Stop-and-go traffic patterns
   - **Use Case**: Emergency response in congested, intermittent traffic

11. **highway_merge_heavy**
    - **Description**: Ambulance navigating heavy highway merge areas
    - **Traffic Density**: Heavy (38 vehicles)
    - **Duration**: 50 seconds
    - **Speed Limit**: 25 km/h
    - **Conditions**: Heavy merge traffic
    - **Use Case**: Emergency response through merge zones, coordination with merging traffic

13. **highway_shoulder_use**
    - **Description**: Ambulance using highway shoulder when needed
    - **Traffic Density**: Heavy (35 vehicles)
    - **Duration**: 40 seconds
    - **Speed Limit**: 20 km/h
    - **Conditions**: Shoulder available for emergency use
    - **Use Case**: Emergency shoulder utilization, alternative routing

### Scenario Selection Guidelines

**For Basic Training**: Start with `highway_emergency_light` and `highway_emergency_moderate`
**For Challenging Conditions**: Use `highway_rush_hour`, `highway_stop_and_go`, `highway_emergency_dense`
**For Infrastructure Challenges**: Focus on `highway_lane_closure`, `highway_construction`, `highway_merge_heavy`
**For Behavioral Challenges**: Try `highway_aggressive_drivers`, `highway_time_pressure`
**For Environmental Conditions**: Use `highway_weather_conditions`, `highway_shoulder_use`

### Common Scenario Parameters

All scenarios share these characteristics:
- **Controlled Vehicles**: 4 agents (first agent is ambulance)
- **Highway Configuration**: 4-lane highway (except lane closure: 3 lanes)
- **Image Orientation**: Horizontal (800x600)
- **Multi-Modal Support**: All 3 observation types supported
- **Emergency Priority**: High priority for most scenarios (medium for construction and speed variation)

## Configuration

### Scenario Parameters

Each ambulance scenario includes:
- **Ego Vehicle**: Always configured as ambulance type (first agent)
- **Multi-Agent Setup**: 3-4 controlled agents total
- **Highway Configuration**: 4-lane highway setup
- **Traffic Conditions**: Varied densities and behaviors
- **Emergency Context**: Realistic emergency response situations

### Observation Types

Supports all existing observation modalities with dynamic configuration:
- **Kinematics**: Vehicle positions, velocities, accelerations, heading information
  - Features: presence, x, y, vx, vy, cos_h, sin_h
  - Shape: (15, 7) for multi-agent environments
- **OccupancyGrid**: Spatial occupancy information in grid format
  - Features: presence, on_road status
  - Shape: (2, 11, 11) grid representation
- **GrayscaleObservation**: Visual observations in horizontal orientation
  - Grayscale images from vehicle perspective
  - Shape: (4, 128, 64) with stack size of 4
  - Horizontal orientation: 800x600 screen resolution

**Key Feature**: Observation types are configured dynamically by the environment factory, enabling simultaneous collection of all three data types without hardcoded limitations.

## Integration with Existing Tools

### Data Storage Integration
The ambulance data collection system seamlessly integrates with the existing data storage infrastructure:

```python
# Ambulance data uses the same storage format as standard highway data
from highway_datacollection.storage.manager import DatasetStorageManager

# Load ambulance data with existing tools
storage_manager = DatasetStorageManager("data/ambulance_collection")
ambulance_episodes = storage_manager.load_episodes("highway_emergency_dense")

# Data format is identical - same Parquet schema and JSONL metadata
print(f"Loaded {len(ambulance_episodes)} emergency vehicle episodes")
```

### Visualization Integration
All existing visualization tools work seamlessly with ambulance data:

```python
# Use existing multimodal plotting tools
from visualization.comprehensive_data_plotter import plot_multimodal_data
from visualization.multimodal_parquet_plotter import MultimodalParquetPlotter

# Plot ambulance scenario data
plot_multimodal_data(
    data_path="data/ambulance_collection/highway_rush_hour",
    scenario="highway_rush_hour",
    output_dir="plots/ambulance_analysis"
)

# Create detailed ambulance behavior analysis
plotter = MultimodalParquetPlotter("data/ambulance_collection")
plotter.create_comprehensive_dashboard(
    scenarios=["highway_emergency_light", "highway_accident_scene"],
    output_file="ambulance_dashboard.png"
)
```

### Analysis and Feature Extraction
Ambulance data integrates with existing analysis pipelines:

```python
# Use existing feature extraction on ambulance data
from highway_datacollection.features.engine import FeatureDerivationEngine

engine = FeatureDerivationEngine()

# Extract features from ambulance episodes
ambulance_features = engine.extract_features(
    data_path="data/ambulance_collection/highway_emergency_dense",
    include_natural_language=True
)

# Features include emergency-specific metrics
print("Available features:", ambulance_features.columns.tolist())
# Output includes: ttc, speed, lane_position, traffic_density, summary_text, etc.
```

### Training Pipeline Integration
Emergency vehicle data can be used directly in existing training workflows:

```python
# Load ambulance data for RL training
import pandas as pd
from stable_baselines3 import PPO

# Combine standard and emergency data for robust training
standard_data = pd.read_parquet("data/highway_dataset/dense_commuting/transitions.parquet")
emergency_data = pd.read_parquet("data/ambulance_collection/highway_emergency_dense/transitions.parquet")

# Train on combined dataset for emergency-aware policies
combined_data = pd.concat([standard_data, emergency_data])
```

## Requirements

### System Requirements
- Python 3.8+
- highway-env with ambulance vehicle support
- Existing highway data collection dependencies

### Dependencies
- All existing highway_datacollection package dependencies
- No additional external dependencies required

## Usage Examples

### Basic Data Collection with AmbulanceDataCollector

```python
from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector
from pathlib import Path

# Initialize the ambulance data collector
with AmbulanceDataCollector(n_agents=4) as collector:
    
    # Get available scenarios
    scenarios = collector.get_available_scenarios()
    print(f"Available scenarios: {len(scenarios)}")
    
    # Collect data from a single scenario
    result = collector.collect_single_ambulance_scenario(
        scenario_name="highway_emergency_light",
        episodes=10,
        max_steps=50,
        seed=42
    )
    
    # Store the collected data
    output_dir = Path("data/ambulance_collection")
    storage_info = collector.store_ambulance_data(
        {"highway_emergency_light": result}, 
        output_dir
    )
    
    print(f"Collected {result.successful_episodes} episodes")
    print(f"Stored to: {storage_info['output_dir']}")
```

### Multi-Scenario Data Collection

```python
from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector

# Collect data from multiple scenarios
with AmbulanceDataCollector(n_agents=4) as collector:
    
    # Collect from specific scenarios
    scenarios_to_collect = [
        "highway_emergency_light",
        "highway_emergency_dense", 
        "highway_rush_hour"
    ]
    
    results = collector.collect_ambulance_data(
        scenarios=scenarios_to_collect,
        episodes_per_scenario=20,
        max_steps_per_episode=100,
        base_seed=42,
        batch_size=5
    )
    
    # Display results
    for scenario, result in results.items():
        print(f"{scenario}: {result.successful_episodes}/{result.total_episodes} episodes")
```

### Command Line Usage

Use the provided scripts for easy data collection:

```bash
# Activate the environment first (as required)
source avs_venv/bin/activate

# Collect from all 15 ambulance scenarios (GPU-accelerated)
python collecting_ambulance_data/examples/final_gpu_accelerated_collection.py \
    --episodes 50 \
    --max-steps 100 \
    --gpu-intensity 20 \
    --output-dir data/ambulance_dataset

# Collect from all 15 ambulance scenarios (standard)
python collecting_ambulance_data/examples/basic_ambulance_collection.py \
    --episodes 50 \
    --max-steps 100 \
    --output-dir data/ambulance_dataset

# Collect from specific emergency scenarios
python collecting_ambulance_data/examples/basic_ambulance_collection.py \
    --scenarios highway_emergency_light highway_rush_hour highway_accident_scene \
    --episodes 20 \
    --max-steps 75 \
    --output-dir data/emergency_response_dataset

# Run ambulance demonstration with visualization
python collecting_ambulance_data/examples/ambulance_demo.py

# Test ambulance data collection pipeline
python collecting_ambulance_data/examples/test_ambulance_collector.py
```

### Customizing Ambulance Data Collection

#### Custom Scenario Selection
```python
from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector

# Select scenarios by traffic density
light_traffic_scenarios = [
    "highway_emergency_light", 
    "highway_weather_conditions"
]

moderate_traffic_scenarios = [
    "highway_emergency_moderate",
    "highway_lane_closure", 
    "highway_accident_scene"
]

heavy_traffic_scenarios = [
    "highway_emergency_dense",
    "highway_rush_hour",
    "highway_stop_and_go"
]

# Collect data for specific traffic conditions
with AmbulanceDataCollector(n_agents=4) as collector:
    # Start with light traffic for initial training
    light_results = collector.collect_ambulance_data(
        scenarios=light_traffic_scenarios,
        episodes_per_scenario=30,
        max_steps_per_episode=80
    )
    
    # Progress to moderate traffic
    moderate_results = collector.collect_ambulance_data(
        scenarios=moderate_traffic_scenarios,
        episodes_per_scenario=25,
        max_steps_per_episode=90
    )
```

#### Custom Episode Configuration
```python
# Configure different episode lengths for different scenarios
scenario_configs = {
    "highway_emergency_light": {"episodes": 50, "max_steps": 60},
    "highway_rush_hour": {"episodes": 30, "max_steps": 120},
    "highway_time_pressure": {"episodes": 40, "max_steps": 50}  # Shorter for urgency
}

with AmbulanceDataCollector(n_agents=4) as collector:
    for scenario, config in scenario_configs.items():
        result = collector.collect_single_ambulance_scenario(
            scenario_name=scenario,
            episodes=config["episodes"],
            max_steps=config["max_steps"],
            seed=42
        )
        print(f"{scenario}: {result.successful_episodes} episodes collected")
```

#### Batch Processing for Large Datasets
```python
# Collect large datasets in batches to manage memory
def collect_large_ambulance_dataset():
    all_scenarios = [
        "highway_emergency_light", "highway_emergency_moderate", "highway_emergency_dense",
        "highway_lane_closure", "highway_rush_hour", "highway_accident_scene",
        "highway_construction", "highway_weather_conditions", "highway_stop_and_go",
        "highway_aggressive_drivers", "highway_merge_heavy", "highway_speed_variation",
        "highway_shoulder_use", "highway_truck_heavy", "highway_time_pressure"
    ]
    
    # Process in batches of 5 scenarios
    batch_size = 5
    for i in range(0, len(all_scenarios), batch_size):
        batch_scenarios = all_scenarios[i:i+batch_size]
        
        with AmbulanceDataCollector(n_agents=4) as collector:
            batch_results = collector.collect_ambulance_data(
                scenarios=batch_scenarios,
                episodes_per_scenario=20,
                max_steps_per_episode=100,
                batch_size=3  # Process 3 episodes at a time
            )
            
            # Store batch results
            batch_output = f"data/ambulance_batch_{i//batch_size + 1}"
            collector.store_ambulance_data(batch_results, batch_output)
            print(f"Completed batch {i//batch_size + 1}")

# Run large dataset collection
collect_large_ambulance_dataset()
```

### Collecting Specific Scenarios

```python
# Collect data for emergency response scenarios
emergency_scenarios = [
    "highway_accident_scene",
    "highway_rush_hour", 
    "highway_time_pressure"
]

with AmbulanceDataCollector(n_agents=4) as collector:
    results = collector.collect_ambulance_data(
        scenarios=emergency_scenarios,
        episodes_per_scenario=10,
        max_steps_per_episode=100
    )
    
    # Store all results
    storage_info = collector.store_ambulance_data(
        results, 
        Path("data/emergency_response")
    )
```

### Scenario Information and Validation

```python
from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector

with AmbulanceDataCollector() as collector:
    # Get scenario information
    info = collector.get_scenario_info("highway_emergency_light")
    print(f"Description: {info['description']}")
    print(f"Traffic density: {info['traffic_density']}")
    print(f"Duration: {info['duration']} seconds")
    
    # Validate scenario setup
    validation = collector.validate_ambulance_setup("highway_emergency_light")
    print(f"Valid setup: {validation['valid']}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
```

## Data Output

### File Structure
```
data/ambulance_collection/
├── highway_emergency_light/
│   ├── episode_001_transitions.parquet
│   ├── episode_001_meta.jsonl
│   └── ...
├── highway_rush_hour/
│   ├── episode_001_transitions.parquet
│   ├── episode_001_meta.jsonl
│   └── ...
└── index.json
```

### Data Format
- **Transitions**: Multi-modal observations, actions, rewards in Parquet format
- **Metadata**: Episode information, scenario parameters in JSONL format
- **Index**: Dataset organization and scenario mapping in JSON format

## Integration Notes for Existing Tools and Workflows

### Compatibility with Existing Analysis Tools

#### Natural Language Analysis
```python
# Ambulance data includes natural language summaries
from examples.show_natural_language_data import show_natural_language_summary

# Analyze emergency vehicle behavior descriptions
show_natural_language_summary(
    data_path="data/ambulance_collection/highway_rush_hour",
    scenario="highway_rush_hour"
)

# Example output:
# "Ambulance navigating heavy rush hour traffic, maintaining emergency priority 
#  while coordinating with 3 civilian vehicles. High traffic density requires 
#  careful gap selection and lane changing strategies."
```

#### Multimodal Plotting Integration
```python
# Use existing plotting tools with ambulance data
from visualization.multimodal_parquet_plotter import MultimodalParquetPlotter
from examples.plot_multimodal_data_example import create_comprehensive_plots

# Create ambulance-specific visualizations
plotter = MultimodalParquetPlotter("data/ambulance_collection")

# Plot emergency vehicle trajectories
plotter.plot_trajectories(
    scenario="highway_accident_scene",
    highlight_agent=0,  # Highlight ambulance (first agent)
    output_file="ambulance_trajectories.png"
)

# Create comprehensive emergency scenario analysis
create_comprehensive_plots(
    data_dir="data/ambulance_collection",
    scenarios=["highway_emergency_light", "highway_emergency_dense"],
    output_dir="plots/emergency_analysis"
)
```

#### Training Pipeline Integration
```python
# Integrate ambulance data with existing RL training
from rl.rl_training_example import train_multiagent_policy

# Train on combined standard + emergency data
train_multiagent_policy(
    data_paths=[
        "data/highway_dataset",  # Standard highway data
        "data/ambulance_collection"  # Emergency vehicle data
    ],
    policy_type="emergency_aware",
    training_episodes=10000
)
```

### Workflow Integration Examples

#### Complete Emergency Vehicle Research Workflow
```python
# 1. Collect emergency vehicle data
from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector

with AmbulanceDataCollector(n_agents=4) as collector:
    emergency_data = collector.collect_ambulance_data(
        scenarios=["highway_emergency_light", "highway_rush_hour"],
        episodes_per_scenario=100
    )
    collector.store_ambulance_data(emergency_data, "data/emergency_research")

# 2. Analyze with existing tools
from visualization.comprehensive_data_plotter import plot_multimodal_data
from highway_datacollection.features.engine import FeatureDerivationEngine

# Extract features using existing pipeline
engine = FeatureDerivationEngine()
features = engine.extract_features("data/emergency_research")

# Visualize using existing tools
plot_multimodal_data(
    data_path="data/emergency_research/highway_rush_hour",
    scenario="highway_rush_hour",
    output_dir="plots/emergency_analysis"
)

# 3. Train models using existing infrastructure
from stable_baselines3 import PPO
import pandas as pd

# Load data with existing tools
df = pd.read_parquet("data/emergency_research/highway_rush_hour/transitions.parquet")

# Use existing training pipeline
# (Training code would use existing RL infrastructure)
```

#### Data Quality Validation Integration
```python
# Use existing validation tools with ambulance data
from collecting_ambulance_data.validation import validate_ambulance_data
from tests.test_data_quality_assurance import run_data_quality_tests

# Validate ambulance data collection
validation_results = validate_ambulance_data("data/ambulance_collection")
print(f"Validation passed: {validation_results['all_valid']}")

# Run existing data quality tests on ambulance data
quality_results = run_data_quality_tests("data/ambulance_collection")
print(f"Quality score: {quality_results['overall_score']}")
```

### Performance Considerations

#### Memory Management with Existing Tools
```python
# Ambulance data collection respects existing memory limits
from highway_datacollection.performance.monitor import PerformanceMonitor

# Monitor performance during ambulance data collection
monitor = PerformanceMonitor()
with AmbulanceDataCollector(n_agents=4, max_memory_gb=8.0) as collector:
    # Collection automatically manages memory like existing system
    results = collector.collect_ambulance_data(
        scenarios=["highway_emergency_dense"],
        episodes_per_scenario=50
    )
    
    performance_stats = monitor.get_stats()
    print(f"Peak memory usage: {performance_stats['peak_memory_gb']:.2f} GB")
```

#### Storage Integration
```python
# Ambulance data uses same storage format as existing system
from highway_datacollection.storage.manager import DatasetStorageManager

# Load ambulance and standard data with same tools
storage = DatasetStorageManager("data")

# Both datasets use identical format
standard_episodes = storage.load_episodes("highway_dataset/free_flow")
ambulance_episodes = storage.load_episodes("ambulance_collection/highway_emergency_light")

# Data schemas are identical - seamless integration
print("Standard data columns:", standard_episodes.columns.tolist())
print("Ambulance data columns:", ambulance_episodes.columns.tolist())
# Both will show: ['episode_id', 'step', 'agent_id', 'action', 'reward', 'ttc', ...]
```

## Troubleshooting

### Common Issues

1. **Environment Activation Required**
   - Always run `source avs_venv/bin/activate` before ambulance data collection
   - This ensures proper Python environment and dependencies

2. **Ambulance Vehicle Configuration**
   - The system uses custom logic to treat the first agent as an ambulance
   - No special highway-env ambulance vehicle type is required
   - Verify `controlled_vehicles` is set to 4 and `ambulance_agent_index` is 0

3. **Multi-Agent Setup Issues**
   - Ensure 4 controlled vehicles are configured (not 2 like standard scenarios)
   - First agent (index 0) represents the ambulance
   - Remaining 3 agents are normal vehicles

4. **Data Collection Failures**
   - Check scenario configuration parameters match expected format
   - Verify output directory has write permissions
   - Review error logs for specific failure details
   - Ensure sufficient disk space for large datasets

5. **Integration Issues**
   - Ambulance data uses identical format to standard highway data
   - All existing visualization and analysis tools should work without modification
   - If tools fail, verify data paths and file formats

### Performance Optimization

- **Episode Configuration**: Use appropriate episode counts for scenario complexity
  - Light traffic: 50-100 episodes
  - Heavy traffic: 30-50 episodes (more complex, longer episodes)
- **Memory Management**: Monitor memory usage during long collection runs
  - Use batch processing for large datasets
  - Set appropriate `max_memory_gb` limits
- **Parallel Processing**: Consider collecting different scenarios in parallel
- **Storage Optimization**: Use Parquet format for optimal performance (default)

## Contributing

When adding new ambulance scenarios:
1. Follow existing scenario configuration patterns
2. Ensure ambulance ego vehicle is properly configured
3. Test scenario with all observation modalities
4. Update documentation with new scenario details

## Support

For issues related to:
- **Scenario Configuration**: Check ambulance_scenarios.py
- **Data Collection**: Review ambulance_collector.py
- **Integration Issues**: Verify existing system compatibility
- **Performance**: Consult existing optimization guides

## License

This ambulance data collection system follows the same license as the parent highway data collection project.