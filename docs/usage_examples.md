# Usage Examples and Documentation

This document provides comprehensive examples and documentation for using the HighwayEnv Multi-Modal Data Collection System.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Usage](#basic-usage)
3. [Data Collection Workflows](#data-collection-workflows)
4. [Data Loading and Access](#data-loading-and-access)
5. [Policy Integration](#policy-integration)
6. [Custom Feature Extraction](#custom-feature-extraction)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Running the Main Demonstration

The main demonstration script provides an interactive way to explore all system capabilities:

```bash
# Interactive demonstration with menu
python main.py

# Run specific demonstrations
python main.py --demo basic       # Basic functionality
python main.py --demo collection  # Data collection workflow
python main.py --demo loading     # Data loading examples
python main.py --demo policy      # Policy integration
python main.py --demo features    # Custom feature extraction
python main.py --demo all         # Run all demonstrations
```

### Basic Data Collection

```python
from highway_datacollection.collection.orchestrator import run_full_collection
from pathlib import Path

# Collect data for all scenarios
result = run_full_collection(
    base_storage_path=Path("data/my_dataset"),
    episodes_per_scenario=100,
    n_agents=3,
    max_steps_per_episode=200,
    base_seed=42
)

print(f"Collected {result.successful_episodes} episodes")
print(f"Dataset saved to: {result.dataset_index_path}")
```

## Basic Usage

### Scenario Configuration

The system supports six curriculum scenarios with customizable parameters:

```python
from highway_datacollection import ScenarioRegistry

registry = ScenarioRegistry()

# List available scenarios
scenarios = registry.list_scenarios()
print(scenarios)  # ['free_flow', 'dense_commuting', 'stop_and_go', ...]

# Get scenario configuration
config = registry.get_scenario_config("free_flow")
print(config)
# {
#     'vehicles_count': 50,
#     'lanes_count': 4,
#     'duration': 40,
#     'description': 'Light traffic with smooth flow'
# }

# Validate custom configuration
custom_config = {
    'vehicles_count': 30,
    'lanes_count': 3,
    'duration': 60
}
is_valid = registry.validate_scenario(custom_config)
```

### Environment Creation

```python
from highway_datacollection.environments.factory import MultiAgentEnvFactory

factory = MultiAgentEnvFactory()

# Create single environment
env = factory.create_env("free_flow", "kinematics", n_agents=2)

# Create parallel environments for all modalities
envs = factory.create_parallel_envs("dense_commuting", n_agents=3)
# Returns: {'kinematics': env1, 'occupancy_grid': env2, 'grayscale': env3}
```

## Data Collection Workflows

### Complete Workflow Example

```python
# examples/complete_workflow_demo.py demonstrates:
python examples/complete_workflow_demo.py

# 1. Configuration setup
# 2. Data collection execution
# 3. Data loading and analysis
# 4. Cross-scenario comparison
# 5. Statistical analysis
```

### Custom Collection Parameters

```python
from highway_datacollection.collection.orchestrator import run_full_collection

# Focused collection for specific scenarios
result = run_full_collection(
    base_storage_path=Path("data/focused_dataset"),
    episodes_per_scenario=50,
    n_agents=2,
    max_steps_per_episode=150,
    scenarios=["free_flow", "lane_closure"],  # Subset of scenarios
    batch_size=10,  # Process in batches
    base_seed=12345
)

# Progress tracking
def progress_callback(progress):
    print(f"Scenario {progress.current_scenario}: "
          f"{progress.current_episode}/{progress.total_episodes}")

result = run_full_collection(
    # ... other parameters
    progress_callback=progress_callback
)
```

### Modality Selection

```python
from highway_datacollection.collection.modality_config import ModalityConfig

# Configure specific modalities
config = ModalityConfig(
    kinematics=True,
    occupancy_grid=True,
    grayscale=False  # Skip grayscale for faster collection
)

# Use in collection (implementation depends on collector setup)
```

## Data Loading and Access

### Basic Data Loading

```python
# examples/data_loading_examples.py provides comprehensive examples
python examples/data_loading_examples.py

# Basic loading
from examples.data_loading_examples import DatasetLoader
import pandas as pd

loader = DatasetLoader("data/my_dataset")

# Load scenario data
df = loader.load_scenario_data("free_flow")
print(f"Loaded {len(df)} transitions")

# Load specific episode
episode_df = loader.load_episode_data("free_flow", "episode_001")

# Load agent trajectory
trajectory = loader.load_agent_trajectory("free_flow", "episode_001", agent_id=0)
```

### Binary Data Reconstruction

```python
# Reconstruct occupancy grid
occ_grid = loader.reconstruct_observation(df.iloc[0], 'occ')
print(f"Occupancy grid shape: {occ_grid.shape}")

# Reconstruct grayscale image
grayscale = loader.reconstruct_observation(df.iloc[0], 'gray')
print(f"Grayscale image shape: {grayscale.shape}")
```

### Dataset Analysis

```python
# Get dataset statistics
stats = loader.get_dataset_statistics()
print(f"Total scenarios: {stats['total_scenarios']}")
print(f"Total files: {stats['total_files']}")

# Per-scenario analysis
for scenario, info in stats['scenarios'].items():
    print(f"{scenario}: {info['episodes']} episodes, "
          f"avg reward: {info['avg_reward']:.3f}")
```

## Policy Integration

### Rule-Based Policies

```python
# examples/policy_integration_examples.py demonstrates various approaches
python examples/policy_integration_examples.py

from highway_datacollection.collection.action_samplers import ActionSampler

class SafeDrivingPolicy(ActionSampler):
    def sample_actions(self, observations, n_agents, seed=None):
        actions = []
        kin_obs = observations['kinematics']
        
        for agent_idx in range(n_agents):
            agent_obs = kin_obs[agent_idx]
            
            # Simple safety logic
            if self._is_safe_to_accelerate(agent_obs):
                action = 2  # FASTER
            elif self._should_slow_down(agent_obs):
                action = 0  # SLOWER
            else:
                action = 1  # IDLE
            
            actions.append(action)
        
        return tuple(actions)
    
    def _is_safe_to_accelerate(self, obs):
        # Your safety logic here
        return True
    
    def _should_slow_down(self, obs):
        # Your safety logic here
        return False
```

### Machine Learning Policy Integration

```python
from stable_baselines3 import PPO
from highway_datacollection.collection.action_samplers import ActionSampler

class SB3PolicySampler(ActionSampler):
    def __init__(self, model_path):
        self.model = PPO.load(model_path)
    
    def sample_actions(self, observations, n_agents, seed=None):
        actions = []
        kin_obs = observations['kinematics']
        
        for agent_idx in range(n_agents):
            # Preprocess observation for your model
            obs = self._preprocess(kin_obs[agent_idx])
            
            # Get action from trained model
            action, _ = self.model.predict(obs, deterministic=True)
            actions.append(int(action))
        
        return tuple(actions)
    
    def _preprocess(self, obs):
        # Flatten or reshape observation as needed by your model
        return obs.flatten()

# Use in collection
policy = SB3PolicySampler("path/to/trained_model.zip")
collector = SynchronizedCollector(env_factory, action_sampler=policy)
result = collector.collect_episode_batch("free_flow", episodes=100, seed=42)
```

### Ensemble Policies

```python
from examples.policy_integration_examples import EnsemblePolicy, RuleBasedPolicy, MLPolicyWrapper

# Create component policies
policies = {
    "conservative": RuleBasedPolicy("safe_following"),
    "aggressive": RuleBasedPolicy("lane_changer"),
    "learned": MLPolicyWrapper("model.pth", "pytorch")
}

# Create weighted ensemble
weights = {"conservative": 0.5, "aggressive": 0.2, "learned": 0.3}
ensemble = EnsemblePolicy(policies, weights)

# Use in collection
collector = SynchronizedCollector(env_factory, action_sampler=ensemble)
```

## Custom Feature Extraction

### Basic Custom Features

```python
from highway_datacollection.features.extractors import KinematicsExtractor

class MyFeatureExtractor:
    def extract_features(self, observation, context):
        features = {}
        
        if observation.shape[0] > 0:
            ego = observation[0]
            others = observation[1:]
            
            # Custom feature 1: Acceleration estimate
            features['acceleration'] = self._estimate_acceleration(ego, context)
            
            # Custom feature 2: Risk assessment
            features['risk_level'] = self._assess_risk(ego, others)
            
            # Custom feature 3: Lane change opportunity
            features['lane_change_safe'] = self._check_lane_change_safety(ego, others)
        
        return features
    
    def _estimate_acceleration(self, ego, context):
        # Your acceleration estimation logic
        return 0.0
    
    def _assess_risk(self, ego, others):
        # Your risk assessment logic
        return 0.0
    
    def _check_lane_change_safety(self, ego, others):
        # Your lane change safety logic
        return True
```

### Advanced Feature Extraction

```python
# The main.py script includes CustomFeatureExtractor with examples of:
# - Lateral acceleration estimation
# - Relative speed to traffic flow
# - Congestion level calculation
# - Lane change opportunity assessment

from main import CustomFeatureExtractor

extractor = CustomFeatureExtractor()

# Use with sample observation
observation = np.array([
    [1.0, 100.0, 8.0, 25.0, 0.5],  # Ego vehicle
    [1.0, 120.0, 8.0, 20.0, 0.0],  # Lead vehicle
    # ... more vehicles
])

features = extractor.extract_features(observation, {"scenario": "free_flow"})
print(features)
# {
#     'ttc': 4.0,
#     'lateral_acceleration': 0.5,
#     'relative_speed_to_traffic': 5.0,
#     'congestion_level': 0.3,
#     'lane_change_opportunity': 0.8
# }
```

## Advanced Usage

### Batch Processing

```python
from highway_datacollection.collection.orchestrator import run_full_collection

# Large-scale data collection with memory management
result = run_full_collection(
    base_storage_path=Path("data/large_dataset"),
    episodes_per_scenario=1000,
    n_agents=5,
    max_steps_per_episode=500,
    batch_size=50,  # Process in smaller batches
    scenarios=["free_flow", "dense_commuting", "stop_and_go", 
               "aggressive_neighbors", "lane_closure", "time_budget"]
)
```

### Performance Optimization

```python
from highway_datacollection.performance.monitor import PerformanceMonitor
from highway_datacollection.performance.profiler import MemoryProfiler

# Monitor performance during collection
monitor = PerformanceMonitor()
profiler = MemoryProfiler()

with monitor.time_context("data_collection"):
    with profiler.profile_context():
        result = run_full_collection(
            # ... parameters
        )

print(f"Collection time: {monitor.get_timing('data_collection'):.2f}s")
print(f"Peak memory usage: {profiler.get_peak_memory():.2f} MB")
```

### Error Handling and Validation

```python
from highway_datacollection.collection.validation import DataValidator
from highway_datacollection.collection.error_handling import CollectionErrorHandler

# Validate collected data
validator = DataValidator()
error_handler = CollectionErrorHandler()

try:
    result = run_full_collection(
        # ... parameters
        error_handler=error_handler
    )
    
    # Validate results
    validation_result = validator.validate_collection_result(result)
    if not validation_result.is_valid:
        print(f"Validation errors: {validation_result.errors}")
    
except Exception as e:
    print(f"Collection failed: {e}")
    # Error handler provides detailed error information
    error_info = error_handler.get_error_summary()
    print(f"Error summary: {error_info}")
```

## Troubleshooting

### Common Issues

1. **Environment Creation Fails**
   ```python
   # Check HighwayEnv installation
   import highway_env
   print(highway_env.__version__)
   
   # Verify scenario configuration
   registry = ScenarioRegistry()
   config = registry.get_scenario_config("free_flow")
   is_valid = registry.validate_scenario(config)
   ```

2. **Memory Issues During Collection**
   ```python
   # Reduce batch size
   result = run_full_collection(
       # ... other parameters
       batch_size=10,  # Smaller batches
       episodes_per_scenario=50  # Fewer episodes
   )
   ```

3. **Data Loading Problems**
   ```python
   # Check dataset structure
   from pathlib import Path
   dataset_path = Path("data/my_dataset")
   
   if not (dataset_path / "index.json").exists():
       print("Dataset index not found")
   
   # Verify parquet files
   parquet_files = list(dataset_path.rglob("*.parquet"))
   print(f"Found {len(parquet_files)} parquet files")
   ```

4. **Policy Integration Issues**
   ```python
   # Test policy separately
   policy = MyCustomPolicy()
   sample_obs = {'kinematics': np.random.randn(2, 5, 5)}
   
   try:
       actions = policy.sample_actions(sample_obs, 2, seed=42)
       print(f"Policy test successful: {actions}")
   except Exception as e:
       print(f"Policy test failed: {e}")
   ```

### Performance Tips

1. **Optimize Collection Speed**
   - Use appropriate batch sizes (10-50 episodes)
   - Disable unnecessary modalities
   - Use faster storage (SSD)

2. **Memory Management**
   - Process scenarios sequentially for large datasets
   - Clear environment states between episodes
   - Use memory profiling to identify bottlenecks

3. **Data Access Optimization**
   - Load only required columns from parquet files
   - Use dataset index for efficient scenario filtering
   - Cache frequently accessed data

### Getting Help

1. **Check Examples**
   - Run `python main.py --demo all` for comprehensive examples
   - Examine `examples/` directory for specific use cases
   - Review `docs/` directory for detailed documentation

2. **Debug Mode**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   
   # Run collection with debug logging
   result = run_full_collection(...)
   ```

3. **Validation Tools**
   ```python
   from highway_datacollection.collection.validation import validate_dataset
   
   # Validate entire dataset
   validation_result = validate_dataset("data/my_dataset")
   if not validation_result.is_valid:
       for error in validation_result.errors:
           print(f"Validation error: {error}")
   ```

This documentation provides comprehensive guidance for using the HighwayEnv Multi-Modal Data Collection System. For additional examples and advanced usage patterns, refer to the example scripts and demonstration files included with the system.