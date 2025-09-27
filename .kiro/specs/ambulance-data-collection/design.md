# Design Document

## Overview

The ambulance data collection system extends the existing highway data collection infrastructure to support emergency vehicle scenarios. The design maintains full compatibility with current multi-modal data collection while introducing ambulance-specific vehicle configurations and 15 specialized scenarios. The system leverages highway-env's built-in ambulance vehicle type and creates realistic emergency response situations for comprehensive data collection.

## Architecture

### Core Components

The system builds upon existing architecture with minimal modifications:

```
collecting_ambulance_data/
├── scenarios/
│   ├── ambulance_scenarios.py     # 15 ambulance-specific scenarios
│   └── __init__.py
├── collection/
│   ├── ambulance_collector.py     # Specialized collector for ambulance data
│   └── __init__.py
├── examples/
│   ├── ambulance_demo.py          # Demonstration script
│   └── basic_ambulance_collection.py
└── README.md                      # Documentation and usage guide
```

### Integration Points

1. **Environment Factory Extension**: Modify `MultiAgentEnvFactory` to support ambulance ego vehicle configuration
2. **Scenario Registry Extension**: Add ambulance scenarios to the existing registry system
3. **Data Collection Pipeline**: Utilize existing `SynchronizedCollector` with ambulance-specific configurations
4. **Storage Integration**: Use current `DatasetStorageManager` for seamless data storage

## Components and Interfaces

### AmbulanceScenarioConfig

```python
class AmbulanceScenarioConfig:
    """Configuration class for ambulance-specific scenarios."""
    
    def get_ambulance_scenarios() -> Dict[str, Dict[str, Any]]
    def validate_ambulance_config(config: Dict[str, Any]) -> bool
    def create_emergency_situation(scenario_type: str) -> Dict[str, Any]
```

### Enhanced MultiAgentEnvFactory

```python
class MultiAgentEnvFactory:
    """Extended to support ambulance ego vehicle configuration."""
    
    def create_ambulance_env(scenario_name: str, obs_type: str, n_agents: int) -> gym.Env
    def get_ambulance_base_config(scenario_name: str, n_agents: int) -> Dict[str, Any]
```

### Multi-Modal Data Collection

The ambulance data collection system supports all three observation modalities available in the main highway system:

#### 1. Kinematics Observations
- **Data Type**: Numerical vehicle state information
- **Features**: Position (x, y), velocity (vx, vy), heading (cos_h, sin_h), presence
- **Use Case**: Vehicle dynamics analysis, trajectory planning, behavioral modeling

#### 2. OccupancyGrid Observations  
- **Data Type**: Spatial grid representation of the environment
- **Features**: Binary occupancy information in a grid around the ego vehicle
- **Use Case**: Spatial reasoning, obstacle detection, path planning

#### 3. GrayscaleObservation
- **Data Type**: Visual/image observations of the driving scene
- **Features**: Grayscale images rendered from the vehicle's perspective
- **Use Case**: Computer vision tasks, visual perception, scene understanding

### Visual Configuration

All ambulance scenarios will use horizontal image orientation for visual observations:
- **Screen dimensions**: Configured for horizontal/landscape view (800x600)
- **Observation rendering**: Optimized for horizontal display format
- **Image aspect ratio**: Maintained for horizontal viewing experience
- **Multi-modal support**: Dynamic observation type selection based on collection requirements

### AmbulanceDataCollector

```python
class AmbulanceDataCollector:
    """Specialized collector for ambulance scenario data collection."""
    
    def collect_ambulance_data(scenarios: List[str], output_dir: str) -> None
    def setup_ambulance_environments(scenario_name: str) -> Dict[str, gym.Env]
```

## Data Models

### Ambulance Scenario Configuration

```python
ambulance_scenario = {
    "scenario_name": str,           # Unique identifier
    "observation": {                # Multi-modal observation configuration
        "type": "MultiAgentObservation",
        "observation_config": {
            # Dynamic - supports Kinematics, OccupancyGrid, GrayscaleObservation
            "type": str,            # Configurable observation type
            "features": list,       # Observation-specific features
            "absolute": bool        # Coordinate system setting
        }
    },
    "action": {                     # Multi-agent action configuration
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction"
        }
    },
    "traffic_density": str,         # light, moderate, heavy
    "vehicles_count": int,          # Total vehicles in highway scenario
    "lanes_count": 4,               # Fixed at 4 lanes for all scenarios
    "duration": int,                # Scenario duration in seconds
    "other_vehicles_type": str,     # IDMVehicle, AggressiveVehicle, etc.
    "controlled_vehicles": 4,       # 3-4 controlled agents total
    "highway_conditions": str,      # normal, construction, accident, weather
    "speed_limit": int,             # Highway speed limit
    "_ambulance_config": {          # Ambulance-specific metadata
        "ambulance_agent_index": 0, # First agent (index 0) is always the ambulance
        "emergency_priority": str,  # high, medium (affects ambulance behavior)
        "ambulance_behavior": str   # emergency_response behavior type
    }
}
```

### 15 Highway Ambulance Scenarios (Single Ambulance)

1. **highway_emergency_light**: Single ambulance on highway with light traffic flow
2. **highway_emergency_moderate**: Single ambulance navigating moderate highway traffic
3. **highway_emergency_dense**: Single ambulance in heavy highway congestion
4. **highway_lane_closure**: Single ambulance navigating highway with lane closure
5. **highway_rush_hour**: Single ambulance during peak highway rush hour
6. **highway_accident_scene**: Single ambulance approaching highway accident location
7. **highway_construction**: Single ambulance through highway construction zone
8. **highway_weather_conditions**: Single ambulance on highway with weather challenges
9. **highway_stop_and_go**: Single ambulance in stop-and-go highway traffic
10. **highway_aggressive_drivers**: Single ambulance with aggressive highway drivers
11. **highway_merge_heavy**: Single ambulance navigating heavy highway merge areas
12. **highway_speed_variation**: Single ambulance with varying highway speed zones
13. **highway_shoulder_use**: Single ambulance using highway shoulder when needed
14. **highway_truck_heavy**: Single ambulance on highway with heavy truck traffic
15. **highway_time_pressure**: Single ambulance with high time pressure scenario

## Error Handling

### Configuration Validation

- Validate ambulance vehicle type compatibility with highway-env
- Ensure scenario parameters meet system constraints
- Verify multi-modal observation compatibility with ambulance vehicles

### Runtime Error Management

- Handle ambulance vehicle initialization failures gracefully
- Provide fallback to standard vehicle if ambulance type unavailable
- Maintain data collection continuity during scenario transitions

### Data Collection Robustness

- Implement retry mechanisms for failed ambulance scenario runs
- Ensure partial data collection completion if individual scenarios fail
- Maintain data integrity across all observation modalities

## Testing Strategy

### Unit Testing

- Test ambulance vehicle configuration in isolation
- Validate each of the 15 scenario configurations
- Verify integration with existing factory and registry systems

### Integration Testing

- Test complete ambulance data collection pipeline
- Verify multi-modal data collection with ambulance ego vehicle
- Ensure compatibility with existing visualization and analysis tools

### Scenario Testing

- Execute each ambulance scenario individually
- Validate realistic emergency vehicle behaviors
- Test traffic interaction patterns with ambulance presence

### Performance Testing

- Measure data collection performance with ambulance scenarios
- Compare collection speeds with standard vehicle scenarios
- Validate memory usage and storage requirements

### End-to-End Testing

- Run complete ambulance data collection workflow
- Verify data storage in collecting_ambulance_data folder
- Test integration with existing analysis and plotting tools