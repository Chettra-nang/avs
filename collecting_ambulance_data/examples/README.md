# Ambulance Data Collection Examples

This directory contains example scripts demonstrating the ambulance data collection system.

## Scripts

### `ambulance_demo.py`
**Main demonstration script** showing the complete ambulance data collection system.

**Features demonstrated:**
- ✅ All 15 ambulance highway scenarios
- ✅ Multi-modal data collection (Kinematics, OccupancyGrid, GrayscaleObservation)
- ✅ Integration with existing visualization tools
- ✅ Data storage and management
- ✅ Statistics and monitoring
- ✅ Environment setup and validation

**Usage:**
```bash
python collecting_ambulance_data/examples/ambulance_demo.py
```

**What it does:**
1. Shows all available ambulance scenarios and their configurations
2. Demonstrates multi-modal observation support
3. Performs small-scale data collection (5 episodes, 20 steps each)
4. Stores data in `data/ambulance_demo_output/`
5. Tests integration with visualization tools
6. Shows different scenario examples and usage patterns
7. Demonstrates statistics and monitoring capabilities

### `test_ambulance_demo.py`
**Test script** to verify the demonstration functionality works correctly.

**Usage:**
```bash
python collecting_ambulance_data/examples/test_ambulance_demo.py
```

### `basic_ambulance_collection.py`
**Production data collection script** for collecting larger datasets.

**Usage:**
```bash
python collecting_ambulance_data/examples/basic_ambulance_collection.py
```

## Output

The demonstration creates:
- **Data files**: Stored in `data/ambulance_demo_output/`
- **Log file**: `ambulance_demo.log` with detailed execution logs
- **Dataset index**: JSON file cataloging all collected data

## Requirements

The ambulance data collection system requires:
- All 15 ambulance scenarios properly configured
- Multi-modal observation support (Kinematics, OccupancyGrid, GrayscaleObservation)
- Integration with existing highway data collection infrastructure
- Visualization tool compatibility

## Key Features Demonstrated

### 🚑 Ambulance Scenarios
- 15 distinct highway scenarios with varying traffic conditions
- Emergency vehicle as ego agent (first controlled vehicle)
- Realistic emergency response situations

### 🔍 Multi-Modal Observations
- **Kinematics**: Vehicle state data (position, velocity, heading)
- **OccupancyGrid**: Spatial grid representation of environment
- **GrayscaleObservation**: Visual/image observations of driving scene

### 📊 Data Collection
- Batch processing with configurable episode counts
- Error handling and recovery mechanisms
- Performance monitoring and statistics
- Validation and quality assurance

### 🎨 Visualization Integration
- Compatible with existing `MultimodalParquetPlotter`
- Works with `ComprehensiveDataPlotter`
- Supports all current analysis workflows

## Next Steps

After running the demonstration:

1. **Scale up**: Use `basic_ambulance_collection.py` for larger datasets
2. **Customize**: Modify scenarios in `ambulance_scenarios.py`
3. **Analyze**: Use existing visualization tools with ambulance data
4. **Integrate**: Incorporate into your machine learning workflows

## Troubleshooting

If the demonstration fails:

1. Check that all dependencies are installed
2. Verify the ambulance scenarios are properly configured
3. Ensure sufficient disk space for data storage
4. Review the log file (`ambulance_demo.log`) for detailed error information

## Requirements Satisfied

This demonstration satisfies the following requirements:
- **5.4**: Integration with existing infrastructure and tools
- **6.3**: Easy configuration and extensibility of scenarios