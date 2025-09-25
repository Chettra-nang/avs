# 🚗 Data Collection Scenarios

This directory contains 10 different highway driving scenarios for collecting comprehensive multimodal datasets. Each scenario collects **1000 episodes** with complete multimodal data including kinematics, occupancy grids, and grayscale images.

## 📋 **Scenarios Overview**

| Scenario | Description | Episodes | Vehicles | Lanes | Agents | Focus |
|----------|-------------|----------|----------|-------|---------|-------|
| **01** | Light Free Flow | 1000 | 15 | 4 | 2 | Baseline smooth driving |
| **02** | Heavy Commuting | 1000 | 90 | 4 | 3 | Dense traffic interactions |
| **03** | Stop and Go | 1000 | 65 | 3 | 2 | Traffic congestion patterns |
| **04** | Aggressive Behaviors | 1000 | 50 | 4 | 2 | Safety-critical situations |
| **05** | Lane Closure | 1000 | 55 | 3 | 2 | Merging coordination |
| **06** | Time Budget | 1000 | 40 | 4 | 2 | Efficiency under pressure |
| **07** | Multi-Lane Highway | 1000 | 75 | 6 | 3 | Wide highway dynamics |
| **08** | Mixed Traffic | 1000 | 45 | 4 | 2 | Varied density conditions |
| **09** | High-Speed Extended | 1000 | 35 | 4 | 2 | Long-range planning |
| **10** | Multi-Agent Coordination | 1000 | 70 | 5 | 4 | Complex agent interactions |

**Total: 10,000 episodes across 10 diverse scenarios**

## 🎯 **Data Collection Specifications**

### **Modalities Collected**
All scenarios use the complete multimodal collection setup:
```python
collector = SynchronizedCollector(
    modalities=["kinematics", "occupancy_grid", "grayscale"]
)
```

### **Data Types**
- **Kinematics**: Vehicle positions, velocities, accelerations
- **Occupancy Grid**: Spatial traffic representation (binary grids)
- **Grayscale Images**: Visual observations for computer vision
- **Derived Features**: TTC, lane position, traffic density, vehicle count
- **Actions**: Discrete actions (SLOWER, IDLE, FASTER, LANE_LEFT, LANE_RIGHT)
- **Rewards**: Environment rewards and safety metrics

### **Storage Structure**
Each scenario creates its own data folder:
```
data/
├── scenario_01_light_free_flow/
│   ├── index.json
│   └── free_flow/
│       ├── episode_001_transitions.parquet
│       ├── episode_001_meta.jsonl
│       └── ...
├── scenario_02_heavy_commuting/
├── scenario_03_stop_and_go/
└── ...
```

## 🚀 **Quick Start**

### **Run Single Scenario**
```bash
# Activate environment
source avs_venv/bin/activate

# Run specific scenario
python collecting_data/scenario_01_light_free_flow.py
```

### **Run All Scenarios (Sequential)**
```bash
# Run all 10 scenarios one by one
python collecting_data/run_all_scenarios.py --mode sequential
```

### **Run All Scenarios (Parallel)**
```bash
# Run scenarios in parallel (3 workers by default)
python collecting_data/run_all_scenarios.py --mode parallel --workers 3
```

### **Run Specific Scenarios**
```bash
# Run only scenarios 1, 3, and 5
python collecting_data/run_all_scenarios.py --scenarios scenario_01_light_free_flow scenario_03_stop_and_go scenario_05_lane_closure
```

## 📊 **Expected Output**

### **Collection Statistics**
- **Total Episodes**: 10,000 (1,000 per scenario)
- **Estimated Collection Time**: 
  - Sequential: ~4-6 hours
  - Parallel (3 workers): ~2-3 hours
- **Expected Data Size**: ~50-100 GB (depending on compression)
- **Success Rate**: >95% (with error handling and retry logic)

### **Per-Scenario Output**
Each scenario will produce:
```
✓ Total episodes: 1000
✓ Successful episodes: 980-1000
✓ Collection time: 15-45 minutes
✓ Data saved to: data/scenario_XX_name/index.json
```

## 📁 **Scenario Details**

### **Scenario 1: Light Free Flow** 
- **Purpose**: Baseline smooth driving behavior
- **Characteristics**: Light traffic (15 vehicles), 4 lanes, minimal interactions
- **Use Cases**: Baseline model training, smooth trajectory analysis
- **Duration**: ~15 minutes collection time

### **Scenario 2: Heavy Commuting**
- **Purpose**: Dense traffic with frequent lane changes
- **Characteristics**: Heavy traffic (90 vehicles), 4 lanes, 3 agents
- **Use Cases**: Multi-agent coordination, traffic flow optimization
- **Duration**: ~45 minutes collection time

### **Scenario 3: Stop and Go**
- **Purpose**: Traffic congestion patterns
- **Characteristics**: Congested traffic (65 vehicles), 3 lanes, stop-go dynamics
- **Use Cases**: Congestion management, patience modeling
- **Duration**: ~35 minutes collection time

### **Scenario 4: Aggressive Behaviors**
- **Purpose**: Safety-critical driving situations
- **Characteristics**: Moderate traffic (50 vehicles) with aggressive behaviors
- **Use Cases**: Safety analysis, collision avoidance, risk assessment
- **Duration**: ~25 minutes collection time

### **Scenario 5: Lane Closure**
- **Purpose**: Merging coordination scenarios
- **Characteristics**: Moderate traffic (55 vehicles), 3 lanes (simulating closure)
- **Use Cases**: Merging strategies, coordination algorithms
- **Duration**: ~30 minutes collection time

### **Scenario 6: Time Budget**
- **Purpose**: Efficiency under time pressure
- **Characteristics**: Moderate traffic (40 vehicles), time-pressured driving
- **Use Cases**: Efficiency optimization, urgency modeling
- **Duration**: ~20 minutes collection time

### **Scenario 7: Multi-Lane Highway**
- **Purpose**: Wide highway dynamics
- **Characteristics**: Heavy traffic (75 vehicles), 6 lanes, 3 agents
- **Use Cases**: Wide highway navigation, lane selection strategies
- **Duration**: ~40 minutes collection time

### **Scenario 8: Mixed Traffic**
- **Purpose**: Varied density conditions
- **Characteristics**: Medium traffic (45 vehicles), mixed conditions
- **Use Cases**: Adaptive behavior modeling, condition recognition
- **Duration**: ~30 minutes collection time

### **Scenario 9: High-Speed Extended**
- **Purpose**: Long-range planning analysis
- **Characteristics**: Light-moderate traffic (35 vehicles), extended duration (60s)
- **Use Cases**: Long-term planning, highway cruising behavior
- **Duration**: ~50 minutes collection time

### **Scenario 10: Multi-Agent Coordination**
- **Purpose**: Complex agent interactions
- **Characteristics**: High traffic (70 vehicles), 5 lanes, 4 agents
- **Use Cases**: Advanced multi-agent systems, coordination protocols
- **Duration**: ~60 minutes collection time

## 🔧 **Configuration & Customization**

### **Modify Collection Parameters**
Edit individual scenario files to adjust:
- `episodes_per_scenario`: Number of episodes to collect
- `n_agents`: Number of controlled agents
- `max_steps_per_episode`: Episode length
- `batch_size`: Processing batch size
- `vehicles_count`: Traffic density
- `lanes_count`: Highway width

### **Example Customization**
```python
# In scenario_01_light_free_flow.py
collection_config = {
    "episodes_per_scenario": 500,  # Reduce to 500 episodes
    "vehicles_count": 25,          # Increase traffic
    "batch_size": 10,              # Smaller batches
}
```

## 🐛 **Troubleshooting**

### **Common Issues**

**Memory Issues**
```bash
# Reduce batch size in scenario scripts
batch_size: 5  # Instead of 20
```

**Disk Space**
- Each scenario: ~5-10 GB
- Monitor disk space: `df -h`
- Clean up failed runs: `rm -rf data/scenario_*/failed_*`

**Collection Failures**
```bash
# Check logs
tail -f logs/master_collection.log

# Resume failed scenarios
python collecting_data/run_all_scenarios.py --scenarios scenario_XX_name
```

### **Performance Optimization**
- **Parallel Mode**: Use `--workers 2-4` based on CPU cores
- **Batch Size**: Adjust based on available RAM
- **SSD Storage**: Use SSD for faster I/O during collection

## 📈 **Expected Results**

### **Data Quality Metrics**
- Episode completion rate: >95%
- Average episode length: 100-300 steps
- Data consistency checks: Automated validation
- Missing data: <1% per modality

### **Collection Performance**
- Episodes/hour: 50-200 (depending on complexity)
- Memory usage: 2-8 GB peak
- CPU utilization: 80-100% during collection
- Storage I/O: High during parquet writing

## 🎯 **Next Steps**

After collection completion:

1. **Data Validation**: Run `python main.py --demo loading` to verify data integrity
2. **Data Analysis**: Use examples in `examples/` folder for exploration
3. **Model Training**: Proceed with ML model training using collected datasets
4. **Visualization**: Generate plots and analysis reports

## 📚 **Related Documentation**

- [DATA_COLLECTION_TUTORIAL.md](../docs/DATA_COLLECTION_TUTORIAL.md) - Detailed tutorial
- [examples/](../examples/) - Usage examples and demos
- [docs/SETUP_AND_RUN_GUIDE.md](../docs/SETUP_AND_RUN_GUIDE.md) - Setup guide
- [main.py](../main.py) - System demonstration script

---

**Total Expected Dataset: 10,000 episodes of multimodal highway driving data** 🚗📊