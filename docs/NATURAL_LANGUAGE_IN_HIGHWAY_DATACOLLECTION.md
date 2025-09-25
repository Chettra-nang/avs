# Natural Language in Highway Data Collection

This document explains exactly where and how natural language processing is integrated into your `highway_datacollection` system.

## 🗣️ **Where Natural Language Lives in Your Code**

### **📁 1. Core Implementation**

**Location: `highway_datacollection/features/summarizer.py`**

This is the main natural language engine with:
- `LanguageSummarizer` class - Generates contextual driving descriptions
- Scenario-specific templates (free_flow, dense_commuting, stop_and_go, etc.)
- Rich feature extraction for natural language generation

```python
from highway_datacollection.features.summarizer import LanguageSummarizer

summarizer = LanguageSummarizer(lane_width=4.0, num_lanes=4)
summary = summarizer.summarize(ego_vehicle, other_vehicles, context)
```

### **📁 2. Integration Layer**

**Location: `highway_datacollection/features/engine.py`**

The `FeatureDerivationEngine` integrates natural language with other features:

```python
class FeatureDerivationEngine:
    def __init__(self, lane_width: float = 4.0, num_lanes: int = 4):
        self.summarizer = LanguageSummarizer(lane_width, num_lanes)
    
    def generate_language_summary(self, ego, others, config=None):
        return self.summarizer.summarize(ego, others, config)
```

### **📁 3. Data Collection Integration**

**Location: `highway_datacollection/collection/collector.py` (lines 870-876)**

Natural language is generated during data collection:

```python
# In SynchronizedCollector._process_kinematics_observation()
if config.feature_extraction_enabled:
    kinematics_features = feature_engine.derive_kinematics_features(kin_array)
    ttc = feature_engine.calculate_ttc(ego_vehicle, other_vehicles)
    summary = feature_engine.generate_language_summary(ego_vehicle, other_vehicles)  # ← HERE!
    
    # Add to processed observation
    processed_obs['ttc'] = ttc
    processed_obs['summary_text'] = summary  # ← STORED HERE!
```

### **📁 4. Data Storage**

**Location: `highway_datacollection/storage/types.py`**

Natural language is part of the core data structure:

```python
@dataclass
class ProcessedObservation:
    # ... other fields ...
    ttc: float
    summary_text: str  # ← Natural language stored here
```

## 🔄 **How Natural Language Flows Through Your System**

### **Step 1: Data Collection**
```
Environment → Raw Observations → Feature Engine → Natural Language Summary
```

### **Step 2: Processing Pipeline**
```python
# In collector.py
ego_vehicle = kin_array[0]  # Extract ego vehicle
other_vehicles = kin_array[1:]  # Extract other vehicles

# Generate natural language
summary = feature_engine.generate_language_summary(ego_vehicle, other_vehicles)

# Store in processed observation
processed_obs['summary_text'] = summary
```

### **Step 3: Storage**
```
Processed Observation → Parquet File → summary_text column
```

### **Step 4: Visualization**
```
Parquet File → Your Plotter → Natural Language Display
```

## 🎯 **Natural Language Templates by Scenario**

Your system has specialized templates for different driving scenarios:

### **Free Flow Template**
```python
def _free_flow_template(self, features, context):
    return f"Vehicle is {features['speed_description']} at {features['speed_kmh']:.1f} km/h in the {features['lane_description']}. Traffic is {features['traffic_description']} with {features['num_vehicles']} other vehicles nearby."
```

**Example Output:**
> "Vehicle is moving at highway speed at 85.3 km/h in the left lane. Traffic is light traffic with 3 other vehicles nearby. Large gap to lead vehicle. Maneuver assessment: left lane change available, clear road ahead."

### **Dense Commuting Template**
```python
def _dense_commuting_template(self, features, context):
    return f"In heavy commuter traffic, vehicle is {features['speed_description']} at {features['speed_kmh']:.1f} km/h in the {features['lane_description']}. {features['gap_description'].capitalize()}. Collision assessment: {features['ttc_description']}."
```

**Example Output:**
> "In heavy commuter traffic, vehicle is moving slowly at 25.7 km/h in the right lane. Close to lead vehicle. Collision assessment: moderate collision risk. Available maneuvers: limited maneuver options."

### **Stop and Go Template**
```python
def _stop_and_go_template(self, features, context):
    return f"In stop-and-go traffic, vehicle is nearly stationary at {features['speed_kmh']:.1f} km/h in the {features['lane_description']}. {features['gap_description'].capitalize()}. Following closely due to congested conditions."
```

**Example Output:**
> "In stop-and-go traffic, vehicle is nearly stationary at 8.2 km/h in the middle lane. Very close to lead vehicle. Following closely due to congested conditions."

## 📊 **What Natural Language Captures**

Your natural language system captures:

### **1. Vehicle State**
- Speed and movement description
- Lane position and context
- Heading and trajectory

### **2. Traffic Context**
- Number of nearby vehicles
- Traffic density assessment
- Gap to lead vehicle

### **3. Safety Assessment**
- Time-to-collision analysis
- Collision risk evaluation
- Following distance status

### **4. Maneuver Opportunities**
- Available lane changes
- Overtaking possibilities
- Tactical driving options

### **5. Scenario-Specific Context**
- Free flow conditions
- Commuter traffic patterns
- Stop-and-go situations
- Aggressive driver scenarios

## 🔧 **How to Use Natural Language in Your Code**

### **1. During Data Collection**
Natural language is automatically generated when you collect data:

```python
from highway_datacollection.collection.collector import SynchronizedCollector

collector = SynchronizedCollector(n_agents=2)
results = collector.collect_episodes(n_episodes=10)

# Each observation will have 'summary_text' field
for episode in results.episodes:
    for obs in episode.observations:
        print(f"Step {obs.step}: {obs.summary_text}")
```

### **2. In Data Analysis**
Access natural language from stored data:

```python
import pandas as pd

# Load your collected data
df = pd.read_parquet('episode_transitions.parquet')

# Analyze natural language summaries
summaries = df['summary_text'].tolist()
print(f"Collected {len(summaries)} natural language descriptions")

# Example summaries
for i, summary in enumerate(summaries[:5]):
    print(f"{i+1}. {summary}")
```

### **3. In Visualization**
Your plotter automatically displays natural language:

```python
from visualization.multimodal_parquet_plotter import MultimodalParquetPlotter

plotter = MultimodalParquetPlotter("data/your_dataset")
plotter.plot_episode_overview(episode_df, "episode_001")
# Natural language appears in the bottom panel automatically!
```

## 🎨 **Customizing Natural Language**

### **1. Add New Scenario Templates**
```python
# In highway_datacollection/features/summarizer.py
def _custom_scenario_template(self, features, context):
    return f"Custom scenario: {features['speed_description']} at {features['speed_kmh']:.1f} km/h"

# Register in __init__
self._scenario_templates['custom_scenario'] = self._custom_scenario_template
```

### **2. Modify Feature Descriptions**
```python
def _describe_speed(self, speed_kmh: float) -> str:
    if speed_kmh < 5:
        return "crawling"  # Custom description
    elif speed_kmh < 30:
        return "moving slowly"
    # ... etc
```

### **3. Add Context Information**
```python
# Pass additional context during collection
context = {
    'scenario': 'free_flow',
    'weather': 'rainy',
    'time_of_day': 'rush_hour'
}

summary = feature_engine.generate_language_summary(ego, others, context)
```

## ✅ **Verification Your Natural Language Works**

### **1. Check Data Collection**
```python
# Run a simple collection test
from highway_datacollection.collection.collector import SynchronizedCollector

collector = SynchronizedCollector(n_agents=1)
results = collector.collect_episodes(n_episodes=1)

# Check if natural language is generated
sample_obs = results.episodes[0].observations[0]
print(f"Natural language working: {hasattr(sample_obs, 'summary_text')}")
print(f"Sample summary: {sample_obs.summary_text}")
```

### **2. Check Stored Data**
```python
# Verify natural language in stored Parquet files
import pandas as pd

df = pd.read_parquet('your_episode_file.parquet')
print(f"Has summary_text column: {'summary_text' in df.columns}")
print(f"Non-null summaries: {df['summary_text'].notna().sum()}/{len(df)}")
```

### **3. Check Visualization**
```python
# Test visualization with natural language
from visualization.multimodal_parquet_plotter import MultimodalParquetPlotter

plotter = MultimodalParquetPlotter("your_dataset_path")
plotter.plot_episode_overview(episode_df, "episode_id")
# Look for natural language panel in the generated plot
```

## 🎉 **Summary**

Your natural language system is **fully integrated** into your highway data collection pipeline:

- ✅ **Implemented**: `LanguageSummarizer` in `features/summarizer.py`
- ✅ **Integrated**: Feature engine calls summarizer during collection
- ✅ **Stored**: `summary_text` field in Parquet files
- ✅ **Visualized**: Automatic display in your plotting system
- ✅ **Tested**: Comprehensive test coverage

**Your natural language processing is working and ready to use!** 🚗💬