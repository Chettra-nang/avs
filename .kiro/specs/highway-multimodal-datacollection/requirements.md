# Requirements Document

## Introduction

This feature implements a comprehensive multi-agent data collection system for HighwayEnv that captures synchronized observations across multiple modalities (Kinematics, OccupancyGrid, Grayscale) along with derived metrics and natural language summaries. The system supports curriculum-based scenario generation for training both reinforcement learning agents and large language model planners in autonomous driving contexts.

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to collect synchronized multi-modal observations from multiple agents in highway driving scenarios, so that I can train both RL and LLM-based autonomous driving systems with rich, aligned datasets.

#### Acceptance Criteria

1. WHEN the system runs a multi-agent simulation THEN it SHALL capture Kinematics, OccupancyGrid, and Grayscale observations simultaneously for each controlled vehicle
2. WHEN collecting observations THEN the system SHALL ensure all modalities are synchronized using identical seeds and actions across parallel environments
3. WHEN storing observations THEN the system SHALL maintain alignment by keying records with episode_id, step, and agent_id
4. IF any environment signals done or truncated THEN the system SHALL stop data collection for that episode across all modalities

### Requirement 2

**User Story:** As a researcher, I want to generate curriculum-based scenarios with varying traffic conditions, so that I can train robust autonomous driving models across diverse driving situations.

#### Acceptance Criteria

1. WHEN generating scenarios THEN the system SHALL support six curriculum types: free_flow, dense_commuting, stop_and_go, aggressive_neighbors, lane_closure, and time_budget
2. WHEN configuring scenarios THEN the system SHALL allow customizable vehicles_count, lanes_count, and duration parameters for each scenario type
3. WHEN running scenarios THEN the system SHALL tag each episode with scenario metadata for downstream analysis
4. WHEN collecting data THEN the system SHALL support configurable number of controlled vehicles (minimum 2 for multi-agent scenarios)

### Requirement 3

**User Story:** As a researcher, I want derived metrics and natural language summaries for each observation, so that I can use the data for both numerical analysis and language model training.

#### Acceptance Criteria

1. WHEN processing Kinematics observations THEN the system SHALL calculate Time-to-Collision (TTC) from relative positions and velocities
2. WHEN generating summaries THEN the system SHALL create natural language descriptions including lane position, speed, gaps, TTC, traffic density, and maneuver context
3. WHEN estimating vehicle states THEN the system SHALL derive lane assignments, lead vehicle identification, and gap measurements
4. WHEN calculating traffic metrics THEN the system SHALL provide density measurements and availability of lane change opportunities

### Requirement 4

**User Story:** As a researcher, I want efficient storage of large multi-modal datasets, so that I can manage and access the collected data effectively for training and analysis.

#### Acceptance Criteria

1. WHEN storing transition data THEN the system SHALL use Parquet format with CSV fallback for tabular data
2. WHEN storing large arrays THEN the system SHALL use binary blob encoding with explicit shape and dtype metadata for OccupancyGrid and Grayscale observations
3. WHEN organizing data THEN the system SHALL create separate directories for each scenario type with episode-specific files
4. WHEN indexing data THEN the system SHALL generate a global index.json file listing all scenario files and their metadata

### Requirement 5

**User Story:** As a researcher, I want deterministic and reproducible data collection, so that I can validate results and ensure consistent training datasets.

#### Acceptance Criteria

1. WHEN running episodes THEN the system SHALL use identical seeds across all parallel environments to guarantee synchronized trajectories
2. WHEN configuring environments THEN the system SHALL use consistent multi-agent configurations with identical action spaces and observation parameters
3. WHEN collecting data THEN the system SHALL support configurable episode counts, agent counts, and maximum steps per episode
4. WHEN storing metadata THEN the system SHALL record complete configuration parameters for each episode

### Requirement 6

**User Story:** As a researcher, I want extensible architecture for policy integration, so that I can replace random actions with trained agents and customize data collection strategies.

#### Acceptance Criteria

1. WHEN sampling actions THEN the system SHALL provide hooks to replace random action sampling with trained policy inference
2. WHEN configuring observations THEN the system SHALL support toggling modalities per scenario for focused data collection
3. WHEN extending functionality THEN the system SHALL maintain modular design allowing addition of new observation types or derived metrics
4. IF using trained policies THEN the system SHALL maintain deterministic behavior through proper seed management

### Requirement 7

**User Story:** As a researcher, I want comprehensive observation capture including vehicle dynamics and spatial representations, so that I can support diverse machine learning approaches.

#### Acceptance Criteria

1. WHEN capturing Kinematics THEN the system SHALL record presence, position (x,y), velocity (vx,vy), and heading (cos_h, sin_h) for ego and surrounding vehicles
2. WHEN capturing OccupancyGrid THEN the system SHALL provide spatial encoding around ego vehicle with configurable grid dimensions
3. WHEN capturing Grayscale THEN the system SHALL support vision-based observations with proper image stack handling
4. WHEN normalizing observations THEN the system SHALL apply ego-relative transformations and feature normalization as configured