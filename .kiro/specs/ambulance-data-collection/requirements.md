# Requirements Document

## Introduction

This feature extends the existing highway data collection system to support ambulance ego vehicles with specialized scenarios. The system will maintain all current data collection capabilities while introducing ambulance-specific vehicle behavior and 15 diverse scenarios that capture emergency vehicle interactions with regular traffic. The ambulance ego vehicle will have distinct characteristics while other agents remain as normal vehicles, creating realistic emergency response scenarios for training autonomous systems.

## Requirements

### Requirement 1

**User Story:** As a researcher studying emergency vehicle behavior, I want to collect multi-modal data with an ambulance as the ego vehicle, so that I can train models to understand emergency vehicle dynamics and traffic interactions.

#### Acceptance Criteria

1. WHEN the system creates an environment THEN the ego vehicle SHALL be configured as an ambulance type
2. WHEN the ambulance ego vehicle is active THEN it SHALL maintain all existing observation modalities (Kinematics, OccupancyGrid, GrayscaleObservation)
3. WHEN data collection runs THEN the ambulance SHALL be visually distinguishable from regular vehicles
4. WHEN collecting data THEN all existing data collection features SHALL remain functional

### Requirement 2

**User Story:** As a traffic simulation researcher, I want 3-4 controlled agents with the first agent fixed as an ambulance ego vehicle and others as normal vehicles, so that I can study multi-agent interactions during emergency scenarios.

#### Acceptance Criteria

1. WHEN the environment is configured THEN there SHALL be 3-4 controlled agents total
2. WHEN agents are initialized THEN the first agent (index 0) SHALL always be the ambulance ego vehicle
3. WHEN scenarios run THEN remaining controlled agents SHALL be normal vehicle types
4. WHEN scenarios execute THEN vehicle interactions SHALL reflect realistic emergency vehicle scenarios with multiple controlled agents

### Requirement 3

**User Story:** As a data scientist, I want 15 distinct ambulance scenarios stored in a dedicated folder, so that I can collect diverse emergency vehicle data across different traffic conditions.

#### Acceptance Criteria

1. WHEN the system initializes THEN it SHALL create a "collecting_ambulance_data" folder structure
2. WHEN scenarios are defined THEN there SHALL be exactly 15 unique ambulance scenarios
3. WHEN each scenario runs THEN it SHALL have distinct traffic patterns, densities, and conditions
4. WHEN scenarios are executed THEN they SHALL cover emergency response situations like hospital runs, accident responses, and traffic navigation

### Requirement 4

**User Story:** As a machine learning engineer, I want ambulance scenarios to include varied traffic densities and emergency situations, so that I can train robust models for emergency vehicle navigation.

#### Acceptance Criteria

1. WHEN scenarios are created THEN they SHALL include light, moderate, and heavy traffic conditions
2. WHEN emergency scenarios run THEN they SHALL simulate realistic ambulance response situations
3. WHEN traffic patterns are generated THEN they SHALL include rush hour, accident scenes, and hospital approach scenarios
4. WHEN scenarios execute THEN they SHALL vary in duration, lane configurations, and vehicle behaviors

### Requirement 5

**User Story:** As a system administrator, I want the ambulance data collection to integrate seamlessly with existing infrastructure, so that I can use current tools and workflows without modification.

#### Acceptance Criteria

1. WHEN ambulance collection runs THEN it SHALL use the existing MultiAgentEnvFactory
2. WHEN data is collected THEN it SHALL be stored using the current DatasetStorageManager
3. WHEN scenarios are configured THEN they SHALL extend the existing ScenarioRegistry
4. WHEN the system operates THEN all existing visualization and analysis tools SHALL work with ambulance data

### Requirement 6

**User Story:** As a researcher, I want ambulance scenarios to be easily configurable and extensible, so that I can modify parameters and add new emergency situations as needed.

#### Acceptance Criteria

1. WHEN scenarios are defined THEN they SHALL follow the existing configuration pattern
2. WHEN parameters need modification THEN they SHALL be adjustable through scenario configs
3. WHEN new scenarios are added THEN they SHALL integrate with the existing registry system
4. WHEN configurations change THEN they SHALL be validated against system constraints