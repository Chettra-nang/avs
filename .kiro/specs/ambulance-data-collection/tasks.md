# Implementation Plan

- [x] 1. Create collecting_ambulance_data folder structure
  - Create main directory and subdirectories for scenarios, collection, and examples
  - Set up __init__.py files for proper Python package structure
  - Create README.md with documentation and usage instructions
  - _Requirements: 3.1, 3.2_

- [x] 2. Define 15 ambulance highway scenarios configuration
  - Create ambulance_scenarios.py with 15 distinct highway scenarios
  - Configure each scenario with 4-lane highway, varying traffic densities and conditions
  - Set controlled_vehicles to 4 with first agent as ambulance, others as normal vehicles
  - Include horizontal image orientation settings for all scenarios
  - _Requirements: 3.2, 3.3, 4.1, 4.2, 4.3_

- [x] 3. Extend MultiAgentEnvFactory for ambulance support
  - Add create_ambulance_env method to support ambulance ego vehicle configuration
  - Modify get_base_config to handle ambulance-specific vehicle type settings
  - Ensure first controlled vehicle is configured as ambulance type
  - Maintain compatibility with existing multi-modal observation types
  - _Requirements: 1.1, 1.2, 2.2, 5.1_

- [x] 4. Extend ScenarioRegistry for ambulance scenarios
  - Add ambulance scenarios to the existing registry system
  - Implement validation for ambulance-specific configuration parameters
  - Ensure ambulance scenarios follow existing configuration patterns
  - Add methods to retrieve and customize ambulance scenarios
  - _Requirements: 5.3, 6.1, 6.2, 6.4_

- [x] 4.1 Implement multi-modal data collection for ambulance scenarios
  - Update ambulance scenarios to support all 3 observation types (Kinematics, OccupancyGrid, GrayscaleObservation)
  - Remove hardcoded Kinematics-only observation configuration from ambulance scenarios
  - Enable dynamic observation type selection like the main highway data collection system
  - Ensure ambulance scenarios can collect visual, spatial, and kinematic data simultaneously
  - _Requirements: 1.2, 1.4_

- [x] 5. Create AmbulanceDataCollector class
  - Implement specialized collector that uses existing SynchronizedCollector
  - Add methods for setting up ambulance environments with proper agent configuration
  - Ensure data collection works with all observation modalities (Kinematics, OccupancyGrid, GrayscaleObservation)
  - Integrate with existing DatasetStorageManager for data storage
  - _Requirements: 1.3, 1.4, 5.2_

- [x] 6. Implement ambulance environment configuration
  - Configure highway-env to use ambulance vehicle type for first agent
  - Set up horizontal image rendering for visual observations
  - Ensure 4-lane highway configuration across all scenarios
  - Configure 3-4 controlled agents with ambulance as ego vehicle
  - _Requirements: 1.1, 2.1, 2.3, 4.4_

- [x] 7. Create ambulance demonstration script
  - Build ambulance_demo.py showing basic ambulance data collection
  - Demonstrate multi-modal data collection with ambulance scenarios
  - Show integration with existing visualization tools
  - Include examples of running different ambulance scenarios
  - _Requirements: 5.4, 6.3_

- [x] 8. Create basic ambulance collection script
  - Implement basic_ambulance_collection.py for simple data collection workflows
  - Allow users to specify which ambulance scenarios to run
  - Integrate with existing storage and output directory structure
  - Provide progress tracking and error handling
  - _Requirements: 3.4, 5.4_

- [x] 9. Add ambulance scenario validation and testing
  - Implement validation for ambulance scenario configurations
  - Test each of the 15 scenarios individually for proper execution
  - Verify ambulance ego vehicle is correctly configured in each scenario
  - Ensure multi-agent setup works with ambulance and normal vehicles
  - _Requirements: 6.4, 2.4_

- [x] 10. Integrate with existing data collection pipeline
  - Ensure ambulance data collection works with existing ActionSampler classes
  - Verify compatibility with current feature extraction systems
  - Test integration with existing storage and visualization tools
  - Maintain backward compatibility with existing data collection workflows
  - _Requirements: 5.1, 5.2, 5.4_

- [x] 11. Create comprehensive tests for ambulance system
  - when run please activate the environment source avs_venv/bin/activate
  - Write unit tests for ambulance scenario configurations
  - Test ambulance environment creation and agent setup
  - Verify data collection produces expected multi-modal outputs
  - Test error handling and edge cases
  - _Requirements: 1.4, 2.4, 6.4_

- [x] 12. Document ambulance data collection system
  - when run please activate the environment source avs_venv/bin/activate
  - Update README.md with ambulance-specific usage instructions
  - Document the 15 ambulance scenarios and their characteristics
  - Provide examples of how to run and customize ambulance data collection
  - Include integration notes for existing tools and workflows
  - _Requirements: 3.4, 6.1, 6.3_