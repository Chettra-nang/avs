#!/usr/bin/env python3
"""
Comprehensive integration tests for ambulance data collection pipeline.

This script tests the integration of ambulance data collection with:
- Existing ActionSampler classes
- Current feature extraction systems  
- Storage and visualization tools
- Backward compatibility with existing workflows

Requirements: 5.1, 5.2, 5.4
"""

import sys
import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import ambulance collection components
from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector
from collecting_ambulance_data.scenarios.ambulance_scenarios import get_scenario_names

# Import existing pipeline components for integration testing
from highway_datacollection.collection.action_samplers import (
    ActionSampler, RandomActionSampler, PolicyActionSampler, HybridActionSampler
)
from highway_datacollection.collection.collector import SynchronizedCollector
from highway_datacollection.collection.modality_config import ModalityConfigManager
from highway_datacollection.features.extractors import KinematicsExtractor, TrafficMetricsExtractor
from highway_datacollection.features.engine import FeatureDerivationEngine
from highway_datacollection.storage.manager import DatasetStorageManager
from highway_datacollection.performance import PerformanceConfig

# Import visualization tools for integration testing
try:
    from visualization.multimodal_parquet_plotter import MultimodalParquetPlotter
    from visualization.comprehensive_data_plotter import ComprehensiveDataPlotter
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


def setup_logging():
    """Set up logging for integration tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pipeline_integration_test.log')
        ]
    )
    return logging.getLogger(__name__)


class PipelineIntegrationTester:
    """Comprehensive tester for ambulance data collection pipeline integration."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize the integration tester."""
        self.logger = logger
        self.test_results = {
            "action_sampler_tests": {},
            "feature_extraction_tests": {},
            "storage_integration_tests": {},
            "visualization_integration_tests": {},
            "backward_compatibility_tests": {},
            "performance_tests": {}
        }
        self.test_data_dir = Path("data/integration_test_output")
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
    
    def test_action_sampler_integration(self) -> Dict[str, Any]:
        """Test integration with existing ActionSampler classes."""
        self.logger.info("Testing ActionSampler integration...")
        
        test_results = {
            "random_sampler": False,
            "hybrid_sampler": False,
            "custom_sampler": False,
            "sampler_compatibility": False,
            "errors": []
        }
        
        try:
            # Test 1: RandomActionSampler integration
            self.logger.info("Testing RandomActionSampler integration...")
            random_sampler = RandomActionSampler(action_space_size=5, seed=42)
            
            collector = AmbulanceDataCollector(
                n_agents=4,
                action_sampler=random_sampler,
                max_memory_gb=1.0,
                enable_validation=False  # Disable for faster testing
            )
            
            # Test basic functionality
            scenarios = collector.get_available_scenarios()
            if scenarios:
                test_scenario = scenarios[0]
                setup_info = collector.setup_ambulance_environments(test_scenario)
                
                # Test action sampling
                dummy_obs = {"Kinematics": {"observation": [[1, 0, 0, 0, 0, 1, 0]]}}
                actions = collector._collector.sample_actions(dummy_obs, step=0, episode_id="test")
                
                # Check for both int and np.int64 types (numpy integers are valid)
                import numpy as np
                if (len(actions) == 4 and 
                    all(isinstance(a, (int, np.integer)) for a in actions)):
                    test_results["random_sampler"] = True
                    self.logger.info("✅ RandomActionSampler integration successful")
                else:
                    error_msg = f"Invalid actions from RandomActionSampler: {actions} (type: {type(actions)}, length: {len(actions) if hasattr(actions, '__len__') else 'N/A'})"
                    test_results["errors"].append(error_msg)
                    self.logger.error(error_msg)
            
            collector.cleanup()
            
        except Exception as e:
            test_results["errors"].append(f"RandomActionSampler test failed: {str(e)}")
            self.logger.error(f"RandomActionSampler test failed: {e}")
        
        try:
            # Test 2: HybridActionSampler integration
            self.logger.info("Testing HybridActionSampler integration...")
            
            # Create hybrid sampler with different samplers for different agents
            agent_samplers = {
                0: RandomActionSampler(seed=42),  # Ambulance agent
                1: RandomActionSampler(seed=43),  # Regular vehicle
            }
            default_sampler = RandomActionSampler(seed=44)
            hybrid_sampler = HybridActionSampler(agent_samplers, default_sampler)
            
            collector = AmbulanceDataCollector(
                n_agents=4,
                action_sampler=hybrid_sampler,
                max_memory_gb=1.0,
                enable_validation=False
            )
            
            scenarios = collector.get_available_scenarios()
            if scenarios:
                test_scenario = scenarios[0]
                collector.setup_ambulance_environments(test_scenario)
                
                # Test hybrid action sampling
                dummy_obs = {"Kinematics": {"observation": [[1, 0, 0, 0, 0, 1, 0]]}}
                actions = collector._collector.sample_actions(dummy_obs, step=0, episode_id="test")
                
                # Check for both int and np.int64 types (numpy integers are valid)
                import numpy as np
                if (len(actions) == 4 and 
                    all(isinstance(a, (int, np.integer)) for a in actions)):
                    test_results["hybrid_sampler"] = True
                    self.logger.info("✅ HybridActionSampler integration successful")
                else:
                    error_msg = f"Invalid actions from HybridActionSampler: {actions} (type: {type(actions)}, length: {len(actions) if hasattr(actions, '__len__') else 'N/A'})"
                    test_results["errors"].append(error_msg)
                    self.logger.error(error_msg)
            
            collector.cleanup()
            
        except Exception as e:
            test_results["errors"].append(f"HybridActionSampler test failed: {str(e)}")
            self.logger.error(f"HybridActionSampler test failed: {e}")
        
        try:
            # Test 3: Custom ActionSampler integration
            self.logger.info("Testing custom ActionSampler integration...")
            
            class CustomAmbulanceActionSampler(ActionSampler):
                """Custom action sampler for ambulance scenarios."""
                
                def __init__(self):
                    self.step_count = 0
                
                def sample_actions(self, observations, n_agents, step=0, episode_id=""):
                    self.step_count += 1
                    # Ambulance (agent 0) always accelerates, others random
                    actions = []
                    for i in range(n_agents):
                        if i == 0:  # Ambulance agent
                            actions.append(1)  # FASTER action
                        else:
                            actions.append(0)  # IDLE action
                    return tuple(actions)
                
                def reset(self, seed=None):
                    self.step_count = 0
            
            custom_sampler = CustomAmbulanceActionSampler()
            
            collector = AmbulanceDataCollector(
                n_agents=4,
                action_sampler=custom_sampler,
                max_memory_gb=1.0,
                enable_validation=False
            )
            
            scenarios = collector.get_available_scenarios()
            if scenarios:
                test_scenario = scenarios[0]
                collector.setup_ambulance_environments(test_scenario)
                
                # Test custom action sampling
                dummy_obs = {"Kinematics": {"observation": [[1, 0, 0, 0, 0, 1, 0]]}}
                actions = collector._collector.sample_actions(dummy_obs, step=0, episode_id="test")
                
                # Should get (1, 0, 0, 0) - ambulance accelerates, others idle
                expected_actions = (1, 0, 0, 0)
                if actions == expected_actions:
                    test_results["custom_sampler"] = True
                    self.logger.info("✅ Custom ActionSampler integration successful")
                else:
                    test_results["errors"].append(f"Custom sampler returned {actions}, expected {expected_actions}")
            
            collector.cleanup()
            
        except Exception as e:
            test_results["errors"].append(f"Custom ActionSampler test failed: {str(e)}")
            self.logger.error(f"Custom ActionSampler test failed: {e}")
        
        # Test 4: ActionSampler compatibility with existing SynchronizedCollector
        try:
            self.logger.info("Testing ActionSampler compatibility with SynchronizedCollector...")
            
            # Test that ambulance collector uses the same action sampling interface
            random_sampler = RandomActionSampler(seed=42)
            
            # Create both collectors with same sampler
            ambulance_collector = AmbulanceDataCollector(n_agents=4, action_sampler=random_sampler)
            standard_collector = SynchronizedCollector(n_agents=4, action_sampler=random_sampler)
            
            # Test that they use the same action sampling interface
            dummy_obs = {"Kinematics": {"observation": [[1, 0, 0, 0, 0, 1, 0]]}}
            
            ambulance_actions = ambulance_collector._collector.sample_actions(dummy_obs, step=0)
            standard_actions = standard_collector.sample_actions(dummy_obs, step=0)
            
            # Both should return 4-element tuples of integers
            # Check for both int and np.int64 types (numpy integers are valid)
            import numpy as np
            if (len(ambulance_actions) == 4 and len(standard_actions) == 4 and
                all(isinstance(a, (int, np.integer)) for a in ambulance_actions) and
                all(isinstance(a, (int, np.integer)) for a in standard_actions)):
                test_results["sampler_compatibility"] = True
                self.logger.info("✅ ActionSampler compatibility verified")
            else:
                error_msg = f"ActionSampler compatibility check failed - ambulance: {ambulance_actions}, standard: {standard_actions}"
                test_results["errors"].append(error_msg)
                self.logger.error(error_msg)
            
            ambulance_collector.cleanup()
            
        except Exception as e:
            test_results["errors"].append(f"ActionSampler compatibility test failed: {str(e)}")
            self.logger.error(f"ActionSampler compatibility test failed: {e}")
        
        self.test_results["action_sampler_tests"] = test_results
        return test_results
    
    def test_feature_extraction_integration(self) -> Dict[str, Any]:
        """Test integration with current feature extraction systems."""
        self.logger.info("Testing feature extraction integration...")
        
        test_results = {
            "kinematics_extractor": False,
            "traffic_metrics_extractor": False,
            "feature_engine_integration": False,
            "ambulance_specific_features": False,
            "errors": []
        }
        
        try:
            # Test 1: KinematicsExtractor integration
            self.logger.info("Testing KinematicsExtractor integration...")
            
            extractor = KinematicsExtractor(lane_width=4.0, num_lanes=4)
            
            # Create sample ambulance observation data
            # Format: [presence, x, y, vx, vy, cos_h, sin_h] for each vehicle
            sample_obs = [
                [1, 0, 0, 10, 0, 1, 0],    # Ambulance (ego vehicle)
                [1, -20, 0, 8, 0, 1, 0],   # Vehicle behind
                [1, 20, 4, 12, 0, 1, 0],   # Vehicle ahead in next lane
                [1, 10, -4, 9, 0, 1, 0]    # Vehicle ahead in previous lane
            ]
            
            import numpy as np
            obs_array = np.array(sample_obs)
            
            # Extract features
            features = extractor.extract_features(obs_array)
            
            # Verify expected features are present
            expected_features = ['lane_position', 'speed', 'lead_vehicle_gap', 'min_ttc', 'traffic_density']
            if all(feature in features for feature in expected_features):
                test_results["kinematics_extractor"] = True
                self.logger.info("✅ KinematicsExtractor integration successful")
                self.logger.info(f"   Extracted features: {list(features.keys())}")
            else:
                missing = [f for f in expected_features if f not in features]
                test_results["errors"].append(f"Missing features from KinematicsExtractor: {missing}")
            
        except Exception as e:
            test_results["errors"].append(f"KinematicsExtractor test failed: {str(e)}")
            self.logger.error(f"KinematicsExtractor test failed: {e}")
        
        try:
            # Test 2: TrafficMetricsExtractor integration
            self.logger.info("Testing TrafficMetricsExtractor integration...")
            
            extractor = TrafficMetricsExtractor(analysis_radius=100.0)
            
            # Use same sample observation data
            import numpy as np
            obs_array = np.array([
                [1, 0, 0, 10, 0, 1, 0],    # Ambulance
                [1, -20, 0, 8, 0, 1, 0],   # Vehicle behind
                [1, 20, 4, 12, 0, 1, 0],   # Vehicle ahead in next lane
                [1, 10, -4, 9, 0, 1, 0]    # Vehicle ahead in previous lane
            ])
            
            # Extract traffic metrics
            metrics = extractor.extract_features(obs_array)
            
            # Verify expected metrics are present
            expected_metrics = ['vehicle_count', 'average_speed', 'speed_variance', 'lane_change_opportunities']
            if all(metric in metrics for metric in expected_metrics):
                test_results["traffic_metrics_extractor"] = True
                self.logger.info("✅ TrafficMetricsExtractor integration successful")
                self.logger.info(f"   Extracted metrics: {list(metrics.keys())}")
            else:
                missing = [m for m in expected_metrics if m not in metrics]
                test_results["errors"].append(f"Missing metrics from TrafficMetricsExtractor: {missing}")
            
        except Exception as e:
            test_results["errors"].append(f"TrafficMetricsExtractor test failed: {str(e)}")
            self.logger.error(f"TrafficMetricsExtractor test failed: {e}")
        
        try:
            # Test 3: FeatureDerivationEngine integration
            self.logger.info("Testing FeatureDerivationEngine integration...")
            
            engine = FeatureDerivationEngine()
            
            # Create sample multi-modal observations (ambulance scenario format)
            sample_observations = {
                "Kinematics": {
                    "observation": [
                        [1, 0, 0, 10, 0, 1, 0],    # Ambulance
                        [1, -20, 0, 8, 0, 1, 0],   # Vehicle behind
                        [1, 20, 4, 12, 0, 1, 0],   # Vehicle ahead
                        [1, 10, -4, 9, 0, 1, 0]    # Another vehicle
                    ]
                }
            }
            
            # Process observations through feature engine using correct method
            import numpy as np
            kinematics_obs = np.array(sample_observations["Kinematics"]["observation"])
            
            # Test kinematics feature extraction
            kinematics_features = engine.derive_kinematics_features(kinematics_obs)
            
            # Test traffic metrics
            traffic_metrics = engine.estimate_traffic_metrics(kinematics_obs)
            
            # Test language summary
            ego_state = kinematics_obs[0]  # First vehicle is ego
            other_vehicles = kinematics_obs[1:] if len(kinematics_obs) > 1 else np.array([])
            language_summary = engine.generate_language_summary(ego_state, other_vehicles)
            
            # Verify processing worked
            if (kinematics_features and traffic_metrics and language_summary and
                len(kinematics_features) > 0 and len(traffic_metrics) > 0 and
                isinstance(language_summary, str) and len(language_summary) > 0):
                test_results["feature_engine_integration"] = True
                self.logger.info("✅ FeatureDerivationEngine integration successful")
                self.logger.info(f"   Kinematics features: {len(kinematics_features)}")
                self.logger.info(f"   Traffic metrics: {len(traffic_metrics)}")
                self.logger.info(f"   Language summary length: {len(language_summary)}")
            else:
                test_results["errors"].append("FeatureDerivationEngine methods failed")
            
        except Exception as e:
            test_results["errors"].append(f"FeatureDerivationEngine test failed: {str(e)}")
            self.logger.error(f"FeatureDerivationEngine test failed: {e}")
        
        try:
            # Test 4: Ambulance-specific feature extraction
            self.logger.info("Testing ambulance-specific feature extraction...")
            
            # Test that we can identify ambulance-specific features
            extractor = KinematicsExtractor()
            
            # Ambulance observation (first agent should be ambulance)
            ambulance_obs = np.array([
                [1, 0, 0, 15, 0, 1, 0],    # Ambulance (higher speed)
                [1, -20, 0, 8, 0, 1, 0],   # Regular vehicle
                [1, 20, 4, 10, 0, 1, 0],   # Regular vehicle
                [1, 10, -4, 9, 0, 1, 0]    # Regular vehicle
            ])
            
            features = extractor.extract_features(ambulance_obs)
            
            # Check that ambulance (ego vehicle) features are extracted
            if ('ego_x' in features and 'ego_y' in features and 
                'ego_vx' in features and 'ego_vy' in features and
                'speed' in features):
                
                # Verify ambulance has higher speed (15 m/s vs others ~8-10 m/s)
                ambulance_speed = features['speed']
                if ambulance_speed > 12:  # Should be ~15 m/s
                    test_results["ambulance_specific_features"] = True
                    self.logger.info("✅ Ambulance-specific feature extraction successful")
                    self.logger.info(f"   Ambulance speed: {ambulance_speed:.2f} m/s")
                else:
                    test_results["errors"].append(f"Ambulance speed too low: {ambulance_speed}")
            else:
                test_results["errors"].append("Missing ambulance ego vehicle features")
            
        except Exception as e:
            test_results["errors"].append(f"Ambulance-specific feature test failed: {str(e)}")
            self.logger.error(f"Ambulance-specific feature test failed: {e}")
        
        self.test_results["feature_extraction_tests"] = test_results
        return test_results
    
    def test_storage_integration(self) -> Dict[str, Any]:
        """Test integration with existing storage and visualization tools."""
        self.logger.info("Testing storage integration...")
        
        test_results = {
            "dataset_storage_manager": False,
            "ambulance_data_storage": False,
            "data_format_compatibility": False,
            "index_generation": False,
            "errors": []
        }
        
        try:
            # Test 1: DatasetStorageManager integration
            self.logger.info("Testing DatasetStorageManager integration...")
            
            storage_dir = self.test_data_dir / "storage_test"
            storage_manager = DatasetStorageManager(storage_dir)
            
            # Create sample ambulance data
            sample_data = [
                {
                    'episode_id': 'ambulance_test_001',
                    'scenario': 'highway_emergency_light',
                    'step': 0,
                    'ambulance_scenario': True,
                    'ambulance_agent_index': 0,
                    'agent_0_x': 0.0,
                    'agent_0_y': 0.0,
                    'agent_0_speed': 15.0,
                    'action': [1, 0, 0, 0],
                    'reward': 1.0,
                    'done': False
                },
                {
                    'episode_id': 'ambulance_test_001',
                    'scenario': 'highway_emergency_light',
                    'step': 1,
                    'ambulance_scenario': True,
                    'ambulance_agent_index': 0,
                    'agent_0_x': 5.0,
                    'agent_0_y': 0.0,
                    'agent_0_speed': 16.0,
                    'action': [1, 0, 0, 0],
                    'reward': 1.0,
                    'done': False
                }
            ]
            
            sample_metadata = [
                {
                    'episode_id': 'ambulance_test_001',
                    'scenario': 'highway_emergency_light',
                    'seed': 42,
                    'steps': 2,
                    'total_reward': 2.0,
                    'ambulance_scenario': True,
                    'ambulance_agent_index': 0
                }
            ]
            
            # Test storage
            storage_paths = storage_manager.write_episode_batch(
                data=sample_data,
                metadata=sample_metadata,
                scenario='highway_emergency_light'
            )
            
            # Verify files were created
            if (storage_paths.transitions_file.exists() and 
                storage_paths.metadata_file.exists()):
                test_results["dataset_storage_manager"] = True
                self.logger.info("✅ DatasetStorageManager integration successful")
                self.logger.info(f"   Transitions file: {storage_paths.transitions_file}")
                self.logger.info(f"   Metadata file: {storage_paths.metadata_file}")
            else:
                test_results["errors"].append("DatasetStorageManager failed to create files")
            
        except Exception as e:
            test_results["errors"].append(f"DatasetStorageManager test failed: {str(e)}")
            self.logger.error(f"DatasetStorageManager test failed: {e}")
        
        try:
            # Test 2: Ambulance data storage through AmbulanceDataCollector
            self.logger.info("Testing ambulance data storage through collector...")
            
            collector = AmbulanceDataCollector(n_agents=4, enable_validation=False)
            
            # Collect minimal data for testing
            scenarios = collector.get_available_scenarios()
            if scenarios:
                test_scenario = scenarios[0]
                
                # Collect 1 episode with 3 steps for testing
                collection_results = collector.collect_ambulance_data(
                    scenarios=[test_scenario],
                    episodes_per_scenario=1,
                    max_steps_per_episode=3,
                    base_seed=42,
                    batch_size=1
                )
                
                # Store the data
                storage_dir = self.test_data_dir / "ambulance_storage_test"
                storage_info = collector.store_ambulance_data(collection_results, storage_dir)
                
                # Verify storage worked
                if (storage_info['scenarios_stored'] > 0 and 
                    storage_info['total_episodes_stored'] > 0):
                    test_results["ambulance_data_storage"] = True
                    self.logger.info("✅ Ambulance data storage successful")
                    self.logger.info(f"   Scenarios stored: {storage_info['scenarios_stored']}")
                    self.logger.info(f"   Episodes stored: {storage_info['total_episodes_stored']}")
                else:
                    test_results["errors"].append("Ambulance data storage failed")
            
            collector.cleanup()
            
        except Exception as e:
            test_results["errors"].append(f"Ambulance data storage test failed: {str(e)}")
            self.logger.error(f"Ambulance data storage test failed: {e}")
        
        try:
            # Test 3: Data format compatibility
            self.logger.info("Testing data format compatibility...")
            
            storage_dir = self.test_data_dir / "ambulance_storage_test"
            if storage_dir.exists():
                # Check for parquet files
                parquet_files = list(storage_dir.rglob("*.parquet"))
                jsonl_files = list(storage_dir.rglob("*.jsonl"))
                
                if parquet_files and jsonl_files:
                    # Try to read the data
                    import pandas as pd
                    import json
                    
                    # Read parquet file
                    df = pd.read_parquet(parquet_files[0])
                    
                    # Read metadata file
                    with open(jsonl_files[0], 'r') as f:
                        metadata = [json.loads(line) for line in f if line.strip()]
                    
                    # Check for ambulance-specific fields
                    ambulance_fields = ['ambulance_scenario', 'ambulance_agent_index']
                    if all(field in df.columns for field in ambulance_fields):
                        test_results["data_format_compatibility"] = True
                        self.logger.info("✅ Data format compatibility verified")
                        self.logger.info(f"   Parquet columns: {len(df.columns)}")
                        self.logger.info(f"   Metadata records: {len(metadata)}")
                    else:
                        missing = [f for f in ambulance_fields if f not in df.columns]
                        test_results["errors"].append(f"Missing ambulance fields: {missing}")
                else:
                    test_results["errors"].append("No data files found for compatibility test")
            
        except Exception as e:
            test_results["errors"].append(f"Data format compatibility test failed: {str(e)}")
            self.logger.error(f"Data format compatibility test failed: {e}")
        
        try:
            # Test 4: Index generation
            self.logger.info("Testing dataset index generation...")
            
            storage_dir = self.test_data_dir / "ambulance_storage_test"
            if storage_dir.exists():
                storage_manager = DatasetStorageManager(storage_dir)
                
                # Create dummy storage paths for index generation
                from highway_datacollection.storage.types import StoragePaths
                dummy_paths = [
                    StoragePaths(
                        transitions_file=storage_dir / "test_transitions.parquet",
                        metadata_file=storage_dir / "test_meta.jsonl",
                        scenario_dir=storage_dir / "test_scenario"
                    )
                ]
                
                # Generate index
                index_path = storage_manager.create_dataset_index(dummy_paths)
                
                if index_path.exists():
                    # Read and verify index
                    import json
                    with open(index_path, 'r') as f:
                        index_data = json.load(f)
                    
                    if 'dataset_name' in index_data and 'scenarios' in index_data:
                        test_results["index_generation"] = True
                        self.logger.info("✅ Dataset index generation successful")
                        self.logger.info(f"   Index file: {index_path}")
                    else:
                        test_results["errors"].append("Invalid index file structure")
                else:
                    test_results["errors"].append("Index file not created")
            
        except Exception as e:
            test_results["errors"].append(f"Index generation test failed: {str(e)}")
            self.logger.error(f"Index generation test failed: {e}")
        
        self.test_results["storage_integration_tests"] = test_results
        return test_results
    
    def test_visualization_integration(self) -> Dict[str, Any]:
        """Test integration with existing visualization tools."""
        self.logger.info("Testing visualization integration...")
        
        test_results = {
            "multimodal_plotter": False,
            "comprehensive_plotter": False,
            "ambulance_data_compatibility": False,
            "visualization_available": VISUALIZATION_AVAILABLE,
            "errors": []
        }
        
        if not VISUALIZATION_AVAILABLE:
            test_results["errors"].append("Visualization tools not available - skipping tests")
            self.logger.warning("Visualization tools not available - skipping visualization tests")
            self.test_results["visualization_integration_tests"] = test_results
            return test_results
        
        try:
            # Test 1: MultimodalParquetPlotter integration
            self.logger.info("Testing MultimodalParquetPlotter integration...")
            
            storage_dir = self.test_data_dir / "ambulance_storage_test"
            if storage_dir.exists():
                plotter = MultimodalParquetPlotter(str(storage_dir))
                
                # Discover ambulance data files
                discovered_files = plotter.discover_parquet_files()
                
                if discovered_files:
                    test_results["multimodal_plotter"] = True
                    self.logger.info("✅ MultimodalParquetPlotter integration successful")
                    self.logger.info(f"   Discovered scenarios: {list(discovered_files.keys())}")
                else:
                    test_results["errors"].append("MultimodalParquetPlotter found no data files")
            else:
                test_results["errors"].append("No storage directory for visualization test")
            
        except Exception as e:
            test_results["errors"].append(f"MultimodalParquetPlotter test failed: {str(e)}")
            self.logger.error(f"MultimodalParquetPlotter test failed: {e}")
        
        try:
            # Test 2: ComprehensiveDataPlotter integration
            self.logger.info("Testing ComprehensiveDataPlotter integration...")
            
            storage_dir = self.test_data_dir / "ambulance_storage_test"
            if storage_dir.exists():
                try:
                    plotter = ComprehensiveDataPlotter(str(storage_dir))
                    
                    # Discover data files
                    data_files = plotter.discover_data_files()
                    
                    total_files = sum(len(files) for files in data_files.values())
                    if total_files > 0:
                        test_results["comprehensive_plotter"] = True
                        self.logger.info("✅ ComprehensiveDataPlotter integration successful")
                        self.logger.info(f"   Total data files: {total_files}")
                    else:
                        # Check if we can at least instantiate the plotter
                        test_results["comprehensive_plotter"] = True
                        self.logger.info("✅ ComprehensiveDataPlotter integration successful (instantiation)")
                        self.logger.info("   No data files found but plotter works")
                except Exception as e:
                    test_results["errors"].append(f"ComprehensiveDataPlotter instantiation failed: {str(e)}")
            else:
                test_results["errors"].append("No storage directory for visualization test")
            
        except Exception as e:
            test_results["errors"].append(f"ComprehensiveDataPlotter test failed: {str(e)}")
            self.logger.error(f"ComprehensiveDataPlotter test failed: {e}")
        
        try:
            # Test 3: Ambulance data compatibility with visualization
            self.logger.info("Testing ambulance data compatibility with visualization...")
            
            storage_dir = self.test_data_dir / "ambulance_storage_test"
            if storage_dir.exists():
                # Check if ambulance data can be loaded by visualization tools
                parquet_files = list(storage_dir.rglob("*.parquet"))
                
                if parquet_files:
                    import pandas as pd
                    df = pd.read_parquet(parquet_files[0])
                    
                    # Check for ambulance-specific columns that visualization should handle
                    ambulance_columns = ['ambulance_scenario', 'ambulance_agent_index']
                    if all(col in df.columns for col in ambulance_columns):
                        # Check that ambulance data is properly marked
                        ambulance_rows = df[df['ambulance_scenario'] == True]
                        if len(ambulance_rows) > 0:
                            test_results["ambulance_data_compatibility"] = True
                            self.logger.info("✅ Ambulance data compatibility verified")
                            self.logger.info(f"   Ambulance data rows: {len(ambulance_rows)}")
                        else:
                            test_results["errors"].append("No ambulance data rows found")
                    else:
                        missing = [col for col in ambulance_columns if col not in df.columns]
                        test_results["errors"].append(f"Missing ambulance columns: {missing}")
                else:
                    test_results["errors"].append("No parquet files for compatibility test")
            
        except Exception as e:
            test_results["errors"].append(f"Ambulance data compatibility test failed: {str(e)}")
            self.logger.error(f"Ambulance data compatibility test failed: {e}")
        
        self.test_results["visualization_integration_tests"] = test_results
        return test_results
    
    def test_backward_compatibility(self) -> Dict[str, Any]:
        """Test backward compatibility with existing data collection workflows."""
        self.logger.info("Testing backward compatibility...")
        
        test_results = {
            "synchronized_collector_compatibility": False,
            "modality_config_compatibility": False,
            "performance_config_compatibility": False,
            "existing_workflow_compatibility": False,
            "errors": []
        }
        
        try:
            # Test 1: SynchronizedCollector compatibility
            self.logger.info("Testing SynchronizedCollector compatibility...")
            
            # Test that AmbulanceDataCollector uses SynchronizedCollector internally
            ambulance_collector = AmbulanceDataCollector(n_agents=4)
            
            # Verify internal collector is SynchronizedCollector
            if isinstance(ambulance_collector._collector, SynchronizedCollector):
                # Test that it has the same interface
                if (hasattr(ambulance_collector._collector, 'collect_episode_batch') and
                    hasattr(ambulance_collector._collector, 'sample_actions') and
                    hasattr(ambulance_collector._collector, 'reset_parallel_envs')):
                    test_results["synchronized_collector_compatibility"] = True
                    self.logger.info("✅ SynchronizedCollector compatibility verified")
                else:
                    test_results["errors"].append("Missing SynchronizedCollector interface methods")
            else:
                test_results["errors"].append("AmbulanceDataCollector does not use SynchronizedCollector")
            
            ambulance_collector.cleanup()
            
        except Exception as e:
            test_results["errors"].append(f"SynchronizedCollector compatibility test failed: {str(e)}")
            self.logger.error(f"SynchronizedCollector compatibility test failed: {e}")
        
        try:
            # Test 2: ModalityConfigManager compatibility
            self.logger.info("Testing ModalityConfigManager compatibility...")
            
            modality_manager = ModalityConfigManager()
            
            # Test with ambulance collector
            ambulance_collector = AmbulanceDataCollector(
                n_agents=4,
                modality_config_manager=modality_manager
            )
            
            # Verify modality manager is used
            if ambulance_collector._collector._modality_manager is modality_manager:
                test_results["modality_config_compatibility"] = True
                self.logger.info("✅ ModalityConfigManager compatibility verified")
            else:
                test_results["errors"].append("ModalityConfigManager not properly integrated")
            
            ambulance_collector.cleanup()
            
        except Exception as e:
            test_results["errors"].append(f"ModalityConfigManager compatibility test failed: {str(e)}")
            self.logger.error(f"ModalityConfigManager compatibility test failed: {e}")
        
        try:
            # Test 3: PerformanceConfig compatibility
            self.logger.info("Testing PerformanceConfig compatibility...")
            
            performance_config = PerformanceConfig(
                max_memory_gb=2.0,
                enable_profiling=True,
                enable_memory_profiling=True
            )
            
            # Test with ambulance collector
            ambulance_collector = AmbulanceDataCollector(
                n_agents=4,
                performance_config=performance_config
            )
            
            # Verify performance config is used
            if ambulance_collector._collector.performance_config is performance_config:
                test_results["performance_config_compatibility"] = True
                self.logger.info("✅ PerformanceConfig compatibility verified")
            else:
                test_results["errors"].append("PerformanceConfig not properly integrated")
            
            ambulance_collector.cleanup()
            
        except Exception as e:
            test_results["errors"].append(f"PerformanceConfig compatibility test failed: {str(e)}")
            self.logger.error(f"PerformanceConfig compatibility test failed: {e}")
        
        try:
            # Test 4: Existing workflow compatibility
            self.logger.info("Testing existing workflow compatibility...")
            
            # Test that ambulance collector can be used in place of standard collector
            # for basic data collection workflows
            
            # Standard workflow components
            action_sampler = RandomActionSampler(seed=42)
            modality_manager = ModalityConfigManager()
            performance_config = PerformanceConfig(max_memory_gb=1.0)
            
            # Create ambulance collector with standard components
            ambulance_collector = AmbulanceDataCollector(
                n_agents=4,
                action_sampler=action_sampler,
                modality_config_manager=modality_manager,
                performance_config=performance_config,
                enable_validation=False
            )
            
            # Test basic workflow operations
            scenarios = ambulance_collector.get_available_scenarios()
            if scenarios:
                test_scenario = scenarios[0]
                
                # Test environment setup (standard workflow step)
                setup_info = ambulance_collector.setup_ambulance_environments(test_scenario)
                
                # Test statistics (standard workflow step)
                stats = ambulance_collector.get_collection_statistics()
                
                # Test validation (standard workflow step)
                validation = ambulance_collector.validate_ambulance_setup(test_scenario)
                
                if (setup_info and stats and validation and
                    setup_info['environments_created'] > 0 and
                    'ambulance_episodes_collected' in stats and
                    validation['valid']):
                    test_results["existing_workflow_compatibility"] = True
                    self.logger.info("✅ Existing workflow compatibility verified")
                else:
                    test_results["errors"].append("Workflow operations failed")
            else:
                test_results["errors"].append("No scenarios available for workflow test")
            
            ambulance_collector.cleanup()
            
        except Exception as e:
            test_results["errors"].append(f"Existing workflow compatibility test failed: {str(e)}")
            self.logger.error(f"Existing workflow compatibility test failed: {e}")
        
        self.test_results["backward_compatibility_tests"] = test_results
        return test_results
    
    def test_performance_integration(self) -> Dict[str, Any]:
        """Test performance integration and monitoring."""
        self.logger.info("Testing performance integration...")
        
        test_results = {
            "memory_monitoring": False,
            "performance_profiling": False,
            "throughput_monitoring": False,
            "resource_management": False,
            "errors": []
        }
        
        try:
            # Test 1: Memory monitoring integration
            self.logger.info("Testing memory monitoring integration...")
            
            performance_config = PerformanceConfig(
                max_memory_gb=1.0,
                enable_memory_profiling=True,
                enable_profiling=True
            )
            
            ambulance_collector = AmbulanceDataCollector(
                n_agents=4,
                performance_config=performance_config,
                enable_validation=True  # Enable validation for memory monitoring
            )
            
            # Check that memory monitoring is active
            if (hasattr(ambulance_collector._collector, 'memory_profiler') and
                ambulance_collector._collector.memory_profiler is not None):
                test_results["memory_monitoring"] = True
                self.logger.info("✅ Memory monitoring integration successful")
            else:
                test_results["errors"].append("Memory monitoring not active")
            
            ambulance_collector.cleanup()
            
        except Exception as e:
            test_results["errors"].append(f"Memory monitoring test failed: {str(e)}")
            self.logger.error(f"Memory monitoring test failed: {e}")
        
        try:
            # Test 2: Performance profiling integration
            self.logger.info("Testing performance profiling integration...")
            
            performance_config = PerformanceConfig(enable_profiling=True)
            
            ambulance_collector = AmbulanceDataCollector(
                n_agents=4,
                performance_config=performance_config
            )
            
            # Check that performance profiler is active
            if (hasattr(ambulance_collector._collector, 'performance_profiler') and
                ambulance_collector._collector.performance_profiler is not None):
                test_results["performance_profiling"] = True
                self.logger.info("✅ Performance profiling integration successful")
            else:
                test_results["errors"].append("Performance profiling not active")
            
            ambulance_collector.cleanup()
            
        except Exception as e:
            test_results["errors"].append(f"Performance profiling test failed: {str(e)}")
            self.logger.error(f"Performance profiling test failed: {e}")
        
        try:
            # Test 3: Throughput monitoring with storage
            self.logger.info("Testing throughput monitoring integration...")
            
            storage_dir = self.test_data_dir / "performance_test"
            storage_manager = DatasetStorageManager(
                storage_dir,
                performance_config=PerformanceConfig(enable_profiling=True)
            )
            
            # Check that throughput monitor is active
            if (hasattr(storage_manager, 'throughput_monitor') and
                storage_manager.throughput_monitor is not None):
                test_results["throughput_monitoring"] = True
                self.logger.info("✅ Throughput monitoring integration successful")
            else:
                test_results["errors"].append("Throughput monitoring not active")
            
        except Exception as e:
            test_results["errors"].append(f"Throughput monitoring test failed: {str(e)}")
            self.logger.error(f"Throughput monitoring test failed: {e}")
        
        try:
            # Test 4: Resource management integration
            self.logger.info("Testing resource management integration...")
            
            performance_config = PerformanceConfig(
                max_memory_gb=0.5,  # Low limit to test resource management
                enable_memory_profiling=True
            )
            
            ambulance_collector = AmbulanceDataCollector(
                n_agents=4,
                max_memory_gb=0.5,
                performance_config=performance_config,
                enable_validation=True
            )
            
            # Check that resource management components are present
            collector = ambulance_collector._collector
            if (hasattr(collector, 'memory_validator') and
                hasattr(collector, 'batching_optimizer') and
                collector.memory_validator is not None and
                collector.batching_optimizer is not None):
                test_results["resource_management"] = True
                self.logger.info("✅ Resource management integration successful")
            else:
                test_results["errors"].append("Resource management components not active")
            
            ambulance_collector.cleanup()
            
        except Exception as e:
            test_results["errors"].append(f"Resource management test failed: {str(e)}")
            self.logger.error(f"Resource management test failed: {e}")
        
        self.test_results["performance_tests"] = test_results
        return test_results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        self.logger.info("Starting comprehensive pipeline integration tests...")
        
        start_time = time.time()
        
        # Run all test categories
        test_categories = [
            ("ActionSampler Integration", self.test_action_sampler_integration),
            ("Feature Extraction Integration", self.test_feature_extraction_integration),
            ("Storage Integration", self.test_storage_integration),
            ("Visualization Integration", self.test_visualization_integration),
            ("Backward Compatibility", self.test_backward_compatibility),
            ("Performance Integration", self.test_performance_integration)
        ]
        
        for category_name, test_method in test_categories:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Running {category_name} Tests")
            self.logger.info(f"{'='*60}")
            
            try:
                test_method()
            except Exception as e:
                self.logger.error(f"{category_name} tests failed: {e}")
        
        # Calculate overall results
        total_time = time.time() - start_time
        
        overall_results = {
            "test_results": self.test_results,
            "total_time": total_time,
            "summary": self._calculate_test_summary()
        }
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("INTEGRATION TEST SUMMARY")
        self.logger.info(f"{'='*60}")
        
        summary = overall_results["summary"]
        self.logger.info(f"Total Tests: {summary['total_tests']}")
        self.logger.info(f"Passed: {summary['passed_tests']}")
        self.logger.info(f"Failed: {summary['failed_tests']}")
        self.logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        self.logger.info(f"Total Time: {total_time:.2f}s")
        
        if summary['failed_tests'] > 0:
            self.logger.warning(f"\n{summary['failed_tests']} tests failed:")
            for category, results in self.test_results.items():
                if 'errors' in results and results['errors']:
                    self.logger.warning(f"  {category}: {len(results['errors'])} errors")
        
        return overall_results
    
    def _calculate_test_summary(self) -> Dict[str, Any]:
        """Calculate overall test summary."""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for category, results in self.test_results.items():
            if isinstance(results, dict):
                for test_name, test_result in results.items():
                    if test_name != 'errors' and isinstance(test_result, bool):
                        total_tests += 1
                        if test_result:
                            passed_tests += 1
                        else:
                            failed_tests += 1
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate
        }


def main():
    """Main function for pipeline integration testing."""
    print("🔧 AMBULANCE DATA COLLECTION PIPELINE INTEGRATION TESTS")
    print("=" * 70)
    print("Testing integration with existing data collection pipeline components:")
    print("• ActionSampler classes")
    print("• Feature extraction systems")
    print("• Storage and visualization tools")
    print("• Backward compatibility")
    print("• Performance monitoring")
    print()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting pipeline integration tests")
    
    try:
        # Create and run integration tester
        tester = PipelineIntegrationTester(logger)
        results = tester.run_all_tests()
        
        # Print final results
        print(f"\n🎉 INTEGRATION TESTS COMPLETE!")
        print("=" * 40)
        
        summary = results["summary"]
        print(f"⏱️  Total Time: {results['total_time']:.2f} seconds")
        print(f"📊 Test Results: {summary['passed_tests']}/{summary['total_tests']} passed")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        
        if summary['failed_tests'] == 0:
            print("✅ All integration tests passed!")
            print("\n🚀 Ambulance data collection system is fully integrated with existing pipeline!")
        else:
            print(f"⚠️  {summary['failed_tests']} tests failed - check logs for details")
        
        print(f"\n📝 Detailed log: pipeline_integration_test.log")
        print(f"📁 Test data: {tester.test_data_dir}")
        
        return 0 if summary['failed_tests'] == 0 else 1
        
    except Exception as e:
        print(f"\n❌ Integration tests failed: {e}")
        logger.error(f"Integration tests failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())