"""
Specialized data collector for ambulance scenarios.

This module provides the AmbulanceDataCollector class that extends the existing
SynchronizedCollector to support ambulance-specific data collection with proper
agent configuration and multi-modal observation support.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import time

from highway_datacollection.collection.collector import SynchronizedCollector
from highway_datacollection.collection.types import CollectionResult, EpisodeData
from highway_datacollection.collection.action_samplers import ActionSampler, RandomActionSampler
from highway_datacollection.collection.modality_config import ModalityConfigManager
from highway_datacollection.storage.manager import DatasetStorageManager
from highway_datacollection.environments.factory import MultiAgentEnvFactory
from highway_datacollection.performance import PerformanceConfig

from ..scenarios.ambulance_scenarios import get_all_ambulance_scenarios, get_scenario_names, validate_ambulance_scenario


logger = logging.getLogger(__name__)


class AmbulanceDataCollector:
    """
    Specialized collector for ambulance scenario data collection.
    
    This class extends the functionality of SynchronizedCollector to support
    ambulance-specific scenarios with proper agent configuration where the first
    controlled agent represents an ambulance ego vehicle.
    
    Features:
    - Multi-modal data collection (Kinematics, OccupancyGrid, GrayscaleObservation)
    - Ambulance ego vehicle configuration
    - Integration with existing storage and validation systems
    - Support for all 15 ambulance highway scenarios
    """
    
    def __init__(self, 
                 n_agents: int = 4,
                 action_sampler: Optional[ActionSampler] = None,
                 modality_config_manager: Optional[ModalityConfigManager] = None,
                 max_memory_gb: float = 8.0,
                 enable_validation: bool = True,
                 performance_config: Optional[PerformanceConfig] = None):
        """
        Initialize the ambulance data collector.
        
        Args:
            n_agents: Number of controlled agents (first agent will be ambulance)
            action_sampler: Action sampling strategy (defaults to RandomActionSampler)
            modality_config_manager: Manager for modality configurations
            max_memory_gb: Maximum memory usage allowed in GB
            enable_validation: Whether to enable validation checks
            performance_config: Performance optimization configuration
        """
        self.n_agents = n_agents
        
        # Initialize the underlying synchronized collector
        self._collector = SynchronizedCollector(
            n_agents=n_agents,
            action_sampler=action_sampler,
            modality_config_manager=modality_config_manager,
            max_memory_gb=max_memory_gb,
            enable_validation=enable_validation,
            performance_config=performance_config
        )
        
        # Override the environment factory to support ambulance environments
        self._env_factory = MultiAgentEnvFactory()
        self._collector._env_factory = self._env_factory
        
        # Get available ambulance scenarios
        self._ambulance_scenarios = get_all_ambulance_scenarios()
        self._scenario_names = get_scenario_names()
        
        # Collection statistics specific to ambulance scenarios
        self.ambulance_stats = {
            "ambulance_episodes_collected": 0,
            "ambulance_scenarios_processed": 0,
            "ambulance_validation_errors": 0,
            "ambulance_setup_failures": 0
        }
        
        logger.info(f"Initialized AmbulanceDataCollector with {n_agents} agents")
        logger.info(f"Available ambulance scenarios: {len(self._scenario_names)}")
        logger.info(f"Action sampler: {type(self._collector._action_sampler).__name__}")
    
    def setup_ambulance_environments(self, scenario_name: str) -> Dict[str, Any]:
        """
        Set up ambulance environments with proper agent configuration.
        
        Args:
            scenario_name: Name of the ambulance scenario to set up
            
        Returns:
            Dictionary containing environment setup information
            
        Raises:
            ValueError: If scenario is not a valid ambulance scenario
            RuntimeError: If environment setup fails
        """
        if scenario_name not in self._scenario_names:
            raise ValueError(f"Unknown ambulance scenario: {scenario_name}. "
                           f"Available scenarios: {self._scenario_names}")
        
        # Validate ambulance scenario configuration
        scenario_config = self._ambulance_scenarios[scenario_name]
        if not validate_ambulance_scenario(scenario_config):
            self.ambulance_stats["ambulance_validation_errors"] += 1
            raise ValueError(f"Invalid ambulance scenario configuration: {scenario_name}")
        
        logger.info(f"Setting up ambulance environments for scenario: {scenario_name}")
        
        try:
            # Override the environment setup method to use ambulance environments
            self._setup_ambulance_environments(scenario_name)
            
            setup_info = {
                "scenario_name": scenario_name,
                "n_agents": self.n_agents,
                "ambulance_agent_index": scenario_config.get("_ambulance_config", {}).get("ambulance_agent_index", 0),
                "emergency_priority": scenario_config.get("_ambulance_config", {}).get("emergency_priority", "high"),
                "environments_created": len(self._collector._environments),
                "supported_modalities": list(self._collector._environments.keys()) if self._collector._environments else []
            }
            
            logger.info(f"Successfully set up ambulance environments: {setup_info}")
            return setup_info
            
        except Exception as e:
            self.ambulance_stats["ambulance_setup_failures"] += 1
            logger.error(f"Failed to set up ambulance environments for {scenario_name}: {e}")
            raise RuntimeError(f"Ambulance environment setup failed: {str(e)}")
    
    def _setup_ambulance_environments(self, scenario_name: str) -> None:
        """
        Internal method to set up parallel ambulance environments.
        
        Args:
            scenario_name: Name of the ambulance scenario to set up
        """
        if self._collector._current_scenario == scenario_name and self._collector._environments:
            # Environments already set up for this scenario
            return
        
        # Clean up existing environments
        self._collector._cleanup_environments()
        
        # Get enabled modalities for this scenario
        enabled_modalities = self._collector._modality_manager.get_enabled_modalities(scenario_name)
        
        # Create ambulance environments using the factory
        logger.info(f"Creating ambulance environments for scenario: {scenario_name}")
        logger.info(f"Enabled modalities: {enabled_modalities}")
        
        try:
            # Use the ambulance-specific environment creation method
            self._collector._environments = self._env_factory.create_parallel_ambulance_envs(
                scenario_name, self.n_agents, enabled_modalities
            )
            self._collector._current_scenario = scenario_name
            logger.info(f"Successfully created {len(self._collector._environments)} parallel ambulance environments")
            
        except Exception as e:
            logger.error(f"Failed to create ambulance environments for scenario {scenario_name}: {e}")
            raise
    
    def collect_ambulance_data(self, 
                             scenarios: Optional[List[str]] = None,
                             episodes_per_scenario: int = 100,
                             max_steps_per_episode: int = 100,
                             base_seed: int = 42,
                             batch_size: int = 10) -> Dict[str, CollectionResult]:
        """
        Collect data from ambulance scenarios with multi-modal observations.
        
        Args:
            scenarios: List of ambulance scenario names to collect (None for all)
            episodes_per_scenario: Number of episodes to collect per scenario
            max_steps_per_episode: Maximum steps per episode
            base_seed: Base random seed for reproducibility
            batch_size: Number of episodes to process in each batch
            
        Returns:
            Dictionary mapping scenario names to collection results
            
        Raises:
            ValueError: If invalid scenarios are specified
            RuntimeError: If data collection fails
        """
        # Use all ambulance scenarios if none specified
        if scenarios is None:
            scenarios = self._scenario_names.copy()
        
        # Validate specified scenarios
        invalid_scenarios = [s for s in scenarios if s not in self._scenario_names]
        if invalid_scenarios:
            raise ValueError(f"Invalid ambulance scenarios: {invalid_scenarios}. "
                           f"Available scenarios: {self._scenario_names}")
        
        logger.info(f"Starting ambulance data collection for {len(scenarios)} scenarios")
        logger.info(f"Configuration: {episodes_per_scenario} episodes/scenario, "
                   f"{max_steps_per_episode} max steps/episode, "
                   f"base seed {base_seed}, batch size {batch_size}")
        
        collection_results = {}
        start_time = time.time()
        
        for scenario_idx, scenario_name in enumerate(scenarios):
            logger.info(f"Processing ambulance scenario {scenario_idx + 1}/{len(scenarios)}: {scenario_name}")
            scenario_start_time = time.time()
            
            try:
                # Set up ambulance environments for this scenario
                setup_info = self.setup_ambulance_environments(scenario_name)
                logger.debug(f"Environment setup info: {setup_info}")
                
                # Collect data using the underlying synchronized collector
                scenario_seed = base_seed + scenario_idx * 10000  # Ensure unique seeds per scenario
                result = self._collector.collect_episode_batch(
                    scenario_name=scenario_name,
                    episodes=episodes_per_scenario,
                    seed=scenario_seed,
                    max_steps=max_steps_per_episode,
                    batch_size=batch_size
                )
                
                collection_results[scenario_name] = result
                self.ambulance_stats["ambulance_episodes_collected"] += result.successful_episodes
                self.ambulance_stats["ambulance_scenarios_processed"] += 1
                
                scenario_time = time.time() - scenario_start_time
                logger.info(f"Completed ambulance scenario {scenario_name} in {scenario_time:.2f}s: "
                           f"{result.successful_episodes}/{result.total_episodes} episodes successful")
                
            except Exception as e:
                error_msg = f"Ambulance scenario {scenario_name} failed: {str(e)}"
                logger.error(error_msg)
                
                # Create empty result for failed scenario
                collection_results[scenario_name] = CollectionResult(
                    episodes=[],
                    total_episodes=episodes_per_scenario,
                    successful_episodes=0,
                    failed_episodes=episodes_per_scenario,
                    collection_time=0.0,
                    errors=[error_msg]
                )
        
        total_time = time.time() - start_time
        successful_scenarios = sum(1 for r in collection_results.values() if r.successful_episodes > 0)
        total_episodes = sum(r.successful_episodes for r in collection_results.values())
        
        logger.info(f"Ambulance data collection completed in {total_time:.2f}s")
        logger.info(f"Scenarios: {successful_scenarios}/{len(scenarios)} successful")
        logger.info(f"Episodes: {total_episodes} total successful episodes")
        
        return collection_results
    
    def collect_single_ambulance_scenario(self,
                                        scenario_name: str,
                                        episodes: int = 100,
                                        max_steps: int = 100,
                                        seed: int = 42,
                                        batch_size: int = 10) -> CollectionResult:
        """
        Collect data from a single ambulance scenario.
        
        Args:
            scenario_name: Name of the ambulance scenario to collect
            episodes: Number of episodes to collect
            max_steps: Maximum steps per episode
            seed: Random seed for reproducibility
            batch_size: Number of episodes to process in each batch
            
        Returns:
            CollectionResult for the scenario
            
        Raises:
            ValueError: If scenario_name is not a valid ambulance scenario
            RuntimeError: If data collection fails
        """
        if scenario_name not in self._scenario_names:
            raise ValueError(f"Unknown ambulance scenario: {scenario_name}. "
                           f"Available scenarios: {self._scenario_names}")
        
        logger.info(f"Collecting data from ambulance scenario: {scenario_name}")
        
        # Set up ambulance environments
        setup_info = self.setup_ambulance_environments(scenario_name)
        logger.debug(f"Environment setup info: {setup_info}")
        
        # Collect data using the underlying synchronized collector
        result = self._collector.collect_episode_batch(
            scenario_name=scenario_name,
            episodes=episodes,
            seed=seed,
            max_steps=max_steps,
            batch_size=batch_size
        )
        
        # Update statistics
        self.ambulance_stats["ambulance_episodes_collected"] += result.successful_episodes
        if result.successful_episodes > 0:
            self.ambulance_stats["ambulance_scenarios_processed"] += 1
        
        logger.info(f"Collected {result.successful_episodes}/{result.total_episodes} episodes "
                   f"from ambulance scenario {scenario_name}")
        
        return result
    
    def store_ambulance_data(self, 
                           collection_results: Dict[str, CollectionResult],
                           output_dir: Path) -> Dict[str, Any]:
        """
        Store collected ambulance data using the DatasetStorageManager.
        
        Args:
            collection_results: Dictionary of collection results by scenario
            output_dir: Output directory for storing data
            
        Returns:
            Dictionary with storage information and paths
            
        Raises:
            RuntimeError: If data storage fails
        """
        logger.info(f"Storing ambulance data to: {output_dir}")
        
        # Initialize storage manager
        storage_manager = DatasetStorageManager(output_dir)
        
        storage_info = {
            "output_dir": str(output_dir),
            "scenarios_stored": 0,
            "total_episodes_stored": 0,
            "storage_paths": [],
            "errors": []
        }
        
        for scenario_name, result in collection_results.items():
            if not result.episodes:
                logger.warning(f"No episodes to store for ambulance scenario {scenario_name}")
                continue
            
            try:
                logger.info(f"Storing {len(result.episodes)} episodes for ambulance scenario {scenario_name}")
                
                # Convert episode data to storage format
                all_observations = []
                all_metadata = []
                
                for episode in result.episodes:
                    # Add episode metadata
                    all_metadata.append(episode.metadata)
                    
                    # Process episode observations
                    for step_idx, step_observations in enumerate(episode.observations):
                        for obs in step_observations:
                            # Add episode-level information to each observation
                            obs_record = obs.copy()
                            obs_record.update({
                                'episode_id': episode.episode_id,
                                'scenario': episode.scenario,
                                'step': step_idx,
                                'action': episode.actions[step_idx] if step_idx < len(episode.actions) else None,
                                'reward': episode.rewards[step_idx] if step_idx < len(episode.rewards) else 0.0,
                                'done': episode.dones[step_idx] if step_idx < len(episode.dones) else False,
                                'ambulance_scenario': True,  # Mark as ambulance data
                                'ambulance_agent_index': 0   # First agent is ambulance
                            })
                            all_observations.append(obs_record)
                
                # Store data
                storage_paths = storage_manager.write_episode_batch(
                    data=all_observations,
                    metadata=all_metadata,
                    scenario=scenario_name
                )
                
                storage_info["storage_paths"].append(storage_paths)
                storage_info["scenarios_stored"] += 1
                storage_info["total_episodes_stored"] += len(result.episodes)
                
                logger.info(f"Successfully stored ambulance data for scenario {scenario_name}")
                
            except Exception as e:
                error_msg = f"Failed to store ambulance data for scenario {scenario_name}: {str(e)}"
                logger.error(error_msg)
                storage_info["errors"].append(error_msg)
        
        # Create dataset index if we have stored data
        if storage_info["storage_paths"]:
            try:
                index_path = storage_manager.create_dataset_index(storage_info["storage_paths"])
                storage_info["dataset_index_path"] = str(index_path)
                logger.info(f"Created ambulance dataset index: {index_path}")
            except Exception as e:
                error_msg = f"Failed to create ambulance dataset index: {str(e)}"
                logger.error(error_msg)
                storage_info["errors"].append(error_msg)
        
        logger.info(f"Ambulance data storage completed: {storage_info['scenarios_stored']} scenarios, "
                   f"{storage_info['total_episodes_stored']} episodes stored")
        
        return storage_info
    
    def get_available_scenarios(self) -> List[str]:
        """
        Get list of available ambulance scenarios.
        
        Returns:
            List of ambulance scenario names
        """
        return self._scenario_names.copy()
    
    def get_scenario_info(self, scenario_name: str) -> Dict[str, Any]:
        """
        Get information about a specific ambulance scenario.
        
        Args:
            scenario_name: Name of the ambulance scenario
            
        Returns:
            Dictionary with scenario information
            
        Raises:
            ValueError: If scenario_name is not found
        """
        if scenario_name not in self._scenario_names:
            raise ValueError(f"Unknown ambulance scenario: {scenario_name}. "
                           f"Available scenarios: {self._scenario_names}")
        
        scenario_config = self._ambulance_scenarios[scenario_name]
        
        return {
            "scenario_name": scenario_name,
            "description": scenario_config.get("description", "No description available"),
            "traffic_density": scenario_config.get("traffic_density", "unknown"),
            "vehicles_count": scenario_config.get("vehicles_count", 0),
            "duration": scenario_config.get("duration", 0),
            "lanes_count": scenario_config.get("lanes_count", 4),
            "controlled_vehicles": scenario_config.get("controlled_vehicles", 4),
            "highway_conditions": scenario_config.get("highway_conditions", "normal"),
            "speed_limit": scenario_config.get("speed_limit", 30),
            "ambulance_config": scenario_config.get("_ambulance_config", {}),
            "supports_multi_modal": True,
            "supported_observations": ["Kinematics", "OccupancyGrid", "GrayscaleObservation"]
        }
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """
        Get collection statistics including both general and ambulance-specific metrics.
        
        Returns:
            Dictionary with collection statistics
        """
        # Get base collector statistics
        base_stats = self._collector.collection_stats.copy()
        
        # Combine with ambulance-specific statistics
        combined_stats = {
            **base_stats,
            **self.ambulance_stats,
            "available_ambulance_scenarios": len(self._scenario_names),
            "n_agents": self.n_agents,
            "ambulance_agent_index": 0,
            "supported_modalities": self._collector._obs_types
        }
        
        return combined_stats
    
    def reset_statistics(self) -> None:
        """Reset collection statistics."""
        self.ambulance_stats = {
            "ambulance_episodes_collected": 0,
            "ambulance_scenarios_processed": 0,
            "ambulance_validation_errors": 0,
            "ambulance_setup_failures": 0
        }
        # Reset base collector statistics if available
        if hasattr(self._collector, 'collection_stats'):
            self._collector.collection_stats = {
                "episodes_collected": 0,
                "steps_collected": 0,
                "sync_failures": 0,
                "memory_warnings": 0,
                "validation_errors": 0,
                "recovery_attempts": 0
            }
        logger.info("Ambulance collection statistics reset")
    
    def validate_ambulance_setup(self, scenario_name: str) -> Dict[str, Any]:
        """
        Validate ambulance scenario setup without creating environments.
        
        Args:
            scenario_name: Name of the ambulance scenario to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "scenario_name": scenario_name,
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        # Check if scenario exists
        if scenario_name not in self._scenario_names:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Unknown ambulance scenario: {scenario_name}")
            return validation_result
        
        # Validate scenario configuration
        scenario_config = self._ambulance_scenarios[scenario_name]
        if not validate_ambulance_scenario(scenario_config):
            validation_result["valid"] = False
            validation_result["errors"].append(f"Invalid ambulance scenario configuration")
        
        # Check environment factory validation
        for obs_type in ["Kinematics", "OccupancyGrid", "GrayscaleObservation"]:
            if not self._env_factory.validate_ambulance_configuration(scenario_name, obs_type, self.n_agents):
                validation_result["warnings"].append(f"Potential issue with {obs_type} observation type")
        
        # Add scenario information
        validation_result["info"] = self.get_scenario_info(scenario_name)
        
        return validation_result
    
    def cleanup(self) -> None:
        """Clean up resources and environments."""
        logger.info("Cleaning up ambulance data collector")
        self._collector._cleanup_environments()
        logger.info("Ambulance data collector cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()