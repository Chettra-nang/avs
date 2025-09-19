"""
Main collection orchestrator for processing all curriculum scenarios.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from dataclasses import dataclass

from ..scenarios.registry import ScenarioRegistry
from ..storage.manager import DatasetStorageManager
from .collector import SynchronizedCollector
from .types import CollectionResult, EpisodeData
from .action_samplers import ActionSampler
from .modality_config import ModalityConfigManager


logger = logging.getLogger(__name__)


@dataclass
class CollectionProgress:
    """Progress tracking for collection operations."""
    current_scenario: str
    scenario_index: int
    total_scenarios: int
    current_episode: int
    total_episodes: int
    successful_episodes: int
    failed_episodes: int
    start_time: float
    scenario_start_time: float
    errors: List[str]


@dataclass
class FullCollectionResult:
    """Result of a full collection operation across all scenarios."""
    total_scenarios: int
    successful_scenarios: int
    failed_scenarios: int
    total_episodes: int
    successful_episodes: int
    failed_episodes: int
    collection_time: float
    scenario_results: Dict[str, CollectionResult]
    storage_paths: List[Any]
    dataset_index_path: Optional[Path]
    errors: List[str]


class CollectionOrchestrator:
    """
    Main orchestrator for running complete dataset collection across all curriculum scenarios.
    
    Handles batch processing, progress tracking, error handling, and storage management
    for long-running collection operations.
    """
    
    def __init__(self, base_storage_path: Path, n_agents: int = 2, 
                 action_sampler: Optional[ActionSampler] = None,
                 modality_config_manager: Optional[ModalityConfigManager] = None):
        """
        Initialize the collection orchestrator.
        
        Args:
            base_storage_path: Base directory for dataset storage
            n_agents: Number of controlled agents per scenario
            action_sampler: Action sampling strategy (defaults to RandomActionSampler)
            modality_config_manager: Manager for modality configurations
        """
        self.base_storage_path = Path(base_storage_path)
        self.n_agents = n_agents
        
        # Initialize components
        self.scenario_registry = ScenarioRegistry()
        self.storage_manager = DatasetStorageManager(self.base_storage_path)
        self.collector = SynchronizedCollector(
            n_agents=n_agents, 
            action_sampler=action_sampler,
            modality_config_manager=modality_config_manager
        )
        
        # Progress tracking
        self._progress_callback: Optional[Callable[[CollectionProgress], None]] = None
        self._should_stop = False
        
        logger.info(f"Initialized CollectionOrchestrator with {n_agents} agents")
        logger.info(f"Action sampler: {type(self.collector.get_action_sampler()).__name__}")
        logger.info(f"Storage path: {self.base_storage_path}")
    
    def set_progress_callback(self, callback: Callable[[CollectionProgress], None]) -> None:
        """
        Set callback function for progress updates.
        
        Args:
            callback: Function to call with progress updates
        """
        self._progress_callback = callback
    
    def stop_collection(self) -> None:
        """Request collection to stop gracefully."""
        self._should_stop = True
        logger.info("Collection stop requested")
    
    def set_action_sampler(self, action_sampler: ActionSampler) -> None:
        """
        Set a new action sampling strategy.
        
        Args:
            action_sampler: New action sampling strategy
        """
        self.collector.set_action_sampler(action_sampler)
        logger.info(f"Updated orchestrator action sampler to: {type(action_sampler).__name__}")
    
    def get_action_sampler(self) -> ActionSampler:
        """
        Get the current action sampling strategy.
        
        Returns:
            Current action sampler
        """
        return self.collector.get_action_sampler()
    
    def set_modality_config_manager(self, manager: ModalityConfigManager) -> None:
        """
        Set a new modality configuration manager.
        
        Args:
            manager: New modality configuration manager
        """
        self.collector.set_modality_config_manager(manager)
        logger.info("Updated orchestrator modality configuration manager")
    
    def get_modality_config_manager(self) -> ModalityConfigManager:
        """
        Get the current modality configuration manager.
        
        Returns:
            Current modality configuration manager
        """
        return self.collector.get_modality_config_manager()
    
    def run_full_collection(
        self,
        episodes_per_scenario: int = 100,
        max_steps_per_episode: int = 100,
        base_seed: int = 42,
        scenarios: Optional[List[str]] = None,
        batch_size: int = 10
    ) -> FullCollectionResult:
        """
        Run complete data collection across all curriculum scenarios.
        
        Args:
            episodes_per_scenario: Number of episodes to collect per scenario
            max_steps_per_episode: Maximum steps per episode
            base_seed: Base random seed for reproducibility
            scenarios: List of specific scenarios to run (None for all)
            batch_size: Number of episodes to process in each batch
            
        Returns:
            FullCollectionResult containing comprehensive collection results
        """
        logger.info("Starting full collection across all curriculum scenarios")
        logger.info(f"Configuration: {episodes_per_scenario} episodes/scenario, "
                   f"{max_steps_per_episode} max steps/episode, "
                   f"base seed {base_seed}, batch size {batch_size}")
        
        start_time = time.time()
        self._should_stop = False
        
        # Get scenarios to process
        if scenarios is None:
            scenarios = self.scenario_registry.list_scenarios()
        
        logger.info(f"Processing {len(scenarios)} scenarios: {scenarios}")
        
        # Initialize result tracking
        scenario_results = {}
        storage_paths = []
        total_successful_episodes = 0
        total_failed_episodes = 0
        successful_scenarios = 0
        failed_scenarios = 0
        collection_errors = []
        
        # Process each scenario
        for scenario_idx, scenario_name in enumerate(scenarios):
            if self._should_stop:
                logger.info("Collection stopped by user request")
                break
            
            logger.info(f"Processing scenario {scenario_idx + 1}/{len(scenarios)}: {scenario_name}")
            scenario_start_time = time.time()
            
            try:
                # Initialize progress tracking
                progress = CollectionProgress(
                    current_scenario=scenario_name,
                    scenario_index=scenario_idx,
                    total_scenarios=len(scenarios),
                    current_episode=0,
                    total_episodes=episodes_per_scenario,
                    successful_episodes=0,
                    failed_episodes=0,
                    start_time=start_time,
                    scenario_start_time=scenario_start_time,
                    errors=[]
                )
                
                # Collect data for this scenario
                scenario_result = self._collect_scenario_data(
                    scenario_name=scenario_name,
                    episodes=episodes_per_scenario,
                    max_steps=max_steps_per_episode,
                    base_seed=base_seed + scenario_idx * 10000,  # Ensure unique seeds per scenario
                    batch_size=batch_size,
                    progress=progress
                )
                
                scenario_results[scenario_name] = scenario_result
                
                # Store collected data
                if scenario_result.episodes:
                    storage_result = self._store_scenario_data(scenario_name, scenario_result)
                    storage_paths.append(storage_result)
                    logger.info(f"Stored {len(scenario_result.episodes)} episodes for {scenario_name}")
                
                # Update totals
                total_successful_episodes += scenario_result.successful_episodes
                total_failed_episodes += scenario_result.failed_episodes
                
                if scenario_result.successful_episodes > 0:
                    successful_scenarios += 1
                else:
                    failed_scenarios += 1
                    collection_errors.extend(scenario_result.errors)
                
                scenario_time = time.time() - scenario_start_time
                logger.info(f"Completed scenario {scenario_name} in {scenario_time:.2f}s: "
                           f"{scenario_result.successful_episodes}/{scenario_result.total_episodes} episodes successful")
                
            except Exception as e:
                error_msg = f"Scenario {scenario_name} failed completely: {str(e)}"
                logger.error(error_msg)
                collection_errors.append(error_msg)
                failed_scenarios += 1
                
                # Create empty result for failed scenario
                scenario_results[scenario_name] = CollectionResult(
                    episodes=[],
                    total_episodes=episodes_per_scenario,
                    successful_episodes=0,
                    failed_episodes=episodes_per_scenario,
                    collection_time=0.0,
                    errors=[error_msg]
                )
        
        # Create dataset index
        dataset_index_path = None
        if storage_paths:
            try:
                dataset_index_path = self.storage_manager.create_dataset_index(storage_paths)
                logger.info(f"Created dataset index: {dataset_index_path}")
            except Exception as e:
                error_msg = f"Failed to create dataset index: {str(e)}"
                logger.error(error_msg)
                collection_errors.append(error_msg)
        
        total_time = time.time() - start_time
        
        # Create final result
        result = FullCollectionResult(
            total_scenarios=len(scenarios),
            successful_scenarios=successful_scenarios,
            failed_scenarios=failed_scenarios,
            total_episodes=len(scenarios) * episodes_per_scenario,
            successful_episodes=total_successful_episodes,
            failed_episodes=total_failed_episodes,
            collection_time=total_time,
            scenario_results=scenario_results,
            storage_paths=storage_paths,
            dataset_index_path=dataset_index_path,
            errors=collection_errors
        )
        
        # Log final summary
        logger.info("Full collection completed!")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Scenarios: {successful_scenarios}/{len(scenarios)} successful")
        logger.info(f"Episodes: {total_successful_episodes}/{len(scenarios) * episodes_per_scenario} successful")
        
        if collection_errors:
            logger.warning(f"Collection completed with {len(collection_errors)} errors")
        
        return result
    
    def _collect_scenario_data(
        self,
        scenario_name: str,
        episodes: int,
        max_steps: int,
        base_seed: int,
        batch_size: int,
        progress: CollectionProgress
    ) -> CollectionResult:
        """
        Collect data for a single scenario with batch processing.
        
        Args:
            scenario_name: Name of the scenario to collect
            episodes: Total number of episodes to collect
            max_steps: Maximum steps per episode
            base_seed: Base seed for this scenario
            batch_size: Number of episodes per batch
            progress: Progress tracking object
            
        Returns:
            CollectionResult for this scenario
        """
        logger.info(f"Collecting {episodes} episodes for scenario '{scenario_name}' in batches of {batch_size}")
        
        all_episodes = []
        all_errors = []
        total_successful = 0
        total_failed = 0
        scenario_start_time = time.time()
        
        # Process episodes in batches
        for batch_start in range(0, episodes, batch_size):
            if self._should_stop:
                logger.info(f"Stopping collection for scenario {scenario_name} due to user request")
                break
            
            batch_end = min(batch_start + batch_size, episodes)
            batch_episodes = batch_end - batch_start
            batch_seed = base_seed + batch_start
            
            logger.debug(f"Processing batch {batch_start//batch_size + 1}: "
                        f"episodes {batch_start + 1}-{batch_end}")
            
            # Update progress
            progress.current_episode = batch_start
            progress.successful_episodes = total_successful
            progress.failed_episodes = total_failed
            if self._progress_callback:
                self._progress_callback(progress)
            
            try:
                # Collect batch
                batch_result = self.collector.collect_episode_batch(
                    scenario_name=scenario_name,
                    episodes=batch_episodes,
                    seed=batch_seed,
                    max_steps=max_steps
                )
                
                # Accumulate results
                all_episodes.extend(batch_result.episodes)
                all_errors.extend(batch_result.errors)
                total_successful += batch_result.successful_episodes
                total_failed += batch_result.failed_episodes
                
                logger.debug(f"Batch completed: {batch_result.successful_episodes}/{batch_episodes} episodes successful")
                
            except Exception as e:
                error_msg = f"Batch {batch_start//batch_size + 1} failed: {str(e)}"
                logger.error(error_msg)
                all_errors.append(error_msg)
                total_failed += batch_episodes
        
        scenario_time = time.time() - scenario_start_time
        
        # Final progress update
        progress.current_episode = episodes
        progress.successful_episodes = total_successful
        progress.failed_episodes = total_failed
        if self._progress_callback:
            self._progress_callback(progress)
        
        return CollectionResult(
            episodes=all_episodes,
            total_episodes=episodes,
            successful_episodes=total_successful,
            failed_episodes=total_failed,
            collection_time=scenario_time,
            errors=all_errors
        )
    
    def _store_scenario_data(self, scenario_name: str, result: CollectionResult) -> Any:
        """
        Store collected data for a scenario.
        
        Args:
            scenario_name: Name of the scenario
            result: Collection result to store
            
        Returns:
            Storage paths object
        """
        if not result.episodes:
            logger.warning(f"No episodes to store for scenario {scenario_name}")
            return None
        
        logger.info(f"Storing {len(result.episodes)} episodes for scenario {scenario_name}")
        
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
                        'done': episode.dones[step_idx] if step_idx < len(episode.dones) else False
                    })
                    all_observations.append(obs_record)
        
        # Store data
        try:
            storage_paths = self.storage_manager.write_episode_batch(
                data=all_observations,
                metadata=all_metadata,
                scenario=scenario_name
            )
            logger.info(f"Successfully stored data for scenario {scenario_name}")
            return storage_paths
            
        except Exception as e:
            error_msg = f"Failed to store data for scenario {scenario_name}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """
        Get summary of current dataset state.
        
        Returns:
            Dictionary with dataset summary information
        """
        try:
            structure = self.storage_manager.organize_dataset_structure()
            validation = self.storage_manager.validate_dataset_integrity()
            
            return {
                'dataset_structure': structure,
                'validation': validation,
                'available_scenarios': self.scenario_registry.list_scenarios(),
                'storage_path': str(self.base_storage_path)
            }
        except Exception as e:
            logger.error(f"Failed to generate collection summary: {e}")
            return {
                'error': str(e),
                'storage_path': str(self.base_storage_path)
            }
    
    def cleanup_failed_collections(self) -> Dict[str, Any]:
        """
        Clean up any failed or incomplete collections.
        
        Returns:
            Dictionary with cleanup results
        """
        logger.info("Cleaning up failed collections")
        
        try:
            # Remove empty directories
            removed_dirs = self.storage_manager.cleanup_empty_directories()
            
            # Validate dataset integrity
            validation = self.storage_manager.validate_dataset_integrity()
            
            cleanup_result = {
                'removed_directories': removed_dirs,
                'validation': validation,
                'cleanup_successful': True
            }
            
            logger.info(f"Cleanup completed: removed {len(removed_dirs)} empty directories")
            return cleanup_result
            
        except Exception as e:
            error_msg = f"Cleanup failed: {str(e)}"
            logger.error(error_msg)
            return {
                'error': error_msg,
                'cleanup_successful': False
            }


def run_full_collection(
    base_storage_path: Path,
    episodes_per_scenario: int = 100,
    n_agents: int = 2,
    max_steps_per_episode: int = 100,
    base_seed: int = 42,
    scenarios: Optional[List[str]] = None,
    batch_size: int = 10,
    progress_callback: Optional[Callable[[CollectionProgress], None]] = None,
    action_sampler: Optional[ActionSampler] = None,
    modality_config_manager: Optional[ModalityConfigManager] = None
) -> FullCollectionResult:
    """
    Convenience function to run complete data collection across all curriculum scenarios.
    
    Args:
        base_storage_path: Base directory for dataset storage
        episodes_per_scenario: Number of episodes to collect per scenario
        n_agents: Number of controlled agents per scenario
        max_steps_per_episode: Maximum steps per episode
        base_seed: Base random seed for reproducibility
        scenarios: List of specific scenarios to run (None for all)
        batch_size: Number of episodes to process in each batch
        progress_callback: Optional callback for progress updates
        action_sampler: Action sampling strategy (defaults to RandomActionSampler)
        modality_config_manager: Manager for modality configurations
        
    Returns:
        FullCollectionResult containing comprehensive collection results
    """
    orchestrator = CollectionOrchestrator(base_storage_path, n_agents, action_sampler, modality_config_manager)
    
    if progress_callback:
        orchestrator.set_progress_callback(progress_callback)
    
    return orchestrator.run_full_collection(
        episodes_per_scenario=episodes_per_scenario,
        max_steps_per_episode=max_steps_per_episode,
        base_seed=base_seed,
        scenarios=scenarios,
        batch_size=batch_size
    )