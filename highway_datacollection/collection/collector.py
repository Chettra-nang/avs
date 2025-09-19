"""
Synchronized collector for multi-modal data collection.
"""

from typing import Dict, Any, Tuple, Optional, List
import logging
import numpy as np
import time
from dataclasses import dataclass

from ..environments.factory import MultiAgentEnvFactory
from .types import CollectionResult, EpisodeData
from .action_samplers import ActionSampler, RandomActionSampler
from .modality_config import ModalityConfigManager, ModalityConfig
from .validation import (
    EnvironmentSynchronizationValidator, MemoryValidator, DataIntegrityValidator,
    ValidationResult, ValidationSeverity
)
from .error_handling import (
    EnvironmentSynchronizationError, MemoryError, ValidationError,
    ErrorHandler, ErrorContext, ErrorSeverity, GracefulDegradationManager,
    with_error_handling
)
from ..performance import MemoryProfiler, PerformanceProfiler, BatchingOptimizer, PerformanceConfig


logger = logging.getLogger(__name__)


class SynchronizedCollector:
    """
    Orchestrates parallel environment execution and data synchronization.
    
    Manages multiple HighwayEnv instances with different observation modalities,
    ensuring perfect synchronization through identical seeds and actions.
    """
    
    def __init__(self, n_agents: int = 2, action_sampler: Optional[ActionSampler] = None,
                 modality_config_manager: Optional[ModalityConfigManager] = None,
                 max_memory_gb: float = 8.0, enable_validation: bool = True,
                 performance_config: Optional[PerformanceConfig] = None):
        """
        Initialize the synchronized collector.
        
        Args:
            n_agents: Number of controlled agents in each environment
            action_sampler: Action sampling strategy (defaults to RandomActionSampler)
            modality_config_manager: Manager for modality configurations
            max_memory_gb: Maximum memory usage allowed in GB
            enable_validation: Whether to enable validation checks
            performance_config: Performance optimization configuration
        """
        self._env_factory = MultiAgentEnvFactory()
        self._n_agents = n_agents
        self._environments: Dict[str, Any] = {}
        self._current_seed: Optional[int] = None
        self._current_scenario: Optional[str] = None
        
        # Action sampling strategy
        self._action_sampler = action_sampler or RandomActionSampler()
        
        # Modality configuration
        self._modality_manager = modality_config_manager or ModalityConfigManager()
        
        # Supported observation types
        self._obs_types = self._env_factory.get_supported_observation_types()
        
        # Performance monitoring and optimization
        self.performance_config = performance_config or PerformanceConfig(max_memory_gb=max_memory_gb)
        self.memory_profiler = MemoryProfiler() if self.performance_config.enable_memory_profiling else None
        self.performance_profiler = PerformanceProfiler(self.memory_profiler)
        self.batching_optimizer = BatchingOptimizer(self.performance_config, self.memory_profiler)
        
        # Start memory monitoring if enabled
        if self.memory_profiler and self.performance_config.enable_profiling:
            self.memory_profiler.start_monitoring(interval=2.0)
        
        # Error handling and validation
        self.error_handler = ErrorHandler()
        self.degradation_manager = GracefulDegradationManager()
        self.enable_validation = enable_validation
        
        if enable_validation:
            self.sync_validator = EnvironmentSynchronizationValidator()
            self.memory_validator = MemoryValidator(max_memory_gb)
            self.data_validator = DataIntegrityValidator()
        
        # Register recovery strategies
        self._register_recovery_strategies()
        
        # Collection statistics
        self.collection_stats = {
            "episodes_collected": 0,
            "steps_collected": 0,
            "sync_failures": 0,
            "memory_warnings": 0,
            "validation_errors": 0,
            "recovery_attempts": 0
        }
        
        logger.info(f"Initialized SynchronizedCollector with {n_agents} agents")
        logger.info(f"Action sampler: {type(self._action_sampler).__name__}")
        logger.info(f"Supported observation types: {self._obs_types}")
        logger.info(f"Validation enabled: {enable_validation}")
        logger.info(f"Performance monitoring enabled: {self.performance_config.enable_profiling}")
    
    def _register_recovery_strategies(self) -> None:
        """Register recovery strategies for common errors."""
        from .error_handling import RecoveryAction
        
        # Environment synchronization recovery
        def reset_environments_recovery():
            """Reset all environments with fresh seeds."""
            if self._environments and self._current_seed is not None:
                try:
                    self.reset_parallel_envs(self._current_seed + 1000)  # Use different seed
                    if self.enable_validation:
                        self.sync_validator.reset()
                    logger.info("Successfully reset environments for synchronization recovery")
                    return True
                except Exception as e:
                    logger.error(f"Environment reset recovery failed: {e}")
                    return False
            return False
        
        def recreate_environments_recovery():
            """Recreate environments from scratch."""
            if self._current_scenario:
                try:
                    self._cleanup_environments()
                    self._setup_environments(self._current_scenario)
                    if self._current_seed is not None:
                        self.reset_parallel_envs(self._current_seed)
                    logger.info("Successfully recreated environments")
                    return True
                except Exception as e:
                    logger.error(f"Environment recreation recovery failed: {e}")
                    return False
            return False
        
        def memory_cleanup_recovery():
            """Perform memory cleanup."""
            try:
                if self.enable_validation:
                    gc_result = self.memory_validator.trigger_garbage_collection()
                    logger.info(f"Memory cleanup: {gc_result}")
                    return gc_result["memory_freed_mb"] > 0
                return False
            except Exception as e:
                logger.error(f"Memory cleanup recovery failed: {e}")
                return False
        
        # Register recovery actions
        self.error_handler.register_recovery_strategy(
            EnvironmentSynchronizationError,
            RecoveryAction("reset_environments", "Reset environments with new seed", 
                         reset_environments_recovery, 0.8)
        )
        
        self.error_handler.register_recovery_strategy(
            EnvironmentSynchronizationError,
            RecoveryAction("recreate_environments", "Recreate environments from scratch",
                         recreate_environments_recovery, 0.9)
        )
        
        self.error_handler.register_recovery_strategy(
            MemoryError,
            RecoveryAction("memory_cleanup", "Trigger garbage collection",
                         memory_cleanup_recovery, 0.5)
        )
    
    def _setup_environments(self, scenario_name: str) -> None:
        """
        Set up parallel environments for enabled observation modalities.
        
        Args:
            scenario_name: Name of the scenario to set up environments for
        """
        if self._current_scenario == scenario_name and self._environments:
            # Environments already set up for this scenario
            return
        
        # Clean up existing environments
        self._cleanup_environments()
        
        # Get enabled modalities for this scenario
        enabled_modalities = self._modality_manager.get_enabled_modalities(scenario_name)
        
        # Create new environments
        logger.info(f"Setting up environments for scenario: {scenario_name}")
        logger.info(f"Enabled modalities: {enabled_modalities}")
        try:
            self._environments = self._env_factory.create_parallel_envs(
                scenario_name, self._n_agents, enabled_modalities
            )
            self._current_scenario = scenario_name
            logger.info(f"Successfully created {len(self._environments)} parallel environments")
        except Exception as e:
            logger.error(f"Failed to create environments for scenario {scenario_name}: {e}")
            raise
    
    def _cleanup_environments(self) -> None:
        """Clean up existing environments."""
        if self._environments:
            logger.debug("Cleaning up existing environments")
            for obs_type, env in self._environments.items():
                try:
                    env.close()
                except Exception as e:
                    logger.warning(f"Error closing environment {obs_type}: {e}")
            self._environments.clear()
            self._current_scenario = None
    
    def reset_parallel_envs(self, seed: int) -> Dict[str, Any]:
        """
        Reset all parallel environments with identical seed.
        
        Args:
            seed: Random seed for environment reset
            
        Returns:
            Dictionary containing initial observations from all modalities
            
        Raises:
            RuntimeError: If environments are not set up or reset fails
        """
        if not self._environments:
            raise RuntimeError("Environments not set up. Call _setup_environments first.")
        
        logger.debug(f"Resetting parallel environments with seed: {seed}")
        observations = {}
        
        try:
            for obs_type, env in self._environments.items():
                obs, info = env.reset(seed=seed)
                observations[obs_type] = {
                    'observation': obs,
                    'info': info
                }
                logger.debug(f"Reset {obs_type} environment successfully")
            
            self._current_seed = seed
            # Reset action sampler with seed for deterministic behavior
            self._action_sampler.reset(seed)
            logger.info(f"Successfully reset all {len(self._environments)} environments")
            return observations
            
        except Exception as e:
            logger.error(f"Failed to reset environments: {e}")
            raise RuntimeError(f"Environment reset failed: {e}")
    
    def step_parallel_envs(self, actions: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Step all parallel environments with the same actions.
        
        Args:
            actions: Tuple of actions for each agent
            
        Returns:
            Dictionary containing step results from all modalities
            
        Raises:
            RuntimeError: If environments are not set up or step fails
            ValueError: If actions don't match expected number of agents
        """
        if not self._environments:
            raise RuntimeError("Environments not set up. Call _setup_environments first.")
        
        if len(actions) != self._n_agents:
            raise ValueError(f"Expected {self._n_agents} actions, got {len(actions)}")
        
        logger.debug(f"Stepping parallel environments with actions: {actions}")
        step_results = {}
        
        try:
            for obs_type, env in self._environments.items():
                obs, reward, terminated, truncated, info = env.step(actions)
                step_results[obs_type] = {
                    'observation': obs,
                    'reward': reward,
                    'terminated': terminated,
                    'truncated': truncated,
                    'info': info
                }
                logger.debug(f"Stepped {obs_type} environment successfully")
            
            return step_results
            
        except Exception as e:
            logger.error(f"Failed to step environments: {e}")
            raise RuntimeError(f"Environment step failed: {e}")
    
    def verify_synchronization(self, step_results: Dict[str, Any]) -> bool:
        """
        Verify that all environments are synchronized with comprehensive validation.
        
        Args:
            step_results: Results from stepping all environments
            
        Returns:
            True if environments are synchronized, False otherwise
            
        Raises:
            EnvironmentSynchronizationError: If critical synchronization failure occurs
        """
        if not step_results:
            return True
        
        # Use validator if available
        if self.enable_validation and hasattr(self, 'sync_validator'):
            validation_result = self.sync_validator.validate_step_synchronization(step_results)
            
            # Handle validation issues
            for issue in validation_result.issues:
                if issue.severity == ValidationSeverity.ERROR:
                    self.collection_stats["sync_failures"] += 1
                    logger.error(f"Synchronization error: {issue.message}")
                elif issue.severity == ValidationSeverity.WARNING:
                    logger.warning(f"Synchronization warning: {issue.message}")
                elif issue.severity == ValidationSeverity.CRITICAL:
                    self.collection_stats["sync_failures"] += 1
                    raise EnvironmentSynchronizationError(
                        f"Critical synchronization failure: {issue.message}",
                        issue.details
                    )
            
            return validation_result.is_valid
        
        # Fallback to basic synchronization check
        return self._basic_synchronization_check(step_results)
    
    def _basic_synchronization_check(self, step_results: Dict[str, Any]) -> bool:
        """
        Basic synchronization check (fallback when validation is disabled).
        
        Args:
            step_results: Results from stepping all environments
            
        Returns:
            True if environments are synchronized, False otherwise
        """
        if not step_results:
            return True
        
        # Get reference values from first environment
        first_obs_type = next(iter(step_results.keys()))
        ref_result = step_results[first_obs_type]
        ref_reward = ref_result['reward']
        ref_terminated = ref_result['terminated']
        ref_truncated = ref_result['truncated']
        
        # Check that all environments have same reward, terminated, truncated
        for obs_type, result in step_results.items():
            if result['reward'] != ref_reward:
                logger.warning(f"Reward mismatch: {obs_type} has {result['reward']}, "
                             f"expected {ref_reward}")
                return False
            
            if result['terminated'] != ref_terminated:
                logger.warning(f"Terminated mismatch: {obs_type} has {result['terminated']}, "
                             f"expected {ref_terminated}")
                return False
            
            if result['truncated'] != ref_truncated:
                logger.warning(f"Truncated mismatch: {obs_type} has {result['truncated']}, "
                             f"expected {ref_truncated}")
                return False
        
        return True
    
    def sample_actions(self, observations: Dict[str, Any], step: int = 0, 
                      episode_id: str = "") -> Tuple[int, ...]:
        """
        Sample actions for all agents using the configured action sampler.
        
        Args:
            observations: Current observations from all modalities
            step: Current step in the episode
            episode_id: Current episode ID
            
        Returns:
            Tuple of actions for each agent
        """
        actions = self._action_sampler.sample_actions(
            observations, self._n_agents, step, episode_id
        )
        
        logger.debug(f"Sampled actions: {actions}")
        return actions
    
    def collect_episode_batch(self, scenario_name: str, episodes: int, seed: int, 
                             max_steps: int = 100, batch_size: Optional[int] = None) -> CollectionResult:
        """
        Collect a batch of episodes for a given scenario with comprehensive error handling and performance monitoring.
        
        Args:
            scenario_name: Name of the scenario to collect data for
            episodes: Number of episodes to collect
            seed: Random seed for reproducibility
            max_steps: Maximum steps per episode
            batch_size: Optional batch size override
            
        Returns:
            CollectionResult containing episode data and metadata
        """
        context = ErrorContext(
            operation="collect_episode_batch",
            component="SynchronizedCollector",
            scenario=scenario_name,
            additional_info={"episodes": episodes, "max_steps": max_steps}
        )
        
        logger.info(f"Starting collection of {episodes} episodes for scenario '{scenario_name}'")
        
        # Use performance profiler for the entire batch operation
        with self.performance_profiler.profile_operation("collect_episode_batch") as profiler:
            profiler.set_items_processed(episodes)
            
            # Determine optimal batch size
            if batch_size is None:
                batch_size = self.batching_optimizer.get_optimal_batch_size()
            
            logger.info(f"Using batch size: {batch_size}")
            
            # Validate memory before starting
            if self.enable_validation:
                memory_validation = self.memory_validator.validate_memory_usage()
                if memory_validation.has_errors():
                    for issue in memory_validation.get_issues_by_severity(ValidationSeverity.CRITICAL):
                        raise MemoryError(issue.message, issue.details)
                
                if memory_validation.has_warnings():
                    self.collection_stats["memory_warnings"] += 1
                    for issue in memory_validation.get_issues_by_severity(ValidationSeverity.WARNING):
                        logger.warning(f"Memory warning: {issue.message}")
            
            # Setup environments
            try:
                self._setup_environments(scenario_name)
            except Exception as e:
                error_info = self.error_handler.handle_error(e, context)
                if not error_info["recovery_successful"]:
                    raise
            
            collected_episodes = []
            collection_errors = []
            successful_episodes = 0
            
            start_time = time.time()
            
            # Process episodes in batches for memory efficiency
            for batch_start in range(0, episodes, batch_size):
                batch_end = min(batch_start + batch_size, episodes)
                batch_episodes = batch_end - batch_start
                
                logger.debug(f"Processing batch {batch_start//batch_size + 1}: episodes {batch_start+1}-{batch_end}")
                
                batch_start_time = time.time()
                batch_memory_before = 0.0
                if self.memory_profiler:
                    snapshot = self.memory_profiler.take_snapshot()
                    batch_memory_before = snapshot.rss_mb
                
                batch_successful = 0
                batch_errors = []
                
                for episode_idx in range(batch_start, batch_end):
                    episode_seed = seed + episode_idx
                    logger.debug(f"Collecting episode {episode_idx + 1}/{episodes} with seed {episode_seed}")
                    
                    # Check memory usage periodically
                    if (self.enable_validation and 
                        episode_idx % self.performance_config.memory_check_interval == 0):
                        memory_validation = self.memory_validator.validate_memory_usage()
                        if memory_validation.has_errors():
                            logger.error("Memory limit exceeded, stopping collection")
                            break
                    
                    try:
                        episode_data = self._collect_single_episode_safe(
                            scenario_name, episode_seed, max_steps, episode_idx
                        )
                        collected_episodes.append(episode_data)
                        batch_successful += 1
                        successful_episodes += 1
                        self.collection_stats["episodes_collected"] += 1
                        
                    except Exception as e:
                        error_msg = f"Episode {episode_idx + 1} failed: {str(e)}"
                        logger.error(error_msg)
                        collection_errors.append(error_msg)
                        batch_errors.append(error_msg)
                        
                        # Handle critical errors
                        if isinstance(e, (EnvironmentSynchronizationError, MemoryError)):
                            logger.error("Critical error encountered, stopping collection")
                            break
                
                # Record batch metrics
                batch_time = time.time() - batch_start_time
                batch_memory_after = 0.0
                if self.memory_profiler:
                    snapshot = self.memory_profiler.take_snapshot()
                    batch_memory_after = snapshot.rss_mb
                
                self.batching_optimizer.record_batch_metrics(
                    batch_size=batch_episodes,
                    processing_time=batch_time,
                    memory_usage_mb=max(batch_memory_before, batch_memory_after),
                    items_processed=batch_successful,
                    errors=batch_errors
                )
                
                # Trigger garbage collection if memory usage is high
                if (self.memory_profiler and 
                    self.performance_config.enable_memory_profiling and
                    batch_memory_after > self.performance_config.gc_threshold_mb):
                    gc_result = self.memory_profiler.trigger_gc()
                    logger.info(f"Triggered GC: freed {gc_result['memory_freed_mb']:.1f} MB")
                
                # Update profiler with batch progress
                if hasattr(profiler, 'update_memory_peak'):
                    profiler.update_memory_peak()
            
            collection_time = time.time() - start_time
            
            logger.info(f"Collection completed: {successful_episodes}/{episodes} episodes successful "
                       f"in {collection_time:.2f}s")
            
            return CollectionResult(
                episodes=collected_episodes,
                total_episodes=episodes,
                successful_episodes=successful_episodes,
                failed_episodes=episodes - successful_episodes,
                collection_time=collection_time,
                errors=collection_errors
            )
    
    def _collect_single_episode_safe(self, scenario_name: str, seed: int, max_steps: int, 
                                    episode_idx: int) -> EpisodeData:
        """
        Collect data from a single episode with comprehensive error handling.
        
        Args:
            scenario_name: Name of the scenario
            seed: Random seed for this episode
            max_steps: Maximum steps for this episode
            episode_idx: Index of this episode in the batch
            
        Returns:
            EpisodeData containing all collected information
        """
        context = ErrorContext(
            operation="collect_single_episode",
            component="SynchronizedCollector",
            scenario=scenario_name,
            additional_info={"seed": seed, "max_steps": max_steps, "episode_idx": episode_idx}
        )
        
        try:
            return self._collect_single_episode(scenario_name, seed, max_steps, episode_idx)
        except Exception as e:
            self.collection_stats["recovery_attempts"] += 1
            error_info = self.error_handler.handle_error(e, context)
            
            if error_info["recovery_successful"]:
                # Try again after recovery
                return self._collect_single_episode(scenario_name, seed, max_steps, episode_idx)
            else:
                # Re-raise if recovery failed
                raise
    
    def _collect_single_episode(self, scenario_name: str, seed: int, max_steps: int, 
                               episode_idx: int) -> EpisodeData:
        """
        Collect data from a single episode.
        
        Args:
            scenario_name: Name of the scenario
            seed: Random seed for this episode
            max_steps: Maximum steps for this episode
            episode_idx: Index of this episode in the batch
            
        Returns:
            EpisodeData containing all collected information
        """
        from ..features.engine import FeatureDerivationEngine
        from ..storage.manager import DatasetStorageManager
        
        # Initialize feature engine
        feature_engine = FeatureDerivationEngine()
        
        # Generate unique episode ID
        episode_id = f"ep_{scenario_name}_{seed}_{episode_idx:04d}"
        
        # Reset environments
        initial_observations = self.reset_parallel_envs(seed)
        
        # Initialize episode data storage
        episode_observations = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        episode_infos = []
        
        # Episode loop
        step = 0
        done = False
        
        while step < max_steps and not done:
            # Process current observations and derive features
            processed_obs = self._process_observations(
                initial_observations if step == 0 else step_results,
                feature_engine,
                episode_id,
                step
            )
            
            episode_observations.append(processed_obs)
            
            # Sample actions
            actions = self.sample_actions(
                initial_observations if step == 0 else step_results, 
                step, 
                episode_id
            )
            episode_actions.append(actions)
            
            # Step environments
            step_results = self.step_parallel_envs(actions)
            
            # Verify synchronization
            try:
                if not self.verify_synchronization(step_results):
                    raise EnvironmentSynchronizationError(
                        f"Environment desynchronization detected at step {step}",
                        {"step": step, "episode_id": episode_id}
                    )
            except EnvironmentSynchronizationError:
                raise  # Re-raise synchronization errors
            except Exception as e:
                logger.error(f"Synchronization verification failed: {e}")
                raise EnvironmentSynchronizationError(
                    f"Synchronization verification error at step {step}: {str(e)}"
                )
            
            # Extract common step information
            first_result = next(iter(step_results.values()))
            reward = first_result['reward']
            terminated = first_result['terminated']
            truncated = first_result['truncated']
            info = first_result['info']
            
            episode_rewards.append(reward)
            episode_dones.append(terminated or truncated)
            episode_infos.append(info)
            
            # Check termination
            done = terminated or truncated
            step += 1
            self.collection_stats["steps_collected"] += 1
        
        # Create episode metadata
        metadata = {
            'episode_id': episode_id,
            'scenario': scenario_name,
            'config': self._env_factory.get_base_config(scenario_name, self._n_agents),
            'modalities': list(self._obs_types),
            'n_agents': self._n_agents,
            'total_steps': step,
            'seed': seed,
            'max_steps': max_steps,
            'terminated_early': done and step < max_steps
        }
        
        return EpisodeData(
            episode_id=episode_id,
            scenario=scenario_name,
            observations=episode_observations,
            actions=episode_actions,
            rewards=episode_rewards,
            dones=episode_dones,
            infos=episode_infos,
            metadata=metadata
        )
    
    def _process_observations(self, step_results: Dict[str, Any], feature_engine: Any,
                             episode_id: str, step: int) -> List[Dict[str, Any]]:
        """
        Process observations from all modalities and derive features.
        
        Args:
            step_results: Results from stepping all environments
            feature_engine: Feature derivation engine
            episode_id: Current episode ID
            step: Current step number
            
        Returns:
            List of processed observations for each agent
        """
        processed_observations = []
        
        # Get observations from each modality (only for enabled modalities)
        scenario_name = self._current_scenario or "default"
        enabled_modalities = self._modality_manager.get_enabled_modalities(scenario_name)
        
        # Extract observations for enabled modalities
        modality_observations = {}
        for modality in enabled_modalities:
            if modality in step_results:
                modality_observations[modality] = step_results[modality].get('observation', [])
        
        # Process each agent's observations
        for agent_idx in range(self._n_agents):
            processed_obs = {
                'episode_id': episode_id,
                'step': step,
                'agent_id': agent_idx,
            }
            
            # Process each enabled modality
            for modality, observations in modality_observations.items():
                # Extract agent-specific observation
                if isinstance(observations, (list, tuple)) and len(observations) > agent_idx:
                    agent_obs = observations[agent_idx]
                elif observations:
                    agent_obs = observations
                else:
                    agent_obs = self._get_default_observation(modality)
                
                # Get modality configuration
                modality_config = self._modality_manager.get_modality_config(scenario_name, modality)
                
                # Apply custom processor if available
                if modality_config.processor:
                    try:
                        processed_data = modality_config.processor.process_observation(
                            agent_obs, {'episode_id': episode_id, 'step': step, 'agent_id': agent_idx}
                        )
                        agent_obs = processed_data
                    except Exception as e:
                        logger.warning(f"Custom processor failed for {modality}: {e}")
                
                # Process based on modality type
                if modality == 'Kinematics':
                    self._process_kinematics_observation(
                        agent_obs, processed_obs, feature_engine, modality_config
                    )
                elif modality in ['OccupancyGrid', 'GrayscaleObservation']:
                    self._process_binary_observation(
                        agent_obs, processed_obs, modality, modality_config
                    )
            
            processed_observations.append(processed_obs)
        
        return processed_observations
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "collection_stats": self.collection_stats.copy(),
            "performance_config": self.performance_config.to_dict()
        }
        
        # Memory profiler stats
        if self.memory_profiler:
            stats["memory_stats"] = self.memory_profiler.get_memory_stats()
            stats["memory_recommendations"] = self.memory_profiler.get_optimization_recommendations()
        
        # Performance profiler stats
        if self.performance_profiler:
            stats["performance_stats"] = self.performance_profiler.get_all_stats()
            stats["performance_summary"] = self.performance_profiler.get_performance_summary()
        
        # Batching optimizer stats
        if self.batching_optimizer:
            stats["batch_stats"] = self.batching_optimizer.get_batch_statistics()
            stats["batch_suggestion"] = self.batching_optimizer.suggest_batch_size_adjustment()
        
        return stats
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Perform performance optimization and return results."""
        optimization_results = {
            "memory_optimization": {},
            "batch_optimization": {},
            "recommendations": []
        }
        
        # Memory optimization
        if self.memory_profiler:
            gc_result = self.memory_profiler.trigger_gc()
            optimization_results["memory_optimization"] = gc_result
            
            memory_recommendations = self.memory_profiler.get_optimization_recommendations()
            optimization_results["recommendations"].extend(memory_recommendations)
        
        # Batch size optimization
        if self.batching_optimizer:
            batch_suggestion = self.batching_optimizer.suggest_batch_size_adjustment()
            optimization_results["batch_optimization"] = batch_suggestion
            
            if batch_suggestion.get("adjustment") != "maintain":
                suggested_size = batch_suggestion.get("suggested_batch_size")
                if suggested_size:
                    self.batching_optimizer.update_batch_size(suggested_size)
                    optimization_results["recommendations"].append(
                        f"Updated batch size to {suggested_size}"
                    )
        
        return optimization_results
    
    def reset_performance_metrics(self) -> None:
        """Reset all performance metrics and statistics."""
        self.collection_stats = {
            "episodes_collected": 0,
            "steps_collected": 0,
            "sync_failures": 0,
            "memory_warnings": 0,
            "validation_errors": 0,
            "recovery_attempts": 0
        }
        
        if self.performance_profiler:
            self.performance_profiler.clear_metrics()
        
        if self.batching_optimizer:
            self.batching_optimizer.clear_metrics()
        
        logger.info("Performance metrics reset")
    
    def __del__(self):
        """Cleanup when collector is destroyed."""
        if hasattr(self, 'memory_profiler') and self.memory_profiler:
            self.memory_profiler.stop_monitoring()
        self._cleanup_environments()
    
    def _get_default_observation(self, modality: str) -> Any:
        """Get default observation for a modality when data is missing."""
        if modality == 'Kinematics':
            return []
        elif modality == 'OccupancyGrid':
            return np.zeros((11, 11))
        elif modality == 'GrayscaleObservation':
            return np.zeros((84, 84, 3))
        else:
            return None
    
    def _process_kinematics_observation(self, observation: Any, processed_obs: Dict[str, Any],
                                      feature_engine: Any, config: ModalityConfig) -> None:
        """Process kinematics observation and extract features."""
        kin_array = np.array(observation)
        
        # Store raw kinematics
        processed_obs['kinematics_raw'] = (
            kin_array.flatten().tolist() if len(kin_array.shape) > 1 else kin_array.tolist()
        )
        
        # Extract features if enabled
        if config.feature_extraction_enabled:
            if len(kin_array.shape) == 2 and kin_array.shape[0] > 0:
                # Multi-vehicle observation format
                ego_vehicle = kin_array[0]  # First vehicle is ego
                other_vehicles = kin_array[1:] if len(kin_array) > 1 else np.array([])
                
                # Extract features
                try:
                    kinematics_features = feature_engine.derive_kinematics_features(kin_array)
                    ttc = feature_engine.calculate_ttc(ego_vehicle, other_vehicles)
                    summary = feature_engine.generate_language_summary(ego_vehicle, other_vehicles)
                    traffic_metrics = feature_engine.estimate_traffic_metrics(kin_array)
                    
                    # Add derived features
                    processed_obs['ttc'] = ttc
                    processed_obs['summary_text'] = summary
                    processed_obs.update(kinematics_features)
                    processed_obs.update(traffic_metrics)
                    
                except Exception as e:
                    logger.warning(f"Feature extraction failed for kinematics: {e}")
                    processed_obs['ttc'] = float('inf')
                    processed_obs['summary_text'] = "Feature extraction failed"
            else:
                # Fallback for unexpected observation format
                processed_obs['ttc'] = float('inf')
                processed_obs['summary_text'] = "Unable to process observation"
    
    def _process_binary_observation(self, observation: Any, processed_obs: Dict[str, Any],
                                  modality: str, config: ModalityConfig) -> None:
        """Process binary observations (OccupancyGrid, GrayscaleObservation)."""
        if not config.storage_enabled:
            return
        
        try:
            # Encode binary arrays
            from ..storage.encoders import BinaryArrayEncoder
            encoder = BinaryArrayEncoder()
            
            obs_array = np.array(observation)
            
            if modality == 'OccupancyGrid':
                binary_data = encoder.encode_single(obs_array)
                processed_obs.update({
                    'occupancy_blob': binary_data['blob'],
                    'occupancy_shape': binary_data['shape'],
                    'occupancy_dtype': binary_data['dtype']
                })
            elif modality == 'GrayscaleObservation':
                binary_data = encoder.encode_single(obs_array)
                processed_obs.update({
                    'grayscale_blob': binary_data['blob'],
                    'grayscale_shape': binary_data['shape'],
                    'grayscale_dtype': binary_data['dtype']
                })
                
        except Exception as e:
            logger.warning(f"Binary encoding failed for {modality}: {e}")
    
    def set_action_sampler(self, action_sampler: ActionSampler) -> None:
        """
        Set a new action sampling strategy.
        
        Args:
            action_sampler: New action sampling strategy
        """
        self._action_sampler = action_sampler
        logger.info(f"Updated action sampler to: {type(action_sampler).__name__}")
        
        # Reset with current seed if available
        if self._current_seed is not None:
            self._action_sampler.reset(self._current_seed)
    
    def get_action_sampler(self) -> ActionSampler:
        """
        Get the current action sampling strategy.
        
        Returns:
            Current action sampler
        """
        return self._action_sampler
    
    def set_modality_config_manager(self, manager: ModalityConfigManager) -> None:
        """
        Set a new modality configuration manager.
        
        Args:
            manager: New modality configuration manager
        """
        self._modality_manager = manager
        logger.info("Updated modality configuration manager")
        
        # Clear current environments to force recreation with new config
        if self._environments:
            self._cleanup_environments()
    
    def get_modality_config_manager(self) -> ModalityConfigManager:
        """
        Get the current modality configuration manager.
        
        Returns:
            Current modality configuration manager
        """
        return self._modality_manager
    
    def get_enabled_modalities(self, scenario_name: str) -> List[str]:
        """
        Get list of enabled modalities for a scenario.
        
        Args:
            scenario_name: Name of the scenario
            
        Returns:
            List of enabled modality names
        """
        return self._modality_manager.get_enabled_modalities(scenario_name)
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        stats = dict(self.collection_stats)
        
        if self.enable_validation:
            if hasattr(self, 'sync_validator'):
                stats["synchronization_stats"] = self.sync_validator.get_synchronization_stats()
            if hasattr(self, 'memory_validator'):
                stats["memory_stats"] = self.memory_validator.get_memory_stats()
        
        stats["error_stats"] = self.error_handler.get_error_statistics()
        stats["degradation_status"] = self.degradation_manager.get_degradation_status()
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset collection statistics."""
        self.collection_stats = {
            "episodes_collected": 0,
            "steps_collected": 0,
            "sync_failures": 0,
            "memory_warnings": 0,
            "validation_errors": 0,
            "recovery_attempts": 0
        }
        
        if self.enable_validation:
            if hasattr(self, 'sync_validator'):
                self.sync_validator.reset()
        
        self.error_handler.clear_error_history()
        logger.info("Collection statistics reset")
    
    def validate_episode_data(self, episode_data: EpisodeData) -> ValidationResult:
        """
        Validate collected episode data.
        
        Args:
            episode_data: Episode data to validate
            
        Returns:
            ValidationResult with validation status
        """
        if not self.enable_validation:
            from .validation import ValidationResult
            return ValidationResult(True, [])
        
        # Validate observation data integrity
        all_observations = []
        for obs_list in episode_data.observations:
            all_observations.extend(obs_list)
        
        return self.data_validator.validate_observation_data(all_observations)
    
    def perform_health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check of the collector.
        
        Returns:
            Dictionary with health check results
        """
        health_status = {
            "overall_healthy": True,
            "environments_ready": bool(self._environments),
            "validation_enabled": self.enable_validation,
            "issues": []
        }
        
        # Check environment status
        if not self._environments:
            health_status["issues"].append("No environments initialized")
            health_status["overall_healthy"] = False
        
        # Check memory if validation is enabled
        if self.enable_validation and hasattr(self, 'memory_validator'):
            memory_validation = self.memory_validator.validate_memory_usage()
            if memory_validation.has_errors():
                health_status["issues"].extend([
                    f"Memory issue: {issue.message}" 
                    for issue in memory_validation.get_issues_by_severity(ValidationSeverity.ERROR)
                ])
                health_status["overall_healthy"] = False
        
        # Check for degraded features
        degradation_status = self.degradation_manager.get_degradation_status()
        if degradation_status["total_degraded"] > 0:
            health_status["issues"].append(
                f"{degradation_status['total_degraded']} features are degraded"
            )
        
        # Add statistics
        health_status["statistics"] = self.get_collection_statistics()
        
        return health_status
    
    def __del__(self):
        """Cleanup environments on deletion."""
        self._cleanup_environments()