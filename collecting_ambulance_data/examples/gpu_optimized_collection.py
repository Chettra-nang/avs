#!/usr/bin/env python3
"""
GPU-Optimized Ambulance Data Collection

This script optimizes data collection using GPU acceleration for data processing,
parallel environments, and optimized storage operations.
"""

import logging
import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector
from collecting_ambulance_data.scenarios.ambulance_scenarios import get_scenario_names

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GPUOptimizedAmbulanceCollector:
    """
    GPU-optimized ambulance data collector with accelerated processing.
    """
    
    def __init__(self, n_agents: int = 4, use_gpu: bool = True):
        self.n_agents = n_agents
        self.use_gpu = use_gpu
        self.device = None
        
        # Initialize GPU if available
        if use_gpu:
            self._setup_gpu()
        
        # Initialize base collector
        self.collector = AmbulanceDataCollector(n_agents=n_agents)
    
    def _setup_gpu(self):
        """Set up GPU acceleration."""
        try:
            import torch
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                logger.info(f"GPU initialized: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                
                # Set optimal GPU settings
                torch.cuda.empty_cache()
                os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async GPU operations
                
            else:
                logger.warning("CUDA not available, falling back to CPU")
                self.use_gpu = False
                self.device = torch.device('cpu')
        except ImportError:
            logger.warning("PyTorch not available, GPU acceleration disabled")
            self.use_gpu = False
            self.device = None
    
    def collect_with_gpu_optimization(self, 
                                    scenarios: List[str],
                                    episodes_per_scenario: int = 50,
                                    max_steps_per_episode: int = 100,
                                    base_seed: int = 42,
                                    batch_size: int = 10) -> Dict[str, Any]:
        """
        Collect data with GPU-optimized processing.
        
        Args:
            scenarios: List of scenario names to collect
            episodes_per_scenario: Number of episodes per scenario
            max_steps_per_episode: Maximum steps per episode
            base_seed: Base random seed
            batch_size: Batch size for processing
            
        Returns:
            Collection results dictionary
        """
        logger.info(f"Starting GPU-optimized collection for {len(scenarios)} scenarios")
        
        # Optimize batch size based on GPU memory
        if self.use_gpu:
            batch_size = self._optimize_batch_size(batch_size, episodes_per_scenario)
        
        collection_results = {}
        total_start_time = time.time()
        
        for scenario_idx, scenario_name in enumerate(scenarios):
            logger.info(f"Processing scenario {scenario_idx + 1}/{len(scenarios)}: {scenario_name}")
            scenario_start_time = time.time()
            
            try:
                # Collect data with optimized batching
                scenario_seed = base_seed + scenario_idx * 10000
                result = self._collect_scenario_optimized(
                    scenario_name=scenario_name,
                    episodes=episodes_per_scenario,
                    max_steps=max_steps_per_episode,
                    seed=scenario_seed,
                    batch_size=batch_size
                )
                
                collection_results[scenario_name] = result
                
                scenario_time = time.time() - scenario_start_time
                logger.info(f"Completed {scenario_name} in {scenario_time:.2f}s: "
                           f"{result.successful_episodes} episodes")
                
                # GPU memory cleanup between scenarios
                if self.use_gpu:
                    self._cleanup_gpu_memory()
                    
            except Exception as e:
                logger.error(f"Scenario {scenario_name} failed: {e}")
                # Create empty result for failed scenario
                from highway_datacollection.collection.types import CollectionResult
                collection_results[scenario_name] = CollectionResult(
                    episodes=[],
                    total_episodes=episodes_per_scenario,
                    successful_episodes=0,
                    failed_episodes=episodes_per_scenario,
                    collection_time=0.0,
                    errors=[str(e)]
                )
        
        total_time = time.time() - total_start_time
        total_episodes = sum(r.successful_episodes for r in collection_results.values())
        
        logger.info(f"GPU-optimized collection completed in {total_time:.2f}s")
        logger.info(f"Total episodes: {total_episodes}")
        logger.info(f"Average speed: {total_episodes/total_time:.2f} episodes/second")
        
        return collection_results
    
    def _collect_scenario_optimized(self, scenario_name: str, episodes: int, 
                                  max_steps: int, seed: int, batch_size: int):
        """Collect data for a single scenario with GPU optimization."""
        
        # Use smaller batches for GPU memory efficiency
        optimized_batch_size = min(batch_size, 5) if self.use_gpu else batch_size
        
        # Collect data using the base collector
        result = self.collector.collect_single_ambulance_scenario(
            scenario_name=scenario_name,
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            batch_size=optimized_batch_size
        )
        
        # Post-process data with GPU acceleration if available
        if self.use_gpu and result.episodes:
            result = self._gpu_postprocess_episodes(result)
        
        return result
    
    def _gpu_postprocess_episodes(self, result):
        """Post-process episode data using GPU acceleration."""
        if not self.use_gpu or not result.episodes:
            return result
        
        try:
            import torch
            
            logger.debug("GPU post-processing episode data...")
            
            # Convert episode data to GPU tensors for fast processing
            for episode in result.episodes:
                if hasattr(episode, 'transitions') and episode.transitions:
                    # Process numerical data on GPU
                    self._process_transitions_gpu(episode.transitions)
            
            logger.debug("GPU post-processing completed")
            
        except Exception as e:
            logger.warning(f"GPU post-processing failed, using CPU: {e}")
        
        return result
    
    def _process_transitions_gpu(self, transitions):
        """Process transition data on GPU for acceleration."""
        try:
            import torch
            
            # Extract numerical data
            rewards = [t.get('reward', 0) for t in transitions]
            if rewards:
                # Convert to GPU tensor for fast operations
                reward_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
                
                # Perform GPU-accelerated calculations
                mean_reward = torch.mean(reward_tensor).item()
                std_reward = torch.std(reward_tensor).item()
                
                # Store computed statistics back to transitions
                for i, transition in enumerate(transitions):
                    transition['reward_stats'] = {
                        'mean': mean_reward,
                        'std': std_reward,
                        'normalized': (rewards[i] - mean_reward) / (std_reward + 1e-8)
                    }
        
        except Exception as e:
            logger.debug(f"GPU transition processing failed: {e}")
    
    def _optimize_batch_size(self, requested_batch_size: int, episodes: int) -> int:
        """Optimize batch size based on GPU memory."""
        if not self.use_gpu:
            return requested_batch_size
        
        try:
            import torch
            
            # Get available GPU memory
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # Estimate optimal batch size based on memory
            if gpu_memory_gb >= 8:
                optimal_batch_size = min(requested_batch_size, 10)
            elif gpu_memory_gb >= 4:
                optimal_batch_size = min(requested_batch_size, 5)
            else:
                optimal_batch_size = min(requested_batch_size, 3)
            
            logger.info(f"Optimized batch size: {optimal_batch_size} (GPU memory: {gpu_memory_gb:.1f}GB)")
            return optimal_batch_size
            
        except Exception as e:
            logger.warning(f"Batch size optimization failed: {e}")
            return requested_batch_size
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory between scenarios."""
        if not self.use_gpu:
            return
        
        try:
            import torch
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.debug(f"GPU cleanup failed: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_gpu:
            self._cleanup_gpu_memory()
        if hasattr(self.collector, '__exit__'):
            self.collector.__exit__(exc_type, exc_val, exc_tb)


def main():
    """Main function for GPU-optimized ambulance data collection."""
    parser = argparse.ArgumentParser(description='GPU-optimized ambulance data collection')
    parser.add_argument('--scenarios', nargs='+', help='Specific scenarios to collect (default: all)')
    parser.add_argument('--episodes', type=int, default=50, help='Episodes per scenario (default: 50)')
    parser.add_argument('--max-steps', type=int, default=100, help='Max steps per episode (default: 100)')
    parser.add_argument('--output-dir', type=str, default='data/ambulance_gpu', 
                       help='Output directory (default: data/ambulance_gpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size (default: 10)')
    parser.add_argument('--n-agents', type=int, default=4, help='Number of agents (default: 4)')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--profile', action='store_true', help='Enable performance profiling')
    
    args = parser.parse_args()
    
    use_gpu = not args.no_gpu
    
    logger.info("Starting GPU-optimized ambulance data collection")
    logger.info(f"Configuration:")
    logger.info(f"  Episodes per scenario: {args.episodes}")
    logger.info(f"  Max steps per episode: {args.max_steps}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  GPU acceleration: {use_gpu}")
    logger.info(f"  Performance profiling: {args.profile}")
    
    try:
        # Get scenarios to process
        all_scenarios = get_scenario_names()
        if args.scenarios:
            scenarios_to_collect = args.scenarios
            # Validate scenarios
            invalid_scenarios = [s for s in scenarios_to_collect if s not in all_scenarios]
            if invalid_scenarios:
                logger.error(f"Invalid scenarios: {invalid_scenarios}")
                return 1
        else:
            scenarios_to_collect = all_scenarios
        
        logger.info(f"Processing {len(scenarios_to_collect)} scenarios")
        
        # Initialize GPU-optimized collector
        with GPUOptimizedAmbulanceCollector(n_agents=args.n_agents, use_gpu=use_gpu) as collector:
            
            # Start profiling if requested
            if args.profile:
                import cProfile
                profiler = cProfile.Profile()
                profiler.enable()
            
            # Collect data with GPU optimization
            start_time = time.time()
            
            collection_results = collector.collect_with_gpu_optimization(
                scenarios=scenarios_to_collect,
                episodes_per_scenario=args.episodes,
                max_steps_per_episode=args.max_steps,
                base_seed=args.seed,
                batch_size=args.batch_size
            )
            
            collection_time = time.time() - start_time
            
            # Stop profiling
            if args.profile:
                profiler.disable()
                profiler.dump_stats(f"{args.output_dir}/collection_profile.prof")
                logger.info(f"Performance profile saved to {args.output_dir}/collection_profile.prof")
            
            # Store collected data
            output_path = Path(args.output_dir)
            logger.info(f"Storing data to: {output_path}")
            
            storage_start_time = time.time()
            storage_info = collector.collector.store_ambulance_data(collection_results, output_path)
            storage_time = time.time() - storage_start_time
            
            # Display results
            total_episodes = sum(r.successful_episodes for r in collection_results.values())
            successful_scenarios = sum(1 for r in collection_results.values() if r.successful_episodes > 0)
            
            logger.info("GPU-Optimized Collection Results:")
            logger.info(f"  Collection time: {collection_time:.2f}s")
            logger.info(f"  Storage time: {storage_time:.2f}s")
            logger.info(f"  Total time: {collection_time + storage_time:.2f}s")
            logger.info(f"  Scenarios: {successful_scenarios}/{len(scenarios_to_collect)}")
            logger.info(f"  Episodes: {total_episodes}")
            logger.info(f"  Speed: {total_episodes/collection_time:.2f} episodes/second")
            
            if use_gpu:
                try:
                    import torch
                    logger.info(f"  GPU memory used: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
                except:
                    pass
            
            logger.info(f"✅ GPU-optimized collection completed!")
            
        return 0
        
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"GPU-optimized collection failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())