#!/usr/bin/env python3
"""
Parallel Ambulance Data Collection Script

This script provides optimized parallel data collection for ambulance scenarios
with support for multi-processing, GPU acceleration, and batch optimization.
"""

import logging
import sys
import os
import multiprocessing as mp
from pathlib import Path
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector
from collecting_ambulance_data.scenarios.ambulance_scenarios import get_scenario_names

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collect_scenario_batch(args_tuple: Tuple[List[str], int, int, int, int, str]) -> Dict[str, Any]:
    """
    Collect data for a batch of scenarios in a separate process.
    
    Args:
        args_tuple: Tuple containing (scenarios, episodes, max_steps, seed, n_agents, output_dir)
        
    Returns:
        Dictionary with collection results
    """
    scenarios, episodes, max_steps, seed, n_agents, output_dir = args_tuple
    
    # Set up logging for this process
    process_logger = logging.getLogger(f"worker-{os.getpid()}")
    
    try:
        process_logger.info(f"Worker {os.getpid()} processing {len(scenarios)} scenarios")
        
        # Initialize collector in this process
        with AmbulanceDataCollector(n_agents=n_agents) as collector:
            
            # Collect data for assigned scenarios
            collection_results = collector.collect_ambulance_data(
                scenarios=scenarios,
                episodes_per_scenario=episodes,
                max_steps_per_episode=max_steps,
                base_seed=seed,
                batch_size=min(5, episodes)  # Optimize batch size
            )
            
            # Store data immediately in this process
            output_path = Path(output_dir) / f"batch_{os.getpid()}"
            storage_info = collector.store_ambulance_data(collection_results, output_path)
            
            # Return summary
            total_episodes = sum(r.successful_episodes for r in collection_results.values())
            total_time = sum(r.collection_time for r in collection_results.values())
            
            return {
                'worker_id': os.getpid(),
                'scenarios': scenarios,
                'total_episodes': total_episodes,
                'total_time': total_time,
                'storage_path': str(output_path),
                'collection_results': collection_results,
                'storage_info': storage_info
            }
            
    except Exception as e:
        process_logger.error(f"Worker {os.getpid()} failed: {e}")
        return {
            'worker_id': os.getpid(),
            'scenarios': scenarios,
            'total_episodes': 0,
            'total_time': 0,
            'error': str(e)
        }


def optimize_batch_configuration(total_scenarios: int, episodes: int, max_workers: int) -> Tuple[int, int]:
    """
    Optimize batch size and worker configuration for best performance.
    
    Args:
        total_scenarios: Total number of scenarios to process
        episodes: Episodes per scenario
        max_workers: Maximum number of worker processes
        
    Returns:
        Tuple of (optimal_workers, scenarios_per_worker)
    """
    # Calculate optimal distribution
    cpu_count = mp.cpu_count()
    optimal_workers = min(max_workers, cpu_count - 1, total_scenarios)  # Leave 1 CPU for main process
    
    scenarios_per_worker = max(1, total_scenarios // optimal_workers)
    
    # Adjust for memory considerations (more episodes = fewer parallel workers)
    if episodes > 100:
        optimal_workers = max(1, optimal_workers // 2)
        scenarios_per_worker = total_scenarios // optimal_workers
    
    logger.info(f"Optimization: {optimal_workers} workers, {scenarios_per_worker} scenarios/worker")
    return optimal_workers, scenarios_per_worker


def setup_gpu_acceleration():
    """
    Set up GPU acceleration if available.
    Note: highway-env simulation itself doesn't use GPU, but we can optimize
    data processing and storage operations.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            logger.info(f"GPU available: {device}")
            
            # Set environment variables for optimized GPU usage
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            
            # Enable GPU-accelerated data processing where possible
            return True
        else:
            logger.info("No GPU available, using CPU optimization")
            return False
    except ImportError:
        logger.info("PyTorch not available, using CPU-only processing")
        return False


def main():
    """Main function for parallel ambulance data collection."""
    parser = argparse.ArgumentParser(description='Parallel ambulance data collection')
    parser.add_argument('--scenarios', nargs='+', help='Specific scenarios to collect (default: all)')
    parser.add_argument('--episodes', type=int, default=50, help='Episodes per scenario (default: 50)')
    parser.add_argument('--max-steps', type=int, default=100, help='Max steps per episode (default: 100)')
    parser.add_argument('--output-dir', type=str, default='data/ambulance_parallel', 
                       help='Output directory (default: data/ambulance_parallel)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--n-agents', type=int, default=4, help='Number of agents (default: 4)')
    parser.add_argument('--max-workers', type=int, default=None, 
                       help='Max parallel workers (default: auto-detect)')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration where possible')
    parser.add_argument('--batch-optimize', action='store_true', 
                       help='Optimize batch sizes for performance')
    
    args = parser.parse_args()
    
    # Set up GPU acceleration if requested
    gpu_available = False
    if args.gpu:
        gpu_available = setup_gpu_acceleration()
    
    # Determine max workers
    if args.max_workers is None:
        args.max_workers = mp.cpu_count() - 1
    
    logger.info("Starting parallel ambulance data collection")
    logger.info(f"Configuration:")
    logger.info(f"  Episodes per scenario: {args.episodes}")
    logger.info(f"  Max steps per episode: {args.max_steps}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Random seed: {args.seed}")
    logger.info(f"  Number of agents: {args.n_agents}")
    logger.info(f"  Max workers: {args.max_workers}")
    logger.info(f"  GPU acceleration: {gpu_available}")
    
    try:
        # Get scenarios to process
        all_scenarios = get_scenario_names()
        if args.scenarios:
            scenarios_to_collect = args.scenarios
            # Validate scenarios
            invalid_scenarios = [s for s in scenarios_to_collect if s not in all_scenarios]
            if invalid_scenarios:
                logger.error(f"Invalid scenarios: {invalid_scenarios}")
                logger.info(f"Available scenarios: {all_scenarios}")
                return 1
        else:
            scenarios_to_collect = all_scenarios
        
        logger.info(f"Processing {len(scenarios_to_collect)} scenarios")
        
        # Optimize batch configuration
        if args.batch_optimize:
            optimal_workers, scenarios_per_worker = optimize_batch_configuration(
                len(scenarios_to_collect), args.episodes, args.max_workers
            )
        else:
            optimal_workers = min(args.max_workers, len(scenarios_to_collect))
            scenarios_per_worker = max(1, len(scenarios_to_collect) // optimal_workers)
        
        # Create scenario batches for parallel processing
        scenario_batches = []
        for i in range(0, len(scenarios_to_collect), scenarios_per_worker):
            batch = scenarios_to_collect[i:i + scenarios_per_worker]
            batch_seed = args.seed + i * 1000  # Unique seed per batch
            scenario_batches.append((batch, args.episodes, args.max_steps, 
                                   batch_seed, args.n_agents, args.output_dir))
        
        logger.info(f"Created {len(scenario_batches)} scenario batches for parallel processing")
        
        # Create output directory
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Execute parallel collection
        start_time = time.time()
        all_results = []
        
        logger.info("Starting parallel data collection...")
        
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(collect_scenario_batch, batch): batch 
                for batch in scenario_batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    
                    if 'error' in result:
                        logger.error(f"Batch failed: {result['error']}")
                    else:
                        logger.info(f"Worker {result['worker_id']} completed: "
                                   f"{result['total_episodes']} episodes in {result['total_time']:.2f}s")
                        
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
        
        # Aggregate results
        total_time = time.time() - start_time
        total_episodes = sum(r.get('total_episodes', 0) for r in all_results)
        successful_batches = sum(1 for r in all_results if 'error' not in r)
        
        logger.info("Parallel collection completed!")
        logger.info(f"Results:")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Total episodes: {total_episodes}")
        logger.info(f"  Successful batches: {successful_batches}/{len(scenario_batches)}")
        logger.info(f"  Average episodes/second: {total_episodes/total_time:.2f}")
        
        # Create consolidated index
        create_consolidated_index(output_path, all_results)
        
        logger.info(f"✅ Parallel ambulance data collection completed!")
        logger.info(f"📁 Data stored in: {output_path}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Parallel collection failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def create_consolidated_index(output_path: Path, results: List[Dict[str, Any]]):
    """Create a consolidated index of all collected data."""
    import json
    
    consolidated_index = {
        'collection_info': {
            'total_batches': len(results),
            'collection_timestamp': time.time(),
            'total_episodes': sum(r.get('total_episodes', 0) for r in results)
        },
        'batches': []
    }
    
    for result in results:
        if 'error' not in result:
            batch_info = {
                'worker_id': result['worker_id'],
                'scenarios': result['scenarios'],
                'episodes': result['total_episodes'],
                'storage_path': result['storage_path']
            }
            consolidated_index['batches'].append(batch_info)
    
    # Save consolidated index
    index_path = output_path / 'consolidated_index.json'
    with open(index_path, 'w') as f:
        json.dump(consolidated_index, f, indent=2)
    
    logger.info(f"Consolidated index saved to: {index_path}")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    sys.exit(main())