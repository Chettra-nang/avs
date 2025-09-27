#!/usr/bin/env python3
"""
Basic Ambulance Data Collection Script

This script demonstrates how to use the AmbulanceDataCollector to collect
multi-modal data from ambulance scenarios and store it for analysis.
"""

import logging
import sys
from pathlib import Path
import argparse

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function for basic ambulance data collection."""
    parser = argparse.ArgumentParser(description='Collect ambulance scenario data')
    parser.add_argument('--scenarios', nargs='+', help='Specific scenarios to collect (default: all)')
    parser.add_argument('--episodes', type=int, default=10, help='Episodes per scenario (default: 10)')
    parser.add_argument('--max-steps', type=int, default=50, help='Max steps per episode (default: 50)')
    parser.add_argument('--output-dir', type=str, default='data/ambulance_collection', 
                       help='Output directory (default: data/ambulance_collection)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size (default: 5)')
    parser.add_argument('--n-agents', type=int, default=4, help='Number of agents (default: 4)')
    
    args = parser.parse_args()
    
    logger.info("Starting basic ambulance data collection")
    logger.info(f"Configuration:")
    logger.info(f"  Episodes per scenario: {args.episodes}")
    logger.info(f"  Max steps per episode: {args.max_steps}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Random seed: {args.seed}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Number of agents: {args.n_agents}")
    
    try:
        # Initialize the ambulance data collector
        with AmbulanceDataCollector(n_agents=args.n_agents) as collector:
            
            # Get available scenarios
            available_scenarios = collector.get_available_scenarios()
            logger.info(f"Available ambulance scenarios: {len(available_scenarios)}")
            
            # Determine which scenarios to collect
            if args.scenarios:
                scenarios_to_collect = args.scenarios
                # Validate scenarios
                invalid_scenarios = [s for s in scenarios_to_collect if s not in available_scenarios]
                if invalid_scenarios:
                    logger.error(f"Invalid scenarios specified: {invalid_scenarios}")
                    logger.info(f"Available scenarios: {available_scenarios}")
                    return 1
            else:
                scenarios_to_collect = available_scenarios
            
            logger.info(f"Collecting data from {len(scenarios_to_collect)} scenarios:")
            for i, scenario in enumerate(scenarios_to_collect, 1):
                info = collector.get_scenario_info(scenario)
                logger.info(f"  {i}. {scenario}: {info['description']}")
            
            # Collect data from all specified scenarios
            collection_results = collector.collect_ambulance_data(
                scenarios=scenarios_to_collect,
                episodes_per_scenario=args.episodes,
                max_steps_per_episode=args.max_steps,
                base_seed=args.seed,
                batch_size=args.batch_size
            )
            
            # Display collection results
            logger.info("Collection Results:")
            total_successful = 0
            total_failed = 0
            
            for scenario_name, result in collection_results.items():
                logger.info(f"  {scenario_name}:")
                logger.info(f"    Successful episodes: {result.successful_episodes}")
                logger.info(f"    Failed episodes: {result.failed_episodes}")
                logger.info(f"    Collection time: {result.collection_time:.2f}s")
                
                total_successful += result.successful_episodes
                total_failed += result.failed_episodes
                
                if result.errors:
                    logger.warning(f"    Errors: {len(result.errors)}")
            
            logger.info(f"Total successful episodes: {total_successful}")
            logger.info(f"Total failed episodes: {total_failed}")
            
            # Store the collected data
            output_path = Path(args.output_dir)
            logger.info(f"Storing data to: {output_path}")
            
            storage_info = collector.store_ambulance_data(collection_results, output_path)
            
            logger.info("Storage Results:")
            logger.info(f"  Scenarios stored: {storage_info['scenarios_stored']}")
            logger.info(f"  Episodes stored: {storage_info['total_episodes_stored']}")
            logger.info(f"  Output directory: {storage_info['output_dir']}")
            
            if 'dataset_index_path' in storage_info:
                logger.info(f"  Dataset index: {storage_info['dataset_index_path']}")
            
            if storage_info['errors']:
                logger.warning(f"  Storage errors: {len(storage_info['errors'])}")
                for error in storage_info['errors']:
                    logger.warning(f"    {error}")
            
            # Display final statistics
            stats = collector.get_collection_statistics()
            logger.info("Final Statistics:")
            logger.info(f"  Ambulance episodes collected: {stats['ambulance_episodes_collected']}")
            logger.info(f"  Ambulance scenarios processed: {stats['ambulance_scenarios_processed']}")
            logger.info(f"  Total steps collected: {stats['steps_collected']}")
            
            logger.info("✅ Ambulance data collection completed successfully!")
            return 0
            
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())