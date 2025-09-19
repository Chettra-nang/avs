#!/usr/bin/env python3
"""
Demonstration of the episode data collection pipeline.

This script shows how to use the SynchronizedCollector to collect
multi-modal episode data from HighwayEnv scenarios.
"""

import logging
from highway_datacollection.collection.collector import SynchronizedCollector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate episode data collection."""
    logger.info("Starting episode collection demonstration")
    
    # Initialize collector with 2 agents
    collector = SynchronizedCollector(n_agents=2)
    
    # Collection parameters
    scenario = "free_flow"
    episodes = 3
    seed = 42
    max_steps = 20
    
    logger.info(f"Collecting {episodes} episodes for scenario '{scenario}'")
    logger.info(f"Parameters: seed={seed}, max_steps={max_steps}, n_agents=2")
    
    try:
        # Collect episode batch
        result = collector.collect_episode_batch(
            scenario, episodes, seed, max_steps
        )
        
        # Display results
        logger.info(f"Collection completed in {result.collection_time:.2f}s")
        logger.info(f"Successful episodes: {result.successful_episodes}/{result.total_episodes}")
        
        if result.failed_episodes > 0:
            logger.warning(f"Failed episodes: {result.failed_episodes}")
            for error in result.errors:
                logger.warning(f"  - {error}")
        
        # Analyze collected data
        for i, episode_data in enumerate(result.episodes):
            logger.info(f"\nEpisode {i+1} ({episode_data.episode_id}):")
            logger.info(f"  - Steps: {episode_data.metadata['total_steps']}")
            logger.info(f"  - Terminated early: {episode_data.metadata.get('terminated_early', False)}")
            logger.info(f"  - Observations collected: {len(episode_data.observations)}")
            logger.info(f"  - Actions taken: {len(episode_data.actions)}")
            
            # Show sample observation structure
            if episode_data.observations:
                sample_obs = episode_data.observations[0][0]  # First step, first agent
                logger.info(f"  - Sample observation keys: {list(sample_obs.keys())}")
                logger.info(f"  - TTC value: {sample_obs['ttc']:.3f}")
                logger.info(f"  - Summary: {sample_obs['summary_text'][:50]}...")
                logger.info(f"  - Occupancy grid shape: {sample_obs['occupancy_shape']}")
                logger.info(f"  - Grayscale image shape: {sample_obs['grayscale_shape']}")
        
        logger.info("\nDemonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise
    
    finally:
        # Cleanup
        collector._cleanup_environments()
        logger.info("Cleaned up environments")


if __name__ == "__main__":
    main()