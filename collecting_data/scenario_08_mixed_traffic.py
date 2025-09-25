#!/usr/bin/env python3
"""    logger.info("Starting Scenario 8: Mixed Traffic Patterns data collection")
    
    # Storage configuration
    storage_path = Path("/home/chettra/ITC/Research/AVs/data/scenario_08_mixed_traffic")
    storage_path.mkdir(parents=True, exist_ok=True)ario 8: Mixed Traffic Conditions
Mixed scenario combining moderate traffic with varying densities throughout episodes.
Collects complete multimodal dataset with 1000 episodes for varied condition analysis.
"""

import sys
import logging
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from highway_datacollection.collection.orchestrator import run_full_collection
from highway_datacollection.collection.collector import SynchronizedCollector
from highway_datacollection.collection.modality_config import ModalityConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Collect data for mixed traffic conditions scenario."""
    logger.info("Starting Scenario 8: Mixed Traffic Conditions data collection")
    
    # Storage configuration
    storage_path = Path("data/scenario_08_mixed_traffic")
    storage_path.mkdir(parents=True, exist_ok=True)
    
    # Collection parameters - Mix of free_flow and dense_commuting characteristics
    collection_config = {
        "base_storage_path": storage_path,
        "episodes_per_scenario": 1000,
        "n_agents": 2,
        "max_steps_per_episode": 190,
        "scenarios": ["dense_commuting"],
        "base_seed": 8001,
        "batch_size": 14
    }
    
    # Configure complete dataset collection
    logger.info("Configuring complete multimodal data collection")
    # Configure complete dataset collection
    modality_manager = ModalityConfigManager()
    collector = SynchronizedCollector(
        n_agents=collection_config["n_agents"],
        modality_config_manager=modality_manager
    )
    
    logger.info(f"Collection configuration:")
    logger.info(f"  - Episodes: {collection_config['episodes_per_scenario']}")
    logger.info(f"  - Scenario: Mixed Traffic (45 vehicles, 4 lanes)")
    logger.info(f"  - Modalities: Kinematics, Occupancy Grid, Grayscale")
    logger.info(f"  - Storage: {storage_path}")
    logger.info(f"  - Batch size: {collection_config['batch_size']}")
    
    try:
        # Run data collection
        result = run_full_collection(modality_config_manager=modality_manager, **collection_config)
        
        # Report results
        logger.info("=" * 60)
        logger.info("COLLECTION COMPLETED - SCENARIO 8")
        logger.info("=" * 60)
        logger.info(f"✓ Total episodes: {result.total_episodes}")
        logger.info(f"✓ Successful episodes: {result.successful_episodes}")
        logger.info(f"✓ Failed episodes: {result.failed_episodes}")
        logger.info(f"✓ Collection time: {result.collection_time:.2f}s")
        logger.info(f"✓ Success rate: {(result.successful_episodes/result.total_episodes)*100:.1f}%")
        logger.info(f"✓ Data saved to: {result.dataset_index_path}")
        
        if result.errors:
            logger.warning("Errors encountered:")
            for error in result.errors[:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")
        
        logger.info("Scenario 8 data collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise


if __name__ == "__main__":
    main()