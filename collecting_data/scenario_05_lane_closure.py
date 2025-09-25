#!/usr/bin/env python3
"""    logger.info("Starting Scenario 5: Highway Lane Closure data collection")
    
    # Storage configuration
    storage_path = Path("/home/chettra/ITC/Research/AVs/data/scenario_05_lane_closure")
    storage_path.mkdir(parents=True, exist_ok=True)ario 5: Lane Closure Merging
Complex merging scenario due to lane closures requiring coordination.
Collects complete multimodal dataset with 1000 episodes for merging behavior analysis.
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
    """Collect data for lane closure merging scenario."""
    logger.info("Starting Scenario 5: Lane Closure Merging data collection")
    
    # Storage configuration
    storage_path = Path("data/scenario_05_lane_closure")
    storage_path.mkdir(parents=True, exist_ok=True)
    
    # Collection parameters
    collection_config = {
        "base_storage_path": storage_path,
        "episodes_per_scenario": 1000,
        "n_agents": 2,
        "max_steps_per_episode": 160,
        "scenarios": ["lane_closure"],
        "base_seed": 5001,
        "batch_size": 18
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
    logger.info(f"  - Scenario: Lane Closure (55 vehicles, 3 lanes)")
    logger.info(f"  - Modalities: Kinematics, Occupancy Grid, Grayscale")
    logger.info(f"  - Storage: {storage_path}")
    logger.info(f"  - Batch size: {collection_config['batch_size']}")
    
    try:
        # Run data collection
        result = run_full_collection(modality_config_manager=modality_manager, **collection_config)
        
        # Report results
        logger.info("=" * 60)
        logger.info("COLLECTION COMPLETED - SCENARIO 5")
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
        
        logger.info("Scenario 5 data collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise


if __name__ == "__main__":
    main()