#!/usr/bin/env python3
"""
Test Scenario 1 with just 2 episodes to see what happens
"""

import sys
import logging
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from highway_datacollection.collection.orchestrator import run_full_collection
from highway_datacollection.collection.modality_config import ModalityConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Test data collection for scenario 1."""
    logger.info("Testing Scenario 1: Light Traffic Free Flow data collection (2 episodes)")
    
    # Storage configuration - use absolute path to main data directory
    storage_path = Path("/home/chettra/ITC/Research/AVs/data/scenario_01_light_free_flow")
    storage_path.mkdir(parents=True, exist_ok=True)
    
    # Collection parameters
    collection_config = {
        "base_storage_path": storage_path,
        "episodes_per_scenario": 2,  # Just 2 episodes for testing
        "n_agents": 2,
        "max_steps_per_episode": 50,  # Shorter episodes
        "scenarios": ["free_flow"],
        "base_seed": 1001,
        "batch_size": 1
    }
    
    # Configure complete dataset collection  
    modality_manager = ModalityConfigManager()
    
    logger.info(f"Test configuration:")
    logger.info(f"  - Episodes: {collection_config['episodes_per_scenario']}")
    logger.info(f"  - Scenario: Light Free Flow")
    logger.info(f"  - Storage: {storage_path}")
    logger.info(f"  - Batch size: {collection_config['batch_size']}")
    
    try:
        # Run data collection
        result = run_full_collection(modality_config_manager=modality_manager, **collection_config)
        
        # Report results
        logger.info("=" * 60)
        logger.info("TEST COLLECTION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"✓ Total episodes: {result.total_episodes}")
        logger.info(f"✓ Successful episodes: {result.successful_episodes}")
        logger.info(f"✓ Failed episodes: {result.failed_episodes}")
        logger.info(f"✓ Collection time: {result.collection_time:.2f}s")
        logger.info(f"✓ Success rate: {(result.successful_episodes/result.total_episodes)*100:.1f}%")
        
        # Check what files were created
        if storage_path.exists():
            files = list(storage_path.rglob("*"))
            logger.info(f"✓ Files created: {len(files)}")
            for file in files[:10]:  # Show first 10 files
                logger.info(f"  - {file}")
        
        if result.errors:
            logger.warning("Errors encountered:")
            for error in result.errors[:3]:  # Show first 3 errors
                logger.warning(f"  - {error}")
        
        logger.info("Test completed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()