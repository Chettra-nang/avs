#!/usr/bin/env python3
"""
Quick Test: Minimal data collection test 
Tests 2 episodes for validation that the system works correctly.
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
    """Quick test data collection."""
    logger.info("Starting Quick Test: 2 episodes for system validation")
    
    # Storage configuration
    storage_path = Path("data/quick_validation_test")
    storage_path.mkdir(parents=True, exist_ok=True)
    
    # Collection parameters  
    collection_config = {
        "base_storage_path": storage_path,
        "episodes_per_scenario": 2,  # Just 2 episodes for quick test
        "n_agents": 2,
        "max_steps_per_episode": 30,  # Short episodes
        "scenarios": ["free_flow"],
        "base_seed": 1001,
        "batch_size": 1
    }
    
    # Configure complete dataset collection  
    modality_manager = ModalityConfigManager()
    
    logger.info(f"Quick Test Configuration:")
    logger.info(f"  - Episodes: {collection_config['episodes_per_scenario']}")
    logger.info(f"  - Scenario: Free Flow")
    logger.info(f"  - Max steps: {collection_config['max_steps_per_episode']}")
    logger.info(f"  - Storage: {storage_path}")
    
    try:
        # Run data collection
        result = run_full_collection(modality_config_manager=modality_manager, **collection_config)
        
        # Report results
        logger.info("=" * 60)
        logger.info("QUICK TEST COMPLETED")
        logger.info("=" * 60)
        logger.info(f"✓ Total episodes: {result.total_episodes}")
        logger.info(f"✓ Successful episodes: {result.successful_episodes}")
        logger.info(f"✓ Failed episodes: {result.failed_episodes}")
        logger.info(f"✓ Collection time: {result.collection_time:.2f}s")
        logger.info(f"✓ Success rate: {(result.successful_episodes/result.total_episodes)*100:.1f}%")
        
        if result.errors:
            logger.warning("Errors encountered:")
            for error in result.errors[:3]:  # Show first 3 errors
                logger.warning(f"  - {error}")
        
        if result.successful_episodes == result.total_episodes:
            logger.info("🎉 Quick test passed! System is working correctly!")
            logger.info("Ready to run full data collection with 10,000 episodes!")
        else:
            logger.warning("⚠️  Some episodes failed. Check logs before running full collection.")
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()