#!/usr/bin/env python3
"""
Test script for AmbulanceDataCollector functionality.

This script demonstrates the basic functionality of the AmbulanceDataCollector
and validates that it can properly set up ambulance environments and collect data.
"""

import logging
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_ambulance_collector_initialization():
    """Test basic initialization of AmbulanceDataCollector."""
    logger.info("Testing AmbulanceDataCollector initialization...")
    
    try:
        collector = AmbulanceDataCollector(n_agents=4)
        logger.info(f"✓ Successfully initialized AmbulanceDataCollector with {collector.n_agents} agents")
        
        # Test getting available scenarios
        scenarios = collector.get_available_scenarios()
        logger.info(f"✓ Found {len(scenarios)} available ambulance scenarios")
        
        # Test getting scenario info
        if scenarios:
            first_scenario = scenarios[0]
            scenario_info = collector.get_scenario_info(first_scenario)
            logger.info(f"✓ Successfully retrieved info for scenario: {first_scenario}")
            logger.info(f"  Description: {scenario_info['description']}")
            logger.info(f"  Traffic density: {scenario_info['traffic_density']}")
            logger.info(f"  Vehicles count: {scenario_info['vehicles_count']}")
        
        # Test validation
        if scenarios:
            validation_result = collector.validate_ambulance_setup(scenarios[0])
            logger.info(f"✓ Validation result for {scenarios[0]}: {validation_result['valid']}")
        
        # Test statistics
        stats = collector.get_collection_statistics()
        logger.info(f"✓ Retrieved collection statistics: {len(stats)} metrics")
        
        collector.cleanup()
        logger.info("✓ AmbulanceDataCollector initialization test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ AmbulanceDataCollector initialization test failed: {e}")
        return False


def test_ambulance_environment_setup():
    """Test ambulance environment setup."""
    logger.info("Testing ambulance environment setup...")
    
    try:
        collector = AmbulanceDataCollector(n_agents=4)
        scenarios = collector.get_available_scenarios()
        
        if not scenarios:
            logger.warning("No ambulance scenarios available for testing")
            return True
        
        # Test environment setup for the first scenario
        test_scenario = scenarios[0]
        logger.info(f"Testing environment setup for scenario: {test_scenario}")
        
        setup_info = collector.setup_ambulance_environments(test_scenario)
        logger.info(f"✓ Successfully set up ambulance environments")
        logger.info(f"  Scenario: {setup_info['scenario_name']}")
        logger.info(f"  Agents: {setup_info['n_agents']}")
        logger.info(f"  Ambulance agent index: {setup_info['ambulance_agent_index']}")
        logger.info(f"  Emergency priority: {setup_info['emergency_priority']}")
        logger.info(f"  Environments created: {setup_info['environments_created']}")
        logger.info(f"  Supported modalities: {setup_info['supported_modalities']}")
        
        collector.cleanup()
        logger.info("✓ Ambulance environment setup test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Ambulance environment setup test failed: {e}")
        return False


def test_ambulance_data_collection_dry_run():
    """Test ambulance data collection with minimal episodes."""
    logger.info("Testing ambulance data collection (dry run)...")
    
    try:
        collector = AmbulanceDataCollector(n_agents=4)
        scenarios = collector.get_available_scenarios()
        
        if not scenarios:
            logger.warning("No ambulance scenarios available for testing")
            return True
        
        # Test with just one scenario and minimal episodes
        test_scenario = scenarios[0]
        logger.info(f"Testing data collection for scenario: {test_scenario}")
        
        # Collect just 1 episode with 5 steps for testing
        result = collector.collect_single_ambulance_scenario(
            scenario_name=test_scenario,
            episodes=1,
            max_steps=5,
            seed=42,
            batch_size=1
        )
        
        logger.info(f"✓ Data collection completed")
        logger.info(f"  Total episodes: {result.total_episodes}")
        logger.info(f"  Successful episodes: {result.successful_episodes}")
        logger.info(f"  Failed episodes: {result.failed_episodes}")
        logger.info(f"  Collection time: {result.collection_time:.2f}s")
        
        if result.successful_episodes > 0:
            logger.info(f"  Episode data collected: {len(result.episodes)} episodes")
            if result.episodes:
                episode = result.episodes[0]
                logger.info(f"  Sample episode ID: {episode.episode_id}")
                logger.info(f"  Sample episode steps: {len(episode.observations)}")
        
        # Test statistics after collection
        stats = collector.get_collection_statistics()
        logger.info(f"✓ Updated statistics - ambulance episodes collected: {stats['ambulance_episodes_collected']}")
        
        collector.cleanup()
        logger.info("✓ Ambulance data collection test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Ambulance data collection test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting AmbulanceDataCollector tests...")
    
    tests = [
        test_ambulance_collector_initialization,
        test_ambulance_environment_setup,
        test_ambulance_data_collection_dry_run
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
            failed += 1
        
        logger.info("-" * 50)
    
    logger.info(f"Test results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All AmbulanceDataCollector tests passed!")
        return 0
    else:
        logger.error(f"❌ {failed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())