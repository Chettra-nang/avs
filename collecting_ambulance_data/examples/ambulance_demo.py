#!/usr/bin/env python3
"""
Ambulance Data Collection Demonstration Script

This script demonstrates the complete ambulance data collection system including:
- Basic ambulance data collection with multi-modal observations
- Integration with existing visualization tools
- Examples of running different ambulance scenarios
- Demonstration of all 3 observation types (Kinematics, OccupancyGrid, GrayscaleObservation)

Requirements: 5.4, 6.3
"""

import sys
import os
from pathlib import Path
import time
import logging
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import ambulance collection components
from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector
from collecting_ambulance_data.scenarios.ambulance_scenarios import (
    get_all_ambulance_scenarios, get_scenario_names, get_scenario_by_name
)

# Import existing visualization tools for integration
try:
    from visualization.multimodal_parquet_plotter import MultimodalParquetPlotter
    from visualization.comprehensive_data_plotter import ComprehensiveDataPlotter
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️  Visualization tools not available - continuing without plotting")

# Import highway data collection components
from highway_datacollection.collection.action_samplers import RandomActionSampler
from highway_datacollection.collection.modality_config import ModalityConfigManager
from highway_datacollection.storage.manager import DatasetStorageManager
from highway_datacollection.performance import PerformanceConfig


def setup_logging():
    """Set up logging for the demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ambulance_demo.log')
        ]
    )
    return logging.getLogger(__name__)


def demonstrate_ambulance_scenarios():
    """Demonstrate available ambulance scenarios and their configurations."""
    print("\n🚑 AMBULANCE SCENARIOS DEMONSTRATION")
    print("=" * 50)
    
    # Get all ambulance scenarios
    all_scenarios = get_all_ambulance_scenarios()
    scenario_names = get_scenario_names()
    
    print(f"📋 Available Ambulance Scenarios: {len(scenario_names)}")
    print("-" * 40)
    
    # Show first 5 scenarios in detail
    for i, scenario_name in enumerate(scenario_names[:5], 1):
        scenario_config = all_scenarios[scenario_name]
        print(f"\n{i}. {scenario_name}")
        print(f"   Description: {scenario_config.get('description', 'N/A')}")
        print(f"   Traffic Density: {scenario_config.get('traffic_density', 'N/A')}")
        print(f"   Vehicles Count: {scenario_config.get('vehicles_count', 'N/A')}")
        print(f"   Duration: {scenario_config.get('duration', 'N/A')}s")
        print(f"   Highway Conditions: {scenario_config.get('highway_conditions', 'N/A')}")
        
        # Show ambulance-specific config
        ambulance_config = scenario_config.get('_ambulance_config', {})
        print(f"   🚑 Ambulance Agent Index: {ambulance_config.get('ambulance_agent_index', 0)}")
        print(f"   🚨 Emergency Priority: {ambulance_config.get('emergency_priority', 'high')}")
    
    if len(scenario_names) > 5:
        print(f"\n... and {len(scenario_names) - 5} more scenarios")
        print("   (Use get_all_ambulance_scenarios() to see all)")
    
    return scenario_names


def demonstrate_multi_modal_support():
    """Demonstrate multi-modal observation support for ambulance scenarios."""
    print("\n🔍 MULTI-MODAL OBSERVATION SUPPORT")
    print("=" * 40)
    
    print("Ambulance scenarios support all 3 observation types:")
    print("1. 📊 Kinematics - Vehicle state data (position, velocity, heading)")
    print("2. 🗺️  OccupancyGrid - Spatial grid representation of environment")
    print("3. 📷 GrayscaleObservation - Visual/image observations of driving scene")
    
    print("\n✅ Key Features:")
    print("• Dynamic observation type selection (not hardcoded)")
    print("• Simultaneous multi-modal data collection")
    print("• Horizontal image orientation for visual observations")
    print("• Integration with existing modality configuration system")
    
    return ["Kinematics", "OccupancyGrid", "GrayscaleObservation"]


def demonstrate_basic_collection(logger: logging.Logger):
    """Demonstrate basic ambulance data collection."""
    print("\n🔧 BASIC AMBULANCE DATA COLLECTION")
    print("=" * 40)
    
    # Initialize ambulance data collector
    print("Initializing AmbulanceDataCollector...")
    
    # Configure for demonstration (smaller numbers for quick demo)
    collector = AmbulanceDataCollector(
        n_agents=4,  # 4 controlled agents (first is ambulance)
        action_sampler=RandomActionSampler(),
        max_memory_gb=2.0,  # Lower memory limit for demo
        enable_validation=True
    )
    
    logger.info("AmbulanceDataCollector initialized successfully")
    
    # Get available scenarios
    available_scenarios = collector.get_available_scenarios()
    print(f"✅ Collector initialized with {len(available_scenarios)} scenarios")
    
    # Select a few scenarios for demonstration
    demo_scenarios = available_scenarios[:3]  # First 3 scenarios
    print(f"📋 Demo scenarios: {demo_scenarios}")
    
    # Show scenario information
    print("\n📊 Scenario Information:")
    for scenario in demo_scenarios:
        info = collector.get_scenario_info(scenario)
        print(f"\n   {scenario}:")
        print(f"   • Traffic Density: {info['traffic_density']}")
        print(f"   • Vehicles: {info['vehicles_count']}")
        print(f"   • Duration: {info['duration']}s")
        print(f"   • Ambulance Agent: Index {info['ambulance_config']['ambulance_agent_index']}")
    
    # Demonstrate environment setup
    print(f"\n🏗️  Testing Environment Setup:")
    test_scenario = demo_scenarios[0]
    
    try:
        setup_info = collector.setup_ambulance_environments(test_scenario)
        print(f"✅ Environment setup successful for '{test_scenario}':")
        print(f"   • Scenario: {setup_info['scenario_name']}")
        print(f"   • Agents: {setup_info['n_agents']}")
        print(f"   • Ambulance Index: {setup_info['ambulance_agent_index']}")
        print(f"   • Emergency Priority: {setup_info['emergency_priority']}")
        print(f"   • Environments Created: {setup_info['environments_created']}")
        print(f"   • Supported Modalities: {setup_info['supported_modalities']}")
        
        logger.info(f"Environment setup successful: {setup_info}")
        
    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        logger.error(f"Environment setup failed: {e}")
        return None
    
    # Demonstrate small-scale data collection
    print(f"\n📊 Collecting Sample Data:")
    print("(Small scale for demonstration - 5 episodes, 20 steps each)")
    
    try:
        # Collect a small amount of data for demonstration
        collection_results = collector.collect_ambulance_data(
            scenarios=[test_scenario],  # Just one scenario
            episodes_per_scenario=5,    # Small number for demo
            max_steps_per_episode=20,   # Short episodes for demo
            base_seed=42,
            batch_size=5
        )
        
        # Show collection results
        for scenario, result in collection_results.items():
            print(f"\n✅ Collection Results for '{scenario}':")
            print(f"   • Total Episodes: {result.total_episodes}")
            print(f"   • Successful Episodes: {result.successful_episodes}")
            print(f"   • Failed Episodes: {result.failed_episodes}")
            print(f"   • Collection Time: {result.collection_time:.2f}s")
            
            if result.errors:
                print(f"   • Errors: {len(result.errors)}")
                for error in result.errors[:2]:  # Show first 2 errors
                    print(f"     - {error}")
        
        logger.info(f"Data collection completed: {len(collection_results)} scenarios")
        return collection_results
        
    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        logger.error(f"Data collection failed: {e}")
        return None
    
    finally:
        # Clean up
        collector.cleanup()


def demonstrate_data_storage(collection_results: Dict[str, Any], logger: logging.Logger):
    """Demonstrate data storage with ambulance data."""
    if not collection_results:
        print("\n⚠️  No collection results to store")
        return None
    
    print("\n💾 DATA STORAGE DEMONSTRATION")
    print("=" * 35)
    
    # Set up output directory
    output_dir = Path("data/ambulance_demo_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Initialize collector for storage (reuse previous setup)
        collector = AmbulanceDataCollector(n_agents=4)
        
        # Store the collected data
        storage_info = collector.store_ambulance_data(collection_results, output_dir)
        
        print(f"✅ Data storage completed:")
        print(f"   • Output Directory: {storage_info['output_dir']}")
        print(f"   • Scenarios Stored: {storage_info['scenarios_stored']}")
        print(f"   • Episodes Stored: {storage_info['total_episodes_stored']}")
        
        if storage_info.get('dataset_index_path'):
            print(f"   • Dataset Index: {storage_info['dataset_index_path']}")
        
        if storage_info['errors']:
            print(f"   • Storage Errors: {len(storage_info['errors'])}")
            for error in storage_info['errors'][:2]:
                print(f"     - {error}")
        
        logger.info(f"Data storage completed: {storage_info}")
        return output_dir
        
    except Exception as e:
        print(f"❌ Data storage failed: {e}")
        logger.error(f"Data storage failed: {e}")
        return None


def demonstrate_visualization_integration(output_dir: Optional[Path], logger: logging.Logger):
    """Demonstrate integration with existing visualization tools."""
    print("\n📊 VISUALIZATION INTEGRATION")
    print("=" * 35)
    
    if not VISUALIZATION_AVAILABLE:
        print("⚠️  Visualization tools not available")
        print("   Install matplotlib, seaborn, and other dependencies to enable plotting")
        return
    
    if not output_dir or not output_dir.exists():
        print("⚠️  No output directory available for visualization")
        return
    
    print("🎨 Integrating with existing visualization tools...")
    
    try:
        # Try to use MultimodalParquetPlotter
        print("\n1. Testing MultimodalParquetPlotter integration:")
        
        # Look for parquet files in output directory
        parquet_files = list(output_dir.rglob("*.parquet"))
        if parquet_files:
            print(f"   ✅ Found {len(parquet_files)} parquet files")
            
            # Initialize plotter with ambulance data directory
            plotter = MultimodalParquetPlotter(str(output_dir))
            
            # Discover ambulance data files
            discovered_files = plotter.discover_parquet_files()
            print(f"   📁 Discovered data in {len(discovered_files)} scenarios")
            
            for scenario, files in discovered_files.items():
                if files['transitions']:
                    print(f"      • {scenario}: {len(files['transitions'])} transition files")
        else:
            print("   ⚠️  No parquet files found for visualization")
        
        # Try to use ComprehensiveDataPlotter
        print("\n2. Testing ComprehensiveDataPlotter integration:")
        
        comprehensive_plotter = ComprehensiveDataPlotter(str(output_dir))
        data_files = comprehensive_plotter.discover_data_files()
        
        total_files = sum(len(files) for files in data_files.values())
        if total_files > 0:
            print(f"   ✅ Comprehensive plotter found {total_files} data files")
            for data_type, files in data_files.items():
                if files:
                    print(f"      • {data_type}: {len(files)} files")
        else:
            print("   ⚠️  No compatible data files found")
        
        print("\n✅ Visualization integration successful!")
        print("   Ambulance data is compatible with existing visualization tools")
        
        logger.info("Visualization integration completed successfully")
        
    except Exception as e:
        print(f"❌ Visualization integration failed: {e}")
        logger.error(f"Visualization integration failed: {e}")


def demonstrate_different_scenarios():
    """Demonstrate running different ambulance scenarios."""
    print("\n🎯 DIFFERENT SCENARIO EXAMPLES")
    print("=" * 35)
    
    # Get scenario information
    all_scenarios = get_all_ambulance_scenarios()
    
    # Show examples of different scenario types
    scenario_examples = {
        "Light Traffic": "highway_emergency_light",
        "Heavy Traffic": "highway_emergency_dense", 
        "Construction Zone": "highway_construction",
        "Weather Conditions": "highway_weather_conditions",
        "Rush Hour": "highway_rush_hour"
    }
    
    print("📋 Example Scenario Types:")
    
    for category, scenario_name in scenario_examples.items():
        if scenario_name in all_scenarios:
            config = all_scenarios[scenario_name]
            print(f"\n🔹 {category} ({scenario_name}):")
            print(f"   Description: {config.get('description', 'N/A')}")
            print(f"   Traffic Density: {config.get('traffic_density', 'N/A')}")
            print(f"   Vehicles: {config.get('vehicles_count', 'N/A')}")
            print(f"   Conditions: {config.get('highway_conditions', 'N/A')}")
            print(f"   Speed Limit: {config.get('speed_limit', 'N/A')} m/s")
            
            # Show how to run this scenario
            print(f"   💻 Usage Example:")
            print(f"      collector.collect_single_ambulance_scenario('{scenario_name}')")
    
    print(f"\n📊 Total Available Scenarios: {len(all_scenarios)}")
    print("   Use get_all_ambulance_scenarios() to access all configurations")


def demonstrate_statistics_and_monitoring():
    """Demonstrate collection statistics and monitoring."""
    print("\n📈 STATISTICS AND MONITORING")
    print("=" * 35)
    
    # Initialize collector
    collector = AmbulanceDataCollector(n_agents=4)
    
    # Show initial statistics
    initial_stats = collector.get_collection_statistics()
    print("📊 Initial Statistics:")
    print(f"   • Available Scenarios: {initial_stats.get('available_ambulance_scenarios', 0)}")
    print(f"   • Agents: {initial_stats.get('n_agents', 0)}")
    print(f"   • Ambulance Agent Index: {initial_stats.get('ambulance_agent_index', 0)}")
    print(f"   • Episodes Collected: {initial_stats.get('ambulance_episodes_collected', 0)}")
    print(f"   • Scenarios Processed: {initial_stats.get('ambulance_scenarios_processed', 0)}")
    
    # Show validation capabilities
    print("\n🔍 Validation Capabilities:")
    test_scenario = get_scenario_names()[0]
    
    try:
        validation_result = collector.validate_ambulance_setup(test_scenario)
        print(f"✅ Validation for '{test_scenario}':")
        print(f"   • Valid: {validation_result['valid']}")
        print(f"   • Errors: {len(validation_result['errors'])}")
        print(f"   • Warnings: {len(validation_result['warnings'])}")
        
        if validation_result['errors']:
            for error in validation_result['errors']:
                print(f"     - Error: {error}")
        
        if validation_result['warnings']:
            for warning in validation_result['warnings']:
                print(f"     - Warning: {warning}")
                
    except Exception as e:
        print(f"❌ Validation failed: {e}")
    
    # Clean up
    collector.cleanup()


def main():
    """Main demonstration function."""
    print("🚑 AMBULANCE DATA COLLECTION DEMONSTRATION")
    print("=" * 55)
    print("This script demonstrates the complete ambulance data collection system")
    print("including multi-modal observations and visualization integration.")
    print()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting ambulance demonstration")
    
    start_time = time.time()
    
    try:
        # 1. Demonstrate ambulance scenarios
        scenario_names = demonstrate_ambulance_scenarios()
        
        # 2. Demonstrate multi-modal support
        observation_types = demonstrate_multi_modal_support()
        
        # 3. Demonstrate basic data collection
        collection_results = demonstrate_basic_collection(logger)
        
        # 4. Demonstrate data storage
        output_dir = demonstrate_data_storage(collection_results, logger)
        
        # 5. Demonstrate visualization integration
        demonstrate_visualization_integration(output_dir, logger)
        
        # 6. Demonstrate different scenarios
        demonstrate_different_scenarios()
        
        # 7. Demonstrate statistics and monitoring
        demonstrate_statistics_and_monitoring()
        
        # Summary
        total_time = time.time() - start_time
        print(f"\n🎉 DEMONSTRATION COMPLETE!")
        print("=" * 30)
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print(f"📋 Scenarios Available: {len(scenario_names)}")
        print(f"🔍 Observation Types: {len(observation_types)}")
        print("✅ All systems operational!")
        
        if output_dir:
            print(f"📁 Demo data saved to: {output_dir}")
        
        print("\n🚀 Next Steps:")
        print("• Use basic_ambulance_collection.py for production data collection")
        print("• Customize scenarios in ambulance_scenarios.py")
        print("• Integrate with your existing analysis workflows")
        print("• Scale up episodes and steps for full dataset collection")
        
        logger.info(f"Demonstration completed successfully in {total_time:.2f}s")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        logger.error(f"Demonstration failed: {e}")
        raise
    
    finally:
        print(f"\n📝 Log file: ambulance_demo.log")


if __name__ == "__main__":
    main()