#!/usr/bin/env python3
"""
Demonstration of comprehensive error handling and validation features.

This script shows how the error handling and validation systems work
in the highway data collection framework.
"""

import tempfile
import logging
from pathlib import Path
import numpy as np

from highway_datacollection.collection.error_handling import (
    ErrorHandler, ErrorContext, ErrorSeverity, DataCollectionError,
    EnvironmentSynchronizationError, StorageError, MemoryError,
    GracefulDegradationManager
)
from highway_datacollection.collection.validation import (
    EnvironmentSynchronizationValidator, MemoryValidator, DataIntegrityValidator,
    ValidationSeverity
)
from highway_datacollection.storage.manager import DatasetStorageManager
from highway_datacollection.collection.collector import SynchronizedCollector

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_error_handling():
    """Demonstrate basic error handling functionality."""
    print("\n=== Error Handling Demo ===")
    
    # Create error handler
    handler = ErrorHandler(max_retries=2, retry_delay=0.1)
    
    # Simulate various errors
    errors = [
        DataCollectionError("Data processing failed", ErrorSeverity.MEDIUM),
        EnvironmentSynchronizationError("Environments out of sync"),
        StorageError("Disk write failed"),
        MemoryError("Memory limit exceeded")
    ]
    
    for error in errors:
        context = ErrorContext(
            operation="demo_operation",
            component="demo_component",
            additional_info={"error_type": type(error).__name__}
        )
        
        print(f"\nHandling {type(error).__name__}: {error}")
        result = handler.handle_error(error, context)
        print(f"  Recovery attempted: {result['recovery_attempted']}")
        print(f"  Recovery successful: {result['recovery_successful']}")
    
    # Show error statistics
    stats = handler.get_error_statistics()
    print(f"\nError Statistics:")
    print(f"  Total errors: {stats['total_errors']}")
    print(f"  Error types: {list(stats['error_counts_by_type'].keys())}")
    print(f"  Recovery attempts: {stats['recovery_attempts']}")


def demo_synchronization_validation():
    """Demonstrate environment synchronization validation."""
    print("\n=== Synchronization Validation Demo ===")
    
    validator = EnvironmentSynchronizationValidator(tolerance=1e-6)
    
    # Test synchronized environments
    print("\n1. Testing synchronized environments:")
    sync_results = {
        "Kinematics": {"reward": 1.0, "terminated": False, "truncated": False},
        "OccupancyGrid": {"reward": 1.0, "terminated": False, "truncated": False},
        "GrayscaleObservation": {"reward": 1.0, "terminated": False, "truncated": False}
    }
    
    result = validator.validate_step_synchronization(sync_results)
    print(f"  Validation result: {'PASS' if result.is_valid else 'FAIL'}")
    print(f"  Issues found: {len(result.issues)}")
    
    # Test desynchronized environments
    print("\n2. Testing desynchronized environments:")
    desync_results = {
        "Kinematics": {"reward": 1.0, "terminated": False, "truncated": False},
        "OccupancyGrid": {"reward": 2.0, "terminated": False, "truncated": False},  # Different reward
        "GrayscaleObservation": {"reward": 1.0, "terminated": True, "truncated": False}  # Different termination
    }
    
    result = validator.validate_step_synchronization(desync_results)
    print(f"  Validation result: {'PASS' if result.is_valid else 'FAIL'}")
    print(f"  Issues found: {len(result.issues)}")
    for issue in result.issues:
        print(f"    - {issue.severity.value.upper()}: {issue.message}")
    
    # Show synchronization statistics
    stats = validator.get_synchronization_stats()
    print(f"\nSynchronization Statistics:")
    print(f"  Total checks: {stats['total_checks']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")


def demo_memory_validation():
    """Demonstrate memory usage validation."""
    print("\n=== Memory Validation Demo ===")
    
    # Create validator with low memory limit for demonstration
    validator = MemoryValidator(max_memory_gb=0.1, warning_threshold=0.5)  # Very low limit
    
    print("Testing memory validation with low limits...")
    result = validator.validate_memory_usage()
    
    print(f"Validation result: {'PASS' if result.is_valid else 'FAIL'}")
    print(f"Issues found: {len(result.issues)}")
    
    for issue in result.issues:
        print(f"  - {issue.severity.value.upper()}: {issue.message}")
    
    # Show memory statistics
    stats = validator.get_memory_stats()
    if "no_data" not in stats:
        print(f"\nMemory Statistics:")
        print(f"  Current memory: {stats['current_memory_gb']:.3f} GB")
        print(f"  Memory limit: {stats['memory_limit_gb']:.3f} GB")
        print(f"  Usage percentage: {stats['usage_percentage']:.1%}")


def demo_data_integrity_validation():
    """Demonstrate data integrity validation."""
    print("\n=== Data Integrity Validation Demo ===")
    
    validator = DataIntegrityValidator()
    
    # Test valid observation data
    print("\n1. Testing valid observation data:")
    valid_observations = [
        {"episode_id": "ep1", "step": 0, "agent_id": 0, "reward": 1.0},
        {"episode_id": "ep1", "step": 0, "agent_id": 1, "reward": 1.0},
        {"episode_id": "ep1", "step": 1, "agent_id": 0, "reward": 2.0},
        {"episode_id": "ep1", "step": 1, "agent_id": 1, "reward": 2.0},
    ]
    
    result = validator.validate_observation_data(valid_observations)
    print(f"  Validation result: {'PASS' if result.is_valid else 'FAIL'}")
    print(f"  Issues found: {len(result.issues)}")
    
    # Test invalid observation data
    print("\n2. Testing invalid observation data:")
    invalid_observations = [
        {"episode_id": "ep1", "step": 0, "agent_id": 0},  # Missing reward
        {"step": 0, "agent_id": 1, "reward": 1.0},        # Missing episode_id
        {"episode_id": "ep1", "step": 0, "agent_id": 0, "reward": 1.0},  # Duplicate
    ]
    
    result = validator.validate_observation_data(invalid_observations)
    print(f"  Validation result: {'PASS' if result.is_valid else 'FAIL'}")
    print(f"  Issues found: {len(result.issues)}")
    for issue in result.issues:
        print(f"    - {issue.severity.value.upper()}: {issue.message}")
    
    # Test binary data validation
    print("\n3. Testing binary data validation:")
    test_array = np.random.rand(10, 10).astype(np.float32)
    valid_binary_data = {
        "test_blob": test_array.tobytes(),
        "test_shape": [10, 10],
        "test_dtype": "float32"
    }
    
    result = validator.validate_binary_data(valid_binary_data, ["test"])
    print(f"  Validation result: {'PASS' if result.is_valid else 'FAIL'}")
    print(f"  Issues found: {len(result.issues)}")


def demo_graceful_degradation():
    """Demonstrate graceful degradation functionality."""
    print("\n=== Graceful Degradation Demo ===")
    
    manager = GracefulDegradationManager()
    
    # Register fallback strategies
    def primary_function(x):
        return x * 2
    
    def fallback_function(x):
        return x + 1  # Simpler fallback
    
    manager.register_fallback("math_operation", fallback_function, "Simple addition fallback")
    
    # Test normal operation
    print("\n1. Testing normal operation:")
    result = manager.execute_with_fallback("math_operation", primary_function, 5)
    print(f"  Result: {result} (expected: 10)")
    
    # Degrade feature and test fallback
    print("\n2. Testing degraded operation with fallback:")
    manager.degrade_feature("math_operation", "Primary function failed", ErrorSeverity.MEDIUM)
    result = manager.execute_with_fallback("math_operation", primary_function, 5)
    print(f"  Result: {result} (expected: 6, using fallback)")
    
    # Show degradation status
    status = manager.get_degradation_status()
    print(f"\nDegradation Status:")
    print(f"  Degraded features: {status['total_degraded']}")
    print(f"  Available fallbacks: {len(status['available_fallbacks'])}")


def demo_enhanced_storage():
    """Demonstrate enhanced storage with error handling."""
    print("\n=== Enhanced Storage Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = DatasetStorageManager(Path(temp_dir), max_disk_usage_gb=1.0)
        
        # Test successful storage
        print("\n1. Testing successful storage:")
        test_data = [
            {"episode_id": "ep1", "step": 0, "agent_id": 0, "reward": 1.0},
            {"episode_id": "ep1", "step": 1, "agent_id": 0, "reward": 2.0}
        ]
        test_metadata = [
            {"episode_id": "ep1", "scenario": "test", "total_steps": 2}
        ]
        
        try:
            result = manager.write_episode_batch(test_data, test_metadata, "test_scenario")
            print(f"  Storage successful: {result.transitions_file.exists()}")
        except Exception as e:
            print(f"  Storage failed: {e}")
        
        # Show storage statistics
        stats = manager.get_storage_statistics()
        print(f"\nStorage Statistics:")
        print(f"  Files written: {stats['files_written']}")
        print(f"  Bytes written: {stats['bytes_written']}")
        print(f"  Parquet failures: {stats['parquet_failures']}")
        print(f"  CSV fallbacks: {stats['csv_fallbacks']}")


def demo_enhanced_collector():
    """Demonstrate enhanced collector with error handling."""
    print("\n=== Enhanced Collector Demo ===")
    
    # Create collector with validation enabled
    collector = SynchronizedCollector(
        n_agents=2,
        enable_validation=True,
        max_memory_gb=4.0
    )
    
    print(f"Collector created with validation: {collector.enable_validation}")
    
    # Test synchronization verification
    print("\n1. Testing synchronization verification:")
    sync_results = {
        "Kinematics": {"reward": 1.0, "terminated": False, "truncated": False},
        "OccupancyGrid": {"reward": 1.0, "terminated": False, "truncated": False}
    }
    
    is_synchronized = collector.verify_synchronization(sync_results)
    print(f"  Synchronization check: {'PASS' if is_synchronized else 'FAIL'}")
    
    # Test health check
    print("\n2. Testing health check:")
    health_status = collector.perform_health_check()
    print(f"  Overall healthy: {health_status['overall_healthy']}")
    print(f"  Environments ready: {health_status['environments_ready']}")
    print(f"  Issues found: {len(health_status['issues'])}")
    
    # Show collection statistics
    stats = collector.get_collection_statistics()
    print(f"\nCollection Statistics:")
    print(f"  Episodes collected: {stats['episodes_collected']}")
    print(f"  Sync failures: {stats['sync_failures']}")
    print(f"  Memory warnings: {stats['memory_warnings']}")


def main():
    """Run all demonstrations."""
    print("Highway Data Collection - Error Handling & Validation Demo")
    print("=" * 60)
    
    try:
        demo_error_handling()
        demo_synchronization_validation()
        demo_memory_validation()
        demo_data_integrity_validation()
        demo_graceful_degradation()
        demo_enhanced_storage()
        demo_enhanced_collector()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()