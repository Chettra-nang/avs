#!/usr/bin/env python3
"""
Demonstration of data validation and quality assurance utilities.

This script shows how to use the comprehensive data quality assurance pipeline
to validate multi-modal data collection, binary array integrity, and feature
derivation accuracy.
"""

import numpy as np
import logging
from typing import Dict, Any, List

from highway_datacollection.collection.validation import (
    DataQualityAssurancePipeline,
    TrajectoryComparisonValidator,
    BinaryArrayIntegrityValidator,
    FeatureDerivationValidator,
    ValidationSeverity
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_modality_data() -> Dict[str, List[Dict[str, Any]]]:
    """
    Create sample multi-modal data for demonstration.
    
    Returns:
        Dictionary mapping modality names to observation lists
    """
    episodes = ['episode_001', 'episode_002']
    agents = [0, 1]
    steps = list(range(10))  # 10 steps per episode
    
    modality_data = {}
    
    for modality in ['kinematics', 'occupancy_grid', 'grayscale']:
        modality_data[modality] = []
        
        for episode in episodes:
            for agent in agents:
                for step in steps:
                    # Create realistic trajectory data
                    x_pos = step * 8.0 + agent * 3.0  # Forward movement with agent offset
                    y_pos = agent * 4.0  # Different lanes for different agents
                    velocity = 8.0 + np.random.normal(0, 0.5)  # Slight velocity variation
                    
                    obs = {
                        'episode_id': episode,
                        'agent_id': agent,
                        'step': step,
                        'kin_x': x_pos,
                        'kin_y': y_pos,
                        'kin_vx': velocity,
                        'kin_vy': 0.0,
                        'reward': 1.0,
                        'terminated': step == (len(steps) - 1),  # Terminate at last step
                        'truncated': False
                    }
                    modality_data[modality].append(obs)
    
    return modality_data


def create_sample_binary_data() -> Dict[str, Any]:
    """
    Create sample binary array data for testing reconstruction.
    
    Returns:
        Dictionary with original arrays and encoded data
    """
    # Create sample arrays similar to what would be collected
    occupancy_grid = np.random.rand(64, 64).astype(np.float32)
    grayscale_image = np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)
    
    original_arrays = {
        'occupancy': occupancy_grid,
        'grayscale': grayscale_image
    }
    
    # Encode arrays as binary data (simulating storage encoding)
    encoded_data = {}
    for name, array in original_arrays.items():
        encoded_data[f"{name}_blob"] = array.tobytes()
        encoded_data[f"{name}_shape"] = list(array.shape)
        encoded_data[f"{name}_dtype"] = str(array.dtype)
    
    return {
        'original_arrays': original_arrays,
        'encoded_data': encoded_data
    }


def create_sample_step_results() -> Dict[str, Any]:
    """
    Create sample step results for environment synchronization testing.
    
    Returns:
        Dictionary mapping modality names to step results
    """
    step_results = {}
    
    for modality in ['kinematics', 'occupancy_grid', 'grayscale']:
        step_results[modality] = {
            'reward': 1.0,
            'terminated': False,
            'truncated': False
        }
    
    return step_results


def demonstrate_individual_validators():
    """Demonstrate individual validator components."""
    logger.info("=== Demonstrating Individual Validators ===")
    
    # 1. Trajectory Comparison Validator
    logger.info("1. Testing Trajectory Comparison Validator")
    trajectory_validator = TrajectoryComparisonValidator(
        position_tolerance=1e-4,
        velocity_tolerance=1e-4
    )
    
    modality_data = create_sample_modality_data()
    result = trajectory_validator.validate_trajectory_synchronization(modality_data)
    
    logger.info(f"   Trajectory sync validation: {'PASS' if result.is_valid else 'FAIL'}")
    if result.issues:
        logger.info(f"   Issues found: {len(result.issues)}")
        for issue in result.issues[:3]:  # Show first 3 issues
            logger.info(f"     - {issue.severity.value}: {issue.message}")
    
    # 2. Binary Array Integrity Validator
    logger.info("2. Testing Binary Array Integrity Validator")
    binary_validator = BinaryArrayIntegrityValidator(tolerance=1e-6)
    
    binary_data = create_sample_binary_data()
    result = binary_validator.validate_binary_reconstruction(
        binary_data['original_arrays'],
        binary_data['encoded_data']
    )
    
    logger.info(f"   Binary integrity validation: {'PASS' if result.is_valid else 'FAIL'}")
    if result.issues:
        logger.info(f"   Issues found: {len(result.issues)}")
        for issue in result.issues:
            logger.info(f"     - {issue.severity.value}: {issue.message}")
    
    # 3. Feature Derivation Validator
    logger.info("3. Testing Feature Derivation Validator")
    feature_validator = FeatureDerivationValidator()
    
    result = feature_validator.validate_feature_derivation_accuracy()
    
    logger.info(f"   Feature derivation validation: {'PASS' if result.is_valid else 'FAIL'}")
    if result.issues:
        logger.info(f"   Issues found: {len(result.issues)}")
        for issue in result.issues:
            logger.info(f"     - {issue.severity.value}: {issue.message}")


def demonstrate_full_pipeline():
    """Demonstrate the complete data quality assurance pipeline."""
    logger.info("\n=== Demonstrating Full QA Pipeline ===")
    
    # Initialize pipeline with custom configuration
    config = {
        'position_tolerance': 1e-4,
        'velocity_tolerance': 1e-4,
        'binary_tolerance': 1e-6,
        'sync_tolerance': 1e-6,
        'max_memory_gb': 8.0,
        'memory_warning_threshold': 0.8
    }
    
    pipeline = DataQualityAssurancePipeline(config)
    
    # Create test data
    modality_data = create_sample_modality_data()
    binary_test_data = create_sample_binary_data()
    step_results = create_sample_step_results()
    
    logger.info("Running comprehensive quality assurance...")
    
    # Run full validation
    results = pipeline.run_full_quality_assurance(
        modality_data=modality_data,
        binary_test_data=binary_test_data,
        step_results=step_results
    )
    
    # Generate and display report
    report = pipeline.generate_quality_report(results)
    
    logger.info(f"\n=== Quality Assurance Report ===")
    logger.info(f"Overall Status: {report['overall_status']}")
    logger.info(f"Timestamp: {report['timestamp']}")
    
    # Summary statistics
    summary = report['summary']
    logger.info(f"\nSummary:")
    logger.info(f"  Total Validators: {summary['total_validators']}")
    logger.info(f"  Passed: {summary['passed']}")
    logger.info(f"  Failed: {summary['failed']}")
    logger.info(f"  Warnings: {summary['warnings']}")
    logger.info(f"  Errors: {summary['errors']}")
    logger.info(f"  Critical: {summary['critical']}")
    
    # Individual validator results
    logger.info(f"\nValidator Results:")
    for validator_name, validator_result in report['validator_results'].items():
        status = validator_result['status']
        issues_count = validator_result['issues_count']
        logger.info(f"  {validator_name}: {status} ({issues_count} issues)")
        
        # Show first few issues if any
        if issues_count > 0:
            for issue in validator_result['issues'][:2]:  # Show first 2 issues
                logger.info(f"    - {issue['severity']}: {issue['message']}")
    
    # Recommendations
    if report['recommendations']:
        logger.info(f"\nRecommendations:")
        for rec in report['recommendations']:
            logger.info(f"  • {rec}")
    
    return report


def demonstrate_corrupted_data_detection():
    """Demonstrate detection of various data corruption scenarios."""
    logger.info("\n=== Demonstrating Corrupted Data Detection ===")
    
    pipeline = DataQualityAssurancePipeline()
    
    # Create corrupted modality data
    corrupted_modality_data = {
        'kinematics': [
            {
                'episode_id': 'ep1', 'agent_id': 0, 'step': 0,
                'kin_x': 0.0, 'kin_y': 0.0, 'kin_vx': 10.0, 'kin_vy': 0.0,
                'reward': 1.0, 'terminated': False, 'truncated': False
            }
        ],
        'occupancy_grid': [
            {
                'episode_id': 'ep1', 'agent_id': 0, 'step': 0,
                'kin_x': 100.0, 'kin_y': 50.0,  # Corrupted position data
                'kin_vx': 10.0, 'kin_vy': 0.0,
                'reward': 2.0,  # Corrupted reward
                'terminated': True,  # Corrupted termination
                'truncated': False
            }
        ]
    }
    
    # Create corrupted binary data
    original_array = np.random.rand(64, 64).astype(np.float32)
    corrupted_binary_data = {
        'original_arrays': {'test': original_array},
        'encoded_data': {
            'test_blob': b'corrupted_data_that_is_wrong_size',  # Wrong size
            'test_shape': [64, 64],
            'test_dtype': 'float32'
        }
    }
    
    logger.info("Running validation on corrupted data...")
    
    # Run validation on corrupted data
    results = pipeline.run_full_quality_assurance(
        modality_data=corrupted_modality_data,
        binary_test_data=corrupted_binary_data
    )
    
    # Generate report
    report = pipeline.generate_quality_report(results)
    
    logger.info(f"Corruption Detection Results:")
    logger.info(f"  Overall Status: {report['overall_status']}")
    logger.info(f"  Failed Validators: {report['summary']['failed']}")
    logger.info(f"  Total Errors: {report['summary']['errors']}")
    
    # Show detected corruption issues
    logger.info(f"\nDetected Corruption Issues:")
    for validator_name, validator_result in report['validator_results'].items():
        if validator_result['status'] == 'FAIL':
            logger.info(f"  {validator_name}:")
            for issue in validator_result['issues'][:3]:  # Show first 3 issues
                logger.info(f"    - {issue['message']}")


def main():
    """Main demonstration function."""
    logger.info("Highway Multi-Modal Data Collection - Quality Assurance Demo")
    logger.info("=" * 60)
    
    try:
        # Demonstrate individual validators
        demonstrate_individual_validators()
        
        # Demonstrate full pipeline
        report = demonstrate_full_pipeline()
        
        # Demonstrate corruption detection
        demonstrate_corrupted_data_detection()
        
        logger.info("\n" + "=" * 60)
        logger.info("Demo completed successfully!")
        
        if report['overall_status'] == 'PASS':
            logger.info("✅ All quality assurance checks passed!")
        else:
            logger.info("⚠️  Some quality assurance checks failed - review the report above")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()