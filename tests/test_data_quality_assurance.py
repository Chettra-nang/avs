"""
Integration tests for data quality assurance pipeline.
"""

import pytest
import numpy as np
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from highway_datacollection.collection.validation import (
    TrajectoryComparisonValidator, BinaryArrayIntegrityValidator,
    FeatureDerivationValidator, DataQualityAssurancePipeline,
    ValidationSeverity, ValidationResult
)


class TestTrajectoryComparisonValidator:
    """Test trajectory comparison validation."""
    
    def test_synchronized_trajectories(self):
        """Test validation of properly synchronized trajectories."""
        validator = TrajectoryComparisonValidator()
        
        # Create synchronized trajectory data
        modality_data = {
            'kinematics': [
                {
                    'episode_id': 'ep1', 'agent_id': 0, 'step': 0,
                    'kin_x': 0.0, 'kin_y': 0.0, 'kin_vx': 10.0, 'kin_vy': 0.0,
                    'reward': 1.0, 'terminated': False, 'truncated': False
                },
                {
                    'episode_id': 'ep1', 'agent_id': 0, 'step': 1,
                    'kin_x': 10.0, 'kin_y': 0.0, 'kin_vx': 10.0, 'kin_vy': 0.0,
                    'reward': 1.0, 'terminated': False, 'truncated': False
                }
            ],
            'occupancy': [
                {
                    'episode_id': 'ep1', 'agent_id': 0, 'step': 0,
                    'kin_x': 0.0, 'kin_y': 0.0, 'kin_vx': 10.0, 'kin_vy': 0.0,
                    'reward': 1.0, 'terminated': False, 'truncated': False
                },
                {
                    'episode_id': 'ep1', 'agent_id': 0, 'step': 1,
                    'kin_x': 10.0, 'kin_y': 0.0, 'kin_vx': 10.0, 'kin_vy': 0.0,
                    'reward': 1.0, 'terminated': False, 'truncated': False
                }
            ]
        }
        
        result = validator.validate_trajectory_synchronization(modality_data)
        
        assert result.is_valid
        assert len(result.issues) == 0
    
    def test_desynchronized_positions(self):
        """Test detection of position desynchronization."""
        validator = TrajectoryComparisonValidator(position_tolerance=1e-4)
        
        # Create desynchronized trajectory data
        modality_data = {
            'kinematics': [
                {
                    'episode_id': 'ep1', 'agent_id': 0, 'step': 0,
                    'kin_x': 0.0, 'kin_y': 0.0, 'kin_vx': 10.0, 'kin_vy': 0.0,
                    'reward': 1.0, 'terminated': False, 'truncated': False
                }
            ],
            'occupancy': [
                {
                    'episode_id': 'ep1', 'agent_id': 0, 'step': 0,
                    'kin_x': 0.1, 'kin_y': 0.0, 'kin_vx': 10.0, 'kin_vy': 0.0,  # Different x position
                    'reward': 1.0, 'terminated': False, 'truncated': False
                }
            ]
        }
        
        result = validator.validate_trajectory_synchronization(modality_data)
        
        assert not result.is_valid
        assert len(result.issues) > 0
        assert any("Position desynchronization" in issue.message for issue in result.issues)
    
    def test_missing_episodes(self):
        """Test detection of missing episodes in modalities."""
        validator = TrajectoryComparisonValidator()
        
        modality_data = {
            'kinematics': [
                {
                    'episode_id': 'ep1', 'agent_id': 0, 'step': 0,
                    'kin_x': 0.0, 'kin_y': 0.0, 'kin_vx': 10.0, 'kin_vy': 0.0,
                    'reward': 1.0, 'terminated': False, 'truncated': False
                }
            ],
            'occupancy': [
                {
                    'episode_id': 'ep2', 'agent_id': 0, 'step': 0,  # Different episode
                    'kin_x': 0.0, 'kin_y': 0.0, 'kin_vx': 10.0, 'kin_vy': 0.0,
                    'reward': 1.0, 'terminated': False, 'truncated': False
                }
            ]
        }
        
        result = validator.validate_trajectory_synchronization(modality_data)
        
        assert not result.is_valid
        assert any("missing in" in issue.message for issue in result.issues)
    
    def test_reward_desynchronization(self):
        """Test detection of reward desynchronization."""
        validator = TrajectoryComparisonValidator()
        
        modality_data = {
            'kinematics': [
                {
                    'episode_id': 'ep1', 'agent_id': 0, 'step': 0,
                    'kin_x': 0.0, 'kin_y': 0.0, 'kin_vx': 10.0, 'kin_vy': 0.0,
                    'reward': 1.0, 'terminated': False, 'truncated': False
                }
            ],
            'occupancy': [
                {
                    'episode_id': 'ep1', 'agent_id': 0, 'step': 0,
                    'kin_x': 0.0, 'kin_y': 0.0, 'kin_vx': 10.0, 'kin_vy': 0.0,
                    'reward': 2.0, 'terminated': False, 'truncated': False  # Different reward
                }
            ]
        }
        
        result = validator.validate_trajectory_synchronization(modality_data)
        
        assert not result.is_valid
        assert any("Reward desynchronization" in issue.message for issue in result.issues)


class TestBinaryArrayIntegrityValidator:
    """Test binary array integrity validation."""
    
    def test_successful_reconstruction(self):
        """Test successful binary array reconstruction."""
        validator = BinaryArrayIntegrityValidator()
        
        # Create test arrays
        original_arrays = {
            'occupancy': np.random.rand(64, 64).astype(np.float32),
            'grayscale': np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)
        }
        
        # Simulate encoding
        encoded_data = {}
        for name, array in original_arrays.items():
            encoded_data[f"{name}_blob"] = array.tobytes()
            encoded_data[f"{name}_shape"] = list(array.shape)
            encoded_data[f"{name}_dtype"] = str(array.dtype)
        
        result = validator.validate_binary_reconstruction(original_arrays, encoded_data)
        
        assert result.is_valid
        assert len(result.issues) == 0
    
    def test_shape_mismatch(self):
        """Test detection of shape mismatch in reconstruction."""
        validator = BinaryArrayIntegrityValidator()
        
        original_array = np.random.rand(64, 64).astype(np.float32)
        original_arrays = {'test': original_array}
        
        # Create encoded data with wrong shape
        encoded_data = {
            'test_blob': original_array.tobytes(),
            'test_shape': [32, 32],  # Wrong shape
            'test_dtype': str(original_array.dtype)
        }
        
        result = validator.validate_binary_reconstruction(original_arrays, encoded_data)
        
        assert not result.is_valid
        assert any("size mismatch" in issue.message for issue in result.issues)
    
    def test_dtype_mismatch(self):
        """Test detection of dtype mismatch in reconstruction."""
        validator = BinaryArrayIntegrityValidator()
        
        original_array = np.random.rand(64, 64).astype(np.float32)
        original_arrays = {'test': original_array}
        
        # Create encoded data with wrong dtype
        encoded_data = {
            'test_blob': original_array.tobytes(),
            'test_shape': list(original_array.shape),
            'test_dtype': 'float64'  # Wrong dtype
        }
        
        result = validator.validate_binary_reconstruction(original_arrays, encoded_data)
        
        assert not result.is_valid
        assert any("size mismatch" in issue.message for issue in result.issues)
    
    def test_missing_encoded_data(self):
        """Test detection of missing encoded data."""
        validator = BinaryArrayIntegrityValidator()
        
        original_arrays = {'test': np.random.rand(10, 10)}
        encoded_data = {}  # Missing all encoded data
        
        result = validator.validate_binary_reconstruction(original_arrays, encoded_data)
        
        assert not result.is_valid
        assert any("Missing encoded data" in issue.message for issue in result.issues)
    
    def test_floating_point_tolerance(self):
        """Test floating point tolerance in reconstruction validation."""
        validator = BinaryArrayIntegrityValidator(tolerance=1e-6)
        
        original_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        original_arrays = {'test': original_array}
        
        # Create slightly modified array within tolerance
        modified_array = original_array + 1e-7  # Within tolerance
        encoded_data = {
            'test_blob': modified_array.tobytes(),
            'test_shape': list(original_array.shape),
            'test_dtype': str(original_array.dtype)
        }
        
        result = validator.validate_binary_reconstruction(original_arrays, encoded_data)
        
        assert result.is_valid
        assert len(result.issues) == 0


class TestFeatureDerivationValidator:
    """Test feature derivation accuracy validation."""
    
    def test_ttc_calculation_accuracy(self):
        """Test TTC calculation accuracy against known cases."""
        validator = FeatureDerivationValidator()
        
        result = validator.validate_feature_derivation_accuracy()
        
        # Should pass all test cases
        assert result.is_valid
        ttc_issues = [issue for issue in result.issues if "TTC test" in issue.message]
        assert len(ttc_issues) == 0
    
    def test_lane_estimation_accuracy(self):
        """Test lane estimation accuracy."""
        validator = FeatureDerivationValidator()
        
        result = validator.validate_feature_derivation_accuracy()
        
        # Check lane estimation specifically
        lane_issues = [issue for issue in result.issues if "Lane estimation test" in issue.message]
        assert len(lane_issues) == 0
    
    def test_lead_vehicle_detection_accuracy(self):
        """Test lead vehicle detection accuracy."""
        validator = FeatureDerivationValidator()
        
        result = validator.validate_feature_derivation_accuracy()
        
        # Check lead vehicle detection specifically
        lead_issues = [issue for issue in result.issues if "Lead vehicle" in issue.message]
        assert len(lead_issues) == 0
    
    def test_ttc_exception_handling(self):
        """Test handling of exceptions during feature validation."""
        validator = FeatureDerivationValidator()
        
        # Modify a test case to cause an exception by using invalid test case structure
        original_test_cases = validator.test_cases
        validator.test_cases = [
            {
                "name": "simple_ttc_head_on",
                "ego_state": "invalid_data",  # This will cause an exception in TTC calculation
                "other_state": np.array([1.0, 50.0, 0.0, -10.0, 0.0, -1.0, 0.0]),
                "expected_ttc": 2.5,
                "tolerance": 0.1
            }
        ]
        
        result = validator.validate_feature_derivation_accuracy()
        
        # Should handle the exception gracefully
        assert not result.is_valid
        assert any("Exception in test case" in issue.message for issue in result.issues)
        
        # Restore original test cases
        validator.test_cases = original_test_cases


class TestDataQualityAssurancePipeline:
    """Test comprehensive data quality assurance pipeline."""
    
    def test_full_pipeline_success(self):
        """Test successful execution of full QA pipeline."""
        pipeline = DataQualityAssurancePipeline()
        
        # Create test data
        modality_data = {
            'kinematics': [
                {
                    'episode_id': 'ep1', 'agent_id': 0, 'step': 0,
                    'kin_x': 0.0, 'kin_y': 0.0, 'kin_vx': 10.0, 'kin_vy': 0.0,
                    'reward': 1.0, 'terminated': False, 'truncated': False
                }
            ],
            'occupancy': [
                {
                    'episode_id': 'ep1', 'agent_id': 0, 'step': 0,
                    'kin_x': 0.0, 'kin_y': 0.0, 'kin_vx': 10.0, 'kin_vy': 0.0,
                    'reward': 1.0, 'terminated': False, 'truncated': False
                }
            ]
        }
        
        binary_test_data = {
            'original_arrays': {'test': np.array([1, 2, 3])},
            'encoded_data': {
                'test_blob': np.array([1, 2, 3]).tobytes(),
                'test_shape': [3],
                'test_dtype': 'int64'
            }
        }
        
        step_results = {
            'kinematics': {'reward': 1.0, 'terminated': False, 'truncated': False},
            'occupancy': {'reward': 1.0, 'terminated': False, 'truncated': False}
        }
        
        results = pipeline.run_full_quality_assurance(
            modality_data, binary_test_data, step_results
        )
        
        # Check that all validators ran
        expected_validators = [
            'trajectory_sync', 'binary_integrity', 'feature_accuracy',
            'env_sync', 'memory', 'data_integrity'
        ]
        
        for validator in expected_validators:
            assert validator in results
            assert isinstance(results[validator], ValidationResult)
    
    def test_quality_report_generation(self):
        """Test quality report generation."""
        pipeline = DataQualityAssurancePipeline()
        
        # Create mock results
        results = {
            'test_validator_pass': ValidationResult(True, []),
            'test_validator_fail': ValidationResult(False, [
                Mock(severity=ValidationSeverity.ERROR, message="Test error", 
                     component="TestComponent", details={"key": "value"})
            ])
        }
        
        report = pipeline.generate_quality_report(results)
        
        # Check report structure
        assert 'timestamp' in report
        assert 'overall_status' in report
        assert 'summary' in report
        assert 'validator_results' in report
        assert 'recommendations' in report
        
        # Check summary
        assert report['summary']['total_validators'] == 2
        assert report['summary']['passed'] == 1
        assert report['summary']['failed'] == 1
        assert report['summary']['errors'] == 1
        
        # Check overall status
        assert report['overall_status'] == 'FAIL'
        
        # Check recommendations
        assert any("ERROR:" in rec for rec in report['recommendations'])
    
    def test_pipeline_with_missing_data(self):
        """Test pipeline behavior with missing optional data."""
        pipeline = DataQualityAssurancePipeline()
        
        # Minimal data
        modality_data = {
            'kinematics': [
                {
                    'episode_id': 'ep1', 'agent_id': 0, 'step': 0,
                    'kin_x': 0.0, 'kin_y': 0.0, 'kin_vx': 10.0, 'kin_vy': 0.0,
                    'reward': 1.0, 'terminated': False, 'truncated': False
                }
            ]
        }
        
        # Run without binary test data and step results
        results = pipeline.run_full_quality_assurance(modality_data)
        
        # Should still run core validators
        assert 'trajectory_sync' in results
        assert 'feature_accuracy' in results
        assert 'memory' in results
        assert 'data_integrity' in results
        
        # Binary and env sync validators should not run
        assert 'binary_integrity' not in results
        assert 'env_sync' not in results
    
    def test_pipeline_configuration(self):
        """Test pipeline configuration options."""
        config = {
            'position_tolerance': 1e-3,
            'velocity_tolerance': 1e-3,
            'binary_tolerance': 1e-5,
            'sync_tolerance': 1e-5,
            'max_memory_gb': 16.0,
            'memory_warning_threshold': 0.9
        }
        
        pipeline = DataQualityAssurancePipeline(config)
        
        # Check that validators are configured correctly
        assert pipeline.trajectory_validator.position_tolerance == 1e-3
        assert pipeline.trajectory_validator.velocity_tolerance == 1e-3
        assert pipeline.binary_validator.tolerance == 1e-5
        assert pipeline.sync_validator.tolerance == 1e-5
        assert pipeline.memory_validator.max_memory_bytes == 16.0 * 1024**3
        assert pipeline.memory_validator.warning_threshold == 0.9


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple validators."""
    
    def test_complete_data_collection_validation(self):
        """Test validation of a complete data collection scenario."""
        pipeline = DataQualityAssurancePipeline()
        
        # Simulate complete multi-modal data collection
        episodes = ['ep1', 'ep2']
        agents = [0, 1]
        steps = [0, 1, 2]
        
        modality_data = {}
        for modality in ['kinematics', 'occupancy', 'grayscale']:
            modality_data[modality] = []
            
            for episode in episodes:
                for agent in agents:
                    for step in steps:
                        # Create synchronized observations
                        x_pos = step * 10.0 + agent * 5.0
                        y_pos = agent * 4.0  # Different lanes
                        
                        obs = {
                            'episode_id': episode,
                            'agent_id': agent,
                            'step': step,
                            'kin_x': x_pos,
                            'kin_y': y_pos,
                            'kin_vx': 10.0,
                            'kin_vy': 0.0,
                            'reward': 1.0,
                            'terminated': step == 2,  # Terminate at last step
                            'truncated': False
                        }
                        modality_data[modality].append(obs)
        
        # Create binary test data
        test_arrays = {
            'occupancy': np.random.rand(64, 64).astype(np.float32),
            'grayscale': np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)
        }
        
        encoded_data = {}
        for name, array in test_arrays.items():
            encoded_data[f"{name}_blob"] = array.tobytes()
            encoded_data[f"{name}_shape"] = list(array.shape)
            encoded_data[f"{name}_dtype"] = str(array.dtype)
        
        binary_test_data = {
            'original_arrays': test_arrays,
            'encoded_data': encoded_data
        }
        
        # Create step results
        step_results = {}
        for modality in ['kinematics', 'occupancy', 'grayscale']:
            step_results[modality] = {
                'reward': 1.0,
                'terminated': False,
                'truncated': False
            }
        
        # Run full validation
        results = pipeline.run_full_quality_assurance(
            modality_data, binary_test_data, step_results
        )
        
        # Generate report
        report = pipeline.generate_quality_report(results)
        
        # All validations should pass for properly synchronized data
        assert report['overall_status'] == 'PASS'
        assert report['summary']['failed'] == 0
        
        # Check specific validator results
        assert results['trajectory_sync'].is_valid
        assert results['binary_integrity'].is_valid
        assert results['feature_accuracy'].is_valid
        assert results['env_sync'].is_valid
    
    def test_data_corruption_detection(self):
        """Test detection of various data corruption scenarios."""
        pipeline = DataQualityAssurancePipeline()
        
        # Create corrupted data scenarios
        corrupted_modality_data = {
            'kinematics': [
                {
                    'episode_id': 'ep1', 'agent_id': 0, 'step': 0,
                    'kin_x': 0.0, 'kin_y': 0.0, 'kin_vx': 10.0, 'kin_vy': 0.0,
                    'reward': 1.0, 'terminated': False, 'truncated': False
                }
            ],
            'occupancy': [
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
        corrupted_encoded_data = {
            'test_blob': b'corrupted_data',  # Wrong size
            'test_shape': [64, 64],
            'test_dtype': 'float32'
        }
        
        corrupted_binary_data = {
            'original_arrays': {'test': original_array},
            'encoded_data': corrupted_encoded_data
        }
        
        # Run validation on corrupted data
        results = pipeline.run_full_quality_assurance(
            corrupted_modality_data, corrupted_binary_data
        )
        
        # Generate report
        report = pipeline.generate_quality_report(results)
        
        # Should detect multiple corruption issues
        assert report['overall_status'] == 'FAIL'
        assert report['summary']['failed'] > 0
        assert report['summary']['errors'] > 0
        
        # Check specific corruption detection
        assert not results['trajectory_sync'].is_valid  # Position/reward corruption
        assert not results['binary_integrity'].is_valid  # Binary corruption