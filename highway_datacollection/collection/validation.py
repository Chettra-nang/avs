"""
Validation utilities for data collection operations.
"""

import logging
import numpy as np
import psutil
import gc
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: ValidationSeverity
    message: str
    component: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    issues: List[ValidationIssue]
    
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                  for issue in self.issues)
    
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues of a specific severity."""
        return [issue for issue in self.issues if issue.severity == severity]


class EnvironmentSynchronizationValidator:
    """
    Validates synchronization across parallel environments.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize validator.
        
        Args:
            tolerance: Numerical tolerance for floating point comparisons
        """
        self.tolerance = tolerance
        self.sync_history: List[bool] = []
        self.desync_count = 0
        self.max_desync_threshold = 3  # Maximum consecutive desyncs before critical error
    
    def validate_step_synchronization(self, step_results: Dict[str, Any]) -> ValidationResult:
        """
        Validate that all environments are synchronized after a step.
        
        Args:
            step_results: Results from stepping all environments
            
        Returns:
            ValidationResult with synchronization status
        """
        issues = []
        
        if not step_results:
            issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                "No step results provided for synchronization validation",
                "EnvironmentSynchronizationValidator"
            ))
            return ValidationResult(False, issues)
        
        if len(step_results) < 2:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                f"Only {len(step_results)} environment(s) available for synchronization check",
                "EnvironmentSynchronizationValidator"
            ))
            return ValidationResult(True, issues)
        
        # Get reference values from first environment
        first_obs_type = next(iter(step_results.keys()))
        ref_result = step_results[first_obs_type]
        
        # Validate required fields exist
        required_fields = ['reward', 'terminated', 'truncated']
        for field in required_fields:
            if field not in ref_result:
                issues.append(ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Missing required field '{field}' in step results",
                    "EnvironmentSynchronizationValidator",
                    {"missing_field": field, "environment": first_obs_type}
                ))
        
        if issues:
            return ValidationResult(False, issues)
        
        ref_reward = ref_result['reward']
        ref_terminated = ref_result['terminated']
        ref_truncated = ref_result['truncated']
        
        # Check synchronization across all environments
        sync_valid = True
        for obs_type, result in step_results.items():
            # Validate reward synchronization
            if not self._compare_values(result.get('reward'), ref_reward):
                issues.append(ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Reward desynchronization detected",
                    "EnvironmentSynchronizationValidator",
                    {
                        "environment": obs_type,
                        "expected_reward": ref_reward,
                        "actual_reward": result.get('reward'),
                        "reference_env": first_obs_type
                    }
                ))
                sync_valid = False
            
            # Validate termination synchronization
            if result.get('terminated') != ref_terminated:
                issues.append(ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Termination status desynchronization detected",
                    "EnvironmentSynchronizationValidator",
                    {
                        "environment": obs_type,
                        "expected_terminated": ref_terminated,
                        "actual_terminated": result.get('terminated'),
                        "reference_env": first_obs_type
                    }
                ))
                sync_valid = False
            
            # Validate truncation synchronization
            if result.get('truncated') != ref_truncated:
                issues.append(ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Truncation status desynchronization detected",
                    "EnvironmentSynchronizationValidator",
                    {
                        "environment": obs_type,
                        "expected_truncated": ref_truncated,
                        "actual_truncated": result.get('truncated'),
                        "reference_env": first_obs_type
                    }
                ))
                sync_valid = False
        
        # Update synchronization history
        self.sync_history.append(sync_valid)
        if not sync_valid:
            self.desync_count += 1
        else:
            self.desync_count = 0  # Reset counter on successful sync
        
        # Check for critical desynchronization pattern
        if self.desync_count >= self.max_desync_threshold:
            issues.append(ValidationIssue(
                ValidationSeverity.CRITICAL,
                f"Critical desynchronization: {self.desync_count} consecutive failures",
                "EnvironmentSynchronizationValidator",
                {"consecutive_failures": self.desync_count}
            ))
        
        # Keep history manageable
        if len(self.sync_history) > 100:
            self.sync_history = self.sync_history[-50:]
        
        return ValidationResult(sync_valid, issues)
    
    def _compare_values(self, val1: Any, val2: Any) -> bool:
        """
        Compare two values with appropriate tolerance.
        
        Args:
            val1: First value
            val2: Second value
            
        Returns:
            True if values are considered equal
        """
        if val1 is None or val2 is None:
            return val1 == val2
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return abs(val1 - val2) <= self.tolerance
        
        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if val1.shape != val2.shape:
                return False
            return np.allclose(val1, val2, atol=self.tolerance)
        
        return val1 == val2
    
    def get_synchronization_stats(self) -> Dict[str, Any]:
        """
        Get synchronization statistics.
        
        Returns:
            Dictionary with synchronization statistics
        """
        if not self.sync_history:
            return {"total_checks": 0, "success_rate": 0.0, "current_desync_count": 0}
        
        successful_syncs = sum(self.sync_history)
        total_checks = len(self.sync_history)
        success_rate = successful_syncs / total_checks
        
        return {
            "total_checks": total_checks,
            "successful_syncs": successful_syncs,
            "failed_syncs": total_checks - successful_syncs,
            "success_rate": success_rate,
            "current_desync_count": self.desync_count,
            "recent_history": self.sync_history[-10:] if len(self.sync_history) >= 10 else self.sync_history
        }
    
    def reset(self) -> None:
        """Reset validation state."""
        self.sync_history.clear()
        self.desync_count = 0


class MemoryValidator:
    """
    Validates and monitors memory usage during data collection.
    """
    
    def __init__(self, max_memory_gb: float = 8.0, warning_threshold: float = 0.8):
        """
        Initialize memory validator.
        
        Args:
            max_memory_gb: Maximum allowed memory usage in GB
            warning_threshold: Threshold (0-1) for memory usage warnings
        """
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.warning_threshold = warning_threshold
        self.memory_history: List[float] = []
    
    def validate_memory_usage(self) -> ValidationResult:
        """
        Validate current memory usage.
        
        Returns:
            ValidationResult with memory status
        """
        issues = []
        
        try:
            # Get current memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            current_memory = memory_info.rss  # Resident Set Size
            
            # Get system memory info
            system_memory = psutil.virtual_memory()
            
            # Record memory usage
            self.memory_history.append(current_memory)
            
            # Keep history manageable
            if len(self.memory_history) > 100:
                self.memory_history = self.memory_history[-50:]
            
            # Check against absolute limit
            if current_memory > self.max_memory_bytes:
                issues.append(ValidationIssue(
                    ValidationSeverity.CRITICAL,
                    f"Memory usage exceeded limit: {current_memory / 1024**3:.2f}GB > {self.max_memory_bytes / 1024**3:.2f}GB",
                    "MemoryValidator",
                    {
                        "current_memory_gb": current_memory / 1024**3,
                        "max_memory_gb": self.max_memory_bytes / 1024**3,
                        "system_available_gb": system_memory.available / 1024**3
                    }
                ))
            
            # Check against warning threshold
            elif current_memory > self.max_memory_bytes * self.warning_threshold:
                issues.append(ValidationIssue(
                    ValidationSeverity.WARNING,
                    f"Memory usage approaching limit: {current_memory / 1024**3:.2f}GB (threshold: {self.max_memory_bytes * self.warning_threshold / 1024**3:.2f}GB)",
                    "MemoryValidator",
                    {
                        "current_memory_gb": current_memory / 1024**3,
                        "threshold_gb": self.max_memory_bytes * self.warning_threshold / 1024**3,
                        "usage_percentage": current_memory / self.max_memory_bytes
                    }
                ))
            
            # Check system memory availability
            if system_memory.percent > 90:
                issues.append(ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"System memory critically low: {system_memory.percent:.1f}% used",
                    "MemoryValidator",
                    {
                        "system_memory_percent": system_memory.percent,
                        "available_gb": system_memory.available / 1024**3
                    }
                ))
            elif system_memory.percent > 80:
                issues.append(ValidationIssue(
                    ValidationSeverity.WARNING,
                    f"System memory usage high: {system_memory.percent:.1f}% used",
                    "MemoryValidator",
                    {
                        "system_memory_percent": system_memory.percent,
                        "available_gb": system_memory.available / 1024**3
                    }
                ))
            
            # Check for memory leaks (rapid growth)
            if len(self.memory_history) >= 10:
                recent_growth = self.memory_history[-1] - self.memory_history[-10]
                if recent_growth > 100 * 1024 * 1024:  # 100MB growth in 10 checks
                    issues.append(ValidationIssue(
                        ValidationSeverity.WARNING,
                        f"Potential memory leak detected: {recent_growth / 1024**2:.1f}MB growth in recent checks",
                        "MemoryValidator",
                        {
                            "growth_mb": recent_growth / 1024**2,
                            "checks_analyzed": 10
                        }
                    ))
            
            is_valid = not any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
            return ValidationResult(is_valid, issues)
            
        except Exception as e:
            issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                f"Failed to validate memory usage: {str(e)}",
                "MemoryValidator"
            ))
            return ValidationResult(False, issues)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self.memory_history:
            return {"no_data": True}
        
        current_memory = self.memory_history[-1]
        max_memory = max(self.memory_history)
        min_memory = min(self.memory_history)
        avg_memory = sum(self.memory_history) / len(self.memory_history)
        
        return {
            "current_memory_gb": current_memory / 1024**3,
            "max_memory_gb": max_memory / 1024**3,
            "min_memory_gb": min_memory / 1024**3,
            "avg_memory_gb": avg_memory / 1024**3,
            "memory_limit_gb": self.max_memory_bytes / 1024**3,
            "usage_percentage": current_memory / self.max_memory_bytes,
            "samples_count": len(self.memory_history)
        }
    
    def trigger_garbage_collection(self) -> Dict[str, Any]:
        """
        Trigger garbage collection and return statistics.
        
        Returns:
            Dictionary with garbage collection results
        """
        before_memory = psutil.Process().memory_info().rss
        
        # Force garbage collection
        collected = gc.collect()
        
        after_memory = psutil.Process().memory_info().rss
        freed_memory = before_memory - after_memory
        
        return {
            "objects_collected": collected,
            "memory_freed_mb": freed_memory / 1024**2,
            "before_memory_gb": before_memory / 1024**3,
            "after_memory_gb": after_memory / 1024**3
        }


class TrajectoryComparisonValidator:
    """
    Validates synchronization across modalities using trajectory comparison.
    """
    
    def __init__(self, position_tolerance: float = 1e-4, velocity_tolerance: float = 1e-4):
        """
        Initialize trajectory comparison validator.
        
        Args:
            position_tolerance: Tolerance for position comparisons
            velocity_tolerance: Tolerance for velocity comparisons
        """
        self.position_tolerance = position_tolerance
        self.velocity_tolerance = velocity_tolerance
    
    def validate_trajectory_synchronization(self, 
                                          modality_data: Dict[str, List[Dict[str, Any]]]) -> ValidationResult:
        """
        Validate that trajectories are synchronized across modalities.
        
        Args:
            modality_data: Dictionary mapping modality names to observation lists
            
        Returns:
            ValidationResult with synchronization status
        """
        issues = []
        
        if len(modality_data) < 2:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                f"Need at least 2 modalities for trajectory comparison, got {len(modality_data)}",
                "TrajectoryComparisonValidator"
            ))
            return ValidationResult(True, issues)
        
        # Extract trajectory data for each modality
        trajectories = {}
        for modality, observations in modality_data.items():
            trajectories[modality] = self._extract_trajectory_data(observations)
        
        # Get reference modality (first one)
        reference_modality = next(iter(trajectories.keys()))
        reference_trajectory = trajectories[reference_modality]
        
        # Compare all other modalities against reference
        for modality, trajectory in trajectories.items():
            if modality == reference_modality:
                continue
                
            sync_result = self._compare_trajectories(
                reference_trajectory, trajectory, reference_modality, modality
            )
            issues.extend(sync_result.issues)
        
        is_valid = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          for issue in issues)
        return ValidationResult(is_valid, issues)
    
    def _extract_trajectory_data(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract trajectory data from observations.
        
        Args:
            observations: List of observation records
            
        Returns:
            Dictionary with trajectory data organized by episode and agent
        """
        trajectories = {}
        
        for obs in observations:
            episode_id = obs.get('episode_id')
            agent_id = obs.get('agent_id')
            step = obs.get('step')
            
            if not all([episode_id, agent_id is not None, step is not None]):
                continue
            
            key = (episode_id, agent_id)
            if key not in trajectories:
                trajectories[key] = {}
            
            # Extract position and velocity from kinematics data
            position = None
            velocity = None
            
            if 'kin_x' in obs and 'kin_y' in obs:
                position = (obs['kin_x'], obs['kin_y'])
            
            if 'kin_vx' in obs and 'kin_vy' in obs:
                velocity = (obs['kin_vx'], obs['kin_vy'])
            
            trajectories[key][step] = {
                'position': position,
                'velocity': velocity,
                'reward': obs.get('reward'),
                'terminated': obs.get('terminated'),
                'truncated': obs.get('truncated')
            }
        
        return trajectories
    
    def _compare_trajectories(self, ref_trajectory: Dict[str, Any], 
                            test_trajectory: Dict[str, Any],
                            ref_modality: str, test_modality: str) -> ValidationResult:
        """
        Compare two trajectories for synchronization.
        
        Args:
            ref_trajectory: Reference trajectory data
            test_trajectory: Test trajectory data
            ref_modality: Name of reference modality
            test_modality: Name of test modality
            
        Returns:
            ValidationResult with comparison results
        """
        issues = []
        
        # Check if same episodes and agents exist
        ref_keys = set(ref_trajectory.keys())
        test_keys = set(test_trajectory.keys())
        
        missing_in_test = ref_keys - test_keys
        extra_in_test = test_keys - ref_keys
        
        for key in missing_in_test:
            issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                f"Episode/agent {key} missing in {test_modality}",
                "TrajectoryComparisonValidator",
                {"reference_modality": ref_modality, "test_modality": test_modality, "missing_key": key}
            ))
        
        for key in extra_in_test:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                f"Extra episode/agent {key} in {test_modality}",
                "TrajectoryComparisonValidator",
                {"reference_modality": ref_modality, "test_modality": test_modality, "extra_key": key}
            ))
        
        # Compare common trajectories
        common_keys = ref_keys & test_keys
        for key in common_keys:
            ref_steps = ref_trajectory[key]
            test_steps = test_trajectory[key]
            
            # Check step alignment
            ref_step_nums = set(ref_steps.keys())
            test_step_nums = set(test_steps.keys())
            
            if ref_step_nums != test_step_nums:
                issues.append(ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Step mismatch for {key} between modalities",
                    "TrajectoryComparisonValidator",
                    {
                        "reference_modality": ref_modality,
                        "test_modality": test_modality,
                        "key": key,
                        "ref_steps": sorted(ref_step_nums),
                        "test_steps": sorted(test_step_nums)
                    }
                ))
                continue
            
            # Compare step-by-step data
            for step_num in ref_step_nums:
                ref_step = ref_steps[step_num]
                test_step = test_steps[step_num]
                
                # Compare positions
                if ref_step['position'] and test_step['position']:
                    pos_diff = np.array(ref_step['position']) - np.array(test_step['position'])
                    pos_error = np.linalg.norm(pos_diff)
                    
                    if pos_error > self.position_tolerance:
                        issues.append(ValidationIssue(
                            ValidationSeverity.ERROR,
                            f"Position desynchronization at {key}, step {step_num}",
                            "TrajectoryComparisonValidator",
                            {
                                "reference_modality": ref_modality,
                                "test_modality": test_modality,
                                "key": key,
                                "step": step_num,
                                "position_error": pos_error,
                                "tolerance": self.position_tolerance,
                                "ref_position": ref_step['position'],
                                "test_position": test_step['position']
                            }
                        ))
                
                # Compare velocities
                if ref_step['velocity'] and test_step['velocity']:
                    vel_diff = np.array(ref_step['velocity']) - np.array(test_step['velocity'])
                    vel_error = np.linalg.norm(vel_diff)
                    
                    if vel_error > self.velocity_tolerance:
                        issues.append(ValidationIssue(
                            ValidationSeverity.ERROR,
                            f"Velocity desynchronization at {key}, step {step_num}",
                            "TrajectoryComparisonValidator",
                            {
                                "reference_modality": ref_modality,
                                "test_modality": test_modality,
                                "key": key,
                                "step": step_num,
                                "velocity_error": vel_error,
                                "tolerance": self.velocity_tolerance,
                                "ref_velocity": ref_step['velocity'],
                                "test_velocity": test_step['velocity']
                            }
                        ))
                
                # Compare control signals (rewards, termination)
                if ref_step['reward'] != test_step['reward']:
                    issues.append(ValidationIssue(
                        ValidationSeverity.ERROR,
                        f"Reward desynchronization at {key}, step {step_num}",
                        "TrajectoryComparisonValidator",
                        {
                            "reference_modality": ref_modality,
                            "test_modality": test_modality,
                            "key": key,
                            "step": step_num,
                            "ref_reward": ref_step['reward'],
                            "test_reward": test_step['reward']
                        }
                    ))
                
                if ref_step['terminated'] != test_step['terminated']:
                    issues.append(ValidationIssue(
                        ValidationSeverity.ERROR,
                        f"Termination desynchronization at {key}, step {step_num}",
                        "TrajectoryComparisonValidator",
                        {
                            "reference_modality": ref_modality,
                            "test_modality": test_modality,
                            "key": key,
                            "step": step_num,
                            "ref_terminated": ref_step['terminated'],
                            "test_terminated": test_step['terminated']
                        }
                    ))
        
        is_valid = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          for issue in issues)
        return ValidationResult(is_valid, issues)


class BinaryArrayIntegrityValidator:
    """
    Validates binary array reconstruction integrity.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize binary array integrity validator.
        
        Args:
            tolerance: Tolerance for numerical comparisons
        """
        self.tolerance = tolerance
    
    def validate_binary_reconstruction(self, 
                                     original_arrays: Dict[str, np.ndarray],
                                     encoded_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate that binary arrays can be reconstructed accurately.
        
        Args:
            original_arrays: Original arrays before encoding
            encoded_data: Encoded binary data with metadata
            
        Returns:
            ValidationResult with reconstruction integrity status
        """
        issues = []
        
        for array_name, original_array in original_arrays.items():
            blob_key = f"{array_name}_blob"
            shape_key = f"{array_name}_shape"
            dtype_key = f"{array_name}_dtype"
            
            # Check if encoded data exists
            if not all(key in encoded_data for key in [blob_key, shape_key, dtype_key]):
                missing_keys = [key for key in [blob_key, shape_key, dtype_key] 
                              if key not in encoded_data]
                issues.append(ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Missing encoded data for array '{array_name}'",
                    "BinaryArrayIntegrityValidator",
                    {"array_name": array_name, "missing_keys": missing_keys}
                ))
                continue
            
            try:
                # Reconstruct array from binary data
                blob = encoded_data[blob_key]
                shape = tuple(encoded_data[shape_key])
                dtype = encoded_data[dtype_key]
                
                # Check if blob size matches expected size
                expected_size = np.prod(shape) * np.dtype(dtype).itemsize
                if len(blob) != expected_size:
                    issues.append(ValidationIssue(
                        ValidationSeverity.ERROR,
                        f"Binary data size mismatch for array '{array_name}'",
                        "BinaryArrayIntegrityValidator",
                        {
                            "array_name": array_name,
                            "expected_size": expected_size,
                            "actual_size": len(blob),
                            "shape": shape,
                            "dtype": dtype
                        }
                    ))
                    continue
                
                reconstructed_array = np.frombuffer(blob, dtype=dtype).reshape(shape)
                
                # Validate reconstruction
                reconstruction_result = self._validate_array_match(
                    original_array, reconstructed_array, array_name
                )
                issues.extend(reconstruction_result.issues)
                
            except Exception as e:
                issues.append(ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Failed to reconstruct array '{array_name}': {str(e)}",
                    "BinaryArrayIntegrityValidator",
                    {"array_name": array_name, "error": str(e)}
                ))
        
        is_valid = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          for issue in issues)
        return ValidationResult(is_valid, issues)
    
    def _validate_array_match(self, original: np.ndarray, reconstructed: np.ndarray, 
                            array_name: str) -> ValidationResult:
        """
        Validate that two arrays match within tolerance.
        
        Args:
            original: Original array
            reconstructed: Reconstructed array
            array_name: Name of the array for error reporting
            
        Returns:
            ValidationResult with match status
        """
        issues = []
        
        # Check shapes
        if original.shape != reconstructed.shape:
            issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                f"Shape mismatch for array '{array_name}'",
                "BinaryArrayIntegrityValidator",
                {
                    "array_name": array_name,
                    "original_shape": original.shape,
                    "reconstructed_shape": reconstructed.shape
                }
            ))
            return ValidationResult(False, issues)
        
        # Check dtypes
        if original.dtype != reconstructed.dtype:
            issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                f"Dtype mismatch for array '{array_name}'",
                "BinaryArrayIntegrityValidator",
                {
                    "array_name": array_name,
                    "original_dtype": str(original.dtype),
                    "reconstructed_dtype": str(reconstructed.dtype)
                }
            ))
            return ValidationResult(False, issues)
        
        # Check values
        if np.issubdtype(original.dtype, np.floating):
            # Use tolerance for floating point arrays
            if not np.allclose(original, reconstructed, atol=self.tolerance, rtol=self.tolerance):
                max_diff = np.max(np.abs(original - reconstructed))
                issues.append(ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Value mismatch for floating point array '{array_name}'",
                    "BinaryArrayIntegrityValidator",
                    {
                        "array_name": array_name,
                        "max_difference": max_diff,
                        "tolerance": self.tolerance
                    }
                ))
        else:
            # Exact match for integer arrays
            if not np.array_equal(original, reconstructed):
                diff_count = np.sum(original != reconstructed)
                issues.append(ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Value mismatch for integer array '{array_name}'",
                    "BinaryArrayIntegrityValidator",
                    {
                        "array_name": array_name,
                        "different_elements": int(diff_count),
                        "total_elements": original.size
                    }
                ))
        
        is_valid = len(issues) == 0
        return ValidationResult(is_valid, issues)


class FeatureDerivationValidator:
    """
    Validates accuracy of feature derivation against known test cases.
    """
    
    def __init__(self):
        """Initialize feature derivation validator."""
        self.test_cases = self._create_test_cases()
    
    def _create_test_cases(self) -> List[Dict[str, Any]]:
        """
        Create known test cases for feature validation.
        
        Returns:
            List of test cases with expected results
        """
        test_cases = []
        
        # Test case 1: Simple TTC calculation
        test_cases.append({
            "name": "simple_ttc_head_on",
            "ego_state": np.array([1.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0]),  # Moving forward at 10 m/s
            "other_state": np.array([1.0, 50.0, 0.0, -10.0, 0.0, -1.0, 0.0]),  # 50m ahead, moving backward at 10 m/s
            "expected_ttc": 2.5,  # 50m / (10 + 10) m/s = 2.5s
            "tolerance": 0.1
        })
        
        # Test case 2: No collision (parallel movement)
        test_cases.append({
            "name": "no_collision_parallel",
            "ego_state": np.array([1.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0]),
            "other_state": np.array([1.0, 0.0, 4.0, 10.0, 0.0, 1.0, 0.0]),  # Same speed, different lane
            "expected_ttc": float('inf'),
            "tolerance": None
        })
        
        # Test case 3: Lane estimation
        test_cases.append({
            "name": "lane_estimation_center",
            "y_position": -2.0,  # Center of lane 1 (assuming 4m lanes, rightmost at -8 to -4)
            "expected_lane": 1,
            "lane_width": 4.0,
            "num_lanes": 4
        })
        
        # Test case 4: Lead vehicle detection
        test_cases.append({
            "name": "lead_vehicle_detection",
            "ego_state": np.array([1.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0]),
            "other_vehicles": np.array([
                [1.0, 20.0, 0.5, 8.0, 0.0, 1.0, 0.0],  # Lead vehicle 20m ahead, same lane
                [1.0, -10.0, 0.0, 12.0, 0.0, 1.0, 0.0],  # Vehicle behind
                [1.0, 15.0, 8.0, 9.0, 0.0, 1.0, 0.0]   # Vehicle ahead but different lane
            ]),
            "expected_lead_idx": 0,
            "expected_gap": 20.0,
            "same_lane_threshold": 2.0
        })
        
        return test_cases
    
    def validate_feature_derivation_accuracy(self) -> ValidationResult:
        """
        Validate feature derivation accuracy against known test cases.
        
        Returns:
            ValidationResult with accuracy validation status
        """
        issues = []
        
        # Import feature extraction functions
        try:
            from ..features.extractors import (
                calculate_time_to_collision, estimate_lane_from_position, 
                detect_lead_vehicle
            )
        except ImportError as e:
            issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                f"Failed to import feature extraction functions: {str(e)}",
                "FeatureDerivationValidator"
            ))
            return ValidationResult(False, issues)
        
        # Test each case
        for test_case in self.test_cases:
            try:
                if test_case["name"].startswith("simple_ttc") or test_case["name"].startswith("no_collision"):
                    # Test TTC calculation
                    calculated_ttc = calculate_time_to_collision(
                        test_case["ego_state"], test_case["other_state"]
                    )
                    expected_ttc = test_case["expected_ttc"]
                    
                    if expected_ttc == float('inf'):
                        if calculated_ttc != float('inf'):
                            issues.append(ValidationIssue(
                                ValidationSeverity.ERROR,
                                f"TTC test '{test_case['name']}' failed: expected inf, got {calculated_ttc}",
                                "FeatureDerivationValidator",
                                {
                                    "test_case": test_case["name"],
                                    "expected": expected_ttc,
                                    "calculated": calculated_ttc
                                }
                            ))
                    else:
                        tolerance = test_case.get("tolerance", 0.1)
                        if abs(calculated_ttc - expected_ttc) > tolerance:
                            issues.append(ValidationIssue(
                                ValidationSeverity.ERROR,
                                f"TTC test '{test_case['name']}' failed: expected {expected_ttc}, got {calculated_ttc}",
                                "FeatureDerivationValidator",
                                {
                                    "test_case": test_case["name"],
                                    "expected": expected_ttc,
                                    "calculated": calculated_ttc,
                                    "tolerance": tolerance,
                                    "error": abs(calculated_ttc - expected_ttc)
                                }
                            ))
                
                elif test_case["name"].startswith("lane_estimation"):
                    # Test lane estimation
                    calculated_lane = estimate_lane_from_position(
                        test_case["y_position"], 
                        test_case["lane_width"], 
                        test_case["num_lanes"]
                    )
                    expected_lane = test_case["expected_lane"]
                    
                    if calculated_lane != expected_lane:
                        issues.append(ValidationIssue(
                            ValidationSeverity.ERROR,
                            f"Lane estimation test '{test_case['name']}' failed: expected {expected_lane}, got {calculated_lane}",
                            "FeatureDerivationValidator",
                            {
                                "test_case": test_case["name"],
                                "y_position": test_case["y_position"],
                                "expected": expected_lane,
                                "calculated": calculated_lane
                            }
                        ))
                
                elif test_case["name"].startswith("lead_vehicle"):
                    # Test lead vehicle detection
                    lead_idx, gap = detect_lead_vehicle(
                        test_case["ego_state"],
                        test_case["other_vehicles"],
                        test_case["same_lane_threshold"]
                    )
                    expected_idx = test_case["expected_lead_idx"]
                    expected_gap = test_case["expected_gap"]
                    
                    if lead_idx != expected_idx:
                        issues.append(ValidationIssue(
                            ValidationSeverity.ERROR,
                            f"Lead vehicle detection test '{test_case['name']}' failed: expected idx {expected_idx}, got {lead_idx}",
                            "FeatureDerivationValidator",
                            {
                                "test_case": test_case["name"],
                                "expected_idx": expected_idx,
                                "calculated_idx": lead_idx,
                                "expected_gap": expected_gap,
                                "calculated_gap": gap
                            }
                        ))
                    
                    if abs(gap - expected_gap) > 0.1:
                        issues.append(ValidationIssue(
                            ValidationSeverity.ERROR,
                            f"Lead vehicle gap test '{test_case['name']}' failed: expected {expected_gap}, got {gap}",
                            "FeatureDerivationValidator",
                            {
                                "test_case": test_case["name"],
                                "expected_gap": expected_gap,
                                "calculated_gap": gap,
                                "error": abs(gap - expected_gap)
                            }
                        ))
                
            except Exception as e:
                issues.append(ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Exception in test case '{test_case['name']}': {str(e)}",
                    "FeatureDerivationValidator",
                    {"test_case": test_case["name"], "error": str(e)}
                ))
        
        is_valid = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          for issue in issues)
        return ValidationResult(is_valid, issues)


class DataQualityAssurancePipeline:
    """
    Comprehensive data quality assurance pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data quality assurance pipeline.
        
        Args:
            config: Configuration for validators
        """
        config = config or {}
        
        # Initialize validators
        self.trajectory_validator = TrajectoryComparisonValidator(
            position_tolerance=config.get('position_tolerance', 1e-4),
            velocity_tolerance=config.get('velocity_tolerance', 1e-4)
        )
        
        self.binary_validator = BinaryArrayIntegrityValidator(
            tolerance=config.get('binary_tolerance', 1e-6)
        )
        
        self.feature_validator = FeatureDerivationValidator()
        
        self.sync_validator = EnvironmentSynchronizationValidator(
            tolerance=config.get('sync_tolerance', 1e-6)
        )
        
        self.memory_validator = MemoryValidator(
            max_memory_gb=config.get('max_memory_gb', 8.0),
            warning_threshold=config.get('memory_warning_threshold', 0.8)
        )
        
        self.data_integrity_validator = DataIntegrityValidator()
    
    def run_full_quality_assurance(self, 
                                 modality_data: Dict[str, List[Dict[str, Any]]],
                                 binary_test_data: Optional[Dict[str, Any]] = None,
                                 step_results: Optional[Dict[str, Any]] = None) -> Dict[str, ValidationResult]:
        """
        Run comprehensive data quality assurance.
        
        Args:
            modality_data: Data from different modalities for trajectory comparison
            binary_test_data: Test data for binary array validation
            step_results: Step results for synchronization validation
            
        Returns:
            Dictionary mapping validator names to their results
        """
        results = {}
        
        # 1. Trajectory synchronization validation
        logger.info("Running trajectory synchronization validation...")
        results['trajectory_sync'] = self.trajectory_validator.validate_trajectory_synchronization(
            modality_data
        )
        
        # 2. Binary array integrity validation
        if binary_test_data:
            logger.info("Running binary array integrity validation...")
            original_arrays = binary_test_data.get('original_arrays', {})
            encoded_data = binary_test_data.get('encoded_data', {})
            results['binary_integrity'] = self.binary_validator.validate_binary_reconstruction(
                original_arrays, encoded_data
            )
        
        # 3. Feature derivation accuracy validation
        logger.info("Running feature derivation accuracy validation...")
        results['feature_accuracy'] = self.feature_validator.validate_feature_derivation_accuracy()
        
        # 4. Environment synchronization validation
        if step_results:
            logger.info("Running environment synchronization validation...")
            results['env_sync'] = self.sync_validator.validate_step_synchronization(step_results)
        
        # 5. Memory usage validation
        logger.info("Running memory usage validation...")
        results['memory'] = self.memory_validator.validate_memory_usage()
        
        # 6. Data integrity validation (use first modality data to avoid duplicates)
        logger.info("Running data integrity validation...")
        first_modality_data = next(iter(modality_data.values())) if modality_data else []
        results['data_integrity'] = self.data_integrity_validator.validate_observation_data(
            first_modality_data
        )
        
        return results
    
    def generate_quality_report(self, results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """
        Generate a comprehensive quality assurance report.
        
        Args:
            results: Results from quality assurance pipeline
            
        Returns:
            Dictionary with quality report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "PASS",
            "summary": {
                "total_validators": len(results),
                "passed": 0,
                "failed": 0,
                "warnings": 0,
                "errors": 0,
                "critical": 0
            },
            "validator_results": {},
            "recommendations": []
        }
        
        # Analyze results
        for validator_name, result in results.items():
            report["validator_results"][validator_name] = {
                "status": "PASS" if result.is_valid else "FAIL",
                "issues_count": len(result.issues),
                "issues": [
                    {
                        "severity": issue.severity.value,
                        "message": issue.message,
                        "component": issue.component,
                        "details": issue.details
                    }
                    for issue in result.issues
                ]
            }
            
            if result.is_valid:
                report["summary"]["passed"] += 1
            else:
                report["summary"]["failed"] += 1
                report["overall_status"] = "FAIL"
            
            # Count issues by severity
            for issue in result.issues:
                if issue.severity == ValidationSeverity.WARNING:
                    report["summary"]["warnings"] += 1
                elif issue.severity == ValidationSeverity.ERROR:
                    report["summary"]["errors"] += 1
                elif issue.severity == ValidationSeverity.CRITICAL:
                    report["summary"]["critical"] += 1
        
        # Generate recommendations
        if report["summary"]["critical"] > 0:
            report["recommendations"].append(
                "CRITICAL: Address critical issues immediately before proceeding with data collection"
            )
        
        if report["summary"]["errors"] > 0:
            report["recommendations"].append(
                "ERROR: Fix error-level issues to ensure data quality and reliability"
            )
        
        if report["summary"]["warnings"] > 0:
            report["recommendations"].append(
                "WARNING: Review warning-level issues for potential improvements"
            )
        
        if report["overall_status"] == "PASS":
            report["recommendations"].append(
                "SUCCESS: All quality assurance checks passed. Data collection system is ready."
            )
        
        return report


class DataIntegrityValidator:
    """
    Validates data integrity during collection and storage.
    """
    
    def validate_observation_data(self, observations: List[Dict[str, Any]]) -> ValidationResult:
        """
        Validate observation data integrity.
        
        Args:
            observations: List of observation records
            
        Returns:
            ValidationResult with data integrity status
        """
        issues = []
        
        if not observations:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                "No observations provided for validation",
                "DataIntegrityValidator"
            ))
            return ValidationResult(True, issues)
        
        # Check for required fields
        required_fields = ['episode_id', 'step', 'agent_id']
        for i, obs in enumerate(observations):
            for field in required_fields:
                if field not in obs:
                    issues.append(ValidationIssue(
                        ValidationSeverity.ERROR,
                        f"Missing required field '{field}' in observation {i}",
                        "DataIntegrityValidator",
                        {"observation_index": i, "missing_field": field}
                    ))
        
        # Check for data consistency
        episode_ids = set()
        step_ranges = {}
        agent_ids = set()
        
        for i, obs in enumerate(observations):
            if 'episode_id' in obs:
                episode_ids.add(obs['episode_id'])
            
            if 'step' in obs and 'episode_id' in obs:
                ep_id = obs['episode_id']
                step = obs['step']
                if ep_id not in step_ranges:
                    step_ranges[ep_id] = [step, step]
                else:
                    step_ranges[ep_id][0] = min(step_ranges[ep_id][0], step)
                    step_ranges[ep_id][1] = max(step_ranges[ep_id][1], step)
            
            if 'agent_id' in obs:
                agent_ids.add(obs['agent_id'])
        
        # Validate step sequences
        for ep_id, (min_step, max_step) in step_ranges.items():
            expected_steps = max_step - min_step + 1
            actual_steps = sum(1 for obs in observations 
                             if obs.get('episode_id') == ep_id)
            
            if actual_steps != expected_steps * len(agent_ids):
                issues.append(ValidationIssue(
                    ValidationSeverity.WARNING,
                    f"Incomplete step sequence for episode {ep_id}",
                    "DataIntegrityValidator",
                    {
                        "episode_id": ep_id,
                        "expected_records": expected_steps * len(agent_ids),
                        "actual_records": actual_steps,
                        "step_range": [min_step, max_step],
                        "agent_count": len(agent_ids)
                    }
                ))
        
        # Check for duplicate records
        seen_keys = set()
        for i, obs in enumerate(observations):
            if all(field in obs for field in required_fields):
                key = (obs['episode_id'], obs['step'], obs['agent_id'])
                if key in seen_keys:
                    issues.append(ValidationIssue(
                        ValidationSeverity.ERROR,
                        f"Duplicate observation record found",
                        "DataIntegrityValidator",
                        {
                            "observation_index": i,
                            "duplicate_key": key
                        }
                    ))
                seen_keys.add(key)
        
        is_valid = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          for issue in issues)
        return ValidationResult(is_valid, issues)
    
    def validate_binary_data(self, data: Dict[str, Any], expected_keys: List[str]) -> ValidationResult:
        """
        Validate binary data integrity.
        
        Args:
            data: Dictionary containing binary data
            expected_keys: List of expected binary data keys
            
        Returns:
            ValidationResult with binary data integrity status
        """
        issues = []
        
        for key in expected_keys:
            blob_key = f"{key}_blob"
            shape_key = f"{key}_shape"
            dtype_key = f"{key}_dtype"
            
            # Check if all required keys exist
            if blob_key not in data:
                issues.append(ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Missing binary blob for key '{key}'",
                    "DataIntegrityValidator",
                    {"missing_key": blob_key}
                ))
                continue
            
            if shape_key not in data:
                issues.append(ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Missing shape information for key '{key}'",
                    "DataIntegrityValidator",
                    {"missing_key": shape_key}
                ))
                continue
            
            if dtype_key not in data:
                issues.append(ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Missing dtype information for key '{key}'",
                    "DataIntegrityValidator",
                    {"missing_key": dtype_key}
                ))
                continue
            
            # Validate data consistency
            try:
                blob = data[blob_key]
                shape = data[shape_key]
                dtype = data[dtype_key]
                
                if not isinstance(blob, bytes):
                    issues.append(ValidationIssue(
                        ValidationSeverity.ERROR,
                        f"Binary blob for '{key}' is not bytes type",
                        "DataIntegrityValidator",
                        {"key": key, "actual_type": type(blob).__name__}
                    ))
                
                if not isinstance(shape, (list, tuple)):
                    issues.append(ValidationIssue(
                        ValidationSeverity.ERROR,
                        f"Shape for '{key}' is not list or tuple",
                        "DataIntegrityValidator",
                        {"key": key, "actual_type": type(shape).__name__}
                    ))
                
                # Try to reconstruct array to validate integrity
                if isinstance(blob, bytes) and isinstance(shape, (list, tuple)):
                    expected_size = np.prod(shape) * np.dtype(dtype).itemsize
                    if len(blob) != expected_size:
                        issues.append(ValidationIssue(
                            ValidationSeverity.ERROR,
                            f"Binary data size mismatch for '{key}'",
                            "DataIntegrityValidator",
                            {
                                "key": key,
                                "expected_size": expected_size,
                                "actual_size": len(blob),
                                "shape": shape,
                                "dtype": dtype
                            }
                        ))
                
            except Exception as e:
                issues.append(ValidationIssue(
                    ValidationSeverity.ERROR,
                    f"Failed to validate binary data for '{key}': {str(e)}",
                    "DataIntegrityValidator",
                    {"key": key, "error": str(e)}
                ))
        
        is_valid = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                          for issue in issues)
        return ValidationResult(is_valid, issues)