"""
Unit tests for error handling and validation functionality.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import psutil

from highway_datacollection.collection.error_handling import (
    ErrorHandler, ErrorContext, ErrorSeverity, RecoveryAction,
    DataCollectionError, EnvironmentSynchronizationError, StorageError,
    MemoryError, ValidationError, GracefulDegradationManager
)
from highway_datacollection.collection.validation import (
    EnvironmentSynchronizationValidator, MemoryValidator, DataIntegrityValidator,
    ValidationResult, ValidationSeverity, ValidationIssue
)


class TestErrorHandler:
    """Test cases for ErrorHandler class."""
    
    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = ErrorHandler(max_retries=5, retry_delay=0.5)
        assert handler.max_retries == 5
        assert handler.retry_delay == 0.5
        assert len(handler.error_history) == 0
        assert len(handler.recovery_strategies) > 0  # Default strategies registered
    
    def test_handle_error_basic(self):
        """Test basic error handling."""
        handler = ErrorHandler()
        error = DataCollectionError("Test error", ErrorSeverity.MEDIUM)
        context = ErrorContext("test_operation", "test_component")
        
        result = handler.handle_error(error, context)
        
        assert result["error_type"] == "DataCollectionError"
        assert result["error_message"] == "Test error"
        assert result["severity"] == "medium"
        assert result["context"] == context
        assert len(handler.error_history) == 1
    
    def test_handle_error_with_recovery(self):
        """Test error handling with successful recovery."""
        handler = ErrorHandler()
        
        # Mock recovery action
        recovery_called = False
        def mock_recovery():
            nonlocal recovery_called
            recovery_called = True
            return True
        
        recovery_action = RecoveryAction(
            "test_recovery", "Test recovery", mock_recovery, 0.8
        )
        
        error = DataCollectionError("Test error")
        context = ErrorContext("test_operation", "test_component")
        
        result = handler.handle_error(error, context, [recovery_action])
        
        assert result["recovery_attempted"] is True
        assert result["recovery_successful"] is True
        assert recovery_called is True
        assert len(result["recovery_actions_tried"]) == 1
    
    def test_handle_error_recovery_failure(self):
        """Test error handling with failed recovery."""
        handler = ErrorHandler()
        
        def failing_recovery():
            raise Exception("Recovery failed")
        
        recovery_action = RecoveryAction(
            "failing_recovery", "Failing recovery", failing_recovery, 0.5
        )
        
        error = DataCollectionError("Test error")
        context = ErrorContext("test_operation", "test_component")
        
        result = handler.handle_error(error, context, [recovery_action])
        
        assert result["recovery_attempted"] is True
        assert result["recovery_successful"] is False
        assert len(result["recovery_actions_tried"]) == 1
        assert result["recovery_actions_tried"][0]["success"] is False
    
    def test_register_recovery_strategy(self):
        """Test registering custom recovery strategies."""
        handler = ErrorHandler()
        
        def custom_recovery():
            return True
        
        strategy = RecoveryAction("custom", "Custom recovery", custom_recovery, 0.9)
        handler.register_recovery_strategy(CustomError, strategy)
        
        assert CustomError in handler.recovery_strategies
        assert len(handler.recovery_strategies[CustomError]) == 1
    
    def test_get_error_statistics(self):
        """Test error statistics generation."""
        handler = ErrorHandler()
        
        # Add some test errors
        error1 = DataCollectionError("Error 1")
        error2 = StorageError("Error 2")
        context = ErrorContext("test", "test")
        
        handler.handle_error(error1, context)
        handler.handle_error(error2, context)
        
        stats = handler.get_error_statistics()
        
        assert stats["total_errors"] == 2
        assert "DataCollectionError" in stats["error_counts_by_type"]
        assert "StorageError" in stats["error_counts_by_type"]
        assert stats["error_counts_by_type"]["DataCollectionError"] == 1
        assert stats["error_counts_by_type"]["StorageError"] == 1


class CustomError(Exception):
    """Custom error for testing."""
    pass


class TestEnvironmentSynchronizationValidator:
    """Test cases for EnvironmentSynchronizationValidator."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = EnvironmentSynchronizationValidator(tolerance=1e-5)
        assert validator.tolerance == 1e-5
        assert len(validator.sync_history) == 0
        assert validator.desync_count == 0
    
    def test_validate_synchronized_environments(self):
        """Test validation of synchronized environments."""
        validator = EnvironmentSynchronizationValidator()
        
        step_results = {
            "Kinematics": {
                "reward": 1.0,
                "terminated": False,
                "truncated": False
            },
            "OccupancyGrid": {
                "reward": 1.0,
                "terminated": False,
                "truncated": False
            }
        }
        
        result = validator.validate_step_synchronization(step_results)
        
        assert result.is_valid is True
        assert len(result.issues) == 0
        assert validator.sync_history[-1] is True
    
    def test_validate_desynchronized_environments(self):
        """Test validation of desynchronized environments."""
        validator = EnvironmentSynchronizationValidator()
        
        step_results = {
            "Kinematics": {
                "reward": 1.0,
                "terminated": False,
                "truncated": False
            },
            "OccupancyGrid": {
                "reward": 2.0,  # Different reward
                "terminated": False,
                "truncated": False
            }
        }
        
        result = validator.validate_step_synchronization(step_results)
        
        assert result.is_valid is False
        assert len(result.issues) > 0
        assert any(issue.severity == ValidationSeverity.ERROR for issue in result.issues)
        assert validator.sync_history[-1] is False
    
    def test_critical_desynchronization_detection(self):
        """Test detection of critical desynchronization patterns."""
        validator = EnvironmentSynchronizationValidator()
        validator.max_desync_threshold = 2
        
        step_results = {
            "Kinematics": {"reward": 1.0, "terminated": False, "truncated": False},
            "OccupancyGrid": {"reward": 2.0, "terminated": False, "truncated": False}
        }
        
        # Trigger multiple desyncs
        for _ in range(3):
            result = validator.validate_step_synchronization(step_results)
        
        # Last result should have critical issue
        assert any(issue.severity == ValidationSeverity.CRITICAL for issue in result.issues)
    
    def test_empty_step_results(self):
        """Test validation with empty step results."""
        validator = EnvironmentSynchronizationValidator()
        
        result = validator.validate_step_synchronization({})
        
        assert len(result.issues) == 1
        assert result.issues[0].severity == ValidationSeverity.ERROR
    
    def test_single_environment(self):
        """Test validation with single environment."""
        validator = EnvironmentSynchronizationValidator()
        
        step_results = {
            "Kinematics": {
                "reward": 1.0,
                "terminated": False,
                "truncated": False
            }
        }
        
        result = validator.validate_step_synchronization(step_results)
        
        assert result.is_valid is True
        assert len(result.issues) == 1  # Warning about single environment
        assert result.issues[0].severity == ValidationSeverity.WARNING
    
    def test_get_synchronization_stats(self):
        """Test synchronization statistics."""
        validator = EnvironmentSynchronizationValidator()
        
        # Add some history
        validator.sync_history = [True, True, False, True, False]
        
        stats = validator.get_synchronization_stats()
        
        assert stats["total_checks"] == 5
        assert stats["successful_syncs"] == 3
        assert stats["failed_syncs"] == 2
        assert stats["success_rate"] == 0.6


class TestMemoryValidator:
    """Test cases for MemoryValidator."""
    
    def test_validator_initialization(self):
        """Test memory validator initialization."""
        validator = MemoryValidator(max_memory_gb=4.0, warning_threshold=0.7)
        assert validator.max_memory_bytes == 4.0 * 1024**3
        assert validator.warning_threshold == 0.7
        assert len(validator.memory_history) == 0
    
    @patch('psutil.Process')
    @patch('psutil.virtual_memory')
    def test_validate_memory_usage_normal(self, mock_virtual_memory, mock_process):
        """Test memory validation under normal conditions."""
        # Mock memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024**3  # 1GB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        mock_virtual_memory.return_value.percent = 50.0
        mock_virtual_memory.return_value.available = 4 * 1024**3
        
        validator = MemoryValidator(max_memory_gb=4.0)
        result = validator.validate_memory_usage()
        
        assert result.is_valid is True
        assert len(validator.memory_history) == 1
    
    @patch('psutil.Process')
    @patch('psutil.virtual_memory')
    def test_validate_memory_usage_warning(self, mock_virtual_memory, mock_process):
        """Test memory validation with warning threshold exceeded."""
        # Mock memory info - 3.5GB usage with 4GB limit and 0.8 threshold
        mock_memory_info = Mock()
        mock_memory_info.rss = int(3.5 * 1024**3)
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        mock_virtual_memory.return_value.percent = 70.0
        mock_virtual_memory.return_value.available = 2 * 1024**3
        
        validator = MemoryValidator(max_memory_gb=4.0, warning_threshold=0.8)
        result = validator.validate_memory_usage()
        
        assert result.is_valid is True
        assert result.has_warnings() is True
        warning_issues = result.get_issues_by_severity(ValidationSeverity.WARNING)
        assert len(warning_issues) > 0
    
    @patch('psutil.Process')
    @patch('psutil.virtual_memory')
    def test_validate_memory_usage_critical(self, mock_virtual_memory, mock_process):
        """Test memory validation with critical memory usage."""
        # Mock memory info - 5GB usage with 4GB limit
        mock_memory_info = Mock()
        mock_memory_info.rss = 5 * 1024**3
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        mock_virtual_memory.return_value.percent = 95.0
        mock_virtual_memory.return_value.available = 0.5 * 1024**3
        
        validator = MemoryValidator(max_memory_gb=4.0)
        result = validator.validate_memory_usage()
        
        assert result.is_valid is False
        assert result.has_errors() is True
        critical_issues = result.get_issues_by_severity(ValidationSeverity.CRITICAL)
        assert len(critical_issues) > 0
    
    @patch('psutil.Process')
    def test_memory_leak_detection(self, mock_process):
        """Test memory leak detection."""
        validator = MemoryValidator(max_memory_gb=8.0)
        
        # Simulate growing memory usage
        memory_values = [1024**3 * (1 + i * 0.2) for i in range(15)]  # Growing memory
        
        for memory_val in memory_values:
            mock_memory_info = Mock()
            mock_memory_info.rss = int(memory_val)
            mock_process.return_value.memory_info.return_value = mock_memory_info
            
            with patch('psutil.virtual_memory') as mock_virtual_memory:
                mock_virtual_memory.return_value.percent = 50.0
                mock_virtual_memory.return_value.available = 4 * 1024**3
                validator.validate_memory_usage()
        
        # Last validation should detect potential memory leak
        result = validator.validate_memory_usage()
        warning_issues = result.get_issues_by_severity(ValidationSeverity.WARNING)
        leak_warnings = [issue for issue in warning_issues if "memory leak" in issue.message.lower()]
        assert len(leak_warnings) > 0
    
    def test_get_memory_stats(self):
        """Test memory statistics generation."""
        validator = MemoryValidator(max_memory_gb=4.0)
        
        # Add some memory history
        validator.memory_history = [1024**3, 1.5*1024**3, 2*1024**3]
        
        stats = validator.get_memory_stats()
        
        assert stats["current_memory_gb"] == 2.0
        assert stats["max_memory_gb"] == 2.0
        assert stats["min_memory_gb"] == 1.0
        assert stats["memory_limit_gb"] == 4.0
        assert stats["samples_count"] == 3


class TestDataIntegrityValidator:
    """Test cases for DataIntegrityValidator."""
    
    def test_validate_observation_data_valid(self):
        """Test validation of valid observation data."""
        validator = DataIntegrityValidator()
        
        observations = [
            {"episode_id": "ep1", "step": 0, "agent_id": 0, "reward": 1.0},
            {"episode_id": "ep1", "step": 0, "agent_id": 1, "reward": 1.0},
            {"episode_id": "ep1", "step": 1, "agent_id": 0, "reward": 2.0},
            {"episode_id": "ep1", "step": 1, "agent_id": 1, "reward": 2.0},
        ]
        
        result = validator.validate_observation_data(observations)
        
        assert result.is_valid is True
        assert not result.has_errors()
    
    def test_validate_observation_data_missing_fields(self):
        """Test validation with missing required fields."""
        validator = DataIntegrityValidator()
        
        observations = [
            {"episode_id": "ep1", "step": 0},  # Missing agent_id
            {"step": 0, "agent_id": 1},        # Missing episode_id
        ]
        
        result = validator.validate_observation_data(observations)
        
        assert result.is_valid is False
        assert result.has_errors()
        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        assert len(error_issues) == 2
    
    def test_validate_observation_data_duplicates(self):
        """Test validation with duplicate records."""
        validator = DataIntegrityValidator()
        
        observations = [
            {"episode_id": "ep1", "step": 0, "agent_id": 0, "reward": 1.0},
            {"episode_id": "ep1", "step": 0, "agent_id": 0, "reward": 1.0},  # Duplicate
        ]
        
        result = validator.validate_observation_data(observations)
        
        assert result.is_valid is False
        assert result.has_errors()
        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        duplicate_errors = [issue for issue in error_issues if "duplicate" in issue.message.lower()]
        assert len(duplicate_errors) > 0
    
    def test_validate_binary_data_valid(self):
        """Test validation of valid binary data."""
        validator = DataIntegrityValidator()
        
        # Create test binary data
        test_array = np.random.rand(10, 10).astype(np.float32)
        test_blob = test_array.tobytes()
        
        data = {
            "test_blob": test_blob,
            "test_shape": [10, 10],
            "test_dtype": "float32"
        }
        
        result = validator.validate_binary_data(data, ["test"])
        
        assert result.is_valid is True
        assert not result.has_errors()
    
    def test_validate_binary_data_missing_keys(self):
        """Test validation with missing binary data keys."""
        validator = DataIntegrityValidator()
        
        data = {
            "test_blob": b"some_data",
            # Missing shape and dtype
        }
        
        result = validator.validate_binary_data(data, ["test"])
        
        assert result.is_valid is False
        assert result.has_errors()
        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        assert len(error_issues) == 2  # Missing shape and dtype
    
    def test_validate_binary_data_size_mismatch(self):
        """Test validation with binary data size mismatch."""
        validator = DataIntegrityValidator()
        
        data = {
            "test_blob": b"wrong_size_data",  # Wrong size
            "test_shape": [10, 10],
            "test_dtype": "float32"
        }
        
        result = validator.validate_binary_data(data, ["test"])
        
        assert result.is_valid is False
        assert result.has_errors()
        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        size_errors = [issue for issue in error_issues if "size mismatch" in issue.message.lower()]
        assert len(size_errors) > 0


class TestGracefulDegradationManager:
    """Test cases for GracefulDegradationManager."""
    
    def test_manager_initialization(self):
        """Test degradation manager initialization."""
        manager = GracefulDegradationManager()
        assert len(manager.degraded_features) == 0
        assert len(manager.fallback_strategies) == 0
    
    def test_register_fallback(self):
        """Test registering fallback strategies."""
        manager = GracefulDegradationManager()
        
        def test_fallback():
            return "fallback_result"
        
        manager.register_fallback("test_feature", test_fallback, "Test fallback")
        
        assert "test_feature" in manager.fallback_strategies
        assert manager.fallback_strategies["test_feature"] == test_fallback
    
    def test_degrade_and_restore_feature(self):
        """Test degrading and restoring features."""
        manager = GracefulDegradationManager()
        
        # Degrade feature
        manager.degrade_feature("test_feature", "Test reason", ErrorSeverity.HIGH)
        
        assert manager.is_feature_degraded("test_feature") is True
        assert "test_feature" in manager.degraded_features
        assert manager.degraded_features["test_feature"]["reason"] == "Test reason"
        
        # Restore feature
        manager.restore_feature("test_feature")
        
        assert manager.is_feature_degraded("test_feature") is False
        assert "test_feature" not in manager.degraded_features
    
    def test_execute_with_fallback_normal(self):
        """Test executing function normally when feature is not degraded."""
        manager = GracefulDegradationManager()
        
        def primary_func(x):
            return x * 2
        
        result = manager.execute_with_fallback("test_feature", primary_func, 5)
        
        assert result == 10
    
    def test_execute_with_fallback_degraded(self):
        """Test executing with fallback when feature is degraded."""
        manager = GracefulDegradationManager()
        
        def primary_func(x):
            return x * 2
        
        def fallback_func(x):
            return x + 1
        
        manager.register_fallback("test_feature", fallback_func)
        manager.degrade_feature("test_feature", "Test degradation")
        
        result = manager.execute_with_fallback("test_feature", primary_func, 5)
        
        assert result == 6  # Fallback result
    
    def test_execute_with_fallback_no_fallback(self):
        """Test executing when feature is degraded but no fallback exists."""
        manager = GracefulDegradationManager()
        
        def primary_func(x):
            return x * 2
        
        manager.degrade_feature("test_feature", "Test degradation")
        
        with pytest.raises(RuntimeError, match="no fallback is available"):
            manager.execute_with_fallback("test_feature", primary_func, 5)
    
    def test_get_degradation_status(self):
        """Test getting degradation status."""
        manager = GracefulDegradationManager()
        
        def test_fallback():
            return "test"
        
        manager.register_fallback("feature1", test_fallback)
        manager.degrade_feature("feature2", "Test reason")
        
        status = manager.get_degradation_status()
        
        assert status["total_degraded"] == 1
        assert "feature2" in status["degraded_features"]
        assert "feature1" in status["available_fallbacks"]


if __name__ == "__main__":
    pytest.main([__file__])