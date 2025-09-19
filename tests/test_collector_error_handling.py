"""
Unit tests for collector error handling functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

from highway_datacollection.collection.collector import SynchronizedCollector
from highway_datacollection.collection.error_handling import (
    EnvironmentSynchronizationError, MemoryError, ValidationError
)
from highway_datacollection.collection.validation import ValidationSeverity
from highway_datacollection.collection.action_samplers import RandomActionSampler
from highway_datacollection.collection.modality_config import ModalityConfigManager


class TestSynchronizedCollectorErrorHandling:
    """Test cases for SynchronizedCollector error handling."""
    
    def setup_method(self):
        """Set up test environment."""
        self.collector = SynchronizedCollector(
            n_agents=2, 
            enable_validation=True,
            max_memory_gb=4.0
        )
    
    def test_initialization_with_error_handling(self):
        """Test collector initialization with error handling components."""
        assert self.collector.error_handler is not None
        assert self.collector.degradation_manager is not None
        assert self.collector.enable_validation is True
        assert hasattr(self.collector, 'sync_validator')
        assert hasattr(self.collector, 'memory_validator')
        assert hasattr(self.collector, 'data_validator')
        assert "episodes_collected" in self.collector.collection_stats
    
    def test_initialization_without_validation(self):
        """Test collector initialization with validation disabled."""
        collector = SynchronizedCollector(n_agents=2, enable_validation=False)
        
        assert collector.enable_validation is False
        assert not hasattr(collector, 'sync_validator')
        assert not hasattr(collector, 'memory_validator')
        assert not hasattr(collector, 'data_validator')
    
    def test_verify_synchronization_success(self):
        """Test successful synchronization verification."""
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
        
        result = self.collector.verify_synchronization(step_results)
        
        assert result is True
        assert self.collector.collection_stats["sync_failures"] == 0
    
    def test_verify_synchronization_failure(self):
        """Test synchronization verification failure."""
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
        
        result = self.collector.verify_synchronization(step_results)
        
        assert result is False
        assert self.collector.collection_stats["sync_failures"] > 0
    
    def test_verify_synchronization_critical_failure(self):
        """Test critical synchronization failure."""
        # Mock validator to return critical issue
        mock_issue = Mock()
        mock_issue.severity = ValidationSeverity.CRITICAL
        mock_issue.message = "Critical sync failure"
        mock_issue.details = {"test": "data"}
        
        mock_result = Mock()
        mock_result.is_valid = False
        mock_result.issues = [mock_issue]
        
        with patch.object(self.collector.sync_validator, 'validate_step_synchronization') as mock_validate:
            mock_validate.return_value = mock_result
            
            step_results = {"Kinematics": {"reward": 1.0, "terminated": False, "truncated": False}}
            
            with pytest.raises(EnvironmentSynchronizationError, match="Critical synchronization failure"):
                self.collector.verify_synchronization(step_results)
    
    def test_verify_synchronization_without_validation(self):
        """Test synchronization verification with validation disabled."""
        collector = SynchronizedCollector(n_agents=2, enable_validation=False)
        
        step_results = {
            "Kinematics": {"reward": 1.0, "terminated": False, "truncated": False},
            "OccupancyGrid": {"reward": 2.0, "terminated": False, "truncated": False}
        }
        
        result = collector.verify_synchronization(step_results)
        
        assert result is False  # Should use basic check
    
    @patch('highway_datacollection.collection.collector.MultiAgentEnvFactory')
    def test_setup_environments_with_error_handling(self, mock_factory_class):
        """Test environment setup with error handling."""
        mock_factory = Mock()
        mock_factory.create_parallel_envs.side_effect = Exception("Environment creation failed")
        mock_factory_class.return_value = mock_factory
        
        collector = SynchronizedCollector(n_agents=2)
        
        with pytest.raises(Exception, match="Environment creation failed"):
            collector._setup_environments("test_scenario")
    
    @patch('psutil.Process')
    @patch('psutil.virtual_memory')
    def test_collect_episode_batch_memory_validation(self, mock_virtual_memory, mock_process):
        """Test episode batch collection with memory validation."""
        # Mock high memory usage
        mock_memory_info = Mock()
        mock_memory_info.rss = 5 * 1024**3  # 5GB (exceeds 4GB limit)
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        mock_virtual_memory.return_value.percent = 95.0
        mock_virtual_memory.return_value.available = 0.5 * 1024**3
        
        with pytest.raises(MemoryError):
            self.collector.collect_episode_batch("test_scenario", 1, 42, 10)
    
    @patch('highway_datacollection.collection.collector.SynchronizedCollector._setup_environments')
    def test_collect_episode_batch_setup_failure_recovery(self, mock_setup):
        """Test episode batch collection with setup failure and recovery."""
        # First call fails, second succeeds
        mock_setup.side_effect = [Exception("Setup failed"), None]
        
        # Mock successful recovery
        def mock_recovery():
            mock_setup.side_effect = [None]  # Next call will succeed
            return True
        
        with patch.object(self.collector.error_handler, 'handle_error') as mock_handle:
            mock_handle.return_value = {"recovery_successful": True}
            
            # This should not raise an exception due to recovery
            with patch.object(self.collector, '_collect_single_episode_safe') as mock_collect:
                mock_collect.return_value = Mock()
                result = self.collector.collect_episode_batch("test_scenario", 1, 42, 10)
                
                assert result.successful_episodes >= 0  # Should not crash
    
    def test_collect_single_episode_safe_with_recovery(self):
        """Test safe single episode collection with error recovery."""
        # Mock the actual collection method to fail first, then succeed
        episode_data = Mock()
        
        with patch.object(self.collector, '_collect_single_episode') as mock_collect:
            mock_collect.side_effect = [Exception("Collection failed"), episode_data]
            
            with patch.object(self.collector.error_handler, 'handle_error') as mock_handle:
                mock_handle.return_value = {"recovery_successful": True}
                
                result = self.collector._collect_single_episode_safe("test", 42, 100, 0)
                
                assert result == episode_data
                assert self.collector.collection_stats["recovery_attempts"] == 1
    
    def test_collect_single_episode_safe_recovery_failure(self):
        """Test safe single episode collection with failed recovery."""
        with patch.object(self.collector, '_collect_single_episode') as mock_collect:
            mock_collect.side_effect = Exception("Collection failed")
            
            with patch.object(self.collector.error_handler, 'handle_error') as mock_handle:
                mock_handle.return_value = {"recovery_successful": False}
                
                with pytest.raises(Exception, match="Collection failed"):
                    self.collector._collect_single_episode_safe("test", 42, 100, 0)
    
    def test_get_collection_statistics(self):
        """Test collection statistics retrieval."""
        # Set some test statistics
        self.collector.collection_stats["episodes_collected"] = 5
        self.collector.collection_stats["sync_failures"] = 2
        
        stats = self.collector.get_collection_statistics()
        
        assert stats["episodes_collected"] == 5
        assert stats["sync_failures"] == 2
        assert "synchronization_stats" in stats
        assert "memory_stats" in stats
        assert "error_stats" in stats
        assert "degradation_status" in stats
    
    def test_get_collection_statistics_without_validation(self):
        """Test collection statistics without validation enabled."""
        collector = SynchronizedCollector(n_agents=2, enable_validation=False)
        collector.collection_stats["episodes_collected"] = 3
        
        stats = collector.get_collection_statistics()
        
        assert stats["episodes_collected"] == 3
        assert "error_stats" in stats
        assert "degradation_status" in stats
        # Should not have validation-specific stats
        assert "synchronization_stats" not in stats
        assert "memory_stats" not in stats
    
    def test_reset_statistics(self):
        """Test statistics reset."""
        # Set some test statistics
        self.collector.collection_stats["episodes_collected"] = 10
        self.collector.collection_stats["sync_failures"] = 5
        
        self.collector.reset_statistics()
        
        assert self.collector.collection_stats["episodes_collected"] == 0
        assert self.collector.collection_stats["sync_failures"] == 0
    
    def test_validate_episode_data(self):
        """Test episode data validation."""
        from highway_datacollection.collection.types import EpisodeData
        
        episode_data = EpisodeData(
            episode_id="ep1",
            scenario="test",
            observations=[[
                {"episode_id": "ep1", "step": 0, "agent_id": 0},
                {"episode_id": "ep1", "step": 0, "agent_id": 1}
            ]],
            actions=[1, 2],
            rewards=[1.0, 2.0],
            dones=[False, True],
            infos=[{}, {}],
            metadata={"test": "data"}
        )
        
        result = self.collector.validate_episode_data(episode_data)
        
        assert result.is_valid is True
    
    def test_validate_episode_data_without_validation(self):
        """Test episode data validation with validation disabled."""
        collector = SynchronizedCollector(n_agents=2, enable_validation=False)
        
        from highway_datacollection.collection.types import EpisodeData
        episode_data = EpisodeData(
            episode_id="ep1", scenario="test", observations=[], actions=[],
            rewards=[], dones=[], infos=[], metadata={}
        )
        
        result = collector.validate_episode_data(episode_data)
        
        assert result.is_valid is True  # Should always return valid when disabled
    
    def test_perform_health_check(self):
        """Test comprehensive health check."""
        # Set up some test conditions
        self.collector._environments = {"Kinematics": Mock()}
        
        health_status = self.collector.perform_health_check()
        
        assert "overall_healthy" in health_status
        assert "environments_ready" in health_status
        assert "validation_enabled" in health_status
        assert "issues" in health_status
        assert "statistics" in health_status
        
        assert health_status["environments_ready"] is True
        assert health_status["validation_enabled"] is True
    
    def test_perform_health_check_no_environments(self):
        """Test health check with no environments."""
        health_status = self.collector.perform_health_check()
        
        assert health_status["overall_healthy"] is False
        assert health_status["environments_ready"] is False
        assert "No environments initialized" in health_status["issues"]
    
    @patch('psutil.Process')
    @patch('psutil.virtual_memory')
    def test_perform_health_check_memory_issues(self, mock_virtual_memory, mock_process):
        """Test health check with memory issues."""
        # Mock high memory usage
        mock_memory_info = Mock()
        mock_memory_info.rss = 5 * 1024**3  # 5GB (exceeds 4GB limit)
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        mock_virtual_memory.return_value.percent = 95.0
        mock_virtual_memory.return_value.available = 0.5 * 1024**3
        
        health_status = self.collector.perform_health_check()
        
        assert health_status["overall_healthy"] is False
        assert any("Memory issue" in issue for issue in health_status["issues"])
    
    def test_perform_health_check_degraded_features(self):
        """Test health check with degraded features."""
        # Degrade a feature
        self.collector.degradation_manager.degrade_feature("test_feature", "Test degradation")
        
        health_status = self.collector.perform_health_check()
        
        assert any("features are degraded" in issue for issue in health_status["issues"])
    
    def test_recovery_strategies_registration(self):
        """Test that recovery strategies are properly registered."""
        from highway_datacollection.collection.error_handling import EnvironmentSynchronizationError, MemoryError
        
        assert EnvironmentSynchronizationError in self.collector.error_handler.recovery_strategies
        assert MemoryError in self.collector.error_handler.recovery_strategies
        
        sync_strategies = self.collector.error_handler.recovery_strategies[EnvironmentSynchronizationError]
        memory_strategies = self.collector.error_handler.recovery_strategies[MemoryError]
        
        assert len(sync_strategies) >= 2  # reset_environments, recreate_environments
        assert len(memory_strategies) >= 1  # memory_cleanup
    
    @patch('gc.collect')
    def test_memory_cleanup_recovery(self, mock_gc_collect):
        """Test memory cleanup recovery strategy."""
        mock_gc_collect.return_value = 100  # Objects collected
        
        # Trigger memory cleanup through validator
        if hasattr(self.collector, 'memory_validator'):
            with patch('psutil.Process') as mock_process:
                mock_memory_info = Mock()
                mock_memory_info.rss = 2 * 1024**3  # 2GB
                mock_process.return_value.memory_info.return_value = mock_memory_info
                
                gc_result = self.collector.memory_validator.trigger_garbage_collection()
                
                assert "objects_collected" in gc_result
                assert gc_result["objects_collected"] == 100


if __name__ == "__main__":
    pytest.main([__file__])