"""
Unit tests for storage error handling functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import json

from highway_datacollection.storage.manager import DatasetStorageManager
from highway_datacollection.collection.error_handling import StorageError


class TestDatasetStorageManagerErrorHandling:
    """Test cases for DatasetStorageManager error handling."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)
        self.manager = DatasetStorageManager(self.base_path, max_disk_usage_gb=1.0)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization_with_error_handling(self):
        """Test storage manager initialization with error handling components."""
        assert self.manager.error_handler is not None
        assert self.manager.degradation_manager is not None
        assert self.manager.max_disk_usage_bytes == 1.0 * 1024**3
        assert "files_written" in self.manager.storage_stats
    
    @patch('shutil.disk_usage')
    def test_check_disk_space_normal(self, mock_disk_usage):
        """Test disk space checking under normal conditions."""
        # Mock disk usage: 100GB total, 50GB used, 50GB free
        mock_disk_usage.return_value = (100 * 1024**3, 50 * 1024**3, 50 * 1024**3)
        
        disk_info = self.manager.check_disk_space()
        
        assert disk_info["total_disk_gb"] == 100.0
        assert disk_info["used_disk_gb"] == 50.0
        assert disk_info["free_disk_gb"] == 50.0
        assert disk_info["space_available"] is True
        assert disk_info["within_limits"] is True
    
    @patch('shutil.disk_usage')
    def test_check_disk_space_low(self, mock_disk_usage):
        """Test disk space checking with low available space."""
        # Mock disk usage: 100GB total, 99.5GB used, 0.5GB free
        mock_disk_usage.return_value = (100 * 1024**3, int(99.5 * 1024**3), int(0.5 * 1024**3))
        
        disk_info = self.manager.check_disk_space()
        
        assert disk_info["space_available"] is False
    
    def test_write_episode_batch_success(self):
        """Test successful episode batch writing."""
        data = [
            {"episode_id": "ep1", "step": 0, "agent_id": 0, "reward": 1.0},
            {"episode_id": "ep1", "step": 1, "agent_id": 0, "reward": 2.0}
        ]
        metadata = [
            {"episode_id": "ep1", "scenario": "test", "total_steps": 2}
        ]
        
        with patch.object(self.manager, 'check_disk_space') as mock_check:
            mock_check.return_value = {"space_available": True, "within_limits": True}
            
            result = self.manager.write_episode_batch(data, metadata, "test_scenario")
            
            assert result.transitions_file.exists()
            assert result.metadata_file.exists()
            assert self.manager.storage_stats["files_written"] == 2
    
    @patch('pandas.DataFrame.to_parquet')
    def test_write_transitions_data_parquet_failure(self, mock_to_parquet):
        """Test transitions data writing with Parquet failure and CSV fallback."""
        mock_to_parquet.side_effect = Exception("Parquet write failed")
        
        data = [{"episode_id": "ep1", "step": 0, "agent_id": 0, "reward": 1.0}]
        file_path = self.base_path / "test_transitions.parquet"
        
        self.manager._write_transitions_data_safe(data, file_path)
        
        # Should create CSV file instead
        csv_path = file_path.with_suffix('.csv')
        assert csv_path.exists()
        assert self.manager.storage_stats["parquet_failures"] == 1
        assert self.manager.storage_stats["csv_fallbacks"] == 1
        assert self.manager.degradation_manager.is_feature_degraded("parquet_storage")
    
    @patch('pandas.DataFrame.to_parquet')
    @patch('pandas.DataFrame.to_csv')
    def test_write_transitions_data_both_fail(self, mock_to_csv, mock_to_parquet):
        """Test transitions data writing when both Parquet and CSV fail."""
        mock_to_parquet.side_effect = Exception("Parquet failed")
        mock_to_csv.side_effect = Exception("CSV failed")
        
        data = [{"episode_id": "ep1", "step": 0, "agent_id": 0, "reward": 1.0}]
        file_path = self.base_path / "test_transitions.parquet"
        
        with pytest.raises(StorageError, match="Both Parquet and CSV writes failed"):
            self.manager._write_transitions_data_safe(data, file_path)
    
    def test_write_metadata_safe_success(self):
        """Test safe metadata writing."""
        metadata = [
            {"episode_id": "ep1", "scenario": "test"},
            {"episode_id": "ep2", "scenario": "test"}
        ]
        file_path = self.base_path / "test_meta.jsonl"
        
        self.manager._write_metadata_safe(metadata, file_path)
        
        assert file_path.exists()
        
        # Verify content
        with open(file_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert json.loads(lines[0])["episode_id"] == "ep1"
            assert json.loads(lines[1])["episode_id"] == "ep2"
    
    def test_write_metadata_safe_serialization_error(self):
        """Test metadata writing with serialization errors."""
        # Create metadata with non-serializable object
        class NonSerializable:
            pass
        
        metadata = [
            {"episode_id": "ep1", "scenario": "test"},
            {"episode_id": "ep2", "bad_data": NonSerializable()}  # Non-serializable
        ]
        file_path = self.base_path / "test_meta.jsonl"
        
        self.manager._write_metadata_safe(metadata, file_path)
        
        assert file_path.exists()
        
        # Verify that error record was written for bad data
        with open(file_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2
            
            # First record should be normal
            record1 = json.loads(lines[0])
            assert record1["episode_id"] == "ep1"
            
            # Second record should be error placeholder
            record2 = json.loads(lines[1])
            assert record2["error"] == "serialization_failed"
    
    @patch('builtins.open')
    def test_write_metadata_safe_file_error(self, mock_open):
        """Test metadata writing with file I/O error."""
        mock_open.side_effect = IOError("File write failed")
        
        metadata = [{"episode_id": "ep1", "scenario": "test"}]
        file_path = self.base_path / "test_meta.jsonl"
        
        with pytest.raises(StorageError, match="Failed to write metadata file"):
            self.manager._write_metadata_safe(metadata, file_path)
    
    @patch.object(DatasetStorageManager, 'check_disk_space')
    def test_write_episode_batch_insufficient_space(self, mock_check_disk):
        """Test episode batch writing with insufficient disk space."""
        mock_check_disk.return_value = {"space_available": False, "within_limits": False}
        
        data = [{"episode_id": "ep1", "step": 0, "agent_id": 0, "reward": 1.0}]
        metadata = [{"episode_id": "ep1", "scenario": "test"}]
        
        with pytest.raises(StorageError, match="Insufficient disk space"):
            self.manager.write_episode_batch(data, metadata, "test_scenario")
    
    def test_get_storage_statistics(self):
        """Test storage statistics retrieval."""
        # Perform some operations to generate statistics
        data = [{"episode_id": "ep1", "step": 0, "agent_id": 0, "reward": 1.0}]
        metadata = [{"episode_id": "ep1", "scenario": "test"}]
        
        with patch.object(self.manager, 'check_disk_space') as mock_check:
            mock_check.return_value = {"space_available": True, "within_limits": True}
            self.manager.write_episode_batch(data, metadata, "test_scenario")
        
        stats = self.manager.get_storage_statistics()
        
        assert "files_written" in stats
        assert "disk_info" in stats
        assert "degraded_features" in stats
        assert "error_stats" in stats
        assert stats["files_written"] == 2
    
    def test_reset_statistics(self):
        """Test statistics reset."""
        # Generate some statistics
        self.manager.storage_stats["files_written"] = 10
        self.manager.storage_stats["storage_errors"] = 5
        
        self.manager.reset_statistics()
        
        assert self.manager.storage_stats["files_written"] == 0
        assert self.manager.storage_stats["storage_errors"] == 0
    
    def test_perform_maintenance(self):
        """Test storage maintenance operations."""
        # Create some test files and directories
        test_scenario_dir = self.base_path / "test_scenario"
        test_scenario_dir.mkdir()
        
        # Create empty file
        empty_file = test_scenario_dir / "empty.txt"
        empty_file.touch()
        
        maintenance_result = self.manager.perform_maintenance()
        
        assert "cleanup_performed" in maintenance_result
        assert "errors_found" in maintenance_result
        assert "files_removed" in maintenance_result
    
    def test_validate_dataset_integrity(self):
        """Test dataset integrity validation."""
        # Create test dataset structure
        scenario_dir = self.base_path / "test_scenario"
        scenario_dir.mkdir()
        
        # Create matching transitions and metadata files
        transitions_file = scenario_dir / "20240101_120000-abcd1234_transitions.parquet"
        metadata_file = scenario_dir / "20240101_120000-abcd1234_meta.jsonl"
        
        # Create test data
        df = pd.DataFrame([{"episode_id": "ep1", "step": 0, "reward": 1.0}])
        df.to_parquet(transitions_file, index=False)
        
        with open(metadata_file, 'w') as f:
            json.dump({"episode_id": "ep1", "scenario": "test"}, f)
            f.write('\n')
        
        validation = self.manager.validate_dataset_integrity()
        
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
        assert validation["statistics"]["total_scenarios"] == 1
    
    def test_validate_dataset_integrity_mismatch(self):
        """Test dataset integrity validation with file mismatches."""
        # Create test dataset structure with mismatched files
        scenario_dir = self.base_path / "test_scenario"
        scenario_dir.mkdir()
        
        # Create only transitions file (missing metadata)
        transitions_file = scenario_dir / "20240101_120000-abcd1234_transitions.parquet"
        df = pd.DataFrame([{"episode_id": "ep1", "step": 0, "reward": 1.0}])
        df.to_parquet(transitions_file, index=False)
        
        validation = self.manager.validate_dataset_integrity()
        
        assert validation["valid"] is False
        assert len(validation["errors"]) > 0
        assert any("mismatch" in error.lower() for error in validation["errors"])
    
    def test_cleanup_empty_directories(self):
        """Test cleanup of empty directories."""
        # Create empty directory
        empty_dir = self.base_path / "empty_scenario"
        empty_dir.mkdir()
        
        # Create directory with empty file
        empty_file_dir = self.base_path / "empty_file_scenario"
        empty_file_dir.mkdir()
        (empty_file_dir / "empty.txt").touch()
        
        # Create directory with non-empty file
        non_empty_dir = self.base_path / "non_empty_scenario"
        non_empty_dir.mkdir()
        with open(non_empty_dir / "data.txt", 'w') as f:
            f.write("some data")
        
        removed_dirs = self.manager.cleanup_empty_directories()
        
        assert "empty_scenario" in removed_dirs
        assert "empty_file_scenario" in removed_dirs
        assert "non_empty_scenario" not in removed_dirs
        
        # Verify directories were actually removed
        assert not empty_dir.exists()
        assert not empty_file_dir.exists()
        assert non_empty_dir.exists()


if __name__ == "__main__":
    pytest.main([__file__])