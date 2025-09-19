"""
Unit tests for dataset organization and indexing functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from datetime import datetime

from highway_datacollection.storage.manager import DatasetStorageManager
from highway_datacollection.storage.types import StoragePaths, EpisodeMetadata


class TestDatasetOrganization:
    """Test dataset organization and indexing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_manager = DatasetStorageManager(Path(self.temp_dir))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_scenario_data(self, scenario_name: str, num_episodes: int = 2) -> List[StoragePaths]:
        """Helper to create test scenario data."""
        paths_list = []
        
        for i in range(num_episodes):
            data = [
                {
                    'episode_id': f'ep_{scenario_name}_{i}',
                    'step': j,
                    'agent_id': 0,
                    'action': j % 3,
                    'reward': 0.1 * j,
                    'kin_x': 10.0 + j,
                    'kin_y': 5.0,
                    'ttc': 2.5 - j * 0.1,
                    'summary_text': f'Step {j} in {scenario_name}'
                }
                for j in range(5)  # 5 steps per episode
            ]
            
            metadata = [
                {
                    'episode_id': f'ep_{scenario_name}_{i}',
                    'scenario': scenario_name,
                    'config': {'vehicles_count': 10 + i, 'lanes_count': 3},
                    'modalities': ['kinematics', 'occupancy'],
                    'n_agents': 1,
                    'total_steps': 5,
                    'seed': 42 + i
                }
            ]
            
            paths = self.storage_manager.write_episode_batch(data, metadata, scenario_name)
            paths_list.append(paths)
        
        return paths_list
    
    def test_scenario_directory_creation(self):
        """Test that scenario directories are created with proper structure."""
        scenario_names = ['free_flow', 'dense_commuting', 'lane_closure']
        
        for scenario in scenario_names:
            self._create_test_scenario_data(scenario, num_episodes=1)
        
        # Check directory structure
        for scenario in scenario_names:
            scenario_dir = self.storage_manager.base_path / scenario
            assert scenario_dir.exists()
            assert scenario_dir.is_dir()
            
            # Check for expected files
            files = list(scenario_dir.iterdir())
            assert len(files) >= 2  # At least transitions and metadata files
            
            # Check file naming patterns
            transitions_files = [f for f in files if 'transitions' in f.name]
            metadata_files = [f for f in files if 'meta' in f.name]
            
            assert len(transitions_files) >= 1
            assert len(metadata_files) >= 1
    
    def test_unique_episode_id_generation(self):
        """Test that episode IDs are unique across scenarios and time."""
        # Generate multiple episode IDs
        ids = []
        for _ in range(100):
            episode_id = self.storage_manager.generate_episode_id()
            ids.append(episode_id)
        
        # Check uniqueness
        assert len(set(ids)) == len(ids)
        
        # Check format
        for episode_id in ids:
            assert episode_id.startswith('ep_')
            parts = episode_id.split('_')
            assert len(parts) >= 3
    
    def test_file_path_management(self):
        """Test file path generation and management."""
        scenario_name = 'test_scenario'
        paths = self._create_test_scenario_data(scenario_name, num_episodes=1)[0]
        
        # Check path structure
        assert paths.scenario_dir.name == scenario_name
        assert paths.scenario_dir.parent == self.storage_manager.base_path
        
        # Check file naming
        assert 'transitions' in paths.transitions_file.name
        assert 'meta' in paths.metadata_file.name
        
        # Check that files exist
        assert paths.transitions_file.exists()
        assert paths.metadata_file.exists()
        
        # Check file extensions
        assert paths.transitions_file.suffix in ['.parquet', '.csv']
        assert paths.metadata_file.suffix == '.jsonl'
    
    def test_organize_dataset_structure(self):
        """Test dataset structure organization analysis."""
        # Create test data for multiple scenarios
        scenarios = ['free_flow', 'dense_commuting', 'lane_closure']
        episodes_per_scenario = [2, 3, 1]
        
        for scenario, num_episodes in zip(scenarios, episodes_per_scenario):
            self._create_test_scenario_data(scenario, num_episodes)
        
        # Analyze structure
        structure = self.storage_manager.organize_dataset_structure()
        
        # Check top-level structure
        assert 'base_path' in structure
        assert 'scenarios' in structure
        assert 'total_episodes' in structure
        assert 'total_files' in structure
        
        # Check scenario information
        assert len(structure['scenarios']) == len(scenarios)
        
        for scenario, expected_episodes in zip(scenarios, episodes_per_scenario):
            assert scenario in structure['scenarios']
            scenario_info = structure['scenarios'][scenario]
            
            assert 'episode_count' in scenario_info
            assert 'file_count' in scenario_info
            assert 'transitions_files' in scenario_info
            assert 'metadata_files' in scenario_info
            
            assert scenario_info['episode_count'] == expected_episodes
            assert len(scenario_info['transitions_files']) >= 1
            assert len(scenario_info['metadata_files']) >= 1
        
        # Check totals
        expected_total_episodes = sum(episodes_per_scenario)
        assert structure['total_episodes'] == expected_total_episodes
        assert structure['total_files'] > 0
    
    def test_global_index_generation(self):
        """Test global dataset index creation."""
        # Create test data
        scenarios = ['free_flow', 'dense_commuting']
        all_paths = []
        
        for scenario in scenarios:
            paths_list = self._create_test_scenario_data(scenario, num_episodes=2)
            all_paths.extend(paths_list)
        
        # Create index
        index_path = self.storage_manager.create_dataset_index(all_paths)
        
        # Verify index file
        assert index_path.exists()
        assert index_path.name == 'index.json'
        
        # Load and verify index content
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        # Check index structure
        required_keys = ['dataset_name', 'created', 'base_path', 'scenarios', 'total_files']
        for key in required_keys:
            assert key in index_data
        
        # Check dataset metadata
        assert index_data['dataset_name'] == 'highway_multimodal_datacollection'
        assert index_data['base_path'] == str(self.storage_manager.base_path)
        assert index_data['total_files'] == len(all_paths)
        
        # Check scenario listings
        for scenario in scenarios:
            assert scenario in index_data['scenarios']
            scenario_files = index_data['scenarios'][scenario]
            assert len(scenario_files) == 2  # 2 episodes per scenario
            
            for file_info in scenario_files:
                assert 'transitions_file' in file_info
                assert 'metadata_file' in file_info
                assert 'created' in file_info
                
                # Check that file paths are relative
                assert not file_info['transitions_file'].startswith('/')
                assert not file_info['metadata_file'].startswith('/')
    
    def test_scenario_file_listing(self):
        """Test listing files for specific scenarios."""
        scenario_name = 'test_scenario'
        self._create_test_scenario_data(scenario_name, num_episodes=3)
        
        # Get scenario files
        files = self.storage_manager.get_scenario_files(scenario_name)
        
        # Check file count (should have transitions and metadata files)
        assert len(files) >= 6  # At least 3 transitions + 3 metadata files
        
        # Check file types
        transitions_files = [f for f in files if 'transitions' in f.name]
        metadata_files = [f for f in files if 'meta' in f.name]
        
        assert len(transitions_files) == 3
        assert len(metadata_files) == 3
        
        # Check that all files exist
        for file_path in files:
            assert file_path.exists()
    
    def test_dataset_integrity_validation(self):
        """Test dataset integrity validation."""
        # Create valid dataset
        self._create_test_scenario_data('valid_scenario', num_episodes=2)
        
        # Create scenario with missing metadata file
        invalid_scenario_dir = self.storage_manager.base_path / 'invalid_scenario'
        invalid_scenario_dir.mkdir()
        
        # Create only transitions file (missing metadata)
        transitions_file = invalid_scenario_dir / 'test_transitions.parquet'
        pd.DataFrame({'episode_id': ['ep_001'], 'step': [0]}).to_parquet(transitions_file)
        
        # Validate dataset
        validation = self.storage_manager.validate_dataset_integrity()
        
        # Check validation results
        assert 'valid' in validation
        assert 'errors' in validation
        assert 'warnings' in validation
        assert 'statistics' in validation
        
        # Should detect the mismatch
        assert not validation['valid']
        assert len(validation['errors']) > 0
        
        # Check that error mentions the mismatch
        error_text = ' '.join(validation['errors'])
        assert 'invalid_scenario' in error_text
        assert 'Mismatch' in error_text
    
    def test_empty_directory_cleanup(self):
        """Test cleanup of empty directories."""
        # Create empty scenario directory
        empty_dir = self.storage_manager.base_path / 'empty_scenario'
        empty_dir.mkdir()
        
        # Create directory with empty files
        empty_files_dir = self.storage_manager.base_path / 'empty_files_scenario'
        empty_files_dir.mkdir()
        (empty_files_dir / 'empty.txt').touch()
        
        # Create valid scenario
        self._create_test_scenario_data('valid_scenario', num_episodes=1)
        
        # Run cleanup
        removed_dirs = self.storage_manager.cleanup_empty_directories()
        
        # Check results
        assert 'empty_scenario' in removed_dirs
        assert 'empty_files_scenario' in removed_dirs
        assert 'valid_scenario' not in removed_dirs
        
        # Check that directories were actually removed
        assert not empty_dir.exists()
        assert not empty_files_dir.exists()
        assert (self.storage_manager.base_path / 'valid_scenario').exists()
    
    def test_index_file_relative_paths(self):
        """Test that index file contains relative paths."""
        # Create test data
        paths_list = self._create_test_scenario_data('test_scenario', num_episodes=1)
        
        # Create index
        index_path = self.storage_manager.create_dataset_index(paths_list)
        
        # Load index
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        # Check that all file paths are relative
        for scenario_files in index_data['scenarios'].values():
            for file_info in scenario_files:
                transitions_path = file_info['transitions_file']
                metadata_path = file_info['metadata_file']
                
                # Should not start with / (absolute path)
                assert not transitions_path.startswith('/')
                assert not metadata_path.startswith('/')
                
                # Should be valid relative paths
                full_transitions_path = self.storage_manager.base_path / transitions_path
                full_metadata_path = self.storage_manager.base_path / metadata_path
                
                assert full_transitions_path.exists()
                assert full_metadata_path.exists()
    
    def test_multiple_batches_same_scenario(self):
        """Test handling multiple batches for the same scenario."""
        scenario_name = 'multi_batch_scenario'
        
        # Create multiple batches
        all_paths = []
        for batch in range(3):
            paths_list = self._create_test_scenario_data(f'{scenario_name}_batch_{batch}', num_episodes=1)
            all_paths.extend(paths_list)
        
        # Also create multiple batches for the same scenario name
        for batch in range(2):
            data = [{'episode_id': f'ep_same_{batch}', 'step': 0, 'agent_id': 0}]
            metadata = [{
                'episode_id': f'ep_same_{batch}',
                'scenario': scenario_name,
                'config': {},
                'modalities': ['kinematics'],
                'n_agents': 1,
                'total_steps': 1,
                'seed': 42
            }]
            paths = self.storage_manager.write_episode_batch(data, metadata, scenario_name)
            all_paths.append(paths)
        
        # Create index
        index_path = self.storage_manager.create_dataset_index(all_paths)
        
        # Load and verify
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        # Check that scenario with multiple batches is handled correctly
        assert scenario_name in index_data['scenarios']
        assert len(index_data['scenarios'][scenario_name]) == 2  # 2 batches for same scenario
        
        # Check total file count
        assert index_data['total_files'] == len(all_paths)