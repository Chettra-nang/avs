"""
Unit tests for dataset storage manager.
"""

import pytest
import tempfile
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from highway_datacollection.storage.manager import DatasetStorageManager
from highway_datacollection.storage.types import StoragePaths, EpisodeMetadata


class TestDatasetStorageManager:
    """Test dataset storage manager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_manager = DatasetStorageManager(Path(self.temp_dir))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test storage manager initialization."""
        assert self.storage_manager.base_path.exists()
        assert self.storage_manager.base_path.is_dir()
    
    def test_generate_episode_id(self):
        """Test episode ID generation."""
        episode_id = self.storage_manager.generate_episode_id()
        
        assert episode_id.startswith('ep_')
        parts = episode_id.split('_')
        assert len(parts) >= 3  # ep_timestamp_parts_uuid
        assert parts[0] == 'ep'
        
        # Generate multiple IDs to ensure uniqueness
        ids = [self.storage_manager.generate_episode_id() for _ in range(10)]
        assert len(set(ids)) == 10  # All unique
    
    def test_encode_decode_binary_arrays(self):
        """Test binary array encoding and decoding."""
        # Create test arrays
        arrays = {
            'occ': np.random.rand(10, 10).astype(np.float32),
            'gray': np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        }
        
        # Encode
        encoded = self.storage_manager.encode_binary_arrays(arrays)
        
        # Verify encoded structure
        expected_keys = [
            'occ_blob', 'occ_shape', 'occ_dtype',
            'gray_blob', 'gray_shape', 'gray_dtype'
        ]
        for key in expected_keys:
            assert key in encoded
        
        # Decode
        decoded = self.storage_manager.decode_binary_arrays(encoded, ['occ', 'gray'])
        
        # Verify reconstruction
        np.testing.assert_array_almost_equal(arrays['occ'], decoded['occ'], decimal=6)
        np.testing.assert_array_equal(arrays['gray'], decoded['gray'])
    
    def test_write_episode_batch_basic(self):
        """Test basic episode batch writing."""
        # Create test data
        data = [
            {
                'episode_id': 'ep_001',
                'step': 0,
                'agent_id': 0,
                'action': 1,
                'reward': 0.5,
                'kin_x': 10.0,
                'kin_y': 5.0,
                'ttc': 2.5,
                'summary_text': 'Vehicle in right lane'
            },
            {
                'episode_id': 'ep_001',
                'step': 1,
                'agent_id': 0,
                'action': 2,
                'reward': 0.3,
                'kin_x': 11.0,
                'kin_y': 5.0,
                'ttc': 2.3,
                'summary_text': 'Vehicle accelerating'
            }
        ]
        
        metadata = [
            {
                'episode_id': 'ep_001',
                'scenario': 'free_flow',
                'config': {'vehicles_count': 10},
                'modalities': ['kinematics', 'occupancy'],
                'n_agents': 1,
                'total_steps': 2,
                'seed': 42
            }
        ]
        
        # Write data
        paths = self.storage_manager.write_episode_batch(data, metadata, 'free_flow')
        
        # Verify paths
        assert isinstance(paths, StoragePaths)
        assert paths.transitions_file.exists()
        assert paths.metadata_file.exists()
        assert paths.scenario_dir.exists()
        assert paths.scenario_dir.name == 'free_flow'
    
    def test_write_episode_batch_parquet_format(self):
        """Test that data is written in Parquet format when possible."""
        data = [
            {
                'episode_id': 'ep_001',
                'step': 0,
                'agent_id': 0,
                'action': 1,
                'reward': 0.5
            }
        ]
        
        metadata = [
            {
                'episode_id': 'ep_001',
                'scenario': 'test',
                'config': {},
                'modalities': ['kinematics'],
                'n_agents': 1,
                'total_steps': 1,
                'seed': 42
            }
        ]
        
        paths = self.storage_manager.write_episode_batch(data, metadata, 'test_scenario')
        
        # Check that Parquet file was created
        assert paths.transitions_file.suffix == '.parquet'
        
        # Verify data can be read back
        df = pd.read_parquet(paths.transitions_file)
        assert len(df) == 1
        assert df.iloc[0]['episode_id'] == 'ep_001'
        assert df.iloc[0]['action'] == 1
    
    def test_write_metadata_jsonl_format(self):
        """Test metadata writing in JSONL format."""
        metadata = [
            {
                'episode_id': 'ep_001',
                'scenario': 'free_flow',
                'config': {'vehicles_count': 10},
                'modalities': ['kinematics'],
                'n_agents': 1,
                'total_steps': 100,
                'seed': 42
            },
            {
                'episode_id': 'ep_002',
                'scenario': 'free_flow',
                'config': {'vehicles_count': 15},
                'modalities': ['kinematics', 'occupancy'],
                'n_agents': 2,
                'total_steps': 150,
                'seed': 43
            }
        ]
        
        paths = self.storage_manager.write_episode_batch([], metadata, 'test_scenario')
        
        # Read back metadata
        loaded_metadata = self.storage_manager.load_episode_metadata(paths.metadata_file)
        
        assert len(loaded_metadata) == 2
        assert loaded_metadata[0].episode_id == 'ep_001'
        assert loaded_metadata[0].scenario == 'free_flow'
        assert loaded_metadata[0].n_agents == 1
        assert loaded_metadata[1].episode_id == 'ep_002'
        assert loaded_metadata[1].n_agents == 2
    
    def test_create_dataset_index(self):
        """Test dataset index creation."""
        # Create some test files first
        scenario_paths = []
        
        for scenario in ['free_flow', 'dense_commuting']:
            paths = self.storage_manager.write_episode_batch(
                [{'episode_id': f'ep_{scenario}', 'step': 0, 'agent_id': 0}],
                [{'episode_id': f'ep_{scenario}', 'scenario': scenario, 'config': {}, 
                  'modalities': ['kinematics'], 'n_agents': 1, 'total_steps': 1, 'seed': 42}],
                scenario
            )
            scenario_paths.append(paths)
        
        # Create index
        index_path = self.storage_manager.create_dataset_index(scenario_paths)
        
        # Verify index file
        assert index_path.exists()
        assert index_path.name == 'index.json'
        
        # Load and verify index content
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        assert index_data['dataset_name'] == 'highway_multimodal_datacollection'
        assert 'created' in index_data
        assert 'scenarios' in index_data
        assert 'total_files' in index_data
        
        # Check scenarios
        assert 'free_flow' in index_data['scenarios']
        assert 'dense_commuting' in index_data['scenarios']
        assert len(index_data['scenarios']['free_flow']) == 1
        assert len(index_data['scenarios']['dense_commuting']) == 1
        
        # Check file references
        free_flow_files = index_data['scenarios']['free_flow'][0]
        assert 'transitions_file' in free_flow_files
        assert 'metadata_file' in free_flow_files
        assert 'created' in free_flow_files
    
    def test_get_scenario_files(self):
        """Test retrieving files for a specific scenario."""
        # Create test files
        self.storage_manager.write_episode_batch(
            [{'episode_id': 'ep_001', 'step': 0, 'agent_id': 0}],
            [{'episode_id': 'ep_001', 'scenario': 'test', 'config': {}, 
              'modalities': ['kinematics'], 'n_agents': 1, 'total_steps': 1, 'seed': 42}],
            'test_scenario'
        )
        
        # Get files
        files = self.storage_manager.get_scenario_files('test_scenario')
        
        assert len(files) >= 2  # At least transitions and metadata files
        file_names = [f.name for f in files]
        
        # Check for expected file patterns
        transitions_files = [f for f in file_names if 'transitions' in f]
        metadata_files = [f for f in file_names if 'meta' in f]
        
        assert len(transitions_files) >= 1
        assert len(metadata_files) >= 1
    
    def test_get_scenario_files_nonexistent(self):
        """Test retrieving files for non-existent scenario."""
        files = self.storage_manager.get_scenario_files('nonexistent_scenario')
        assert files == []
    
    def test_write_empty_data(self):
        """Test handling of empty data."""
        paths = self.storage_manager.write_episode_batch([], [], 'empty_scenario')
        
        # Files should still be created
        assert paths.transitions_file.parent.exists()
        assert paths.metadata_file.exists()
        
        # Metadata file should be empty
        with open(paths.metadata_file, 'r') as f:
            content = f.read().strip()
            assert content == ''
    
    def test_scenario_directory_creation(self):
        """Test that scenario directories are created properly."""
        scenario_name = 'test_scenario_with_underscores'
        
        paths = self.storage_manager.write_episode_batch(
            [{'episode_id': 'ep_001', 'step': 0, 'agent_id': 0}],
            [{'episode_id': 'ep_001', 'scenario': scenario_name, 'config': {}, 
              'modalities': ['kinematics'], 'n_agents': 1, 'total_steps': 1, 'seed': 42}],
            scenario_name
        )
        
        assert paths.scenario_dir.exists()
        assert paths.scenario_dir.name == scenario_name
        assert paths.scenario_dir.parent == self.storage_manager.base_path
    
    def test_filename_uniqueness(self):
        """Test that generated filenames are unique."""
        # Create multiple batches for the same scenario
        paths_list = []
        for i in range(3):
            paths = self.storage_manager.write_episode_batch(
                [{'episode_id': f'ep_{i}', 'step': 0, 'agent_id': 0}],
                [{'episode_id': f'ep_{i}', 'scenario': 'test', 'config': {}, 
                  'modalities': ['kinematics'], 'n_agents': 1, 'total_steps': 1, 'seed': 42}],
                'test_scenario'
            )
            paths_list.append(paths)
        
        # Check that all filenames are unique
        transition_files = [p.transitions_file.name for p in paths_list]
        metadata_files = [p.metadata_file.name for p in paths_list]
        
        assert len(set(transition_files)) == 3
        assert len(set(metadata_files)) == 3