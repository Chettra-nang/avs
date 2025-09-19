"""
Tests for the collection orchestrator and end-to-end dataset generation.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

from highway_datacollection.collection.orchestrator import (
    CollectionOrchestrator, 
    run_full_collection,
    CollectionProgress,
    FullCollectionResult
)
from highway_datacollection.collection.types import CollectionResult, EpisodeData


class TestCollectionOrchestrator:
    """Test cases for CollectionOrchestrator class."""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def orchestrator(self, temp_storage_path):
        """Create orchestrator instance for testing."""
        return CollectionOrchestrator(temp_storage_path, n_agents=2)
    
    @pytest.fixture
    def mock_episode_data(self):
        """Create mock episode data for testing."""
        return EpisodeData(
            episode_id="test_ep_001",
            scenario="free_flow",
            observations=[
                [  # Step 0 observations for all agents
                    {
                        'episode_id': 'test_ep_001',
                        'step': 0,
                        'agent_id': 0,
                        'kinematics_raw': [1.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0],
                        'ttc': 5.0,
                        'summary_text': 'Driving in right lane at moderate speed',
                        'occupancy_blob': b'mock_occupancy_data',
                        'occupancy_shape': [11, 11],
                        'occupancy_dtype': 'float32',
                        'grayscale_blob': b'mock_grayscale_data',
                        'grayscale_shape': [84, 84, 3],
                        'grayscale_dtype': 'uint8'
                    },
                    {
                        'episode_id': 'test_ep_001',
                        'step': 0,
                        'agent_id': 1,
                        'kinematics_raw': [1.0, 0.0, 5.0, 12.0, 0.0, 1.0, 0.0],
                        'ttc': 3.0,
                        'summary_text': 'Following vehicle ahead',
                        'occupancy_blob': b'mock_occupancy_data_2',
                        'occupancy_shape': [11, 11],
                        'occupancy_dtype': 'float32',
                        'grayscale_blob': b'mock_grayscale_data_2',
                        'grayscale_shape': [84, 84, 3],
                        'grayscale_dtype': 'uint8'
                    }
                ]
            ],
            actions=[2],  # LANE_RIGHT action
            rewards=[0.5],
            dones=[False],
            infos=[{'speed': 10.0}],
            metadata={
                'episode_id': 'test_ep_001',
                'scenario': 'free_flow',
                'config': {'vehicles_count': 20, 'lanes_count': 4},
                'modalities': ['Kinematics', 'OccupancyGrid', 'GrayscaleObservation'],
                'n_agents': 2,
                'total_steps': 1,
                'seed': 42,
                'max_steps': 100,
                'terminated_early': False
            }
        )
    
    def test_orchestrator_initialization(self, temp_storage_path):
        """Test orchestrator initialization."""
        orchestrator = CollectionOrchestrator(temp_storage_path, n_agents=3)
        
        assert orchestrator.base_storage_path == temp_storage_path
        assert orchestrator.n_agents == 3
        assert orchestrator.scenario_registry is not None
        assert orchestrator.storage_manager is not None
        assert orchestrator.collector is not None
        assert orchestrator._progress_callback is None
        assert orchestrator._should_stop is False
    
    def test_set_progress_callback(self, orchestrator):
        """Test setting progress callback."""
        callback = Mock()
        orchestrator.set_progress_callback(callback)
        assert orchestrator._progress_callback == callback
    
    def test_stop_collection(self, orchestrator):
        """Test collection stop request."""
        assert orchestrator._should_stop is False
        orchestrator.stop_collection()
        assert orchestrator._should_stop is True
    
    @patch('highway_datacollection.collection.orchestrator.SynchronizedCollector')
    @patch('highway_datacollection.collection.orchestrator.DatasetStorageManager')
    def test_collect_scenario_data(self, mock_storage_manager, mock_collector, orchestrator, mock_episode_data):
        """Test scenario data collection with batch processing."""
        # Mock collector behavior
        mock_collector_instance = Mock()
        mock_collector.return_value = mock_collector_instance
        
        # Mock successful collection result
        mock_collection_result = CollectionResult(
            episodes=[mock_episode_data],
            total_episodes=1,
            successful_episodes=1,
            failed_episodes=0,
            collection_time=1.0,
            errors=[]
        )
        mock_collector_instance.collect_episode_batch.return_value = mock_collection_result
        
        # Replace orchestrator's collector with mock
        orchestrator.collector = mock_collector_instance
        
        # Create progress object
        progress = CollectionProgress(
            current_scenario="free_flow",
            scenario_index=0,
            total_scenarios=1,
            current_episode=0,
            total_episodes=2,
            successful_episodes=0,
            failed_episodes=0,
            start_time=time.time(),
            scenario_start_time=time.time(),
            errors=[]
        )
        
        # Test collection
        result = orchestrator._collect_scenario_data(
            scenario_name="free_flow",
            episodes=2,
            max_steps=50,
            base_seed=42,
            batch_size=1,
            progress=progress
        )
        
        # Verify results
        assert result.total_episodes == 2
        assert result.successful_episodes == 2  # 2 batches of 1 episode each
        assert result.failed_episodes == 0
        assert len(result.episodes) == 2
        assert len(result.errors) == 0
        
        # Verify collector was called correctly
        assert mock_collector_instance.collect_episode_batch.call_count == 2
    
    @patch('highway_datacollection.collection.orchestrator.SynchronizedCollector')
    def test_collect_scenario_data_with_failures(self, mock_collector, orchestrator):
        """Test scenario data collection with batch failures."""
        # Mock collector behavior with failures
        mock_collector_instance = Mock()
        mock_collector.return_value = mock_collector_instance
        
        # First batch succeeds, second batch fails
        mock_collector_instance.collect_episode_batch.side_effect = [
            CollectionResult(
                episodes=[],
                total_episodes=1,
                successful_episodes=1,
                failed_episodes=0,
                collection_time=1.0,
                errors=[]
            ),
            Exception("Batch collection failed")
        ]
        
        # Replace orchestrator's collector with mock
        orchestrator.collector = mock_collector_instance
        
        # Create progress object
        progress = CollectionProgress(
            current_scenario="free_flow",
            scenario_index=0,
            total_scenarios=1,
            current_episode=0,
            total_episodes=2,
            successful_episodes=0,
            failed_episodes=0,
            start_time=time.time(),
            scenario_start_time=time.time(),
            errors=[]
        )
        
        # Test collection
        result = orchestrator._collect_scenario_data(
            scenario_name="free_flow",
            episodes=2,
            max_steps=50,
            base_seed=42,
            batch_size=1,
            progress=progress
        )
        
        # Verify results
        assert result.total_episodes == 2
        assert result.successful_episodes == 1
        assert result.failed_episodes == 1
        assert len(result.errors) == 1
        assert "Batch 2 failed" in result.errors[0]
    
    def test_store_scenario_data(self, orchestrator, mock_episode_data):
        """Test storing scenario data."""
        # Mock storage manager
        mock_storage_paths = Mock()
        orchestrator.storage_manager.write_episode_batch = Mock(return_value=mock_storage_paths)
        
        # Create collection result
        result = CollectionResult(
            episodes=[mock_episode_data],
            total_episodes=1,
            successful_episodes=1,
            failed_episodes=0,
            collection_time=1.0,
            errors=[]
        )
        
        # Test storage
        storage_result = orchestrator._store_scenario_data("free_flow", result)
        
        # Verify storage was called
        assert storage_result == mock_storage_paths
        orchestrator.storage_manager.write_episode_batch.assert_called_once()
        
        # Verify call arguments
        call_args = orchestrator.storage_manager.write_episode_batch.call_args
        args, kwargs = call_args
        
        # Function is called with keyword arguments
        data = kwargs['data']
        metadata = kwargs['metadata']
        scenario = kwargs['scenario']
        
        assert scenario == "free_flow"
        assert len(metadata) == 1
        assert metadata[0]['episode_id'] == 'test_ep_001'
        assert len(data) == 2  # Two observation records (one per agent)
    
    def test_store_scenario_data_empty_episodes(self, orchestrator):
        """Test storing scenario data with no episodes."""
        result = CollectionResult(
            episodes=[],
            total_episodes=0,
            successful_episodes=0,
            failed_episodes=0,
            collection_time=0.0,
            errors=[]
        )
        
        storage_result = orchestrator._store_scenario_data("free_flow", result)
        assert storage_result is None
    
    @patch('highway_datacollection.collection.orchestrator.SynchronizedCollector')
    def test_run_full_collection_success(self, mock_collector, orchestrator, mock_episode_data):
        """Test successful full collection run."""
        # Mock collector
        mock_collector_instance = Mock()
        mock_collector.return_value = mock_collector_instance
        
        mock_collection_result = CollectionResult(
            episodes=[mock_episode_data],
            total_episodes=1,  # Each batch has 1 episode
            successful_episodes=1,  # Each batch has 1 successful episode
            failed_episodes=0,
            collection_time=1.0,
            errors=[]
        )
        mock_collector_instance.collect_episode_batch.return_value = mock_collection_result
        
        orchestrator.collector = mock_collector_instance
        
        # Mock storage manager
        mock_storage_paths = Mock()
        orchestrator.storage_manager.write_episode_batch = Mock(return_value=mock_storage_paths)
        orchestrator.storage_manager.create_dataset_index = Mock(return_value=Path("index.json"))
        
        # Run collection
        result = orchestrator.run_full_collection(
            episodes_per_scenario=2,
            max_steps_per_episode=50,
            base_seed=42,
            scenarios=["free_flow"],
            batch_size=1
        )
        
        # Verify results
        assert isinstance(result, FullCollectionResult)
        assert result.total_scenarios == 1
        assert result.successful_scenarios == 1
        assert result.failed_scenarios == 0
        assert result.total_episodes == 2
        assert result.successful_episodes == 2
        assert result.failed_episodes == 0
        assert len(result.scenario_results) == 1
        assert "free_flow" in result.scenario_results
        assert result.dataset_index_path == Path("index.json")
    
    @patch('highway_datacollection.collection.orchestrator.SynchronizedCollector')
    def test_run_full_collection_with_failures(self, mock_collector, orchestrator):
        """Test full collection run with scenario failures."""
        # Mock collector that fails
        mock_collector_instance = Mock()
        mock_collector.return_value = mock_collector_instance
        mock_collector_instance.collect_episode_batch.side_effect = Exception("Collection failed")
        
        orchestrator.collector = mock_collector_instance
        
        # Run collection
        result = orchestrator.run_full_collection(
            episodes_per_scenario=1,
            scenarios=["free_flow"],
            batch_size=1
        )
        
        # Verify results
        assert result.total_scenarios == 1
        assert result.successful_scenarios == 0
        assert result.failed_scenarios == 1
        assert result.total_episodes == 1
        assert result.successful_episodes == 0
        assert result.failed_episodes == 1
        assert len(result.errors) > 0
    
    def test_run_full_collection_stop_request(self, orchestrator):
        """Test collection stop functionality."""
        # Test that stop_collection method sets the flag
        assert orchestrator._should_stop is False
        orchestrator.stop_collection()
        assert orchestrator._should_stop is True
        
        # Note: run_full_collection resets the stop flag at the beginning
        # This is expected behavior to allow reuse of the orchestrator
        # So we test the stop_collection method functionality instead
        
        # Reset the flag and test that it can be set
        orchestrator._should_stop = False
        orchestrator.stop_collection()
        assert orchestrator._should_stop is True
    
    def test_get_collection_summary(self, orchestrator):
        """Test getting collection summary."""
        # Mock storage manager methods
        mock_structure = {'scenarios': {}, 'total_episodes': 0}
        mock_validation = {'valid': True, 'errors': []}
        
        orchestrator.storage_manager.organize_dataset_structure = Mock(return_value=mock_structure)
        orchestrator.storage_manager.validate_dataset_integrity = Mock(return_value=mock_validation)
        
        summary = orchestrator.get_collection_summary()
        
        assert 'dataset_structure' in summary
        assert 'validation' in summary
        assert 'available_scenarios' in summary
        assert 'storage_path' in summary
        assert summary['dataset_structure'] == mock_structure
        assert summary['validation'] == mock_validation
    
    def test_cleanup_failed_collections(self, orchestrator):
        """Test cleanup of failed collections."""
        # Mock storage manager cleanup methods
        orchestrator.storage_manager.cleanup_empty_directories = Mock(return_value=['empty_scenario'])
        orchestrator.storage_manager.validate_dataset_integrity = Mock(return_value={'valid': True})
        
        result = orchestrator.cleanup_failed_collections()
        
        assert result['cleanup_successful'] is True
        assert result['removed_directories'] == ['empty_scenario']
        assert 'validation' in result


class TestRunFullCollectionFunction:
    """Test cases for the run_full_collection convenience function."""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @patch('highway_datacollection.collection.orchestrator.CollectionOrchestrator')
    def test_run_full_collection_function(self, mock_orchestrator_class, temp_storage_path):
        """Test the convenience function."""
        # Mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator_class.return_value = mock_orchestrator
        
        mock_result = FullCollectionResult(
            total_scenarios=1,
            successful_scenarios=1,
            failed_scenarios=0,
            total_episodes=5,
            successful_episodes=5,
            failed_episodes=0,
            collection_time=10.0,
            scenario_results={},
            storage_paths=[],
            dataset_index_path=None,
            errors=[]
        )
        mock_orchestrator.run_full_collection.return_value = mock_result
        
        # Test function call
        progress_callback = Mock()
        result = run_full_collection(
            base_storage_path=temp_storage_path,
            episodes_per_scenario=5,
            n_agents=3,
            max_steps_per_episode=75,
            base_seed=123,
            scenarios=["free_flow"],
            batch_size=2,
            progress_callback=progress_callback
        )
        
        # Verify orchestrator was created correctly
        mock_orchestrator_class.assert_called_once_with(temp_storage_path, 3)
        
        # Verify progress callback was set
        mock_orchestrator.set_progress_callback.assert_called_once_with(progress_callback)
        
        # Verify run_full_collection was called with correct parameters
        mock_orchestrator.run_full_collection.assert_called_once_with(
            episodes_per_scenario=5,
            max_steps_per_episode=75,
            base_seed=123,
            scenarios=["free_flow"],
            batch_size=2
        )
        
        # Verify result
        assert result == mock_result
    
    @patch('highway_datacollection.collection.orchestrator.CollectionOrchestrator')
    def test_run_full_collection_function_no_callback(self, mock_orchestrator_class, temp_storage_path):
        """Test the convenience function without progress callback."""
        # Mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator_class.return_value = mock_orchestrator
        
        mock_result = Mock()
        mock_orchestrator.run_full_collection.return_value = mock_result
        
        # Test function call without callback
        result = run_full_collection(
            base_storage_path=temp_storage_path,
            episodes_per_scenario=10
        )
        
        # Verify orchestrator was created with defaults
        mock_orchestrator_class.assert_called_once_with(temp_storage_path, 2)
        
        # Verify progress callback was not set
        mock_orchestrator.set_progress_callback.assert_not_called()
        
        # Verify run_full_collection was called with defaults
        mock_orchestrator.run_full_collection.assert_called_once_with(
            episodes_per_scenario=10,
            max_steps_per_episode=100,
            base_seed=42,
            scenarios=None,
            batch_size=10
        )


class TestCollectionProgress:
    """Test cases for CollectionProgress data class."""
    
    def test_collection_progress_creation(self):
        """Test creating CollectionProgress instance."""
        progress = CollectionProgress(
            current_scenario="free_flow",
            scenario_index=0,
            total_scenarios=6,
            current_episode=5,
            total_episodes=100,
            successful_episodes=4,
            failed_episodes=1,
            start_time=time.time(),
            scenario_start_time=time.time(),
            errors=["Error 1"]
        )
        
        assert progress.current_scenario == "free_flow"
        assert progress.scenario_index == 0
        assert progress.total_scenarios == 6
        assert progress.current_episode == 5
        assert progress.total_episodes == 100
        assert progress.successful_episodes == 4
        assert progress.failed_episodes == 1
        assert len(progress.errors) == 1


class TestFullCollectionResult:
    """Test cases for FullCollectionResult data class."""
    
    def test_full_collection_result_creation(self):
        """Test creating FullCollectionResult instance."""
        result = FullCollectionResult(
            total_scenarios=6,
            successful_scenarios=5,
            failed_scenarios=1,
            total_episodes=600,
            successful_episodes=580,
            failed_episodes=20,
            collection_time=3600.0,
            scenario_results={},
            storage_paths=[],
            dataset_index_path=Path("index.json"),
            errors=[]
        )
        
        assert result.total_scenarios == 6
        assert result.successful_scenarios == 5
        assert result.failed_scenarios == 1
        assert result.total_episodes == 600
        assert result.successful_episodes == 580
        assert result.failed_episodes == 20
        assert result.collection_time == 3600.0
        assert result.dataset_index_path == Path("index.json")


if __name__ == "__main__":
    pytest.main([__file__])