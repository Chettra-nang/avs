"""
Unit tests for modality configuration system.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from highway_datacollection.collection.modality_config import (
    ModalityConfig,
    ScenarioModalityConfig,
    ModalityConfigManager,
    ObservationProcessor,
    KinematicsProcessor,
    OccupancyGridProcessor,
    GrayscaleProcessor,
    create_kinematics_only_config,
    create_vision_only_config,
    create_minimal_config
)


class TestModalityConfig:
    """Test cases for ModalityConfig."""
    
    def test_default_initialization(self):
        """Test default ModalityConfig initialization."""
        config = ModalityConfig()
        assert config.enabled == True
        assert config.processor is None
        assert config.parameters == {}
        assert config.storage_enabled == True
        assert config.feature_extraction_enabled == True
    
    def test_custom_initialization(self):
        """Test ModalityConfig with custom values."""
        processor = Mock(spec=ObservationProcessor)
        config = ModalityConfig(
            enabled=False,
            processor=processor,
            parameters={'param1': 'value1'},
            storage_enabled=False,
            feature_extraction_enabled=False
        )
        
        assert config.enabled == False
        assert config.processor == processor
        assert config.parameters == {'param1': 'value1'}
        assert config.storage_enabled == False
        assert config.feature_extraction_enabled == False


class TestScenarioModalityConfig:
    """Test cases for ScenarioModalityConfig."""
    
    def test_initialization(self):
        """Test ScenarioModalityConfig initialization."""
        config = ScenarioModalityConfig(scenario_name="test_scenario")
        assert config.scenario_name == "test_scenario"
        assert config.modalities == {}
        assert config.default_enabled == True
    
    def test_is_modality_enabled_default(self):
        """Test modality enabled check with default behavior."""
        config = ScenarioModalityConfig(scenario_name="test", default_enabled=True)
        assert config.is_modality_enabled("Kinematics") == True
        
        config.default_enabled = False
        assert config.is_modality_enabled("Kinematics") == False
    
    def test_is_modality_enabled_specific(self):
        """Test modality enabled check with specific configuration."""
        config = ScenarioModalityConfig(scenario_name="test")
        config.modalities["Kinematics"] = ModalityConfig(enabled=False)
        
        assert config.is_modality_enabled("Kinematics") == False
        assert config.is_modality_enabled("OccupancyGrid") == True  # Uses default
    
    def test_get_modality_config(self):
        """Test getting modality configuration."""
        config = ScenarioModalityConfig(scenario_name="test")
        processor = Mock(spec=ObservationProcessor)
        config.modalities["Kinematics"] = ModalityConfig(enabled=False, processor=processor)
        
        # Get specific config
        kin_config = config.get_modality_config("Kinematics")
        assert kin_config.enabled == False
        assert kin_config.processor == processor
        
        # Get default config
        occ_config = config.get_modality_config("OccupancyGrid")
        assert occ_config.enabled == True
        assert occ_config.processor is None


class TestObservationProcessors:
    """Test cases for observation processors."""
    
    def test_kinematics_processor(self):
        """Test KinematicsProcessor."""
        processor = KinematicsProcessor()
        
        # Test process_observation (pass-through)
        obs = [[1, 2, 3, 4, 5]]
        result = processor.process_observation(obs, {})
        assert result == obs
        
        # Test get_output_schema
        schema = processor.get_output_schema()
        assert 'kinematics_raw' in schema
        assert 'x' in schema
        assert 'y' in schema
        
        # Test validate_input
        assert processor.validate_input(obs) == True
    
    def test_occupancy_grid_processor_default(self):
        """Test OccupancyGridProcessor with default settings."""
        processor = OccupancyGridProcessor()
        
        # Test with simple grid
        obs = np.array([[0, 1], [1, 0]])
        result = processor.process_observation(obs, {})
        
        # Should normalize to [0, 1]
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        
        # Test get_output_schema
        schema = processor.get_output_schema()
        assert 'occupancy_blob' in schema
        assert 'occupancy_shape' in schema
        assert 'occupancy_dtype' in schema
    
    def test_occupancy_grid_processor_flatten(self):
        """Test OccupancyGridProcessor with flattening."""
        processor = OccupancyGridProcessor(flatten=True)
        
        obs = np.array([[0, 1], [1, 0]])
        result = processor.process_observation(obs, {})
        
        assert result.ndim == 1
        assert len(result) == 4  # 2x2 flattened
    
    def test_grayscale_processor_default(self):
        """Test GrayscaleProcessor with default settings."""
        processor = GrayscaleProcessor()
        
        # Test with simple image
        obs = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]])
        result = processor.process_observation(obs, {})
        
        # Should normalize to [0, 1]
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        
        # Test get_output_schema
        schema = processor.get_output_schema()
        assert 'grayscale_blob' in schema
        assert 'grayscale_shape' in schema
        assert 'grayscale_dtype' in schema
    
    def test_grayscale_processor_resize(self):
        """Test GrayscaleProcessor with resizing."""
        processor = GrayscaleProcessor(resize_shape=(32, 32))
        
        obs = np.zeros((64, 64, 3))
        result = processor.process_observation(obs, {})
        
        # Should resize and normalize
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (32, 32, 3)  # Resized shape


class TestModalityConfigManager:
    """Test cases for ModalityConfigManager."""
    
    def test_initialization(self):
        """Test ModalityConfigManager initialization."""
        manager = ModalityConfigManager()
        
        # Should have default processors registered
        assert manager.get_processor('Kinematics') is not None
        assert manager.get_processor('OccupancyGrid') is not None
        assert manager.get_processor('GrayscaleObservation') is not None
    
    def test_set_global_modality_config(self):
        """Test setting global modality configuration."""
        manager = ModalityConfigManager()
        config = ModalityConfig(enabled=False)
        
        manager.set_global_modality_config('Kinematics', config)
        
        assert manager._global_config['Kinematics'] == config
        assert manager.is_modality_enabled('test_scenario', 'Kinematics') == False
    
    def test_set_scenario_modality_config(self):
        """Test setting scenario-specific modality configuration."""
        manager = ModalityConfigManager()
        scenario_config = ScenarioModalityConfig(scenario_name="test")
        scenario_config.modalities['Kinematics'] = ModalityConfig(enabled=False)
        
        manager.set_scenario_modality_config('test', scenario_config)
        
        assert manager.is_modality_enabled('test', 'Kinematics') == False
        assert manager.is_modality_enabled('other', 'Kinematics') == True  # Uses default
    
    def test_get_enabled_modalities(self):
        """Test getting enabled modalities for a scenario."""
        manager = ModalityConfigManager()
        
        # All enabled by default
        enabled = manager.get_enabled_modalities('test')
        assert 'Kinematics' in enabled
        assert 'OccupancyGrid' in enabled
        assert 'GrayscaleObservation' in enabled
        
        # Disable one globally
        manager.disable_modality_globally('OccupancyGrid')
        enabled = manager.get_enabled_modalities('test')
        assert 'Kinematics' in enabled
        assert 'OccupancyGrid' not in enabled
        assert 'GrayscaleObservation' in enabled
    
    def test_register_processor(self):
        """Test registering custom processor."""
        manager = ModalityConfigManager()
        custom_processor = Mock(spec=ObservationProcessor)
        
        manager.register_processor('CustomModality', custom_processor)
        
        assert manager.get_processor('CustomModality') == custom_processor
    
    def test_create_scenario_config(self):
        """Test creating scenario configuration."""
        manager = ModalityConfigManager()
        
        config = manager.create_scenario_config(
            scenario_name='test',
            enabled_modalities=['Kinematics'],
            disabled_modalities=['OccupancyGrid']
        )
        
        assert config.scenario_name == 'test'
        assert config.is_modality_enabled('Kinematics') == True
        assert config.is_modality_enabled('OccupancyGrid') == False
        # When enabled_modalities is specified, only those are enabled
        assert config.is_modality_enabled('GrayscaleObservation') == False
    
    def test_disable_enable_modality_globally(self):
        """Test global modality enable/disable."""
        manager = ModalityConfigManager()
        
        # Disable
        manager.disable_modality_globally('Kinematics')
        assert manager.is_modality_enabled('any_scenario', 'Kinematics') == False
        
        # Enable
        manager.enable_modality_globally('Kinematics')
        assert manager.is_modality_enabled('any_scenario', 'Kinematics') == True
    
    def test_get_configuration_summary(self):
        """Test getting configuration summary."""
        manager = ModalityConfigManager()
        manager.disable_modality_globally('OccupancyGrid')
        
        scenario_config = ScenarioModalityConfig(scenario_name="test")
        scenario_config.modalities['Kinematics'] = ModalityConfig(enabled=False)
        manager.set_scenario_modality_config('test', scenario_config)
        
        summary = manager.get_configuration_summary()
        
        assert 'global_configs' in summary
        assert 'scenario_configs' in summary
        assert 'registered_processors' in summary
        
        assert 'OccupancyGrid' in summary['global_configs']
        assert summary['global_configs']['OccupancyGrid']['enabled'] == False
        
        assert 'test' in summary['scenario_configs']


class TestConvenienceFunctions:
    """Test cases for convenience configuration functions."""
    
    def test_create_kinematics_only_config(self):
        """Test creating kinematics-only configuration."""
        config = create_kinematics_only_config('test_scenario')
        
        assert config.scenario_name == 'test_scenario'
        assert config.is_modality_enabled('Kinematics') == True
        assert config.is_modality_enabled('OccupancyGrid') == False
        assert config.is_modality_enabled('GrayscaleObservation') == False
    
    def test_create_vision_only_config(self):
        """Test creating vision-only configuration."""
        config = create_vision_only_config('test_scenario')
        
        assert config.scenario_name == 'test_scenario'
        assert config.is_modality_enabled('Kinematics') == False
        assert config.is_modality_enabled('OccupancyGrid') == True
        assert config.is_modality_enabled('GrayscaleObservation') == True
    
    def test_create_minimal_config(self):
        """Test creating minimal configuration."""
        config = create_minimal_config('test_scenario')
        
        assert config.scenario_name == 'test_scenario'
        assert config.is_modality_enabled('Kinematics') == True
        assert config.is_modality_enabled('OccupancyGrid') == False
        assert config.is_modality_enabled('GrayscaleObservation') == False
        
        # Should have custom processor
        kin_config = config.get_modality_config('Kinematics')
        assert kin_config.processor is not None
        assert isinstance(kin_config.processor, KinematicsProcessor)


class TestModalityConfigIntegration:
    """Integration tests for modality configuration system."""
    
    def test_scenario_override_global(self):
        """Test that scenario configuration overrides global configuration."""
        manager = ModalityConfigManager()
        
        # Set global config
        manager.disable_modality_globally('Kinematics')
        assert manager.is_modality_enabled('test', 'Kinematics') == False
        
        # Set scenario-specific config that overrides global
        scenario_config = ScenarioModalityConfig(scenario_name='test')
        scenario_config.modalities['Kinematics'] = ModalityConfig(enabled=True)
        manager.set_scenario_modality_config('test', scenario_config)
        
        # Scenario should override global
        assert manager.is_modality_enabled('test', 'Kinematics') == True
        # Other scenarios should still use global
        assert manager.is_modality_enabled('other', 'Kinematics') == False
    
    def test_processor_chain(self):
        """Test that processors can be chained or customized."""
        manager = ModalityConfigManager()
        
        # Create custom processor
        custom_processor = Mock(spec=ObservationProcessor)
        custom_processor.process_observation.return_value = "processed"
        custom_processor.get_output_schema.return_value = {'custom': str}
        
        # Register and use in scenario
        manager.register_processor('Kinematics', custom_processor)
        
        scenario_config = manager.create_scenario_config(
            scenario_name='test',
            custom_processors={'Kinematics': custom_processor}
        )
        
        # Verify processor is set
        kin_config = scenario_config.get_modality_config('Kinematics')
        assert kin_config.processor == custom_processor


if __name__ == "__main__":
    pytest.main([__file__])