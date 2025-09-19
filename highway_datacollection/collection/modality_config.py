"""
Modality configuration and selection system.

This module provides flexible configuration for enabling/disabling specific
observation modalities during data collection, supporting focused data collection
per scenario and custom observation processors through a plugin architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModalityConfig:
    """
    Configuration for a specific observation modality.
    
    Defines whether a modality is enabled, its processing parameters,
    and any custom processors to apply.
    """
    enabled: bool = True
    processor: Optional['ObservationProcessor'] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    storage_enabled: bool = True
    feature_extraction_enabled: bool = True


@dataclass
class ScenarioModalityConfig:
    """
    Modality configuration for a specific scenario.
    
    Allows per-scenario customization of which modalities to collect
    and how to process them.
    """
    scenario_name: str
    modalities: Dict[str, ModalityConfig] = field(default_factory=dict)
    default_enabled: bool = True
    
    def is_modality_enabled(self, modality_name: str) -> bool:
        """Check if a modality is enabled for this scenario."""
        if modality_name in self.modalities:
            return self.modalities[modality_name].enabled
        return self.default_enabled
    
    def get_modality_config(self, modality_name: str) -> ModalityConfig:
        """Get configuration for a specific modality."""
        if modality_name in self.modalities:
            return self.modalities[modality_name]
        return ModalityConfig(enabled=self.default_enabled)


class ObservationProcessor(ABC):
    """
    Abstract base class for custom observation processors.
    
    Allows pluggable processing of observations before storage or feature extraction.
    """
    
    @abstractmethod
    def process_observation(self, observation: Any, metadata: Dict[str, Any]) -> Any:
        """
        Process an observation.
        
        Args:
            observation: Raw observation data
            metadata: Additional metadata about the observation
            
        Returns:
            Processed observation data
        """
        pass
    
    @abstractmethod
    def get_output_schema(self) -> Dict[str, type]:
        """
        Get the schema of the processed output.
        
        Returns:
            Dictionary mapping field names to their types
        """
        pass
    
    def validate_input(self, observation: Any) -> bool:
        """
        Validate input observation format.
        
        Args:
            observation: Observation to validate
            
        Returns:
            True if observation is valid for this processor
        """
        return True


class KinematicsProcessor(ObservationProcessor):
    """Default processor for Kinematics observations."""
    
    def process_observation(self, observation: Any, metadata: Dict[str, Any]) -> Any:
        """Process kinematics observation (pass-through by default)."""
        return observation
    
    def get_output_schema(self) -> Dict[str, type]:
        """Get output schema for kinematics data."""
        return {
            'kinematics_raw': list,
            'presence': float,
            'x': float,
            'y': float,
            'vx': float,
            'vy': float,
            'cos_h': float,
            'sin_h': float
        }


class OccupancyGridProcessor(ObservationProcessor):
    """Default processor for OccupancyGrid observations."""
    
    def __init__(self, normalize: bool = True, flatten: bool = False):
        """
        Initialize occupancy grid processor.
        
        Args:
            normalize: Whether to normalize grid values to [0, 1]
            flatten: Whether to flatten the grid to 1D
        """
        self.normalize = normalize
        self.flatten = flatten
    
    def process_observation(self, observation: Any, metadata: Dict[str, Any]) -> Any:
        """Process occupancy grid observation."""
        obs_array = np.array(observation)
        
        if self.normalize:
            # Normalize to [0, 1] range
            obs_array = obs_array.astype(np.float32)
            if obs_array.max() > obs_array.min():
                obs_array = (obs_array - obs_array.min()) / (obs_array.max() - obs_array.min())
        
        if self.flatten:
            obs_array = obs_array.flatten()
        
        return obs_array
    
    def get_output_schema(self) -> Dict[str, type]:
        """Get output schema for occupancy grid data."""
        return {
            'occupancy_blob': bytes,
            'occupancy_shape': list,
            'occupancy_dtype': str
        }


class GrayscaleProcessor(ObservationProcessor):
    """Default processor for Grayscale observations."""
    
    def __init__(self, resize_shape: Optional[tuple] = None, normalize: bool = True):
        """
        Initialize grayscale processor.
        
        Args:
            resize_shape: Target shape for resizing (height, width)
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.resize_shape = resize_shape
        self.normalize = normalize
    
    def process_observation(self, observation: Any, metadata: Dict[str, Any]) -> Any:
        """Process grayscale observation."""
        obs_array = np.array(observation)
        
        if self.resize_shape and obs_array.shape[:2] != self.resize_shape:
            # Simple resize using numpy (for more advanced resizing, use cv2 or PIL)
            try:
                from scipy.ndimage import zoom
                zoom_factors = (
                    self.resize_shape[0] / obs_array.shape[0],
                    self.resize_shape[1] / obs_array.shape[1],
                    1.0 if len(obs_array.shape) > 2 else None
                )
                zoom_factors = [f for f in zoom_factors if f is not None]
                obs_array = zoom(obs_array, zoom_factors)
            except ImportError:
                logger.warning("scipy not available for image resizing, skipping resize")
        
        if self.normalize:
            obs_array = obs_array.astype(np.float32) / 255.0
        
        return obs_array
    
    def get_output_schema(self) -> Dict[str, type]:
        """Get output schema for grayscale data."""
        return {
            'grayscale_blob': bytes,
            'grayscale_shape': list,
            'grayscale_dtype': str
        }


class ModalityConfigManager:
    """
    Manager for modality configurations across scenarios.
    
    Provides centralized configuration management for observation modalities,
    supporting per-scenario customization and plugin architecture for processors.
    """
    
    def __init__(self):
        """Initialize modality configuration manager."""
        self._scenario_configs: Dict[str, ScenarioModalityConfig] = {}
        self._global_config: Dict[str, ModalityConfig] = {}
        self._processor_registry: Dict[str, ObservationProcessor] = {}
        
        # Register default processors
        self._register_default_processors()
        
        logger.info("Initialized ModalityConfigManager")
    
    def _register_default_processors(self) -> None:
        """Register default observation processors."""
        self._processor_registry['Kinematics'] = KinematicsProcessor()
        self._processor_registry['OccupancyGrid'] = OccupancyGridProcessor()
        self._processor_registry['GrayscaleObservation'] = GrayscaleProcessor()
    
    def set_global_modality_config(self, modality_name: str, config: ModalityConfig) -> None:
        """
        Set global configuration for a modality.
        
        Args:
            modality_name: Name of the modality
            config: Configuration for the modality
        """
        self._global_config[modality_name] = config
        logger.info(f"Set global config for modality '{modality_name}': enabled={config.enabled}")
    
    def set_scenario_modality_config(self, scenario_name: str, 
                                   config: ScenarioModalityConfig) -> None:
        """
        Set modality configuration for a specific scenario.
        
        Args:
            scenario_name: Name of the scenario
            config: Scenario-specific modality configuration
        """
        self._scenario_configs[scenario_name] = config
        logger.info(f"Set modality config for scenario '{scenario_name}'")
    
    def get_enabled_modalities(self, scenario_name: str) -> List[str]:
        """
        Get list of enabled modalities for a scenario.
        
        Args:
            scenario_name: Name of the scenario
            
        Returns:
            List of enabled modality names
        """
        enabled_modalities = []
        
        # Get all available modalities
        all_modalities = ['Kinematics', 'OccupancyGrid', 'GrayscaleObservation']
        
        for modality in all_modalities:
            if self.is_modality_enabled(scenario_name, modality):
                enabled_modalities.append(modality)
        
        return enabled_modalities
    
    def is_modality_enabled(self, scenario_name: str, modality_name: str) -> bool:
        """
        Check if a modality is enabled for a scenario.
        
        Args:
            scenario_name: Name of the scenario
            modality_name: Name of the modality
            
        Returns:
            True if modality is enabled
        """
        # Check scenario-specific configuration first
        if scenario_name in self._scenario_configs:
            return self._scenario_configs[scenario_name].is_modality_enabled(modality_name)
        
        # Check global configuration
        if modality_name in self._global_config:
            return self._global_config[modality_name].enabled
        
        # Default to enabled
        return True
    
    def get_modality_config(self, scenario_name: str, modality_name: str) -> ModalityConfig:
        """
        Get configuration for a specific modality in a scenario.
        
        Args:
            scenario_name: Name of the scenario
            modality_name: Name of the modality
            
        Returns:
            Modality configuration
        """
        # Check scenario-specific configuration first
        if scenario_name in self._scenario_configs:
            return self._scenario_configs[scenario_name].get_modality_config(modality_name)
        
        # Check global configuration
        if modality_name in self._global_config:
            return self._global_config[modality_name]
        
        # Return default configuration
        return ModalityConfig()
    
    def register_processor(self, modality_name: str, processor: ObservationProcessor) -> None:
        """
        Register a custom observation processor.
        
        Args:
            modality_name: Name of the modality
            processor: Custom processor for the modality
        """
        self._processor_registry[modality_name] = processor
        logger.info(f"Registered custom processor for modality '{modality_name}': "
                   f"{type(processor).__name__}")
    
    def get_processor(self, modality_name: str) -> Optional[ObservationProcessor]:
        """
        Get processor for a modality.
        
        Args:
            modality_name: Name of the modality
            
        Returns:
            Observation processor or None if not found
        """
        return self._processor_registry.get(modality_name)
    
    def create_scenario_config(self, scenario_name: str, 
                             enabled_modalities: Optional[List[str]] = None,
                             disabled_modalities: Optional[List[str]] = None,
                             custom_processors: Optional[Dict[str, ObservationProcessor]] = None) -> ScenarioModalityConfig:
        """
        Create a scenario-specific modality configuration.
        
        Args:
            scenario_name: Name of the scenario
            enabled_modalities: List of modalities to enable (None for all)
            disabled_modalities: List of modalities to disable
            custom_processors: Custom processors for specific modalities
            
        Returns:
            Scenario modality configuration
        """
        config = ScenarioModalityConfig(scenario_name=scenario_name)
        
        all_modalities = ['Kinematics', 'OccupancyGrid', 'GrayscaleObservation']
        
        for modality in all_modalities:
            # Start with default enabled state
            enabled = True
            
            # If enabled_modalities is specified, only those are enabled
            if enabled_modalities is not None:
                enabled = modality in enabled_modalities
            
            # If disabled_modalities is specified, disable those
            if disabled_modalities is not None and modality in disabled_modalities:
                enabled = False
            
            # Get custom processor if provided
            processor = None
            if custom_processors and modality in custom_processors:
                processor = custom_processors[modality]
            
            config.modalities[modality] = ModalityConfig(
                enabled=enabled,
                processor=processor
            )
        
        return config
    
    def disable_modality_globally(self, modality_name: str) -> None:
        """
        Disable a modality globally across all scenarios.
        
        Args:
            modality_name: Name of the modality to disable
        """
        self._global_config[modality_name] = ModalityConfig(enabled=False)
        logger.info(f"Disabled modality '{modality_name}' globally")
    
    def enable_modality_globally(self, modality_name: str) -> None:
        """
        Enable a modality globally across all scenarios.
        
        Args:
            modality_name: Name of the modality to enable
        """
        self._global_config[modality_name] = ModalityConfig(enabled=True)
        logger.info(f"Enabled modality '{modality_name}' globally")
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get summary of current modality configurations.
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            'global_configs': {
                name: {'enabled': config.enabled, 'has_processor': config.processor is not None}
                for name, config in self._global_config.items()
            },
            'scenario_configs': {
                name: {
                    'default_enabled': config.default_enabled,
                    'modality_overrides': {
                        mod_name: {'enabled': mod_config.enabled}
                        for mod_name, mod_config in config.modalities.items()
                    }
                }
                for name, config in self._scenario_configs.items()
            },
            'registered_processors': list(self._processor_registry.keys())
        }


# Convenience functions for common configurations

def create_kinematics_only_config(scenario_name: str) -> ScenarioModalityConfig:
    """Create configuration that only enables Kinematics modality."""
    manager = ModalityConfigManager()
    return manager.create_scenario_config(
        scenario_name=scenario_name,
        enabled_modalities=['Kinematics']
    )


def create_vision_only_config(scenario_name: str) -> ScenarioModalityConfig:
    """Create configuration that only enables vision-based modalities."""
    manager = ModalityConfigManager()
    return manager.create_scenario_config(
        scenario_name=scenario_name,
        enabled_modalities=['OccupancyGrid', 'GrayscaleObservation']
    )


def create_minimal_config(scenario_name: str) -> ScenarioModalityConfig:
    """Create minimal configuration with only essential modalities."""
    manager = ModalityConfigManager()
    return manager.create_scenario_config(
        scenario_name=scenario_name,
        enabled_modalities=['Kinematics'],
        custom_processors={
            'Kinematics': KinematicsProcessor()
        }
    )