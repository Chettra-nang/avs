"""
Modality configuration and selection system.

This module provides flexible configuration for enabling/disabling specific
observation modalities during data collection, supporting focused data collection
per scenario and custom observation processors through a plugin architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, Callable, Union, Tuple
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
    """Processor for Kinematics observations with multi-agent support."""

    def __init__(self, n_agents: int = 1, extract_per_agent: bool = True):
        """
        Initialize kinematics processor.

        Args:
            n_agents: Number of agents (for multi-agent scenarios)
            extract_per_agent: Whether to extract features per agent
        """
        self.n_agents = n_agents
        self.extract_per_agent = extract_per_agent
        
        # Initialize language summarizer for generating natural language descriptions
        from ..features.summarizer import LanguageSummarizer
        self.summarizer = LanguageSummarizer()

    def process_observation(self, observation: Any, metadata: Dict[str, Any]) -> Any:
        """
        Process kinematics observation with proper multi-agent handling.

        In multi-agent mode, observation is a tuple of arrays (one per agent).
        Each agent's observation contains kinematic features for all vehicles.
        """
        if isinstance(observation, tuple) and len(observation) > 1:
            # Multi-agent observation: tuple of arrays
            return self._process_multi_agent_kinematics(observation, metadata)
        else:
            # Single-agent observation: single array
            return self._process_single_agent_kinematics(observation, metadata)

    def _process_multi_agent_kinematics(self, observation: Tuple[np.ndarray, ...],
                                       metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process multi-agent kinematics observations.

        Args:
            observation: Tuple of kinematic arrays (one per agent)
            metadata: Additional processing metadata

        Returns:
            Dictionary with processed kinematics for each agent
        """
        agent_kinematics = {}

        for agent_idx, agent_obs in enumerate(observation):
            try:
                # Convert to numpy array if needed
                agent_obs_array = np.array(agent_obs)
                agent_kinematics[f'agent_{agent_idx}'] = self._extract_agent_kinematics(agent_obs_array, agent_idx, metadata)
            except Exception as e:
                logger.warning(f"Failed to process agent {agent_idx} kinematics: {e}")
                agent_kinematics[f'agent_{agent_idx}'] = self._get_default_kinematics()

        # Store raw observation for debugging (convert to list to avoid serialization issues)
        try:
            agent_kinematics['kinematics_raw'] = [np.array(obs).flatten().tolist() for obs in observation]
        except Exception:
            agent_kinematics['kinematics_raw'] = []

        return agent_kinematics

    def _process_single_agent_kinematics(self, observation: np.ndarray,
                                        metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process single-agent kinematics observation.

        Args:
            observation: Kinematic array for single agent
            metadata: Additional processing metadata

        Returns:
            Dictionary with processed kinematics
        """
        try:
            # Convert to numpy array if needed
            kinematics_array = np.array(observation)
            kinematics = self._extract_agent_kinematics(kinematics_array, 0, metadata)
            kinematics['kinematics_raw'] = kinematics_array.flatten().tolist()
            return kinematics
        except Exception as e:
            logger.warning(f"Failed to process single agent kinematics: {e}")
            default_kinematics = self._get_default_kinematics()
            default_kinematics['kinematics_raw'] = []
            return default_kinematics

    def _extract_agent_kinematics(self, kinematics_array: np.ndarray, agent_index: int, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract kinematic features for a specific agent.

        Highway-env kinematics format: [presence, x, y, vx, vy, cos_h, sin_h] per vehicle
        Vehicles are ordered with controlled vehicles first.

        Args:
            kinematics_array: Raw kinematics array (can be 1D flattened or 2D)
            agent_index: Index of the agent to extract features for

        Returns:
            Dictionary with extracted kinematic features
        """
        try:
            # Handle both 1D and 2D input arrays
            if kinematics_array.ndim == 1:
                # 1D flattened array
                n_vehicles = len(kinematics_array) // 7
                kinematics_reshaped = kinematics_array.reshape(n_vehicles, 7)
            elif kinematics_array.ndim == 2 and kinematics_array.shape[1] == 7:
                # Already 2D with correct shape
                kinematics_reshaped = kinematics_array
                n_vehicles = kinematics_reshaped.shape[0]
            else:
                # Try to reshape if possible
                kinematics_flat = kinematics_array.flatten()
                n_vehicles = len(kinematics_flat) // 7
                if n_vehicles > 0:
                    kinematics_reshaped = kinematics_flat.reshape(n_vehicles, 7)
                else:
                    raise ValueError(f"Cannot reshape kinematics array of size {len(kinematics_flat)} into (n, 7) format")

            # Extract features for the specified agent (controlled vehicle)
            if agent_index < n_vehicles:
                agent_features = kinematics_reshaped[agent_index]
                presence, x, y, vx, vy, cos_h, sin_h = agent_features

                # Calculate derived features
                # Check if velocities are normalized (Highway-Env default) and de-normalize if needed
                vx_real, vy_real = self._denormalize_velocities(vx, vy, kinematics_reshaped)
                speed = np.sqrt(vx_real**2 + vy_real**2)
                heading = np.arctan2(sin_h, cos_h)

                # Calculate TTC (Time To Collision) with nearest vehicle ahead
                ttc = self._calculate_ttc(kinematics_reshaped, agent_index)

                # Calculate traffic density and other features
                traffic_density = self._calculate_traffic_density(kinematics_reshaped)
                min_ttc = self._calculate_min_ttc(kinematics_reshaped, agent_index)
                average_speed, speed_variance = self._calculate_speed_stats(kinematics_reshaped)
                lane_change_opportunities = self._calculate_lane_change_opportunities(kinematics_reshaped, agent_index)

                return {
                    'presence': float(presence),
                    'ego_x': float(x),
                    'ego_y': float(y),
                    'ego_vx': float(vx_real),
                    'ego_vy': float(vy_real),
                    'speed': float(speed),
                    'heading': float(heading),
                    'cos_h': float(cos_h),
                    'sin_h': float(sin_h),
                    'lane_position': int(agent_index % 4),  # Assume 4 lanes
                    'ttc': float(ttc) if np.isfinite(ttc) else float('inf'),
                    'min_ttc': float(min_ttc) if np.isfinite(min_ttc) else float('inf'),
                    'traffic_density': float(traffic_density),
                    'vehicle_count': int(np.sum(kinematics_reshaped[:, 0])),  # Count present vehicles
                    'average_speed': float(average_speed),
                    'speed_variance': float(speed_variance),
                    'lane_change_opportunities': int(lane_change_opportunities),
                    'summary_text': self._generate_summary_text(kinematics_reshaped, agent_index, metadata)
                }
            else:
                # Agent index out of bounds, return default values
                return self._get_default_kinematics()

        except Exception as e:
            logger.warning(f"Failed to extract kinematics for agent {agent_index}: {e}")
            return self._get_default_kinematics()

    def _generate_summary_text(self, kinematics_reshaped: np.ndarray, agent_index: int, metadata: Dict[str, Any] = None) -> str:
        """
        Generate natural language summary of the driving context.
        
        Args:
            kinematics_reshaped: Full kinematics array for all vehicles
            agent_index: Index of the agent to summarize for
            metadata: Additional metadata about the observation
            
        Returns:
            Natural language description of the driving situation
        """
        try:
            if agent_index >= len(kinematics_reshaped):
                return "Unable to process observation - agent index out of bounds"
            
            # Extract ego vehicle state
            ego_vehicle = kinematics_reshaped[agent_index].copy()  # Make a copy to avoid modifying original
            
            # De-normalize ego vehicle velocities before summarization
            ego_vx, ego_vy = ego_vehicle[3:5]  # vx, vy from kinematics
            ego_vx_real, ego_vy_real = self._denormalize_velocities(ego_vx, ego_vy, kinematics_reshaped)
            ego_vehicle[3:5] = ego_vx_real, ego_vy_real  # Update with de-normalized velocities
            
            # Extract other vehicles (exclude ego)
            other_vehicles = np.concatenate([
                kinematics_reshaped[:agent_index],
                kinematics_reshaped[agent_index + 1:]
            ]) if len(kinematics_reshaped) > 1 else np.array([])
            
            # Filter to only present vehicles and de-normalize their velocities
            if len(other_vehicles) > 0:
                other_vehicles = other_vehicles[other_vehicles[:, 0] > 0]
                # De-normalize velocities for all other vehicles
                for i in range(len(other_vehicles)):
                    vx, vy = other_vehicles[i, 3:5]
                    vx_real, vy_real = self._denormalize_velocities(vx, vy, kinematics_reshaped)
                    other_vehicles[i, 3:5] = vx_real, vy_real
            
            # Get scenario information from metadata
            scenario_context = {}
            if metadata:
                scenario_name = metadata.get('scenario', 'default')
                scenario_context['scenario'] = scenario_name
            
            # Generate summary using the language summarizer with de-normalized velocities
            summary = self.summarizer.summarize(ego_vehicle, other_vehicles, scenario_context)
            return summary
            
        except Exception as e:
            logger.warning(f"Failed to generate summary text: {e}")
            return "Features extracted by KinematicsProcessor"

    def _denormalize_velocities(self, vx: float, vy: float, kinematics_reshaped: np.ndarray) -> Tuple[float, float]:
        """
        De-normalize velocities if they appear to be normalized by Highway-Env.
        
        Highway-Env normalizes velocities by speed_limit when absolute=False.
        We detect normalization by checking if max speeds are unrealistically low (< 2.0 m/s).
        
        Args:
            vx, vy: Normalized velocities
            kinematics_reshaped: Full kinematics array to estimate speed_limit
            
        Returns:
            Tuple of (real_vx, real_vy) in m/s
        """
        # Calculate speeds for all vehicles to detect if normalized
        all_speeds = []
        for vehicle in kinematics_reshaped:
            if vehicle[0] > 0:  # Vehicle present
                v_x, v_y = vehicle[3:5]
                all_speeds.append(np.sqrt(v_x**2 + v_y**2))
        
        if not all_speeds:
            return vx, vy
            
        max_speed = max(all_speeds)
        
        # If max speed is unrealistically low (< 2 m/s), assume normalized
        # Highway speeds should be 15-25 m/s, normalized would be 0-1
        if max_speed < 2.0:
            # Estimate speed_limit from scenario (typical highway speeds)
            # This is a heuristic since we don't have access to the exact speed_limit
            estimated_speed_limit = 20.0  # Conservative estimate for highway scenarios
            vx_real = vx * estimated_speed_limit
            vy_real = vy * estimated_speed_limit
            logger.debug(f"De-normalizing velocities: vx {vx:.3f} -> {vx_real:.3f}, vy {vy:.3f} -> {vy_real:.3f} (max_speed={max_speed:.3f})")
            return vx_real, vy_real
        
        # Already absolute velocities
        logger.debug(f"Velocities already absolute: vx {vx:.3f}, vy {vy:.3f} (max_speed={max_speed:.3f})")
        return vx, vy

    def _get_default_kinematics(self) -> Dict[str, Any]:
        """Return default kinematics values when extraction fails."""
        return {
            'presence': 0.0,
            'ego_x': 0.0,
            'ego_y': 0.0,
            'ego_vx': 0.0,
            'ego_vy': 0.0,
            'speed': 0.0,
            'heading': 0.0,
            'cos_h': 1.0,
            'sin_h': 0.0,
            'lane_position': 0,
            'ttc': float('inf'),
            'min_ttc': float('inf'),
            'traffic_density': 0.0,
            'vehicle_count': 0,
            'average_speed': 0.0,
            'speed_variance': 0.0,
            'lane_change_opportunities': 0,
            'summary_text': 'Unable to process observation'
        }

    def _calculate_ttc(self, kinematics: np.ndarray, agent_index: int) -> float:
        """Calculate Time To Collision with nearest vehicle ahead."""
        try:
            agent_x, agent_y = kinematics[agent_index, 1:3]
            agent_vx, agent_vy = kinematics[agent_index, 3:5]

            min_ttc = float('inf')

            for i, vehicle in enumerate(kinematics):
                if i == agent_index:
                    continue

                presence, x, y, vx, vy = vehicle[:5]
                if presence <= 0:
                    continue

                # Check if vehicle is ahead (same lane, positive relative x)
                if abs(y - agent_y) < 1.0:  # Same lane (within 1 unit)
                    relative_x = x - agent_x
                    relative_vx = vx - agent_vx

                    if relative_x > 0 and relative_vx < 0:  # Vehicle ahead and approaching
                        # TTC = relative_distance / relative_speed
                        ttc = abs(relative_x) / abs(relative_vx)
                        min_ttc = min(min_ttc, ttc)

            return min_ttc if np.isfinite(min_ttc) else float('inf')

        except Exception:
            return float('inf')

    def _calculate_min_ttc(self, kinematics: np.ndarray, agent_index: int) -> float:
        """Calculate minimum TTC with any vehicle."""
        try:
            agent_x, agent_y = kinematics[agent_index, 1:3]
            agent_vx, agent_vy = kinematics[agent_index, 3:5]

            min_ttc = float('inf')

            for i, vehicle in enumerate(kinematics):
                if i == agent_index:
                    continue

                presence, x, y, vx, vy = vehicle[:5]
                if presence <= 0:
                    continue

                # Calculate relative motion
                relative_x = x - agent_x
                relative_y = y - agent_y
                relative_vx = vx - agent_vx
                relative_vy = vy - agent_vy

                # Simple TTC calculation (distance / relative speed)
                distance = np.sqrt(relative_x**2 + relative_y**2)
                relative_speed = np.sqrt(relative_vx**2 + relative_vy**2)

                if relative_speed > 0.1:  # Avoid division by very small numbers
                    ttc = distance / relative_speed
                    min_ttc = min(min_ttc, ttc)

            return min_ttc if np.isfinite(min_ttc) else float('inf')

        except Exception:
            return float('inf')

    def _calculate_traffic_density(self, kinematics: np.ndarray) -> float:
        """Calculate traffic density (vehicles per unit area)."""
        try:
            present_vehicles = kinematics[kinematics[:, 0] > 0]
            if len(present_vehicles) <= 1:
                return 0.0

            # Calculate spread in x and y directions
            x_spread = np.ptp(present_vehicles[:, 1])  # Peak-to-peak in x
            y_spread = np.ptp(present_vehicles[:, 2])  # Peak-to-peak in y

            area = max(x_spread * y_spread, 1.0)  # Avoid division by zero
            return len(present_vehicles) / area

        except Exception:
            return 0.0

    def _calculate_speed_stats(self, kinematics: np.ndarray) -> Tuple[float, float]:
        """Calculate average speed and speed variance."""
        try:
            present_vehicles = kinematics[kinematics[:, 0] > 0]
            if len(present_vehicles) == 0:
                return 0.0, 0.0

            speeds = np.sqrt(present_vehicles[:, 3]**2 + present_vehicles[:, 4]**2)
            return float(np.mean(speeds)), float(np.var(speeds))

        except Exception:
            return 0.0, 0.0

    def _calculate_lane_change_opportunities(self, kinematics: np.ndarray, agent_index: int) -> int:
        """Calculate number of lane change opportunities."""
        try:
            agent_y = kinematics[agent_index, 2]
            opportunities = 0

            # Check adjacent lanes (simplified - assume 4 lanes)
            for lane_offset in [-1, 1]:
                target_lane = agent_y + lane_offset * 3.5  # Assume 3.5m lane width

                # Check if lane is clear (no vehicles within 10m ahead/behind)
                clear = True
                for i, vehicle in enumerate(kinematics):
                    if i == agent_index:
                        continue

                    presence, x, y, vx, vy = vehicle[:5]
                    if presence <= 0:
                        continue

                    if abs(y - target_lane) < 1.0:  # Vehicle in target lane
                        agent_x = kinematics[agent_index, 1]
                        if abs(x - agent_x) < 10.0:  # Within 10m
                            clear = False
                            break

                if clear:
                    opportunities += 1

            return opportunities

        except Exception:
            return 0

    def _get_default_kinematics(self) -> Dict[str, Any]:
        """Get default kinematics values for error cases."""
        return {
            'presence': 0.0,
            'ego_x': 0.0,
            'ego_y': 0.0,
            'ego_vx': 0.0,
            'ego_vy': 0.0,
            'speed': 0.0,
            'heading': 0.0,
            'cos_h': 1.0,
            'sin_h': 0.0,
            'lane_position': 0,
            'ttc': float('inf'),
            'min_ttc': float('inf'),
            'traffic_density': 0.0,
            'vehicle_count': 0,
            'average_speed': 0.0,
            'speed_variance': 0.0,
            'lane_change_opportunities': 0
        }

    def get_output_schema(self) -> Dict[str, type]:
        """Get output schema for kinematics data."""
        return {
            'kinematics_raw': list,
            'presence': float,
            'ego_x': float,
            'ego_y': float,
            'ego_vx': float,
            'ego_vy': float,
            'speed': float,
            'heading': float,
            'cos_h': float,
            'sin_h': float,
            'lane_position': int,
            'ttc': float,
            'min_ttc': float,
            'traffic_density': float,
            'vehicle_count': int,
            'average_speed': float,
            'speed_variance': float,
            'lane_change_opportunities': int,
            'summary_text': str
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
        # Use multi-agent aware kinematics processor by default
        self._processor_registry['Kinematics'] = KinematicsProcessor(n_agents=4, extract_per_agent=True)
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
            config = self._scenario_configs[scenario_name].get_modality_config(modality_name)
            # If no processor is set but one is registered, use it
            if config.processor is None and modality_name in self._processor_registry:
                config.processor = self._processor_registry[modality_name]
            return config
        
        # Check global configuration
        if modality_name in self._global_config:
            config = self._global_config[modality_name]
            # If no processor is set but one is registered, use it
            if config.processor is None and modality_name in self._processor_registry:
                config.processor = self._processor_registry[modality_name]
            return config
        
        # Return default configuration with registered processor if available
        default_config = ModalityConfig()
        if modality_name in self._processor_registry:
            default_config.processor = self._processor_registry[modality_name]
        return default_config
    
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