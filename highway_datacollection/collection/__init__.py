"""
Data collection module for synchronized multi-modal observation capture.
"""

from .collector import SynchronizedCollector
from .types import CollectionResult, EpisodeData
from .action_samplers import (
    ActionSampler, 
    RandomActionSampler, 
    PolicyActionSampler, 
    HybridActionSampler
)
from .modality_config import (
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

__all__ = [
    "SynchronizedCollector", 
    "CollectionResult", 
    "EpisodeData",
    "ActionSampler",
    "RandomActionSampler",
    "PolicyActionSampler", 
    "HybridActionSampler",
    "ModalityConfig",
    "ScenarioModalityConfig", 
    "ModalityConfigManager",
    "ObservationProcessor",
    "KinematicsProcessor",
    "OccupancyGridProcessor",
    "GrayscaleProcessor",
    "create_kinematics_only_config",
    "create_vision_only_config",
    "create_minimal_config"
]