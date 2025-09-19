"""
HighwayEnv Multi-Modal Data Collection System

A comprehensive framework for capturing synchronized observations across multiple 
modalities in multi-agent highway driving scenarios.
"""

__version__ = "0.1.0"
__author__ = "Highway Data Collection Team"

from .scenarios import ScenarioRegistry
from .environments.factory import MultiAgentEnvFactory
from .collection.collector import SynchronizedCollector
from .collection.orchestrator import CollectionOrchestrator, run_full_collection
from .features.engine import FeatureDerivationEngine
from .storage.manager import DatasetStorageManager

__all__ = [
    "ScenarioRegistry",
    "MultiAgentEnvFactory", 
    "SynchronizedCollector",
    "CollectionOrchestrator",
    "run_full_collection",
    "FeatureDerivationEngine",
    "DatasetStorageManager"
]