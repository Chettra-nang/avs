"""
Type definitions for data collection.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class EpisodeData:
    """Data collected from a single episode."""
    episode_id: str
    scenario: str
    observations: List[Dict[str, Any]]
    actions: List[int]
    rewards: List[float]
    dones: List[bool]
    infos: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class CollectionResult:
    """Result of a data collection operation."""
    episodes: List[EpisodeData]
    total_episodes: int
    successful_episodes: int
    failed_episodes: int
    collection_time: float
    errors: List[str]