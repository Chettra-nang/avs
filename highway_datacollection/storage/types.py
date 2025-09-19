"""
Type definitions for storage operations.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
from pathlib import Path
import numpy as np


@dataclass
class StoragePaths:
    """Paths for stored dataset files."""
    transitions_file: Path
    metadata_file: Path
    scenario_dir: Path


@dataclass
class ObservationRecord:
    """Single observation record for storage."""
    episode_id: str
    step: int
    agent_id: int
    action: int
    reward: float
    
    # Kinematics features
    kin_presence: float
    kin_x: float
    kin_y: float
    kin_vx: float
    kin_vy: float
    kin_cos_h: float
    kin_sin_h: float
    
    # Derived features
    ttc: float
    summary_text: str
    
    # Binary blob references
    occ_blob: bytes
    occ_shape: List[int]
    occ_dtype: str
    gray_blob: bytes
    gray_shape: List[int]
    gray_dtype: str


@dataclass
class EpisodeMetadata:
    """Metadata for a complete episode."""
    episode_id: str
    scenario: str
    config: Dict[str, Any]
    modalities: List[str]
    n_agents: int
    total_steps: int
    seed: int