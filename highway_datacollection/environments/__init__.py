"""
Environment factory module for creating multi-agent HighwayEnv instances.
"""

from .factory import MultiAgentEnvFactory
from .config import ENV_CONFIGS, OBSERVATION_CONFIGS

__all__ = ["MultiAgentEnvFactory", "ENV_CONFIGS", "OBSERVATION_CONFIGS"]