"""
Global configuration constants for the highway data collection system.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
LOGS_ROOT = PROJECT_ROOT / "logs"

# Default data collection parameters
DEFAULT_COLLECTION_CONFIG = {
    "episodes_per_scenario": 100,
    "max_steps_per_episode": 1000,
    "n_agents": 2,
    "seed_base": 42,
    "batch_size": 10,
    "save_frequency": 50,
}

# Storage configuration
STORAGE_CONFIG = {
    "use_compression": True,
    "parquet_engine": "pyarrow",
    "csv_fallback": True,
    "binary_encoding": "bytes",
    "index_filename": "index.json",
    "metadata_format": "jsonl",
}

# Environment configuration
ENV_CONFIG = {
    "render_mode": None,
    "normalize_obs": True,
    "normalize_reward": True,
    "clip_actions": True,
    "frame_stack": 1,
}

# Feature derivation configuration
FEATURE_CONFIG = {
    "ttc_threshold": 10.0,  # seconds
    "gap_threshold": 50.0,  # meters
    "speed_threshold": 30.0,  # m/s
    "lane_width": 4.0,  # meters
    "summary_template": "default",
}

# Performance optimization configuration
PERFORMANCE_CONFIG = {
    "max_memory_gb": 8.0,
    "memory_check_interval": 10,
    "gc_threshold_mb": 1000.0,
    "enable_memory_profiling": True,
    "default_batch_size": 10,
    "adaptive_batching": True,
    "min_batch_size": 1,
    "max_batch_size": 100,
    "memory_based_batching": True,
    "enable_compression": True,
    "compression_level": 6,
    "storage_buffer_size": 8192,
    "throughput_monitoring": True,
    "enable_profiling": True,
    "profile_interval": 50,
    "collect_detailed_stats": False,
    "log_performance_metrics": True,
    "memory_warning_threshold": 0.8,
    "memory_critical_threshold": 0.95,
    "throughput_warning_threshold": 1.0,
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_handler": True,
    "console_handler": True,
}

# Create necessary directories
def ensure_directories():
    """Create necessary directories if they don't exist."""
    DATA_ROOT.mkdir(exist_ok=True)
    LOGS_ROOT.mkdir(exist_ok=True)
    (DATA_ROOT / "scenarios").mkdir(exist_ok=True)
    (LOGS_ROOT / "collection").mkdir(exist_ok=True)