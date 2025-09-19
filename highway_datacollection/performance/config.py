"""
Performance configuration settings.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization and monitoring."""
    
    # Memory management
    max_memory_gb: float = 8.0
    memory_check_interval: int = 10  # episodes
    gc_threshold_mb: float = 1000.0
    enable_memory_profiling: bool = True
    
    # Batch processing
    default_batch_size: int = 10
    adaptive_batching: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 100
    memory_based_batching: bool = True
    
    # Storage optimization
    enable_compression: bool = True
    compression_level: int = 6
    storage_buffer_size: int = 8192
    throughput_monitoring: bool = True
    
    # Performance monitoring
    enable_profiling: bool = True
    profile_interval: int = 50  # steps
    collect_detailed_stats: bool = False
    log_performance_metrics: bool = True
    
    # Optimization thresholds
    memory_warning_threshold: float = 0.8  # 80% of max memory
    memory_critical_threshold: float = 0.95  # 95% of max memory
    throughput_warning_threshold: float = 1.0  # MB/s
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'max_memory_gb': self.max_memory_gb,
            'memory_check_interval': self.memory_check_interval,
            'gc_threshold_mb': self.gc_threshold_mb,
            'enable_memory_profiling': self.enable_memory_profiling,
            'default_batch_size': self.default_batch_size,
            'adaptive_batching': self.adaptive_batching,
            'min_batch_size': self.min_batch_size,
            'max_batch_size': self.max_batch_size,
            'memory_based_batching': self.memory_based_batching,
            'enable_compression': self.enable_compression,
            'compression_level': self.compression_level,
            'storage_buffer_size': self.storage_buffer_size,
            'throughput_monitoring': self.throughput_monitoring,
            'enable_profiling': self.enable_profiling,
            'profile_interval': self.profile_interval,
            'collect_detailed_stats': self.collect_detailed_stats,
            'log_performance_metrics': self.log_performance_metrics,
            'memory_warning_threshold': self.memory_warning_threshold,
            'memory_critical_threshold': self.memory_critical_threshold,
            'throughput_warning_threshold': self.throughput_warning_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceConfig':
        """Create from dictionary."""
        return cls(**data)
    
    def update(self, **kwargs) -> 'PerformanceConfig':
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self