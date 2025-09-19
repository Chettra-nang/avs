# Performance Optimization and Monitoring

This document describes the performance optimization and monitoring features implemented in the Highway Multi-Modal Data Collection system.

## Overview

The performance optimization system provides comprehensive monitoring and optimization capabilities for:

- **Memory Usage**: Track memory consumption, detect leaks, and optimize garbage collection
- **Storage Throughput**: Monitor I/O performance and optimize compression settings
- **Batch Processing**: Dynamically optimize batch sizes for memory-efficient processing
- **Operation Profiling**: Track timing and throughput of individual operations

## Components

### 1. Memory Profiler (`MemoryProfiler`)

Monitors system memory usage and provides optimization recommendations.

#### Features:
- Real-time memory monitoring with configurable intervals
- Memory usage snapshots with RSS, VMS, and garbage collection statistics
- Growth rate calculation and leak detection
- Manual garbage collection triggering
- Optimization recommendations based on usage patterns

#### Usage:
```python
from highway_datacollection.performance import MemoryProfiler

profiler = MemoryProfiler()

# Start continuous monitoring
profiler.start_monitoring(interval=1.0)

# Take manual snapshots
snapshot = profiler.take_snapshot()
print(f"Memory usage: {snapshot.rss_mb:.1f} MB")

# Get statistics and recommendations
stats = profiler.get_memory_stats()
recommendations = profiler.get_optimization_recommendations()

# Trigger garbage collection
gc_result = profiler.trigger_gc()
print(f"Memory freed: {gc_result['memory_freed_mb']:.1f} MB")

profiler.stop_monitoring()
```

### 2. Performance Profiler (`PerformanceProfiler`)

Profiles individual operations for timing and throughput analysis.

#### Features:
- Context manager for easy operation profiling
- Automatic memory usage tracking during operations
- Throughput calculation for batch operations
- Operation statistics aggregation
- Performance summary generation

#### Usage:
```python
from highway_datacollection.performance import PerformanceProfiler

profiler = PerformanceProfiler()

# Profile an operation
with profiler.profile_operation("data_processing") as op_profiler:
    # Your operation here
    process_data()
    op_profiler.set_items_processed(1000)

# Get operation statistics
stats = profiler.get_operation_stats("data_processing")
print(f"Mean duration: {stats['mean_duration']:.4f}s")

# Get overall summary
summary = profiler.get_performance_summary()
```

### 3. Storage Throughput Monitor (`StorageThroughputMonitor`)

Monitors storage I/O performance and provides optimization recommendations.

#### Features:
- Throughput measurement for different file operations (Parquet, CSV, JSONL)
- Time-windowed statistics
- Operation-type filtering
- Low throughput warnings
- Compression optimization recommendations

#### Usage:
```python
from highway_datacollection.performance import StorageThroughputMonitor, PerformanceConfig

config = PerformanceConfig()
monitor = StorageThroughputMonitor(config)

# Record write operations (automatically done by storage manager)
monitor.record_write_operation(bytes_written=1024*1024, duration=1.0, operation_type="parquet")

# Get statistics
stats = monitor.get_throughput_stats()
print(f"Mean throughput: {stats['throughput_stats']['mean_mbps']:.2f} MB/s")

# Get recommendations
recommendations = monitor.get_optimization_recommendations()
```

### 4. Batching Optimizer (`BatchingOptimizer`)

Optimizes batch sizes for memory-efficient processing.

#### Features:
- Adaptive batch size optimization based on performance metrics
- Memory-based batch size adjustment
- Performance score calculation
- Batch size recommendations
- Comprehensive batch statistics

#### Usage:
```python
from highway_datacollection.performance import BatchingOptimizer, PerformanceConfig

config = PerformanceConfig(adaptive_batching=True)
optimizer = BatchingOptimizer(config)

# Record batch metrics
optimizer.record_batch_metrics(
    batch_size=10,
    processing_time=1.0,
    memory_usage_mb=100.0,
    items_processed=10
)

# Get optimal batch size
optimal_size = optimizer.get_optimal_batch_size()
print(f"Optimal batch size: {optimal_size}")

# Get adjustment suggestions
suggestion = optimizer.suggest_batch_size_adjustment()
print(f"Suggestion: {suggestion['message']}")
```

## Integration with Data Collection

### Collector Integration

The `SynchronizedCollector` integrates all performance monitoring components:

```python
from highway_datacollection.performance import PerformanceConfig
from highway_datacollection.collection.collector import SynchronizedCollector

# Create performance configuration
config = PerformanceConfig(
    max_memory_gb=8.0,
    enable_memory_profiling=True,
    adaptive_batching=True,
    throughput_monitoring=True
)

# Create collector with performance monitoring
collector = SynchronizedCollector(
    n_agents=2,
    performance_config=config
)

# Collect data with automatic performance optimization
result = collector.collect_episode_batch(
    scenario_name="free_flow",
    episodes=100,
    seed=42,
    max_steps=1000
)

# Get performance statistics
stats = collector.get_performance_statistics()
print(f"Memory usage: {stats['memory_stats']['current']['rss_mb']:.1f} MB")
print(f"Optimal batch size: {stats['batch_suggestion']['suggested_batch_size']}")

# Perform optimization
optimization_results = collector.optimize_performance()
```

### Storage Manager Integration

The `DatasetStorageManager` includes throughput monitoring:

```python
from highway_datacollection.performance import PerformanceConfig
from highway_datacollection.storage.manager import DatasetStorageManager

config = PerformanceConfig(
    throughput_monitoring=True,
    enable_compression=True
)

manager = DatasetStorageManager(
    base_path=Path("data"),
    performance_config=config
)

# Write data with automatic throughput monitoring
storage_paths = manager.write_episode_batch(data, metadata, "scenario")

# Get storage statistics including throughput
stats = manager.get_storage_statistics()
print(f"Throughput: {stats['throughput_stats']['throughput_stats']['mean_mbps']:.2f} MB/s")
```

## Configuration

### Performance Configuration (`PerformanceConfig`)

All performance settings are controlled through the `PerformanceConfig` class:

```python
from highway_datacollection.performance import PerformanceConfig

config = PerformanceConfig(
    # Memory management
    max_memory_gb=8.0,
    memory_check_interval=10,
    gc_threshold_mb=1000.0,
    enable_memory_profiling=True,
    
    # Batch processing
    default_batch_size=10,
    adaptive_batching=True,
    min_batch_size=1,
    max_batch_size=100,
    memory_based_batching=True,
    
    # Storage optimization
    enable_compression=True,
    compression_level=6,
    throughput_monitoring=True,
    
    # Performance monitoring
    enable_profiling=True,
    profile_interval=50,
    log_performance_metrics=True,
    
    # Thresholds
    memory_warning_threshold=0.8,
    memory_critical_threshold=0.95,
    throughput_warning_threshold=1.0
)
```

### Global Configuration

Performance settings can also be configured globally in `highway_datacollection/config.py`:

```python
PERFORMANCE_CONFIG = {
    "max_memory_gb": 8.0,
    "enable_memory_profiling": True,
    "adaptive_batching": True,
    "throughput_monitoring": True,
    # ... other settings
}
```

## Performance Optimization Strategies

### Memory Optimization

1. **Monitor Memory Growth**: Track memory usage over time to detect leaks
2. **Trigger Garbage Collection**: Manually trigger GC when memory usage is high
3. **Batch Size Adjustment**: Reduce batch sizes when memory usage exceeds thresholds
4. **Object Count Monitoring**: Track Python object count to identify accumulation

### Storage Optimization

1. **Compression Settings**: Adjust compression level based on throughput requirements
2. **File Format Selection**: Use Parquet for better compression, CSV as fallback
3. **Buffer Size Optimization**: Tune I/O buffer sizes for better throughput
4. **Parallel Writing**: Consider parallel writing for large datasets

### Batch Processing Optimization

1. **Adaptive Batch Sizes**: Automatically adjust batch sizes based on performance
2. **Memory-Based Scaling**: Reduce batch sizes when memory usage is high
3. **Performance Scoring**: Use composite scores to balance throughput and resource usage
4. **Error Rate Monitoring**: Adjust batch sizes based on success rates

## Monitoring and Alerts

### Memory Alerts

- **Warning**: Memory usage > 80% of maximum
- **Critical**: Memory usage > 95% of maximum
- **Growth Rate**: Memory growing > 10 MB/min

### Storage Alerts

- **Low Throughput**: Storage throughput < 1 MB/s
- **High Variance**: Inconsistent storage performance
- **Operation-Specific**: Parquet throughput < 5 MB/s

### Batch Processing Alerts

- **High Memory Usage**: Batch memory usage > threshold
- **Low Success Rate**: Batch success rate < 90%
- **Performance Degradation**: Throughput decreasing over time

## Best Practices

### Memory Management

1. **Enable Continuous Monitoring**: Use memory profiler with appropriate intervals
2. **Set Reasonable Limits**: Configure memory limits based on system capacity
3. **Monitor Growth Rates**: Watch for unexpected memory growth patterns
4. **Regular Cleanup**: Trigger garbage collection periodically

### Storage Performance

1. **Monitor All Operations**: Track throughput for all file types
2. **Adjust Compression**: Balance compression ratio vs. speed
3. **Use Appropriate Formats**: Parquet for analytics, CSV for compatibility
4. **Monitor Disk Space**: Ensure adequate free space for optimal performance

### Batch Processing

1. **Enable Adaptive Batching**: Let the system optimize batch sizes automatically
2. **Set Appropriate Bounds**: Configure min/max batch sizes for your use case
3. **Monitor Success Rates**: Ensure batch processing remains reliable
4. **Consider Memory Constraints**: Enable memory-based batch size adjustment

## Troubleshooting

### High Memory Usage

1. Check memory profiler recommendations
2. Reduce batch sizes
3. Trigger garbage collection
4. Look for memory leaks in custom code

### Low Storage Throughput

1. Check disk I/O performance
2. Adjust compression settings
3. Consider using faster storage
4. Monitor for disk space issues

### Poor Batch Performance

1. Review batch size optimization
2. Check for memory constraints
3. Monitor error rates
4. Consider reducing complexity per batch

## Examples

See `examples/performance_optimization_demo.py` for comprehensive examples of all performance monitoring features.

## Testing

Performance optimization features are tested in `tests/test_performance_optimization.py`, covering:

- Memory profiling accuracy
- Performance measurement precision
- Throughput monitoring functionality
- Batch optimization algorithms
- Integration with collector and storage manager