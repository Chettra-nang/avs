"""
Demonstration of performance optimization and monitoring features.

This example shows how to use the performance monitoring and optimization
components to improve data collection efficiency and monitor system resources.
"""

import time
import logging
from pathlib import Path
import numpy as np

from highway_datacollection.performance import (
    MemoryProfiler, PerformanceProfiler, StorageThroughputMonitor,
    BatchingOptimizer, PerformanceConfig
)
from highway_datacollection.collection.collector import SynchronizedCollector
from highway_datacollection.storage.manager import DatasetStorageManager
from highway_datacollection.scenarios.registry import ScenarioRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_memory_profiling():
    """Demonstrate memory profiling capabilities."""
    print("\n=== Memory Profiling Demo ===")
    
    # Create memory profiler
    profiler = MemoryProfiler(max_snapshots=100)
    
    # Start continuous monitoring
    profiler.start_monitoring(interval=0.5)
    
    print("Starting memory monitoring...")
    
    # Simulate memory-intensive operations
    data_arrays = []
    for i in range(10):
        # Create large arrays to show memory growth
        large_array = np.random.random((1000, 1000))
        data_arrays.append(large_array)
        
        # Take snapshot and show current memory
        snapshot = profiler.take_snapshot()
        print(f"Step {i+1}: Memory usage = {snapshot.rss_mb:.1f} MB, "
              f"Objects = {snapshot.gc_objects:,}")
        
        time.sleep(0.1)
    
    # Get memory statistics
    stats = profiler.get_memory_stats()
    print(f"\nMemory Statistics:")
    print(f"  Current RSS: {stats['current']['rss_mb']:.1f} MB")
    print(f"  Peak RSS: {stats['statistics']['rss_max']:.1f} MB")
    print(f"  Growth rate: {stats['statistics']['growth_rate_mb_per_min']:.1f} MB/min")
    
    # Get optimization recommendations
    recommendations = profiler.get_optimization_recommendations()
    print(f"\nOptimization Recommendations:")
    for rec in recommendations:
        print(f"  - {rec}")
    
    # Trigger garbage collection
    print(f"\nTriggering garbage collection...")
    gc_result = profiler.trigger_gc()
    print(f"  Objects collected: {gc_result['objects_collected']}")
    print(f"  Memory freed: {gc_result['memory_freed_mb']:.1f} MB")
    
    # Stop monitoring
    profiler.stop_monitoring()
    
    # Clean up
    del data_arrays


def demonstrate_performance_profiling():
    """Demonstrate performance profiling of operations."""
    print("\n=== Performance Profiling Demo ===")
    
    # Create performance profiler with memory monitoring
    memory_profiler = MemoryProfiler()
    profiler = PerformanceProfiler(memory_profiler)
    
    # Profile different types of operations
    operations = [
        ("data_processing", lambda: [x**2 for x in range(10000)]),
        ("array_operations", lambda: np.random.random((1000, 1000)).sum()),
        ("string_operations", lambda: "".join([str(i) for i in range(1000)]))
    ]
    
    for op_name, op_func in operations:
        print(f"\nProfiling operation: {op_name}")
        
        # Profile multiple runs
        for run in range(3):
            with profiler.profile_operation(op_name) as op_profiler:
                result = op_func()
                op_profiler.set_items_processed(1000)  # Assume 1000 items processed
        
        # Get statistics for this operation
        stats = profiler.get_operation_stats(op_name)
        print(f"  Runs: {stats['count']}")
        print(f"  Mean duration: {stats['mean_duration']:.4f}s")
        print(f"  Min/Max duration: {stats['min_duration']:.4f}s / {stats['max_duration']:.4f}s")
    
    # Get overall performance summary
    summary = profiler.get_performance_summary()
    print(f"\nOverall Performance Summary:")
    print(f"  Total operations: {summary['recent_operations']}")
    print(f"  Average duration: {summary['average_duration']:.4f}s")
    print(f"  Total time: {summary['total_duration']:.2f}s")


def demonstrate_throughput_monitoring():
    """Demonstrate storage throughput monitoring."""
    print("\n=== Storage Throughput Monitoring Demo ===")
    
    # Create throughput monitor
    config = PerformanceConfig(throughput_monitoring=True, throughput_warning_threshold=2.0)
    monitor = StorageThroughputMonitor(config)
    
    # Simulate different storage operations with varying performance
    operations = [
        ("parquet", 1024*1024*5, 1.0),   # 5 MB in 1s = 5 MB/s
        ("parquet", 1024*1024*2, 2.0),   # 2 MB in 2s = 1 MB/s (slow)
        ("csv", 1024*1024*3, 0.5),       # 3 MB in 0.5s = 6 MB/s
        ("jsonl", 1024*512, 0.1),        # 0.5 MB in 0.1s = 5 MB/s
    ]
    
    print("Recording storage operations...")
    for op_type, bytes_written, duration in operations:
        monitor.record_write_operation(bytes_written, duration, op_type)
        throughput = (bytes_written / (1024*1024)) / duration
        print(f"  {op_type}: {bytes_written/(1024*1024):.1f} MB in {duration:.1f}s = {throughput:.1f} MB/s")
    
    # Get throughput statistics
    stats = monitor.get_throughput_stats()
    print(f"\nThroughput Statistics:")
    print(f"  Measurements: {stats['measurement_count']}")
    print(f"  Mean throughput: {stats['throughput_stats']['mean_mbps']:.2f} MB/s")
    print(f"  Max throughput: {stats['throughput_stats']['max_mbps']:.2f} MB/s")
    print(f"  Min throughput: {stats['throughput_stats']['min_mbps']:.2f} MB/s")
    
    # Get operation-specific stats
    parquet_stats = monitor.get_throughput_stats(operation_type="parquet")
    if "error" not in parquet_stats:
        print(f"  Parquet mean: {parquet_stats['throughput_stats']['mean_mbps']:.2f} MB/s")
    
    # Get optimization recommendations
    recommendations = monitor.get_optimization_recommendations()
    print(f"\nThroughput Recommendations:")
    for rec in recommendations:
        print(f"  - {rec}")


def demonstrate_batch_optimization():
    """Demonstrate batch size optimization."""
    print("\n=== Batch Size Optimization Demo ===")
    
    # Create batch optimizer with adaptive batching enabled
    config = PerformanceConfig(
        adaptive_batching=True,
        min_batch_size=5,
        max_batch_size=50,
        memory_based_batching=True
    )
    memory_profiler = MemoryProfiler()
    optimizer = BatchingOptimizer(config, memory_profiler)
    
    print(f"Initial batch size: {optimizer.current_batch_size}")
    
    # Simulate batch processing with different sizes and performance
    batch_scenarios = [
        # (batch_size, processing_time, memory_usage, success_rate)
        (5, 0.5, 50.0, 1.0),    # Small batch: fast, low memory, good success
        (10, 0.8, 100.0, 1.0),  # Medium batch: good balance
        (20, 1.2, 200.0, 0.95), # Large batch: slower, more memory, some failures
        (30, 2.0, 400.0, 0.8),  # Very large: slow, high memory, more failures
    ]
    
    print("\nSimulating batch processing scenarios...")
    for batch_size, proc_time, memory_mb, success_rate in batch_scenarios:
        # Simulate multiple runs for each batch size
        for run in range(3):
            items_processed = int(batch_size * success_rate)
            errors = [] if success_rate == 1.0 else [f"Error in batch {run}"]
            
            optimizer.record_batch_metrics(
                batch_size=batch_size,
                processing_time=proc_time + np.random.normal(0, 0.1),  # Add some variance
                memory_usage_mb=memory_mb + np.random.normal(0, 10),
                items_processed=items_processed,
                errors=errors
            )
        
        print(f"  Batch size {batch_size}: {proc_time:.1f}s, {memory_mb:.0f}MB, {success_rate:.1%} success")
    
    # Get optimal batch size
    optimal_size = optimizer.get_optimal_batch_size()
    print(f"\nOptimal batch size: {optimal_size}")
    
    # Get batch size adjustment suggestion
    suggestion = optimizer.suggest_batch_size_adjustment()
    print(f"Adjustment suggestion: {suggestion['adjustment']}")
    print(f"Message: {suggestion['message']}")
    
    # Get batch statistics
    stats = optimizer.get_batch_statistics()
    print(f"\nBatch Statistics:")
    print(f"  Total batches processed: {stats['total_batches']}")
    print(f"  Mean throughput: {stats['throughput_stats']['mean_items_per_sec']:.1f} items/s")
    print(f"  Mean memory usage: {stats['memory_stats']['mean_mb']:.1f} MB")
    print(f"  Mean success rate: {stats['success_stats']['mean_rate']:.1%}")


def demonstrate_integrated_performance_monitoring():
    """Demonstrate integrated performance monitoring in data collection."""
    print("\n=== Integrated Performance Monitoring Demo ===")
    
    # Create performance configuration
    config = PerformanceConfig(
        max_memory_gb=4.0,
        enable_memory_profiling=True,
        enable_profiling=True,
        adaptive_batching=True,
        default_batch_size=5,
        throughput_monitoring=True
    )
    
    # Create collector with performance monitoring
    collector = SynchronizedCollector(
        n_agents=2,
        performance_config=config
    )
    
    print("Created collector with performance monitoring enabled")
    
    # Get initial performance statistics
    initial_stats = collector.get_performance_statistics()
    if 'memory_stats' in initial_stats and 'current' in initial_stats['memory_stats']:
        print(f"Initial memory usage: {initial_stats['memory_stats']['current']['rss_mb']:.1f} MB")
    else:
        print("Memory profiling data not yet available")
    
    # Simulate some data collection (mock since we don't have real environments)
    print("\nSimulating data collection operations...")
    
    # Simulate batch processing metrics
    for i in range(5):
        batch_size = config.default_batch_size
        processing_time = 1.0 + np.random.normal(0, 0.2)
        memory_usage = 100 + i * 20  # Increasing memory usage
        items_processed = batch_size
        
        collector.batching_optimizer.record_batch_metrics(
            batch_size=batch_size,
            processing_time=processing_time,
            memory_usage_mb=memory_usage,
            items_processed=items_processed
        )
        
        print(f"  Batch {i+1}: {processing_time:.2f}s, {memory_usage:.0f}MB")
    
    # Get updated performance statistics
    final_stats = collector.get_performance_statistics()
    print(f"\nFinal Performance Statistics:")
    
    if "batch_stats" in final_stats and "error" not in final_stats["batch_stats"]:
        batch_stats = final_stats["batch_stats"]
        print(f"  Batches processed: {batch_stats['total_batches']}")
        print(f"  Mean processing time: {batch_stats['timing_stats']['mean_seconds']:.2f}s")
        print(f"  Mean memory usage: {batch_stats['memory_stats']['mean_mb']:.1f}MB")
    
    # Perform optimization
    print(f"\nPerforming performance optimization...")
    optimization_results = collector.optimize_performance()
    
    print(f"Optimization Results:")
    for recommendation in optimization_results["recommendations"]:
        print(f"  - {recommendation}")
    
    # Clean up
    collector.reset_performance_metrics()


def main():
    """Run all performance optimization demonstrations."""
    print("Highway Data Collection - Performance Optimization Demo")
    print("=" * 60)
    
    try:
        demonstrate_memory_profiling()
        demonstrate_performance_profiling()
        demonstrate_throughput_monitoring()
        demonstrate_batch_optimization()
        demonstrate_integrated_performance_monitoring()
        
        print("\n" + "=" * 60)
        print("Performance optimization demo completed successfully!")
        print("\nKey takeaways:")
        print("- Memory profiling helps identify memory leaks and optimization opportunities")
        print("- Performance profiling tracks operation timing and throughput")
        print("- Storage throughput monitoring optimizes I/O performance")
        print("- Batch optimization balances throughput and memory usage")
        print("- Integrated monitoring provides comprehensive system insights")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()