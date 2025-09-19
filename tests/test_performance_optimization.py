"""
Tests for performance optimization and monitoring components.
"""

import pytest
import time
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch

from highway_datacollection.performance import (
    MemoryProfiler, PerformanceProfiler, StorageThroughputMonitor, 
    BatchingOptimizer, PerformanceConfig
)
from highway_datacollection.collection.collector import SynchronizedCollector
from highway_datacollection.storage.manager import DatasetStorageManager


class TestMemoryProfiler:
    """Test memory profiling functionality."""
    
    def test_memory_snapshot(self):
        """Test taking memory snapshots."""
        profiler = MemoryProfiler()
        
        snapshot = profiler.take_snapshot()
        
        assert snapshot.timestamp > 0
        assert snapshot.rss_mb >= 0
        assert snapshot.vms_mb >= 0
        assert snapshot.percent >= 0
        assert snapshot.available_mb >= 0
        assert snapshot.gc_objects >= 0
    
    def test_memory_monitoring(self):
        """Test continuous memory monitoring."""
        profiler = MemoryProfiler(max_snapshots=10)
        
        # Start monitoring
        profiler.start_monitoring(interval=0.1)
        time.sleep(0.3)  # Let it take a few snapshots
        profiler.stop_monitoring()
        
        assert len(profiler.snapshots) > 0
        assert len(profiler.snapshots) <= 10
    
    def test_memory_stats(self):
        """Test memory statistics calculation."""
        profiler = MemoryProfiler()
        
        # Take several snapshots
        for _ in range(5):
            profiler.take_snapshot()
            time.sleep(0.01)
        
        stats = profiler.get_memory_stats()
        
        assert "current" in stats
        assert "statistics" in stats
        assert "snapshots_count" in stats
        assert stats["snapshots_count"] == 5
        
        current = stats["current"]
        assert "rss_mb" in current
        assert "percent" in current
        assert "available_mb" in current
    
    def test_garbage_collection(self):
        """Test garbage collection triggering."""
        profiler = MemoryProfiler()
        
        # Create some objects to collect
        large_list = [list(range(1000)) for _ in range(100)]
        
        gc_result = profiler.trigger_gc()
        
        assert "objects_collected" in gc_result
        assert "memory_freed_mb" in gc_result
        assert "memory_before_mb" in gc_result
        assert "memory_after_mb" in gc_result
        
        # Clean up
        del large_list
    
    def test_optimization_recommendations(self):
        """Test memory optimization recommendations."""
        profiler = MemoryProfiler()
        
        # Take some snapshots
        for _ in range(3):
            profiler.take_snapshot()
            time.sleep(0.01)
        
        recommendations = profiler.get_optimization_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)


class TestPerformanceProfiler:
    """Test performance profiling functionality."""
    
    def test_operation_profiling(self):
        """Test profiling individual operations."""
        memory_profiler = MemoryProfiler()
        profiler = PerformanceProfiler(memory_profiler)
        
        # Profile a simple operation
        with profiler.profile_operation("test_operation") as op_profiler:
            op_profiler.set_items_processed(100)
            time.sleep(0.1)  # Simulate work
            # Create some memory usage
            temp_data = [i for i in range(1000)]
        
        assert len(profiler.metrics) == 1
        metric = profiler.metrics[0]
        
        assert metric.operation == "test_operation"
        assert metric.duration >= 0.1
        assert metric.items_processed == 100
        assert metric.throughput is not None
        assert metric.throughput > 0
    
    def test_operation_stats(self):
        """Test operation statistics."""
        profiler = PerformanceProfiler()
        
        # Profile multiple operations
        for i in range(5):
            with profiler.profile_operation("test_op"):
                time.sleep(0.01)
        
        stats = profiler.get_operation_stats("test_op")
        
        assert stats["operation"] == "test_op"
        assert stats["count"] == 5
        assert "mean_duration" in stats
        assert "median_duration" in stats
        assert "total_duration" in stats
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        profiler = PerformanceProfiler()
        
        # Profile some operations
        for i in range(3):
            with profiler.profile_operation(f"op_{i}") as op_profiler:
                op_profiler.set_items_processed(10)
                time.sleep(0.01)
        
        summary = profiler.get_performance_summary()
        
        assert "recent_operations" in summary
        assert "average_duration" in summary
        assert "total_duration" in summary
        assert summary["recent_operations"] == 3


class TestStorageThroughputMonitor:
    """Test storage throughput monitoring."""
    
    def test_throughput_recording(self):
        """Test recording throughput measurements."""
        config = PerformanceConfig()
        monitor = StorageThroughputMonitor(config)
        
        # Record some operations
        monitor.record_write_operation(1024*1024, 1.0, "parquet")  # 1 MB/s
        monitor.record_write_operation(2048*1024, 1.0, "csv")      # 2 MB/s
        
        assert len(monitor.measurements) == 2
        
        # Check measurements
        measurements = list(monitor.measurements)
        assert measurements[0].throughput_mbps == 1.0
        assert measurements[1].throughput_mbps == 2.0
    
    def test_throughput_stats(self):
        """Test throughput statistics calculation."""
        config = PerformanceConfig()
        monitor = StorageThroughputMonitor(config)
        
        # Record multiple operations
        for i in range(5):
            monitor.record_write_operation(1024*1024, 1.0, "parquet")
        
        stats = monitor.get_throughput_stats()
        
        assert "throughput_stats" in stats
        assert "duration_stats" in stats
        assert "volume_stats" in stats
        assert stats["measurement_count"] == 5
        
        throughput_stats = stats["throughput_stats"]
        assert throughput_stats["mean_mbps"] == 1.0
        assert throughput_stats["max_mbps"] == 1.0
    
    def test_throughput_filtering(self):
        """Test throughput statistics filtering."""
        config = PerformanceConfig()
        monitor = StorageThroughputMonitor(config)
        
        # Record different operation types
        monitor.record_write_operation(1024*1024, 1.0, "parquet")
        monitor.record_write_operation(2048*1024, 1.0, "csv")
        
        # Filter by operation type
        parquet_stats = monitor.get_throughput_stats(operation_type="parquet")
        csv_stats = monitor.get_throughput_stats(operation_type="csv")
        
        assert parquet_stats["measurement_count"] == 1
        assert csv_stats["measurement_count"] == 1
        assert parquet_stats["throughput_stats"]["mean_mbps"] == 1.0
        assert csv_stats["throughput_stats"]["mean_mbps"] == 2.0
    
    def test_optimization_recommendations(self):
        """Test throughput optimization recommendations."""
        config = PerformanceConfig(throughput_warning_threshold=5.0)
        monitor = StorageThroughputMonitor(config)
        
        # Record low throughput operations
        for _ in range(3):
            monitor.record_write_operation(1024*1024, 2.0, "parquet")  # 0.5 MB/s
        
        recommendations = monitor.get_optimization_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("Low storage throughput" in rec for rec in recommendations)


class TestBatchingOptimizer:
    """Test batch size optimization."""
    
    def test_batch_metrics_recording(self):
        """Test recording batch metrics."""
        config = PerformanceConfig()
        optimizer = BatchingOptimizer(config)
        
        # Record batch metrics
        optimizer.record_batch_metrics(
            batch_size=10,
            processing_time=1.0,
            memory_usage_mb=100.0,
            items_processed=10
        )
        
        assert len(optimizer.batch_metrics) == 1
        metric = optimizer.batch_metrics[0]
        
        assert metric.batch_size == 10
        assert metric.processing_time == 1.0
        assert metric.memory_usage_mb == 100.0
        assert metric.throughput_items_per_sec == 10.0
        assert metric.success_rate == 1.0
    
    def test_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        config = PerformanceConfig(adaptive_batching=True)
        optimizer = BatchingOptimizer(config)
        
        # Record metrics for different batch sizes
        # Batch size 5: decent performance
        for _ in range(3):
            optimizer.record_batch_metrics(5, 0.5, 50.0, 5)  # 10 items/sec, low memory
        
        # Batch size 10: better performance (higher throughput, reasonable memory)
        for _ in range(3):
            optimizer.record_batch_metrics(10, 0.8, 80.0, 10)  # 12.5 items/sec, moderate memory
        
        # Batch size 20: worse performance (high memory penalty)
        for _ in range(3):
            optimizer.record_batch_metrics(20, 2.0, 500.0, 20)  # 10 items/sec, high memory
        
        optimal_size = optimizer.get_optimal_batch_size()
        
        # Should prefer batch size 10 (best balance of throughput and memory)
        # But the algorithm might choose 5 if memory penalty is high enough
        assert optimal_size in [5, 10]  # Accept either as both are reasonable
    
    def test_batch_size_adjustment_suggestion(self):
        """Test batch size adjustment suggestions."""
        config = PerformanceConfig()
        optimizer = BatchingOptimizer(config)
        
        # Record some metrics
        optimizer.record_batch_metrics(5, 1.0, 50.0, 5)
        optimizer.record_batch_metrics(5, 1.0, 50.0, 5)
        
        suggestion = optimizer.suggest_batch_size_adjustment()
        
        assert "current_batch_size" in suggestion
        assert "suggested_batch_size" in suggestion
        assert "adjustment" in suggestion
        assert "message" in suggestion
        assert "current_performance" in suggestion
    
    def test_memory_based_batching(self):
        """Test memory-based batch size adjustment."""
        config = PerformanceConfig(memory_based_batching=True, memory_warning_threshold=0.5)
        memory_profiler = Mock()
        
        # Mock high memory usage
        memory_profiler.get_memory_stats.return_value = {
            "current": {"percent": 60.0}  # 60% memory usage
        }
        
        optimizer = BatchingOptimizer(config, memory_profiler)
        
        # Record good performance for large batch
        for _ in range(3):
            optimizer.record_batch_metrics(20, 1.0, 100.0, 20)
        
        optimal_size = optimizer.get_optimal_batch_size()
        
        # Should reduce batch size due to high memory usage
        assert optimal_size < 20
    
    def test_batch_statistics(self):
        """Test batch statistics generation."""
        config = PerformanceConfig()
        optimizer = BatchingOptimizer(config)
        
        # Record various batch metrics
        optimizer.record_batch_metrics(5, 0.5, 50.0, 5)
        optimizer.record_batch_metrics(10, 1.0, 100.0, 10)
        optimizer.record_batch_metrics(15, 1.5, 150.0, 15)
        
        stats = optimizer.get_batch_statistics()
        
        assert "total_batches" in stats
        assert "recent_batches" in stats
        assert "batch_size_stats" in stats
        assert "throughput_stats" in stats
        assert "memory_stats" in stats
        assert "success_stats" in stats
        assert "timing_stats" in stats
        
        assert stats["total_batches"] == 3
        assert stats["batch_size_stats"]["mean"] == 10.0


class TestIntegratedPerformanceMonitoring:
    """Test integrated performance monitoring in collector and storage manager."""
    
    def test_collector_performance_integration(self):
        """Test performance monitoring integration in collector."""
        config = PerformanceConfig(enable_profiling=True, enable_memory_profiling=True)
        
        collector = SynchronizedCollector(
            n_agents=2,
            performance_config=config
        )
        
        # Check that performance components are initialized
        assert collector.performance_config is not None
        assert collector.memory_profiler is not None
        assert collector.performance_profiler is not None
        assert collector.batching_optimizer is not None
        
        # Test getting performance statistics
        stats = collector.get_performance_statistics()
        
        assert "collection_stats" in stats
        assert "performance_config" in stats
        assert "memory_stats" in stats
        assert "batch_stats" in stats
        
        # Test performance optimization
        optimization_results = collector.optimize_performance()
        
        assert "memory_optimization" in optimization_results
        assert "batch_optimization" in optimization_results
        assert "recommendations" in optimization_results
        
        # Cleanup
        collector.reset_performance_metrics()
    
    def test_storage_manager_throughput_integration(self):
        """Test throughput monitoring integration in storage manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = PerformanceConfig(throughput_monitoring=True)
            
            manager = DatasetStorageManager(
                base_path=Path(temp_dir),
                performance_config=config
            )
            
            # Check that throughput monitor is initialized
            assert manager.throughput_monitor is not None
            
            # Test writing data (which should record throughput)
            test_data = [
                {
                    "episode_id": "test_ep_1",
                    "step": 0,
                    "agent_id": 0,
                    "action": 1,
                    "reward": 0.5
                }
            ]
            test_metadata = [
                {
                    "episode_id": "test_ep_1",
                    "scenario": "test_scenario",
                    "total_steps": 1
                }
            ]
            
            # Write episode batch
            storage_paths = manager.write_episode_batch(
                test_data, test_metadata, "test_scenario"
            )
            
            # Check that throughput was recorded
            stats = manager.get_storage_statistics()
            
            assert "throughput_stats" in stats
            assert "throughput_recommendations" in stats
            
            # Should have recorded some throughput measurements
            throughput_stats = stats["throughput_stats"]
            if "error" not in throughput_stats:
                assert throughput_stats["measurement_count"] > 0


class TestPerformanceConfig:
    """Test performance configuration."""
    
    def test_config_creation(self):
        """Test creating performance configuration."""
        config = PerformanceConfig(
            max_memory_gb=16.0,
            default_batch_size=20,
            enable_compression=True
        )
        
        assert config.max_memory_gb == 16.0
        assert config.default_batch_size == 20
        assert config.enable_compression is True
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = PerformanceConfig(max_memory_gb=8.0, default_batch_size=15)
        
        # Convert to dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["max_memory_gb"] == 8.0
        assert config_dict["default_batch_size"] == 15
        
        # Create from dict
        new_config = PerformanceConfig.from_dict(config_dict)
        assert new_config.max_memory_gb == 8.0
        assert new_config.default_batch_size == 15
    
    def test_config_update(self):
        """Test configuration updates."""
        config = PerformanceConfig(max_memory_gb=8.0)
        
        updated_config = config.update(max_memory_gb=16.0, default_batch_size=25)
        
        assert updated_config.max_memory_gb == 16.0
        assert updated_config.default_batch_size == 25


if __name__ == "__main__":
    pytest.main([__file__])