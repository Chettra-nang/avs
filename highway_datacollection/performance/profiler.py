"""
Memory and performance profiling utilities.
"""

import gc
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory percentage
    available_mb: float
    gc_objects: int
    gc_collections: List[int] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a specific operation."""
    operation: str
    duration: float
    memory_before: float
    memory_after: float
    memory_peak: float
    cpu_percent: float
    throughput: Optional[float] = None
    items_processed: Optional[int] = None


class MemoryProfiler:
    """
    Memory usage profiler with optimization recommendations.
    """
    
    def __init__(self, max_snapshots: int = 1000):
        """
        Initialize memory profiler.
        
        Args:
            max_snapshots: Maximum number of snapshots to keep in memory
        """
        self.max_snapshots = max_snapshots
        self.snapshots: deque = deque(maxlen=max_snapshots)
        self.process = psutil.Process()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_interval = 1.0  # seconds
        
    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            virtual_memory = psutil.virtual_memory()
            
            # Get garbage collection stats
            gc_stats = gc.get_stats()
            gc_collections = [stat['collections'] for stat in gc_stats]
            
            snapshot = MemorySnapshot(
                timestamp=time.time(),
                rss_mb=memory_info.rss / 1024 / 1024,
                vms_mb=memory_info.vms / 1024 / 1024,
                percent=memory_percent,
                available_mb=virtual_memory.available / 1024 / 1024,
                gc_objects=len(gc.get_objects()),
                gc_collections=gc_collections
            )
            
            self.snapshots.append(snapshot)
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to take memory snapshot: {e}")
            # Return empty snapshot
            return MemorySnapshot(
                timestamp=time.time(),
                rss_mb=0.0,
                vms_mb=0.0,
                percent=0.0,
                available_mb=0.0,
                gc_objects=0
            )
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """
        Start continuous memory monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring:
            return
        
        self._monitor_interval = interval
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Started memory monitoring with {interval}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop continuous memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Stopped memory monitoring")
    
    def _monitor_loop(self) -> None:
        """Continuous monitoring loop."""
        while self._monitoring:
            self.take_snapshot()
            time.sleep(self._monitor_interval)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        if not self.snapshots:
            return {"error": "No snapshots available"}
        
        recent_snapshots = list(self.snapshots)[-100:]  # Last 100 snapshots
        
        rss_values = [s.rss_mb for s in recent_snapshots]
        percent_values = [s.percent for s in recent_snapshots]
        
        current = recent_snapshots[-1]
        
        return {
            "current": {
                "rss_mb": current.rss_mb,
                "vms_mb": current.vms_mb,
                "percent": current.percent,
                "available_mb": current.available_mb,
                "gc_objects": current.gc_objects
            },
            "statistics": {
                "rss_mean": np.mean(rss_values),
                "rss_max": np.max(rss_values),
                "rss_min": np.min(rss_values),
                "rss_std": np.std(rss_values),
                "percent_mean": np.mean(percent_values),
                "percent_max": np.max(percent_values),
                "growth_rate_mb_per_min": self._calculate_growth_rate(recent_snapshots)
            },
            "snapshots_count": len(self.snapshots),
            "monitoring_active": self._monitoring
        }
    
    def _calculate_growth_rate(self, snapshots: List[MemorySnapshot]) -> float:
        """Calculate memory growth rate in MB per minute."""
        if len(snapshots) < 2:
            return 0.0
        
        first = snapshots[0]
        last = snapshots[-1]
        
        time_diff_minutes = (last.timestamp - first.timestamp) / 60.0
        memory_diff_mb = last.rss_mb - first.rss_mb
        
        if time_diff_minutes > 0:
            return memory_diff_mb / time_diff_minutes
        return 0.0
    
    def trigger_gc(self) -> Dict[str, Any]:
        """Trigger garbage collection and return statistics."""
        before_snapshot = self.take_snapshot()
        
        # Force garbage collection
        collected = gc.collect()
        
        after_snapshot = self.take_snapshot()
        
        memory_freed = before_snapshot.rss_mb - after_snapshot.rss_mb
        objects_freed = before_snapshot.gc_objects - after_snapshot.gc_objects
        
        return {
            "objects_collected": collected,
            "objects_freed": objects_freed,
            "memory_freed_mb": memory_freed,
            "memory_before_mb": before_snapshot.rss_mb,
            "memory_after_mb": after_snapshot.rss_mb
        }
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get memory optimization recommendations based on profiling data."""
        recommendations = []
        
        if not self.snapshots:
            return ["No profiling data available"]
        
        stats = self.get_memory_stats()
        current = stats["current"]
        statistics = stats["statistics"]
        
        # High memory usage
        if current["percent"] > 80:
            recommendations.append(
                f"High memory usage ({current['percent']:.1f}%). Consider reducing batch size or enabling garbage collection."
            )
        
        # Memory growth
        growth_rate = statistics["growth_rate_mb_per_min"]
        if growth_rate > 10:  # More than 10 MB/min growth
            recommendations.append(
                f"Memory growing rapidly ({growth_rate:.1f} MB/min). Check for memory leaks."
            )
        
        # High object count
        if current["gc_objects"] > 100000:
            recommendations.append(
                f"High object count ({current['gc_objects']:,}). Consider manual garbage collection."
            )
        
        # Memory fragmentation
        if current["vms_mb"] > current["rss_mb"] * 2:
            recommendations.append(
                "Possible memory fragmentation detected. Consider process restart."
            )
        
        # Low available memory
        if current["available_mb"] < 1000:  # Less than 1GB available
            recommendations.append(
                f"Low system memory available ({current['available_mb']:.0f} MB). Reduce memory usage."
            )
        
        if not recommendations:
            recommendations.append("Memory usage appears optimal.")
        
        return recommendations


class PerformanceProfiler:
    """
    Performance profiler for operations and throughput monitoring.
    """
    
    def __init__(self, memory_profiler: Optional[MemoryProfiler] = None):
        """
        Initialize performance profiler.
        
        Args:
            memory_profiler: Optional memory profiler instance
        """
        self.memory_profiler = memory_profiler or MemoryProfiler()
        self.metrics: List[PerformanceMetrics] = []
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        
    def profile_operation(self, operation_name: str) -> 'OperationProfiler':
        """
        Create a context manager for profiling an operation.
        
        Args:
            operation_name: Name of the operation to profile
            
        Returns:
            OperationProfiler context manager
        """
        return OperationProfiler(self, operation_name)
    
    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        self.metrics.append(metrics)
        self.operation_stats[metrics.operation].append(metrics.duration)
        
        # Keep only recent metrics to prevent memory growth
        if len(self.metrics) > 10000:
            self.metrics = self.metrics[-5000:]  # Keep last 5000
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        if operation not in self.operation_stats:
            return {"error": f"No data for operation: {operation}"}
        
        durations = self.operation_stats[operation]
        
        return {
            "operation": operation,
            "count": len(durations),
            "mean_duration": np.mean(durations),
            "median_duration": np.median(durations),
            "min_duration": np.min(durations),
            "max_duration": np.max(durations),
            "std_duration": np.std(durations),
            "total_duration": np.sum(durations)
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all operations."""
        stats = {}
        for operation in self.operation_stats.keys():
            stats[operation] = self.get_operation_stats(operation)
        
        return {
            "operations": stats,
            "total_metrics": len(self.metrics),
            "memory_stats": self.memory_profiler.get_memory_stats() if self.memory_profiler else None
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        if not self.metrics:
            return {"error": "No performance data available"}
        
        recent_metrics = self.metrics[-100:]  # Last 100 operations
        
        durations = [m.duration for m in recent_metrics]
        memory_usage = [m.memory_after for m in recent_metrics if m.memory_after > 0]
        throughputs = [m.throughput for m in recent_metrics if m.throughput is not None]
        
        summary = {
            "recent_operations": len(recent_metrics),
            "average_duration": np.mean(durations) if durations else 0,
            "total_duration": np.sum(durations) if durations else 0,
        }
        
        if memory_usage:
            summary["average_memory_mb"] = np.mean(memory_usage)
            summary["peak_memory_mb"] = np.max(memory_usage)
        
        if throughputs:
            summary["average_throughput"] = np.mean(throughputs)
            summary["peak_throughput"] = np.max(throughputs)
        
        return summary
    
    def clear_metrics(self) -> None:
        """Clear all recorded metrics."""
        self.metrics.clear()
        self.operation_stats.clear()
        logger.info("Performance metrics cleared")


class OperationProfiler:
    """Context manager for profiling individual operations."""
    
    def __init__(self, profiler: PerformanceProfiler, operation_name: str):
        """
        Initialize operation profiler.
        
        Args:
            profiler: Parent performance profiler
            operation_name: Name of the operation
        """
        self.profiler = profiler
        self.operation_name = operation_name
        self.start_time: Optional[float] = None
        self.memory_before: Optional[float] = None
        self.memory_peak: float = 0.0
        self.items_processed: Optional[int] = None
        
    def __enter__(self) -> 'OperationProfiler':
        """Start profiling."""
        self.start_time = time.time()
        
        # Take memory snapshot
        if self.profiler.memory_profiler:
            snapshot = self.profiler.memory_profiler.take_snapshot()
            self.memory_before = snapshot.rss_mb
            self.memory_peak = snapshot.rss_mb
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End profiling and record metrics."""
        if self.start_time is None:
            return
        
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Take final memory snapshot
        memory_after = 0.0
        if self.profiler.memory_profiler:
            snapshot = self.profiler.memory_profiler.take_snapshot()
            memory_after = snapshot.rss_mb
            self.memory_peak = max(self.memory_peak, memory_after)
        
        # Calculate throughput if items were processed
        throughput = None
        if self.items_processed is not None and duration > 0:
            throughput = self.items_processed / duration
        
        # Get CPU usage (approximate)
        try:
            cpu_percent = psutil.Process().cpu_percent()
        except:
            cpu_percent = 0.0
        
        # Record metrics
        metrics = PerformanceMetrics(
            operation=self.operation_name,
            duration=duration,
            memory_before=self.memory_before or 0.0,
            memory_after=memory_after,
            memory_peak=self.memory_peak,
            cpu_percent=cpu_percent,
            throughput=throughput,
            items_processed=self.items_processed
        )
        
        self.profiler.record_metrics(metrics)
    
    def set_items_processed(self, count: int) -> None:
        """Set the number of items processed for throughput calculation."""
        self.items_processed = count
    
    def update_memory_peak(self) -> None:
        """Update peak memory usage during operation."""
        if self.profiler.memory_profiler:
            snapshot = self.profiler.memory_profiler.take_snapshot()
            self.memory_peak = max(self.memory_peak, snapshot.rss_mb)