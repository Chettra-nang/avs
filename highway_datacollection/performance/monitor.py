"""
Storage throughput monitoring and batch optimization.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging
import numpy as np
from pathlib import Path

from .config import PerformanceConfig
from .profiler import MemoryProfiler

logger = logging.getLogger(__name__)


@dataclass
class ThroughputMeasurement:
    """Single throughput measurement."""
    timestamp: float
    bytes_written: int
    duration: float
    throughput_mbps: float
    operation_type: str
    file_path: Optional[str] = None


@dataclass
class BatchMetrics:
    """Metrics for a batch processing operation."""
    batch_size: int
    processing_time: float
    memory_usage_mb: float
    throughput_items_per_sec: float
    success_rate: float
    errors: List[str] = field(default_factory=list)


class StorageThroughputMonitor:
    """
    Monitor storage throughput and optimize compression settings.
    """
    
    def __init__(self, config: PerformanceConfig):
        """
        Initialize throughput monitor.
        
        Args:
            config: Performance configuration
        """
        self.config = config
        self.measurements: deque = deque(maxlen=1000)
        self._monitoring = False
        self._lock = threading.Lock()
        
    def record_write_operation(self, bytes_written: int, duration: float, 
                              operation_type: str, file_path: Optional[Path] = None) -> None:
        """
        Record a storage write operation.
        
        Args:
            bytes_written: Number of bytes written
            duration: Time taken for the operation
            operation_type: Type of operation (parquet, csv, jsonl, etc.)
            file_path: Optional path to the file
        """
        if duration <= 0:
            return
        
        throughput_mbps = (bytes_written / (1024 * 1024)) / duration
        
        measurement = ThroughputMeasurement(
            timestamp=time.time(),
            bytes_written=bytes_written,
            duration=duration,
            throughput_mbps=throughput_mbps,
            operation_type=operation_type,
            file_path=str(file_path) if file_path else None
        )
        
        with self._lock:
            self.measurements.append(measurement)
        
        # Log warning if throughput is low
        if (self.config.throughput_monitoring and 
            throughput_mbps < self.config.throughput_warning_threshold):
            logger.warning(f"Low storage throughput: {throughput_mbps:.2f} MB/s for {operation_type}")
    
    def get_throughput_stats(self, operation_type: Optional[str] = None, 
                           time_window_minutes: Optional[float] = None) -> Dict[str, Any]:
        """
        Get throughput statistics.
        
        Args:
            operation_type: Filter by operation type
            time_window_minutes: Only include measurements from last N minutes
            
        Returns:
            Dictionary with throughput statistics
        """
        with self._lock:
            measurements = list(self.measurements)
        
        if not measurements:
            return {"error": "No throughput measurements available"}
        
        # Filter by time window
        if time_window_minutes:
            cutoff_time = time.time() - (time_window_minutes * 60)
            measurements = [m for m in measurements if m.timestamp >= cutoff_time]
        
        # Filter by operation type
        if operation_type:
            measurements = [m for m in measurements if m.operation_type == operation_type]
        
        if not measurements:
            return {"error": f"No measurements found for criteria"}
        
        throughputs = [m.throughput_mbps for m in measurements]
        durations = [m.duration for m in measurements]
        bytes_written = [m.bytes_written for m in measurements]
        
        return {
            "operation_type": operation_type or "all",
            "measurement_count": len(measurements),
            "time_window_minutes": time_window_minutes,
            "throughput_stats": {
                "mean_mbps": np.mean(throughputs),
                "median_mbps": np.median(throughputs),
                "min_mbps": np.min(throughputs),
                "max_mbps": np.max(throughputs),
                "std_mbps": np.std(throughputs)
            },
            "duration_stats": {
                "mean_seconds": np.mean(durations),
                "median_seconds": np.median(durations),
                "min_seconds": np.min(durations),
                "max_seconds": np.max(durations)
            },
            "volume_stats": {
                "total_mb": sum(bytes_written) / (1024 * 1024),
                "mean_mb": np.mean(bytes_written) / (1024 * 1024),
                "max_mb": np.max(bytes_written) / (1024 * 1024)
            }
        }
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get storage optimization recommendations."""
        recommendations = []
        
        stats = self.get_throughput_stats(time_window_minutes=10)  # Last 10 minutes
        if "error" in stats:
            return ["No recent throughput data available for recommendations"]
        
        throughput_stats = stats["throughput_stats"]
        mean_throughput = throughput_stats["mean_mbps"]
        
        # Low throughput recommendations
        if mean_throughput < self.config.throughput_warning_threshold:
            recommendations.append(
                f"Low storage throughput ({mean_throughput:.2f} MB/s). "
                "Consider reducing compression level or using faster storage."
            )
        
        # High variance recommendations
        std_throughput = throughput_stats["std_mbps"]
        if std_throughput > mean_throughput * 0.5:  # High variance
            recommendations.append(
                "High throughput variance detected. Storage performance may be inconsistent."
            )
        
        # Operation-specific recommendations
        for op_type in ["parquet", "csv", "jsonl"]:
            op_stats = self.get_throughput_stats(operation_type=op_type, time_window_minutes=10)
            if "error" not in op_stats:
                op_throughput = op_stats["throughput_stats"]["mean_mbps"]
                if op_type == "parquet" and op_throughput < 5.0:  # Parquet should be faster
                    recommendations.append(
                        f"Parquet throughput is low ({op_throughput:.2f} MB/s). "
                        "Consider adjusting compression settings."
                    )
        
        if not recommendations:
            recommendations.append("Storage throughput appears optimal.")
        
        return recommendations
    
    def clear_measurements(self) -> None:
        """Clear all throughput measurements."""
        with self._lock:
            self.measurements.clear()
        logger.info("Throughput measurements cleared")


class BatchingOptimizer:
    """
    Optimize batch sizes for memory-efficient processing.
    """
    
    def __init__(self, config: PerformanceConfig, memory_profiler: Optional[MemoryProfiler] = None):
        """
        Initialize batching optimizer.
        
        Args:
            config: Performance configuration
            memory_profiler: Optional memory profiler for memory-based optimization
        """
        self.config = config
        self.memory_profiler = memory_profiler
        self.batch_metrics: List[BatchMetrics] = []
        self.current_batch_size = config.default_batch_size
        self._optimization_history: List[Tuple[int, float]] = []  # (batch_size, performance_score)
        
    def record_batch_metrics(self, batch_size: int, processing_time: float, 
                           memory_usage_mb: float, items_processed: int, 
                           errors: Optional[List[str]] = None) -> None:
        """
        Record metrics for a batch processing operation.
        
        Args:
            batch_size: Size of the processed batch
            processing_time: Time taken to process the batch
            memory_usage_mb: Peak memory usage during processing
            items_processed: Number of items successfully processed
            errors: List of errors encountered
        """
        if processing_time <= 0 or batch_size <= 0:
            return
        
        throughput = items_processed / processing_time
        success_rate = items_processed / batch_size if batch_size > 0 else 0.0
        
        metrics = BatchMetrics(
            batch_size=batch_size,
            processing_time=processing_time,
            memory_usage_mb=memory_usage_mb,
            throughput_items_per_sec=throughput,
            success_rate=success_rate,
            errors=errors or []
        )
        
        self.batch_metrics.append(metrics)
        
        # Keep only recent metrics
        if len(self.batch_metrics) > 1000:
            self.batch_metrics = self.batch_metrics[-500:]
        
        # Calculate performance score and update history
        performance_score = self._calculate_performance_score(metrics)
        self._optimization_history.append((batch_size, performance_score))
        
        # Keep optimization history manageable
        if len(self._optimization_history) > 100:
            self._optimization_history = self._optimization_history[-50:]
    
    def _calculate_performance_score(self, metrics: BatchMetrics) -> float:
        """
        Calculate a performance score for batch metrics.
        Higher score is better.
        
        Args:
            metrics: Batch metrics to score
            
        Returns:
            Performance score (0-100)
        """
        # Base score from throughput (items/sec)
        throughput_score = min(metrics.throughput_items_per_sec * 10, 50)  # Max 50 points
        
        # Success rate score
        success_score = metrics.success_rate * 30  # Max 30 points
        
        # Memory efficiency score (lower memory usage is better)
        max_reasonable_memory = self.config.max_memory_gb * 1024 * 0.8  # 80% of max
        if metrics.memory_usage_mb <= max_reasonable_memory:
            memory_score = 20  # Max 20 points
        else:
            # Penalize high memory usage
            memory_score = max(0, 20 - (metrics.memory_usage_mb - max_reasonable_memory) / 100)
        
        return throughput_score + success_score + memory_score
    
    def get_optimal_batch_size(self) -> int:
        """
        Get the optimal batch size based on recorded metrics.
        
        Returns:
            Recommended batch size
        """
        if not self.config.adaptive_batching:
            return self.config.default_batch_size
        
        if len(self._optimization_history) < 3:
            return self.current_batch_size
        
        # Find batch size with highest average performance score
        batch_scores: Dict[int, List[float]] = {}
        for batch_size, score in self._optimization_history[-20:]:  # Last 20 measurements
            if batch_size not in batch_scores:
                batch_scores[batch_size] = []
            batch_scores[batch_size].append(score)
        
        # Calculate average scores
        avg_scores = {
            batch_size: np.mean(scores) 
            for batch_size, scores in batch_scores.items()
            if len(scores) >= 2  # Need at least 2 measurements
        }
        
        if not avg_scores:
            return self.current_batch_size
        
        # Find optimal batch size
        optimal_batch_size = max(avg_scores.keys(), key=lambda k: avg_scores[k])
        
        # Apply constraints
        optimal_batch_size = max(self.config.min_batch_size, optimal_batch_size)
        optimal_batch_size = min(self.config.max_batch_size, optimal_batch_size)
        
        # Memory-based adjustment
        if self.config.memory_based_batching and self.memory_profiler:
            memory_stats = self.memory_profiler.get_memory_stats()
            if "current" in memory_stats:
                memory_percent = memory_stats["current"]["percent"]
                
                # Reduce batch size if memory usage is high
                if memory_percent > self.config.memory_warning_threshold * 100:
                    optimal_batch_size = max(
                        self.config.min_batch_size,
                        int(optimal_batch_size * 0.7)  # Reduce by 30%
                    )
                elif memory_percent > self.config.memory_critical_threshold * 100:
                    optimal_batch_size = self.config.min_batch_size
        
        return optimal_batch_size
    
    def suggest_batch_size_adjustment(self) -> Dict[str, Any]:
        """
        Suggest batch size adjustment based on recent performance.
        
        Returns:
            Dictionary with adjustment suggestion
        """
        if not self.batch_metrics:
            return {"suggestion": "no_data", "message": "No batch metrics available"}
        
        recent_metrics = self.batch_metrics[-10:]  # Last 10 batches
        current_avg_throughput = np.mean([m.throughput_items_per_sec for m in recent_metrics])
        current_avg_memory = np.mean([m.memory_usage_mb for m in recent_metrics])
        current_avg_success = np.mean([m.success_rate for m in recent_metrics])
        
        optimal_size = self.get_optimal_batch_size()
        
        suggestion = {
            "current_batch_size": self.current_batch_size,
            "suggested_batch_size": optimal_size,
            "current_performance": {
                "throughput_items_per_sec": current_avg_throughput,
                "memory_usage_mb": current_avg_memory,
                "success_rate": current_avg_success
            }
        }
        
        # Determine adjustment type
        if optimal_size > self.current_batch_size:
            suggestion["adjustment"] = "increase"
            suggestion["message"] = f"Consider increasing batch size to {optimal_size} for better throughput"
        elif optimal_size < self.current_batch_size:
            suggestion["adjustment"] = "decrease"
            suggestion["message"] = f"Consider decreasing batch size to {optimal_size} to reduce memory usage"
        else:
            suggestion["adjustment"] = "maintain"
            suggestion["message"] = "Current batch size appears optimal"
        
        return suggestion
    
    def update_batch_size(self, new_size: int) -> None:
        """
        Update the current batch size.
        
        Args:
            new_size: New batch size to use
        """
        new_size = max(self.config.min_batch_size, new_size)
        new_size = min(self.config.max_batch_size, new_size)
        
        if new_size != self.current_batch_size:
            logger.info(f"Updating batch size from {self.current_batch_size} to {new_size}")
            self.current_batch_size = new_size
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get comprehensive batch processing statistics."""
        if not self.batch_metrics:
            return {"error": "No batch metrics available"}
        
        recent_metrics = self.batch_metrics[-50:]  # Last 50 batches
        
        batch_sizes = [m.batch_size for m in recent_metrics]
        throughputs = [m.throughput_items_per_sec for m in recent_metrics]
        memory_usage = [m.memory_usage_mb for m in recent_metrics]
        success_rates = [m.success_rate for m in recent_metrics]
        processing_times = [m.processing_time for m in recent_metrics]
        
        return {
            "total_batches": len(self.batch_metrics),
            "recent_batches": len(recent_metrics),
            "current_batch_size": self.current_batch_size,
            "batch_size_stats": {
                "mean": np.mean(batch_sizes),
                "min": np.min(batch_sizes),
                "max": np.max(batch_sizes),
                "std": np.std(batch_sizes)
            },
            "throughput_stats": {
                "mean_items_per_sec": np.mean(throughputs),
                "max_items_per_sec": np.max(throughputs),
                "min_items_per_sec": np.min(throughputs),
                "std_items_per_sec": np.std(throughputs)
            },
            "memory_stats": {
                "mean_mb": np.mean(memory_usage),
                "max_mb": np.max(memory_usage),
                "min_mb": np.min(memory_usage),
                "std_mb": np.std(memory_usage)
            },
            "success_stats": {
                "mean_rate": np.mean(success_rates),
                "min_rate": np.min(success_rates),
                "total_errors": sum(len(m.errors) for m in recent_metrics)
            },
            "timing_stats": {
                "mean_seconds": np.mean(processing_times),
                "max_seconds": np.max(processing_times),
                "min_seconds": np.min(processing_times),
                "total_seconds": np.sum(processing_times)
            }
        }
    
    def clear_metrics(self) -> None:
        """Clear all batch metrics."""
        self.batch_metrics.clear()
        self._optimization_history.clear()
        logger.info("Batch metrics cleared")