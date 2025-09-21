"""
Dataset storage manager for efficient multi-modal data persistence.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import uuid
import time
from datetime import datetime
import pandas as pd
import numpy as np
import logging
import shutil
import os

from .types import StoragePaths, ObservationRecord, EpisodeMetadata
from .encoders import BinaryArrayEncoder
from ..collection.error_handling import (
    StorageError, ErrorHandler, ErrorContext, ErrorSeverity,
    GracefulDegradationManager, with_error_handling
)
from ..performance import StorageThroughputMonitor, PerformanceConfig

logger = logging.getLogger(__name__)


class DatasetStorageManager:
    """
    Handles efficient storage of multi-modal observations and metadata.
    
    Supports Parquet format with CSV fallback for tabular data,
    binary blob encoding for large arrays, and JSONL metadata logging.
    """
    
    def __init__(self, base_path: Path, max_disk_usage_gb: float = 50.0,
                 performance_config: Optional[PerformanceConfig] = None):
        """
        Initialize storage manager.
        
        Args:
            base_path: Base directory for dataset storage
            max_disk_usage_gb: Maximum disk usage allowed in GB
            performance_config: Performance configuration for throughput monitoring
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._encoder = BinaryArrayEncoder()
        self.max_disk_usage_bytes = max_disk_usage_gb * 1024 * 1024 * 1024
        
        # Performance monitoring
        self.performance_config = performance_config or PerformanceConfig()
        self.throughput_monitor = StorageThroughputMonitor(self.performance_config)
        
        # Error handling
        self.error_handler = ErrorHandler()
        self.degradation_manager = GracefulDegradationManager()
        
        # Register fallback strategies
        self._register_fallback_strategies()
        
        # Storage statistics
        self.storage_stats = {
            "files_written": 0,
            "bytes_written": 0,
            "parquet_failures": 0,
            "csv_fallbacks": 0,
            "storage_errors": 0
        }
    
    def _register_fallback_strategies(self) -> None:
        """Register fallback strategies for storage operations."""
        
        def csv_fallback(*args, **kwargs):
            """Fallback to CSV when Parquet fails."""
            logger.info("Using CSV fallback for data storage")
            return "csv_fallback_used"
        
        def memory_storage_fallback(*args, **kwargs):
            """Fallback to in-memory storage when disk fails."""
            logger.warning("Using in-memory storage fallback - data will not persist")
            return "memory_fallback_used"
        
        self.degradation_manager.register_fallback(
            "parquet_storage", csv_fallback, "Use CSV format when Parquet fails"
        )
        self.degradation_manager.register_fallback(
            "disk_storage", memory_storage_fallback, "Use memory when disk is full"
        )
    
    def check_disk_space(self) -> Dict[str, Any]:
        """
        Check available disk space and usage.
        
        Returns:
            Dictionary with disk space information
        """
        try:
            # Get disk usage for the base path
            total, used, free = shutil.disk_usage(self.base_path)
            
            # Calculate dataset size
            dataset_size = self._calculate_dataset_size()
            
            return {
                "total_disk_gb": total / 1024**3,
                "used_disk_gb": used / 1024**3,
                "free_disk_gb": free / 1024**3,
                "dataset_size_gb": dataset_size / 1024**3,
                "disk_usage_percent": (used / total) * 100,
                "dataset_usage_percent": (dataset_size / self.max_disk_usage_bytes) * 100,
                "space_available": free > 1024**3,  # At least 1GB free
                "within_limits": dataset_size < self.max_disk_usage_bytes
            }
        except Exception as e:
            logger.error(f"Failed to check disk space: {e}")
            return {"error": str(e)}
    
    def _calculate_dataset_size(self) -> int:
        """Calculate total size of the dataset in bytes."""
        total_size = 0
        try:
            for root, dirs, files in os.walk(self.base_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
        except Exception as e:
            logger.warning(f"Failed to calculate dataset size: {e}")
        return total_size
    
    def write_episode_batch(self, data: List[Dict], metadata: List[Dict], scenario: str) -> StoragePaths:
        """
        Write a batch of episode data to storage with error handling.
        
        Args:
            data: List of observation records as dictionaries
            metadata: List of episode metadata as dictionaries
            scenario: Scenario name for organization
            
        Returns:
            StoragePaths containing file locations
            
        Raises:
            StorageError: If storage operation fails and cannot be recovered
        """
        context = ErrorContext(
            operation="write_episode_batch",
            component="DatasetStorageManager",
            scenario=scenario,
            additional_info={"data_records": len(data), "metadata_records": len(metadata)}
        )
        
        try:
            # Check disk space before writing
            disk_info = self.check_disk_space()
            if not disk_info.get("space_available", True):
                raise StorageError(
                    "Insufficient disk space for storage operation",
                    {"disk_info": disk_info}
                )
            
            if not disk_info.get("within_limits", True):
                logger.warning(f"Dataset approaching size limit: {disk_info.get('dataset_usage_percent', 0):.1f}%")
            
            # Create scenario directory
            scenario_dir = self.base_path / scenario
            scenario_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename with timestamp and UUID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            base_filename = f"{timestamp}-{unique_id}"
            
            transitions_file = scenario_dir / f"{base_filename}_transitions.parquet"
            metadata_file = scenario_dir / f"{base_filename}_meta.jsonl"
            
            # Write transitions data with error handling
            self._write_transitions_data_safe(data, transitions_file)
            
            # Write metadata with error handling
            self._write_metadata_safe(metadata, metadata_file)
            
            # Update statistics
            self.storage_stats["files_written"] += 2
            file_sizes = 0
            if transitions_file.exists():
                file_sizes += transitions_file.stat().st_size
            if metadata_file.exists():
                file_sizes += metadata_file.stat().st_size
            self.storage_stats["bytes_written"] += file_sizes
            
            return StoragePaths(
                transitions_file=transitions_file,
                metadata_file=metadata_file,
                scenario_dir=scenario_dir
            )
            
        except Exception as e:
            self.storage_stats["storage_errors"] += 1
            error_info = self.error_handler.handle_error(e, context)
            
            if not error_info["recovery_successful"]:
                raise StorageError(f"Failed to write episode batch: {str(e)}", {"context": context})
            
            # If recovery was successful, try again
            return self.write_episode_batch(data, metadata, scenario)
    
    def _write_transitions_data_safe(self, data: List[Dict], file_path: Path) -> None:
        """
        Write transitions data to Parquet with CSV fallback and error handling.
        
        Args:
            data: List of observation records
            file_path: Path to write the data
        """
        if not data:
            logger.warning("No data provided for transitions file")
            return
        
        # Convert to DataFrame
        try:
            df = pd.DataFrame(data)
        except Exception as e:
            raise StorageError(f"Failed to create DataFrame from data: {str(e)}")
        
        # Try Parquet first
        if not self.degradation_manager.is_feature_degraded("parquet_storage"):
            try:
                start_time = time.time()
                
                # Configure compression if enabled
                compression = None
                if self.performance_config.enable_compression:
                    compression = 'snappy'  # Fast compression
                
                df.to_parquet(file_path, index=False, engine='pyarrow', compression=compression)
                
                # Record throughput
                duration = time.time() - start_time
                file_size = file_path.stat().st_size if file_path.exists() else 0
                self.throughput_monitor.record_write_operation(
                    file_size, duration, "parquet", file_path
                )
                
                logger.debug(f"Successfully wrote Parquet file: {file_path}")
                return
            except Exception as e:
                logger.warning(f"Parquet write failed: {e}")
                self.storage_stats["parquet_failures"] += 1
                self.degradation_manager.degrade_feature(
                    "parquet_storage", 
                    f"Parquet write failed: {str(e)}", 
                    ErrorSeverity.MEDIUM
                )
        
        # Fallback to CSV
        try:
            csv_path = file_path.with_suffix('.csv')
            start_time = time.time()
            
            df.to_csv(csv_path, index=False)
            
            # Record throughput for CSV fallback
            duration = time.time() - start_time
            file_size = csv_path.stat().st_size if csv_path.exists() else 0
            self.throughput_monitor.record_write_operation(
                file_size, duration, "csv", csv_path
            )
            
            self.storage_stats["csv_fallbacks"] += 1
            logger.info(f"Used CSV fallback: {csv_path}")
        except Exception as e:
            raise StorageError(f"Both Parquet and CSV writes failed: {str(e)}")
    
    def _write_transitions_data(self, data: List[Dict], file_path: Path) -> None:
        """
        Legacy method - redirects to safe version.
        
        Args:
            data: List of observation records
            file_path: Path to write the data
        """
        self._write_transitions_data_safe(data, file_path)
    
    def _write_metadata_safe(self, metadata: List[Dict], file_path: Path) -> None:
        """
        Write episode metadata as JSONL with error handling.
        
        Args:
            metadata: List of episode metadata dictionaries
            file_path: Path to write the metadata
        """
        if not metadata:
            logger.warning("No metadata provided for metadata file")
            return
        
        try:
            # Write to temporary file first
            temp_path = file_path.with_suffix('.tmp')
            start_time = time.time()
            
            with open(temp_path, 'w') as f:
                for meta in metadata:
                    try:
                        json.dump(meta, f)
                        f.write('\n')
                    except (TypeError, ValueError) as e:
                        logger.error(f"Failed to serialize metadata record: {e}")
                        # Write error placeholder
                        error_record = {
                            "error": "serialization_failed",
                            "original_error": str(e),
                            "timestamp": datetime.now().isoformat()
                        }
                        json.dump(error_record, f)
                        f.write('\n')
            
            # Atomic move to final location
            temp_path.rename(file_path)
            
            # Record throughput
            duration = time.time() - start_time
            file_size = file_path.stat().st_size if file_path.exists() else 0
            self.throughput_monitor.record_write_operation(
                file_size, duration, "jsonl", file_path
            )
            
            logger.debug(f"Successfully wrote metadata file: {file_path}")
            
        except Exception as e:
            # Clean up temp file if it exists
            temp_path = file_path.with_suffix('.tmp')
            if temp_path.exists():
                temp_path.unlink()
            raise StorageError(f"Failed to write metadata file: {str(e)}")
    
    def _write_metadata(self, metadata: List[Dict], file_path: Path) -> None:
        """
        Legacy method - redirects to safe version.
        
        Args:
            metadata: List of episode metadata dictionaries
            file_path: Path to write the metadata
        """
        self._write_metadata_safe(metadata, file_path)
    
    def encode_binary_arrays(self, arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Encode large arrays as binary blobs with metadata.
        
        Args:
            arrays: Dictionary of arrays to encode (e.g., {'occ': array, 'gray': array})
            
        Returns:
            Dictionary with encoded blobs and metadata
        """
        return self._encoder.encode_multiple(arrays)
    
    def decode_binary_arrays(self, data: Dict[str, Any], array_keys: List[str]) -> Dict[str, np.ndarray]:
        """
        Decode binary blobs back to arrays.
        
        Args:
            data: Dictionary containing encoded array data
            array_keys: List of array keys to decode
            
        Returns:
            Dictionary of decoded arrays
        """
        return self._encoder.decode_multiple(data, array_keys)
    
    def create_dataset_index(self, scenario_paths: List[StoragePaths]) -> Path:
        """
        Create global dataset index file.
        
        Args:
            scenario_paths: List of storage paths for all scenarios
            
        Returns:
            Path to created index file
        """
        index_path = self.base_path / "index.json"
        
        # Group paths by scenario
        scenarios = {}
        for paths in scenario_paths:
            scenario_name = paths.scenario_dir.name
            if scenario_name not in scenarios:
                scenarios[scenario_name] = []
            
            scenarios[scenario_name].append({
                'transitions_file': str(paths.transitions_file.relative_to(self.base_path)),
                'metadata_file': str(paths.metadata_file.relative_to(self.base_path)),
                'created': datetime.now().isoformat()
            })
        
        # Create index structure
        index_data = {
            'dataset_name': 'highway_multimodal_datacollection',
            'created': datetime.now().isoformat(),
            'base_path': str(self.base_path),
            'scenarios': scenarios,
            'total_files': len(scenario_paths)
        }
        
        # Write index file
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        return index_path
    
    def generate_episode_id(self) -> str:
        """
        Generate unique episode ID.
        
        Returns:
            Unique episode identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"ep_{timestamp}_{unique_id}"
    
    def get_scenario_files(self, scenario: str) -> List[Path]:
        """
        Get all files for a specific scenario.
        
        Args:
            scenario: Scenario name
            
        Returns:
            List of file paths for the scenario
        """
        scenario_dir = self.base_path / scenario
        if not scenario_dir.exists():
            return []
        
        files = []
        files.extend(scenario_dir.glob("*_transitions.parquet"))
        files.extend(scenario_dir.glob("*_transitions.csv"))
        files.extend(scenario_dir.glob("*_meta.jsonl"))
        
        return sorted(files)
    
    def load_episode_metadata(self, metadata_file: Path) -> List[EpisodeMetadata]:
        """
        Load episode metadata from JSONL file.
        
        Args:
            metadata_file: Path to metadata file
            
        Returns:
            List of EpisodeMetadata objects
        """
        metadata_list = []
        
        with open(metadata_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    data = json.loads(line)
                    metadata_list.append(EpisodeMetadata(**data))
        
        return metadata_list
    
    def organize_dataset_structure(self) -> Dict[str, Any]:
        """
        Analyze and organize the current dataset structure.
        
        Returns:
            Dictionary with dataset organization information
        """
        structure = {
            'base_path': str(self.base_path),
            'scenarios': {},
            'total_episodes': 0,
            'total_files': 0
        }
        
        # Scan all scenario directories
        for scenario_dir in self.base_path.iterdir():
            if scenario_dir.is_dir() and scenario_dir.name != '__pycache__':
                scenario_info = self._analyze_scenario_directory(scenario_dir)
                structure['scenarios'][scenario_dir.name] = scenario_info
                structure['total_episodes'] += scenario_info['episode_count']
                structure['total_files'] += scenario_info['file_count']
        
        return structure
    
    def _analyze_scenario_directory(self, scenario_dir: Path) -> Dict[str, Any]:
        """
        Analyze a single scenario directory.
        
        Args:
            scenario_dir: Path to scenario directory
            
        Returns:
            Dictionary with scenario information
        """
        info = {
            'path': str(scenario_dir),
            'episode_count': 0,
            'file_count': 0,
            'transitions_files': [],
            'metadata_files': [],
            'file_sizes': {}
        }
        
        # Count files and gather information
        for file_path in scenario_dir.iterdir():
            if file_path.is_file():
                info['file_count'] += 1
                info['file_sizes'][file_path.name] = file_path.stat().st_size
                
                if 'transitions' in file_path.name:
                    info['transitions_files'].append(file_path.name)
                elif 'meta' in file_path.name:
                    info['metadata_files'].append(file_path.name)
                    # Count episodes from metadata file
                    try:
                        metadata = self.load_episode_metadata(file_path)
                        info['episode_count'] += len(metadata)
                    except Exception:
                        pass  # Skip corrupted files
        
        return info
    
    def validate_dataset_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity of the dataset structure.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        structure = self.organize_dataset_structure()
        
        for scenario_name, scenario_info in structure['scenarios'].items():
            # Check for orphaned files
            transitions_count = len(scenario_info['transitions_files'])
            metadata_count = len(scenario_info['metadata_files'])
            
            if transitions_count != metadata_count:
                validation['errors'].append(
                    f"Scenario '{scenario_name}': Mismatch between transitions files "
                    f"({transitions_count}) and metadata files ({metadata_count})"
                )
                validation['valid'] = False
            
            # Check for empty files
            for filename, size in scenario_info['file_sizes'].items():
                if size == 0:
                    validation['warnings'].append(
                        f"Scenario '{scenario_name}': Empty file '{filename}'"
                    )
        
        validation['statistics'] = {
            'total_scenarios': len(structure['scenarios']),
            'total_episodes': structure['total_episodes'],
            'total_files': structure['total_files']
        }
        
        return validation
    
    def cleanup_empty_directories(self) -> List[str]:
        """
        Remove empty scenario directories.
        
        Returns:
            List of removed directory names
        """
        removed_dirs = []
        
        for scenario_dir in self.base_path.iterdir():
            if scenario_dir.is_dir() and scenario_dir.name != '__pycache__':
                # Check if directory is empty or contains only empty files
                files = list(scenario_dir.iterdir())
                if not files or all(f.stat().st_size == 0 for f in files if f.is_file()):
                    import shutil
                    shutil.rmtree(scenario_dir)
                    removed_dirs.append(scenario_dir.name)
        
        return removed_dirs
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get storage operation statistics including throughput metrics.
        
        Returns:
            Dictionary with storage statistics
        """
        disk_info = self.check_disk_space()
        throughput_stats = self.throughput_monitor.get_throughput_stats()
        
        return {
            **self.storage_stats,
            "disk_info": disk_info,
            "throughput_stats": throughput_stats,
            "throughput_recommendations": self.throughput_monitor.get_optimization_recommendations(),
            "degraded_features": self.degradation_manager.get_degradation_status(),
            "error_stats": self.error_handler.get_error_statistics()
        }
    
    def reset_statistics(self) -> None:
        """Reset storage statistics including throughput measurements."""
        self.storage_stats = {
            "files_written": 0,
            "bytes_written": 0,
            "parquet_failures": 0,
            "csv_fallbacks": 0,
            "storage_errors": 0
        }
        self.throughput_monitor.clear_measurements()
        self.error_handler.clear_error_history()
        logger.info("Storage statistics reset")
    
    def perform_maintenance(self) -> Dict[str, Any]:
        """
        Perform storage maintenance operations.
        
        Returns:
            Dictionary with maintenance results
        """
        maintenance_results = {
            "cleanup_performed": False,
            "errors_found": 0,
            "space_freed_mb": 0,
            "files_removed": 0
        }
        
        try:
            # Validate dataset integrity
            validation = self.validate_dataset_integrity()
            maintenance_results["errors_found"] = len(validation["errors"])
            
            # Clean up empty directories
            removed_dirs = self.cleanup_empty_directories()
            maintenance_results["files_removed"] = len(removed_dirs)
            maintenance_results["cleanup_performed"] = len(removed_dirs) > 0
            
            # Calculate space freed (approximate)
            if removed_dirs:
                maintenance_results["space_freed_mb"] = len(removed_dirs) * 0.1  # Rough estimate
            
            logger.info(f"Maintenance completed: {maintenance_results}")
            
        except Exception as e:
            logger.error(f"Maintenance failed: {e}")
            maintenance_results["maintenance_error"] = str(e)
        
        return maintenance_results