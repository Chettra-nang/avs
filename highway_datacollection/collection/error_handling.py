"""
Error handling utilities and custom exceptions for data collection.
"""

import logging
import traceback
import time
from typing import Dict, Any, List, Optional, Callable, Type
from dataclasses import dataclass
from enum import Enum
import functools

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DataCollectionError(Exception):
    """Base exception for data collection errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.severity = severity
        self.details = details or {}
        self.timestamp = time.time()


class EnvironmentSynchronizationError(DataCollectionError):
    """Raised when environment synchronization fails."""
    
    def __init__(self, message: str, desync_details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorSeverity.HIGH, desync_details)


class StorageError(DataCollectionError):
    """Raised when storage operations fail."""
    
    def __init__(self, message: str, storage_details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorSeverity.MEDIUM, storage_details)


class MemoryError(DataCollectionError):
    """Raised when memory limits are exceeded."""
    
    def __init__(self, message: str, memory_details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorSeverity.HIGH, memory_details)


class ValidationError(DataCollectionError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, validation_details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorSeverity.MEDIUM, validation_details)


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    component: str
    episode_id: Optional[str] = None
    step: Optional[int] = None
    scenario: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


@dataclass
class RecoveryAction:
    """Represents a recovery action that can be taken."""
    name: str
    description: str
    action: Callable[[], Any]
    success_probability: float  # 0.0 to 1.0


class ErrorHandler:
    """
    Centralized error handling and recovery system.
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize error handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retry attempts in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_history: List[Dict[str, Any]] = []
        self.recovery_strategies: Dict[Type[Exception], List[RecoveryAction]] = {}
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self) -> None:
        """Register default recovery strategies for common errors."""
        
        # Environment synchronization error recovery
        self.recovery_strategies[EnvironmentSynchronizationError] = [
            RecoveryAction(
                "reset_environments",
                "Reset all environments with fresh seeds",
                lambda: None,  # Placeholder - actual implementation in collector
                0.8
            ),
            RecoveryAction(
                "recreate_environments", 
                "Recreate environments from scratch",
                lambda: None,  # Placeholder
                0.9
            )
        ]
        
        # Storage error recovery
        self.recovery_strategies[StorageError] = [
            RecoveryAction(
                "retry_with_csv",
                "Retry storage operation with CSV fallback",
                lambda: None,  # Placeholder
                0.7
            ),
            RecoveryAction(
                "change_storage_location",
                "Try alternative storage location",
                lambda: None,  # Placeholder
                0.6
            )
        ]
        
        # Memory error recovery
        self.recovery_strategies[MemoryError] = [
            RecoveryAction(
                "garbage_collect",
                "Force garbage collection",
                lambda: None,  # Placeholder
                0.5
            ),
            RecoveryAction(
                "reduce_batch_size",
                "Reduce processing batch size",
                lambda: None,  # Placeholder
                0.8
            )
        ]
    
    def handle_error(self, error: Exception, context: ErrorContext, 
                    recovery_actions: Optional[List[RecoveryAction]] = None) -> Dict[str, Any]:
        """
        Handle an error with appropriate recovery strategies.
        
        Args:
            error: The exception that occurred
            context: Context information about the error
            recovery_actions: Custom recovery actions (overrides defaults)
            
        Returns:
            Dictionary with error handling results
        """
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": getattr(error, 'severity', ErrorSeverity.MEDIUM).value,
            "context": context,
            "timestamp": time.time(),
            "traceback": traceback.format_exc(),
            "recovery_attempted": False,
            "recovery_successful": False,
            "recovery_actions_tried": []
        }
        
        # Log the error
        logger.error(f"Error in {context.operation} ({context.component}): {str(error)}")
        
        # Add to error history
        self.error_history.append(error_info)
        
        # Keep error history manageable
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-50:]
        
        # Attempt recovery if strategies are available
        strategies = recovery_actions or self.recovery_strategies.get(type(error), [])
        
        if strategies:
            error_info["recovery_attempted"] = True
            
            for strategy in strategies:
                try:
                    logger.info(f"Attempting recovery strategy: {strategy.name}")
                    strategy.action()
                    error_info["recovery_successful"] = True
                    error_info["recovery_actions_tried"].append({
                        "name": strategy.name,
                        "success": True
                    })
                    logger.info(f"Recovery strategy '{strategy.name}' succeeded")
                    break
                    
                except Exception as recovery_error:
                    error_info["recovery_actions_tried"].append({
                        "name": strategy.name,
                        "success": False,
                        "error": str(recovery_error)
                    })
                    logger.warning(f"Recovery strategy '{strategy.name}' failed: {recovery_error}")
        
        return error_info
    
    def register_recovery_strategy(self, error_type: Type[Exception], 
                                 strategy: RecoveryAction) -> None:
        """
        Register a recovery strategy for a specific error type.
        
        Args:
            error_type: Type of exception this strategy handles
            strategy: Recovery action to register
        """
        if error_type not in self.recovery_strategies:
            self.recovery_strategies[error_type] = []
        
        self.recovery_strategies[error_type].append(strategy)
        logger.info(f"Registered recovery strategy '{strategy.name}' for {error_type.__name__}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics from the error history.
        
        Returns:
            Dictionary with error statistics
        """
        if not self.error_history:
            return {"total_errors": 0}
        
        # Count errors by type
        error_counts = {}
        severity_counts = {}
        recovery_success_rate = 0
        recovery_attempts = 0
        
        for error_info in self.error_history:
            error_type = error_info["error_type"]
            severity = error_info["severity"]
            
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            if error_info["recovery_attempted"]:
                recovery_attempts += 1
                if error_info["recovery_successful"]:
                    recovery_success_rate += 1
        
        if recovery_attempts > 0:
            recovery_success_rate = recovery_success_rate / recovery_attempts
        
        return {
            "total_errors": len(self.error_history),
            "error_counts_by_type": error_counts,
            "error_counts_by_severity": severity_counts,
            "recovery_attempts": recovery_attempts,
            "recovery_success_rate": recovery_success_rate,
            "recent_errors": self.error_history[-5:] if len(self.error_history) >= 5 else self.error_history
        }
    
    def clear_error_history(self) -> None:
        """Clear the error history."""
        self.error_history.clear()
        logger.info("Error history cleared")


def with_error_handling(error_handler: ErrorHandler, context: ErrorContext, 
                       max_retries: Optional[int] = None):
    """
    Decorator for adding error handling to functions.
    
    Args:
        error_handler: ErrorHandler instance to use
        context: Error context information
        max_retries: Maximum retry attempts (overrides handler default)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = max_retries or error_handler.max_retries
            last_error = None
            
            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_error = e
                    
                    if attempt < retries:
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {error_handler.retry_delay}s: {str(e)}")
                        time.sleep(error_handler.retry_delay)
                    else:
                        # Final attempt failed, handle error
                        error_info = error_handler.handle_error(e, context)
                        
                        # Re-raise if recovery was not successful
                        if not error_info["recovery_successful"]:
                            raise e
                        else:
                            # Try one more time after successful recovery
                            try:
                                return func(*args, **kwargs)
                            except Exception as recovery_error:
                                logger.error(f"Function failed even after successful recovery: {recovery_error}")
                                raise recovery_error
            
            # This should not be reached, but just in case
            if last_error:
                raise last_error
                
        return wrapper
    return decorator


class GracefulDegradationManager:
    """
    Manages graceful degradation of functionality when errors occur.
    """
    
    def __init__(self):
        """Initialize graceful degradation manager."""
        self.degraded_features: Dict[str, Dict[str, Any]] = {}
        self.fallback_strategies: Dict[str, Callable] = {}
    
    def register_fallback(self, feature: str, fallback_func: Callable, 
                         description: str = "") -> None:
        """
        Register a fallback strategy for a feature.
        
        Args:
            feature: Name of the feature
            fallback_func: Function to call as fallback
            description: Description of the fallback behavior
        """
        self.fallback_strategies[feature] = fallback_func
        logger.info(f"Registered fallback for feature '{feature}': {description}")
    
    def degrade_feature(self, feature: str, reason: str, 
                       severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> None:
        """
        Mark a feature as degraded.
        
        Args:
            feature: Name of the feature to degrade
            reason: Reason for degradation
            severity: Severity of the degradation
        """
        self.degraded_features[feature] = {
            "reason": reason,
            "severity": severity.value,
            "timestamp": time.time(),
            "has_fallback": feature in self.fallback_strategies
        }
        
        logger.warning(f"Feature '{feature}' degraded: {reason}")
    
    def restore_feature(self, feature: str) -> None:
        """
        Restore a degraded feature.
        
        Args:
            feature: Name of the feature to restore
        """
        if feature in self.degraded_features:
            del self.degraded_features[feature]
            logger.info(f"Feature '{feature}' restored")
    
    def is_feature_degraded(self, feature: str) -> bool:
        """
        Check if a feature is currently degraded.
        
        Args:
            feature: Name of the feature to check
            
        Returns:
            True if feature is degraded
        """
        return feature in self.degraded_features
    
    def execute_with_fallback(self, feature: str, primary_func: Callable, 
                            *args, **kwargs) -> Any:
        """
        Execute a function with fallback if the feature is degraded.
        
        Args:
            feature: Name of the feature
            primary_func: Primary function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of primary or fallback function
        """
        if not self.is_feature_degraded(feature):
            try:
                return primary_func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Primary function for '{feature}' failed: {e}")
                # Don't auto-degrade here, let caller decide
                raise
        
        # Feature is degraded, use fallback if available
        if feature in self.fallback_strategies:
            logger.info(f"Using fallback for degraded feature '{feature}'")
            return self.fallback_strategies[feature](*args, **kwargs)
        else:
            raise RuntimeError(f"Feature '{feature}' is degraded and no fallback is available")
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """
        Get current degradation status.
        
        Returns:
            Dictionary with degradation information
        """
        return {
            "degraded_features": dict(self.degraded_features),
            "available_fallbacks": list(self.fallback_strategies.keys()),
            "total_degraded": len(self.degraded_features)
        }