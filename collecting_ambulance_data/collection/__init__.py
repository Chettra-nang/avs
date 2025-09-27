"""
Ambulance data collection module.

This module provides specialized data collection functionality for ambulance scenarios,
extending the existing highway data collection infrastructure to support emergency
vehicle data collection with multi-modal observations.
"""

from .ambulance_collector import AmbulanceDataCollector

__all__ = ['AmbulanceDataCollector']