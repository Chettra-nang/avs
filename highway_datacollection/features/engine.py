"""
Feature derivation engine for processing observations.
"""

from typing import Dict, Any
import numpy as np
from .extractors import (
    KinematicsExtractor, 
    TrafficMetricsExtractor,
    calculate_time_to_collision,
    calculate_traffic_density
)
from .summarizer import LanguageSummarizer


class FeatureDerivationEngine:
    """
    Processes raw observations into derived metrics and summaries.
    """
    
    def __init__(self, lane_width: float = 4.0, num_lanes: int = 4):
        """
        Initialize feature derivation engine.
        
        Args:
            lane_width: Width of each lane in meters
            num_lanes: Total number of lanes
        """
        self.kinematics_extractor = KinematicsExtractor(lane_width, num_lanes)
        self.traffic_extractor = TrafficMetricsExtractor()
        self.summarizer = LanguageSummarizer(lane_width, num_lanes)
        self.lane_width = lane_width
        self.num_lanes = num_lanes
    
    def derive_kinematics_features(self, obs: np.ndarray) -> Dict[str, float]:
        """
        Derive features from kinematics observations.
        
        Args:
            obs: Kinematics observation array, shape (n_vehicles, 7)
                 Format: [presence, x, y, vx, vy, cos_h, sin_h]
            
        Returns:
            Dictionary of derived features
        """
        return self.kinematics_extractor.extract_features(obs)
    
    def calculate_ttc(self, ego: np.ndarray, others: np.ndarray) -> float:
        """
        Calculate minimum Time-to-Collision from relative dynamics.
        
        Args:
            ego: Ego vehicle state [presence, x, y, vx, vy, cos_h, sin_h]
            others: Other vehicles states array, shape (n_vehicles, 7)
            
        Returns:
            Minimum time to collision in seconds
        """
        if len(others) == 0:
            return float('inf')
        
        min_ttc = float('inf')
        for other_vehicle in others:
            ttc = calculate_time_to_collision(ego, other_vehicle)
            if ttc < min_ttc:
                min_ttc = ttc
        
        return min_ttc
    
    def generate_language_summary(self, ego: np.ndarray, others: np.ndarray, config: Dict = None) -> str:
        """
        Generate natural language description of driving context.
        
        Args:
            ego: Ego vehicle state [presence, x, y, vx, vy, cos_h, sin_h]
            others: Other vehicles states array
            config: Configuration parameters
            
        Returns:
            Natural language summary string
        """
        return self.summarizer.summarize(ego, others, config)
    
    def estimate_traffic_metrics(self, observations: np.ndarray) -> Dict[str, float]:
        """
        Estimate traffic density and flow metrics.
        
        Args:
            observations: Full observation array, shape (n_vehicles, 7)
            
        Returns:
            Dictionary of traffic metrics
        """
        return self.traffic_extractor.extract_features(observations)