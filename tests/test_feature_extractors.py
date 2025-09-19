"""
Unit tests for feature extraction utilities.
"""

import unittest
import numpy as np
from highway_datacollection.features.extractors import (
    estimate_lane_from_position,
    detect_lead_vehicle,
    calculate_time_to_collision,
    calculate_traffic_density,
    KinematicsExtractor,
    TrafficMetricsExtractor
)


class TestFeatureExtractors(unittest.TestCase):
    """Test cases for feature extraction functions."""
    
    def test_estimate_lane_from_position(self):
        """Test lane estimation from lateral position."""
        # Test with 4 lanes, 4m width each
        # Lane boundaries: [-8, -4, 0, 4, 8]
        # Lane 0: y in [-8, -4), Lane 1: y in [-4, 0), Lane 2: y in [0, 4), Lane 3: y in [4, 8]
        
        # Test rightmost lane (lane 0)
        self.assertEqual(estimate_lane_from_position(-6.0), 0)
        self.assertEqual(estimate_lane_from_position(-5.0), 0)
        
        # Test second lane (lane 1)
        self.assertEqual(estimate_lane_from_position(-2.0), 1)
        self.assertEqual(estimate_lane_from_position(-1.0), 1)
        
        # Test third lane (lane 2)
        self.assertEqual(estimate_lane_from_position(2.0), 2)
        self.assertEqual(estimate_lane_from_position(1.0), 2)
        
        # Test leftmost lane (lane 3)
        self.assertEqual(estimate_lane_from_position(6.0), 3)
        self.assertEqual(estimate_lane_from_position(7.0), 3)
        
        # Test boundary conditions
        self.assertEqual(estimate_lane_from_position(-10.0), 0)  # Far right
        self.assertEqual(estimate_lane_from_position(10.0), 3)   # Far left
    
    def test_detect_lead_vehicle(self):
        """Test lead vehicle detection and gap measurement."""
        # Ego vehicle at origin, lane 1
        ego_state = np.array([1, 0, -2, 10, 0, 1, 0])  # [presence, x, y, vx, vy, cos_h, sin_h]
        
        # Other vehicles
        other_vehicles = np.array([
            [1, 20, -2, 8, 0, 1, 0],   # Vehicle ahead in same lane
            [1, 30, -2, 5, 0, 1, 0],   # Another vehicle ahead in same lane
            [1, 15, 2, 12, 0, 1, 0],   # Vehicle ahead in different lane
            [1, -10, -2, 15, 0, 1, 0], # Vehicle behind in same lane
        ])
        
        lead_idx, gap = detect_lead_vehicle(ego_state, other_vehicles)
        
        # Should detect first vehicle (index 0) as lead vehicle
        self.assertEqual(lead_idx, 0)
        self.assertAlmostEqual(gap, 20.0, places=1)
        
        # Test with no vehicles ahead
        behind_vehicles = np.array([
            [1, -10, -2, 15, 0, 1, 0], # Vehicle behind
            [1, -5, 2, 12, 0, 1, 0],   # Vehicle behind in different lane
        ])
        
        lead_idx, gap = detect_lead_vehicle(ego_state, behind_vehicles)
        self.assertIsNone(lead_idx)
        self.assertEqual(gap, float('inf'))
        
        # Test with empty array
        lead_idx, gap = detect_lead_vehicle(ego_state, np.array([]))
        self.assertIsNone(lead_idx)
        self.assertEqual(gap, float('inf'))
    
    def test_calculate_time_to_collision(self):
        """Test TTC calculation with known scenarios."""
        # Test head-on collision scenario
        ego_state = np.array([1, 0, 0, 10, 0, 1, 0])      # Moving right at 10 m/s
        other_state = np.array([1, 50, 0, -10, 0, -1, 0]) # Moving left at 10 m/s, 50m ahead
        
        ttc = calculate_time_to_collision(ego_state, other_state)
        expected_ttc = 50.0 / 20.0  # 50m gap, 20 m/s closing speed
        self.assertAlmostEqual(ttc, expected_ttc, places=2)
        
        # Test parallel movement (no collision)
        ego_state = np.array([1, 0, 0, 10, 0, 1, 0])
        other_state = np.array([1, 0, 4, 10, 0, 1, 0])  # Same speed, different lane
        
        ttc = calculate_time_to_collision(ego_state, other_state)
        self.assertEqual(ttc, float('inf'))
        
        # Test vehicles moving apart
        ego_state = np.array([1, 0, 0, 10, 0, 1, 0])
        other_state = np.array([1, -20, 0, 5, 0, 1, 0])  # Behind and slower
        
        ttc = calculate_time_to_collision(ego_state, other_state)
        self.assertEqual(ttc, float('inf'))
        
        # Test with non-present vehicle
        ego_state = np.array([1, 0, 0, 10, 0, 1, 0])
        other_state = np.array([0, 50, 0, -10, 0, -1, 0])  # Not present
        
        ttc = calculate_time_to_collision(ego_state, other_state)
        self.assertEqual(ttc, float('inf'))
        
        # Test near-miss scenario (vehicles pass by safely)
        ego_state = np.array([1, 0, 0, 10, 0, 1, 0])
        other_state = np.array([1, 50, 3, -10, 0, -1, 0])  # 3m lateral offset
        
        ttc = calculate_time_to_collision(ego_state, other_state)
        self.assertEqual(ttc, float('inf'))  # Should be safe
    
    def test_calculate_traffic_density(self):
        """Test traffic density calculation."""
        ego_position = np.array([0, 0])
        
        # Test with vehicles at various distances
        observations = np.array([
            [1, 0, 0, 10, 0, 1, 0],    # Ego vehicle
            [1, 10, 0, 8, 0, 1, 0],    # 10m away
            [1, 30, 0, 5, 0, 1, 0],    # 30m away
            [1, 60, 0, 12, 0, 1, 0],   # 60m away (outside 50m radius)
            [0, 5, 0, 0, 0, 0, 0],     # Not present
        ])
        
        density = calculate_traffic_density(observations, ego_position, radius=50.0)
        
        # Should count 3 vehicles (ego + 2 within radius) in π * 0.05² km² area
        expected_area_km2 = np.pi * (0.05) ** 2  # 50m = 0.05km
        expected_density = 3 / expected_area_km2
        
        self.assertAlmostEqual(density, expected_density, places=1)
        
        # Test with empty observations
        density = calculate_traffic_density(np.array([]), ego_position)
        self.assertEqual(density, 0.0)
    
    def test_kinematics_extractor(self):
        """Test KinematicsExtractor class."""
        extractor = KinematicsExtractor(lane_width=4.0, num_lanes=4)
        
        # Create test observation with ego vehicle and others
        obs = np.array([
            [1, 0, -2, 15, 0, 1, 0],   # Ego vehicle in lane 1
            [1, 25, -2, 10, 0, 1, 0],  # Lead vehicle
            [1, 50, 2, 8, 0, 1, 0],    # Vehicle in different lane
            [1, 100, -2, -5, 0, -1, 0] # Oncoming vehicle (for TTC test)
        ])
        
        features = extractor.extract_features(obs)
        
        # Check expected features
        self.assertIn('lane_position', features)
        self.assertIn('speed', features)
        self.assertIn('lead_vehicle_gap', features)
        self.assertIn('min_ttc', features)
        self.assertIn('traffic_density', features)
        
        # Verify specific values
        self.assertEqual(features['lane_position'], 1)  # Lane 1
        self.assertAlmostEqual(features['speed'], 15.0, places=1)  # Speed magnitude
        self.assertAlmostEqual(features['lead_vehicle_gap'], 25.0, places=1)  # Gap to lead vehicle
        
        # Test with empty observation
        empty_features = extractor.extract_features(np.array([]))
        self.assertEqual(empty_features['lane_position'], 0)
        self.assertEqual(empty_features['speed'], 0.0)
        self.assertEqual(empty_features['lead_vehicle_gap'], float('inf'))
    
    def test_traffic_metrics_extractor(self):
        """Test TrafficMetricsExtractor class."""
        extractor = TrafficMetricsExtractor()
        
        # Create test observation
        obs = np.array([
            [1, 0, 0, 15, 0, 1, 0],    # Vehicle 1
            [1, 20, 0, 12, 0, 1, 0],   # Vehicle 2
            [1, 40, 4, 18, 0, 1, 0],   # Vehicle 3
            [0, 60, 0, 0, 0, 0, 0],    # Not present
        ])
        
        metrics = extractor.extract_features(obs)
        
        # Check expected metrics
        self.assertIn('vehicle_count', metrics)
        self.assertIn('average_speed', metrics)
        self.assertIn('speed_variance', metrics)
        self.assertIn('lane_change_opportunities', metrics)
        
        # Verify values
        self.assertEqual(metrics['vehicle_count'], 3)  # 3 present vehicles
        
        # Calculate expected average speed
        speeds = [15, 12, 18]  # Speeds of present vehicles
        expected_avg = sum(speeds) / len(speeds)
        self.assertAlmostEqual(metrics['average_speed'], expected_avg, places=1)
        
        # Test with empty observation
        empty_metrics = extractor.extract_features(np.array([]))
        self.assertEqual(empty_metrics['vehicle_count'], 0)
        self.assertEqual(empty_metrics['average_speed'], 0.0)


if __name__ == '__main__':
    unittest.main()