"""
Unit tests for FeatureDerivationEngine.
"""

import unittest
import numpy as np
from highway_datacollection.features.engine import FeatureDerivationEngine


class TestFeatureDerivationEngine(unittest.TestCase):
    """Test cases for FeatureDerivationEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = FeatureDerivationEngine(lane_width=4.0, num_lanes=4)
    
    def test_derive_kinematics_features(self):
        """Test kinematics feature derivation."""
        # Create test observation
        obs = np.array([
            [1, 0, -2, 15, 0, 1, 0],   # Ego vehicle
            [1, 30, -2, 10, 0, 1, 0],  # Lead vehicle
            [1, 50, 2, 12, 0, 1, 0],   # Vehicle in different lane
        ])
        
        features = self.engine.derive_kinematics_features(obs)
        
        # Verify feature structure
        expected_keys = [
            'lane_position', 'speed', 'lead_vehicle_gap', 
            'min_ttc', 'traffic_density', 'ego_x', 'ego_y', 'ego_vx', 'ego_vy'
        ]
        for key in expected_keys:
            self.assertIn(key, features)
        
        # Verify specific values
        self.assertEqual(features['lane_position'], 1)
        self.assertAlmostEqual(features['speed'], 15.0, places=1)
        self.assertAlmostEqual(features['lead_vehicle_gap'], 30.0, places=1)
        self.assertEqual(features['ego_x'], 0.0)
        self.assertEqual(features['ego_y'], -2.0)
    
    def test_calculate_ttc(self):
        """Test TTC calculation method."""
        ego = np.array([1, 0, 0, 10, 0, 1, 0])
        others = np.array([
            [1, 50, 0, -10, 0, -1, 0],  # Head-on collision
            [1, 100, 4, 5, 0, 1, 0],    # Parallel, no collision
        ])
        
        ttc = self.engine.calculate_ttc(ego, others)
        
        # Should return the minimum TTC (from head-on collision)
        expected_ttc = 50.0 / 20.0  # 50m gap, 20 m/s closing speed
        self.assertAlmostEqual(ttc, expected_ttc, places=2)
        
        # Test with no other vehicles
        ttc_empty = self.engine.calculate_ttc(ego, np.array([]))
        self.assertEqual(ttc_empty, float('inf'))
    
    def test_estimate_traffic_metrics(self):
        """Test traffic metrics estimation."""
        obs = np.array([
            [1, 0, 0, 15, 0, 1, 0],
            [1, 20, 0, 12, 0, 1, 0],
            [1, 40, 4, 18, 0, 1, 0],
            [0, 60, 0, 0, 0, 0, 0],  # Not present
        ])
        
        metrics = self.engine.estimate_traffic_metrics(obs)
        
        # Verify metric structure
        expected_keys = [
            'vehicle_count', 'average_speed', 'speed_variance', 'lane_change_opportunities'
        ]
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Verify values
        self.assertEqual(metrics['vehicle_count'], 3)
        
        # Calculate expected average speed
        speeds = [15, 12, 18]
        expected_avg = sum(speeds) / len(speeds)
        self.assertAlmostEqual(metrics['average_speed'], expected_avg, places=1)
    
    def test_generate_language_summary(self):
        """Test language summary generation."""
        ego = np.array([1, 0, -2, 15, 0, 1, 0])  # Right lane, 54 km/h
        others = np.array([[1, 30, -2, 12, 0, 1, 0]])  # Lead vehicle
        
        summary = self.engine.generate_language_summary(ego, others)
        
        # Should return a non-empty string
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 30)
        
        # Should contain relevant information
        self.assertIn("right lane", summary.lower())
        self.assertIn("54", summary)  # Speed in km/h
        
        # Test with scenario context
        context = {'scenario': 'free_flow'}
        summary_with_context = self.engine.generate_language_summary(ego, others, context)
        self.assertIsInstance(summary_with_context, str)
        self.assertGreater(len(summary_with_context), 30)
    
    def test_empty_observations(self):
        """Test behavior with empty observations."""
        empty_obs = np.array([])
        
        # Kinematics features should handle empty input gracefully
        features = self.engine.derive_kinematics_features(empty_obs)
        self.assertEqual(features['lane_position'], 0)
        self.assertEqual(features['speed'], 0.0)
        self.assertEqual(features['lead_vehicle_gap'], float('inf'))
        
        # Traffic metrics should handle empty input gracefully
        metrics = self.engine.estimate_traffic_metrics(empty_obs)
        self.assertEqual(metrics['vehicle_count'], 0)
        self.assertEqual(metrics['average_speed'], 0.0)


if __name__ == '__main__':
    unittest.main()