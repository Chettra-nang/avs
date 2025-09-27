#!/usr/bin/env python3
"""
Test script for ambulance demonstration.

This script tests the ambulance_demo.py functionality to ensure it works correctly.
"""

import sys
import os
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the demo functions
from collecting_ambulance_data.examples.ambulance_demo import (
    demonstrate_ambulance_scenarios,
    demonstrate_multi_modal_support,
    setup_logging
)


class TestAmbulanceDemo(unittest.TestCase):
    """Test cases for ambulance demonstration script."""
    
    def test_demonstrate_ambulance_scenarios(self):
        """Test that ambulance scenarios demonstration works."""
        # Capture output
        with patch('builtins.print') as mock_print:
            scenario_names = demonstrate_ambulance_scenarios()
        
        # Verify we got scenario names
        self.assertIsInstance(scenario_names, list)
        self.assertGreater(len(scenario_names), 0)
        
        # Verify print was called (output was generated)
        self.assertTrue(mock_print.called)
    
    def test_demonstrate_multi_modal_support(self):
        """Test that multi-modal support demonstration works."""
        with patch('builtins.print') as mock_print:
            observation_types = demonstrate_multi_modal_support()
        
        # Verify we got the expected observation types
        expected_types = ["Kinematics", "OccupancyGrid", "GrayscaleObservation"]
        self.assertEqual(observation_types, expected_types)
        
        # Verify print was called
        self.assertTrue(mock_print.called)
    
    def test_setup_logging(self):
        """Test that logging setup works."""
        logger = setup_logging()
        
        # Verify we got a logger
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, 'collecting_ambulance_data.examples.test_ambulance_demo')
    
    def test_ambulance_demo_imports(self):
        """Test that all required imports work."""
        try:
            from collecting_ambulance_data.collection.ambulance_collector import AmbulanceDataCollector
            from collecting_ambulance_data.scenarios.ambulance_scenarios import get_all_ambulance_scenarios
            
            # Verify classes can be instantiated
            scenarios = get_all_ambulance_scenarios()
            self.assertIsInstance(scenarios, dict)
            self.assertGreater(len(scenarios), 0)
            
        except ImportError as e:
            self.fail(f"Required imports failed: {e}")


def run_demo_test():
    """Run a quick test of the demo functionality."""
    print("🧪 Testing Ambulance Demo Functionality")
    print("=" * 40)
    
    try:
        # Test scenario demonstration
        print("1. Testing scenario demonstration...")
        scenario_names = demonstrate_ambulance_scenarios()
        print(f"   ✅ Found {len(scenario_names)} scenarios")
        
        # Test multi-modal demonstration
        print("\n2. Testing multi-modal demonstration...")
        observation_types = demonstrate_multi_modal_support()
        print(f"   ✅ Found {len(observation_types)} observation types")
        
        # Test logging setup
        print("\n3. Testing logging setup...")
        logger = setup_logging()
        logger.info("Test log message")
        print("   ✅ Logging configured successfully")
        
        print("\n✅ All demo tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Demo test failed: {e}")
        return False


if __name__ == "__main__":
    # Run quick test
    success = run_demo_test()
    
    if success:
        print("\n🎉 Demo is ready to run!")
        print("Execute: python collecting_ambulance_data/examples/ambulance_demo.py")
    else:
        print("\n⚠️  Demo has issues - check the errors above")
        sys.exit(1)