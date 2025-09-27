#!/usr/bin/env python3
"""
Test script for ambulance scenario validation.

This script runs comprehensive validation tests on all 15 ambulance scenarios
to verify proper configuration, environment creation, and multi-agent behavior.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from collecting_ambulance_data.validation import run_ambulance_scenario_validation
from collecting_ambulance_data.scenarios.ambulance_scenarios import get_scenario_names


def main():
    """Run ambulance scenario validation tests."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("AMBULANCE SCENARIO VALIDATION TEST")
    print("=" * 80)
    
    # Get all scenario names
    scenario_names = get_scenario_names()
    print(f"\nFound {len(scenario_names)} ambulance scenarios to validate:")
    for i, name in enumerate(scenario_names, 1):
        print(f"  {i:2d}. {name}")
    
    print(f"\nStarting comprehensive validation...")
    print("This will test:")
    print("  - Configuration validation for all scenarios")
    print("  - Environment creation for all observation types")
    print("  - Multi-agent behavior with ambulance ego vehicle")
    print("  - Proper ambulance agent configuration (index 0)")
    print("  - Multi-modal data collection compatibility")
    
    # Run validation with reduced test steps for faster execution
    try:
        results = run_ambulance_scenario_validation(
            scenarios=None,  # Test all scenarios
            test_steps=5,    # Reduced steps for faster testing
            n_agents=4       # Standard 4-agent setup
        )
        
        # Print results summary
        print("\n" + "=" * 80)
        print("VALIDATION RESULTS SUMMARY")
        print("=" * 80)
        
        total_scenarios = results.get('total_scenarios', 0)
        summary = results.get('summary', {})
        
        print(f"\nOverall Results:")
        print(f"  Total scenarios tested: {total_scenarios}")
        print(f"  Scenarios passed: {summary.get('scenarios_passed', 0)}")
        print(f"  Scenarios failed: {summary.get('scenarios_failed', 0)}")
        print(f"  Success rate: {results.get('success_rate', 0):.1f}%")
        
        print(f"\nDetailed Results:")
        print(f"  Configuration tests passed: {summary.get('configurations_passed', 0)}")
        print(f"  Configuration tests failed: {summary.get('configurations_failed', 0)}")
        print(f"  Environment tests passed: {summary.get('environments_passed', 0)}")
        print(f"  Environment tests failed: {summary.get('environments_failed', 0)}")
        print(f"  Behavior tests passed: {summary.get('behaviors_passed', 0)}")
        print(f"  Behavior tests failed: {summary.get('behaviors_failed', 0)}")
        
        # Print individual scenario results
        print(f"\nIndividual Scenario Results:")
        scenarios_results = results.get('scenarios', {})
        
        for scenario_name in scenario_names:
            if scenario_name in scenarios_results:
                scenario_result = scenarios_results[scenario_name]
                status = "✅ PASSED" if scenario_result.get('overall_valid', False) else "❌ FAILED"
                print(f"  {status} {scenario_name}")
                
                # Show configuration details
                config_result = scenario_result.get('configuration', {})
                if config_result:
                    config_status = "✅" if config_result.get('valid', False) else "❌"
                    print(f"    {config_status} Configuration: {config_result.get('tests_passed', 0)} tests passed")
                
                # Show environment results
                env_results = scenario_result.get('environments', {})
                for obs_type in ['Kinematics', 'OccupancyGrid', 'GrayscaleObservation']:
                    if obs_type in env_results:
                        env_result = env_results[obs_type]
                        env_status = "✅" if env_result.get('valid', False) else "❌"
                        print(f"    {env_status} Environment ({obs_type}): {env_result.get('tests_passed', 0)} tests passed")
                
                # Show behavior results
                behavior_results = scenario_result.get('behaviors', {})
                for obs_type in ['Kinematics', 'OccupancyGrid', 'GrayscaleObservation']:
                    if obs_type in behavior_results:
                        behavior_result = behavior_results[obs_type]
                        behavior_status = "✅" if behavior_result.get('valid', False) else "❌"
                        steps_completed = behavior_result.get('info', {}).get('steps_completed', 0)
                        print(f"    {behavior_status} Behavior ({obs_type}): {behavior_result.get('tests_passed', 0)} tests passed, {steps_completed} steps")
            else:
                print(f"  ❓ MISSING {scenario_name}")
        
        # Print errors if any
        errors = results.get('errors', [])
        if errors:
            print(f"\nErrors Encountered:")
            for error in errors:
                print(f"  ❌ {error}")
        
        # Print warnings if any
        warnings = results.get('warnings', [])
        if warnings:
            print(f"\nWarnings:")
            for warning in warnings:
                print(f"  ⚠️  {warning}")
        
        # Final status
        print("\n" + "=" * 80)
        if results.get('success_rate', 0) == 100.0:
            print("🎉 ALL AMBULANCE SCENARIOS PASSED VALIDATION!")
            print("✅ All 15 scenarios are properly configured and functional")
            print("✅ Ambulance ego vehicle configuration verified")
            print("✅ Multi-agent setup working correctly")
            print("✅ All observation types supported")
            return 0
        else:
            print("⚠️  SOME AMBULANCE SCENARIOS FAILED VALIDATION")
            print("Please review the failed scenarios and fix any issues.")
            return 1
            
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)