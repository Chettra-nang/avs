#!/usr/bin/env python3
"""
Demonstration of the extended ScenarioRegistry with ambulance scenario support.

This script shows how to use the ScenarioRegistry to work with ambulance scenarios,
including listing, retrieving, validating, and customizing ambulance configurations.
"""

import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from highway_datacollection.scenarios.registry import ScenarioRegistry


def main():
    """Demonstrate ambulance scenario registry functionality."""
    
    print("Ambulance Scenario Registry Demonstration")
    print("=" * 50)
    
    # Initialize the registry
    registry = ScenarioRegistry()
    
    # 1. Show all available scenarios
    print("\n1. Available Scenarios:")
    print("-" * 25)
    all_scenarios = registry.list_scenarios()
    regular_scenarios = registry.list_regular_scenarios()
    ambulance_scenarios = registry.list_ambulance_scenarios()
    
    print(f"Total scenarios: {len(all_scenarios)}")
    print(f"Regular scenarios: {len(regular_scenarios)}")
    print(f"Ambulance scenarios: {len(ambulance_scenarios)}")
    
    # 2. Show ambulance scenarios by category
    print("\n2. Ambulance Scenarios by Traffic Density:")
    print("-" * 40)
    for density in ["light", "moderate", "heavy"]:
        scenarios = registry.get_ambulance_scenarios_by_traffic_density(density)
        print(f"{density.capitalize()} traffic: {len(scenarios)} scenarios")
        if scenarios:
            print(f"  Examples: {scenarios[:2]}")
    
    # 3. Show ambulance scenarios by conditions
    print("\n3. Ambulance Scenarios by Highway Conditions:")
    print("-" * 45)
    conditions = ["normal", "construction", "rush_hour", "accident"]
    for condition in conditions:
        scenarios = registry.get_ambulance_scenarios_by_conditions(condition)
        if scenarios:
            print(f"{condition.capitalize()}: {scenarios}")
    
    # 4. Retrieve and examine a specific ambulance scenario
    print("\n4. Examining a Specific Ambulance Scenario:")
    print("-" * 42)
    if ambulance_scenarios:
        scenario_name = ambulance_scenarios[0]
        config = registry.get_ambulance_scenario_config(scenario_name)
        
        print(f"Scenario: {scenario_name}")
        print(f"Description: {config.get('description', 'N/A')}")
        print(f"Traffic density: {config.get('traffic_density', 'N/A')}")
        print(f"Vehicle count: {config.get('vehicles_count', 'N/A')}")
        print(f"Duration: {config.get('duration', 'N/A')} seconds")
        print(f"Controlled vehicles: {config.get('controlled_vehicles', 'N/A')}")
        
        # Show ambulance-specific configuration
        ambulance_config = config.get('_ambulance_config', {})
        print(f"Ambulance agent index: {ambulance_config.get('ambulance_agent_index', 'N/A')}")
        print(f"Emergency priority: {ambulance_config.get('emergency_priority', 'N/A')}")
    
    # 5. Validate ambulance scenario
    print("\n5. Ambulance Scenario Validation:")
    print("-" * 35)
    if ambulance_scenarios:
        config = registry.get_ambulance_scenario_config(ambulance_scenarios[0])
        is_valid = registry.validate_scenario(config)
        print(f"Scenario '{ambulance_scenarios[0]}' is valid: {is_valid}")
        
        # Test validation with invalid config
        invalid_config = config.copy()
        invalid_config['controlled_vehicles'] = 2
        is_invalid = registry.validate_scenario(invalid_config)
        print(f"Modified scenario (2 controlled vehicles) is valid: {is_invalid}")
    
    # 6. Customize ambulance scenario
    print("\n6. Ambulance Scenario Customization:")
    print("-" * 37)
    if ambulance_scenarios:
        base_scenario = ambulance_scenarios[0]
        try:
            custom_config = registry.customize_ambulance_scenario(
                base_scenario,
                duration=60,
                vehicles_count=30,
                description="Custom ambulance scenario with extended duration"
            )
            print(f"✓ Successfully customized '{base_scenario}'")
            print(f"  Original duration: {config.get('duration')} seconds")
            print(f"  Custom duration: {custom_config.get('duration')} seconds")
            print(f"  Original vehicle count: {config.get('vehicles_count')}")
            print(f"  Custom vehicle count: {custom_config.get('vehicles_count')}")
            print(f"  Maintained controlled vehicles: {custom_config.get('controlled_vehicles')}")
        except Exception as e:
            print(f"✗ Customization failed: {e}")
    
    # 7. Show ambulance constraints
    print("\n7. Ambulance Scenario Constraints:")
    print("-" * 35)
    constraints = registry.get_ambulance_constraints()
    print("Key constraints for ambulance scenarios:")
    for param, constraint in constraints.items():
        if 'required' in constraint:
            print(f"  {param}: required = {constraint['required']}")
        if 'min' in constraint and 'max' in constraint:
            print(f"  {param}: {constraint['min']} - {constraint['max']}")
        elif 'allowed' in constraint:
            print(f"  {param}: allowed values = {constraint['allowed'][:3]}...")
    
    # 8. Show import status
    print("\n8. Import Status:")
    print("-" * 17)
    status = registry.get_ambulance_import_status()
    print(f"Ambulance scenarios available: {status['available']}")
    print(f"Scenarios loaded: {status['scenario_count']}")
    if status.get('import_error'):
        print(f"Import error: {status['import_error']}")
    
    print("\n" + "=" * 50)
    print("Demonstration completed successfully!")


if __name__ == "__main__":
    main()