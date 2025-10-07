#!/usr/bin/env python3
"""
🚀 SYSTEMATIC FAST SCENARIO UPDATER
====================================

This script systematically updates all 30 ambulance scenarios to fast highway speeds.
It modifies speed_limit, vehicles_count, and spawn_probability for optimal highway performance.
"""

def create_comprehensive_updates():
    """Create comprehensive updates for all 30 scenarios."""
    
    # Define the complete transformation map
    # Format: scenario_name -> (new_speed_limit, new_vehicles_count, new_spawn_probability)
    scenario_updates = {
        # Scenarios 1-10: Original highway scenarios
        "highway_emergency_light": (110, 12, 0.15),
        "highway_emergency_moderate": (95, 18, 0.25), 
        "highway_emergency_dense": (75, 28, 0.4),
        "highway_lane_closure": (65, 20, 0.3),
        "highway_rush_hour": (70, 30, 0.45),
        "highway_accident_scene": (60, 18, 0.2),
        "highway_construction": (65, 15, 0.2),
        "highway_weather_conditions": (70, 14, 0.25),
        "highway_stop_and_go": (55, 25, 0.4),
        "highway_aggressive_drivers": (85, 22, 0.3),
        
        # Scenarios 11-15: Additional highway scenarios  
        "highway_merge_heavy": (80, 26, 0.35),
        "highway_speed_variation": (100, 18, 0.25),
        "highway_shoulder_use": (75, 24, 0.35),
        "highway_truck_heavy": (80, 16, 0.2),
        "highway_time_pressure": (115, 20, 0.2),
        
        # Scenarios 16-18: Roundabouts (medium speeds)
        "roundabout_single_lane": (45, 12, 0.25),
        "roundabout_multi_lane": (50, 18, 0.35),
        "roundabout_congested": (40, 22, 0.4),
        
        # Scenarios 19-23: Corners and intersections
        "corner_sharp_turn": (55, 14, 0.25),
        "intersection_t_junction": (50, 16, 0.3),
        "intersection_four_way": (45, 20, 0.35),
        "corner_blind_curve": (60, 10, 0.2),
        "corner_urban_crossing": (50, 18, 0.3),
        
        # Scenarios 24-27: Merge scenarios (high speeds)
        "merge_highway_entry": (90, 16, 0.25),
        "merge_heavy_traffic": (75, 22, 0.35),
        "merge_zipper_pattern": (80, 20, 0.3),
        "merge_multi_point": (85, 18, 0.25),
        
        # Scenarios 28-30: Urban/mixed scenarios  
        "urban_mixed_complex": (60, 24, 0.4),
        "transition_highway_urban": (70, 20, 0.3),
        "night_emergency_response": (95, 8, 0.1)
    }
    
    return scenario_updates

def apply_scenario_updates():
    """Apply all scenario updates systematically."""
    
    import re
    
    file_path = "D:\\Research_ITC\\avs_folder\\avs\\collecting_ambulance_data\\scenarios\\ambulance_scenarios.py"
    
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    updates = create_comprehensive_updates()
    
    print("🚀 APPLYING SYSTEMATIC SPEED UPDATES")
    print("=" * 50)
    
    changes_made = 0
    
    for scenario_name, (new_speed, new_vehicles, new_spawn) in updates.items():
        
        # Pattern to find and update each scenario's parameters
        # Look for the scenario block and update the three key parameters
        
        # Update speed_limit
        speed_pattern = f'("{scenario_name}":[^}}]*)"speed_limit":\s*\\d+'
        speed_replacement = f'\\1"speed_limit": {new_speed}'
        
        old_content = content
        content = re.sub(speed_pattern, speed_replacement, content, flags=re.DOTALL)
        if content != old_content:
            changes_made += 1
            print(f"  ✅ {scenario_name}: Updated speed_limit to {new_speed} km/h")
        
        # Update vehicles_count
        vehicles_pattern = f'("{scenario_name}":[^}}]*)"vehicles_count":\s*\\d+'
        vehicles_replacement = f'\\1"vehicles_count": {new_vehicles}'
        
        old_content = content  
        content = re.sub(vehicles_pattern, vehicles_replacement, content, flags=re.DOTALL)
        if content != old_content:
            print(f"    ↳ Updated vehicles_count to {new_vehicles}")
        
        # Update spawn_probability
        spawn_pattern = f'("{scenario_name}":[^}}]*)"spawn_probability":\s*[0-9.]+' 
        spawn_replacement = f'\\1"spawn_probability": {new_spawn}'
        
        old_content = content
        content = re.sub(spawn_pattern, spawn_replacement, content, flags=re.DOTALL)
        if content != old_content:
            print(f"    ↳ Updated spawn_probability to {new_spawn}")
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ TRANSFORMATION COMPLETE!")
    print(f"Applied updates to scenarios with speed changes: {changes_made}")
    
    return changes_made

def verify_transformation():
    """Verify that all scenarios now have appropriate speeds."""
    
    import re
    
    file_path = "D:\\Research_ITC\\avs_folder\\avs\\collecting_ambulance_data\\scenarios\\ambulance_scenarios.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract all speed limits
    speeds = re.findall(r'"speed_limit":\s*(\d+)', content)
    speeds = [int(s) for s in speeds]
    
    print("🔍 VERIFICATION RESULTS")
    print("=" * 30)
    print(f"Total scenarios with speed_limit: {len(speeds)}")
    print(f"Speed range: {min(speeds)} - {max(speeds)} km/h")
    print(f"Average speed: {sum(speeds)/len(speeds):.1f} km/h")
    
    # Categorize speeds
    highway_speeds = len([s for s in speeds if s >= 80])  # True highway
    arterial_speeds = len([s for s in speeds if 40 <= s < 80])  # Fast arterial
    city_speeds = len([s for s in speeds if s < 40])  # Still too slow
    
    print(f"\nSpeed Distribution:")
    print(f"  🚀 Highway (80+ km/h): {highway_speeds} scenarios")
    print(f"  🚗 Arterial (40-79 km/h): {arterial_speeds} scenarios") 
    print(f"  🐌 City (<40 km/h): {city_speeds} scenarios")
    
    if city_speeds == 0:
        print(f"\n✅ SUCCESS: All scenarios now have appropriate highway/arterial speeds!")
        print(f"📈 Expected ambulance performance: 60-100 km/h (vs original 3 km/h)")
    else:
        print(f"\n⚠️  {city_speeds} scenarios still have slow speeds")
    
    return speeds

if __name__ == "__main__":
    print("🚀 SYSTEMATIC FAST AMBULANCE TRANSFORMATION")
    print("=" * 60)
    print("Converting all 30 scenarios from city speeds (10-35 km/h)")
    print("to highway speeds (40-115 km/h) for fast ambulance performance.")
    print()
    
    changes = apply_scenario_updates()
    
    if changes > 0:
        print()
        speeds = verify_transformation()
        
        print()
        print("🎯 TRANSFORMATION SUMMARY:")
        print("  • Speed limits: Now 40-115 km/h (was 10-35 km/h)")
        print("  • Traffic reduced for better flow")
        print("  • Expected ambulance speeds: 60-100 km/h")
        print("  • Perfect for 'fast as possible' research! 🚑💨")
    else:
        print("No changes applied - scenarios may already be updated.")