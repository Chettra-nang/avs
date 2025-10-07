#!/usr/bin/env python3
"""
🚀 APPLY REMAINING FAST SCENARIO UPDATES
=========================================

This script applies the remaining speed updates to scenarios 19-30
to complete the transformation from slow city speeds to fast highway speeds.
"""

import re

def apply_all_remaining_updates():
    """Apply all remaining scenario speed updates."""
    
    file_path = "D:\\Research_ITC\\avs_folder\\avs\\collecting_ambulance_data\\scenarios\\ambulance_scenarios.py"
    
    # Read the current file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("🔧 Applying remaining speed updates to scenarios 19-30...")
    
    # Define all remaining updates as (old_pattern, new_replacement) tuples
    updates = [
        # Scenario 19: corner_sharp_turn
        (r'"speed_limit":\s*20,\s*\n(\s*)"spawn_probability":\s*0\.4,',
         r'"speed_limit": 55,\n\1"spawn_probability": 0.25,'),
        
        # Scenario 20: intersection_t_junction  
        (r'"speed_limit":\s*18,\s*\n(\s*)"spawn_probability":\s*0\.5,',
         r'"speed_limit": 50,\n\1"spawn_probability": 0.3,'),
        
        # Scenario 21: intersection_four_way
        (r'"speed_limit":\s*15,\s*\n(\s*)"spawn_probability":\s*0\.6,',
         r'"speed_limit": 45,\n\1"spawn_probability": 0.35,'),
        
        # Scenario 22: corner_blind_curve
        (r'"speed_limit":\s*22,\s*\n(\s*)"spawn_probability":\s*0\.4,',
         r'"speed_limit": 60,\n\1"spawn_probability": 0.2,'),
        
        # Scenario 23: corner_urban_crossing
        (r'"speed_limit":\s*18,\s*\n(\s*)"spawn_probability":\s*0\.6,',
         r'"speed_limit": 50,\n\1"spawn_probability": 0.3,'),
        
        # Scenario 24: merge_highway_entry
        (r'"speed_limit":\s*25,\s*\n(\s*)"spawn_probability":\s*0\.5,',
         r'"speed_limit": 90,\n\1"spawn_probability": 0.25,'),
        
        # Scenario 25: merge_heavy_traffic
        (r'"speed_limit":\s*22,\s*\n(\s*)"spawn_probability":\s*0\.7,',
         r'"speed_limit": 75,\n\1"spawn_probability": 0.35,'),
        
        # Scenario 26: merge_zipper_pattern
        (r'"speed_limit":\s*20,\s*\n(\s*)"spawn_probability":\s*0\.6,',
         r'"speed_limit": 80,\n\1"spawn_probability": 0.3,'),
        
        # Scenario 27: merge_multi_point
        (r'"speed_limit":\s*24,\s*\n(\s*)"spawn_probability":\s*0\.5,',
         r'"speed_limit": 85,\n\1"spawn_probability": 0.25,'),
        
        # Scenario 28: urban_mixed_complex
        (r'"speed_limit":\s*20,\s*\n(\s*)"spawn_probability":\s*0\.7,',
         r'"speed_limit": 60,\n\1"spawn_probability": 0.4,'),
        
        # Scenario 29: transition_highway_urban
        (r'"speed_limit":\s*23,\s*\n(\s*)"spawn_probability":\s*0\.6,',
         r'"speed_limit": 70,\n\1"spawn_probability": 0.3,'),
        
        # Scenario 30: night_emergency_response
        (r'"speed_limit":\s*25,\s*\n(\s*)"spawn_probability":\s*0\.3,',
         r'"speed_limit": 95,\n\1"spawn_probability": 0.1,')
    ]
    
    # Also update vehicle counts for remaining scenarios
    vehicle_updates = [
        # Reduce vehicle counts for better flow
        (r'"vehicles_count":\s*20,\s*\n(\s*)"initial_lane_id":\s*None,\s*\n(\s*)"duration":\s*45,\s*\n(\s*)"highway_conditions":\s*"sharp_turn"',
         r'"vehicles_count": 14,\n\1"initial_lane_id": None,\n\2"duration": 45,\n\3"highway_conditions": "sharp_turn"'),
        
        (r'"vehicles_count":\s*22,\s*\n(\s*)"initial_lane_id":\s*None,\s*\n(\s*)"duration":\s*50,\s*\n(\s*)"highway_conditions":\s*"t_junction"',
         r'"vehicles_count": 16,\n\1"initial_lane_id": None,\n\2"duration": 50,\n\3"highway_conditions": "t_junction"'),
        
        (r'"vehicles_count":\s*30,\s*\n(\s*)"initial_lane_id":\s*None,\s*\n(\s*)"duration":\s*45,\s*\n(\s*)"highway_conditions":\s*"four_way_intersection"',
         r'"vehicles_count": 20,\n\1"initial_lane_id": None,\n\2"duration": 45,\n\3"highway_conditions": "four_way_intersection"'),
        
        (r'"vehicles_count":\s*15,\s*\n(\s*)"initial_lane_id":\s*None,\s*\n(\s*)"duration":\s*40,\s*\n(\s*)"highway_conditions":\s*"blind_corner"',
         r'"vehicles_count": 10,\n\1"initial_lane_id": None,\n\2"duration": 40,\n\3"highway_conditions": "blind_corner"'),
        
        (r'"vehicles_count":\s*25,\s*\n(\s*)"initial_lane_id":\s*None,\s*\n(\s*)"duration":\s*45,\s*\n(\s*)"highway_conditions":\s*"urban_corner"',
         r'"vehicles_count": 18,\n\1"initial_lane_id": None,\n\2"duration": 45,\n\3"highway_conditions": "urban_corner"'),
        
        (r'"vehicles_count":\s*24,\s*\n(\s*)"initial_lane_id":\s*None,\s*\n(\s*)"duration":\s*45,\s*\n(\s*)"highway_conditions":\s*"on_ramp_merge"',
         r'"vehicles_count": 16,\n\1"initial_lane_id": None,\n\2"duration": 45,\n\3"highway_conditions": "on_ramp_merge"'),
        
        (r'"vehicles_count":\s*32,\s*\n(\s*)"initial_lane_id":\s*None,\s*\n(\s*)"duration":\s*50,\s*\n(\s*)"highway_conditions":\s*"merge_congested"',
         r'"vehicles_count": 22,\n\1"initial_lane_id": None,\n\2"duration": 50,\n\3"highway_conditions": "merge_congested"'),
        
        (r'"vehicles_count":\s*28,\s*\n(\s*)"initial_lane_id":\s*None,\s*\n(\s*)"duration":\s*45,\s*\n(\s*)"highway_conditions":\s*"zipper_merge"',
         r'"vehicles_count": 20,\n\1"initial_lane_id": None,\n\2"duration": 45,\n\3"highway_conditions": "zipper_merge"'),
        
        (r'"vehicles_count":\s*26,\s*\n(\s*)"initial_lane_id":\s*None,\s*\n(\s*)"duration":\s*55,\s*\n(\s*)"highway_conditions":\s*"multi_merge"',
         r'"vehicles_count": 18,\n\1"initial_lane_id": None,\n\2"duration": 55,\n\3"highway_conditions": "multi_merge"'),
        
        (r'"vehicles_count":\s*34,\s*\n(\s*)"initial_lane_id":\s*None,\s*\n(\s*)"duration":\s*55,\s*\n(\s*)"highway_conditions":\s*"urban_mixed"',
         r'"vehicles_count": 24,\n\1"initial_lane_id": None,\n\2"duration": 55,\n\3"highway_conditions": "urban_mixed"'),
        
        (r'"vehicles_count":\s*27,\s*\n(\s*)"initial_lane_id":\s*None,\s*\n(\s*)"duration":\s*50,\s*\n(\s*)"highway_conditions":\s*"highway_urban_transition"',
         r'"vehicles_count": 20,\n\1"initial_lane_id": None,\n\2"duration": 50,\n\3"highway_conditions": "highway_urban_transition"'),
        
        (r'"vehicles_count":\s*18,\s*\n(\s*)"initial_lane_id":\s*None,\s*\n(\s*)"duration":\s*40,\s*\n(\s*)"highway_conditions":\s*"night_emergency"',
         r'"vehicles_count": 8,\n\1"initial_lane_id": None,\n\2"duration": 40,\n\3"highway_conditions": "night_emergency"')
    ]
    
    # Apply all updates
    changes_made = 0
    
    print("  📊 Updating speed limits...")
    for old_pattern, new_replacement in updates:
        old_content = content
        content = re.sub(old_pattern, new_replacement, content)
        if content != old_content:
            changes_made += 1
            print(f"    ✅ Updated speed limit pattern")
    
    print("  🚗 Updating vehicle counts...")
    for old_pattern, new_replacement in vehicle_updates:
        old_content = content
        content = re.sub(old_pattern, new_replacement, content)
        if content != old_content:
            changes_made += 1
            print(f"    ✅ Updated vehicle count pattern")
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ Applied {changes_made} updates to scenarios 19-30")
    return changes_made

def verify_speed_updates():
    """Verify that all speed updates were applied correctly."""
    
    file_path = "D:\\Research_ITC\\avs_folder\\avs\\collecting_ambulance_data\\scenarios\\ambulance_scenarios.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all speed_limit values
    speed_limits = re.findall(r'"speed_limit":\s*(\d+)', content)
    speed_limits = [int(s) for s in speed_limits]
    
    print("🔍 SPEED VERIFICATION:")
    print("=" * 40)
    
    print(f"Speed limits found: {len(speed_limits)}")
    print(f"Speed range: {min(speed_limits)} - {max(speed_limits)} km/h")
    print(f"Average speed: {sum(speed_limits)/len(speed_limits):.1f} km/h")
    
    # Check for any remaining slow speeds
    slow_speeds = [s for s in speed_limits if s < 40]
    if slow_speeds:
        print(f"⚠️  Still some slow speeds: {slow_speeds}")
    else:
        print("✅ All speeds are now 40+ km/h (highway appropriate)")
    
    # Count by speed range
    fast_speeds = len([s for s in speed_limits if s >= 80])  # Highway speeds
    medium_speeds = len([s for s in speed_limits if 40 <= s < 80])  # Arterial speeds
    slow_speeds = len([s for s in speed_limits if s < 40])  # City speeds
    
    print(f"\nSpeed distribution:")
    print(f"  🚀 Highway speeds (80+ km/h): {fast_speeds} scenarios")
    print(f"  🚗 Arterial speeds (40-79 km/h): {medium_speeds} scenarios")
    print(f"  🐌 City speeds (<40 km/h): {slow_speeds} scenarios")
    
    return speed_limits

if __name__ == "__main__":
    print("🚀 COMPLETING FAST SCENARIO TRANSFORMATION")
    print("=" * 60)
    
    changes = apply_all_remaining_updates()
    
    if changes > 0:
        print("\n📊 Verifying all updates...")
        speeds = verify_speed_updates()
        
        print("\n🎯 TRANSFORMATION COMPLETE!")
        print("Your 30 ambulance scenarios are now configured for:")
        print("  • Highway speeds: 40-115 km/h (vs original 10-35 km/h)")
        print("  • Reduced traffic congestion")  
        print("  • Expected ambulance performance: 60-100 km/h")
        print("  • Perfect for 'fast as possible' research! 🚑💨")
    else:
        print("⚠️  No additional changes needed - scenarios may already be updated")