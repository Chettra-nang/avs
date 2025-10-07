#!/usr/bin/env python3
"""
Check current speed status of all 30 ambulance scenarios
"""

import sys
import re
sys.path.append('.')

def check_current_speeds():
    """Check all current speed limits in the scenarios file."""
    
    file_path = "D:\\Research_ITC\\avs_folder\\avs\\collecting_ambulance_data\\scenarios\\ambulance_scenarios.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all speed_limit values with context
    pattern = r'"([^"]*(?:highway|roundabout|corner|intersection|merge|urban|transition|night)[^"]*)":[^}]*"speed_limit":\s*(\d+)'
    
    matches = re.findall(pattern, content, re.IGNORECASE)
    
    print("🔍 CURRENT SPEED STATUS")
    print("=" * 50)
    
    if matches:
        for scenario_name, speed in matches:
            print(f"{scenario_name}: {speed} km/h")
        
        speeds = [int(speed) for _, speed in matches]
        print(f"\n📊 STATISTICS:")
        print(f"  Scenarios found: {len(speeds)}")
        print(f"  Speed range: {min(speeds)} - {max(speeds)} km/h")
        print(f"  Average speed: {sum(speeds)/len(speeds):.1f} km/h")
        
        # Categorize speeds
        highway_speeds = len([s for s in speeds if s >= 80])
        arterial_speeds = len([s for s in speeds if 40 <= s < 80])
        city_speeds = len([s for s in speeds if s < 40])
        
        print(f"\n🎯 SPEED CATEGORIES:")
        print(f"  🚀 Highway (80+ km/h): {highway_speeds}")
        print(f"  🚗 Arterial (40-79 km/h): {arterial_speeds}")
        print(f"  🐌 City (<40 km/h): {city_speeds}")
        
        if city_speeds > 0:
            slow_scenarios = [(name, speed) for name, speed in matches if int(speed) < 40]
            print(f"\n⚠️  SLOW SCENARIOS REMAINING:")
            for name, speed in slow_scenarios:
                print(f"    {name}: {speed} km/h")
    
    # Try alternative pattern to catch all speed_limit entries
    all_speeds = re.findall(r'"speed_limit":\s*(\d+)', content)
    all_speeds = [int(s) for s in all_speeds]
    
    print(f"\n📋 ALL SPEED_LIMIT ENTRIES:")
    print(f"  Total entries: {len(all_speeds)}")
    print(f"  Speeds: {all_speeds}")
    
    return matches

if __name__ == "__main__":
    matches = check_current_speeds()