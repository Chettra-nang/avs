#!/usr/bin/env python3
"""
Fix ambulance scenarios by adding reward_speed_range parameters
that match the speed limits for fast emergency responses.
"""

import re
import os

def fix_ambulance_scenarios():
    """Add missing reward_speed_range parameters to match speed_limit values"""
    
    scenario_file = r"d:\Research_ITC\avs_folder\avs\collecting_ambulance_data\scenarios\ambulance_scenarios.py"
    
    # Read the file
    with open(scenario_file, 'r') as f:
        content = f.read()
    
    # Pattern to find scenarios with speed_limit but no reward_speed_range
    # Look for speed_limit followed by spawn_probability (no reward_speed_range in between)
    pattern = r'("speed_limit": (\d+),)\s+("spawn_probability":)'
    
    def replace_func(match):
        speed_limit_line = match.group(1)
        speed_limit_value = int(match.group(2))
        spawn_probability_line = match.group(3)
        
        # Calculate appropriate reward speed range (10-20 km/h below speed limit to speed limit)
        min_reward_speed = max(20, speed_limit_value - 20)  # Don't go below 20 km/h
        max_reward_speed = speed_limit_value
        
        # Add reward_speed_range line
        reward_line = f'"reward_speed_range": [{min_reward_speed}, {max_reward_speed}],'
        
        return f"{speed_limit_line}\n            {reward_line}\n            {spawn_probability_line}"
    
    # Apply the replacement
    new_content = re.sub(pattern, replace_func, content)
    
    # Count how many replacements were made
    matches_found = len(re.findall(pattern, content))
    
    if matches_found > 0:
        # Write back to file
        with open(scenario_file, 'w') as f:
            f.write(new_content)
        
        print(f"✅ Fixed {matches_found} ambulance scenarios by adding reward_speed_range parameters")
        print("📊 Speed range mappings:")
        
        # Show what speed ranges were added
        speed_pattern = r'"speed_limit": (\d+),\s+"reward_speed_range": \[(\d+), (\d+)\],'
        speed_matches = re.findall(speed_pattern, new_content)
        
        for speed_limit, min_speed, max_speed in speed_matches[:10]:  # Show first 10
            print(f"   Speed limit {speed_limit} km/h → Reward range [{min_speed}, {max_speed}] km/h")
        
        if len(speed_matches) > 10:
            print(f"   ... and {len(speed_matches) - 10} more scenarios")
    else:
        print("ℹ️ All scenarios already have reward_speed_range parameters")
    
    return matches_found

if __name__ == "__main__":
    print("🚑 Fixing ambulance scenario reward speed ranges...")
    fixes_applied = fix_ambulance_scenarios()
    
    if fixes_applied > 0:
        print(f"\n🎯 Ready to test! {fixes_applied} scenarios now have fast reward speed ranges")
        print("💡 Next step: Run data collection to verify fast ambulance speeds")
    else:
        print("\n✨ No fixes needed - scenarios are already configured correctly")