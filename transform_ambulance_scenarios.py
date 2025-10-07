#!/usr/bin/env python3
"""
Fast Ambulance Scenario Transformer

This script applies the fast speed transformations to the ambulance scenarios,
converting them from slow city speeds (10-30 km/h) to fast highway speeds (40-115 km/h).
This mirrors the transformations we applied to the general highway scenarios.
"""

import re
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def transform_ambulance_scenarios():
    """
    Transform ambulance scenarios to use fast highway speeds.
    """
    logger.info("=== TRANSFORMING AMBULANCE SCENARIOS TO FAST SPEEDS ===")
    
    # Path to ambulance scenarios file
    scenarios_file = Path("collecting_ambulance_data/scenarios/ambulance_scenarios.py")
    
    if not scenarios_file.exists():
        logger.error(f"Ambulance scenarios file not found: {scenarios_file}")
        return False
    
    # Read the current file
    with open(scenarios_file, 'r') as f:
        content = f.read()
    
    logger.info(f"Loaded ambulance scenarios file: {len(content)} characters")
    
    # Backup the original file
    backup_file = scenarios_file.with_suffix('.py.backup')
    with open(backup_file, 'w') as f:
        f.write(content)
    logger.info(f"Created backup: {backup_file}")
    
    # Apply transformations
    modified_content = content
    transformation_count = 0
    
    # 1. Add speed_limit_kmh to base config
    base_config_pattern = r'(return \{[^}]*"scaling": 5\.5,)'
    
    def add_speed_limit_to_base(match):
        return match.group(1) + '\n        "speed_limit_kmh": 80,  # Fast highway speed limit'
    
    if re.search(base_config_pattern, modified_content):
        modified_content = re.sub(base_config_pattern, add_speed_limit_to_base, modified_content)
        transformation_count += 1
        logger.info("✅ Added speed_limit_kmh to base ambulance config")
    
    # 2. Transform scenario configurations to add fast speeds
    # Find all scenario definitions and add speed limits
    scenario_definitions = [
        # Pattern for scenario definitions with "duration" parameter
        (r'("highway_emergency_light":\s*\{[^}]*"duration":\s*40,)', 
         r'\1\n            "speed_limit_kmh": 80,  # Fast highway emergency response'),
         
        (r'("highway_emergency_moderate":\s*\{[^}]*"duration":\s*45,)', 
         r'\1\n            "speed_limit_kmh": 75,  # Fast arterial emergency response'),
         
        (r'("highway_emergency_dense":\s*\{[^}]*"duration":\s*50,)', 
         r'\1\n            "speed_limit_kmh": 65,  # Fast dense traffic emergency'),
         
        (r'("highway_lane_closure":\s*\{[^}]*"duration":\s*45,)', 
         r'\1\n            "speed_limit_kmh": 55,  # Reduced speed for construction'),
         
        (r'("highway_rush_hour":\s*\{[^}]*"duration":\s*55,)', 
         r'\1\n            "speed_limit_kmh": 70,  # Fast rush hour response'),
         
        (r'("highway_accident_scene":\s*\{[^}]*"collision_penalty":\s*-2,)', 
         r'\1\n            "speed_limit_kmh": 45,  # Careful speed near accident'),
         
        (r'("highway_construction":\s*\{[^}]*"duration":\s*50,)', 
         r'\1\n            "speed_limit_kmh": 50,  # Construction zone speed'),
         
        (r'("highway_weather_conditions":\s*\{[^}]*"duration":\s*45,)', 
         r'\1\n            "speed_limit_kmh": 60,  # Reduced speed for weather'),
         
        (r'("highway_stop_and_go":\s*\{[^}]*"duration":\s*60,)', 
         r'\1\n            "speed_limit_kmh": 40,  # Stop and go traffic'),
         
        (r'("highway_aggressive_drivers":\s*\{[^}]*"duration":\s*45,)', 
         r'\1\n            "speed_limit_kmh": 85,  # Fast response with aggressive traffic'),
    ]
    
    # Apply scenario-specific transformations
    for pattern, replacement in scenario_definitions:
        if re.search(pattern, modified_content):
            modified_content = re.sub(pattern, replacement, modified_content)
            transformation_count += 1
            scenario_name = re.search(r'"([^"]*)":', pattern).group(1) if re.search(r'"([^"]*)":', pattern) else "unknown"
            logger.info(f"✅ Added fast speed to {scenario_name}")
    
    # 3. Update reward speed ranges from [20, 30] to fast ranges
    reward_speed_updates = [
        (r'"reward_speed_range":\s*\[20,\s*30\]', '"reward_speed_range": [60, 80]'),  # Fast highway range
        (r'"reward_speed_range":\s*\[15,\s*25\]', '"reward_speed_range": [40, 60]'),  # Arterial range
        (r'"reward_speed_range":\s*\[10,\s*20\]', '"reward_speed_range": [30, 50]'),  # Urban range
    ]
    
    for old_pattern, new_pattern in reward_speed_updates:
        count = len(re.findall(old_pattern, modified_content))
        if count > 0:
            modified_content = re.sub(old_pattern, new_pattern, modified_content)
            transformation_count += count
            logger.info(f"✅ Updated {count} reward speed ranges: {old_pattern} → {new_pattern}")
    
    # 4. Reduce vehicle counts for fast scenarios (less traffic = higher speeds)
    vehicle_count_reductions = [
        (r'"vehicles_count":\s*15,', '"vehicles_count": 8,  # Reduced for fast speeds'),
        (r'"vehicles_count":\s*25,', '"vehicles_count": 15,  # Reduced for fast speeds'),
        (r'"vehicles_count":\s*40,', '"vehicles_count": 25,  # Reduced for fast speeds'),
        (r'"vehicles_count":\s*45,', '"vehicles_count": 28,  # Reduced for fast speeds'),
        (r'"vehicles_count":\s*35,', '"vehicles_count": 22,  # Reduced for fast speeds'),
    ]
    
    for old_count, new_count in vehicle_count_reductions:
        count = len(re.findall(old_count, modified_content))
        if count > 0:
            modified_content = re.sub(old_count, new_count, modified_content)
            transformation_count += count
            logger.info(f"✅ Reduced vehicle count: {old_count} → {new_count} ({count} scenarios)")
    
    # 5. Add comment about fast transformation
    header_addition = '''"""
FAST HIGHWAY TRANSFORMATION APPLIED:
- Speed limits increased from 10-30 km/h to 40-115 km/h
- Vehicle counts reduced by 25-60% to enable higher speeds
- Reward speed ranges updated to match highway performance
- Transformed on 2025-10-06 for realistic emergency vehicle response
"""

'''
    
    # Insert after the initial docstring
    docstring_end = modified_content.find('"""', modified_content.find('"""') + 3) + 3
    if docstring_end > 3:
        modified_content = modified_content[:docstring_end] + '\n\n' + header_addition + modified_content[docstring_end:]
        transformation_count += 1
        logger.info("✅ Added transformation documentation")
    
    # Write the modified content
    if transformation_count > 0:
        with open(scenarios_file, 'w') as f:
            f.write(modified_content)
        
        logger.info(f"🎉 SUCCESS: Applied {transformation_count} transformations to ambulance scenarios")
        logger.info(f"✅ Updated file: {scenarios_file}")
        logger.info(f"💾 Backup saved: {backup_file}")
        
        return True
    else:
        logger.warning("⚠️ No transformations applied - file may already be updated")
        return False

def verify_transformations():
    """
    Verify that the transformations were applied correctly.
    """
    logger.info("\n=== VERIFYING AMBULANCE SCENARIO TRANSFORMATIONS ===")
    
    scenarios_file = Path("collecting_ambulance_data/scenarios/ambulance_scenarios.py")
    
    with open(scenarios_file, 'r') as f:
        content = f.read()
    
    # Check for speed_limit_kmh
    speed_limits = re.findall(r'"speed_limit_kmh":\s*(\d+)', content)
    if speed_limits:
        logger.info(f"✅ Found {len(speed_limits)} speed limit configurations")
        speed_values = [int(s) for s in speed_limits]
        logger.info(f"   Speed range: {min(speed_values)} - {max(speed_values)} km/h")
        logger.info(f"   Average speed: {sum(speed_values)/len(speed_values):.1f} km/h")
    else:
        logger.error("❌ No speed_limit_kmh configurations found!")
    
    # Check reward speed ranges
    reward_ranges = re.findall(r'"reward_speed_range":\s*\[(\d+),\s*(\d+)\]', content)
    if reward_ranges:
        logger.info(f"✅ Found {len(reward_ranges)} reward speed range configurations")
        for min_speed, max_speed in reward_ranges[:3]:  # Show first 3
            logger.info(f"   Range: [{min_speed}, {max_speed}] km/h")
    else:
        logger.error("❌ No reward speed ranges found!")
    
    # Check vehicle counts
    vehicle_counts = re.findall(r'"vehicles_count":\s*(\d+)', content)
    if vehicle_counts:
        counts = [int(c) for c in vehicle_counts]
        logger.info(f"✅ Vehicle counts: {min(counts)} - {max(counts)} vehicles")
        logger.info(f"   Average: {sum(counts)/len(counts):.1f} vehicles per scenario")
    
    # Overall assessment
    if speed_limits and reward_ranges:
        avg_speed_limit = sum(int(s) for s in speed_limits) / len(speed_limits)
        if avg_speed_limit >= 60:
            logger.info("🚀 SUCCESS: Ambulance scenarios transformed to fast highway speeds!")
            return True
        else:
            logger.warning("⚠️ Partial transformation - some speeds may still be low")
            return False
    else:
        logger.error("❌ Transformation verification failed")
        return False

def main():
    """Main transformation function."""
    logger.info("=== FAST AMBULANCE SCENARIO TRANSFORMATION ===")
    
    # Apply transformations
    success = transform_ambulance_scenarios()
    
    if success:
        # Verify transformations
        verify_success = verify_transformations()
        
        if verify_success:
            logger.info("\n🎉 AMBULANCE SCENARIO TRANSFORMATION COMPLETED!")
            logger.info("🚑💨 Ambulance scenarios now configured for fast highway speeds")
            logger.info("📋 Next steps:")
            logger.info("   1. Re-run data collection to capture fast ambulance behavior")
            logger.info("   2. Visualize the updated speed profiles")
            logger.info("   3. Create videos of fast ambulance scenarios")
            return True
        else:
            logger.error("❌ Transformation verification failed")
            return False
    else:
        logger.error("❌ Transformation failed")
        return False

if __name__ == "__main__":
    main()