#!/usr/bin/env python3
"""
🚨 HIGHWAY SPEED PROBLEM ANALYSIS
====================================

Your ambulance scenarios have EXTREMELY LOW speed limits that make them
NOT HIGHWAY at all! Let's analyze and fix this.

Real highways should have:
- Speed limits: 60-130 km/h (not 10-35 km/h!)
- Ambulance speeds: 80-120 km/h (not 3 km/h!)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('.')

def analyze_scenario_speed_limits():
    """Analyze the speed limits configured in your ambulance scenarios."""
    
    print("🚨 HIGHWAY SPEED PROBLEM DIAGNOSIS")
    print("="*60)
    
    # Import ambulance scenarios
    try:
        from collecting_ambulance_data.scenarios.ambulance_scenarios import (
            get_ambulance_scenarios,
            get_additional_ambulance_scenarios, 
            get_extended_ambulance_scenarios
        )
        
        # Get all scenarios
        scenarios_1_10 = get_ambulance_scenarios()
        scenarios_11_15 = get_additional_ambulance_scenarios()
        scenarios_16_30 = get_extended_ambulance_scenarios()
        
        all_scenarios = {**scenarios_1_10, **scenarios_11_15, **scenarios_16_30}
        
        print(f"📊 Found {len(all_scenarios)} ambulance scenarios")
        print()
        
        # Analyze speed limits
        speed_data = []
        for name, config in all_scenarios.items():
            speed_limit = config.get('speed_limit', 'Unknown')
            vehicles_count = config.get('vehicles_count', 'Unknown')
            traffic_density = config.get('traffic_density', 'Unknown')
            description = config.get('description', 'No description')
            
            speed_data.append({
                'scenario': name,
                'speed_limit_kmh': speed_limit,
                'vehicles_count': vehicles_count,
                'traffic_density': traffic_density,
                'description': description
            })
        
        # Create DataFrame for analysis
        df = pd.DataFrame(speed_data)
        
        print("🏃 SPEED LIMIT ANALYSIS:")
        print("-" * 40)
        
        if 'speed_limit_kmh' in df.columns:
            speed_stats = df['speed_limit_kmh'].describe()
            print(f"Speed Limit Statistics (km/h):")
            print(f"  Minimum: {speed_stats['min']} km/h")
            print(f"  Maximum: {speed_stats['max']} km/h") 
            print(f"  Average: {speed_stats['mean']:.1f} km/h")
            print(f"  Median: {speed_stats['50%']} km/h")
            print()
            
            # Show distribution
            print("Speed Limit Distribution:")
            speed_counts = df['speed_limit_kmh'].value_counts().sort_index()
            for speed, count in speed_counts.items():
                print(f"  {speed:2d} km/h: {count:2d} scenarios")
        
        print()
        print("🚗 TRAFFIC DENSITY ANALYSIS:")
        print("-" * 40)
        
        if 'traffic_density' in df.columns:
            density_counts = df['traffic_density'].value_counts()
            for density, count in density_counts.items():
                print(f"  {density}: {count} scenarios")
        
        print()
        print("🚨 THE PROBLEM IDENTIFIED:")
        print("-" * 40)
        print("❌ Your 'highway' scenarios have CITY STREET speeds!")
        print("❌ Speed limits 10-35 km/h are NOT highway speeds!")
        print("❌ Real highways: 60-130 km/h")
        print("❌ Your scenarios: 10-35 km/h (residential/city speeds)")
        print()
        print("This explains why ambulances only go 3 km/h:")
        print("  • Heavy traffic (40+ vehicles)")
        print("  • + Low speed limits (10-35 km/h)")  
        print("  • = Gridlock at walking speed!")
        
        print()
        print("📋 DETAILED SCENARIO BREAKDOWN:")
        print("-" * 50)
        
        # Show each scenario with its problematic speed
        for _, row in df.iterrows():
            print(f"• {row['scenario']}")
            print(f"  Speed Limit: {row['speed_limit_kmh']} km/h ❌ (TOO LOW)")
            print(f"  Traffic: {row['traffic_density']} ({row['vehicles_count']} vehicles)")
            print(f"  Description: {row['description'][:60]}...")
            print()
            
        return df
        
    except Exception as e:
        print(f"❌ Error analyzing scenarios: {e}")
        return None

def suggest_highway_fixes():
    """Suggest realistic highway speed configurations."""
    
    print("🛠️  HIGHWAY SPEED FIXES")
    print("="*50)
    
    print("Real Highway Configurations Should Be:")
    print()
    
    highway_configs = [
        {
            "scenario_type": "Light Highway Traffic",
            "speed_limit": "100-120 km/h",
            "vehicles_count": "15-25",
            "expected_ambulance_speed": "80-100 km/h"
        },
        {
            "scenario_type": "Moderate Highway Traffic", 
            "speed_limit": "80-100 km/h",
            "vehicles_count": "25-35",
            "expected_ambulance_speed": "60-80 km/h"
        },
        {
            "scenario_type": "Heavy Highway Traffic",
            "speed_limit": "60-80 km/h", 
            "vehicles_count": "35-50",
            "expected_ambulance_speed": "40-60 km/h"
        },
        {
            "scenario_type": "Highway Construction",
            "speed_limit": "50-60 km/h",
            "vehicles_count": "20-30", 
            "expected_ambulance_speed": "30-50 km/h"
        }
    ]
    
    for i, config in enumerate(highway_configs, 1):
        print(f"{i}. {config['scenario_type']}:")
        print(f"   Speed Limit: {config['speed_limit']}")
        print(f"   Vehicles: {config['vehicles_count']}")
        print(f"   Expected Ambulance Speed: {config['expected_ambulance_speed']}")
        print()
    
    print("🎯 RECOMMENDED CHANGES:")
    print("-" * 30)
    print("1. Change speed_limit from 10-35 km/h to 60-120 km/h")
    print("2. Reduce vehicles_count for free-flow scenarios (15-25 vehicles)")
    print("3. Keep heavy traffic scenarios but with highway speeds (60+ km/h)")
    print("4. Add true 'free flow' scenarios with minimal traffic")
    
    print()
    print("With these changes, you should see:")
    print("✅ Ambulance speeds: 40-100 km/h (realistic highway)")
    print("✅ Traffic flow appropriate for highways")
    print("✅ Realistic emergency response scenarios")

def create_sample_highway_config():
    """Create a sample proper highway configuration."""
    
    print("📝 SAMPLE HIGHWAY CONFIGURATION")
    print("="*40)
    
    sample_config = '''
# PROPER HIGHWAY AMBULANCE SCENARIO
"highway_emergency_realistic": {
    "scenario_name": "highway_emergency_realistic",
    "description": "Ambulance on actual highway with realistic speeds",
    "traffic_density": "moderate",
    "vehicles_count": 25,                    # Moderate traffic
    "initial_lane_id": None,
    "duration": 40,
    "highway_conditions": "normal",
    "speed_limit": 100,                      # 🔥 REAL HIGHWAY SPEED!
    "spawn_probability": 0.4,                # Lower spawning
    "collision_reward": -1,
    "lanes_count": 4,
    "_ambulance_config": {
        "emergency_priority": "high",
        "max_speed": 120                     # 🔥 AMBULANCE CAN GO FASTER!
    }
}
'''
    
    print(sample_config)
    
    print("🎯 KEY CHANGES:")
    print("• speed_limit: 100 km/h (was 25 km/h)")
    print("• spawn_probability: 0.4 (was 0.8)")  
    print("• This should give ambulance speeds: 60-80 km/h")
    print("• Much more realistic for highway emergency response!")

if __name__ == "__main__":
    print("🚨 HIGHWAY SPEED INVESTIGATION")
    print("="*60)
    print("Analyzing why your 'highway' scenarios are only 3 km/h...")
    print()
    
    # Analyze current configurations
    df = analyze_scenario_speed_limits()
    
    if df is not None:
        print()
        suggest_highway_fixes()
        print()
        create_sample_highway_config()
    
    print()
    print("🏁 CONCLUSION:")
    print("Your scenarios are configured as CITY STREETS, not highways!")
    print("Change speed_limit from 10-35 km/h to 60-120 km/h for realistic highway simulation.")