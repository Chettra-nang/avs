#!/usr/bin/env python3
"""
🚀 FAST AMBULANCE SCENARIOS
============================

For ambulances that need to be "fast as possible", we need:
- High speed limits (80-120 km/h)
- Light traffic (10-20 vehicles)
- Low spawn probability (0.1-0.3)
- Free-flowing highway conditions

This will create scenarios where ambulances can reach 60-100 km/h!
"""

def get_fast_ambulance_scenarios():
    """
    HIGH-SPEED ambulance scenarios for maximum velocity research.
    
    These scenarios prioritize speed over traffic complexity:
    - Speed limits: 80-120 km/h (real highway speeds)
    - Light traffic: 10-25 vehicles maximum
    - Low congestion: spawn_probability 0.1-0.3
    - Open lanes: 4+ lanes for maneuvering
    """
    
    base_fast_config = {
        "lanes_count": 4,
        "controlled_vehicles": 4,
        "simulation_frequency": 15,
        "policy_frequency": 1,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "action": {"type": "DiscreteMetaAction"},
        "collision_reward": -5,  # High penalty to encourage safe high-speed driving
        "reward_speed_range": [60, 120],  # Reward high speeds
        "_ambulance_config": {
            "emergency_priority": "critical",
            "max_speed": 140,  # Allow very high speeds
            "min_speed": 40,   # Don't get stuck slow
        }
    }
    
    fast_scenarios = {
        
        # SCENARIO 1: Empty Highway - Maximum Speed
        "highway_empty_sprint": {
            **base_fast_config,
            "scenario_name": "highway_empty_sprint",
            "description": "Ambulance on nearly empty highway - maximum speed possible",
            "traffic_density": "minimal",
            "vehicles_count": 8,           # Very few vehicles
            "duration": 30,
            "speed_limit": 120,            # 🚀 MAXIMUM HIGHWAY SPEED
            "spawn_probability": 0.1,      # Almost no new traffic
            "highway_conditions": "free_flow",
        },
        
        # SCENARIO 2: Light Morning Traffic - High Speed
        "highway_light_fast": {
            **base_fast_config,
            "scenario_name": "highway_light_fast",
            "description": "Ambulance in light morning traffic - high speed corridor",
            "traffic_density": "light",
            "vehicles_count": 15,          # Light traffic
            "duration": 35,
            "speed_limit": 100,            # 🚀 HIGH SPEED
            "spawn_probability": 0.2,      # Low spawning
            "highway_conditions": "light_flow",
        },
        
        # SCENARIO 3: Open Highway Lanes - Fast Overtaking
        "highway_fast_overtake": {
            **base_fast_config,
            "scenario_name": "highway_fast_overtake",
            "description": "Ambulance with fast overtaking opportunities",
            "traffic_density": "light",
            "vehicles_count": 18,
            "duration": 40,
            "speed_limit": 110,            # 🚀 VERY HIGH SPEED
            "spawn_probability": 0.15,     # Very low spawning
            "highway_conditions": "overtaking_lanes",
        },
        
        # SCENARIO 4: Express Lane - Speed Priority
        "highway_express_lane": {
            **base_fast_config,
            "scenario_name": "highway_express_lane",
            "description": "Ambulance using express/HOV lane for maximum speed",
            "traffic_density": "light",
            "vehicles_count": 12,
            "duration": 30,
            "speed_limit": 115,            # 🚀 EXPRESS SPEED
            "spawn_probability": 0.1,      # Minimal traffic
            "highway_conditions": "express_lane",
        },
        
        # SCENARIO 5: Off-Peak Highway - Speed Run
        "highway_off_peak_speed": {
            **base_fast_config,
            "scenario_name": "highway_off_peak_speed",
            "description": "Ambulance during off-peak hours - speed focused",
            "traffic_density": "minimal",
            "vehicles_count": 10,
            "duration": 35,
            "speed_limit": 105,            # 🚀 HIGH SPEED
            "spawn_probability": 0.12,     # Very low traffic
            "highway_conditions": "off_peak",
        },
        
        # SCENARIO 6: Country Highway - Open Road
        "highway_country_open": {
            **base_fast_config,
            "scenario_name": "highway_country_open",
            "description": "Ambulance on open country highway - minimal obstacles",
            "traffic_density": "sparse",
            "vehicles_count": 6,           # Very sparse traffic
            "duration": 40,
            "speed_limit": 95,             # 🚀 OPEN ROAD SPEED
            "spawn_probability": 0.08,     # Almost empty
            "highway_conditions": "rural_highway",
        },
        
        # SCENARIO 7: Late Night Emergency - Clear Roads
        "highway_night_clear": {
            **base_fast_config,
            "scenario_name": "highway_night_clear",
            "description": "Ambulance late night emergency - clear roads",
            "traffic_density": "minimal",
            "vehicles_count": 5,           # Almost empty
            "duration": 25,
            "speed_limit": 110,            # 🚀 NIGHT HIGH SPEED
            "spawn_probability": 0.05,     # Nearly no traffic
            "highway_conditions": "night_clear",
        },
        
        # SCENARIO 8: Highway Straightaway - Speed Test
        "highway_straightaway_speed": {
            **base_fast_config,
            "scenario_name": "highway_straightaway_speed",
            "description": "Ambulance on long straightaway - speed optimization",
            "traffic_density": "light",
            "vehicles_count": 14,
            "duration": 45,
            "speed_limit": 100,            # 🚀 SUSTAINED HIGH SPEED
            "spawn_probability": 0.18,
            "highway_conditions": "straightaway",
        },
        
        # SCENARIO 9: Multi-Lane Fast Flow
        "highway_multi_lane_fast": {
            **base_fast_config,
            "scenario_name": "highway_multi_lane_fast",
            "description": "Ambulance on 6-lane highway with fast traffic flow",
            "traffic_density": "light",
            "vehicles_count": 20,
            "lanes_count": 6,              # More lanes = more speed options
            "duration": 40,
            "speed_limit": 90,             # 🚀 MULTI-LANE SPEED
            "spawn_probability": 0.25,
            "highway_conditions": "multi_lane_flow",
        },
        
        # SCENARIO 10: Emergency Corridor - Maximum Priority
        "highway_emergency_corridor": {
            **base_fast_config,
            "scenario_name": "highway_emergency_corridor",
            "description": "Ambulance with emergency corridor - maximum speed priority",
            "traffic_density": "light",
            "vehicles_count": 16,
            "duration": 30,
            "speed_limit": 125,            # 🚀 EMERGENCY MAXIMUM
            "spawn_probability": 0.1,
            "highway_conditions": "emergency_corridor",
            "collision_reward": -10,       # Extra safety at high speed
        }
    }
    
    return fast_scenarios

def print_speed_comparison():
    """Show the dramatic speed difference between current and fast scenarios."""
    
    print("🚀 SPEED COMPARISON")
    print("="*50)
    
    print("❌ CURRENT SCENARIOS (SLOW):")
    print("   Speed Limits: 10-35 km/h")
    print("   Vehicles: 15-45 (heavy traffic)")
    print("   Spawn Rate: 0.3-0.9 (high congestion)")
    print("   Result: 3 km/h ambulance speed")
    print()
    
    print("✅ NEW FAST SCENARIOS:")
    print("   Speed Limits: 90-125 km/h")
    print("   Vehicles: 5-20 (light traffic)")
    print("   Spawn Rate: 0.05-0.25 (low congestion)")
    print("   Expected Result: 60-100 km/h ambulance speed!")
    print()
    
    print("📈 EXPECTED PERFORMANCE:")
    print("   • Empty Highway: 80-100 km/h")
    print("   • Light Traffic: 70-90 km/h")
    print("   • Express Lane: 75-95 km/h")
    print("   • Night Clear: 85-105 km/h")
    print()
    
    print("🎯 KEY CHANGES FOR SPEED:")
    print("   1. Speed limits: 90-125 km/h (3x-4x higher)")
    print("   2. Traffic count: 5-20 vehicles (1/2 to 1/8 less)")
    print("   3. Spawn rate: 0.05-0.25 (1/4 to 1/18 less)")
    print("   4. Reward high speeds: 60-120 km/h range")

if __name__ == "__main__":
    print("🚀 FAST AMBULANCE SCENARIO GENERATOR")
    print("="*60)
    print("Creating HIGH-SPEED scenarios for maximum ambulance velocity...")
    print()
    
    scenarios = get_fast_ambulance_scenarios()
    
    print(f"📊 Generated {len(scenarios)} fast ambulance scenarios:")
    print()
    
    for name, config in scenarios.items():
        print(f"• {name}")
        print(f"  Speed Limit: {config['speed_limit']} km/h 🚀")
        print(f"  Vehicles: {config['vehicles_count']} (light traffic)")
        print(f"  Spawn Rate: {config['spawn_probability']} (low congestion)")
        print(f"  Expected Ambulance Speed: 60-90 km/h")
        print()
    
    print()
    print_speed_comparison()
    
    print()
    print("🏁 NEXT STEPS:")
    print("1. Replace your current ambulance_scenarios.py with these fast scenarios")
    print("2. Collect new dataset with high-speed configurations")
    print("3. Expect ambulance speeds of 60-100 km/h instead of 3 km/h!")
    print("4. Perfect for 'fast as possible' ambulance research! 🚑💨")