#!/usr/bin/env python3
"""
Deep Ambulance Environment Diagnostic Script

This script investigates why ambulance vehicles are moving slowly
despite proper speed_limit configurations.
"""

import sys
from pathlib import Path
import logging
import json

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_ambulance_environment():
    """
    Deep diagnostic of ambulance environment creation and configuration.
    """
    logger.info("=== DEEP AMBULANCE ENVIRONMENT DIAGNOSTIC ===")
    
    try:
        # Import necessary modules
        from highway_datacollection.environments.factory import MultiAgentEnvFactory
        from collecting_ambulance_data.scenarios.ambulance_scenarios import get_scenario_by_name
        
        # Initialize factory
        factory = MultiAgentEnvFactory()
        
        # Test scenario
        scenario_name = "highway_emergency_light"
        n_agents = 4
        obs_type = "Kinematics"
        
        logger.info(f"Testing scenario: {scenario_name}")
        
        # 1. Get ambulance scenario configuration
        logger.info("\n1. CHECKING AMBULANCE SCENARIO CONFIGURATION:")
        ambulance_config = get_scenario_by_name(scenario_name)
        
        logger.info(f"   Scenario config keys: {list(ambulance_config.keys())}")
        logger.info(f"   Speed limit: {ambulance_config.get('speed_limit', 'NOT SET')}")
        logger.info(f"   Speed limit (kmh): {ambulance_config.get('speed_limit_kmh', 'NOT SET')}")
        logger.info(f"   Vehicles count: {ambulance_config.get('vehicles_count', 'NOT SET')}")
        logger.info(f"   Duration: {ambulance_config.get('duration', 'NOT SET')}")
        
        # 2. Get base configuration from factory
        logger.info("\n2. CHECKING FACTORY BASE CONFIGURATION:")
        base_config = factory.get_ambulance_base_config(scenario_name, n_agents)
        
        logger.info(f"   Base config keys: {list(base_config.keys())}")
        logger.info(f"   Speed limit in base config: {base_config.get('speed_limit', 'NOT SET')}")
        logger.info(f"   Vehicles count in base config: {base_config.get('vehicles_count', 'NOT SET')}")
        logger.info(f"   Reward speed range: {base_config.get('reward_speed_range', 'NOT SET')}")
        
        # 3. Create actual environment and inspect it
        logger.info("\n3. CREATING ENVIRONMENT AND INSPECTING:")
        env = factory.create_ambulance_env(scenario_name, obs_type, n_agents)
        
        # Inspect environment configuration
        if hasattr(env, 'config'):
            logger.info(f"   Environment config keys: {list(env.config.keys())}")
            logger.info(f"   Environment speed_limit: {env.config.get('speed_limit', 'NOT SET')}")
            logger.info(f"   Environment vehicles_count: {env.config.get('vehicles_count', 'NOT SET')}")
            logger.info(f"   Environment controlled_vehicles: {env.config.get('controlled_vehicles', 'NOT SET')}")
        
        # Inspect road network
        if hasattr(env, 'road') and env.road:
            logger.info(f"   Road network exists: True")
            logger.info(f"   Road speed_limit: {getattr(env.road, 'speed_limit', 'NOT SET')}")
            
            # Check lanes
            if hasattr(env.road, 'network') and env.road.network:
                logger.info(f"   Road network exists: True")
                lanes = env.road.network.lanes_list()
                if lanes:
                    first_lane = lanes[0]
                    logger.info(f"   First lane speed limit: {getattr(first_lane, 'speed_limit', 'NOT SET')}")
                    logger.info(f"   Lane type: {type(first_lane)}")
        
        # 4. Reset environment and check initial vehicle speeds
        logger.info("\n4. TESTING ENVIRONMENT RESET AND VEHICLE SPEEDS:")
        obs, info = env.reset()
        
        if hasattr(env, 'road') and env.road and hasattr(env.road, 'vehicles'):
            vehicles = env.road.vehicles
            logger.info(f"   Total vehicles: {len(vehicles)}")
            
            for i, vehicle in enumerate(vehicles[:5]):  # Check first 5 vehicles
                if hasattr(vehicle, 'speed') and hasattr(vehicle, 'target_speed'):
                    speed_ms = vehicle.speed
                    speed_kmh = speed_ms * 3.6 if speed_ms else 0
                    target_speed = getattr(vehicle, 'target_speed', 'NOT SET')
                    target_kmh = target_speed * 3.6 if isinstance(target_speed, (int, float)) else target_speed
                    
                    logger.info(f"     Vehicle {i}: speed={speed_kmh:.1f} km/h, target_speed={target_kmh}")
                    
                    # Check if it's a controlled vehicle
                    controlled = vehicle in getattr(env, 'controlled_vehicles', [])
                    logger.info(f"                 controlled={controlled}, type={type(vehicle).__name__}")
        
        # 5. Run a few steps and monitor speeds
        logger.info("\n5. RUNNING ENVIRONMENT STEPS TO MONITOR SPEEDS:")
        
        for step in range(5):
            # Random actions for all agents
            actions = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(actions)
            
            if hasattr(env, 'road') and env.road and hasattr(env.road, 'vehicles'):
                controlled_vehicles = getattr(env, 'controlled_vehicles', [])
                if controlled_vehicles:
                    ambulance = controlled_vehicles[0]  # First vehicle should be ambulance
                    if hasattr(ambulance, 'speed'):
                        speed_kmh = ambulance.speed * 3.6
                        logger.info(f"   Step {step}: Ambulance speed = {speed_kmh:.1f} km/h")
        
        env.close()
        
        # 6. Check if highway-env version or configuration might be the issue
        logger.info("\n6. CHECKING HIGHWAY-ENV VERSION AND CONFIGURATION:")
        try:
            import highway_env
            logger.info(f"   highway-env version: {highway_env.__version__ if hasattr(highway_env, '__version__') else 'Unknown'}")
        except Exception as e:
            logger.error(f"   Error checking highway-env: {e}")
        
        logger.info("\n=== DIAGNOSTIC COMPLETE ===")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_highway_env():
    """
    Test creating a highway-env directly with fast speed limits.
    """
    logger.info("\n=== TESTING DIRECT HIGHWAY-ENV CREATION ===")
    
    try:
        import gymnasium as gym
        import highway_env
        
        # Create a simple highway environment with fast speed limit
        config = {
            "lanes_count": 4,
            "vehicles_count": 20,
            "controlled_vehicles": 1,
            "speed_limit": 100,  # 100 km/h
            "duration": 40,
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute": False,
            },
            "action": {
                "type": "DiscreteMetaAction"
            }
        }
        
        logger.info("Creating direct highway-v0 environment...")
        env = gym.make("highway-v0", config=config)
        
        logger.info("Environment created, checking configuration...")
        logger.info(f"Environment speed_limit: {env.config.get('speed_limit', 'NOT SET')}")
        
        # Reset and check vehicles
        obs, info = env.reset()
        
        if hasattr(env, 'road') and env.road:
            if hasattr(env.road, 'vehicles'):
                vehicles = env.road.vehicles
                logger.info(f"Total vehicles: {len(vehicles)}")
                
                for i, vehicle in enumerate(vehicles[:3]):
                    speed_kmh = vehicle.speed * 3.6 if hasattr(vehicle, 'speed') else 0
                    target_speed = getattr(vehicle, 'target_speed', None)
                    target_kmh = target_speed * 3.6 if target_speed else 'None'
                    
                    logger.info(f"  Vehicle {i}: speed={speed_kmh:.1f} km/h, target={target_kmh}")
        
        env.close()
        logger.info("Direct highway-env test complete")
        
        return True
        
    except Exception as e:
        logger.error(f"Direct highway-env test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main diagnostic function."""
    logger.info("=== AMBULANCE SPEED DIAGNOSTIC SYSTEM ===")
    
    # Run diagnostics
    success1 = diagnose_ambulance_environment()
    success2 = test_direct_highway_env()
    
    if success1 and success2:
        logger.info("\n✅ Diagnostic completed successfully!")
        logger.info("📊 Check the output above for speed configuration details")
    else:
        logger.error("❌ Some diagnostics failed - check logs above")

if __name__ == "__main__":
    main()