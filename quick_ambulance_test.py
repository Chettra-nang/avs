#!/usr/bin/env python3
"""
Quick ambulance data collection test to verify the system works.
Collects 1-2 episodes and analyzes the speed data.
"""

import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.join(os.getcwd(), 'collecting_ambulance_data'))

def test_real_ambulance_collection():
    """Test real data collection with 1 episode"""
    
    print("🚑 TESTING REAL AMBULANCE DATA COLLECTION")
    print("=" * 50)
    
    # Method 1: Try using the collecting_ambulance_data validation script
    try:
        print("📋 Testing with validation script...")
        
        # Import the validation module
        sys.path.append('collecting_ambulance_data')
        from validation import main as validation_main
        
        # Override sys.argv to pass parameters
        original_argv = sys.argv
        sys.argv = ['validation.py', '--episodes', '1', '--scenarios', 'highway_emergency_light']
        
        try:
            validation_main()
            print("✅ Validation script worked!")
            return True
        except Exception as e:
            print(f"❌ Validation script failed: {e}")
        finally:
            sys.argv = original_argv
            
    except ImportError as e:
        print(f"❌ Cannot import validation: {e}")
    
    # Method 2: Try direct environment creation
    try:
        print("\n🎮 Testing direct environment creation...")
        
        import gymnasium as gym
        import highway_env
        from scenarios.ambulance_scenarios import get_ambulance_scenarios
        
        scenarios = get_ambulance_scenarios()
        scenario_name = "highway_emergency_light"
        
        if scenario_name not in scenarios:
            print(f"❌ Scenario {scenario_name} not found")
            return False
            
        config = scenarios[scenario_name]
        print(f"📊 Using scenario: {scenario_name}")
        print(f"   Speed limit: {config.get('speed_limit', 'N/A')}")
        print(f"   Reward range: {config.get('reward_speed_range', 'N/A')}")
        
        # Create environment
        env = gym.make('highway-v0')
        env.unwrapped.configure(config)
        
        # Test one episode
        obs, info = env.reset()
        
        speeds = []
        positions = []
        
        print("🏃 Running test episode...")
        
        for step in range(50):  # Short episode
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Try to extract ambulance speed
            if hasattr(env.unwrapped, 'road') and hasattr(env.unwrapped.road, 'vehicles'):
                vehicles = env.unwrapped.road.vehicles
                if len(vehicles) > 0:
                    ambulance = vehicles[0]  # First vehicle is ambulance
                    if hasattr(ambulance, 'speed'):
                        speed_kmh = ambulance.speed * 3.6  # Convert to km/h
                        speeds.append(speed_kmh)
                    
                    if hasattr(ambulance, 'position'):
                        positions.append(ambulance.position)
            
            if terminated or truncated:
                break
        
        env.close()
        
        # Analyze results
        if speeds:
            import numpy as np
            mean_speed = np.mean(speeds)
            max_speed = np.max(speeds)
            
            print(f"\n📊 RESULTS:")
            print(f"   Steps completed: {len(speeds)}")
            print(f"   Mean speed: {mean_speed:.1f} km/h")
            print(f"   Max speed: {max_speed:.1f} km/h")
            print(f"   Speed range: [{np.min(speeds):.1f}, {max_speed:.1f}] km/h")
            
            # Check if speeds are fast
            fast_speeds = [s for s in speeds if s >= 60]
            fast_percentage = len(fast_speeds) / len(speeds) * 100
            
            print(f"   Fast speeds (≥60 km/h): {len(fast_speeds)} ({fast_percentage:.1f}%)")
            
            if fast_percentage > 50:
                print(f"✅ SUCCESS: Ambulance achieving fast speeds!")
                return True
            elif mean_speed > 40:
                print(f"🔶 PARTIAL: Moderate speeds achieved")
                return True
            else:
                print(f"❌ SLOW: Ambulance speeds still slow")
                return False
        else:
            print(f"❌ No speed data collected")
            return False
            
    except Exception as e:
        print(f"❌ Direct environment test failed: {e}")
        return False

def run_visual_test():
    """Run a visual test to see the ambulance behavior"""
    
    print("\n🎬 RUNNING VISUAL TEST")
    print("=" * 30)
    
    try:
        import gymnasium as gym
        import highway_env
        import time
        from scenarios.ambulance_scenarios import get_ambulance_scenarios
        
        scenarios = get_ambulance_scenarios()
        config = scenarios["highway_emergency_light"]
        
        # Modify config for better visualization
        config_copy = config.copy()
        config_copy["render_mode"] = "human"
        config_copy["real_time_rendering"] = True
        config_copy["screen_width"] = 1000
        config_copy["screen_height"] = 300
        
        print("🚗 Opening visualization window...")
        print("   You should see a highway-env window with moving vehicles")
        print("   Red vehicle = Ambulance, Blue/White = Other vehicles")
        print("   Close window or wait 10 seconds to continue")
        
        env = gym.make('highway-v0', render_mode='human')
        env.unwrapped.configure(config_copy)
        
        obs, info = env.reset()
        
        start_time = time.time()
        step = 0
        
        while time.time() - start_time < 10 and step < 100:  # 10 seconds max
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            env.render()
            time.sleep(0.05)  # Small delay
            step += 1
            
            if terminated or truncated:
                break
        
        env.close()
        
        print(f"✅ Visual test completed ({step} steps)")
        return True
        
    except Exception as e:
        print(f"❌ Visual test failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🧪 AMBULANCE DATA COLLECTION & VISUALIZATION TEST")
    print("=" * 55)
    
    # Test 1: Real data collection
    collection_works = test_real_ambulance_collection()
    
    # Test 2: Visual test (if collection works)
    visual_works = False
    if collection_works:
        visual_works = run_visual_test()
    
    # Summary
    print(f"\n📋 TEST SUMMARY")
    print("=" * 20)
    print(f"Data Collection: {'✅ Working' if collection_works else '❌ Issues'}")
    print(f"Visualization: {'✅ Working' if visual_works else '❌ Issues'}")
    
    if collection_works:
        print(f"\n🎉 SUCCESS! Data collection system is working!")
        print(f"💡 You can now run full ambulance data collection")
        
        if visual_works:
            print(f"🎬 Visualization also working - you can see the ambulance behavior")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"1. Run: python collecting_ambulance_data/validation.py --episodes 10")
        print(f"2. Check collected data for fast ambulance speeds")
        print(f"3. Generate videos from collected episodes")
        print(f"4. Scale up to full dataset collection")
        
        return True
    else:
        print(f"\n❌ Issues detected with data collection system")
        print(f"🔧 Check configuration and dependencies")
        return False

if __name__ == "__main__":
    success = main()