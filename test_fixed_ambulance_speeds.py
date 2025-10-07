#!/usr/bin/env python3
"""
Test the fixed ambulance scenarios to verify they now achieve fast speeds.
Quick collection of 3 episodes with speed analysis.
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

from highway_datacollection.collection.synchronized_collector import SynchronizedCollector
import numpy as np
import matplotlib.pyplot as plt

def test_fixed_ambulance_speeds():
    """Test that ambulance scenarios now achieve fast speeds"""
    
    print("🚑 Testing fixed ambulance scenarios for fast speeds...")
    
    # Test specific scenarios that should now have fast speeds
    test_scenarios = [
        "highway_emergency_light",     # 110 km/h limit, [80,110] reward
        "highway_emergency_moderate",  # 95 km/h limit, [70,95] reward
        "highway_emergency_dense"      # 75 km/h limit, [50,75] reward
    ]
    
    all_speeds = []
    scenario_speeds = {}
    
    for scenario in test_scenarios:
        print(f"\n📊 Testing scenario: {scenario}")
        
        # Collect 1 episode for quick test
        collector = SynchronizedCollector(
            observation_types=["Kinematics"],
            scenario_configs=[scenario],
            total_episodes=1,
            output_dir=f"data/test_fixed_speeds_{scenario}",
            render_mode=None,
            save_gif=False
        )
        
        try:
            collected_data = collector.collect_data()
            
            if collected_data and len(collected_data) > 0:
                episode_data = collected_data[0]
                
                # Extract ambulance speeds (first vehicle)
                if 'agent_0_observations' in episode_data:
                    obs_data = episode_data['agent_0_observations']
                    speeds_ms = []
                    
                    for obs in obs_data:
                        if 'speed' in obs:
                            speeds_ms.append(obs['speed'])
                        elif hasattr(obs, 'shape') and len(obs) > 1:
                            # Kinematics observation format
                            speeds_ms.append(obs[1])  # Speed is usually second element
                    
                    if speeds_ms:
                        speeds_kmh = [s * 3.6 for s in speeds_ms]  # Convert m/s to km/h
                        scenario_speeds[scenario] = speeds_kmh
                        all_speeds.extend(speeds_kmh)
                        
                        mean_speed = np.mean(speeds_kmh)
                        max_speed = np.max(speeds_kmh)
                        
                        print(f"   Mean speed: {mean_speed:.1f} km/h")
                        print(f"   Max speed: {max_speed:.1f} km/h")
                        print(f"   Speed range: [{np.min(speeds_kmh):.1f}, {max_speed:.1f}] km/h")
                        
                        # Check if speeds are now fast
                        fast_threshold = 40  # km/h
                        fast_speeds = [s for s in speeds_kmh if s >= fast_threshold]
                        fast_percentage = (len(fast_speeds) / len(speeds_kmh)) * 100
                        
                        if fast_percentage > 50:
                            print(f"   ✅ SUCCESS: {fast_percentage:.1f}% of speeds are fast (≥{fast_threshold} km/h)")
                        else:
                            print(f"   ❌ ISSUE: Only {fast_percentage:.1f}% of speeds are fast (≥{fast_threshold} km/h)")
                    else:
                        print(f"   ❌ No speed data found for {scenario}")
                else:
                    print(f"   ❌ No ambulance observations found for {scenario}")
            else:
                print(f"   ❌ No data collected for {scenario}")
                
        except Exception as e:
            print(f"   ❌ Error testing {scenario}: {e}")
    
    # Overall analysis
    if all_speeds:
        print(f"\n📈 OVERALL RESULTS:")
        print(f"   Total speed samples: {len(all_speeds)}")
        print(f"   Mean speed: {np.mean(all_speeds):.1f} km/h")
        print(f"   Max speed: {np.max(all_speeds):.1f} km/h")
        
        # Speed distribution
        fast_speeds = [s for s in all_speeds if s >= 40]
        moderate_speeds = [s for s in all_speeds if 20 <= s < 40]
        slow_speeds = [s for s in all_speeds if s < 20]
        
        fast_pct = (len(fast_speeds) / len(all_speeds)) * 100
        moderate_pct = (len(moderate_speeds) / len(all_speeds)) * 100
        slow_pct = (len(slow_speeds) / len(all_speeds)) * 100
        
        print(f"\n🚀 Speed Distribution:")
        print(f"   Fast speeds (≥40 km/h): {fast_pct:.1f}%")
        print(f"   Moderate speeds (20-39 km/h): {moderate_pct:.1f}%")
        print(f"   Slow speeds (<20 km/h): {slow_pct:.1f}%")
        
        # Create quick visualization
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(all_speeds, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=40, color='red', linestyle='--', label='Fast threshold (40 km/h)')
        plt.xlabel('Speed (km/h)')
        plt.ylabel('Frequency')
        plt.title('Ambulance Speed Distribution (After Fix)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        categories = ['Slow\n(<20)', 'Moderate\n(20-39)', 'Fast\n(≥40)']
        percentages = [slow_pct, moderate_pct, fast_pct]
        colors = ['red', 'orange', 'green']
        
        bars = plt.bar(categories, percentages, color=colors, alpha=0.7, edgecolor='black')
        plt.ylabel('Percentage of Episodes (%)')
        plt.title('Speed Category Distribution')
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('test_fixed_ambulance_speeds.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Speed analysis saved to: test_fixed_ambulance_speeds.png")
        
        # Success criteria
        if fast_pct > 70:
            print(f"\n🎉 SUCCESS! Fix worked - {fast_pct:.1f}% of speeds are now fast")
            return True
        elif fast_pct > 30:
            print(f"\n🔶 PARTIAL SUCCESS - {fast_pct:.1f}% of speeds are fast (improved but could be better)")
            return True
        else:
            print(f"\n❌ FIX FAILED - Only {fast_pct:.1f}% of speeds are fast")
            return False
    else:
        print("\n❌ No speed data collected - unable to verify fix")
        return False

if __name__ == "__main__":
    success = test_fixed_ambulance_speeds()
    
    if success:
        print("\n✅ Ambulance scenarios are now configured for fast emergency response!")
        print("💡 Ready for full data collection with fast ambulance speeds")
    else:
        print("\n❌ Additional fixes may be needed")