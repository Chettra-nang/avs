#!/usr/bin/env python3
"""
Quick test to verify the fixed ambulance scenario configurations.
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'collecting_ambulance_data'))

from scenarios.ambulance_scenarios import get_ambulance_scenarios

def test_reward_speed_ranges():
    """Verify all scenarios now have reward_speed_range parameters"""
    
    print("🚑 Testing ambulance scenario configurations...")
    
    scenarios = get_ambulance_scenarios()
    
    print(f"\n📊 Found {len(scenarios)} ambulance scenarios")
    
    missing_reward_ranges = []
    speed_comparisons = []
    
    for name, config in scenarios.items():
        speed_limit = config.get('speed_limit', 'N/A')
        reward_range = config.get('reward_speed_range', 'MISSING')
        
        if reward_range == 'MISSING':
            missing_reward_ranges.append(name)
        else:
            speed_comparisons.append((name, speed_limit, reward_range))
    
    # Report missing reward ranges
    if missing_reward_ranges:
        print(f"\n❌ {len(missing_reward_ranges)} scenarios still missing reward_speed_range:")
        for name in missing_reward_ranges[:5]:  # Show first 5
            print(f"   - {name}")
        if len(missing_reward_ranges) > 5:
            print(f"   ... and {len(missing_reward_ranges) - 5} more")
    else:
        print(f"\n✅ All scenarios have reward_speed_range parameters!")
    
    # Show speed comparisons
    print(f"\n🚀 Speed Limit vs Reward Range Comparison:")
    print(f"{'Scenario':<25} {'Speed Limit':<12} {'Reward Range':<15} {'Status'}")
    print("-" * 70)
    
    fast_scenarios = 0
    
    for name, speed_limit, reward_range in speed_comparisons[:10]:  # Show first 10
        if isinstance(reward_range, list) and len(reward_range) == 2:
            max_reward = reward_range[1]
            if max_reward >= 60:  # Fast emergency response
                status = "✅ Fast"
                fast_scenarios += 1
            elif max_reward >= 40:
                status = "🔶 Moderate"
            else:
                status = "❌ Slow"
        else:
            status = "❓ Invalid"
        
        # Truncate long scenario names
        short_name = name[:24] if len(name) > 24 else name
        
        print(f"{short_name:<25} {speed_limit:<12} {str(reward_range):<15} {status}")
    
    if len(speed_comparisons) > 10:
        print(f"... and {len(speed_comparisons) - 10} more scenarios")
    
    # Summary
    fast_percentage = (fast_scenarios / len(speed_comparisons)) * 100 if speed_comparisons else 0
    
    print(f"\n📈 Summary:")
    print(f"   Total scenarios: {len(scenarios)}")
    print(f"   With reward_speed_range: {len(speed_comparisons)}")
    print(f"   Fast scenarios (≥60 km/h max reward): {fast_scenarios} ({fast_percentage:.1f}%)")
    
    if fast_percentage >= 70:
        print(f"\n🎉 SUCCESS! {fast_percentage:.1f}% of scenarios configured for fast speeds")
        return True
    elif fast_percentage >= 40:
        print(f"\n🔶 PARTIAL SUCCESS: {fast_percentage:.1f}% configured for fast speeds")
        return True
    else:
        print(f"\n❌ NEEDS IMPROVEMENT: Only {fast_percentage:.1f}% configured for fast speeds")
        return False

if __name__ == "__main__":
    success = test_reward_speed_ranges()
    
    if success:
        print("\n✅ Ambulance scenarios are properly configured!")
        print("💡 Ready to run data collection with fast emergency response speeds")
    else:
        print("\n❌ Additional configuration fixes may be needed")