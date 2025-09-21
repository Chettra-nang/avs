#!/usr/bin/env python3
"""
Test script to verify multi-agent setup works correctly
"""

import gymnasium as gym
import highway_env
import numpy as np

def test_multiagent_config():
    """Test the multi-agent configuration."""
    print("🧪 Testing Multi-Agent Configuration")
    print("=" * 40)
    
    # Create environment with multi-agent config
    env = gym.make('highway-v0', render_mode='rgb_array')
    
    config = {
        'observation': {
            'type': 'MultiAgentObservation',
            'observation_config': {
                'type': 'Kinematics',
                'vehicles_count': 15,
                'features': ['presence', 'x', 'y', 'vx', 'vy'],
                'absolute': False,
                'normalize': True
            }
        },
        'action': {
            'type': 'MultiAgentAction',
            'action_config': {
                'type': 'DiscreteMetaAction'
            }
        },
        'controlled_vehicles': 4,
        'lanes_count': 4,
        'vehicles_count': 20,
        'duration': 20,
    }
    
    env.unwrapped.configure(config)
    
    # Test reset
    print("🔄 Testing environment reset...")
    obs, info = env.reset()
    
    print(f"✅ Reset successful!")
    print(f"   Observation type: {type(obs)}")
    
    if isinstance(obs, tuple):
        print(f"   Number of agent observations: {len(obs)}")
        for i, agent_obs in enumerate(obs):
            print(f"   Agent {i} observation shape: {agent_obs.shape}")
    else:
        print(f"   Single observation shape: {obs.shape}")
    
    # Test step with multi-agent actions
    print("\n🎮 Testing multi-agent step...")
    
    # Create actions for each agent
    actions = (0, 1, 2, 1)  # Different actions for each agent
    
    try:
        next_obs, reward, terminated, truncated, info = env.step(actions)
        
        print(f"✅ Step successful!")
        print(f"   Next observation type: {type(next_obs)}")
        print(f"   Reward type: {type(reward)}")
        print(f"   Reward value: {reward}")
        print(f"   Terminated: {terminated}")
        print(f"   Truncated: {truncated}")
        
        if isinstance(next_obs, tuple):
            print(f"   Number of next observations: {len(next_obs)}")
        
    except Exception as e:
        print(f"❌ Step failed: {e}")
        print("   This might indicate the environment doesn't support true multi-agent")
    
    # Test a few more steps
    print("\n🔄 Testing multiple steps...")
    for step in range(5):
        try:
            actions = tuple(np.random.randint(0, 5) for _ in range(4))
            obs, reward, terminated, truncated, info = env.step(actions)
            print(f"   Step {step + 1}: Actions {actions}, Reward {reward}")
            
            if terminated or truncated:
                print("   Episode ended")
                break
                
        except Exception as e:
            print(f"   Step {step + 1} failed: {e}")
            break
    
    env.close()
    print("\n🏁 Test completed!")

if __name__ == "__main__":
    test_multiagent_config()