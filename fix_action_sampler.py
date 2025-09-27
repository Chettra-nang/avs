#!/usr/bin/env python3
"""
Fix for Action Sampler Tuple Issue

This script fixes the action sampler to ensure proper Python integer types
are used instead of numpy integers, which can cause C API errors.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def fix_action_sampler():
    """Fix the action sampler to use Python integers instead of numpy integers."""
    
    action_sampler_path = Path("highway_datacollection/collection/action_samplers.py")
    
    if not action_sampler_path.exists():
        print(f"❌ Action sampler file not found: {action_sampler_path}")
        return False
    
    print("🔧 Fixing action sampler...")
    
    # Read the current file
    with open(action_sampler_path, 'r') as f:
        content = f.read()
    
    # Fix the tuple creation to use Python integers
    old_code = """        actions = tuple(
            self._rng.integers(0, self._action_space_size) 
            for _ in range(n_agents)
        )"""
    
    new_code = """        actions = tuple(
            int(self._rng.integers(0, self._action_space_size))  # Convert to Python int
            for _ in range(n_agents)
        )"""
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        
        # Write the fixed file
        with open(action_sampler_path, 'w') as f:
            f.write(content)
        
        print("✅ Action sampler fixed!")
        print("   Changed numpy integers to Python integers in action tuple")
        return True
    else:
        print("⚠️  Action sampler code pattern not found - may already be fixed")
        return False

def create_robust_action_sampler():
    """Create a more robust action sampler as a backup."""
    
    robust_sampler_path = Path("highway_datacollection/collection/robust_action_sampler.py")
    
    robust_sampler_code = '''"""
Robust Action Sampler

This module provides a more robust action sampler that handles type conversion
issues and provides better error handling.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RobustRandomActionSampler:
    """
    Robust random action sampler that handles type conversion issues.
    
    This sampler ensures that actions are always Python integers,
    preventing C API errors that can occur with numpy integer types.
    """
    
    def __init__(self, action_space_size: int = 5, seed: Optional[int] = None):
        """
        Initialize robust random action sampler.
        
        Args:
            action_space_size: Size of the action space
            seed: Initial random seed
        """
        self._action_space_size = action_space_size
        self._rng = np.random.Generator(np.random.PCG64())
        if seed is not None:
            self.reset(seed)
        
        logger.info(f"Initialized RobustRandomActionSampler with action space size {action_space_size}")
    
    def sample_actions(self, observations: Dict[str, Any], n_agents: int, 
                      step: int = 0, episode_id: str = "") -> Tuple[int, ...]:
        """
        Sample random actions for all agents with robust type handling.
        
        Args:
            observations: Current observations (not used for random sampling)
            n_agents: Number of agents to sample actions for
            step: Current step (not used for random sampling)
            episode_id: Episode ID (not used for random sampling)
            
        Returns:
            Tuple of Python integer actions for each agent
        """
        try:
            # Generate actions and ensure they are Python integers
            actions = tuple(
                int(self._rng.integers(0, self._action_space_size, dtype=np.int32))
                for _ in range(n_agents)
            )
            
            # Validate action types
            for i, action in enumerate(actions):
                if not isinstance(action, int):
                    logger.warning(f"Action {i} is not a Python int: {type(action)}")
                    # Force conversion
                    actions = tuple(int(a) if not isinstance(a, int) else a for a in actions)
                    break
            
            logger.debug(f"Sampled robust actions: {actions} (types: {[type(a).__name__ for a in actions]})")
            return actions
            
        except Exception as e:
            logger.error(f"Action sampling failed: {e}")
            # Fallback to simple Python random
            import random
            fallback_actions = tuple(random.randint(0, self._action_space_size - 1) for _ in range(n_agents))
            logger.warning(f"Using fallback actions: {fallback_actions}")
            return fallback_actions
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the random number generator.
        
        Args:
            seed: Random seed for deterministic behavior
        """
        if seed is not None:
            # Use seed + offset to avoid correlation with environment seeds
            self._rng = np.random.Generator(np.random.PCG64(seed + 1000))
            logger.debug(f"Reset RobustRandomActionSampler with seed {seed}")
    
    def get_action_space_size(self) -> int:
        """
        Get the size of the action space.
        
        Returns:
            Number of possible actions
        """
        return self._action_space_size


def create_robust_sampler(action_space_size: int = 5, seed: Optional[int] = None):
    """
    Factory function to create a robust action sampler.
    
    Args:
        action_space_size: Size of the action space
        seed: Random seed
        
    Returns:
        RobustRandomActionSampler instance
    """
    return RobustRandomActionSampler(action_space_size, seed)
'''
    
    with open(robust_sampler_path, 'w') as f:
        f.write(robust_sampler_code)
    
    print(f"✅ Created robust action sampler: {robust_sampler_path}")
    return True

def main():
    """Main function to fix the action sampler issue."""
    print("🚑 Fixing Ambulance Data Collection Action Sampler Issue")
    print("=" * 55)
    print()
    print("This script fixes the tuple/action type issue that causes:")
    print("  '../Objects/tupleobject.c:909: bad argument to internal function'")
    print()
    
    # Try to fix the existing action sampler
    fixed = fix_action_sampler()
    
    # Create a robust backup sampler
    robust_created = create_robust_action_sampler()
    
    if fixed or robust_created:
        print()
        print("🎉 Action sampler fixes applied!")
        print()
        print("Next steps:")
        print("1. Try running your data collection again")
        print("2. If issues persist, you can use the robust sampler:")
        print()
        print("   from highway_datacollection.collection.robust_action_sampler import create_robust_sampler")
        print("   sampler = create_robust_sampler()")
        print("   collector = AmbulanceDataCollector(action_sampler=sampler)")
        print()
    else:
        print("❌ Could not apply fixes")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''