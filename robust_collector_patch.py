#!/usr/bin/env python3
"""
Robust Collector Patch

This script patches the SynchronizedCollector to add robust action validation
and conversion to prevent the tuple C API error.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def patch_step_parallel_envs():
    """Patch the step_parallel_envs method to add robust action validation."""
    
    collector_path = Path("highway_datacollection/collection/collector.py")
    
    if not collector_path.exists():
        print(f"❌ Collector file not found: {collector_path}")
        return False
    
    print("🔧 Patching SynchronizedCollector.step_parallel_envs...")
    
    # Read the current file
    with open(collector_path, 'r') as f:
        content = f.read()
    
    # Find the step_parallel_envs method
    old_step_method = '''    def step_parallel_envs(self, actions: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Step all parallel environments with the same actions.
        
        Args:
            actions: Tuple of actions for each agent
            
        Returns:
            Dictionary mapping observation types to step results
            
        Raises:
            ValueError: If number of actions doesn't match number of agents
            RuntimeError: If environment step fails
        """
        if len(actions) != self._n_agents:
            raise ValueError(f"Expected {self._n_agents} actions, got {len(actions)}")
        
        logger.debug(f"Stepping parallel environments with actions: {actions}")
        step_results = {}
        
        try:
            for obs_type, env in self._environments.items():
                obs, reward, terminated, truncated, info = env.step(actions)
                step_results[obs_type] = {
                    'observation': obs,
                    'reward': reward,
                    'terminated': terminated,
                    'truncated': truncated,
                    'info': info
                }
                logger.debug(f"Stepped {obs_type} environment successfully")
            
            return step_results
            
        except Exception as e:
            logger.error(f"Failed to step environments: {e}")
            raise RuntimeError(f"Environment step failed: {e}")'''
    
    new_step_method = '''    def step_parallel_envs(self, actions: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Step all parallel environments with the same actions.
        
        Args:
            actions: Tuple of actions for each agent
            
        Returns:
            Dictionary mapping observation types to step results
            
        Raises:
            ValueError: If number of actions doesn't match number of agents
            RuntimeError: If environment step fails
        """
        if len(actions) != self._n_agents:
            raise ValueError(f"Expected {self._n_agents} actions, got {len(actions)}")
        
        # Robust action validation and conversion
        validated_actions = self._validate_and_convert_actions(actions)
        
        logger.debug(f"Stepping parallel environments with actions: {validated_actions}")
        step_results = {}
        
        try:
            for obs_type, env in self._environments.items():
                obs, reward, terminated, truncated, info = env.step(validated_actions)
                step_results[obs_type] = {
                    'observation': obs,
                    'reward': reward,
                    'terminated': terminated,
                    'truncated': truncated,
                    'info': info
                }
                logger.debug(f"Stepped {obs_type} environment successfully")
            
            return step_results
            
        except Exception as e:
            logger.error(f"Failed to step environments: {e}")
            # Try fallback action formats
            fallback_result = self._try_fallback_step(actions)
            if fallback_result is not None:
                logger.warning("Used fallback action format")
                return fallback_result
            raise RuntimeError(f"Environment step failed: {e}")'''
    
    # Add the validation method
    validation_method = '''
    def _validate_and_convert_actions(self, actions: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Validate and convert actions to ensure they are proper Python integers.
        
        Args:
            actions: Input actions tuple
            
        Returns:
            Validated actions tuple with Python integers
        """
        try:
            # Convert all actions to Python integers
            validated_actions = tuple(int(action) for action in actions)
            
            # Additional validation
            for i, action in enumerate(validated_actions):
                if not isinstance(action, int) or isinstance(action, bool):
                    logger.warning(f"Action {i} type issue: {type(action)}, converting...")
                    validated_actions = tuple(
                        int(a) if not (isinstance(a, int) and not isinstance(a, bool)) else a 
                        for a in validated_actions
                    )
                    break
            
            return validated_actions
            
        except Exception as e:
            logger.error(f"Action validation failed: {e}")
            # Fallback to original actions
            return actions
    
    def _try_fallback_step(self, actions: Tuple[int, ...]) -> Optional[Dict[str, Any]]:
        """
        Try alternative action formats as fallback.
        
        Args:
            actions: Original actions
            
        Returns:
            Step results if successful, None if all formats fail
        """
        fallback_formats = [
            list(actions),  # Try as list
            [int(a) for a in actions],  # Try as list of explicit ints
            tuple(int(a) for a in actions),  # Try as tuple of explicit ints
        ]
        
        for i, fallback_actions in enumerate(fallback_formats):
            try:
                logger.debug(f"Trying fallback format {i}: {type(fallback_actions)} {fallback_actions}")
                step_results = {}
                
                for obs_type, env in self._environments.items():
                    obs, reward, terminated, truncated, info = env.step(fallback_actions)
                    step_results[obs_type] = {
                        'observation': obs,
                        'reward': reward,
                        'terminated': terminated,
                        'truncated': truncated,
                        'info': info
                    }
                
                logger.info(f"Fallback format {i} successful")
                return step_results
                
            except Exception as e:
                logger.debug(f"Fallback format {i} failed: {e}")
                continue
        
        return None'''
    
    if old_step_method in content:
        # Replace the method
        content = content.replace(old_step_method, new_step_method)
        
        # Add the validation methods before the cleanup method
        cleanup_method_pos = content.find("    def cleanup(self):")
        if cleanup_method_pos != -1:
            content = content[:cleanup_method_pos] + validation_method + "\n" + content[cleanup_method_pos:]
        else:
            # Add at the end of the class
            content = content + validation_method
        
        # Write the patched file
        with open(collector_path, 'w') as f:
            f.write(content)
        
        print("✅ SynchronizedCollector patched successfully!")
        return True
    else:
        print("⚠️  Could not find step_parallel_envs method to patch")
        return False

def main():
    """Main patching function."""
    print("🔧 Robust Collector Patch")
    print("=" * 25)
    print()
    print("This patch adds robust action validation to prevent tuple C API errors.")
    print()
    
    success = patch_step_parallel_envs()
    
    if success:
        print()
        print("🎉 Patch applied successfully!")
        print()
        print("The collector now includes:")
        print("• Robust action validation and conversion")
        print("• Fallback action formats if primary format fails")
        print("• Enhanced error handling and logging")
        print()
        print("Try running your data collection again:")
        print("  python collecting_ambulance_data/examples/basic_ambulance_collection.py \\")
        print("    --episodes 50 --max-steps 100 --output-dir data/ambulance_dataset")
    else:
        print()
        print("❌ Patch failed - manual intervention may be required")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())