"""
Validation and testing module for ambulance scenarios.

This module provides comprehensive validation and testing functionality for
ambulance scenario configurations, environment setup, and multi-agent behavior.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import traceback

from highway_datacollection.environments.factory import MultiAgentEnvFactory
from highway_datacollection.collection.action_samplers import RandomActionSampler
from .scenarios.ambulance_scenarios import (
    get_all_ambulance_scenarios, 
    get_scenario_names, 
    validate_ambulance_scenario,
    get_supported_observation_types
)


logger = logging.getLogger(__name__)


class AmbulanceScenarioValidator:
    """
    Comprehensive validator for ambulance scenarios and configurations.
    
    This class provides validation for:
    - Ambulance scenario configurations
    - Environment creation and setup
    - Multi-agent behavior verification
    - Observation type compatibility
    """
    
    def __init__(self, n_agents: int = 4):
        """
        Initialize the ambulance scenario validator.
        
        Args:
            n_agents: Number of controlled agents (first agent will be ambulance)
        """
        self.n_agents = n_agents
        self.env_factory = MultiAgentEnvFactory()
        self.action_sampler = RandomActionSampler()
        
        # Get all ambulance scenarios
        self.ambulance_scenarios = get_all_ambulance_scenarios()
        self.scenario_names = get_scenario_names()
        self.supported_obs_types = get_supported_observation_types()
        
        # Validation statistics
        self.validation_stats = {
            "scenarios_validated": 0,
            "scenarios_passed": 0,
            "scenarios_failed": 0,
            "configuration_errors": 0,
            "environment_errors": 0,
            "execution_errors": 0,
            "total_tests_run": 0,
            "total_tests_passed": 0
        }
        
        logger.info(f"Initialized AmbulanceScenarioValidator with {n_agents} agents")
        logger.info(f"Found {len(self.scenario_names)} ambulance scenarios to validate")
    
    def validate_scenario_configuration(self, scenario_name: str) -> Dict[str, Any]:
        """
        Validate a single ambulance scenario configuration.
        
        Args:
            scenario_name: Name of the ambulance scenario to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "scenario_name": scenario_name,
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": {},
            "tests_passed": 0,
            "tests_failed": 0
        }
        
        try:
            # Test 1: Check if scenario exists
            if scenario_name not in self.scenario_names:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Scenario '{scenario_name}' not found")
                validation_result["tests_failed"] += 1
                return validation_result
            
            validation_result["tests_passed"] += 1
            
            # Test 2: Validate scenario configuration structure
            scenario_config = self.ambulance_scenarios[scenario_name]
            if not validate_ambulance_scenario(scenario_config):
                validation_result["valid"] = False
                validation_result["errors"].append("Invalid ambulance scenario configuration structure")
                validation_result["tests_failed"] += 1
            else:
                validation_result["tests_passed"] += 1
            
            # Test 3: Check required ambulance fields
            required_fields = [
                "scenario_name", "controlled_vehicles", "_ambulance_config",
                "lanes_count", "screen_width", "screen_height"
            ]
            
            missing_fields = [field for field in required_fields if field not in scenario_config]
            if missing_fields:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required fields: {missing_fields}")
                validation_result["tests_failed"] += 1
            else:
                validation_result["tests_passed"] += 1
            
            # Test 4: Validate ambulance-specific constraints
            ambulance_config = scenario_config.get("_ambulance_config", {})
            
            # Check ambulance agent index
            if ambulance_config.get("ambulance_agent_index") != 0:
                validation_result["valid"] = False
                validation_result["errors"].append("Ambulance agent index must be 0 (first agent)")
                validation_result["tests_failed"] += 1
            else:
                validation_result["tests_passed"] += 1
            
            # Check controlled vehicles count
            if scenario_config.get("controlled_vehicles") != 4:
                validation_result["valid"] = False
                validation_result["errors"].append("Ambulance scenarios must have exactly 4 controlled vehicles")
                validation_result["tests_failed"] += 1
            else:
                validation_result["tests_passed"] += 1
            
            # Test 5: Validate horizontal orientation
            screen_width = scenario_config.get("screen_width", 0)
            screen_height = scenario_config.get("screen_height", 0)
            if screen_width <= screen_height:
                validation_result["warnings"].append("Screen configuration should use horizontal orientation (width > height)")
            else:
                validation_result["tests_passed"] += 1
            
            # Test 6: Validate lanes count (should be 4 for most scenarios)
            lanes_count = scenario_config.get("lanes_count", 0)
            if lanes_count != 4 and scenario_name != "highway_lane_closure":
                validation_result["warnings"].append(f"Expected 4 lanes, got {lanes_count}")
            else:
                validation_result["tests_passed"] += 1
            
            # Test 7: Check emergency priority
            emergency_priority = ambulance_config.get("emergency_priority")
            if emergency_priority not in ["high", "medium", "low"]:
                validation_result["warnings"].append(f"Unexpected emergency priority: {emergency_priority}")
            else:
                validation_result["tests_passed"] += 1
            
            # Add scenario information
            validation_result["info"] = {
                "description": scenario_config.get("description", "No description"),
                "traffic_density": scenario_config.get("traffic_density", "unknown"),
                "vehicles_count": scenario_config.get("vehicles_count", 0),
                "duration": scenario_config.get("duration", 0),
                "highway_conditions": scenario_config.get("highway_conditions", "normal"),
                "speed_limit": scenario_config.get("speed_limit", 30),
                "ambulance_config": ambulance_config
            }
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Configuration validation failed: {str(e)}")
            validation_result["tests_failed"] += 1
            logger.error(f"Configuration validation error for {scenario_name}: {e}")
        
        return validation_result
    
    def validate_environment_creation(self, scenario_name: str, obs_type: str) -> Dict[str, Any]:
        """
        Validate environment creation for a specific scenario and observation type.
        
        Args:
            scenario_name: Name of the ambulance scenario
            obs_type: Observation type to test
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "scenario_name": scenario_name,
            "obs_type": obs_type,
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": {},
            "tests_passed": 0,
            "tests_failed": 0
        }
        
        env = None
        try:
            # Test 1: Environment factory validation
            if not self.env_factory.validate_ambulance_configuration(scenario_name, obs_type, self.n_agents):
                validation_result["warnings"].append("Environment factory validation raised concerns")
            else:
                validation_result["tests_passed"] += 1
            
            # Test 2: Create ambulance environment
            start_time = time.time()
            env = self.env_factory.create_ambulance_env(scenario_name, obs_type, self.n_agents)
            creation_time = time.time() - start_time
            
            if env is None:
                validation_result["valid"] = False
                validation_result["errors"].append("Environment creation returned None")
                validation_result["tests_failed"] += 1
                return validation_result
            
            validation_result["tests_passed"] += 1
            validation_result["info"]["creation_time"] = creation_time
            
            # Test 3: Environment reset
            start_time = time.time()
            obs, info = env.reset()
            reset_time = time.time() - start_time
            
            validation_result["tests_passed"] += 1
            validation_result["info"]["reset_time"] = reset_time
            
            # Test 4: Check observation structure
            if obs is None:
                validation_result["valid"] = False
                validation_result["errors"].append("Environment reset returned None observation")
                validation_result["tests_failed"] += 1
            else:
                validation_result["tests_passed"] += 1
                
                # For multi-agent environments, check observation structure
                if self.n_agents > 1:
                    if not isinstance(obs, (list, tuple)) or len(obs) != self.n_agents:
                        validation_result["valid"] = False
                        validation_result["errors"].append(f"Expected {self.n_agents} agent observations, got {type(obs)} with length {len(obs) if hasattr(obs, '__len__') else 'unknown'}")
                        validation_result["tests_failed"] += 1
                    else:
                        validation_result["tests_passed"] += 1
            
            # Test 5: Check ambulance configuration in environment
            if hasattr(env, 'unwrapped'):
                env_unwrapped = env.unwrapped
                
                # Check ambulance metadata
                if hasattr(env_unwrapped, '_is_ambulance_env') and env_unwrapped._is_ambulance_env:
                    validation_result["tests_passed"] += 1
                    validation_result["info"]["ambulance_configured"] = True
                else:
                    validation_result["warnings"].append("Ambulance configuration metadata not found")
                
                # Check controlled vehicles
                if hasattr(env_unwrapped, 'controlled_vehicles'):
                    controlled_count = len(env_unwrapped.controlled_vehicles) if env_unwrapped.controlled_vehicles else 0
                    if controlled_count != self.n_agents:
                        validation_result["warnings"].append(f"Expected {self.n_agents} controlled vehicles, found {controlled_count}")
                    else:
                        validation_result["tests_passed"] += 1
                        validation_result["info"]["controlled_vehicles_count"] = controlled_count
            
            # Test 6: Action space validation
            action_space = env.action_space
            if action_space is None:
                validation_result["valid"] = False
                validation_result["errors"].append("Environment has no action space")
                validation_result["tests_failed"] += 1
            else:
                validation_result["tests_passed"] += 1
                validation_result["info"]["action_space_type"] = str(type(action_space))
            
            # Test 7: Observation space validation
            observation_space = env.observation_space
            if observation_space is None:
                validation_result["valid"] = False
                validation_result["errors"].append("Environment has no observation space")
                validation_result["tests_failed"] += 1
            else:
                validation_result["tests_passed"] += 1
                validation_result["info"]["observation_space_type"] = str(type(observation_space))
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Environment creation failed: {str(e)}")
            validation_result["tests_failed"] += 1
            logger.error(f"Environment creation error for {scenario_name}/{obs_type}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        finally:
            # Clean up environment
            if env is not None:
                try:
                    env.close()
                except:
                    pass
        
        return validation_result
    
    def validate_multi_agent_behavior(self, scenario_name: str, obs_type: str, test_steps: int = 10) -> Dict[str, Any]:
        """
        Validate multi-agent behavior in ambulance scenarios.
        
        Args:
            scenario_name: Name of the ambulance scenario
            obs_type: Observation type to test
            test_steps: Number of steps to run for behavior testing
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "scenario_name": scenario_name,
            "obs_type": obs_type,
            "test_steps": test_steps,
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": {},
            "tests_passed": 0,
            "tests_failed": 0
        }
        
        env = None
        try:
            # Create environment
            env = self.env_factory.create_ambulance_env(scenario_name, obs_type, self.n_agents)
            obs, info = env.reset()
            
            validation_result["tests_passed"] += 1  # Environment creation successful
            
            # Test multi-agent step execution
            step_results = []
            total_rewards = [0.0] * self.n_agents
            
            for step in range(test_steps):
                try:
                    # Sample actions for all agents
                    # Create dummy observations dict for action sampling
                    dummy_observations = {"Kinematics": {"observation": obs}}
                    actions = self.action_sampler.sample_actions(dummy_observations, self.n_agents, step)
                    
                    # Execute step
                    step_start = time.time()
                    obs, rewards, dones, truncated, info = env.step(actions)
                    step_time = time.time() - step_start
                    
                    # Record step results
                    step_result = {
                        "step": step,
                        "step_time": step_time,
                        "rewards": rewards if isinstance(rewards, (list, tuple)) else [rewards],
                        "dones": dones if isinstance(dones, (list, tuple)) else [dones],
                        "truncated": truncated if isinstance(truncated, (list, tuple)) else [truncated]
                    }
                    step_results.append(step_result)
                    
                    # Update total rewards
                    if isinstance(rewards, (list, tuple)):
                        for i, reward in enumerate(rewards):
                            if i < len(total_rewards):
                                total_rewards[i] += reward
                    else:
                        # For single reward, distribute to all agents (common in highway-env)
                        for i in range(len(total_rewards)):
                            total_rewards[i] += rewards
                    
                    # Check if episode is done
                    if isinstance(dones, (list, tuple)):
                        if any(dones) or any(truncated if isinstance(truncated, (list, tuple)) else [truncated]):
                            break
                    else:
                        if dones or truncated:
                            break
                            
                except Exception as e:
                    validation_result["errors"].append(f"Step {step} execution failed: {str(e)}")
                    validation_result["tests_failed"] += 1
                    break
            
            if step_results:
                validation_result["tests_passed"] += 1
                validation_result["info"]["steps_completed"] = len(step_results)
                validation_result["info"]["total_rewards"] = total_rewards
                validation_result["info"]["avg_step_time"] = sum(r["step_time"] for r in step_results) / len(step_results)
                
                # Check for ambulance-specific behavior (first agent should be ambulance)
                ambulance_rewards = total_rewards[0] if total_rewards else 0.0
                validation_result["info"]["ambulance_total_reward"] = ambulance_rewards
                
                # Validate reward structure
                if self.n_agents > 1:
                    if len(total_rewards) != self.n_agents:
                        validation_result["warnings"].append(f"Expected {self.n_agents} reward values, got {len(total_rewards)}")
                    else:
                        validation_result["tests_passed"] += 1
                
                # Check for reasonable reward values
                if all(r == 0.0 for r in total_rewards):
                    validation_result["warnings"].append("All agents received zero rewards - check reward configuration")
                else:
                    validation_result["tests_passed"] += 1
            else:
                validation_result["valid"] = False
                validation_result["errors"].append("No steps completed successfully")
                validation_result["tests_failed"] += 1
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Multi-agent behavior validation failed: {str(e)}")
            validation_result["tests_failed"] += 1
            logger.error(f"Multi-agent behavior validation error for {scenario_name}/{obs_type}: {e}")
        
        finally:
            # Clean up environment
            if env is not None:
                try:
                    env.close()
                except:
                    pass
        
        return validation_result
    
    def validate_all_scenarios(self, test_steps: int = 10) -> Dict[str, Any]:
        """
        Validate all ambulance scenarios across all observation types.
        
        Args:
            test_steps: Number of steps to run for behavior testing
            
        Returns:
            Dictionary with comprehensive validation results
        """
        logger.info(f"Starting comprehensive validation of {len(self.scenario_names)} ambulance scenarios")
        
        validation_results = {
            "total_scenarios": len(self.scenario_names),
            "total_obs_types": len(self.supported_obs_types),
            "total_tests": len(self.scenario_names) * len(self.supported_obs_types),
            "scenarios": {},
            "summary": {
                "scenarios_passed": 0,
                "scenarios_failed": 0,
                "configurations_passed": 0,
                "configurations_failed": 0,
                "environments_passed": 0,
                "environments_failed": 0,
                "behaviors_passed": 0,
                "behaviors_failed": 0
            },
            "errors": [],
            "warnings": []
        }
        
        start_time = time.time()
        
        for scenario_idx, scenario_name in enumerate(self.scenario_names):
            logger.info(f"Validating scenario {scenario_idx + 1}/{len(self.scenario_names)}: {scenario_name}")
            
            scenario_results = {
                "scenario_name": scenario_name,
                "configuration": {},
                "environments": {},
                "behaviors": {},
                "overall_valid": True
            }
            
            try:
                # Test 1: Configuration validation
                config_result = self.validate_scenario_configuration(scenario_name)
                scenario_results["configuration"] = config_result
                
                if config_result["valid"]:
                    validation_results["summary"]["configurations_passed"] += 1
                else:
                    validation_results["summary"]["configurations_failed"] += 1
                    scenario_results["overall_valid"] = False
                
                # Test 2: Environment creation for each observation type
                for obs_type in self.supported_obs_types:
                    logger.debug(f"Testing {scenario_name} with {obs_type} observation")
                    
                    # Environment creation test
                    env_result = self.validate_environment_creation(scenario_name, obs_type)
                    scenario_results["environments"][obs_type] = env_result
                    
                    if env_result["valid"]:
                        validation_results["summary"]["environments_passed"] += 1
                    else:
                        validation_results["summary"]["environments_failed"] += 1
                        scenario_results["overall_valid"] = False
                    
                    # Multi-agent behavior test (only if environment creation succeeded)
                    if env_result["valid"]:
                        behavior_result = self.validate_multi_agent_behavior(scenario_name, obs_type, test_steps)
                        scenario_results["behaviors"][obs_type] = behavior_result
                        
                        if behavior_result["valid"]:
                            validation_results["summary"]["behaviors_passed"] += 1
                        else:
                            validation_results["summary"]["behaviors_failed"] += 1
                            scenario_results["overall_valid"] = False
                    else:
                        # Skip behavior test if environment creation failed
                        validation_results["summary"]["behaviors_failed"] += 1
                        scenario_results["overall_valid"] = False
                
                # Update scenario-level statistics
                if scenario_results["overall_valid"]:
                    validation_results["summary"]["scenarios_passed"] += 1
                else:
                    validation_results["summary"]["scenarios_failed"] += 1
                
            except Exception as e:
                error_msg = f"Scenario {scenario_name} validation failed: {str(e)}"
                logger.error(error_msg)
                validation_results["errors"].append(error_msg)
                scenario_results["overall_valid"] = False
                validation_results["summary"]["scenarios_failed"] += 1
            
            validation_results["scenarios"][scenario_name] = scenario_results
        
        # Calculate final statistics
        total_time = time.time() - start_time
        validation_results["validation_time"] = total_time
        validation_results["success_rate"] = (
            validation_results["summary"]["scenarios_passed"] / 
            validation_results["total_scenarios"] * 100
        )
        
        logger.info(f"Validation completed in {total_time:.2f}s")
        logger.info(f"Success rate: {validation_results['success_rate']:.1f}% "
                   f"({validation_results['summary']['scenarios_passed']}/{validation_results['total_scenarios']} scenarios)")
        
        return validation_results
    
    def validate_individual_scenario(self, scenario_name: str, test_steps: int = 10) -> Dict[str, Any]:
        """
        Validate a single ambulance scenario across all observation types.
        
        Args:
            scenario_name: Name of the ambulance scenario to validate
            test_steps: Number of steps to run for behavior testing
            
        Returns:
            Dictionary with validation results for the scenario
        """
        if scenario_name not in self.scenario_names:
            return {
                "scenario_name": scenario_name,
                "valid": False,
                "error": f"Scenario '{scenario_name}' not found. Available scenarios: {self.scenario_names}"
            }
        
        logger.info(f"Validating individual scenario: {scenario_name}")
        
        scenario_results = {
            "scenario_name": scenario_name,
            "configuration": {},
            "environments": {},
            "behaviors": {},
            "overall_valid": True,
            "summary": {
                "tests_passed": 0,
                "tests_failed": 0,
                "obs_types_passed": 0,
                "obs_types_failed": 0
            }
        }
        
        try:
            # Configuration validation
            config_result = self.validate_scenario_configuration(scenario_name)
            scenario_results["configuration"] = config_result
            scenario_results["summary"]["tests_passed"] += config_result["tests_passed"]
            scenario_results["summary"]["tests_failed"] += config_result["tests_failed"]
            
            if not config_result["valid"]:
                scenario_results["overall_valid"] = False
            
            # Test each observation type
            for obs_type in self.supported_obs_types:
                logger.debug(f"Testing {scenario_name} with {obs_type} observation")
                
                # Environment creation
                env_result = self.validate_environment_creation(scenario_name, obs_type)
                scenario_results["environments"][obs_type] = env_result
                scenario_results["summary"]["tests_passed"] += env_result["tests_passed"]
                scenario_results["summary"]["tests_failed"] += env_result["tests_failed"]
                
                # Multi-agent behavior (if environment creation succeeded)
                if env_result["valid"]:
                    behavior_result = self.validate_multi_agent_behavior(scenario_name, obs_type, test_steps)
                    scenario_results["behaviors"][obs_type] = behavior_result
                    scenario_results["summary"]["tests_passed"] += behavior_result["tests_passed"]
                    scenario_results["summary"]["tests_failed"] += behavior_result["tests_failed"]
                    
                    if behavior_result["valid"]:
                        scenario_results["summary"]["obs_types_passed"] += 1
                    else:
                        scenario_results["summary"]["obs_types_failed"] += 1
                        scenario_results["overall_valid"] = False
                else:
                    scenario_results["summary"]["obs_types_failed"] += 1
                    scenario_results["overall_valid"] = False
            
        except Exception as e:
            error_msg = f"Individual scenario validation failed: {str(e)}"
            logger.error(error_msg)
            scenario_results["error"] = error_msg
            scenario_results["overall_valid"] = False
        
        logger.info(f"Individual scenario validation completed for {scenario_name}: "
                   f"{'PASSED' if scenario_results['overall_valid'] else 'FAILED'}")
        
        return scenario_results
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """
        Get current validation statistics.
        
        Returns:
            Dictionary with validation statistics
        """
        return self.validation_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset validation statistics."""
        self.validation_stats = {
            "scenarios_validated": 0,
            "scenarios_passed": 0,
            "scenarios_failed": 0,
            "configuration_errors": 0,
            "environment_errors": 0,
            "execution_errors": 0,
            "total_tests_run": 0,
            "total_tests_passed": 0
        }
        logger.info("Validation statistics reset")


def run_ambulance_scenario_validation(scenarios: Optional[List[str]] = None, 
                                    test_steps: int = 10,
                                    n_agents: int = 4) -> Dict[str, Any]:
    """
    Run ambulance scenario validation with specified parameters.
    
    Args:
        scenarios: List of scenario names to validate (None for all)
        test_steps: Number of steps to run for behavior testing
        n_agents: Number of controlled agents
        
    Returns:
        Dictionary with validation results
    """
    validator = AmbulanceScenarioValidator(n_agents=n_agents)
    
    if scenarios is None:
        # Validate all scenarios
        return validator.validate_all_scenarios(test_steps=test_steps)
    else:
        # Validate specific scenarios
        results = {
            "scenarios": {},
            "summary": {
                "scenarios_passed": 0,
                "scenarios_failed": 0
            }
        }
        
        for scenario_name in scenarios:
            result = validator.validate_individual_scenario(scenario_name, test_steps=test_steps)
            results["scenarios"][scenario_name] = result
            
            if result.get("overall_valid", False):
                results["summary"]["scenarios_passed"] += 1
            else:
                results["summary"]["scenarios_failed"] += 1
        
        return results


if __name__ == "__main__":
    # Run validation when script is executed directly
    logging.basicConfig(level=logging.INFO)
    
    print("Running ambulance scenario validation...")
    results = run_ambulance_scenario_validation(test_steps=5)
    
    print(f"\nValidation Results:")
    print(f"Total scenarios: {results['total_scenarios']}")
    print(f"Scenarios passed: {results['summary']['scenarios_passed']}")
    print(f"Scenarios failed: {results['summary']['scenarios_failed']}")
    print(f"Success rate: {results['success_rate']:.1f}%")
    
    if results["errors"]:
        print(f"\nErrors encountered:")
        for error in results["errors"]:
            print(f"  - {error}")