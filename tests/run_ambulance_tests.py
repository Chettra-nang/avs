#!/usr/bin/env python3
"""
Test runner for comprehensive ambulance system tests.

This script runs all ambulance-related tests and provides a summary report.
It activates the required environment and runs tests in the correct order.

Requirements covered: 1.4, 2.4, 6.4
"""

import os
import sys
import subprocess
import unittest
import logging
from pathlib import Path
from typing import List, Dict, Any
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class AmbulanceTestRunner:
    """Comprehensive test runner for ambulance system."""
    
    def __init__(self):
        self.project_root = project_root
        self.test_modules = [
            'tests.test_ambulance_comprehensive',
            'tests.test_ambulance_environment_factory', 
            'tests.test_ambulance_multimodal_outputs',
            'tests.test_ambulance_scenario_registry',
            'tests.test_ambulance_scenario_validation'
        ]
        self.results = {}
        self.total_tests = 0
        self.total_failures = 0
        self.total_errors = 0
        
    def setup_logging(self):
        """Set up logging for test execution."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.project_root / 'tests' / 'ambulance_test_results.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def check_environment(self) -> bool:
        """Check if the required environment is available."""
        logger = logging.getLogger(__name__)
        
        # Check if we're in the correct environment
        venv_path = self.project_root / 'avs_venv'
        if venv_path.exists():
            logger.info(f"Found virtual environment at: {venv_path}")
            return True
        else:
            logger.warning(f"Virtual environment not found at: {venv_path}")
            logger.info("Proceeding with current Python environment")
            return False
    
    def activate_environment(self) -> bool:
        """Activate the required environment if available."""
        logger = logging.getLogger(__name__)
        
        venv_path = self.project_root / 'avs_venv'
        if venv_path.exists():
            # Check if we're already in the virtual environment
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                logger.info("Already running in virtual environment")
                return True
            
            # Try to activate the environment
            activate_script = venv_path / 'bin' / 'activate'
            if activate_script.exists():
                logger.info(f"Activating virtual environment: {venv_path}")
                # Note: In Python, we can't directly activate a venv from within a script
                # The environment should be activated before running this script
                return True
        
        logger.info("Using current Python environment")
        return True
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        logger = logging.getLogger(__name__)
        
        required_modules = [
            'highway_env',
            'gymnasium', 
            'numpy',
            'pandas',
            'collecting_ambulance_data',
            'highway_datacollection'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
                logger.debug(f"✓ {module} available")
            except ImportError:
                missing_modules.append(module)
                logger.error(f"✗ {module} not available")
        
        if missing_modules:
            logger.error(f"Missing required modules: {missing_modules}")
            return False
        
        logger.info("All required dependencies are available")
        return True
    
    def run_test_module(self, module_name: str) -> Dict[str, Any]:
        """Run a single test module and return results."""
        logger = logging.getLogger(__name__)
        logger.info(f"Running test module: {module_name}")
        
        start_time = time.time()
        
        try:
            # Load the test module
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromName(module_name)
            
            # Run the tests
            runner = unittest.TextTestRunner(
                verbosity=2,
                stream=sys.stdout,
                buffer=True
            )
            result = runner.run(suite)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Collect results
            test_result = {
                'module': module_name,
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
                'success': result.wasSuccessful(),
                'duration': duration,
                'failure_details': [str(failure[1]) for failure in result.failures],
                'error_details': [str(error[1]) for error in result.errors]
            }
            
            logger.info(f"Completed {module_name}: {result.testsRun} tests, "
                       f"{len(result.failures)} failures, {len(result.errors)} errors "
                       f"in {duration:.2f}s")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Failed to run test module {module_name}: {e}")
            return {
                'module': module_name,
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'skipped': 0,
                'success': False,
                'duration': time.time() - start_time,
                'failure_details': [],
                'error_details': [str(e)]
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all ambulance tests and return comprehensive results."""
        logger = logging.getLogger(__name__)
        logger.info("Starting comprehensive ambulance system tests")
        
        overall_start_time = time.time()
        
        # Run each test module
        for module_name in self.test_modules:
            result = self.run_test_module(module_name)
            self.results[module_name] = result
            
            # Update totals
            self.total_tests += result['tests_run']
            self.total_failures += result['failures']
            self.total_errors += result['errors']
        
        overall_duration = time.time() - overall_start_time
        
        # Generate summary
        summary = {
            'total_modules': len(self.test_modules),
            'total_tests': self.total_tests,
            'total_failures': self.total_failures,
            'total_errors': self.total_errors,
            'total_duration': overall_duration,
            'overall_success': self.total_failures == 0 and self.total_errors == 0,
            'module_results': self.results
        }
        
        logger.info(f"Completed all tests in {overall_duration:.2f}s")
        logger.info(f"Summary: {self.total_tests} tests, {self.total_failures} failures, {self.total_errors} errors")
        
        return summary
    
    def print_summary_report(self, summary: Dict[str, Any]):
        """Print a comprehensive summary report."""
        print("\n" + "="*80)
        print("AMBULANCE SYSTEM TEST SUMMARY REPORT")
        print("="*80)
        
        print(f"Total Test Modules: {summary['total_modules']}")
        print(f"Total Tests Run: {summary['total_tests']}")
        print(f"Total Failures: {summary['total_failures']}")
        print(f"Total Errors: {summary['total_errors']}")
        print(f"Total Duration: {summary['total_duration']:.2f} seconds")
        print(f"Overall Success: {'✓ PASS' if summary['overall_success'] else '✗ FAIL'}")
        
        print("\n" + "-"*80)
        print("MODULE BREAKDOWN")
        print("-"*80)
        
        for module_name, result in summary['module_results'].items():
            status = "✓ PASS" if result['success'] else "✗ FAIL"
            print(f"{module_name:<50} {status}")
            print(f"  Tests: {result['tests_run']}, "
                  f"Failures: {result['failures']}, "
                  f"Errors: {result['errors']}, "
                  f"Duration: {result['duration']:.2f}s")
            
            # Print failure details if any
            if result['failures'] or result['errors']:
                print("  Issues:")
                for failure in result['failure_details']:
                    print(f"    FAILURE: {failure.split(chr(10))[0]}")  # First line only
                for error in result['error_details']:
                    print(f"    ERROR: {error.split(chr(10))[0]}")  # First line only
        
        print("\n" + "="*80)
        
        if summary['overall_success']:
            print("🎉 ALL AMBULANCE SYSTEM TESTS PASSED!")
            print("The ambulance data collection system is ready for use.")
        else:
            print("❌ SOME TESTS FAILED")
            print("Please review the failures above and fix issues before using the system.")
        
        print("="*80)
    
    def save_detailed_report(self, summary: Dict[str, Any]):
        """Save detailed test report to file."""
        report_file = self.project_root / 'tests' / 'ambulance_test_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("AMBULANCE SYSTEM COMPREHENSIVE TEST REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Modules: {summary['total_modules']}\n")
            f.write(f"Total Tests: {summary['total_tests']}\n")
            f.write(f"Total Failures: {summary['total_failures']}\n")
            f.write(f"Total Errors: {summary['total_errors']}\n")
            f.write(f"Duration: {summary['total_duration']:.2f} seconds\n")
            f.write(f"Success: {summary['overall_success']}\n\n")
            
            for module_name, result in summary['module_results'].items():
                f.write(f"MODULE: {module_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Tests Run: {result['tests_run']}\n")
                f.write(f"Failures: {result['failures']}\n")
                f.write(f"Errors: {result['errors']}\n")
                f.write(f"Duration: {result['duration']:.2f}s\n")
                f.write(f"Success: {result['success']}\n")
                
                if result['failure_details']:
                    f.write("\nFailures:\n")
                    for i, failure in enumerate(result['failure_details'], 1):
                        f.write(f"{i}. {failure}\n")
                
                if result['error_details']:
                    f.write("\nErrors:\n")
                    for i, error in enumerate(result['error_details'], 1):
                        f.write(f"{i}. {error}\n")
                
                f.write("\n" + "="*80 + "\n\n")
        
        print(f"Detailed report saved to: {report_file}")


def main():
    """Main test execution function."""
    print("🚑 AMBULANCE SYSTEM COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Initialize test runner
    runner = AmbulanceTestRunner()
    logger = runner.setup_logging()
    
    # Check environment and dependencies
    logger.info("Checking test environment...")
    
    if not runner.check_environment():
        logger.warning("Environment check failed, but continuing...")
    
    if not runner.activate_environment():
        logger.error("Failed to activate environment")
        return 1
    
    if not runner.check_dependencies():
        logger.error("Dependency check failed")
        return 1
    
    logger.info("Environment setup complete")
    
    # Run all tests
    try:
        summary = runner.run_all_tests()
        
        # Print and save reports
        runner.print_summary_report(summary)
        runner.save_detailed_report(summary)
        
        # Return appropriate exit code
        return 0 if summary['overall_success'] else 1
        
    except KeyboardInterrupt:
        logger.info("Test execution interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)