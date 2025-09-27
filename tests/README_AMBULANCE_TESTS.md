# Ambulance System Comprehensive Test Suite

This directory contains comprehensive tests for the ambulance data collection system, covering all requirements specified in the ambulance data collection specification.

## Test Coverage

The test suite covers the following requirements:
- **Requirement 1.4**: Multi-modal data collection functionality
- **Requirement 2.4**: Multi-agent setup validation
- **Requirement 6.4**: Configuration validation and extensibility

## Test Files

### Core Test Modules

1. **`test_ambulance_comprehensive.py`**
   - Comprehensive integration tests
   - Scenario configuration validation
   - Environment creation and setup
   - Data collection functionality
   - Error handling and edge cases
   - System integration tests

2. **`test_ambulance_environment_factory.py`**
   - Specialized tests for ambulance environment factory
   - Environment creation consistency
   - Configuration validation
   - Resource management
   - Edge cases and error conditions

3. **`test_ambulance_multimodal_outputs.py`**
   - Multi-modal data collection output validation
   - Observation type testing (Kinematics, OccupancyGrid, GrayscaleObservation)
   - Data format and structure validation
   - Storage integration testing
   - Analysis compatibility testing

4. **`test_ambulance_scenario_registry.py`** (existing)
   - Scenario registry integration tests
   - Scenario listing and retrieval
   - Configuration validation
   - Traffic density and condition filtering

5. **`test_ambulance_scenario_validation.py`** (existing)
   - Individual scenario validation
   - Environment creation testing
   - Multi-agent behavior validation
   - Comprehensive validation workflows

### Test Runners

6. **`test_ambulance_basic_functionality.py`**
   - Quick basic functionality verification
   - Import and initialization tests
   - Core component integration
   - Lightweight validation

7. **`run_ambulance_tests.py`**
   - Comprehensive test runner
   - Environment setup and validation
   - Detailed reporting and logging
   - Summary generation

## Running Tests

### Prerequisites

1. **Activate the virtual environment** (recommended):
   ```bash
   source avs_venv/bin/activate
   ```

2. **Ensure all dependencies are installed**:
   ```bash
   pip install -r requirements.txt
   ```

### Quick Basic Test

For a quick verification that the ambulance system is working:

```bash
python tests/test_ambulance_basic_functionality.py
```

This runs lightweight tests to verify core functionality without extensive setup.

### Comprehensive Test Suite

For full comprehensive testing:

```bash
python tests/run_ambulance_tests.py
```

This runs all ambulance tests and generates detailed reports.

### Individual Test Modules

To run specific test modules:

```bash
# Test environment factory
python -m unittest tests.test_ambulance_environment_factory -v

# Test multi-modal outputs
python -m unittest tests.test_ambulance_multimodal_outputs -v

# Test comprehensive functionality
python -m unittest tests.test_ambulance_comprehensive -v

# Test scenario registry integration
python -m unittest tests.test_ambulance_scenario_registry -v

# Test scenario validation
python -m unittest tests.test_ambulance_scenario_validation -v
```

### Running with Coverage

To run tests with coverage analysis:

```bash
# Install coverage if not already installed
pip install coverage

# Run tests with coverage
coverage run -m unittest discover tests/ -p "test_ambulance_*.py"
coverage report
coverage html  # Generate HTML report
```

## Test Results and Reporting

### Output Files

The comprehensive test runner generates several output files:

- **`tests/ambulance_test_results.log`**: Detailed test execution log
- **`tests/ambulance_test_report.txt`**: Comprehensive test report
- **Console output**: Real-time test progress and summary

### Understanding Test Results

#### Success Indicators
- ✓ All tests pass
- No failures or errors reported
- All 15 ambulance scenarios validate successfully
- All 3 observation types work correctly
- Multi-agent setup (4 agents) functions properly

#### Common Issues and Solutions

1. **Import Errors**
   - Ensure virtual environment is activated
   - Verify all dependencies are installed
   - Check PYTHONPATH includes project root

2. **Environment Creation Failures**
   - Verify highway-env is properly installed
   - Check that ambulance scenarios are properly configured
   - Ensure sufficient system resources

3. **Validation Failures**
   - Check ambulance scenario configurations
   - Verify environment factory ambulance methods
   - Ensure scenario registry integration

## Test Architecture

### Test Categories

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **System Tests**: Test complete workflows
4. **Validation Tests**: Test against requirements
5. **Edge Case Tests**: Test error conditions and boundaries

### Test Data

Tests use minimal data collection (1-3 episodes, 2-5 steps) to:
- Minimize test execution time
- Reduce resource requirements
- Focus on functionality rather than performance
- Enable rapid development feedback

### Mock and Stub Usage

Tests use mocking sparingly and prefer real component testing to:
- Ensure actual system functionality
- Catch integration issues
- Validate real-world behavior
- Provide confidence in system reliability

## Continuous Integration

### Automated Testing

The test suite is designed to be run in CI/CD environments:

```bash
# CI-friendly test execution
python tests/run_ambulance_tests.py --ci-mode
```

### Performance Considerations

- Tests are optimized for speed while maintaining coverage
- Resource usage is minimized through small episode counts
- Parallel test execution is supported where possible
- Test isolation prevents interference between tests

## Troubleshooting

### Common Test Failures

1. **Module Import Failures**
   ```
   Solution: Activate virtual environment and install dependencies
   ```

2. **Environment Creation Timeouts**
   ```
   Solution: Increase timeout values or check system resources
   ```

3. **Validation Errors**
   ```
   Solution: Check ambulance scenario configurations and factory methods
   ```

### Debug Mode

Run tests with debug logging:

```bash
PYTHONPATH=. python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
import unittest
unittest.main(module='tests.test_ambulance_comprehensive', verbosity=2)
"
```

### Test Isolation

Each test class and method is designed to be independent:
- Setup and teardown methods clean resources
- No shared state between tests
- Temporary directories are used and cleaned up
- Environment activation is handled per test

## Contributing

When adding new tests:

1. Follow the existing naming convention: `test_ambulance_*.py`
2. Include comprehensive docstrings
3. Use descriptive test method names
4. Add appropriate setup and teardown
5. Include both positive and negative test cases
6. Update this README with new test descriptions

## Requirements Traceability

| Requirement | Test Coverage |
|-------------|---------------|
| 1.4 - Multi-modal data collection | `test_ambulance_multimodal_outputs.py`, `test_ambulance_comprehensive.py` |
| 2.4 - Multi-agent setup validation | `test_ambulance_scenario_validation.py`, `test_ambulance_environment_factory.py` |
| 6.4 - Configuration validation | `test_ambulance_scenario_registry.py`, `test_ambulance_comprehensive.py` |

All tests are designed to validate these requirements comprehensively and provide confidence that the ambulance data collection system meets its specifications.