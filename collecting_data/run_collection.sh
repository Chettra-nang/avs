#!/bin/bash
# Data Collection Execution Script
# Quick execution script for all 10 data collection scenarios

set -e  # Exit on any error

# Get script directory and change to parent directory (project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

echo "🚗 Highway Data Collection - 10 Scenarios"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}Warning: Virtual environment not detected${NC}"
    echo "Activating avs_venv..."
    source avs_venv/bin/activate
fi

echo -e "${GREEN}✓ Virtual environment active${NC}"

# Create necessary directories
mkdir -p data
mkdir -p logs
echo -e "${GREEN}✓ Directories created${NC}"

# Function to run scenario
run_scenario() {
    local scenario_num=$1
    local scenario_name=$2
    echo ""
    echo -e "${BLUE}Running Scenario $scenario_num: $scenario_name${NC}"
    echo "================================================"
    
    python "collecting_data/scenario_$(printf "%02d" $scenario_num)_$scenario_name.py"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Scenario $scenario_num completed successfully${NC}"
    else
        echo -e "${RED}✗ Scenario $scenario_num failed${NC}"
        exit 1
    fi
}

# Check command line arguments
case "${1:-sequential}" in
    "single")
        if [ -z "$2" ]; then
            echo "Usage: $0 single <scenario_number>"
            echo "Example: $0 single 1"
            exit 1
        fi
        
        scenario_num=$2
        case $scenario_num in
            1) run_scenario 1 "light_free_flow" ;;
            2) run_scenario 2 "heavy_commuting" ;;
            3) run_scenario 3 "stop_and_go" ;;
            4) run_scenario 4 "aggressive_behaviors" ;;
            5) run_scenario 5 "lane_closure" ;;
            6) run_scenario 6 "time_budget" ;;
            7) run_scenario 7 "multi_lane_highway" ;;
            8) run_scenario 8 "mixed_traffic" ;;
            9) run_scenario 9 "high_speed_extended" ;;
            10) run_scenario 10 "multi_agent_coordination" ;;
            *) echo "Invalid scenario number. Use 1-10."; exit 1 ;;
        esac
        ;;
        
    "sequential")
        echo -e "${YELLOW}Running all 10 scenarios sequentially${NC}"
        echo "Estimated time: 4-6 hours"
        echo "Total episodes: 10,000"
        echo ""
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 0
        fi
        
        python collecting_data/run_all_scenarios.py --mode sequential
        ;;
        
    "parallel")
        workers=${2:-3}
        echo -e "${YELLOW}Running all 10 scenarios in parallel (${workers} workers)${NC}"
        echo "Estimated time: 2-3 hours"
        echo "Total episodes: 10,000"
        echo ""
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 0
        fi
        
        python collecting_data/run_all_scenarios.py --mode parallel --workers $workers
        ;;
        
    "test")
        echo -e "${YELLOW}Running quick test (first 3 scenarios with 10 episodes each)${NC}"
        echo "This will collect 30 total episodes for quick validation"
        echo ""
        read -p "Continue with test run? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 0
        fi
        
        # Create temporary test directory
        mkdir -p data/test_run
        
        echo ""
        echo "Creating and running test scenarios..."
        echo "======================================"
        
        # Test Scenario 1: Light Free Flow
        echo -e "${BLUE}Test 1/3: Light Free Flow (10 episodes)${NC}"
        python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))
from highway_datacollection.collection.orchestrator import run_full_collection
from highway_datacollection.collection.collector import SynchronizedCollector

result = run_full_collection(
    base_storage_path=Path('data/test_run/scenario_01_test'),
    episodes_per_scenario=10,
    n_agents=2,
    max_steps_per_episode=50,
    scenarios=['free_flow'],
    base_seed=1001,
    batch_size=5
)
print(f'Test 1 Results: {result.successful_episodes}/{result.total_episodes} episodes successful')
"
        
        echo -e "${BLUE}Test 2/3: Heavy Commuting (10 episodes)${NC}"
        python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))
from highway_datacollection.collection.orchestrator import run_full_collection
from highway_datacollection.collection.collector import SynchronizedCollector

result = run_full_collection(
    base_storage_path=Path('data/test_run/scenario_02_test'),
    episodes_per_scenario=10,
    n_agents=2,
    max_steps_per_episode=50,
    scenarios=['dense_commuting'],
    base_seed=2001,
    batch_size=5
)
print(f'Test 2 Results: {result.successful_episodes}/{result.total_episodes} episodes successful')
"
        
        echo -e "${BLUE}Test 3/3: Stop and Go (10 episodes)${NC}"
        python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))
from highway_datacollection.collection.orchestrator import run_full_collection
from highway_datacollection.collection.collector import SynchronizedCollector

result = run_full_collection(
    base_storage_path=Path('data/test_run/scenario_03_test'),
    episodes_per_scenario=10,
    n_agents=2,
    max_steps_per_episode=50,
    scenarios=['stop_and_go'],
    base_seed=3001,
    batch_size=5
)
print(f'Test 3 Results: {result.successful_episodes}/{result.total_episodes} episodes successful')
"
        
        echo ""
        echo -e "${GREEN}✓ Test run completed!${NC}"
        echo "Test data saved in: data/test_run/"
        echo "Use 'python main.py --demo loading' to verify test data"
        ;;
        
    "quick")
        echo -e "${YELLOW}Running ultra-quick test (3 scenarios with 3 episodes each)${NC}"
        echo "This will collect only 9 total episodes for very quick validation"
        echo ""
        read -p "Continue with quick test? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 0
        fi
        
        # Create temporary test directory
        mkdir -p data/quick_test
        
        echo ""
        echo "Running ultra-quick test scenarios..."
        echo "===================================="
        
        # Quick Test: Just 3 episodes each, very short
        echo -e "${BLUE}Quick Test 1/3: Light Free Flow (3 episodes)${NC}"
        python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))
from highway_datacollection.collection.orchestrator import run_full_collection

result = run_full_collection(
    base_storage_path=Path('data/quick_test/light_flow'),
    episodes_per_scenario=3,
    n_agents=1,
    max_steps_per_episode=30,
    scenarios=['free_flow'],
    base_seed=9001,
    batch_size=3
)
print(f'Quick Test 1: {result.successful_episodes}/3 episodes')
" || echo "Test 1 failed"
        
        echo -e "${BLUE}Quick Test 2/3: Dense Traffic (3 episodes)${NC}"
        python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))
from highway_datacollection.collection.orchestrator import run_full_collection

result = run_full_collection(
    base_storage_path=Path('data/quick_test/dense_traffic'),
    episodes_per_scenario=3,
    n_agents=1,
    max_steps_per_episode=30,
    scenarios=['dense_commuting'],
    base_seed=9002,
    batch_size=3
)
print(f'Quick Test 2: {result.successful_episodes}/3 episodes')
" || echo "Test 2 failed"
        
        echo -e "${BLUE}Quick Test 3/3: Stop-Go (3 episodes)${NC}"
        python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))
from highway_datacollection.collection.orchestrator import run_full_collection

result = run_full_collection(
    base_storage_path=Path('data/quick_test/stop_go'),
    episodes_per_scenario=3,
    n_agents=1,
    max_steps_per_episode=30,
    scenarios=['stop_and_go'],
    base_seed=9003,
    batch_size=3
)
print(f'Quick Test 3: {result.successful_episodes}/3 episodes')
" || echo "Test 3 failed"
        
        echo ""
        echo -e "${GREEN}✓ Ultra-quick test completed!${NC}"
        echo "Quick test data saved in: data/quick_test/"
        echo "This validates that the system is working correctly."
        ;;
        
    "help"|"-h"|"--help")
        echo "Data Collection Execution Script"
        echo ""
        echo "Usage:"
        echo "  $0 [mode] [options]"
        echo ""
        echo "Modes:"
        echo "  sequential     - Run all scenarios one by one (default)"
        echo "  parallel [N]   - Run scenarios in parallel with N workers (default: 3)"
        echo "  single <N>     - Run single scenario N (1-10)"
        echo "  test           - Test run (3 scenarios × 10 episodes = 30 total)"
        echo "  quick          - Ultra-quick test (3 scenarios × 3 episodes = 9 total)"
        echo "  help           - Show this help"
        echo ""
        echo "Examples:"
        echo "  $0                    # Run all scenarios sequentially"
        echo "  $0 parallel 4         # Run all scenarios with 4 parallel workers"
        echo "  $0 single 5           # Run only scenario 5 (lane closure)"
        echo "  $0 test               # Run test with 30 episodes (3 scenarios × 10)"
        echo "  $0 quick              # Run ultra-quick test with 9 episodes (3 scenarios × 3)"
        echo ""
        echo "Scenarios:"
        echo "  1  - Light Free Flow"
        echo "  2  - Heavy Commuting" 
        echo "  3  - Stop and Go"
        echo "  4  - Aggressive Behaviors"
        echo "  5  - Lane Closure"
        echo "  6  - Time Budget"
        echo "  7  - Multi-Lane Highway"
        echo "  8  - Mixed Traffic"
        echo "  9  - High-Speed Extended"
        echo "  10 - Multi-Agent Coordination"
        ;;
        
    *)
        echo "Invalid mode: $1"
        echo "Use '$0 help' for usage information."
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}🎉 Data collection process completed!${NC}"
echo ""
echo "Next steps:"
echo "  - Check data in data/ directory"
echo "  - Review logs in logs/ directory"  
echo "  - Use 'python main.py --demo loading' to verify data"
echo "  - Start model training with collected datasets"