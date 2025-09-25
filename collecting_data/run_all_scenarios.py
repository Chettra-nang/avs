#!/usr/bin/env python3
"""
Master Data Collection Script
Runs all 10 data collection scenarios sequentially or in parallel.
Collects 10,000 total episodes (1000 each) with complete multimodal data.
"""

import sys
import os
import time
import logging
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/master_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_scenario_script(script_path: Path) -> Tuple[str, bool, str]:
    """Run a single scenario data collection script."""
    script_name = script_path.name
    logger.info(f"Starting {script_name}")
    
    try:
        # Run the scenario script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=True,
            cwd=script_path.parent.parent
        )
        
        logger.info(f"✓ {script_name} completed successfully")
        return script_name, True, result.stdout
        
    except subprocess.CalledProcessError as e:
        error_msg = f"✗ {script_name} failed: {e.stderr}"
        logger.error(error_msg)
        return script_name, False, e.stderr


def run_sequential(scenario_scripts: List[Path]):
    """Run all scenario scripts sequentially."""
    logger.info("=" * 80)
    logger.info("SEQUENTIAL DATA COLLECTION - ALL 10 SCENARIOS")
    logger.info("=" * 80)
    logger.info(f"Total scenarios: {len(scenario_scripts)}")
    logger.info("Each scenario: 1000 episodes with complete multimodal data")
    logger.info("Total expected episodes: 10,000")
    
    start_time = time.time()
    results = []
    
    for i, script_path in enumerate(scenario_scripts, 1):
        logger.info(f"\nProgress: {i}/{len(scenario_scripts)} scenarios")
        logger.info("-" * 50)
        
        script_name, success, output = run_scenario_script(script_path)
        results.append((script_name, success, output))
        
        if success:
            logger.info(f"✓ Scenario {i} completed successfully")
        else:
            logger.error(f"✗ Scenario {i} failed")
    
    # Summary report
    total_time = time.time() - start_time
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    logger.info("=" * 80)
    logger.info("SEQUENTIAL COLLECTION COMPLETED")
    logger.info("=" * 80)
    logger.info(f"✓ Successful scenarios: {successful}/{len(scenario_scripts)}")
    logger.info(f"✗ Failed scenarios: {failed}")
    logger.info(f"⏱ Total collection time: {total_time/3600:.2f} hours")
    logger.info(f"📊 Expected total episodes: {successful * 1000}")
    
    return results


def run_parallel(scenario_scripts: List[Path], max_workers: int = 3):
    """Run scenario scripts in parallel."""
    logger.info("=" * 80)
    logger.info("PARALLEL DATA COLLECTION - ALL 10 SCENARIOS")
    logger.info("=" * 80)
    logger.info(f"Total scenarios: {len(scenario_scripts)}")
    logger.info(f"Parallel workers: {max_workers}")
    logger.info("Each scenario: 1000 episodes with complete multimodal data")
    logger.info("Total expected episodes: 10,000")
    
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_script = {
            executor.submit(run_scenario_script, script): script
            for script in scenario_scripts
        }
        
        # Process completed tasks
        for i, future in enumerate(as_completed(future_to_script), 1):
            script_path = future_to_script[future]
            
            try:
                script_name, success, output = future.result()
                results.append((script_name, success, output))
                
                if success:
                    logger.info(f"✓ Completed: {script_name} ({i}/{len(scenario_scripts)})")
                else:
                    logger.error(f"✗ Failed: {script_name} ({i}/{len(scenario_scripts)})")
                    
            except Exception as e:
                logger.error(f"✗ Exception in {script_path.name}: {e}")
                results.append((script_path.name, False, str(e)))
    
    # Summary report
    total_time = time.time() - start_time
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    logger.info("=" * 80)
    logger.info("PARALLEL COLLECTION COMPLETED")
    logger.info("=" * 80)
    logger.info(f"✓ Successful scenarios: {successful}/{len(scenario_scripts)}")
    logger.info(f"✗ Failed scenarios: {failed}")
    logger.info(f"⏱ Total collection time: {total_time/3600:.2f} hours")
    logger.info(f"📊 Expected total episodes: {successful * 1000}")
    
    return results


def main():
    """Main data collection orchestrator."""
    parser = argparse.ArgumentParser(description='Run all 10 data collection scenarios')
    parser.add_argument('--mode', choices=['sequential', 'parallel'], default='sequential',
                       help='Collection mode (default: sequential)')
    parser.add_argument('--workers', type=int, default=3,
                       help='Number of parallel workers (default: 3)')
    parser.add_argument('--scenarios', type=str, nargs='*',
                       help='Specific scenarios to run (default: all)')
    
    args = parser.parse_args()
    
    # Setup directories
    collecting_data_dir = Path(__file__).parent
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Find all scenario scripts
    all_scenario_scripts = sorted(collecting_data_dir.glob("scenario_*.py"))
    
    if args.scenarios:
        # Filter specific scenarios
        scenario_scripts = []
        for scenario_name in args.scenarios:
            script_path = collecting_data_dir / f"{scenario_name}.py"
            if script_path.exists():
                scenario_scripts.append(script_path)
            else:
                logger.warning(f"Scenario script not found: {script_path}")
    else:
        scenario_scripts = all_scenario_scripts
    
    if not scenario_scripts:
        logger.error("No scenario scripts found!")
        return 1
    
    logger.info(f"Found {len(scenario_scripts)} scenario scripts:")
    for script in scenario_scripts:
        logger.info(f"  - {script.name}")
    
    # Ensure data directory exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Run collection
    try:
        if args.mode == 'parallel':
            results = run_parallel(scenario_scripts, args.workers)
        else:
            results = run_sequential(scenario_scripts)
        
        # Save results summary
        summary_file = logs_dir / "collection_summary.log"
        with open(summary_file, 'w') as f:
            f.write("Data Collection Summary\\n")
            f.write("=" * 40 + "\\n")
            for script_name, success, output in results:
                status = "SUCCESS" if success else "FAILED"
                f.write(f"{script_name}: {status}\\n")
        
        logger.info(f"Collection summary saved to: {summary_file}")
        
        # Check if all succeeded
        successful = sum(1 for _, success, _ in results if success)
        if successful == len(scenario_scripts):
            logger.info("🎉 ALL SCENARIOS COMPLETED SUCCESSFULLY!")
            return 0
        else:
            logger.warning(f"⚠️  {len(scenario_scripts) - successful} scenarios failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Collection failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())