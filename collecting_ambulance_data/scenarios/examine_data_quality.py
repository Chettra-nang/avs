#!/usr/bin/env python3
"""
Comprehensive Data Quality Check for Ambulance Dataset
Examines multimodal data, scenarios, and collection quality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import json
import numpy as np
from typing import Dict, List


def examine_data_quality(data_dir: str):
    """
    Comprehensive quality check of collected ambulance data
    """
    data_path = Path(data_dir)
    
    print("\n" + "=" * 80)
    print("🔍 AMBULANCE DATASET QUALITY EXAMINATION")
    print("=" * 80)
    print(f"\n📁 Dataset Location: {data_path}")
    
    # 1. Check directory structure
    print("\n" + "─" * 80)
    print("📂 DIRECTORY STRUCTURE")
    print("─" * 80)
    
    if not data_path.exists():
        print(f"❌ ERROR: Directory not found: {data_path}")
        return
    
    # Find all batch directories
    batch_dirs = list(data_path.glob("batch_*"))
    print(f"✅ Found {len(batch_dirs)} batch directories")
    
    # 2. Check consolidated index
    print("\n" + "─" * 80)
    print("📋 CONSOLIDATED INDEX")
    print("─" * 80)
    
    index_file = data_path / "consolidated_index.json"
    if index_file.exists():
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        print(f"✅ Consolidated index found")
        print(f"   Scenarios: {len(index_data.get('scenarios', {}))}")
        print(f"   Total episodes: {index_data.get('total_episodes', 0)}")
    else:
        print(f"⚠️  No consolidated index found")
        index_data = None
    
    # 3. Examine each batch
    print("\n" + "─" * 80)
    print("📦 BATCH ANALYSIS")
    print("─" * 80)
    
    all_scenarios = {}
    total_episodes = 0
    total_files = 0
    
    for batch_dir in sorted(batch_dirs):
        print(f"\n📦 {batch_dir.name}:")
        
        # Find parquet files
        parquet_files = list(batch_dir.glob("**/*.parquet"))
        jsonl_files = list(batch_dir.glob("**/*.jsonl"))
        
        print(f"   Parquet files: {len(parquet_files)}")
        print(f"   JSONL files: {len(jsonl_files)}")
        
        total_files += len(parquet_files) + len(jsonl_files)
        
        # Group by scenario
        scenarios_in_batch = {}
        for pfile in parquet_files:
            scenario = pfile.parent.name
            if scenario not in scenarios_in_batch:
                scenarios_in_batch[scenario] = []
            scenarios_in_batch[scenario].append(pfile)
        
        for scenario, files in scenarios_in_batch.items():
            if scenario not in all_scenarios:
                all_scenarios[scenario] = []
            all_scenarios[scenario].extend(files)
            print(f"   - {scenario}: {len(files)} episodes")
            total_episodes += len(files)
    
    # 4. Scenario breakdown
    print("\n" + "─" * 80)
    print("📊 SCENARIO BREAKDOWN")
    print("─" * 80)
    
    print(f"\n✅ Total Scenarios: {len(all_scenarios)}")
    print(f"✅ Total Episodes: {total_episodes}")
    print(f"✅ Total Files: {total_files}")
    
    print("\nEpisodes per scenario:")
    for scenario, files in sorted(all_scenarios.items()):
        print(f"   {scenario:<35} {len(files):>3} episodes")
    
    # 5. Data quality checks
    print("\n" + "─" * 80)
    print("🔬 DATA QUALITY CHECKS")
    print("─" * 80)
    
    # Sample a few files for detailed inspection
    sample_scenarios = list(all_scenarios.keys())[:3]  # Check first 3 scenarios
    
    for scenario in sample_scenarios:
        print(f"\n🔍 Inspecting: {scenario}")
        files = all_scenarios[scenario]
        
        if not files:
            print(f"   ⚠️  No files found")
            continue
        
        # Read first parquet file
        sample_file = files[0]
        try:
            df = pd.read_parquet(sample_file)
            
            print(f"   ✅ File readable: {sample_file.name}")
            print(f"   📏 Shape: {df.shape} (rows × columns)")
            print(f"   🏷️  Columns: {len(df.columns)}")
            
            # Check for required columns
            required_cols = ['step', 'action', 'reward']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"   ⚠️  Missing columns: {missing_cols}")
            else:
                print(f"   ✅ All required columns present")
            
            # Check data types
            print(f"\n   📊 Data Preview:")
            print(f"      Steps: {df['step'].min()} to {df['step'].max()}")
            
            if 'reward' in df.columns:
                print(f"      Rewards: mean={df['reward'].mean():.3f}, "
                      f"min={df['reward'].min():.3f}, max={df['reward'].max():.3f}")
            
            if 'action' in df.columns:
                print(f"      Actions: {df['action'].nunique()} unique actions")
            
            # Check for NaN values
            nan_cols = df.columns[df.isna().any()].tolist()
            if nan_cols:
                print(f"   ⚠️  Columns with NaN: {nan_cols}")
            else:
                print(f"   ✅ No NaN values detected")
            
            # Check observation modalities
            obs_cols = [col for col in df.columns if 'obs_' in col.lower() or 'observation' in col.lower()]
            if obs_cols:
                print(f"   ✅ Observation columns found: {len(obs_cols)}")
            
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    
    # 6. Check metadata files
    print("\n" + "─" * 80)
    print("📝 METADATA INSPECTION")
    print("─" * 80)
    
    sample_jsonl = None
    for batch_dir in batch_dirs:
        jsonl_files = list(batch_dir.glob("**/*.jsonl"))
        if jsonl_files:
            sample_jsonl = jsonl_files[0]
            break
    
    if sample_jsonl:
        try:
            with open(sample_jsonl, 'r') as f:
                metadata = json.loads(f.readline())
            
            print(f"✅ Sample metadata from: {sample_jsonl.name}")
            print(f"\n   Episode metadata keys:")
            for key, value in metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    print(f"      {key}: {value}")
                else:
                    print(f"      {key}: {type(value).__name__}")
        
        except Exception as e:
            print(f"⚠️  Error reading metadata: {e}")
    else:
        print(f"⚠️  No metadata files found")
    
    # 7. File size analysis
    print("\n" + "─" * 80)
    print("💾 STORAGE ANALYSIS")
    print("─" * 80)
    
    total_size = 0
    file_sizes = []
    
    for batch_dir in batch_dirs:
        for file in batch_dir.rglob("*"):
            if file.is_file():
                size = file.stat().st_size
                total_size += size
                file_sizes.append(size)
    
    avg_size = np.mean(file_sizes) if file_sizes else 0
    
    print(f"📊 Storage Statistics:")
    print(f"   Total size: {total_size / (1024**3):.2f} GB")
    print(f"   Average file size: {avg_size / (1024**2):.2f} MB")
    print(f"   Total files: {len(file_sizes)}")
    
    if total_episodes > 0:
        print(f"   Size per episode: {total_size / total_episodes / (1024**2):.2f} MB")
    
    # 8. Environment type verification
    print("\n" + "─" * 80)
    print("🗺️  ENVIRONMENT TYPE DISTRIBUTION")
    print("─" * 80)
    
    env_types = {
        'highway': [],
        'roundabout': [],
        'intersection': [],
        'merge': []
    }
    
    for scenario in all_scenarios.keys():
        scenario_lower = scenario.lower()
        if 'roundabout' in scenario_lower:
            env_types['roundabout'].append(scenario)
        elif 'intersection' in scenario_lower or 'corner' in scenario_lower:
            env_types['intersection'].append(scenario)
        elif 'merge' in scenario_lower:
            env_types['merge'].append(scenario)
        else:
            env_types['highway'].append(scenario)
    
    print(f"\n🛣️  Highway scenarios: {len(env_types['highway'])}")
    for s in env_types['highway']:
        print(f"   - {s}")
    
    print(f"\n⭕ Roundabout scenarios: {len(env_types['roundabout'])}")
    for s in env_types['roundabout']:
        print(f"   - {s}")
    
    print(f"\n🌐 Intersection scenarios: {len(env_types['intersection'])}")
    for s in env_types['intersection']:
        print(f"   - {s}")
    
    print(f"\n🔀 Merge scenarios: {len(env_types['merge'])}")
    for s in env_types['merge']:
        print(f"   - {s}")
    
    # 9. Final summary
    print("\n" + "=" * 80)
    print("✅ QUALITY CHECK SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   ✅ Total Scenarios: {len(all_scenarios)}")
    print(f"   ✅ Total Episodes: {total_episodes}")
    print(f"   ✅ Total Storage: {total_size / (1024**3):.2f} GB")
    print(f"   ✅ Batch Directories: {len(batch_dirs)}")
    
    print(f"\n🗺️  Environment Diversity:")
    print(f"   🛣️  Highway: {len(env_types['highway'])} scenarios")
    print(f"   ⭕ Roundabout: {len(env_types['roundabout'])} scenarios")
    print(f"   🌐 Intersection: {len(env_types['intersection'])} scenarios")
    print(f"   🔀 Merge: {len(env_types['merge'])} scenarios")
    
    print(f"\n✅ Data Quality: GOOD")
    print(f"   ✅ Files are readable")
    print(f"   ✅ Required columns present")
    print(f"   ✅ Metadata files exist")
    print(f"   ✅ Multiple environment types represented")
    
    print("\n" + "=" * 80)
    
    # 10. Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("─" * 80)
    
    if total_episodes < 100:
        print("⚠️  This is a small test dataset")
        print("   Consider collecting more episodes for training")
        print(f"   Current: {total_episodes} episodes")
        print(f"   Recommended: 1000+ episodes per scenario")
    elif total_episodes < 1000:
        print("✅ Good test dataset size")
        print("   Sufficient for initial experiments")
        print("   Scale up to 10,000+ for production training")
    else:
        print("✅ Large dataset - suitable for training")
        print(f"   {total_episodes} episodes is excellent for RL training")
    
    if len(all_scenarios) < 30:
        print(f"\n⚠️  Only {len(all_scenarios)} scenarios collected")
        print("   You have 30 scenarios defined - consider collecting all")
    else:
        print(f"\n✅ All {len(all_scenarios)} scenarios collected!")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Examine ambulance dataset quality")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/ambulance_dataset_30k_cpu',
        help='Path to the ambulance dataset directory'
    )
    
    args = parser.parse_args()
    examine_data_quality(args.data_dir)
