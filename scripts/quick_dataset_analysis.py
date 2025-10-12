#!/usr/bin/env python3
"""
Quick Dataset Quality Analysis for Batch-Structured Data

Analyzes ambulance dataset organized in batch directories.

Usage:
    python3 scripts/quick_dataset_analysis.py --data-dir data/ambulance_dataset_diagnose
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from glob import glob


def analyze_batch_dataset(data_dir: str):
    """Analyze dataset organized in batch directories."""
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Directory not found: {data_path}")
        return
    
    # Find all parquet files
    parquet_files = list(data_path.glob('*/*/*.parquet'))
    parquet_files += list(data_path.glob('*/*.parquet'))
    parquet_files += list(data_path.glob('*.parquet'))
    
    if not parquet_files:
        print(f"❌ No parquet files found in {data_dir}")
        return
    
    print(f"📂 Found {len(parquet_files)} parquet files")
    print(f"📊 Loading data...")
    print()
    
    # Load all data
    dfs = []
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️  Skipped {f.name}: {e}")
    
    if not dfs:
        print("❌ No data loaded")
        return
    
    # Combine
    df = pd.concat(dfs, ignore_index=True)
    
    print("=" * 70)
    print("📊 DATASET QUALITY ANALYSIS")
    print("=" * 70)
    
    # Basic statistics
    print(f"\n📈 Dataset Size:")
    print(f"  Total Transitions: {len(df):,}")
    
    # Check if we have episode column
    if 'episode' in df.columns:
        n_episodes = df['episode'].nunique()
        print(f"  Total Episodes: {n_episodes:,}")
        
        # Episode statistics
        print(f"\n📏 Episode Length Distribution:")
        episodes = df.groupby('episode')
        lengths = episodes.size()
        print(f"  Mean: {lengths.mean():.1f} steps")
        print(f"  Median: {lengths.median():.1f} steps")
        print(f"  Std: {lengths.std():.1f} steps")
        print(f"  Min: {lengths.min()} steps")
        print(f"  Max: {lengths.max()} steps")
        print(f"  25th percentile: {lengths.quantile(0.25):.1f} steps")
        print(f"  75th percentile: {lengths.quantile(0.75):.1f} steps")
        
        # Episode completion rate
        print(f"\n✅ Episode Completion:")
        complete_episodes = (lengths >= 95).sum()
        long_episodes = (lengths >= 80).sum()
        medium_episodes = (lengths >= 50).sum()
        short_episodes = (lengths < 50).sum()
        
        print(f"  Complete (≥95 steps): {complete_episodes:,} ({100*complete_episodes/len(lengths):.1f}%)")
        print(f"  Long (≥80 steps): {long_episodes:,} ({100*long_episodes/len(lengths):.1f}%)")
        print(f"  Medium (50-79 steps): {medium_episodes-long_episodes:,} ({100*(medium_episodes-long_episodes)/len(lengths):.1f}%)")
        print(f"  Short (<50 steps): {short_episodes:,} ({100*short_episodes/len(lengths):.1f}%)")
    else:
        print("  ⚠️  No episode column found")
    
    # Reward statistics
    if 'reward' in df.columns:
        print(f"\n💰 Reward Distribution:")
        print(f"  Mean: {df['reward'].mean():.3f}")
        print(f"  Median: {df['reward'].median():.3f}")
        print(f"  Std: {df['reward'].std():.3f}")
        print(f"  Min: {df['reward'].min():.3f}")
        print(f"  Max: {df['reward'].max():.3f}")
        
        # Analyze crashes
        crash_transitions = (df['reward'] < -0.5).sum()
        negative_rewards = (df['reward'] < 0).sum()
        positive_rewards = (df['reward'] > 0).sum()
        zero_rewards = (df['reward'] == 0).sum()
        
        print(f"\n💥 Crash Analysis:")
        print(f"  Severe crashes (reward < -0.5): {crash_transitions:,} ({100*crash_transitions/len(df):.2f}%)")
        print(f"  Negative rewards (< 0): {negative_rewards:,} ({100*negative_rewards/len(df):.2f}%)")
        print(f"  Positive rewards (> 0): {positive_rewards:,} ({100*positive_rewards/len(df):.2f}%)")
        print(f"  Zero rewards: {zero_rewards:,} ({100*zero_rewards/len(df):.2f}%)")
        
        # Episode-level rewards
        if 'episode' in df.columns:
            episode_rewards = episodes['reward'].sum()
            print(f"\n📊 Episode Cumulative Rewards:")
            print(f"  Mean: {episode_rewards.mean():.2f}")
            print(f"  Median: {episode_rewards.median():.2f}")
            print(f"  Std: {episode_rewards.std():.2f}")
            print(f"  Min: {episode_rewards.min():.2f}")
            print(f"  Max: {episode_rewards.max():.2f}")
            
            good_episodes = (episode_rewards > 0).sum()
            bad_episodes = (episode_rewards < 0).sum()
            print(f"  Good episodes (total reward > 0): {good_episodes:,} ({100*good_episodes/len(episode_rewards):.1f}%)")
            print(f"  Bad episodes (total reward < 0): {bad_episodes:,} ({100*bad_episodes/len(episode_rewards):.1f}%)")
    
    # Action distribution
    if 'action' in df.columns:
        print(f"\n🎮 Action Distribution:")
        action_counts = df['action'].value_counts().sort_index()
        for action, count in action_counts.items():
            print(f"  Action {action}: {count:,} ({100*count/len(df):.1f}%)")
    
    # Dataset quality assessment
    if 'episode' in df.columns and 'reward' in df.columns:
        print("\n" + "=" * 70)
        print("🎯 DATASET QUALITY ASSESSMENT")
        print("=" * 70)
        
        completion_rate = complete_episodes / len(lengths)
        positive_reward_rate = positive_rewards / len(df)
        avg_episode_reward = episode_rewards.mean()
        
        print(f"\nQuality Metrics:")
        print(f"  ✓ Completion Rate: {completion_rate:.1%} (target: >70%)")
        print(f"  ✓ Positive Reward Rate: {positive_reward_rate:.1%} (target: >60%)")
        print(f"  ✓ Avg Episode Reward: {avg_episode_reward:.2f} (target: >20)")
        
        # Overall assessment
        print(f"\n📋 Overall Assessment:")
        if completion_rate > 0.7 and positive_reward_rate > 0.6 and avg_episode_reward > 20:
            print("  🟢 GOOD QUALITY - Dataset should support decent offline RL")
            print("     Expected performance: 30-50% success rate")
        elif completion_rate > 0.5 and positive_reward_rate > 0.4 and avg_episode_reward > 10:
            print("  🟡 MEDIUM QUALITY - Dataset has mixed demonstrations")
            print("     Expected performance: 10-30% success rate")
            print("     Recommendation: Filter high-quality episodes or collect more data")
        else:
            print("  🔴 LOW QUALITY - Dataset contains many crashes/poor demonstrations")
            print("     Expected performance: 0-10% success rate")
            print("     Recommendation: Collect better data with longer episodes")
        
        print("\n" + "=" * 70)
        
        # Recommendations
        print("\n💡 Recommendations:")
        
        if completion_rate < 0.7:
            print(f"\n1. ⚠️  Low completion rate ({completion_rate:.1%})")
            print("   → Collect with --max-steps 200 for longer episodes")
            print("   → Or filter: keep only episodes with length ≥ 80")
        
        if positive_reward_rate < 0.6:
            print(f"\n2. ⚠️  Many negative rewards ({100-positive_reward_rate*100:.1f}% negative)")
            print("   → Filter: keep only episodes with mean reward > 0")
            print("   → Or improve collection policy")
        
        if avg_episode_reward < 20:
            print(f"\n3. ⚠️  Low average episode reward ({avg_episode_reward:.1f})")
            print("   → Collect from better policy (not random)")
            print("   → Or use online RL to fine-tune")
    
    print("\n✅ Analysis complete!")


def main():
    parser = argparse.ArgumentParser(description='Quick dataset quality analysis')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset directory')
    
    args = parser.parse_args()
    
    analyze_batch_dataset(args.data_dir)


if __name__ == '__main__':
    main()
