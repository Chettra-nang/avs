#!/usr/bin/env python3
"""
Analyze Dataset Quality

Checks the quality of collected ambulance dataset to understand
why offline RL performance is low.

Usage:
    python3 scripts/analyze_dataset_quality.py --data-dir data/ambulance_dataset_30k_cpu
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze_dataset(data_dir: str):
    """Analyze dataset quality metrics."""
    
    data_path = Path(data_dir) / 'combined_ambulance_data.parquet'
    
    if not data_path.exists():
        print(f"❌ Dataset not found: {data_path}")
        print(f"\nSearching for parquet files in {data_dir}...")
        parquet_files = list(Path(data_dir).glob('*.parquet'))
        if parquet_files:
            print(f"Found {len(parquet_files)} parquet files:")
            for f in parquet_files[:5]:
                print(f"  - {f.name}")
            if len(parquet_files) > 5:
                print(f"  ... and {len(parquet_files)-5} more")
            data_path = parquet_files[0]
            print(f"\n✅ Using: {data_path.name}")
        else:
            print(f"❌ No parquet files found in {data_dir}")
            return
    
    print(f"📂 Loading dataset: {data_path}")
    df = pd.read_parquet(data_path)
    
    print("\n" + "=" * 70)
    print("📊 DATASET QUALITY ANALYSIS")
    print("=" * 70)
    
    # Basic statistics
    print(f"\n📈 Dataset Size:")
    print(f"  Total Transitions: {len(df):,}")
    print(f"  Total Episodes: {df['episode'].nunique():,}")
    print(f"  Columns: {', '.join(df.columns[:10])}...")
    
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
    complete_episodes = (lengths >= 95).sum()  # Near max_steps=100
    long_episodes = (lengths >= 80).sum()
    medium_episodes = (lengths >= 50).sum()
    short_episodes = (lengths < 50).sum()
    
    print(f"  Complete (≥95 steps): {complete_episodes:,} ({100*complete_episodes/len(lengths):.1f}%)")
    print(f"  Long (≥80 steps): {long_episodes:,} ({100*long_episodes/len(lengths):.1f}%)")
    print(f"  Medium (50-79 steps): {medium_episodes-long_episodes:,} ({100*(medium_episodes-long_episodes)/len(lengths):.1f}%)")
    print(f"  Short (<50 steps): {short_episodes:,} ({100*short_episodes/len(lengths):.1f}%)")
    
    # Reward statistics
    print(f"\n💰 Reward Distribution:")
    print(f"  Mean: {df['reward'].mean():.3f}")
    print(f"  Median: {df['reward'].median():.3f}")
    print(f"  Std: {df['reward'].std():.3f}")
    print(f"  Min: {df['reward'].min():.3f}")
    print(f"  Max: {df['reward'].max():.3f}")
    
    # Analyze crashes (negative rewards typically indicate collision)
    crash_transitions = (df['reward'] < -0.5).sum()
    negative_rewards = (df['reward'] < 0).sum()
    positive_rewards = (df['reward'] > 0).sum()
    zero_rewards = (df['reward'] == 0).sum()
    
    print(f"\n💥 Crash Analysis:")
    print(f"  Severe crashes (reward < -0.5): {crash_transitions:,} ({100*crash_transitions/len(df):.2f}%)")
    print(f"  Negative rewards (< 0): {negative_rewards:,} ({100*negative_rewards/len(df):.2f}%)")
    print(f"  Positive rewards (> 0): {positive_rewards:,} ({100*positive_rewards/len(df):.2f}%)")
    print(f"  Zero rewards: {zero_rewards:,} ({100*zero_rewards/len(df):.2f}%)")
    
    # Episode-level reward analysis
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
    print("\n" + "=" * 70)
    print("🎯 DATASET QUALITY ASSESSMENT")
    print("=" * 70)
    
    # Calculate quality score
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
    
    # Suggest improvements
    print("\n💡 Recommendations:")
    
    if completion_rate < 0.7:
        print(f"\n1. ⚠️  Low completion rate ({completion_rate:.1%})")
        print("   → Collect with --max-steps 200 for longer episodes")
        print("   → Or filter: keep only episodes with length ≥ 80")
    
    if positive_reward_rate < 0.6:
        print(f"\n2. ⚠️  Many negative rewards ({100-positive_reward_rate*100:.1f}% negative)")
        print("   → Filter: keep only episodes with mean reward > 0")
        print("   → Or improve collection policy (use better agent)")
    
    if avg_episode_reward < 20:
        print(f"\n3. ⚠️  Low average episode reward ({avg_episode_reward:.1f})")
        print("   → Collect from better policy (not random)")
        print("   → Or use online RL to fine-tune from this checkpoint")
    
    print("\n✅ Analysis complete!")
    
    return df, lengths, episode_rewards


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset quality')
    parser.add_argument('--data-dir', type=str, 
                       default='data/ambulance_dataset_30k_cpu',
                       help='Path to dataset directory')
    
    args = parser.parse_args()
    
    analyze_dataset(args.data_dir)


if __name__ == '__main__':
    main()
