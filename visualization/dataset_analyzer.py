#!/usr/bin/env python3
"""
Dataset Analyzer for Training Purposes
======================================

Analyzes all available datasets and provides training recommendations.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json

def analyze_dataset(file_path):
    """Analyze a single dataset file."""
    try:
        df = pd.read_parquet(file_path)
        
        analysis = {
            'file_path': file_path,
            'file_name': Path(file_path).name,
            'shape': df.shape,
            'columns': list(df.columns),
            'episodes': df['episode_id'].nunique() if 'episode_id' in df.columns else 'N/A',
            'steps_range': (int(df['step'].min()), int(df['step'].max())) if 'step' in df.columns else 'N/A',
            'agents': df['agent_id'].nunique() if 'agent_id' in df.columns else 'N/A',
            'scenarios': list(df['scenario'].unique())[:5] if 'scenario' in df.columns else 'N/A',  # Limit scenarios
            'has_images': 'grayscale_blob' in df.columns,
            'has_occupancy': 'occupancy_blob' in df.columns,
            'has_actions': 'action' in df.columns,
            'has_rewards': 'reward' in df.columns,
            'has_natural_language': 'summary_text' in df.columns,
            'memory_size_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Additional analysis for specific columns
        if 'action' in df.columns:
            actions = df['action'].dropna()
            if len(actions) > 0:
                if isinstance(actions.iloc[0], str):
                    analysis['action_types'] = list(actions.unique())[:10]  # First 10
                else:
                    analysis['action_range'] = (float(actions.min()), float(actions.max()))
        
        if 'reward' in df.columns:
            rewards = df['reward'].dropna()
            if len(rewards) > 0:
                analysis['reward_stats'] = {
                    'mean': float(rewards.mean()),
                    'std': float(rewards.std()),
                    'min': float(rewards.min()),
                    'max': float(rewards.max())
                }
        
        if 'summary_text' in df.columns:
            text_data = df['summary_text'].dropna()
            if len(text_data) > 0:
                analysis['text_samples'] = list(text_data.head(3))
                analysis['avg_text_length'] = float(text_data.str.len().mean())
        
        return analysis
        
    except Exception as e:
        return {
            'file_path': file_path,
            'error': str(e)
        }

def find_all_datasets(data_root="/home/chettra/ITC/Research/AVs/data"):
    """Find all dataset files."""
    datasets = []
    
    # Find Parquet files
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.parquet'):
                datasets.append(os.path.join(root, file))
    
    return sorted(datasets)

def categorize_datasets(analyses):
    """Categorize datasets by their training purpose."""
    categories = {
        'multimodal_learning': [],  # Has images + other modalities
        'language_conditioned': [],  # Has natural language
        'reinforcement_learning': [],  # Has actions + rewards
        'behavior_cloning': [],  # Has actions but no rewards
        'representation_learning': [],  # Has images but minimal actions
        'scenario_based': []  # Different traffic scenarios
    }
    
    for analysis in analyses:
        if 'error' in analysis:
            continue
            
        # Multimodal learning (images + other rich features)
        if analysis['has_images'] and analysis['has_occupancy']:
            categories['multimodal_learning'].append(analysis)
        
        # Language-conditioned learning
        if analysis['has_natural_language']:
            categories['language_conditioned'].append(analysis)
        
        # Reinforcement learning
        if analysis['has_actions'] and analysis['has_rewards']:
            categories['reinforcement_learning'].append(analysis)
        
        # Behavior cloning
        elif analysis['has_actions'] and not analysis['has_rewards']:
            categories['behavior_cloning'].append(analysis)
        
        # Representation learning
        elif analysis['has_images'] and not analysis['has_actions']:
            categories['representation_learning'].append(analysis)
        
        # Scenario-based learning
        if analysis['scenarios'] != 'N/A' and len(analysis['scenarios']) > 1:
            categories['scenario_based'].append(analysis)
    
    return categories

def main():
    print("🚗 Highway Dataset Analysis for Training")
    print("=" * 60)
    
    # Find all datasets
    datasets = find_all_datasets()
    print(f"Found {len(datasets)} dataset files")
    
    # Analyze each dataset
    print("\n📊 Analyzing datasets...")
    analyses = []
    for dataset in datasets:
        print(f"  Analyzing {Path(dataset).name}...")
        analysis = analyze_dataset(dataset)
        analyses.append(analysis)
    
    # Categorize for training purposes
    categories = categorize_datasets(analyses)
    
    print(f"\n{'='*60}")
    print("DATASET ANALYSIS FOR TRAINING")
    print(f"{'='*60}")
    
    # Print detailed analysis for each dataset
    for i, analysis in enumerate(analyses, 1):
        if 'error' in analysis:
            print(f"\n❌ {i}. {analysis['file_path']}")
            print(f"   Error: {analysis['error']}")
            continue
            
        print(f"\n📁 {i}. {analysis['file_name']}")
        print(f"   Path: {analysis['file_path']}")
        print(f"   Shape: {analysis['shape']} ({analysis['memory_size_mb']:.1f} MB)")
        print(f"   Episodes: {analysis['episodes']}, Steps: {analysis['steps_range']}")
        print(f"   Agents: {analysis['agents']}")
        print(f"   Scenarios: {analysis['scenarios']}")
        
        # Data modalities
        modalities = []
        if analysis['has_images']: modalities.append('Images')
        if analysis['has_occupancy']: modalities.append('Occupancy')
        if analysis['has_actions']: modalities.append('Actions')
        if analysis['has_rewards']: modalities.append('Rewards')
        if analysis['has_natural_language']: modalities.append('Natural Language')
        print(f"   Modalities: {', '.join(modalities) if modalities else 'None'}")
        
        # Training-specific info
        if 'action_types' in analysis:
            print(f"   Action types: {analysis['action_types']}")
        elif 'action_range' in analysis:
            print(f"   Action range: {analysis['action_range']}")
            
        if 'reward_stats' in analysis:
            stats = analysis['reward_stats']
            print(f"   Rewards: μ={stats['mean']:.3f}, σ={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")
        
        if 'avg_text_length' in analysis:
            print(f"   Avg text length: {analysis['avg_text_length']:.1f} chars")
            if 'text_samples' in analysis:
                print(f"   Text sample: \"{analysis['text_samples'][0][:100]}...\"")
    
    # Training recommendations
    print(f"\n{'='*60}")
    print("TRAINING RECOMMENDATIONS BY CATEGORY")
    print(f"{'='*60}")
    
    for category, datasets in categories.items():
        if not datasets:
            continue
            
        print(f"\n🎯 {category.upper().replace('_', ' ')} ({len(datasets)} datasets)")
        print("-" * 50)
        
        for dataset in datasets:
            print(f"   • {dataset['file_name']}")
        
        # Specific training recommendations
        if category == 'multimodal_learning':
            print("\n   💡 Training Approach:")
            print("   - Use for multimodal fusion architectures")
            print("   - Combine visual features (CNN) + spatial features (occupancy)")
            print("   - Good for representation learning and feature extraction")
            print("   - Can train encoders for downstream tasks")
            
        elif category == 'language_conditioned':
            print("\n   💡 Training Approach:")
            print("   - Language-conditioned driving policies")
            print("   - Natural language instruction following")
            print("   - Vision-language models for driving")
            print("   - Text-to-action generation")
            
        elif category == 'reinforcement_learning':
            print("\n   💡 Training Approach:")
            print("   - Direct policy optimization (PPO, SAC, etc.)")
            print("   - Value function estimation")
            print("   - Reward modeling and inverse RL")
            print("   - Multi-agent RL if multiple agents present")
            
        elif category == 'behavior_cloning':
            print("\n   💡 Training Approach:")
            print("   - Supervised learning from expert demonstrations")
            print("   - Imitation learning algorithms (BC, DAgger)")
            print("   - Action prediction from state observations")
            print("   - Good baseline for comparison with RL methods")
            
        elif category == 'representation_learning':
            print("\n   💡 Training Approach:")
            print("   - Self-supervised visual representation learning")
            print("   - Autoencoder architectures")
            print("   - Contrastive learning methods")
            print("   - Pre-training for downstream tasks")
            
        elif category == 'scenario_based':
            print("\n   💡 Training Approach:")
            print("   - Scenario-specific policy learning")
            print("   - Transfer learning across scenarios")
            print("   - Domain adaptation methods")
            print("   - Meta-learning for quick adaptation")

if __name__ == "__main__":
    main()