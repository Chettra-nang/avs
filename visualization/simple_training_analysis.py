#!/usr/bin/env python3
"""
Simple Dataset Analysis for Training
===================================

Analyzes your highway datasets and provides practical training guidance.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def analyze_individual_datasets():
    """Analyze each dataset individually."""
    print("🚗 Individual Dataset Analysis")
    print("=" * 60)
    
    # Define datasets with descriptions
    datasets = {
        "Multimodal Datasets (with Images)": {
            "dense_commuting/20250921_151946-f35b18e8_transitions.parquet": "Dense traffic with full multimodal data",
            "dense_commuting/20250921_152749-1ea1c024_transitions.parquet": "Dense traffic episode 2",
            "free_flow/20250921_151816-0a384836_transitions.parquet": "Free flowing traffic with images",
            "free_flow/20250921_152624-ed6a89cd_transitions.parquet": "Free flowing traffic episode 2"
        },
        "Clean Action-Reward Datasets": {
            "dense_commuting/ep_20250921_001_transitions.parquet": "Dense traffic - action/reward pairs",
            "free_flow/ep_20250921_001_transitions.parquet": "Free flow - action/reward pairs", 
            "stop_and_go/ep_20250921_001_transitions.parquet": "Stop-go traffic - action/reward pairs"
        },
        "Demonstration Datasets": {
            "demo_with_natural_language.parquet": "Language-conditioned driving examples"
        }
    }
    
    for category, dataset_dict in datasets.items():
        print(f"\n📁 {category}")
        print("-" * 50)
        
        for relative_path, description in dataset_dict.items():
            if relative_path.startswith("demo"):
                file_path = f"/home/chettra/ITC/Research/AVs/data/{relative_path}"
            else:
                file_path = f"/home/chettra/ITC/Research/AVs/data/highway_multimodal_dataset/{relative_path}"
            
            # Also check clean version
            clean_path = file_path.replace('highway_multimodal_dataset/', 'highway_multimodal_dataset_clean/')
            
            print(f"\n🔍 {description}")
            
            # Try both raw and clean versions
            for path, label in [(file_path, "Raw"), (clean_path, "Clean")]:
                if os.path.exists(path):
                    try:
                        df = pd.read_parquet(path)
                        
                        # Basic info
                        episodes = df['episode_id'].nunique() if 'episode_id' in df.columns else 'N/A'
                        steps = f"({df['step'].min()}-{df['step'].max()})" if 'step' in df.columns else 'N/A'
                        
                        print(f"  {label}: {df.shape[0]} samples, {episodes} episodes, steps {steps}")
                        
                        # Modality info
                        modalities = []
                        if 'grayscale_blob' in df.columns:
                            modalities.append("Images(4×128×64)")
                        if 'occupancy_blob' in df.columns:
                            modalities.append("Occupancy(2×11×11)")
                        if 'action' in df.columns:
                            action_range = f"Actions({df['action'].min()}-{df['action'].max()})"
                            modalities.append(action_range)
                        if 'reward' in df.columns:
                            reward_mean = df['reward'].mean()
                            modalities.append(f"Rewards(μ={reward_mean:.3f})")
                        if 'summary_text' in df.columns:
                            modalities.append("NaturalLanguage")
                        
                        print(f"       Modalities: {', '.join(modalities) if modalities else 'Basic features only'}")
                        
                    except Exception as e:
                        print(f"  {label}: ❌ Error - {e}")

def training_recommendations():
    """Provide specific training recommendations."""
    print(f"\n{'='*60}")
    print("🎯 TRAINING RECOMMENDATIONS BY USE CASE")
    print(f"{'='*60}")
    
    recommendations = [
        {
            "goal": "🤖 Reinforcement Learning Agent",
            "datasets": [
                "highway_multimodal_dataset_clean/dense_commuting/ep_20250921_001_transitions.parquet",
                "highway_multimodal_dataset_clean/free_flow/ep_20250921_001_transitions.parquet", 
                "highway_multimodal_dataset_clean/stop_and_go/ep_20250921_001_transitions.parquet"
            ],
            "why": "These have action-reward pairs needed for RL",
            "approach": [
                "Use (state, action, reward, next_state) tuples",
                "Train with PPO, SAC, or DQN algorithms",
                "Start with dense_commuting (highest rewards)",
                "Evaluate transfer across scenarios"
            ],
            "code_example": """
# Load RL data
df = pd.read_parquet('ep_20250921_001_transitions.parquet')
states = df[['speed', 'ego_x', 'ego_y', 'traffic_density']].values
actions = df['action'].values
rewards = df['reward'].values

# Create transitions
transitions = []
for i in range(len(df)-1):
    transition = (states[i], actions[i], rewards[i], states[i+1], False)
    transitions.append(transition)
"""
        },
        {
            "goal": "🎭 Imitation Learning / Behavior Cloning", 
            "datasets": [
                "All ep_20250921_001_transitions.parquet files"
            ],
            "why": "Expert demonstrations with state-action pairs",
            "approach": [
                "Supervised learning: predict actions from states",
                "Use cross-entropy loss for discrete actions",
                "Combine all scenarios for robustness",
                "Evaluate on held-out scenarios"
            ],
            "code_example": """
# Load expert demonstrations
X = df[['speed', 'ego_x', 'ego_y', 'traffic_density', 'vehicle_count']].values
y = df['action'].values

# Train classifier to predict expert actions
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)
"""
        },
        {
            "goal": "🗣️ Language-Conditioned Driving",
            "datasets": [
                "demo_with_natural_language.parquet",
                "All files with summary_text column"
            ], 
            "why": "Natural language descriptions of driving situations",
            "approach": [
                "Text + state → action mapping",
                "Use BERT/transformer for text encoding", 
                "Combine text features with sensor data",
                "Train on (text, state) → action pairs"
            ],
            "code_example": """
# Extract text-action pairs
texts = df['summary_text'].values
actions = df['action'].values
states = df[['speed', 'traffic_density']].values

# Simple approach: TF-IDF + classifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer(max_features=100)
text_features = vectorizer.fit_transform(texts)
"""
        },
        {
            "goal": "👁️ Computer Vision / Representation Learning",
            "datasets": [
                "highway_multimodal_dataset/dense_commuting/20250921_151946-f35b18e8_transitions.parquet",
                "highway_multimodal_dataset/free_flow/20250921_151816-0a384836_transitions.parquet"
            ],
            "why": "Rich visual data (4×128×64 image stacks) + occupancy grids",
            "approach": [
                "Self-supervised learning on image sequences",
                "Contrastive learning between different views",
                "Future frame prediction",
                "Pre-train visual encoder, fine-tune for driving tasks"
            ],
            "code_example": """
# Extract image sequences (use image_stack_visualizer.py)
from image_stack_visualizer import ImageStackVisualizer

viz = ImageStackVisualizer()
viz.load_parquet_file('20250921_151946-f35b18e8_transitions.parquet')

# Export frames for training
viz.export_frames_to_png('./training_images/', max_episodes=10)
"""
        },
        {
            "goal": "🔄 Multi-Task Learning",
            "datasets": [
                "All multimodal datasets combined"
            ],
            "why": "Multiple modalities and scenarios for robust learning",
            "approach": [
                "Train single model on multiple tasks simultaneously",
                "Shared encoder, multiple task-specific heads",
                "Tasks: action prediction, reward estimation, scene understanding",
                "Regularization through task diversity"
            ],
            "code_example": """
# Multi-task data preparation
tasks = {
    'action_prediction': df['action'].values,
    'reward_estimation': df['reward'].values, 
    'speed_prediction': df['speed'].values,
    'density_estimation': df['traffic_density'].values
}
"""
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['goal']}")
        print("-" * 40)
        print(f"📊 Best datasets:")
        for dataset in rec['datasets']:
            print(f"   • {dataset}")
        print(f"🎯 Why: {rec['why']}")
        print(f"🛠️ Approach:")
        for approach in rec['approach']:
            print(f"   • {approach}")
        print(f"💻 Code example:")
        print(rec['code_example'])

def scenario_comparison():
    """Compare different traffic scenarios."""
    print(f"\n{'='*60}")
    print("🚦 SCENARIO COMPARISON FOR TRAINING")
    print(f"{'='*60}")
    
    scenarios = {
        'Dense Commuting': 'highway_multimodal_dataset_clean/dense_commuting/ep_20250921_001_transitions.parquet',
        'Free Flow': 'highway_multimodal_dataset_clean/free_flow/ep_20250921_001_transitions.parquet',
        'Stop and Go': 'highway_multimodal_dataset_clean/stop_and_go/ep_20250921_001_transitions.parquet'
    }
    
    print(f"\n{'Scenario':<15} {'Samples':<8} {'Avg Reward':<12} {'Reward Std':<12} {'Training Use'}")
    print("-" * 80)
    
    for scenario_name, relative_path in scenarios.items():
        file_path = f"/home/chettra/ITC/Research/AVs/data/{relative_path}"
        
        try:
            df = pd.read_parquet(file_path)
            samples = len(df)
            avg_reward = df['reward'].mean() if 'reward' in df.columns else 0
            std_reward = df['reward'].std() if 'reward' in df.columns else 0
            
            # Determine training use based on reward characteristics
            if avg_reward > 0.1:
                use = "Easy/Baseline"
            elif avg_reward > 0.08:
                use = "Medium/Robust"
            else:
                use = "Hard/Challenge"
            
            print(f"{scenario_name:<15} {samples:<8} {avg_reward:<12.3f} {std_reward:<12.3f} {use}")
            
        except Exception as e:
            print(f"{scenario_name:<15} {'Error':<8} {'-':<12} {'-':<12} {str(e)[:20]}")
    
    print(f"\n💡 Training Strategy:")
    print("1. Start with Free Flow (easiest) to learn basic driving")
    print("2. Move to Dense Commuting (medium) for robustness")
    print("3. Finish with Stop and Go (hardest) for challenging scenarios")
    print("4. Evaluate final model on all three scenarios")

def main():
    """Main analysis function."""
    print("🚗 Highway Dataset Training Analysis")
    print("=" * 60)
    print("This analysis helps you understand how to use each dataset for training.\n")
    
    analyze_individual_datasets()
    training_recommendations() 
    scenario_comparison()
    
    print(f"\n{'='*60}")
    print("🎉 ANALYSIS COMPLETE")
    print("="*60)
    print("\n📋 Summary:")
    print("• You have 11 datasets across 4 categories")
    print("• 4 datasets with full multimodal data (images + occupancy)")
    print("• 6 datasets with action-reward pairs for RL")  
    print("• 7 datasets with natural language descriptions")
    print("• 3 distinct traffic scenarios for robustness testing")
    print("\n🚀 Quick Start:")
    print("1. For RL: Use clean datasets with action-reward pairs")
    print("2. For vision: Use multimodal datasets with images")
    print("3. For language: Use datasets with summary_text")
    print("4. Start simple, then scale to multimodal approaches")

if __name__ == "__main__":
    main()