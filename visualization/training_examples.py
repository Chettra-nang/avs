#!/usr/bin/env python3
"""
Training Examples for Highway Datasets
=====================================

Practical examples showing how to train different types of models
using your highway simulation datasets.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
sys.path.append('.')
from image_stack_visualizer import ImageStackVisualizer

class HighwayDataset(Dataset):
    """PyTorch Dataset for highway multimodal data."""
    
    def __init__(self, parquet_file, use_images=True, use_text=True):
        self.df = pd.read_parquet(parquet_file)
        self.use_images = use_images and 'grayscale_blob' in self.df.columns
        self.use_text = use_text and 'summary_text' in self.df.columns
        self.viz = ImageStackVisualizer()
        
        print(f"Loaded dataset: {os.path.basename(parquet_file)}")
        print(f"  Samples: {len(self.df)}")
        print(f"  Features: Images={self.use_images}, Text={self.use_text}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Basic features (always available)
        features = {
            'speed': float(row.get('speed', 0)),
            'ego_x': float(row.get('ego_x', 0)),
            'ego_y': float(row.get('ego_y', 0)),
            'traffic_density': float(row.get('traffic_density', 0)),
            'action': int(row.get('action', 0)) if 'action' in row else 0,
            'reward': float(row.get('reward', 0)) if 'reward' in row else 0.0,
        }
        
        # Image features
        if self.use_images:
            img_stack = self.viz.decode_blob(row, "grayscale_blob", "grayscale_shape", "grayscale_dtype")
            if img_stack is not None:
                # Normalize to [0, 1] and convert to torch tensor
                features['images'] = torch.FloatTensor(img_stack / 255.0)
            else:
                features['images'] = torch.zeros(4, 128, 64)
        
        # Text features (simple token count for now)
        if self.use_text and 'summary_text' in row:
            text = str(row['summary_text'])
            features['text_length'] = len(text.split())
            features['text'] = text
        
        return features

# Training Examples

def train_behavior_cloning_model():
    """Example 1: Behavior Cloning (Supervised Learning)"""
    print("🎯 Example 1: Behavior Cloning Training")
    print("=" * 50)
    
    # Simple MLP model for behavior cloning
    class BehaviorCloningModel(nn.Module):
        def __init__(self, input_dim=4, num_actions=5):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_actions)
            )
        
        def forward(self, x):
            return self.fc(x)
    
    # Load clean dataset (faster for demo)
    dataset_path = "/home/chettra/ITC/Research/AVs/data/highway_multimodal_dataset_clean/dense_commuting/ep_20250921_001_transitions.parquet"
    
    try:
        dataset = HighwayDataset(dataset_path, use_images=False, use_text=False)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        model = BehaviorCloningModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print("\nTraining for 5 epochs...")
        model.train()
        
        for epoch in range(5):
            total_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                # Create input features vector
                features = torch.stack([
                    batch['speed'],
                    batch['ego_x'], 
                    batch['ego_y'],
                    batch['traffic_density']
                ], dim=1)
                
                actions = batch['action']
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, actions)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch+1}/5, Loss: {avg_loss:.4f}")
        
        print("✅ Behavior cloning training completed!")
        
        # Test prediction
        model.eval()
        with torch.no_grad():
            test_features = torch.FloatTensor([[30.0, 100.0, 50.0, 0.8]])  # speed, x, y, density
            prediction = model(test_features)
            predicted_action = torch.argmax(prediction, dim=1)
            print(f"Test prediction: Action {predicted_action.item()}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def train_language_conditioned_model():
    """Example 2: Language-Conditioned Policy"""
    print("\n🎯 Example 2: Language-Conditioned Training")
    print("=" * 50)
    
    class LanguageConditionedModel(nn.Module):
        def __init__(self, vocab_size=1000, embed_dim=64, num_actions=5):
            super().__init__()
            self.text_embedding = nn.Embedding(vocab_size, embed_dim)
            self.text_encoder = nn.LSTM(embed_dim, 32, batch_first=True)
            self.feature_fc = nn.Linear(4, 32)  # Basic features
            self.fusion = nn.Linear(32 + 32, 64)
            self.policy = nn.Linear(64, num_actions)
            
        def forward(self, features, text_lengths):
            # Simple feature encoding
            feat_encoded = torch.relu(self.feature_fc(features))
            
            # Mock text encoding (in real implementation, would use proper tokenization)
            text_encoded = torch.zeros_like(feat_encoded)  # Placeholder
            
            # Fusion
            combined = torch.cat([feat_encoded, text_encoded], dim=1)
            output = self.policy(torch.relu(self.fusion(combined)))
            return output
    
    try:
        dataset_path = "/home/chettra/ITC/Research/AVs/data/highway_multimodal_dataset_clean/free_flow/ep_20250921_001_transitions.parquet"
        dataset = HighwayDataset(dataset_path, use_images=False, use_text=True)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        model = LanguageConditionedModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print(f"\nTraining language-conditioned model for 3 epochs...")
        model.train()
        
        for epoch in range(3):
            total_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                features = torch.stack([
                    batch['speed'],
                    batch['ego_x'],
                    batch['ego_y'],
                    batch['traffic_density']
                ], dim=1)
                
                text_lengths = batch['text_length']
                actions = batch['action']
                
                optimizer.zero_grad()
                outputs = model(features, text_lengths)
                loss = criterion(outputs, actions)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch+1}/3, Loss: {avg_loss:.4f}")
        
        print("✅ Language-conditioned training completed!")
        
        # Show sample text-action pairs
        print("\nSample text-action pairs from training:")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            if 'text' in sample:
                print(f"  Text: \"{sample['text'][:50]}...\" → Action: {sample['action']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def analyze_reward_distribution():
    """Example 3: Reward Analysis for RL"""
    print("\n🎯 Example 3: Reward Analysis for RL Training")
    print("=" * 50)
    
    scenarios = {
        'Dense Commuting': "/home/chettra/ITC/Research/AVs/data/highway_multimodal_dataset_clean/dense_commuting/ep_20250921_001_transitions.parquet",
        'Free Flow': "/home/chettra/ITC/Research/AVs/data/highway_multimodal_dataset_clean/free_flow/ep_20250921_001_transitions.parquet",
        'Stop and Go': "/home/chettra/ITC/Research/AVs/data/highway_multimodal_dataset_clean/stop_and_go/ep_20250921_001_transitions.parquet"
    }
    
    for scenario_name, path in scenarios.items():
        try:
            df = pd.read_parquet(path)
            if 'reward' in df.columns:
                rewards = df['reward'].dropna()
                print(f"\n{scenario_name}:")
                print(f"  Mean reward: {rewards.mean():.3f}")
                print(f"  Std reward: {rewards.std():.3f}")
                print(f"  Min/Max: [{rewards.min():.3f}, {rewards.max():.3f}]")
                print(f"  Positive rewards: {(rewards > 0).sum()}/{len(rewards)} ({(rewards > 0).mean()*100:.1f}%)")
                
                # Action distribution
                if 'action' in df.columns:
                    actions = df['action'].value_counts().sort_index()
                    print(f"  Action distribution: {dict(actions)}")
        except Exception as e:
            print(f"  ❌ Error loading {scenario_name}: {e}")

def create_training_splits():
    """Example 4: Create Train/Val/Test Splits"""
    print("\n🎯 Example 4: Creating Training Splits")
    print("=" * 50)
    
    try:
        # Load all scenario data
        scenarios = ['dense_commuting', 'free_flow', 'stop_and_go']
        all_data = []
        
        for scenario in scenarios:
            path = f"/home/chettra/ITC/Research/AVs/data/highway_multimodal_dataset_clean/{scenario}/ep_20250921_001_transitions.parquet"
            if os.path.exists(path):
                df = pd.read_parquet(path)
                df['scenario_type'] = scenario
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"Combined dataset: {len(combined_df)} samples")
            
            # Create splits
            np.random.seed(42)
            indices = np.random.permutation(len(combined_df))
            
            train_size = int(0.7 * len(combined_df))
            val_size = int(0.15 * len(combined_df))
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            print(f"Training set: {len(train_indices)} samples")
            print(f"Validation set: {len(val_indices)} samples")
            print(f"Test set: {len(test_indices)} samples")
            
            # Show scenario distribution in splits
            for split_name, split_indices in [('Train', train_indices), ('Val', val_indices), ('Test', test_indices)]:
                split_data = combined_df.iloc[split_indices]
                scenario_counts = split_data['scenario_type'].value_counts()
                print(f"{split_name} scenarios: {dict(scenario_counts)}")
                
        else:
            print("❌ No data files found!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all training examples."""
    print("🚗 Highway Dataset Training Examples")
    print("=" * 60)
    print("This script demonstrates different ways to train models using your datasets.\n")
    
    # Check if datasets exist
    sample_file = "/home/chettra/ITC/Research/AVs/data/highway_multimodal_dataset_clean/dense_commuting/ep_20250921_001_transitions.parquet"
    if not os.path.exists(sample_file):
        print("❌ Sample dataset not found! Please check your data directory.")
        return
    
    # Run examples
    train_behavior_cloning_model()
    train_language_conditioned_model()
    analyze_reward_distribution()
    create_training_splits()
    
    print(f"\n{'='*60}")
    print("🎉 Training Examples Completed!")
    print("="*60)
    print("\n💡 Next Steps:")
    print("1. Choose your training goal (RL, imitation learning, etc.)")
    print("2. Select appropriate datasets based on modalities needed")
    print("3. Implement more sophisticated models based on these examples")
    print("4. Scale up training with full multimodal datasets")
    print("5. Evaluate on different traffic scenarios")

if __name__ == "__main__":
    main()