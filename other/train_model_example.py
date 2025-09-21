#!/usr/bin/env python3
"""
Training Example: Using Collected Highway Data to Train AI Models

This script demonstrates how to use the collected multi-modal highway data
to train different types of AI models for autonomous driving.
"""

import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def load_collected_data(dataset_path: str) -> pd.DataFrame:
    """Load all collected data for training."""
    print("📊 Loading collected data...")
    
    dataset_path = Path(dataset_path)
    index_path = dataset_path / "index.json"
    
    if not index_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        print("Run 'python main.py --demo collection' first to collect data")
        return pd.DataFrame()
    
    # Load index
    with open(index_path, 'r') as f:
        index = json.load(f)
    
    print(f"✅ Found dataset with {len(index['scenarios'])} scenarios")
    
    # Load all scenario data
    all_data = []
    for scenario_name, files in index['scenarios'].items():
        print(f"  Loading scenario: {scenario_name}")
        
        for file_info in files:
            parquet_path = dataset_path / file_info['transitions_file']
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                df['scenario'] = scenario_name
                all_data.append(df)
                print(f"    ✅ Loaded {len(df)} samples from {file_info['transitions_file']}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"✅ Total training samples: {len(combined_df)}")
        return combined_df
    else:
        print("❌ No data files found")
        return pd.DataFrame()

def analyze_data_for_training(df: pd.DataFrame):
    """Analyze the data to understand training opportunities."""
    print("\n🔍 Analyzing Data for Training...")
    
    if df.empty:
        print("❌ No data to analyze")
        return
    
    # Basic statistics
    print(f"📈 Dataset Overview:")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Episodes: {df['episode_id'].nunique()}")
    print(f"  - Scenarios: {df['scenario'].nunique()}")
    print(f"  - Agents: {df['agent_id'].nunique()}")
    
    # Action distribution
    print(f"\n🎮 Action Distribution:")
    # Handle actions that might be lists or arrays
    actions_processed = []
    for action in df['action']:
        if isinstance(action, (list, tuple)):
            actions_processed.append(action[0] if len(action) > 0 else 0)
        elif hasattr(action, '__iter__') and not isinstance(action, str):
            # Handle numpy arrays or other iterables
            try:
                actions_processed.append(int(list(action)[0]))
            except:
                actions_processed.append(0)
        else:
            actions_processed.append(action)
    
    action_series = pd.Series(actions_processed)
    action_counts = action_series.value_counts().sort_index()
    action_names = {0: "SLOWER", 1: "IDLE", 2: "FASTER", 3: "LANE_LEFT", 4: "LANE_RIGHT"}
    
    for action, count in action_counts.items():
        action_name = action_names.get(action, f"Action_{action}")
        percentage = (count / len(df)) * 100
        print(f"  - {action_name}: {count} ({percentage:.1f}%)")
    
    # Reward statistics
    print(f"\n🏆 Reward Statistics:")
    print(f"  - Mean reward: {df['reward'].mean():.3f}")
    print(f"  - Std reward: {df['reward'].std():.3f}")
    print(f"  - Min reward: {df['reward'].min():.3f}")
    print(f"  - Max reward: {df['reward'].max():.3f}")
    
    # Safety metrics
    if 'ttc' in df.columns:
        finite_ttc = df[df['ttc'] != np.inf]['ttc']
        if not finite_ttc.empty:
            print(f"\n🚨 Safety Metrics:")
            print(f"  - Average TTC: {finite_ttc.mean():.2f}s")
            print(f"  - Dangerous situations (TTC < 2s): {(finite_ttc < 2).sum()}")
            print(f"  - Very dangerous (TTC < 1s): {(finite_ttc < 1).sum()}")
    
    # Feature availability
    print(f"\n📊 Available Features for Training:")
    feature_cols = ['speed', 'ttc', 'lane_position', 'traffic_density', 
                   'lead_vehicle_gap', 'ego_x', 'ego_y', 'ego_vx', 'ego_vy']
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"  - Numeric features: {available_features}")
    
    # Binary data availability
    binary_cols = ['occupancy_blob', 'grayscale_blob']
    available_binary = [col for col in binary_cols if col in df.columns]
    print(f"  - Binary data: {available_binary}")

def train_action_prediction_model(df: pd.DataFrame):
    """Train a model to predict driving actions."""
    print("\n🤖 Training Action Prediction Model...")
    
    if df.empty:
        print("❌ No data available for training")
        return None
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.preprocessing import StandardScaler
        
        # Prepare features
        feature_cols = ['speed', 'ttc', 'lane_position', 'traffic_density', 
                       'lead_vehicle_gap', 'ego_x', 'ego_y', 'ego_vx', 'ego_vy']
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            print("❌ No suitable features found for training")
            return None
        
        print(f"📊 Using features: {available_cols}")
        
        # Prepare data
        X = df[available_cols].copy()
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Prepare targets (actions)
        y = df['action'].apply(lambda x: x[0] if isinstance(x, list) else x)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📚 Training set: {len(X_train)} samples")
        print(f"📝 Test set: {len(X_test)} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        print("🔄 Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained successfully!")
        print(f"🎯 Test Accuracy: {accuracy:.3f}")
        
        # Detailed classification report
        print(f"\n📊 Detailed Performance:")
        action_names = {0: "SLOWER", 1: "IDLE", 2: "FASTER", 3: "LANE_LEFT", 4: "LANE_RIGHT"}
        target_names = [action_names.get(i, f"Action_{i}") for i in sorted(y_test.unique())]
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Feature importance
        print(f"\n🔍 Feature Importance:")
        feature_importance = pd.DataFrame({
            'feature': available_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in feature_importance.iterrows():
            print(f"  - {row['feature']}: {row['importance']:.3f}")
        
        # Save model
        import joblib
        model_path = "trained_highway_model.pkl"
        scaler_path = "feature_scaler.pkl"
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"💾 Model saved to: {model_path}")
        print(f"💾 Scaler saved to: {scaler_path}")
        
        return model, scaler
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install scikit-learn")
        return None
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def train_safety_prediction_model(df: pd.DataFrame):
    """Train a model to predict safety metrics (TTC)."""
    print("\n🚨 Training Safety Prediction Model...")
    
    if df.empty or 'ttc' not in df.columns:
        print("❌ No TTC data available for safety training")
        return None
    
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data for TTC prediction
        feature_cols = ['speed', 'lane_position', 'traffic_density', 
                       'lead_vehicle_gap', 'ego_vx', 'ego_vy']
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            print("❌ No suitable features for safety prediction")
            return None
        
        # Filter out infinite TTC values for training
        finite_ttc_mask = (df['ttc'] != np.inf) & (df['ttc'] != -np.inf) & (df['ttc'].notna())
        training_df = df[finite_ttc_mask].copy()
        
        if len(training_df) < 10:
            print("❌ Not enough finite TTC samples for training")
            return None
        
        print(f"📊 Using {len(training_df)} samples with finite TTC values")
        
        X = training_df[available_cols].fillna(0)
        y = training_df['ttc']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        print("🔄 Training Gradient Boosting model for TTC prediction...")
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Safety model trained successfully!")
        print(f"🎯 Test MSE: {mse:.3f}")
        print(f"🎯 Test R²: {r2:.3f}")
        
        # Save model
        import joblib
        safety_model_path = "safety_prediction_model.pkl"
        safety_scaler_path = "safety_scaler.pkl"
        
        joblib.dump(model, safety_model_path)
        joblib.dump(scaler, safety_scaler_path)
        
        print(f"💾 Safety model saved to: {safety_model_path}")
        
        return model, scaler
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return None
    except Exception as e:
        print(f"❌ Safety training failed: {e}")
        return None

def demonstrate_model_usage(model, scaler, df: pd.DataFrame):
    """Demonstrate how to use the trained model."""
    print("\n🎯 Demonstrating Model Usage...")
    
    if model is None or scaler is None:
        print("❌ No trained model available")
        return
    
    # Get a sample from the data
    sample_idx = len(df) // 2  # Middle sample
    sample_row = df.iloc[sample_idx]
    
    print(f"📋 Sample Driving Situation:")
    print(f"  - Scenario: {sample_row.get('scenario', 'Unknown')}")
    print(f"  - Speed: {sample_row.get('speed', 0):.1f} m/s")
    print(f"  - TTC: {sample_row.get('ttc', 'inf')}")
    print(f"  - Lane Position: {sample_row.get('lane_position', 0)}")
    print(f"  - Traffic Density: {sample_row.get('traffic_density', 0):.2f}")
    
    # Prepare features for prediction
    feature_cols = ['speed', 'ttc', 'lane_position', 'traffic_density', 
                   'lead_vehicle_gap', 'ego_x', 'ego_y', 'ego_vx', 'ego_vy']
    available_cols = [col for col in feature_cols if col in df.columns]
    
    sample_features = []
    for col in available_cols:
        value = sample_row.get(col, 0)
        if value == np.inf or value == -np.inf:
            value = 0
        sample_features.append(value)
    
    # Scale features
    sample_scaled = scaler.transform([sample_features])
    
    # Predict action
    predicted_action = model.predict(sample_scaled)[0]
    actual_action = sample_row['action']
    if isinstance(actual_action, list):
        actual_action = actual_action[0]
    
    action_names = {0: "SLOWER", 1: "IDLE", 2: "FASTER", 3: "LANE_LEFT", 4: "LANE_RIGHT"}
    
    print(f"\n🤖 Model Prediction:")
    print(f"  - Predicted Action: {predicted_action} ({action_names.get(predicted_action, 'Unknown')})")
    print(f"  - Actual Action: {actual_action} ({action_names.get(actual_action, 'Unknown')})")
    print(f"  - Match: {'✅ Correct' if predicted_action == actual_action else '❌ Incorrect'}")
    
    # Get prediction confidence (for tree-based models)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(sample_scaled)[0]
        print(f"\n📊 Prediction Confidence:")
        for i, prob in enumerate(probabilities):
            if prob > 0.01:  # Only show significant probabilities
                print(f"  - {action_names.get(i, f'Action_{i}')}: {prob:.3f}")

def create_training_visualization(df: pd.DataFrame):
    """Create visualizations of the training data."""
    print("\n📈 Creating Training Data Visualizations...")
    
    if df.empty:
        print("❌ No data to visualize")
        return
    
    try:
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Highway Training Data Analysis', fontsize=16)
        
        # 1. Action distribution
        ax1 = axes[0, 0]
        actions = df['action'].apply(lambda x: x[0] if isinstance(x, list) else x)
        action_counts = actions.value_counts().sort_index()
        action_names = {0: "SLOWER", 1: "IDLE", 2: "FASTER", 3: "LANE_LEFT", 4: "LANE_RIGHT"}
        labels = [action_names.get(i, f'Action_{i}') for i in action_counts.index]
        
        ax1.bar(range(len(action_counts)), action_counts.values)
        ax1.set_xticks(range(len(action_counts)))
        ax1.set_xticklabels(labels, rotation=45)
        ax1.set_title('Action Distribution')
        ax1.set_ylabel('Count')
        
        # 2. Reward by scenario
        ax2 = axes[0, 1]
        scenarios = df['scenario'].unique()
        for scenario in scenarios:
            scenario_rewards = df[df['scenario'] == scenario]['reward']
            ax2.hist(scenario_rewards, alpha=0.7, label=scenario, bins=20)
        ax2.set_title('Reward Distribution by Scenario')
        ax2.set_xlabel('Reward')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # 3. Speed distribution
        ax3 = axes[0, 2]
        if 'speed' in df.columns:
            ax3.hist(df['speed'], bins=30, alpha=0.7, color='green')
            ax3.set_title('Speed Distribution')
            ax3.set_xlabel('Speed (m/s)')
            ax3.set_ylabel('Frequency')
        else:
            ax3.text(0.5, 0.5, 'Speed data\nnot available', ha='center', va='center')
            ax3.set_title('Speed Distribution')
        
        # 4. TTC distribution
        ax4 = axes[1, 0]
        if 'ttc' in df.columns:
            finite_ttc = df[df['ttc'] != np.inf]['ttc']
            if not finite_ttc.empty:
                ax4.hist(finite_ttc, bins=30, alpha=0.7, color='red')
                ax4.set_title('Time-to-Collision Distribution')
                ax4.set_xlabel('TTC (seconds)')
                ax4.set_ylabel('Frequency')
            else:
                ax4.text(0.5, 0.5, 'No finite TTC\nvalues found', ha='center', va='center')
                ax4.set_title('Time-to-Collision Distribution')
        else:
            ax4.text(0.5, 0.5, 'TTC data\nnot available', ha='center', va='center')
            ax4.set_title('Time-to-Collision Distribution')
        
        # 5. Reward vs Speed correlation
        ax5 = axes[1, 1]
        if 'speed' in df.columns:
            ax5.scatter(df['speed'], df['reward'], alpha=0.5, s=10)
            ax5.set_title('Speed vs Reward')
            ax5.set_xlabel('Speed (m/s)')
            ax5.set_ylabel('Reward')
        else:
            ax5.text(0.5, 0.5, 'Speed data\nnot available', ha='center', va='center')
            ax5.set_title('Speed vs Reward')
        
        # 6. Episode length distribution
        ax6 = axes[1, 2]
        episode_lengths = df.groupby('episode_id')['step'].max()
        ax6.hist(episode_lengths, bins=20, alpha=0.7, color='purple')
        ax6.set_title('Episode Length Distribution')
        ax6.set_xlabel('Episode Length (steps)')
        ax6.set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = "training_data_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved to: {plot_path}")
        
        # Show plot
        plt.show()
        
    except ImportError as e:
        print(f"❌ Visualization failed - missing dependency: {e}")
        print("Install with: pip install matplotlib seaborn")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

def main():
    """Main training demonstration."""
    print("🚗 HighwayEnv Multi-Modal Data Training Example")
    print("=" * 60)
    
    # Load data
    dataset_path = "data/highway_multimodal_dataset"
    df = load_collected_data(dataset_path)
    
    if df.empty:
        print("\n❌ No training data available.")
        print("Please run data collection first:")
        print("  python main.py --demo collection")
        return
    
    # Analyze data
    analyze_data_for_training(df)
    
    # Create visualizations
    create_training_visualization(df)
    
    # Train action prediction model
    model, scaler = train_action_prediction_model(df)
    
    # Train safety prediction model
    safety_model, safety_scaler = train_safety_prediction_model(df)
    
    # Demonstrate model usage
    if model and scaler:
        demonstrate_model_usage(model, scaler, df)
    
    print("\n🎉 Training demonstration completed!")
    print("\n📚 Next Steps:")
    print("1. Collect more data: python main.py --demo collection")
    print("2. Try different scenarios: python main.py --demo basic")
    print("3. Use trained models in your applications")
    print("4. Experiment with different ML algorithms")
    print("5. Integrate models back into the simulation:")
    print("   python main.py --demo policy")

if __name__ == "__main__":
    main()