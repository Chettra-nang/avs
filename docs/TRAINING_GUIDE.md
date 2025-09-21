# Training Guide: Using HighwayEnv Multi-Modal Data

This guide shows you how to use the collected data to train AI models for autonomous driving.

## 🎯 **Training Applications**

### **1. Reinforcement Learning (RL)**
Train agents to make driving decisions based on rewards and observations.

### **2. Supervised Learning**
Train models to predict actions, trajectories, or safety metrics.

### **3. Computer Vision**
Train vision models to understand road scenes and detect objects.

### **4. Multi-Modal Learning**
Combine different data types (vision + sensors) for robust models.

## 📊 **Data Loading for Training**

### **Basic Data Loading**

```python
import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_training_data(dataset_path):
    """Load collected data for training."""
    
    # Load dataset index
    index_path = Path(dataset_path) / "index.json"
    with open(index_path, 'r') as f:
        index = json.load(f)
    
    all_data = []
    
    # Load all scenarios
    for scenario_name, files in index['scenarios'].items():
        for file_info in files:
            # Load transitions (main training data)
            parquet_path = Path(dataset_path) / file_info['transitions_file']
            df = pd.read_parquet(parquet_path)
            df['scenario'] = scenario_name
            all_data.append(df)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

# Usage
data = load_training_data("data/highway_multimodal_dataset")
print(f"Loaded {len(data)} training samples")
```

### **Reconstruct Multi-Modal Observations**

```python
def reconstruct_observations(df):
    """Reconstruct binary observations for training."""
    
    observations = {}
    
    for idx, row in df.iterrows():
        obs_dict = {}
        
        # Reconstruct occupancy grid
        if 'occupancy_blob' in row and pd.notna(row['occupancy_blob']):
            blob = row['occupancy_blob']
            shape = eval(row['occupancy_shape']) if isinstance(row['occupancy_shape'], str) else row['occupancy_shape']
            dtype = row['occupancy_dtype']
            obs_dict['occupancy'] = np.frombuffer(blob, dtype=dtype).reshape(shape)
        
        # Reconstruct grayscale image
        if 'grayscale_blob' in row and pd.notna(row['grayscale_blob']):
            blob = row['grayscale_blob']
            shape = eval(row['grayscale_shape']) if isinstance(row['grayscale_shape'], str) else row['grayscale_shape']
            dtype = row['grayscale_dtype']
            obs_dict['grayscale'] = np.frombuffer(blob, dtype=dtype).reshape(shape)
        
        # Add kinematics (already in readable format)
        if 'kinematics_raw' in row:
            obs_dict['kinematics'] = eval(row['kinematics_raw']) if isinstance(row['kinematics_raw'], str) else row['kinematics_raw']
        
        observations[idx] = obs_dict
    
    return observations

# Usage
observations = reconstruct_observations(data)
```

## 🤖 **Training Examples**

### **1. Reinforcement Learning with Stable-Baselines3**

```python
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

# Create training environment
def make_env():
    env = gym.make('highway-v0')
    return env

# Set up training
env = DummyVecEnv([make_env])

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1
)

# Train the model
model.learn(total_timesteps=100000)

# Save trained model
model.save("highway_ppo_model")

# Test the trained model
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

### **2. Supervised Learning for Action Prediction**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def prepare_supervised_data(df):
    """Prepare data for supervised learning."""
    
    # Features: use extracted features
    feature_cols = ['speed', 'ttc', 'lane_position', 'traffic_density', 
                   'lead_vehicle_gap', 'ego_x', 'ego_y', 'ego_vx', 'ego_vy']
    
    # Filter available columns
    available_cols = [col for col in feature_cols if col in df.columns]
    X = df[available_cols].fillna(0)
    
    # Target: actions (convert from list to single values)
    y = df['action'].apply(lambda x: x[0] if isinstance(x, list) else x)
    
    return X, y

# Prepare data
X, y = prepare_supervised_data(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Action prediction accuracy: {accuracy:.3f}")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(feature_importance)
```

### **3. Computer Vision with PyTorch**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class HighwayVisionDataset(Dataset):
    """Dataset for computer vision training."""
    
    def __init__(self, df, observations, transform=None):
        self.df = df
        self.observations = observations
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get grayscale image
        if idx in self.observations and 'grayscale' in self.observations[idx]:
            image = self.observations[idx]['grayscale']
            
            # Convert to tensor
            if len(image.shape) == 3:
                image = torch.FloatTensor(image).permute(2, 0, 1)  # HWC to CHW
            else:
                image = torch.FloatTensor(image).unsqueeze(0)  # Add channel dim
        else:
            # Placeholder image if not available
            image = torch.zeros(1, 84, 84)
        
        # Get action label
        action = self.df.iloc[idx]['action']
        if isinstance(action, list):
            action = action[0]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.LongTensor([action])

# Create dataset
transform = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

dataset = HighwayVisionDataset(data, observations, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Simple CNN model
class HighwayCNN(nn.Module):
    def __init__(self, num_classes=5):  # 5 actions in highway-env
        super(HighwayCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HighwayCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(10):
    total_loss = 0
    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device).squeeze()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}/10, Loss: {total_loss/len(dataloader):.4f}')

# Save model
torch.save(model.state_dict(), 'highway_cnn_model.pth')
```

### **4. Multi-Modal Learning**

```python
class MultiModalModel(nn.Module):
    """Multi-modal model combining vision and sensor data."""
    
    def __init__(self, num_classes=5, sensor_dim=9):
        super(MultiModalModel, self).__init__()
        
        # Vision branch (CNN)
        self.vision_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Sensor branch (MLP)
        self.sensor_branch = nn.Sequential(
            nn.Linear(sensor_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(64 * 4 * 4 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, image, sensors):
        # Process vision
        vision_features = self.vision_branch(image)
        vision_features = vision_features.view(vision_features.size(0), -1)
        
        # Process sensors
        sensor_features = self.sensor_branch(sensors)
        
        # Fuse features
        combined = torch.cat([vision_features, sensor_features], dim=1)
        output = self.fusion(combined)
        
        return output

# Usage with multi-modal data
class MultiModalDataset(Dataset):
    def __init__(self, df, observations):
        self.df = df
        self.observations = observations
    
    def __getitem__(self, idx):
        # Get image
        if idx in self.observations and 'grayscale' in self.observations[idx]:
            image = torch.FloatTensor(self.observations[idx]['grayscale'])
            if len(image.shape) == 3:
                image = image.permute(2, 0, 1)
            else:
                image = image.unsqueeze(0)
        else:
            image = torch.zeros(1, 84, 84)
        
        # Get sensor data (features)
        row = self.df.iloc[idx]
        sensor_data = torch.FloatTensor([
            row.get('speed', 0),
            row.get('ttc', 0) if row.get('ttc', float('inf')) != float('inf') else 0,
            row.get('lane_position', 0),
            row.get('traffic_density', 0),
            row.get('ego_x', 0),
            row.get('ego_y', 0),
            row.get('ego_vx', 0),
            row.get('ego_vy', 0),
            row.get('lead_vehicle_gap', 0)
        ])
        
        # Get action
        action = row['action']
        if isinstance(action, list):
            action = action[0]
        
        return image, sensor_data, torch.LongTensor([action])
```

## 📈 **Training Best Practices**

### **1. Data Preprocessing**
```python
def preprocess_data(df):
    """Clean and preprocess training data."""
    
    # Handle infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    
    feature_cols = ['speed', 'ttc', 'lane_position', 'traffic_density']
    available_cols = [col for col in feature_cols if col in df.columns]
    
    if available_cols:
        scaler = StandardScaler()
        df[available_cols] = scaler.fit_transform(df[available_cols])
    
    return df, scaler

# Usage
processed_data, scaler = preprocess_data(data)
```

### **2. Evaluation Metrics**
```python
def evaluate_model(model, test_data):
    """Comprehensive model evaluation."""
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    # Predictions
    predictions = model.predict(test_data)
    
    # Metrics
    accuracy = accuracy_score(y_true, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, predictions, average='weighted')
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```

### **3. Model Deployment**
```python
def deploy_model(model, model_type='sklearn'):
    """Deploy trained model for inference."""
    
    if model_type == 'sklearn':
        import joblib
        joblib.dump(model, 'highway_model.pkl')
        
    elif model_type == 'pytorch':
        torch.save(model.state_dict(), 'highway_model.pth')
        
    elif model_type == 'stable_baselines':
        model.save('highway_sb3_model')
    
    print(f"Model saved successfully!")

def load_and_test_model(model_path, test_data):
    """Load and test deployed model."""
    
    # Load model
    model = joblib.load(model_path)  # For sklearn
    
    # Test on new data
    predictions = model.predict(test_data)
    
    return predictions
```

## 🎯 **Training Scenarios**

### **Scenario 1: Collision Avoidance**
- **Goal**: Train model to avoid collisions
- **Data**: Focus on episodes with low TTC values
- **Features**: TTC, relative speeds, distances
- **Target**: Actions that increase TTC

### **Scenario 2: Lane Changing**
- **Goal**: Learn when and how to change lanes
- **Data**: Episodes with lane change actions (3, 4)
- **Features**: Traffic density, lane positions, gaps
- **Target**: Successful lane change decisions

### **Scenario 3: Speed Control**
- **Goal**: Maintain optimal speed for traffic conditions
- **Data**: Speed and reward correlations
- **Features**: Traffic density, lead vehicle gap, speed limits
- **Target**: Speed adjustment actions (0, 1, 2)

## 🔄 **Continuous Learning**

### **Online Learning Setup**
```python
def online_learning_loop():
    """Continuous learning from new data."""
    
    while True:
        # Collect new data
        new_data = collect_new_episodes()
        
        # Update model
        model.partial_fit(new_data)
        
        # Evaluate performance
        performance = evaluate_model(model, validation_data)
        
        # Log results
        print(f"Updated model performance: {performance}")
        
        # Sleep before next iteration
        time.sleep(3600)  # Update hourly
```

## 📊 **Data Analysis for Training**

### **Exploratory Data Analysis**
```python
def analyze_training_data(df):
    """Analyze data for training insights."""
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Action distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    action_counts = df['action'].value_counts()
    plt.bar(action_counts.index, action_counts.values)
    plt.title('Action Distribution')
    plt.xlabel('Action')
    plt.ylabel('Count')
    
    # Reward distribution by scenario
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='scenario', y='reward')
    plt.title('Reward by Scenario')
    plt.xticks(rotation=45)
    
    # TTC distribution
    plt.subplot(2, 2, 3)
    finite_ttc = df[df['ttc'] != np.inf]['ttc']
    plt.hist(finite_ttc, bins=50, alpha=0.7)
    plt.title('Time-to-Collision Distribution')
    plt.xlabel('TTC (seconds)')
    
    # Speed vs Reward correlation
    plt.subplot(2, 2, 4)
    plt.scatter(df['speed'], df['reward'], alpha=0.5)
    plt.title('Speed vs Reward')
    plt.xlabel('Speed')
    plt.ylabel('Reward')
    
    plt.tight_layout()
    plt.savefig('training_data_analysis.png')
    plt.show()

# Usage
analyze_training_data(data)
```

This guide provides a comprehensive foundation for using your collected highway data to train various types of AI models for autonomous driving research!