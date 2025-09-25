# Highway Dataset Training Guide 🚗

## Dataset Overview and Training Applications

Based on the analysis of your highway multimodal datasets, here's a comprehensive guide on how to use each dataset type for training different AI models.

---

## 📊 Dataset Categories & Training Uses

### 1. **Multimodal Datasets** (with Images + Occupancy Grids)
**Files:**
- `20250921_151946-f35b18e8_transitions.parquet` (Dense Commuting)
- `20250921_152749-1ea1c024_transitions.parquet` (Dense Commuting) 
- `20250921_151816-0a384836_transitions.parquet` (Free Flow)
- `20250921_152624-ed6a89cd_transitions.parquet` (Free Flow)

**Data Contents:**
- **Images**: 4×128×64 grayscale image stacks (uint8)
- **Occupancy Grids**: 2×11×11 spatial occupancy maps (float32)
- **Actions**: Discrete actions (0-4 range)
- **Rewards**: Continuous rewards
- **Rich Features**: Speed, position, traffic density, etc.
- **Natural Language**: Driving situation descriptions

**Training Applications:**

#### 🎯 **Vision-Based Reinforcement Learning**
```python
# Example architecture
class MultimodalDrivingAgent(nn.Module):
    def __init__(self):
        # Image encoder (CNN)
        self.image_encoder = CNN(input_channels=4, output_dim=256)
        # Occupancy encoder  
        self.occupancy_encoder = CNN(input_channels=2, output_dim=128)
        # Fusion layer
        self.fusion = nn.Linear(256 + 128 + feature_dim, 512)
        # Policy head
        self.policy = nn.Linear(512, action_dim)
        # Value head
        self.value = nn.Linear(512, 1)

# Training with PPO/SAC
agent = MultimodalDrivingAgent()
# Train using the action-reward pairs
```

#### 🎯 **Computer Vision Pre-training**
- **Self-supervised learning** from image sequences
- **Contrastive learning** between different views
- **Future frame prediction**
- **Scene understanding** and object detection

#### 🎯 **Multimodal Fusion Research**
- **Cross-modal attention** between vision and spatial data
- **Sensor fusion** techniques
- **Robust perception** under different conditions

---

### 2. **Language-Conditioned Datasets**
**Files:** All datasets contain `summary_text` field with driving descriptions

**Natural Language Examples:**
- "Light traffic detected, optimal driving conditions"
- "Heavy traffic detected, reducing speed"
- "Clear road ahead, accelerating to target speed"

**Training Applications:**

#### 🎯 **Language-Conditioned Driving Policies**
```python
class LanguageConditionedAgent(nn.Module):
    def __init__(self):
        self.text_encoder = BERTEncoder()  # or other transformer
        self.vision_encoder = ResNet()
        self.fusion = CrossAttentionLayer()
        self.policy = PolicyHead()

# Train to follow natural language driving instructions
```

#### 🎯 **Vision-Language Models**
- **CLIP-style** contrastive learning between images and descriptions
- **VQA (Visual Question Answering)** for driving scenarios
- **Grounded language understanding** in driving contexts

#### 🎯 **Explainable AI**
- Generate natural language explanations for driving decisions
- Train models to describe their reasoning process

---

### 3. **Scenario-Based Datasets**

#### 🚦 **Dense Commuting Scenarios**
**Files:** All files in `highway_multimodal_dataset/dense_commuting/`
- **Characteristics**: Heavy traffic, frequent lane changes, complex interactions
- **Average Reward**: ~0.065-0.136 (moderate performance)
- **Use Case**: Training robust policies for challenging traffic

#### 🛣️ **Free Flow Scenarios** 
**Files:** All files in `highway_multimodal_dataset/free_flow/`
- **Characteristics**: Light traffic, smooth driving, optimal conditions
- **Average Reward**: ~0.051-0.111 (consistent performance)
- **Use Case**: Learning efficient driving behaviors

#### 🐌 **Stop-and-Go Scenarios**
**Files:** All files in `highway_multimodal_dataset/stop_and_go/`
- **Characteristics**: Congested traffic, frequent stopping, patience required
- **Average Reward**: ~0.067-0.088 (challenging conditions)
- **Use Case**: Training adaptive policies for traffic jams

**Training Applications:**

#### 🎯 **Domain Adaptation**
```python
# Train domain-agnostic features
domain_classifier = DomainDiscriminator()
feature_extractor = FeatureExtractor()

# Adversarial training for scenario transfer
```

#### 🎯 **Meta-Learning**
- **Few-shot adaptation** to new traffic scenarios
- **Model-Agnostic Meta-Learning (MAML)** for quick scenario adaptation
- **Context-dependent** policy learning

#### 🎯 **Curriculum Learning**
1. **Stage 1**: Train on Free Flow (easy)
2. **Stage 2**: Train on Dense Commuting (medium)  
3. **Stage 3**: Train on Stop-and-Go (hard)

---

### 4. **Clean vs Raw Datasets**

#### **Raw Datasets** (`highway_multimodal_dataset/`)
- **Size**: 84-100 steps per episode
- **Content**: Full multimodal data with images
- **Use**: Complete training with all modalities

#### **Clean Datasets** (`highway_multimodal_dataset_clean/`)
- **Size**: 50 steps per episode (filtered)
- **Content**: Curated data without images
- **Use**: Faster training, debugging, baseline models

---

## 🛠️ Specific Training Recipes

### Recipe 1: **Imitation Learning (Behavior Cloning)**
```python
# Use action-observation pairs
X = [images, occupancy, features]  # observations
y = actions                        # expert actions

model = BehaviorCloningModel()
model.fit(X, y)
```

### Recipe 2: **Reinforcement Learning**
```python
# Use full transition tuples
transitions = [(state, action, reward, next_state, done)]

agent = PPOAgent()
agent.train(transitions)
```

### Recipe 3: **Language-Conditioned Learning**
```python
# Use text-observation-action tuples
data = [(text_description, observation, action)]

model = LanguageConditionedPolicy()
model.train(data)
```

### Recipe 4: **Self-Supervised Pretraining**
```python
# Use image sequences for representation learning
image_sequences = extract_image_sequences(dataset)

model = ContrastiveEncoder()
model.pretrain(image_sequences)

# Fine-tune on downstream tasks
model.finetune(action_prediction_task)
```

---

## 📈 Training Recommendations by Goal

### **Goal: Autonomous Driving Agent**
1. **Start with**: Clean datasets for rapid prototyping
2. **Pre-train**: Vision encoder on multimodal datasets
3. **Fine-tune**: On specific scenarios (free_flow → dense_commuting → stop_and_go)
4. **Evaluate**: Across all scenario types

### **Goal: Research on Vision-Language Models**
1. **Focus on**: Natural language descriptions + images
2. **Train**: CLIP-style contrastive learning
3. **Evaluate**: On instruction following tasks

### **Goal: Robust Driving Policies**
1. **Train on**: All scenario types simultaneously
2. **Use**: Domain adaptation techniques
3. **Test**: Transfer across scenarios

### **Goal: Explainable AI for Driving**
1. **Combine**: Actions + natural language descriptions
2. **Train**: Models to generate explanations
3. **Evaluate**: Human interpretability studies

---

## 🔧 Technical Implementation Tips

### **Data Loading**
```python
def load_multimodal_data(file_path):
    df = pd.read_parquet(file_path)
    
    # Extract images
    images = []
    for _, row in df.iterrows():
        img_stack = decode_blob(row, "grayscale_blob", "grayscale_shape", "grayscale_dtype")
        images.append(img_stack)
    
    # Extract other features
    actions = df['action'].values
    rewards = df['reward'].values
    texts = df['summary_text'].values
    
    return images, actions, rewards, texts
```

### **Memory Management**
- **Lazy loading**: Load data in batches to manage memory
- **Image compression**: Consider downsampling images if needed
- **Feature caching**: Pre-compute features to speed up training

### **Evaluation Metrics**
- **Success Rate**: Task completion percentage
- **Average Reward**: Policy performance
- **Safety Metrics**: Collision avoidance
- **Efficiency**: Fuel consumption, travel time
- **Language Alignment**: Text-action correspondence

---

## 🎯 Next Steps for Training

1. **Choose your training goal** from the categories above
2. **Select appropriate datasets** based on your modalities needed
3. **Implement data loaders** using the visualization tools provided
4. **Start with simple baselines** (behavior cloning on clean data)
5. **Scale up to complex models** (multimodal RL with full datasets)
6. **Evaluate thoroughly** across different scenarios

The visualization tools in `/visualization/` folder will help you explore and understand your data before training!